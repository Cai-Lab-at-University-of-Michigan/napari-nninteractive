import os
import warnings
import datetime
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from huggingface_hub import snapshot_download
from napari._qt.layer_controls.qt_layer_controls_container import layer_to_controls
from napari.layers import Labels
from napari.layers.base._base_constants import ActionType
from napari.utils.notifications import show_warning
from napari.utils.transforms import Affine
from napari.viewer import Viewer
from qtpy.QtWidgets import QFileDialog, QWidget, QMessageBox, QDialog, QVBoxLayout, QLabel

from qtpy.QtCore import QTimer, QEvent, Qt

from napari_nninteractive.controls.bbox_controls import CustomQtBBoxControls
from napari_nninteractive.controls.lasso_controls import CustomQtLassoControls
from napari_nninteractive.controls.point_controls import CustomQtPointsControls
from napari_nninteractive.controls.scribble_controls import CustomQtScribbleControls
from napari_nninteractive.layers.bbox_layer import BBoxLayer
from napari_nninteractive.layers.lasso_layer import LassoLayer
from napari_nninteractive.layers.point_layer import SinglePointLayer
from napari_nninteractive.layers.scribble_layer import ScribbleLayer
from napari_nninteractive.utils.affine import is_orthogonal
from napari_nninteractive.utils.utils import ColorMapper, determine_layer_index
from napari_nninteractive.widget_gui import BaseGUI

from napari_nninteractive.gpu_lock import GPUMemoryLock
from filelock import FileLock
from napari_nninteractive.timeout import timeout

layer_to_controls[SinglePointLayer] = CustomQtPointsControls
layer_to_controls[BBoxLayer] = CustomQtBBoxControls
layer_to_controls[ScribbleLayer] = CustomQtScribbleControls
layer_to_controls[LassoLayer] = CustomQtLassoControls


class LayerControls(BaseGUI):
    """
    A class for managing and interacting with different layers in the viewer,
    specifically designed for point, bounding box, and scribble layers.

    Args:
        viewer (Viewer): The Napari viewer instance to which layers will be added.
        parent (Optional[QWidget], optional): The parent widget. Defaults to None.
    """

    def __init__(self, viewer: Viewer, parent: Optional[QWidget] = None):
        super().__init__(viewer, parent)
        self.point_layer_name = "nnInteractive - Point Layer"
        self.bbox_layer_name = "nnInteractive - BBox Layer"
        self.scribble_layer_name = "nnInteractive - Scribble Layer"
        self.lasso_layer_name = "nnInteractive - Lasso Layer"
        self.layer_dict = {
            0: self.point_layer_name,
            1: self.bbox_layer_name,
            2: self.scribble_layer_name,
            3: self.lasso_layer_name,
        }

        self.label_layer_name = "nnInteractive - Label Layer"
        self.semantic_layer_name = "nnInteractive - Label Layer"
        self.archive_layer_name = "Merged Objects - Label Layer"
        self.colormap = ColorMapper(49, seed=0.5, background_value=0)
        self._scribble_brush_size = 5
        self.object_index = 0

        # Initialize auto-save timer
        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.timeout.connect(self._auto_save)
        self.auto_save_interval_minutes = 15

        self.inactivity_timer = QTimer(self)
        self.inactivity_timer.timeout.connect(self._handle_inactivity)
        self.inactivity_timeout_minutes = 30
        self.last_activity_time = datetime.datetime.now()
        
        self.start_auto_save_timer()
        self.start_inactivity_timer()

        self._viewer.window._qt_viewer.installEventFilter(self)
        self._viewer.layers.selection.events.active.connect(self.on_layer_selected)

        self._is_initialized = False

    # Layer Handling
    def _clear_layers(self) -> None:
        """Removes all layers in the viewer that are managed by this class."""
        layer_names = list(self.layer_dict.values())
        for layer_name in layer_names:
            if layer_name in self._viewer.layers:
                self._viewer.layers.remove(layer_name)

    def add_point_layer(self) -> None:
        """Adds a single point layer to the viewer."""
        point_layer = SinglePointLayer(
            name=self.point_layer_name,
            ndim=self.session_cfg["ndim"],
            affine=self.session_cfg["affine"],
            scale=self.session_cfg["scale"],
            translate=self.session_cfg["translate"],
            rotate=self.session_cfg["rotate"],
            shear=self.session_cfg["shear"],
            metadata=self.session_cfg["metadata"],
            opacity=0.7,
            size=3,
            prompt_index=self.prompt_button.index,
        )

        # point_layer.size = 0.2
        point_layer.events.finished.connect(self.on_interaction)
        self._viewer.add_layer(point_layer)

    def add_bbox_layer(self) -> None:
        """Adds a bounding box layer to the viewer."""
        bbox_layer = BBoxLayer(
            name=self.bbox_layer_name,
            ndim=self.session_cfg["ndim"],
            affine=self.session_cfg["affine"],
            scale=self.session_cfg["scale"],
            translate=self.session_cfg["translate"],
            rotate=self.session_cfg["rotate"],
            shear=self.session_cfg["shear"],
            metadata=self.session_cfg["metadata"],
            prompt_index=self.prompt_button.index,
            opacity=0.3,
        )
        bbox_layer.events.data.connect(self.on_interaction)
        self._viewer.add_layer(bbox_layer)

    def add_scribble_layer(self) -> None:
        """Adds a scribble layer to the viewer with an initial blank data array."""
        _data = np.zeros(self.session_cfg["shape"], dtype=np.uint8)
        scribble_layer = ScribbleLayer(
            data=_data,
            name=self.scribble_layer_name,
            affine=self.session_cfg["affine"],
            scale=self.session_cfg["scale"],
            translate=self.session_cfg["translate"],
            rotate=self.session_cfg["rotate"],
            shear=self.session_cfg["shear"],
            metadata=self.session_cfg["metadata"],
            prompt_index=self.prompt_button.index,
        )

        scribble_layer.brush_size = self._scribble_brush_size

        scribble_layer.events.finished.connect(self.on_interaction)
        self._viewer.add_layer(scribble_layer)

    def add_lasso_layer(self) -> None:
        """Adds a lasso layer to the viewer."""
        lasso_layer = LassoLayer(
            shape=self.session_cfg["shape"],
            name=self.lasso_layer_name,
            ndim=self.session_cfg["ndim"],
            affine=self.session_cfg["affine"],
            scale=self.session_cfg["scale"],
            translate=self.session_cfg["translate"],
            rotate=self.session_cfg["rotate"],
            shear=self.session_cfg["shear"],
            metadata=self.session_cfg["metadata"],
            prompt_index=self.prompt_button.index,
            opacity=0.3,
        )
        lasso_layer.events.data.connect(self.on_interaction)
        self._viewer.add_layer(lasso_layer)

    def add_label_layer(self, data, name) -> None:
        """
        Check if a layer with the layer_name already exists. If yes rename this by adding an index
        and afterward create the layer
        :return:
        :rtype:
        """

        label_layer = Labels(
            data,
            # self._data_result,
            name=name,
            opacity=0.3,
            affine=self.session_cfg["affine"],
            scale=self.session_cfg["scale"],
            translate=self.session_cfg["translate"],
            rotate=self.session_cfg["rotate"],
            shear=self.session_cfg["shear"],
            # colormap=self.colormap[index],
            metadata=self.session_cfg["metadata"],
        )
        label_layer._source = self.session_cfg["source"]

        self._viewer.add_layer(label_layer)

    def add_mask_init_layer(self) -> None:
        """
        Check if a layer with the layer_name already exists. If yes rename this by adding an index
        and afterward create the layer
        :return:
        :rtype:
        """

        _layer_res = Labels(
            np.zeros_like(self._data_result),
            name=self.mask_init_layer_name,
            opacity=0.3,
            affine=self.session_cfg["affine"],
            scale=self.session_cfg["scale"],
            translate=self.session_cfg["translate"],
            rotate=self.session_cfg["rotate"],
            shear=self.session_cfg["shear"],
            metadata=self.session_cfg["metadata"],
        )
        _layer_res._source = self.session_cfg["source"]

        self._viewer.add_layer(_layer_res)

    def init_with_mask(self):
        _layer_data = self._viewer.layers[self.label_for_init.currentText()].data

        assert (
            _layer_data.shape == self.session_cfg["shape"]
        )  # Labels and Image should have same shape

        self._data_result = (_layer_data == self.class_for_init.value()).astype(np.uint8)
        self.session.set_target_buffer(self._data_result)
        self._viewer.layers[self.label_layer_name].data = self._data_result

    # Event Handlers
    def on_init(self, *args, **kwargs) -> None:
        """
        Initializes the session by configuring the selected model and image and creating a label layer.

        Retrieves the selected model and image names from the GUI, extracts relevant data from the
        image layer, and creates a corresponding label layer in the viewer.
        """
        # --- MODEL HANDLING --- #
        # Get all model and image from the GUI
        image_name = self.image_selection.currentText()

        if image_name == "":
            raise ValueError("No Image Layer selected")

        model_name = self.model_selection.currentText()
        model_name_local = self.model_selection_local.text()
        if model_name_local != "" and Path(model_name_local).exists():
            # Use Local Checkpoint
            model_name = Path(model_name_local).name
            self.checkpoint_path = model_name_local
        else:
            # Download Checkpoint
            repo_id = "nnInteractive/nnInteractive"
            force_download = False
            download_path = snapshot_download(
                repo_id=repo_id, allow_patterns=[f"{model_name}/*"], force_download=force_download
            )
            self.checkpoint_path = Path(download_path).joinpath(model_name)
        print(f"Using Model {model_name} at : {self.checkpoint_path}")

        # --- DATA HANDLING --- #
        # Get everything we need from the image layer
        image_layer = self._viewer.layers[image_name]
        self.source_cfg = {
            "name": image_name,
            "model": model_name,
            "ndim": image_layer.ndim,
            "shape": image_layer.data.shape,
            "affine": image_layer.affine,
            "scale": image_layer.scale,
            "translate": image_layer.translate,
            "rotate": image_layer.rotate,
            "shear": image_layer.shear,
            "source": image_layer.source,
            "metadata": image_layer.metadata,
        }

        self.session_cfg = self.source_cfg.copy()

        # 1. Non - Othogonal Affine
        if not (
            is_orthogonal(
                self.source_cfg["affine"],
                image_layer.ndim,
                self._viewer.dims.order,
                self._viewer.dims.ndisplay,
            )
        ):
            show_warning(
                "Your data is non-orthogonal. This is not supported by napari. "
                "To fix this the direction and shear is ignored during visualizing which changes the appearance (only visual) of your data."
            )
            # 1. Make affine orthogonal -> ignore rotate and shear
            self.session_cfg["affine"] = Affine(
                scale=self.source_cfg["affine"].scale, translate=self.source_cfg["affine"].translate
            )
            # 2. Apply to Image Layer
            image_layer.affine = self.session_cfg["affine"]
            self._viewer.reset_view()

        # 1. Non - Othogonal Transforms
        # dummy affine to check if transforms are non-orthogonal
        _transform_matrix = Affine(
            scale=self.source_cfg["scale"],
            translate=self.source_cfg["translate"],
            rotate=self.source_cfg["rotate"],
            shear=self.source_cfg["shear"],
        )

        if not is_orthogonal(
            _transform_matrix,
            image_layer.ndim,
            self._viewer.dims.order,
            self._viewer.dims.ndisplay,
        ):
            show_warning(
                "Your data is non-orthogonal. This is not supported by napari. "
                "To fix this the direction and shear is ignored during visualizing which changes the appearance (only visual) of your data."
            )

            # 1. Make transforms orthogonal
            self.session_cfg["rotate"] = np.eye(self.source_cfg["ndim"])
            self.session_cfg["shear"] = np.zeros(self.source_cfg["ndim"])

            # 2. Apply to Image Layer
            image_layer.rotate = self.session_cfg["rotate"]
            image_layer.shear = self.session_cfg["shear"]
            self._viewer.reset_view()

        # 2. Convert 2D Data to dummy 3D Data
        if self.source_cfg["ndim"] == 2:
            self.session_cfg["ndim"] = 3
            self.session_cfg["shape"] = np.insert(self.session_cfg["shape"], 0, 1)

            # 1. to Affine
            self.session_cfg["affine"] = self.session_cfg["affine"].expand_dims([0])

            # 2. to Transforms
            self.session_cfg["scale"] = np.insert(self.session_cfg["scale"], 0, 1)
            self.session_cfg["origin"] = np.insert(self.session_cfg["origin"], 0, 0)
            self.session_cfg["shear"] = np.insert(self.session_cfg["origin"], 0, 0)
            _rot = np.eye(self.session_cfg["ndim"])
            _rot[-2:, -2:] = self.session_cfg["rotate"]
            self.session_cfg["rotate"] = _rot

        # Compute the overall spacing when considering both, affine and scale transform
        self.session_cfg["spacing"] = np.array(self.session_cfg["scale"]) * np.array(
            self.session_cfg["affine"].scale
        )

        # Create the target label array and layer
        self._data_result = np.zeros(self.session_cfg["shape"], dtype=np.uint8)

        # Add Layer
        self.object_index = 0
        if self.label_layer_name in self._viewer.layers:
            self._viewer.layers.remove(self.label_layer_name)
        self.add_label_layer(self._data_result, self.label_layer_name)

        # Lock the Session
        self._lock_session()
        self._is_initialized = True

    def on_reset_interactions(self):
        """Reset only the current interaction"""
        super().on_reset_interactions()
        self.on_layer_selected()

    def on_next(self) -> None:
        """
        Prepares the next label layer for interactions in the viewer.

        Retrieves the index of the last labeled object, renames the current label layer with
        this index, unbinds the original data by creating a deep copy, and clears all interaction
        layers. A new label layer with an updated colormap is then added to the viewer.
        """
        # Rename the current layer and add a new one
        label_layer = self._viewer.layers[self.label_layer_name]
        if not self.instance_aggregation_ckbx.isChecked():

            _name = f"object {self.object_index+1} - {self.session_cfg['name']}"
            self.add_label_layer(label_layer.data.copy(), _name)
            self._viewer.layers[_name].colormap = self.colormap[self.object_index]

        else:
            _sem_name = f"semantic map - {self.session_cfg['name']}"
            if _sem_name not in self._viewer.layers:
                self.add_label_layer(np.zeros_like(label_layer.data, dtype=np.uint16), _sem_name)

            sem_layer = self._viewer.layers[_sem_name]

            sem_layer.data[label_layer.data == 1] = self.object_index + 1
            sem_layer.refresh()

        self.object_index += 1
        label_layer.colormap = self.colormap[self.object_index]

        self._clear_layers()
        self.prompt_button._uncheck()
        self.prompt_button._check(0)

    def on_prompt_selected(self) -> None:
        """
        Updates the prompt index for each layer in the viewer based on the selected prompt.

        Iterates through the layers specified in `layer_dict`, sets the prompt index for each
        corresponding layer using the current prompt button selection, and refreshes each layer to
        apply the updated prompt.
        """
        for layer_name in self.layer_dict.values():
            if layer_name in self._viewer.layers:
                self._viewer.layers[layer_name].set_prompt(self.prompt_button.index)
                self._viewer.layers[layer_name].refresh()

    def on_interaction_selected(self) -> None:
        """
        Activates or creates a layer based on the selected interaction type.

        If a layer of the specified `interaction_type` already exists, it is activated;
        otherwise, a new layer is created.
        """
        self.interaction_type = self.interaction_button.index
        layer_name = self.layer_dict.get(self.interaction_type)

        if layer_name is not None and layer_name in self._viewer.layers:  # Activate the Layer
            self._viewer.layers.selection.clear()
            self._viewer.layers.selection.add(self._viewer.layers[layer_name])
            self._viewer.layers.selection.active = self._viewer.layers[layer_name]

            self._viewer.layers.selection.events.active(value=self._viewer.layers[layer_name])

        elif self.interaction_type == 0:  # Add Point Layer
            self.add_point_layer()
        elif self.interaction_type == 1:  # Add BBox Layer
            self.add_bbox_layer()
        elif self.interaction_type == 2:  # Add Scrible Layer
            self.add_scribble_layer()
        elif self.interaction_type == 3:  # Add Lasso Layer
            self.add_lasso_layer()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._viewer.window._qt_viewer.setFocus()

    def on_run(self):
        if self.session is None:
            return
            
        # Reduced memory footprint if possible
        memory_needed = 12  # GB, adjusted based on model requirements
        
        try:
            lock_manager = GPUMemoryLock(memory_per_lock=12)
            
            # First try non-blocking
            cuda_idx, locks = lock_manager.acquire_memory(memory_needed)
            
            if locks:
                # Non-blocking acquisition succeeded
                try:
                    # Set a timeout for prediction
                    with timeout(30):  # 30s timeout to leave buffer for cleanup
                        self.session._predict()
                except TimeoutError:
                    show_warning("Prediction timed out (30s limit)")
                finally:
                    lock_manager.release_locks(locks)
            else:
                # Try blocking acquisition with shorter timeout
                locks_needed = max(1, int((memory_needed + lock_manager.memory_per_lock - 1) / 
                                        lock_manager.memory_per_lock))
                acquired = []
                
                # Use a shorter timeout to ensure we have time for prediction
                lock_timeout = 5  # seconds
                
                try:
                    # Only use the first visible GPU when blocking
                    for i in range(locks_needed):
                        lock_file = lock_manager.lock_files.get((0, i))
                        if lock_file:
                            lock = FileLock(lock_file, timeout=lock_timeout)
                            lock.acquire()
                            acquired.append(lock)
                    
                    # Set timeout for prediction (remaining time)
                    with timeout(25):  # 25s for prediction
                        self.session._predict()
                except TimeoutError:
                    show_warning("Prediction timed out")
                finally:
                    lock_manager.release_locks(acquired)
                    
        except Exception as e:
            print(f"Error during prediction: {e}")
        finally:
            # Ensure UI is refreshed even if an exception occurred
            self._viewer.layers[self.label_layer_name].refresh()

    def on_interaction(self, event: Any):
        if (
            self.add_ckbx.isChecked()
            and event.action == ActionType.ADDED
            and not self._viewer.layers[event.source.name].is_free()
        ):
            self._viewer.layers[event.source.name].refresh()

            self.add_interaction()

    def on_layer_selected(self, *args, **kwargs) -> None:
        """
        Updates the interaction button and sets the `interaction_type` based on
        the currently selected layer in the viewer.

        Args:
            *args: Additional arguments for the method.
            **kwargs: Additional keyword arguments for the method.
        """
        _layer = self._viewer.layers.selection.active

        if _layer is None:
            key = None
        else:
            key = next((k for k, v in self.layer_dict.items() if v == _layer.name), None)

        self.interaction_type = key
        self.interaction_button._uncheck()
        self.interaction_button._check(self.interaction_type)

    # Inference Behaviour
    def inference(self, data: Any, index: int) -> None:
        """
        Performs inference on the provided data.

        Args:
            data: The data obtained from the layer's run method.
            index (int): The index of the layer type, corresponding to the layer_dict key.
        """
        print(
            f"Inference for interaction {index} and prompt {self.prompt_button.index == 0} and valid data {data is not None} "
        )

    def _export(self) -> None:
        """Export all Label layers belonging to the current image & model pair as separate files
        using the napari plugins"""
        timestamp_now = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
        _img_layer = self._viewer.layers[self.source_cfg["name"]]

        _path = _img_layer.source.path
        if _path is not None:
            # Get the dtype from the input file
            _img_file = Path(_path).name
            _dtype = ".nii.gz" if str(_img_file).endswith(".nii.gz") else Path(_img_file).suffix
            _output_file = _img_file.replace(_dtype, "")
        else:
            # If nothing is defined we save as .nii.gz
            _dtype = ".nii.gz"
            _output_file = self.source_cfg["name"] + _dtype

        _dialog = QFileDialog(self)
        _dialog.setDirectory(os.getcwd())

        _output_dir = _dialog.getExistingDirectory(
            self,
            "Select an Output Directory",
            options=QFileDialog.DontUseNativeDialog | QFileDialog.ShowDirsOnly,
        )

        if _output_dir == "":
            return

        elif Path(_output_dir).is_dir():
            _output_dir = Path(_output_dir).joinpath(f"{_output_file}_nnInteractive_{timestamp_now}")
            Path(_output_dir).mkdir(exist_ok=True)

            for _layer in self._viewer.layers:
                if self.label_layer_name == _layer.name and np.any(_layer.data):
                    _index = determine_layer_index(
                        names=[
                            layer.name for layer in self._viewer.layers if isinstance(layer, Labels)
                        ],
                        prefix="object ",
                        postfix=f" - {self.source_cfg['name']}",
                    )
                    _file_name = f"{_output_file}_{str(_index).zfill(4)}{_dtype}"
                elif _layer.name.startswith("object ") and _layer.name.endswith(
                    f" - {self.source_cfg['name']}"
                ):
                    _index = int(
                        _layer.name.replace("object ", "").replace(
                            f" - {self.source_cfg['name']}", ""
                        )
                    )
                    _file_name = f"{_output_file}_{str(_index).zfill(4)}{_dtype}"
                elif f"semantic map - {self.session_cfg['name']}" == _layer.name:
                    _file_name = f"{_output_file}_semantic_map{_dtype}"
                elif "Merged Objects" in _layer.name:
                    _file_name = f"{_output_file}_merged_objects{_dtype}"
                else:
                    continue

                # _file_name = f"{_output_file}_{str(_index).zfill(4)}{_dtype}"
                _file = str(Path(_output_dir).joinpath(_file_name))

                # reverse the corrections for non-orthogonal data and convert dummy 3d back to 2d
                _data = _layer.data[0] if self.source_cfg["ndim"] == 2 else _layer.data
                _layer_temp = Labels(
                    _data,
                    name="_temp",
                    affine=self.source_cfg["affine"],
                    scale=self.source_cfg["scale"],
                    translate=self.source_cfg["translate"],
                    rotate=self.source_cfg["rotate"],
                    shear=self.source_cfg["shear"],
                    metadata=self.source_cfg["metadata"],
                )

                _layer_temp._source = self.source_cfg["source"]
                _layer_temp.save(_file)
                del _layer_temp
        else:
            raise ValueError("Output path has to be a directory, not a file")

    # activity event filter
    def eventFilter(self, obj, event):
        """
        Event filter to track user activity.
        This method is called for every event in the application.
        """
        
        activity_events = [
            # QEvent.MouseButtonPress, QEvent.MouseButtonRelease, QEvent.MouseMove,
            QEvent.KeyPress, QEvent.KeyRelease
        ]
        
        if event.type() in activity_events:
            self.last_activity_time = datetime.datetime.now()
            
        return False

    def start_inactivity_timer(self):
        """Start the inactivity detection timer."""
        # Check for inactivity every minute
        self.inactivity_timer.start(60000)
    
    def stop_inactivity_timer(self):
        """Stop the inactivity detection timer."""
        self.inactivity_timer.stop()
    
    def set_inactivity_timeout(self, minutes: int):
        """Set the inactivity timeout in minutes."""
        self.inactivity_timeout_minutes = minutes

    def _handle_inactivity(self):
        """
        Handle inactivity by auto-saving and presenting a notice to the user.
        The application will automatically quit if the user doesn't press any key
        within 1 minute. Any key press will continue the session.
        """
        current_time = datetime.datetime.now()
        elapsed_minutes = (current_time - self.last_activity_time).total_seconds() / 60
        
        if elapsed_minutes >= self.inactivity_timeout_minutes:
            self._auto_save()
            self.inactivity_timer.stop()
            
            message = f"No activity detected for {self.inactivity_timeout_minutes} minutes.\n"
            message += "Your work has been auto-saved."
            
            class KeyPressDialog(QDialog):
                def __init__(self, parent=None):
                    super().__init__(parent)
                    self.setWindowTitle("Inactivity Detected")
                    self.setWindowFlag(Qt.WindowStaysOnTopHint)
                    self.setMinimumWidth(400)
                    
                    layout = QVBoxLayout()
                    
                    messageLabel = QLabel(message)
                    messageLabel.setAlignment(Qt.AlignCenter)
                    layout.addWidget(messageLabel)
                    
                    keyPressLabel = QLabel("Press any key to continue working")
                    keyPressLabel.setAlignment(Qt.AlignCenter)
                    keyPressLabel.setStyleSheet("font-weight: bold; color: blue; font-size: 14px;")
                    layout.addWidget(keyPressLabel)
                    
                    self.countdownLabel = QLabel("Time remaining: 60 seconds")
                    self.countdownLabel.setAlignment(Qt.AlignCenter)
                    layout.addWidget(self.countdownLabel)
                    
                    autoCloseLabel = QLabel("Application will close automatically if no key is pressed")
                    autoCloseLabel.setAlignment(Qt.AlignCenter)
                    autoCloseLabel.setStyleSheet("color: gray;")
                    layout.addWidget(autoCloseLabel)
                    
                    self.setLayout(layout)
                
                def keyPressEvent(self, event):
                    # Any key press will continue the session
                    self.accept()
            
            dialog = KeyPressDialog(self)
            
            seconds_left = 30
            countdownTimer = QTimer()
            
            def update_countdown():
                nonlocal seconds_left
                seconds_left -= 1
                dialog.countdownLabel.setText(f"Time remaining: {seconds_left} seconds")
                
                if seconds_left <= 5:
                    dialog.countdownLabel.setStyleSheet("color: red; font-weight: bold;")
                
                if seconds_left <= 0:
                    countdownTimer.stop()
                    dialog.reject()
            
            countdownTimer.timeout.connect(update_countdown)
            countdownTimer.start(1000)
            
            result = dialog.exec_()
            countdownTimer.stop()
            
            if result == QDialog.Accepted:
                self.last_activity_time = datetime.datetime.now()
                self.inactivity_timer.start(60000)
            else:                
                QTimer.singleShot(500, lambda: sys.exit(0))

    def start_auto_save_timer(self):
        """Start the auto-save timer with the current interval."""
        interval_ms = self.auto_save_interval_minutes * 60000
        self.auto_save_timer.start(interval_ms)
    
    def stop_auto_save_timer(self):
        self.auto_save_timer.stop()
    
    def set_auto_save_interval(self, minutes: int):
        self.auto_save_interval_minutes = minutes
        # Restart timer with new interval if it's running
        if self.auto_save_timer.isActive():
            self.stop_auto_save_timer()
            self.start_auto_save_timer()

    def _auto_save(self) -> None:
        """Export all Label layers with timestamp in the filename."""
        # Generate timestamp for the current auto-save
        timestamp_now = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")

        if self._is_initialized:
            _img_layer = self._viewer.layers[self.source_cfg["name"]]

            _output_dir = _img_layer.source.path
            if _output_dir is None:
                show_warning("Auto-save failed: No source path found for the image layer.")
                return

            # Get the dtype from the input file
            _img_file = Path(_output_dir).name
            _dtype = ".nii.gz" if str(_img_file).endswith(".nii.gz") else Path(_img_file).suffix
            _output_file = _img_file.replace(_dtype, "")
            
            _parent_dir = Path(_output_dir).parent
            _output_dir = _parent_dir.joinpath(f"{_output_file}_autosave_{timestamp_now}")
            
            try:
                Path(_output_dir).mkdir(exist_ok=True)
                
                saved_files = []
                for _layer in self._viewer.layers:
                    if self.label_layer_name == _layer.name and np.any(_layer.data):
                        _index = determine_layer_index(
                            names=[
                                layer.name for layer in self._viewer.layers if isinstance(layer, Labels)
                            ],
                            prefix="object ",
                            postfix=f" - {self.source_cfg['name']}",
                        )
                        _file_name = f"{_output_file}_{str(_index).zfill(4)}{_dtype}"
                    elif _layer.name.startswith("object ") and _layer.name.endswith(
                        f" - {self.source_cfg['name']}"
                    ):
                        _index = int(
                            _layer.name.replace("object ", "").replace(
                                f" - {self.source_cfg['name']}", ""
                            )
                        )
                        _file_name = f"{_output_file}_{str(_index).zfill(4)}{_dtype}"
                    elif f"semantic map - {self.session_cfg['name']}" == _layer.name:
                        _file_name = f"{_output_file}_semantic_map{_dtype}"
                    elif "Merged Objects" in _layer.name:
                        _file_name = f"{_output_file}_merged_objects{_dtype}"
                    else:
                        continue

                    _file = str(Path(_output_dir).joinpath(_file_name))
                    saved_files.append(_file_name)

                    # reverse the corrections for non-orthogonal data and convert dummy 3d back to 2d
                    _data = _layer.data[0] if self.source_cfg["ndim"] == 2 else _layer.data
                    _layer_temp = Labels(
                        _data,
                        name="_temp",
                        affine=self.source_cfg["affine"],
                        scale=self.source_cfg["scale"],
                        translate=self.source_cfg["translate"],
                        rotate=self.source_cfg["rotate"],
                        shear=self.source_cfg["shear"],
                        metadata=self.source_cfg["metadata"],
                    )

                    _layer_temp._source = self.source_cfg["source"]
                    _layer_temp.save(_file)
                    del _layer_temp
                
                if saved_files:
                    show_warning(f"Auto-saved {len(saved_files)} files at {timestamp_now}")
            except Exception as e:
                show_warning(f"Auto-save failed: {str(e)}")
        else:
            show_warning(f"Not initialized -- auto-save does nothing")
