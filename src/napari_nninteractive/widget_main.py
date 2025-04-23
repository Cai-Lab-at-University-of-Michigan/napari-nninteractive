import os
import napari
import warnings
from pathlib import Path
from typing import Any, Optional

import nnInteractive
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from napari.utils.notifications import show_warning
from napari.viewer import Viewer
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from qtpy.QtWidgets import QWidget

from napari_nninteractive.widget_controls import LayerControls


class nnInteractiveWidget_(LayerControls):
    """Just a Debug Dummy without all the machine learning stuff"""


class nnInteractiveWidget(LayerControls):
    """
    A widget for the nnInteractive plugin in Napari that manages model inference sessions
    and allows interactive layer-based actions.
    """

    def __init__(self, viewer: Viewer, parent: Optional[QWidget] = None):
        """
        Initialize the nnInteractiveWidget.
        """
        super().__init__(viewer, parent)
        self.session = None
        self._viewer.dims.events.order.connect(self.on_axis_change)

    # Event Handlers
    def on_init(self, *args, **kwargs):
        """
        Initialize the inference session and setup layers for interaction.

        This method sets up the nnInteractiveInferenceSession, loading from a
        pre-trained model folder and initializing properties based on the viewer layer.
        """
        super().on_init(*args, **kwargs)
        if self.session is None:
            # Get inference class from Checkpoint
            if Path(self.checkpoint_path).joinpath("inference_session_class.json").is_file():
                inference_class = load_json(
                    Path(self.checkpoint_path).joinpath("inference_session_class.json")
                )
                if isinstance(inference_class, dict):
                    inference_class = inference_class["inference_class"]
            else:
                inference_class = "nnInteractiveInferenceSession"

            inference_class = recursive_find_python_class(
                join(nnInteractive.__path__[0], "inference"),
                inference_class,
                "nnInteractive.inference",
            )

            # CPU Fallback if noc Cuda is available
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                show_warning(
                    "Cuda is not available. Using CPU instead. This will result in longer runtimes and additionally auto-zoom will be disabled for runtime reasons"
                )

                device = torch.device("cpu")
                self.propagate_ckbx.setChecked(False)

            # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

            # Initialize the Session
            self.session = inference_class(
                device=device,  # can also be cpu or mps. CPU not recommended
                use_torch_compile=False,
                torch_n_threads=os.cpu_count(),
                verbose=False,
                do_autozoom=self.propagate_ckbx.isChecked(),
            )

            self.session.initialize_from_trained_model_folder(
                self.checkpoint_path,
                0,
                "checkpoint_final.pth",
            )

        _data = np.array(self._viewer.layers[self.session_cfg["name"]].data)
        _data = _data[np.newaxis, ...]

        if self.source_cfg["ndim"] == 2:
            _data = _data[np.newaxis, ...]

        self.session.set_image(_data, {"spacing": self.session_cfg["spacing"]})

        self.session.set_target_buffer(self._data_result)
        self._scribble_brush_size = self.session.preferred_scribble_thickness[
            self._viewer.dims.not_displayed[0]
        ]
        # Set the prompt type to positive
        self.prompt_button._uncheck()
        self.prompt_button._check(0)

    def on_model_selected(self):
        """Reset the current session completely"""
        super().on_model_selected()
        self.session = None

    def on_image_selected(self):
        """Reset the current sessions interaction but keep the session itself"""
        super().on_image_selected()
        if self.session is not None:
            self.session.reset_interactions()

    def on_reset_interactions(self):
        """Reset only the current interaction"""
        _ind = self.interaction_button.index
        super().on_reset_interactions()
        if self.session is not None:
            self.session.reset_interactions()

        self._viewer.layers[self.label_layer_name].refresh()

        self.interaction_button._check(_ind)
        self.on_interaction_selected()
        # self.prompt_button._uncheck()
        self.prompt_button._on_button_pressed(0)

    def on_next(self):
        """Reset the Interactions of current session"""
        _ind = self.interaction_button.index
        super().on_next()
        if self.session is not None:
            self.session.reset_interactions()

        # if (
        #     self.use_init_ckbx.isChecked()
        #     and self.label_for_init.currentText() in self._viewer.layers
        # ):
        #     self.init_with_mask()

        self._viewer.layers[self.label_layer_name].refresh()

        self.interaction_button._check(_ind)
        self.on_interaction_selected()
        self.prompt_button._check(0)

    def on_propagate_ckbx(self, *args, **kwargs):
        if self.session is not None:
            self.session.set_do_autozoom(self.propagate_ckbx.isChecked())

    def on_axis_change(self, event: Any):
        """Change the brush size of the scribble layer when the axis changes"""
        if self.session is not None:
            self._scribble_brush_size = self.session.preferred_scribble_thickness[
                self._viewer.dims.not_displayed[0]
            ]
            if self.scribble_layer_name in self._viewer.layers:
                self._viewer.layers[self.scribble_layer_name].brush_size = self._scribble_brush_size

    # Inference Behaviour

    def add_interaction(self):
        _index = self.interaction_button.index
        _layer_name = self.layer_dict.get(_index)
        if (
            _layer_name is not None
            and _layer_name in self._viewer.layers
            and not self._viewer.layers[_layer_name].is_free()
        ):
            data = self._viewer.layers[_layer_name].get_last()

            self._viewer.layers[_layer_name].run()
            # self.inference(_data, _index)

            if data is not None:
                _prompt = self.prompt_button.index == 0
                _auto_run = self.run_ckbx.isChecked()

                if _index == 0:
                    self._viewer.layers[self.point_layer_name].refresh(force=True)
                    self.session.add_point_interaction(data, _prompt, _auto_run)
                elif _index == 1:
                    # add_bbox_interaction expects [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
                    _min = np.min(data, axis=0)
                    _max = np.max(data, axis=0)
                    bbox = [[_min[0], _max[0]], [_min[1], _max[1]], [_min[2], _max[2]]]
                    self.session.add_bbox_interaction(bbox, _prompt, _auto_run)
                elif _index == 2:
                    self.session.add_scribble_interaction(data, _prompt, _auto_run)
                elif _index == 3:
                    self.session.add_lasso_interaction(data, _prompt, _auto_run)

                self._viewer.layers[self.label_layer_name].refresh()

    def on_load_semantic_mask(self):
        selected_layers = list(self._viewer.layers.selection)
        if len(selected_layers) != 1 or not isinstance(selected_layers[0], napari.layers.Labels):
            warnings.warn("Please select exactly one Labels layer", UserWarning, stacklevel=1)
            return
        
        layer = selected_layers[0]
        _layer_data = layer.data
        
        if (_layer_data.shape != self.session_cfg["shape"]):
            warnings.warn("Shape mismatch with session configuration", UserWarning, stacklevel=1)
            return

        semantic_layer_name = f"semantic map - {self.session_cfg['name']}"
        is_semantic_label_layer_existed = any(semantic_layer_name == l.name for l in self._viewer.layers)

        if np.any(_layer_data):
            if is_semantic_label_layer_existed:
                warnings.warn("Replacing the current semantic label layer", UserWarning, stacklevel=1)
                self._viewer.layers[semantic_layer_name].data = _layer_data
                self._viewer.layers[semantic_layer_name].refresh()
            else:
                self.add_label_layer(_layer_data, semantic_layer_name)
            
            self._viewer.layers.remove(layer.name)
        else:
            warnings.warn("No annotation found - result would be empty", UserWarning, stacklevel=1)

    def on_delete_mask(self):
        selected_layers = list(self._viewer.layers.selection)
        if len(selected_layers) != 1 or not isinstance(selected_layers[0], napari.layers.Labels):
            warnings.warn("Please select exactly one Labels layer", UserWarning, stacklevel=1)
            return
            
        layer = selected_layers[0]
        name = layer.name
        target_class = self.class_for_init.value()
        
        _layer_data = layer.data
        mask_indices = np.where(_layer_data == target_class)
        
        if len(mask_indices[0]) > 0:  # Check if any pixels match
            _layer_data[mask_indices] = 0
            layer.refresh()
        else:
            warnings.warn("Selected class is not valid in the layer", UserWarning, stacklevel=1)

    def on_load_mask(self):
        selected_layers = list(self._viewer.layers.selection)
        if len(selected_layers) != 1 or not isinstance(selected_layers[0], napari.layers.Labels):
            warnings.warn("Please select exactly one Labels layer", UserWarning, stacklevel=1)
            return
        
        layer = selected_layers[0]
        _layer_data = layer.data
        
        if (_layer_data.shape != self.session_cfg["shape"]):
            warnings.warn("Shape mismatch with session configuration", UserWarning, stacklevel=1)
            return

        target_class = self.class_for_init.value()
        mask_indices = np.where(_layer_data == target_class)
        
        if len(mask_indices[0]) > 0:  # Check if any pixels match
            if self.session is not None:
                mask = np.zeros(_layer_data.shape, dtype=np.uint8)
                mask[mask_indices] = 1
                
                self.session.add_initial_seg_interaction(
                    mask, run_prediction=self.auto_refine.isChecked()
                )
                self._viewer.layers[self.label_layer_name].refresh()
        else:
            warnings.warn("Mask is not valid - probably its empty", UserWarning, stacklevel=1)

    def on_merge_mask(self):
        shape = self.session_cfg["shape"]
        selected_layers = list(self._viewer.layers.selection)

        if len(selected_layers) < 2:
            warnings.warn("Please select at least two layers to merge", UserWarning, stacklevel=1)
            return
            
        merge_to_layer = selected_layers[0]
        merge_to_layer_name = merge_to_layer.name
        
        result_data = merge_to_layer.data
        
        layers_to_remove = []
        
        for layer in selected_layers[1:]:  # Skip the first layer (target)
            if (layer.data.shape != shape or 
                layer.name in [merge_to_layer_name, self.archive_layer_name] or 
                not isinstance(layer, napari.layers.Labels) or 
                np.amax(layer.data) == 0):
                continue
            
            nonzero_indices = np.where(layer.data > 0)
            
            if len(nonzero_indices[0]) > 0:
                result_data[nonzero_indices] = 1
                layers_to_remove.append(layer.name)
        
        for layer_name in layers_to_remove:
            self._viewer.layers.remove(layer_name)
        
        if np.any(result_data > 0):
            merge_to_layer.refresh()

    def on_archive_object(self):
        warnings.warn("Overlapped mask region will be overridden", UserWarning, stacklevel=1)

        shape = self.session_cfg["shape"]
        selected_layers = list(self._viewer.layers.selection)
        
        is_merged_object_existed = any(self.archive_layer_name == l.name for l in self._viewer.layers)
        
        # Process in two phases to reduce memory usage
        valid_layers = []
        need_new_layer = not is_merged_object_existed
        
        for layer in selected_layers:
            if (layer.data.shape != shape or 
                layer.name == self.archive_layer_name or 
                not isinstance(layer, napari.layers.Labels) or 
                np.amax(layer.data) == 0):
                continue
            valid_layers.append(layer)
        
        if not valid_layers:
            warnings.warn("No valid layers selected for archiving", UserWarning, stacklevel=1)
            return
        
        if need_new_layer:
            archive_data = np.zeros(shape, dtype=np.uint32)
            global_id_now = 1
        else:
            archive_data = self._viewer.layers[self.archive_layer_name].data
            global_id_now = np.amax(archive_data) + 1
            if valid_layers:
                archive_data = archive_data.copy()
        
        for layer in valid_layers:
            _layer_data = layer.data
            
            unique_ids = np.unique(_layer_data)
            unique_ids = unique_ids[unique_ids > 0]
            
            id_mapping = {int(old_id): global_id_now + i for i, old_id in enumerate(unique_ids)}
            
            for old_id in unique_ids:
                id_indices = np.where(_layer_data == old_id)
                if len(id_indices[0]) > 0:
                    archive_data[id_indices] = id_mapping[int(old_id)]
            
            global_id_now += len(unique_ids)
            
            if layer.name not in [self.archive_layer_name, self.label_layer_name]:
                self._viewer.layers.remove(layer.name)
        
        if np.any(archive_data):
            if is_merged_object_existed:
                self._viewer.layers[self.archive_layer_name].data = archive_data
                self._viewer.layers[self.archive_layer_name].refresh()
            else:
                self.add_label_layer(archive_data, self.archive_layer_name)
        else:
            warnings.warn("No objects were archived - result would be empty", UserWarning, stacklevel=1)
