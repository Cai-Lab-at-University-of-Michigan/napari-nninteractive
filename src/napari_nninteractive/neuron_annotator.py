"""
region_grow_plugin_v2.3.4_layout_hotfix.py

[ v2.3.4 紧急修复日志 ]
1. [Fix] 修复 MagicGUI 布局报错 (NotImplementedError):
   - 错误原因: 旧版 MagicGUI 不支持在容器创建后修改 layout 属性。
   - 修复: 将 layout="vertical" 直接作为参数传入 Container 构造函数。
2. [UI] 保持 v2.3.3 的所有美化 (三明治布局、HTML居中、动态滑块)。
"""

import numpy as np
import napari
import json
import SimpleITK as sitk
from skimage.segmentation import flood, watershed
from scipy.ndimage import gaussian_gradient_magnitude, binary_dilation, binary_erosion
from scipy.spatial.distance import cdist
from magicgui import magic_factory
from magicgui.widgets import (
    Container, PushButton, Label, CheckBox, FloatSlider, RadioButtons, IntSlider, ComboBox
)
from napari.layers import Image
from qtpy.QtWidgets import QFileDialog, QScrollArea

# ==================== 核心标注类 ====================

class NeuronAnnotator:
    def __init__(self, viewer: napari.Viewer, image_layer: Image):
        self.viewer = viewer
        self.image_layer = image_layer
        
        # 1. 单例检查
        if self._check_existing_panel():
            print("[System] Annotator already running. Refocusing.")
            return

        self.viewer.status = f"[System] Initializing for '{image_layer.name}'..."
        
        # 2. 数据初始化
        self.segmentations = {}
        self.current_seg_id = None
        self.next_seg_id = 1
        self.protect_others = True
        self.current_mode = "add"
        self.SEED_COLORS = ["cyan", "magenta", "lime", "blue", "orange", "purple", "white"]
        
        # 核心参数
        self.tol = 0.1
        self.edge_thresh = 0.05
        self.enable_edge_limit = False
        
        self.push_force = 0 
        self.algo_mode = "Competitive (Watershed)" 
        
        self.is_frozen = False
        self.is_redoing = False

        # 3. 图像处理
        print(f"[Init] Processing layer data...")
        raw_data = np.asarray(image_layer.data, dtype=np.float32)
        low, high = np.percentile(raw_data, 1), np.percentile(raw_data, 99.9)
        if high <= low: low, high = raw_data.min(), raw_data.max()
        if high > low:
            self.image = (raw_data - low) / (high - low)
        else:
            self.image = raw_data - low
        self.image = np.clip(self.image, 0.0, 1.0)

        self.viewer.status = "[System] Calculating Gradient..."
        print("[Init] Calculating Gradient Magnitude...")
        raw_grad = gaussian_gradient_magnitude(self.image, sigma=1.0)
        g_min, g_max = raw_grad.min(), raw_grad.max()
        if g_max > g_min:
            self.grad_image = (raw_grad - g_min) / (g_max - g_min)
        else:
            self.grad_image = np.zeros_like(raw_grad)
        
        calc_thresh = float(np.percentile(self.grad_image, 98.0))
        self.edge_thresh = max(0.001, calc_thresh)

        # 4. 图层管理
        seg_name = f"Seg_{image_layer.name}"
        if seg_name in viewer.layers:
            self.labels_layer = viewer.layers[seg_name]
        else:
            self.labels_layer = viewer.add_labels(
                np.zeros(self.image.shape, dtype=np.int32),
                name=seg_name,
                opacity=0.6,
                scale=image_layer.scale,
                translate=image_layer.translate,
                rotate=image_layer.rotate,
                shear=image_layer.shear,
                affine=image_layer.affine
            )
        
        preview_name = "Negative Preview"
        if preview_name in viewer.layers:
            viewer.layers.remove(viewer.layers[preview_name])
            
        self.preview_layer = viewer.add_image(
            np.zeros(self.image.shape, dtype=np.uint8),
            name=preview_name,
            colormap="yellow",
            blending="additive",
            opacity=0.5,
            scale=image_layer.scale,
            translate=image_layer.translate,
            rotate=image_layer.rotate,
            shear=image_layer.shear,
            affine=image_layer.affine,
            visible=False 
        )

        self.edge_layer = None

        # 5. 创建首个 Seg
        self.create_new_segmentation()

        # 6. Auto Focus
        self.viewer.layers.selection.events.active.connect(self._on_layer_selection_change)

        # 7. UI Init
        self._setup_ui()
        self._setup_keybindings()
        
        self._update_status()
        self._update_current_points_color()
        
        self.viewer.status = "[System] Ready."

    def _check_existing_panel(self):
        dock_name = "Annotator Panel"
        try:
            if hasattr(self.viewer.window, '_dock_widgets'):
                widgets = self.viewer.window._dock_widgets
                if dock_name in widgets:
                    self.viewer.window.remove_dock_widget(widgets[dock_name])
                    return False 
        except Exception: pass
        return False

    def _on_layer_selection_change(self, event):
        if self.is_frozen: return
        active_layer = event.value
        for seg_id, seg in self.segmentations.items():
            if seg["points_layer"] == active_layer:
                if self.current_seg_id != seg_id:
                    if self.current_seg_id in self.segmentations:
                        old_seg = self.segmentations[self.current_seg_id]
                        old_seg["saved_selection"] = old_seg["points_layer"].selected_data

                    self.current_seg_id = seg_id
                    self._update_status()
                    self._restore_selection(seg_id)
                    self._update_current_points_color()
                    self._recompute_seg_from_seeds(seg_id)
                    seg["points_layer"].mode = 'add'
                return

    def _restore_selection(self, seg_id):
        seg = self.segmentations[seg_id]
        pts = seg["points_layer"]
        if "saved_selection" in seg and seg["saved_selection"]:
            pts.selected_data = seg["saved_selection"]
        elif len(pts.data) > 0:
            pts.selected_data = {len(pts.data) - 1}

    # ---------- UI 构建 ----------

    def _setup_ui(self):
        # [UI] 标题样式 - HTML Center
        def header_label(text):
            return Label(value=f"<div style='text-align: center; margin-top: 10px; margin-bottom: 5px; color: #888;'><b>{text}</b></div>")

        # [Fix] 使用 HTML 居中
        self.status_label = Label(value="<div style='text-align: center;'>Current: Seg 1</div>")
        
        # --- 组1: 基础工具 (需要标签对齐) ---
        self.mode_radio = RadioButtons(choices=["Add (+)", "Remove (-)"], value="Add (+)", label="Brush", orientation="horizontal")
        self.algo_selector = ComboBox(choices=["Competitive (Watershed)", "Local Cut (Constrained)"], value=self.algo_mode, label="Logic")
        self.push_slider = IntSlider(value=0, min=-30, max=30, step=1, name="Boundary Shift", tracking=False)
        
        # [Fix] layout="vertical" 必须在构造时传入
        group_tools = Container(widgets=[self.mode_radio, self.algo_selector, self.push_slider], labels=True, layout="vertical")

        # --- 组2: 生长参数 (需要标签对齐) ---
        self.tol_slider = FloatSlider(value=self.tol, min=0.0, max=1.0, step=0.05, name="New Tol")
        self.seed_tol_slider = FloatSlider(value=self.tol, min=0.0, max=1.0, step=0.05, name="Edit Tol")
        
        # [Fix] layout="vertical" 必须在构造时传入
        group_growth = Container(widgets=[self.tol_slider, self.seed_tol_slider], labels=True, layout="vertical")
        
        # 全宽按钮
        btn_apply_seed = PushButton(text="Update Selected Seed")

        # --- 组3: 约束 (需要标签对齐) ---
        self.cb_edge = CheckBox(value=False, text="Enable Edge Limit")
        self.edge_slider = FloatSlider(value=self.edge_thresh, min=0.001, max=1.0, step=0.005, name="Edge Thresh", enabled=False)
        
        # [Fix] layout="vertical" 必须在构造时传入
        group_constraints = Container(widgets=[self.cb_edge, self.edge_slider], labels=True, layout="vertical")
        
        # 全宽按钮与选项
        self.btn_preview_edge = PushButton(text="Preview Edges", enabled=False)
        cb_protect = CheckBox(value=True, text="Protect Others")
        self.cb_show_neg = CheckBox(value=False, text="Show Rejected Area", visible=False) 

        # --- 编辑操作 (全宽网格) ---
        btn_new = PushButton(text="New Seg")
        btn_prev = PushButton(text="Prev")
        btn_next = PushButton(text="Next")
        btn_undo = PushButton(text="Undo")
        btn_redo = PushButton(text="Redo")
        self.btn_freeze = PushButton(text="Freeze")
        btn_reset = PushButton(text="Reset")
        btn_delete = PushButton(text="Delete")
        btn_reset_all = PushButton(text="Reset All")
        
        btn_export_nii = PushButton(text="Export .nii.gz")
        btn_export_nii.native.setStyleSheet("background-color: #004444; color: white;")
        btn_save = PushButton(text="Save JSON")
        btn_load = PushButton(text="Load JSON")

        # [Logic] UI Update function
        def update_slider_ui():
            mode = self.algo_selector.value
            if mode == "Competitive (Watershed)":
                self.push_slider.label = "Shift (px)"
                self.push_slider.min = -30
                self.push_slider.max = 30
                self.push_slider.value = 0
            else:
                self.push_slider.label = "Radius (px)"
                self.push_slider.min = 1
                self.push_slider.max = 50
                self.push_slider.value = 5

        # Callbacks
        def on_mode_change(event): self._set_mode(self.mode_radio.value)
        def on_algo_change(event): 
            self.algo_mode = self.algo_selector.value
            update_slider_ui()
            if self.current_seg_id: self._recompute_seg_from_seeds(self.current_seg_id)
        def on_tol_change(event): self.tol = float(self.tol_slider.value)
        def on_push_change(event): 
            self.push_force = int(self.push_slider.value)
            if self.current_seg_id: self._recompute_seg_from_seeds(self.current_seg_id)
        def on_toggle_neg(event): self.preview_layer.visible = self.cb_show_neg.value
        def on_edge_enable(event):
            if self.is_frozen: return
            enabled = self.cb_edge.value
            self.enable_edge_limit = enabled
            self.edge_slider.enabled = enabled
            self.btn_preview_edge.enabled = enabled
            if self.current_seg_id: self._recompute_seg_from_seeds(self.current_seg_id)
        def on_edge_thresh_change(event):
            if self.is_frozen: return
            self.edge_thresh = float(self.edge_slider.value)
            if self.edge_layer and self.edge_layer in self.viewer.layers and self.edge_layer.visible:
                self._show_edge_preview()
            if self.current_seg_id: self._recompute_seg_from_seeds(self.current_seg_id)
        def toggle_edge_preview(event):
            if self.edge_layer is None or self.edge_layer not in self.viewer.layers:
                self._show_edge_preview()
                self.btn_preview_edge.text = "Hide Preview"
            else:
                vis = not self.edge_layer.visible
                self.edge_layer.visible = vis
                self.btn_preview_edge.text = "Hide Preview" if vis else "Preview Edges"

        # Connections
        self.mode_radio.changed.connect(on_mode_change)
        self.algo_selector.changed.connect(on_algo_change)
        self.tol_slider.changed.connect(on_tol_change)
        self.push_slider.changed.connect(on_push_change)
        self.cb_show_neg.changed.connect(on_toggle_neg)
        btn_apply_seed.clicked.connect(self._on_apply_seed_tol)
        self.cb_edge.changed.connect(on_edge_enable)
        self.edge_slider.changed.connect(on_edge_thresh_change)
        self.btn_preview_edge.clicked.connect(toggle_edge_preview)
        cb_protect.changed.connect(lambda e: setattr(self, 'protect_others', cb_protect.value))
        btn_new.clicked.connect(self._on_btn_new)
        btn_undo.clicked.connect(lambda e: self.undo_last_seed())
        btn_redo.clicked.connect(lambda e: self.redo_last_action())
        btn_prev.clicked.connect(lambda e: self.switch_segmentation(-1))
        btn_next.clicked.connect(lambda e: self.switch_segmentation(+1))
        self.btn_freeze.clicked.connect(self._toggle_freeze)
        btn_reset.clicked.connect(self.reset_current_seg)
        btn_delete.clicked.connect(self.delete_current_seg)
        btn_reset_all.clicked.connect(self.reset_all)
        btn_save.clicked.connect(self.save_project)
        btn_load.clicked.connect(self.load_project)
        btn_export_nii.clicked.connect(self.export_nifti)

        btn_reset_all.native.setStyleSheet("background-color: #550000; color: white;") 
        
        # [UI Layout] 2x2 Grids
        row_edit_1 = Container(layout="horizontal", widgets=[btn_undo, btn_redo], labels=False)
        row_edit_2 = Container(layout="horizontal", widgets=[btn_reset, btn_delete], labels=False)
        row_nav = Container(layout="horizontal", widgets=[btn_prev, btn_new, btn_next], labels=False)
        row_file = Container(layout="horizontal", widgets=[btn_save, btn_load], labels=False)

        # [UI MAIN] labels=False 启用 "Full Width" 模式
        container = Container(widgets=[
            Label(value="<div style='text-align: center'><b>Neuron Annotator v2.3.4</b></div>"),
            self.status_label,
            
            group_tools, # 含 Brush, Logic, Shift (内部对齐)
            
            header_label("Growth Parameters"),
            group_growth, # 含 Tols (内部对齐)
            btn_apply_seed, # 全宽按钮
            
            header_label("Constraints"),
            group_constraints, # 含 Edge 设置 (内部对齐)
            self.btn_preview_edge, # 全宽按钮
            cb_protect, # 全宽复选框
            
            header_label("Edit Operations"),
            row_nav, # 全宽行
            row_edit_1, # 全宽行
            row_edit_2, # 全宽行
            self.btn_freeze, # 全宽按钮
            
            header_label("Project & Export"),
            btn_export_nii,
            row_file,
            btn_reset_all
        ], labels=False)
        
        container.max_width = 320

        scroll_area = QScrollArea()
        scroll_area.setWidget(container.native)
        scroll_area.setWidgetResizable(True)
        
        self.viewer.window.add_dock_widget(scroll_area, area="right", name="Annotator Panel")
        update_slider_ui()

    def _toggle_freeze(self):
        if not self.is_frozen:
            self.is_frozen = True
            self.btn_freeze.text = "Unfreeze"
            self.btn_freeze.native.setStyleSheet("background-color: #AA0000; color: white;")
            self.viewer.status = "[System] Frozen. Use Paint Brush to edit."
            self.tol_slider.enabled = False
            self.push_slider.enabled = False
            self.mode_radio.enabled = False
            self.algo_selector.enabled = False
            self.preview_layer.visible = False
        else:
            self.is_frozen = False
            self.btn_freeze.text = "Freeze"
            self.btn_freeze.native.setStyleSheet("")
            self.viewer.status = "[System] Algorithm Mode."
            self.tol_slider.enabled = True
            self.push_slider.enabled = True
            self.mode_radio.enabled = True
            self.algo_selector.enabled = True
            self.preview_layer.visible = self.cb_show_neg.value
            if self.current_seg_id:
                self._recompute_seg_from_seeds(self.current_seg_id)

    # ---------- 核心算法 ----------

    def _get_ellipsoid_footprint(self, radius):
        """Mode 1 Morphology"""
        if radius <= 0: return None
        scale = np.array(self.image_layer.scale)
        scale_ratio = scale / scale.min()
        radii_px = radius / scale_ratio
        r_z, r_y, r_x = radii_px
        z, y, x = np.mgrid[-r_z:r_z+1, -r_y:r_y+1, -r_x:r_x+1]
        ellipsoid = (z/r_z)**2 + (y/r_y)**2 + (x/r_x)**2 <= 1
        return ellipsoid

    def _get_flood_mask(self, seeds, global_tol):
        """Basic Flood Fill"""
        if not seeds: return None
        full_mask = None
        for s in seeds:
            seed_idx = tuple(int(round(c)) for c in s["index"])
            if not (0 <= seed_idx[0] < self.image.shape[0] and 
                    0 <= seed_idx[1] < self.image.shape[1] and 
                    0 <= seed_idx[2] < self.image.shape[2]): continue
            
            t = s["tol"] if "tol" in s else global_tol
            
            try:
                mask = flood(self.image, seed_point=seed_idx, tolerance=t)
                if self.enable_edge_limit:
                    mask &= (self.grad_image <= self.edge_thresh)
                if full_mask is None: full_mask = mask
                else: full_mask |= mask
            except: pass
        return full_mask

    def _recompute_seg_from_seeds(self, seg_id):
        if self.is_frozen: return

        seg = self.segmentations[seg_id]
        label_val = seg["label_value"]
        seeds = seg["seeds"]
        
        labels = self.labels_layer.data
        labels[labels == label_val] = 0 
        self.preview_layer.data[:] = 0 
        
        pos_seeds = [s for s in seeds if s.get("type", "add") == "add"]
        neg_seeds = [s for s in seeds if s.get("type", "add") == "remove"]

        if not pos_seeds:
            self.labels_layer.refresh()
            self.preview_layer.refresh()
            return

        # 1. Pos Base
        mask_pos = self._get_flood_mask(pos_seeds, self.tol)
        if mask_pos is None: 
            self.labels_layer.refresh()
            return

        final_pos_mask = mask_pos
        final_neg_mask = np.zeros_like(mask_pos, dtype=bool)

        # 2. Negative Logic
        if neg_seeds:
            if self.algo_mode == "Competitive (Watershed)":
                # === Mode 1: Watershed ===
                mask_neg = self._get_flood_mask(neg_seeds, self.tol)
                if mask_neg is not None:
                    union_mask = mask_pos | mask_neg
                    markers = np.zeros_like(union_mask, dtype=np.int32)
                    for s in pos_seeds:
                        idx = tuple(int(round(c)) for c in s["index"])
                        if 0<=idx[0]<markers.shape[0] and 0<=idx[1]<markers.shape[1] and 0<=idx[2]<markers.shape[2]:
                            markers[idx] = 1 
                    for s in neg_seeds:
                        idx = tuple(int(round(c)) for c in s["index"])
                        if 0<=idx[0]<markers.shape[0] and 0<=idx[1]<markers.shape[1] and 0<=idx[2]<markers.shape[2]:
                            markers[idx] = 2 
                    
                    try:
                        labels_ws = watershed(self.grad_image, markers, mask=union_mask)
                        region_neg = (labels_ws == 2)
                        
                        if self.push_force != 0:
                            kernel = self._get_ellipsoid_footprint(abs(self.push_force))
                            if self.push_force > 0:
                                region_neg = binary_dilation(region_neg, structure=kernel)
                            else:
                                region_neg = binary_erosion(region_neg, structure=kernel)
                        
                        final_neg_mask = region_neg & union_mask
                        final_pos_mask = union_mask & (~final_neg_mask)
                    except Exception: final_pos_mask = mask_pos
            
            else:
                # === Mode 2: Local Cut ===
                mask_neg_seeds = np.zeros_like(mask_pos, dtype=bool)
                for s in neg_seeds:
                    idx = tuple(int(round(c)) for c in s["index"])
                    if 0<=idx[0]<mask_neg_seeds.shape[0] and 0<=idx[1]<mask_neg_seeds.shape[1] and 0<=idx[2]<mask_neg_seeds.shape[2]:
                        mask_neg_seeds[idx] = True
                
                radius = max(1, self.push_force)
                current_neg = mask_neg_seeds
                
                for _ in range(radius):
                    dilated = binary_dilation(current_neg)
                    current_neg = dilated & mask_pos
                
                final_neg_mask = current_neg
                final_pos_mask = mask_pos & (~final_neg_mask)

        # 3. Protect
        if self.protect_others:
            target_indices = np.where(final_pos_mask)
            existing_labels = labels[target_indices]
            valid_mask = (existing_labels == 0) | (existing_labels == label_val)
            z, y, x = target_indices
            labels[z[valid_mask], y[valid_mask], x[valid_mask]] = label_val
        else:
            labels[final_pos_mask] = label_val

        if np.any(final_neg_mask) and self.cb_show_neg.value:
            self.preview_layer.data[final_neg_mask] = 1
        
        self.labels_layer.refresh()
        self.preview_layer.refresh()

    def _on_points_change_factory(self, seg_id):
        def _on_points_change(event):
            if self.is_frozen: return

            seg = self.segmentations[seg_id]
            pts_layer = seg["points_layer"]
            current_pts = np.asarray(pts_layer.data)
            current_count = len(current_pts)
            last_count = seg["last_count"]
            seeds = seg["seeds"]
            
            if "redo_stack" not in seg: seg["redo_stack"] = []

            # Case A: Delete
            if current_count < last_count:
                if len(seeds) > 0:
                    old_coords = np.array([s["index"] for s in seeds])
                    
                    if current_count == 0:
                        seg["redo_stack"].clear() 
                        seg["seeds"].clear()
                    else:
                        try:
                            dists = cdist(old_coords, current_pts)
                            min_dists = dists.min(axis=1)
                            deleted_idx = np.argmax(min_dists)
                            
                            deleted_seed = seg["seeds"][deleted_idx]
                            seg["redo_stack"].append(deleted_seed)
                            print(f"[Undo] Seed stored in Redo stack.")
                            
                            del seg["seeds"][deleted_idx]
                        except Exception as e:
                            print(f"[Sync Warning] Smart delete failed: {e}")
                            del seg["seeds"][current_count:]
                
            # Case B: Add
            elif current_count > last_count: 
                if not self.is_redoing:
                    seg["redo_stack"].clear()
                
                for i in range(last_count, current_count):
                    coord = current_pts[i]
                    seeds.append({"index": coord.copy(), "tol": self.tol, "type": self.current_mode})
            
            # Case C: Move/Edit
            elif current_count == last_count and current_count > 0:
                for i in range(current_count):
                    if i < len(seeds) and not np.allclose(seeds[i]["index"], current_pts[i]):
                        seeds[i]["index"] = current_pts[i].copy()
                        if not self.is_redoing: seg["redo_stack"].clear()
            
            seg["last_count"] = current_count
            self._recompute_seg_from_seeds(seg_id)
            self._update_status()
        return _on_points_change

    def _setup_keybindings(self):
        self.viewer.bind_key("n", lambda v: self.create_new_segmentation(), overwrite=True)
        self.viewer.bind_key("[", lambda v: self.switch_segmentation(-1), overwrite=True)
        self.viewer.bind_key("]", lambda v: self.switch_segmentation(+1), overwrite=True)
        self.viewer.bind_key("z", lambda v: self.undo_last_seed(), overwrite=True)

    def create_new_segmentation(self):
        seg_id = self.next_seg_id
        self.next_seg_id += 1
        name = f"Seg {seg_id}"
        
        layer_name = f"seeds_{name}"
        if layer_name in self.viewer.layers:
            self.viewer.layers.remove(self.viewer.layers[layer_name])

        base_color = self.SEED_COLORS[(seg_id - 1) % len(self.SEED_COLORS)]

        pts = self.viewer.add_points(
            ndim=3, name=layer_name, size=4, 
            face_color=base_color, n_dimensional=True,
            scale=self.image_layer.scale,
            translate=self.image_layer.translate,
            rotate=self.image_layer.rotate,
            shear=self.image_layer.shear,
            affine=self.image_layer.affine
        )
        self.segmentations[seg_id] = {
            "id": seg_id, 
            "name": name, 
            "label_value": seg_id, 
            "points_layer": pts, 
            "seeds": [], 
            "last_count": 0,
            "redo_stack": []
        }
        
        pts.events.data.connect(self._on_points_change_factory(seg_id))
        self.current_seg_id = seg_id
        self.viewer.layers.selection.active = pts
        
        # [UX] Force 'add' mode
        pts.mode = 'add'
        
        self.current_mode = "add"
        if hasattr(self, 'mode_radio'):
            self.mode_radio.value = "Add (+)"
            
        self._update_current_points_color()
        self._update_status()
        print(f"[System] Created {name}")

    def switch_segmentation(self, direction):
        if not self.segmentations: return
        ids = sorted(self.segmentations.keys())
        if self.current_seg_id not in ids: self.current_seg_id = ids[0]
        else: self.current_seg_id = ids[(ids.index(self.current_seg_id) + direction) % len(ids)]
        
        if self.current_seg_id in self.segmentations:
            old_seg = self.segmentations[self.current_seg_id]
            old_seg["saved_selection"] = old_seg["points_layer"].selected_data

        seg = self.segmentations[self.current_seg_id]
        self.viewer.layers.selection.active = seg["points_layer"]
        
        # Restore selection
        self._restore_selection(self.current_seg_id)
        
        # [UX] Force 'add' mode
        seg["points_layer"].mode = 'add'
        
        self.current_mode = "add"
        if hasattr(self, 'mode_radio'):
            self.mode_radio.value = "Add (+)"
            
        self._update_current_points_color()
        self._update_status()
        print(f"[System] Switched to {seg['name']}")

    def undo_last_seed(self):
        if self.is_frozen: return
        if self.current_seg_id and self.current_seg_id in self.segmentations:
            seg = self.segmentations[self.current_seg_id]
            pts = seg["points_layer"]
            if len(pts.data): 
                pts.selected_data = set()
                with pts.events.data.blocker():
                    pts.data = pts.data[:-1]
                
                if len(seg["seeds"]) > len(pts.data):
                    deleted = seg["seeds"].pop()
                    if "redo_stack" not in seg: seg["redo_stack"] = []
                    seg["redo_stack"].append(deleted)
                    print(f"[Undo] Last seed removed. Redo stack size: {len(seg['redo_stack'])}")

                seg["last_count"] = len(pts.data)
                
                self._recompute_seg_from_seeds(self.current_seg_id)
                self._update_status()

    def redo_last_action(self):
        if self.is_frozen: return
        if self.current_seg_id and self.current_seg_id in self.segmentations:
            seg = self.segmentations[self.current_seg_id]
            if "redo_stack" in seg and seg["redo_stack"]:
                seed_info = seg["redo_stack"].pop()
                print(f"[Redo] Restoring seed with Tol={seed_info['tol']}")
                
                self.is_redoing = True
                
                pts = seg["points_layer"]
                new_pt = np.array([seed_info["index"]])
                # Auto select the redone seed
                pts.selected_data = set() 
                pts.data = np.append(pts.data, new_pt, axis=0)
                
                if seg["seeds"]:
                    seg["seeds"][-1] = seed_info
                    self._recompute_seg_from_seeds(self.current_seg_id)
                
                # Highlight the restored point
                if len(pts.data) > 0:
                    pts.selected_data = {len(pts.data) - 1}

                self.is_redoing = False
                self._update_status()
            else:
                print("[Redo] Nothing to redo.")

    def _on_apply_seed_tol(self, event=None):
        if self.is_frozen: return
        if self.current_seg_id and self.current_seg_id in self.segmentations:
            pts = self.segmentations[self.current_seg_id]["points_layer"]
            sel = list(pts.selected_data)
            
            # [UX] Smart Fallback
            if not sel and len(pts.data) > 0:
                print("[Update] No visual selection, assuming last point.")
                idx = len(pts.data) - 1
                pts.selected_data = {idx}
                sel = [idx]
            
            if sel:
                idx = sorted(sel)[0]
                seeds = self.segmentations[self.current_seg_id]["seeds"]
                if idx < len(seeds):
                    new_tol = float(self.seed_tol_slider.value)
                    seeds[idx]["tol"] = new_tol
                    if idx < len(pts.data):
                        seeds[idx]["index"] = pts.data[idx].copy()
                    
                    print(f"[Update] Seed {idx} updated. New Tol: {new_tol}")
                    self._recompute_seg_from_seeds(self.current_seg_id)
            else:
                print("[Update] No seeds exist to update.")
    
    def _on_btn_new(self, event=None):
        self.create_new_segmentation()

    def _update_current_points_color(self):
        if self.current_seg_id and self.current_seg_id in self.segmentations:
            seg = self.segmentations[self.current_seg_id]
            try:
                saved_sel = seg["points_layer"].selected_data
                
                seg["points_layer"].selected_data = set()
                color = "red" if self.current_mode == "remove" else self.SEED_COLORS[(seg["id"]-1)%len(self.SEED_COLORS)]
                seg["points_layer"].current_face_color = color
                
                seg["points_layer"].selected_data = saved_sel
            except IndexError: pass

    # ---------- Helper functions ----------
    
    def _set_mode(self, val):
        self.current_mode = "add" if "Add" in val else "remove"
        self._update_current_points_color()

    def _update_status(self):
        if not hasattr(self, 'status_label'): return
        
        text = "Current: None"
        if self.current_seg_id in self.segmentations:
            seg = self.segmentations[self.current_seg_id]
            stack_size = len(seg.get("redo_stack", []))
            text = f"Current: {seg['name']} (Seeds={len(seg['seeds'])} | Redo={stack_size})"
        
        self.status_label.value = f"<div style='text-align: center;'>{text}</div>"

    def reset_current_seg(self):
        if self.current_seg_id is None or self.current_seg_id not in self.segmentations: return
        seg = self.segmentations[self.current_seg_id]
        seg["points_layer"].selected_data = set()
        with seg["points_layer"].events.data.blocker():
            seg["points_layer"].data = np.empty((0, 3))
        seg["seeds"].clear()
        seg["redo_stack"] = [] 
        seg["last_count"] = 0
        label_val = seg["label_value"]
        self.labels_layer.data[self.labels_layer.data == label_val] = 0
        self.labels_layer.refresh()
        self.preview_layer.data[:] = 0
        self.preview_layer.refresh()
        print(f"[Manager] Reset {seg['name']}.")
        self._update_status()

    def delete_current_seg(self):
        if self.current_seg_id is None or self.current_seg_id not in self.segmentations: return
        if len(self.segmentations) <= 1:
            self.reset_current_seg()
            return
        seg_id_to_del = self.current_seg_id
        seg = self.segmentations[seg_id_to_del]
        label_val = seg["label_value"]
        self.labels_layer.data[self.labels_layer.data == label_val] = 0
        self.labels_layer.refresh()
        if seg["points_layer"] in self.viewer.layers:
            self.viewer.layers.remove(seg["points_layer"])
        del self.segmentations[seg_id_to_del]
        remaining_ids = sorted(self.segmentations.keys())
        if remaining_ids:
            self.current_seg_id = remaining_ids[0]
            self.switch_segmentation(0)
        else: self.create_new_segmentation()
        print(f"[Manager] Deleted Seg {seg_id_to_del}.")

    def reset_all(self):
        print("[Manager] Resetting ALL...")
        for seg_id, seg in list(self.segmentations.items()):
            if seg["points_layer"] in self.viewer.layers:
                self.viewer.layers.remove(seg["points_layer"])
        self.labels_layer.data[:] = 0
        self.labels_layer.refresh()
        self.preview_layer.data[:] = 0
        self.preview_layer.refresh()
        self.segmentations.clear()
        self.next_seg_id = 1
        self.current_seg_id = None
        self.create_new_segmentation()
        print("[Manager] All reset done.")

    def save_project(self):
        if not self.segmentations: return
        filename, _ = QFileDialog.getSaveFileName(None, "Save Project", "", "JSON Files (*.json)")
        if not filename: return
        if not filename.endswith(".json"): filename += ".json"
        data_to_save = {
            "next_seg_id": self.next_seg_id,
            "global_tol": self.tol,
            "edge_thresh": self.edge_thresh,
            "enable_edge": self.enable_edge_limit,
            "segments": []
        }
        for seg_id, seg in self.segmentations.items():
            serializable_seeds = []
            for s in seg["seeds"]:
                serializable_seeds.append({
                    "index": [float(x) for x in s["index"]],
                    "tol": float(s["tol"]),
                    "type": s.get("type", "add")
                })
            data_to_save["segments"].append({
                "id": seg_id,
                "name": seg["name"],
                "label_value": int(seg["label_value"]),
                "seeds": serializable_seeds
            })
        try:
            with open(filename, 'w') as f: json.dump(data_to_save, f, indent=2)
            print(f"[Save] Project saved to {filename}")
        except Exception as e: print(f"[Save Error] {e}")

    def load_project(self):
        filename, _ = QFileDialog.getOpenFileName(None, "Load Project", "", "JSON Files (*.json);;All Files (*)")
        if not filename: return
        try:
            with open(filename, 'r') as f: data = json.load(f)
            self.reset_all()
            if 1 in self.segmentations:
                if self.segmentations[1]["points_layer"] in self.viewer.layers:
                    self.viewer.layers.remove(self.segmentations[1]["points_layer"])
                del self.segmentations[1]
            self.current_seg_id = None
            self.next_seg_id = data.get("next_seg_id", 1)
            self.tol = data.get("global_tol", 0.1)
            self.edge_thresh = data.get("edge_thresh", 0.1)
            self.enable_edge_limit = data.get("enable_edge", False)
            self.tol_slider.value = self.tol
            self.edge_slider.value = self.edge_thresh
            self.cb_edge.value = self.enable_edge_limit
            for seg_data in data["segments"]:
                seg_id = seg_data["id"]
                name = seg_data["name"]
                label_val = seg_data["label_value"]
                layer_name = f"seeds_{name}"
                if layer_name in self.viewer.layers: self.viewer.layers.remove(self.viewer.layers[layer_name])
                
                base_color = self.SEED_COLORS[(seg_id - 1) % len(self.SEED_COLORS)]
                
                pts_layer = self.viewer.add_points(
                    ndim=3, name=layer_name, size=4, 
                    face_color=base_color, n_dimensional=True,
                    scale=self.image_layer.scale, translate=self.image_layer.translate,
                    rotate=self.image_layer.rotate, shear=self.image_layer.shear, affine=self.image_layer.affine
                )
                
                loaded_seeds = []
                pts_data_list = []
                for s in seg_data["seeds"]:
                    idx = np.array(s["index"])
                    loaded_seeds.append({"index": idx, "tol": s["tol"], "type": s["type"]})
                    pts_data_list.append(idx)
                if pts_data_list:
                    with pts_layer.events.data.blocker(): pts_layer.data = np.array(pts_data_list)
                self.segmentations[seg_id] = {
                    "id": seg_id, "name": name, "label_value": label_val,
                    "points_layer": pts_layer, "seeds": loaded_seeds, "last_count": len(loaded_seeds),
                    "redo_stack": []
                }
                pts_layer.events.data.connect(self._on_points_change_factory(seg_id))
                self.current_seg_id = seg_id
                self._recompute_seg_from_seeds(seg_id)
            if self.segmentations:
                first_id = sorted(self.segmentations.keys())[0]
                self.current_seg_id = first_id
                self.switch_segmentation(0)
            else: self.create_new_segmentation()
            print("[Load] Project loaded successfully.")
        except Exception as e:
            print(f"[Load Error] {e}")
            import traceback
            traceback.print_exc()

    def export_nifti(self):
        if self.labels_layer is None: return
        
        filename, _ = QFileDialog.getSaveFileName(None, "Export NIfTI", "", "NIfTI Files (*.nii.gz)")
        if not filename: return
        if not filename.endswith(".nii.gz"): filename += ".nii.gz"
        
        try:
            print(f"[Export] Saving {filename}...")
            arr = self.labels_layer.data.astype(np.int16)
            img = sitk.GetImageFromArray(arr)
            
            if hasattr(self.image_layer, 'scale'):
                sp = list(self.image_layer.scale)
                if len(sp) == 3: img.SetSpacing((sp[2], sp[1], sp[0]))
            
            if hasattr(self.image_layer, 'translate'):
                og = list(self.image_layer.translate)
                if len(og) == 3: img.SetOrigin((og[2], og[1], og[0]))
            
            sitk.WriteImage(img, filename)
            print(f"[Export] Successfully saved to {filename}")
            self.viewer.status = f"[System] Saved {filename}"
            
        except Exception as e:
            print(f"[Export Error] {e}")
            import traceback
            traceback.print_exc()

# ==================== 启动逻辑 ====================

@magic_factory(call_button="Start Annotation")
def start_annotator(
    viewer: napari.Viewer,
    image_layer: Image
):
    if image_layer is None:
        print("[Error] Please select an image layer first!")
        return
    NeuronAnnotator(viewer, image_layer)
