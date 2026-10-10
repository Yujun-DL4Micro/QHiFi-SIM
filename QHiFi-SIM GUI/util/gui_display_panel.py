import os
import tifffile as tif
from PyQt6.QtWidgets import *
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)


class DisplayPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.init_ui()
        self.show_placeholder()

    def show_placeholder(self, side='both'):
        if side in ['left', 'both']:
            self.sim_canvas.fig.clear()
            ax = self.sim_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No SIM image loaded\n \nPlease select a file",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='#888888')
            ax.set_facecolor('#f5f5f5')
            ax.axis('off')
            self.sim_canvas.draw()

        if side in ['right', 'both']:
            self.qhifi_canvas.fig.clear()
            ax = self.qhifi_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No QHIFI image loaded\n \nPlease select a file",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='#888888')
            ax.set_facecolor('#f5f5f5')
            ax.axis('off')
            self.qhifi_canvas.draw()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        left_panel = self.create_left_panel()
        layout.addWidget(left_panel, 1)
        right_panel = self.create_right_panel()
        layout.addWidget(right_panel, 1)
        self.setLayout(layout)

    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        sim_label = QLabel("SIM Results")
        sim_label.setFixedHeight(50)
        sim_label.setStyleSheet("""
            QLabel {
                font-weight: bold; 
                font-size: 14pt; 
                qproperty-alignment: 'AlignCenter';
                border: none;
                padding-top: 10px;  
                padding-bottom: 1px; 
            }
        """)
        layout.addWidget(sim_label)
        sim_controls_layout = QHBoxLayout()
        sim_file_label = QLabel("SIM File:")
        self.sim_file_combo = QComboBox()
        self.sim_file_combo.setMinimumWidth(200)
        self.sim_file_combo.currentTextChanged.connect(self.on_sim_file_selected)

        sim_controls_layout.addWidget(sim_file_label)
        sim_controls_layout.addWidget(self.sim_file_combo)
        sim_controls_layout.addStretch()
        layout.addLayout(sim_controls_layout)
        self.sim_canvas = MplCanvas(self, width=6, height=6, dpi=100)
        self.sim_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.sim_canvas)

        panel.setLayout(layout)
        return panel

    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        qhifi_label = QLabel("QHiFi Results")
        qhifi_label.setFixedHeight(50)
        qhifi_label.setStyleSheet("""
            QLabel {
                font-weight: bold; 
                font-size: 14pt; 
                
                qproperty-alignment: 'AlignCenter';
                border: none;
                padding-top: 10px; 
                padding-bottom: 1px; 
            }
        """)
        layout.addWidget(qhifi_label)
        qhifi_controls_layout = QHBoxLayout()
        qhifi_file_label = QLabel("QHiFi File:")
        self.qhifi_file_combo = QComboBox()
        self.qhifi_file_combo.setMinimumWidth(200)
        self.qhifi_file_combo.currentTextChanged.connect(self.on_qhifi_file_selected)

        qhifi_controls_layout.addWidget(qhifi_file_label)
        qhifi_controls_layout.addWidget(self.qhifi_file_combo)
        qhifi_controls_layout.addStretch()
        layout.addLayout(qhifi_controls_layout)
        self.qhifi_canvas = MplCanvas(self, width=6, height=6, dpi=100)
        self.qhifi_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.qhifi_canvas)
        panel.setLayout(layout)
        return panel

    def on_sim_file_selected(self, text):
        if not text:
            return
        index = self.sim_file_combo.currentIndex()
        if index >= 0:
            item_data = self.sim_file_combo.itemData(index)
            if item_data:
                self.update_sim_preview_from_combo(item_data)
            else:
                self.update_sim_preview_from_combo(text)

    def on_qhifi_file_selected(self, text):
        if not text:
            return

        index = self.qhifi_file_combo.currentIndex()
        if index >= 0:
            item_data = self.qhifi_file_combo.itemData(index)
            if item_data:
                self.update_qhifi_preview_from_combo(item_data)
            else:
                self.update_qhifi_preview_from_combo(text)

    def update_sim_preview_from_combo(self, text):
        if not text:
            self.show_placeholder('left')
            return

        if not hasattr(self.parent, 'control_panel'):
            self.show_placeholder('left')
            return

        output_path = self.parent.control_panel.output_path_edit.text()
        if not output_path:
            self.show_placeholder('left')
            return

        full_path = os.path.join(output_path, text)
        if os.path.exists(full_path):
            self.update_sim_preview(full_path, text)
            return
        if ('HiFi_' in text and 'qHiFi' not in text) or ('HIFI_' in text and 'qHiFi' not in text):
            timestamp_dirs = self.find_timestamp_dirs(output_path)
            for timestamp_dir in timestamp_dirs:
                hifi_dir = os.path.join(timestamp_dir, 'HiFi')
                if os.path.exists(hifi_dir):
                    filename = os.path.basename(text)
                    potential_path = os.path.join(hifi_dir, filename)
                    if os.path.exists(potential_path):
                        rel_path = os.path.relpath(potential_path, output_path)
                        self.update_sim_preview(potential_path, rel_path)
                        return
                hifi_sim_dir = os.path.join(timestamp_dir, 'HiFi-SIM')
                if os.path.exists(hifi_sim_dir):
                    filename = os.path.basename(text)
                    potential_path = os.path.join(hifi_sim_dir, filename)
                    if os.path.exists(potential_path):
                        rel_path = os.path.relpath(potential_path, output_path)
                        self.update_sim_preview(potential_path, rel_path)
                        return
                    if filename.startswith('HIFI_'):
                        lowercase_filename = filename.lower()
                        potential_path = os.path.join(hifi_sim_dir, lowercase_filename)
                        if os.path.exists(potential_path):
                            rel_path = os.path.relpath(potential_path, output_path)
                            self.update_sim_preview(potential_path, rel_path)
                            return

        filename = os.path.basename(text)
        for root, dirs, files in os.walk(output_path):
            dirs[:] = [d for d in dirs if 'WF' not in d.upper() and 'wf' not in d.lower()]
            for file in files:
                if file == filename or (file.lower() == filename.lower()):
                    if ('_wf' in file.lower() or 'wf_' in file.lower() or
                            'wf.' in file.lower() or '.wf.' in file.lower()):
                        continue
                    found_path = os.path.join(root, file)
                    rel_path = os.path.relpath(found_path, output_path)
                    self.update_sim_preview(found_path, rel_path)
                    return
        self.show_placeholder('left')

    def update_qhifi_preview_from_combo(self, text):
        if not text or not hasattr(self.parent, 'control_panel'):
            self.show_placeholder('right')
            return

        output_path = self.parent.control_panel.output_path_edit.text()
        if not output_path:
            self.show_placeholder('right')
            return

        full_path = os.path.join(output_path, text)
        if os.path.exists(full_path):
            self.update_qhifi_preview(full_path, text)
            return

        if 'qHiFi_' in text:
            timestamp_dirs = self.find_timestamp_dirs(output_path)
            for timestamp_dir in timestamp_dirs:
                qhifi_dir = os.path.join(timestamp_dir, 'QHiFi')
                if os.path.exists(qhifi_dir):
                    filename = os.path.basename(text)
                    potential_path = os.path.join(qhifi_dir, filename)
                    if os.path.exists(potential_path):
                        rel_path = os.path.relpath(potential_path, output_path)
                        self.update_qhifi_preview(potential_path, rel_path)
                        return

        filename = os.path.basename(text)
        for root, dirs, files in os.walk(output_path):
            dirs[:] = [d for d in dirs if 'WF' not in d.upper() and 'wf' not in d.lower()]

            for file in files:
                if file == filename or (file.lower() == filename.lower()):
                    if ('_wf' in file.lower() or 'wf_' in file.lower() or
                            'wf.' in file.lower() or '.wf.' in file.lower()):
                        continue

                    found_path = os.path.join(root, file)
                    rel_path = os.path.relpath(found_path, output_path)
                    self.update_qhifi_preview(found_path, rel_path)
                    return
        self.show_placeholder('right')

    def find_timestamp_dirs(self, output_path):
        timestamp_dirs = []
        if os.path.exists(output_path):
            for item in os.listdir(output_path):
                item_path = os.path.join(output_path, item)
                if os.path.isdir(item_path) and item.startswith('Results_'):
                    timestamp_dirs.append(item_path)

        timestamp_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return timestamp_dirs

    def update_sim_preview(self, image_path, display_name):
        try:
            image = tif.imread(image_path)
            self.sim_canvas.fig.clear()
            self.sim_canvas.fig.patch.set_facecolor('#f5f9fa')
            self.sim_canvas.fig.subplots_adjust(left=0.05, right=0.95,
                                                bottom=0.05, top=0.95)
            gs = self.sim_canvas.fig.add_gridspec(
                nrows=2, ncols=1,
                height_ratios=[12, 1],
                hspace=0.05
            )
            ax_img = self.sim_canvas.fig.add_subplot(gs[0])
            ax_img.axis('off')
            ax_img.set_facecolor('#ffffff')
            ax_cbar = self.sim_canvas.fig.add_subplot(gs[1])
            ax_cbar.set_facecolor('#f5f9fa')
            cmap = 'hot' if len(image.shape) == 2 else 'viridis'
            im = ax_img.imshow(image, cmap=cmap, aspect='auto')
            height, width = image.shape[:2]
            ax_img.set_aspect('equal')
            if height != width:
                max_side = max(height, width)
                ax_img.set_xlim(-0.5, max_side - 0.5)
                ax_img.set_ylim(-0.5, max_side - 0.5)
                ax_img.set_aspect('equal')
            cbar = self.sim_canvas.fig.colorbar(
                im, cax=ax_cbar,
                orientation='horizontal'
            )
            vmin, vmax = im.get_clim()
            cbar.set_ticks([])
            cbar.set_ticklabels([])
            ax_cbar.text(0.02, 0.5, f'{vmin:.0f}',
                         fontfamily='Times New Roman',
                         fontsize=11,
                         fontweight='bold',
                         color='white',
                         ha='left',
                         va='center',
                         transform=ax_cbar.transAxes,
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7, edgecolor='none'))

            ax_cbar.text(0.97, 0.5, f'{vmax:.0f}',
                         fontfamily='Times New Roman',
                         fontsize=11,
                         fontweight='bold',
                         color='white',
                         ha='right',
                         va='center',
                         transform=ax_cbar.transAxes,
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7, edgecolor='none'))

            ax_cbar.text(0.5, 0.5, 'Intensity',
                         fontfamily='Times New Roman',
                         fontsize=13,
                         fontweight='bold',
                         color='black',
                         ha='center',
                         va='center',
                         transform=ax_cbar.transAxes)

            cbar.outline.set_linewidth(0.5)
            cbar.outline.set_edgecolor('#999999')
            ax_cbar.spines['top'].set_visible(False)
            ax_cbar.spines['bottom'].set_visible(False)
            ax_cbar.spines['left'].set_visible(False)
            ax_cbar.spines['right'].set_visible(False)
            ax_cbar.set_xticks([])
            ax_cbar.set_yticks([])
            self.sim_canvas.draw()

        except Exception as e:
            self.sim_canvas.fig.clear()
            ax = self.sim_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error loading SIM image:\n{str(e)}",
                    ha='center', va='center', transform=ax.transAxes,
                    color='red', fontsize=10, wrap=True)
            ax.set_title('SIM Loading Error', color='red', pad=10)
            ax.axis('off')
            self.sim_canvas.draw()

    def update_qhifi_preview(self, image_path, display_name):
        try:
            image = tif.imread(image_path)
            self.qhifi_canvas.fig.clear()
            self.qhifi_canvas.fig.patch.set_facecolor('#f5f9fa')
            self.qhifi_canvas.fig.subplots_adjust(left=0.05, right=0.95,
                                                  bottom=0.05, top=0.95)
            gs = self.qhifi_canvas.fig.add_gridspec(
                nrows=2, ncols=1,
                height_ratios=[12, 1],
                hspace=0.05
            )

            ax_img = self.qhifi_canvas.fig.add_subplot(gs[0])
            ax_img.axis('off')
            ax_img.set_facecolor('#ffffff')
            ax_cbar = self.qhifi_canvas.fig.add_subplot(gs[1])
            ax_cbar.set_facecolor('#f5f9fa')
            cmap = 'hot' if len(image.shape) == 2 else 'plasma'
            im = ax_img.imshow(image, cmap=cmap, aspect='auto')

            height, width = image.shape[:2]
            ax_img.set_aspect('equal')
            if height != width:
                max_side = max(height, width)
                ax_img.set_xlim(-0.5, max_side - 0.5)
                ax_img.set_ylim(-0.5, max_side - 0.5)
                ax_img.set_aspect('equal')

            cbar = self.qhifi_canvas.fig.colorbar(
                im, cax=ax_cbar,
                orientation='horizontal'
            )

            vmin, vmax = im.get_clim()
            cbar.set_ticks([])
            cbar.set_ticklabels([])
            ax_cbar.text(0.02, 0.5, f'{vmin:.0f}',
                         fontfamily='Times New Roman',
                         fontsize=11,
                         fontweight='bold',
                         color='white',
                         ha='left',
                         va='center',
                         transform=ax_cbar.transAxes,
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7, edgecolor='none'))

            ax_cbar.text(0.97, 0.5, f'{vmax:.0f}',
                         fontfamily='Times New Roman',
                         fontsize=11,
                         fontweight='bold',
                         color='white',
                         ha='right',
                         va='center',
                         transform=ax_cbar.transAxes,
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7, edgecolor='none'))

            ax_cbar.text(0.5, 0.5, 'Intensity',
                         fontfamily='Times New Roman',
                         fontsize=13,
                         fontweight='bold',
                         color='black',
                         ha='center',
                         va='center',
                         transform=ax_cbar.transAxes)

            cbar.outline.set_linewidth(0.5)
            cbar.outline.set_edgecolor('#999999')
            ax_cbar.spines['top'].set_visible(False)
            ax_cbar.spines['bottom'].set_visible(False)
            ax_cbar.spines['left'].set_visible(False)
            ax_cbar.spines['right'].set_visible(False)
            ax_cbar.set_xticks([])
            ax_cbar.set_yticks([])
            self.qhifi_canvas.draw()

        except Exception as e:
            self.qhifi_canvas.fig.clear()
            ax = self.qhifi_canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error loading QHIFI image:\n{str(e)}",
                    ha='center', va='center', transform=ax.transAxes,
                    color='red', fontsize=10, wrap=True)
            ax.set_title('QHIFI Loading Error', color='red', pad=10)
            ax.axis('off')
            self.qhifi_canvas.draw()

    def update_file_combos(self, output_files):
        current_sim = self.sim_file_combo.currentText() if self.sim_file_combo.count() > 0 else ""
        current_qhifi = self.qhifi_file_combo.currentText() if self.qhifi_file_combo.count() > 0 else ""
        self.sim_file_combo.clear()
        self.qhifi_file_combo.clear()
        filtered_files = []
        for f in output_files:
            f_lower = f.lower()
            if not ('_wf' in f_lower or 'wf_' in f_lower or
                    'wf.' in f_lower or '.wf.' in f_lower or
                    '/wf/' in f_lower or '\\wf\\' in f_lower):
                filtered_files.append(f)
        output_files = filtered_files

        if output_files:
            hifi_files = [f for f in output_files if ('HiFi_' in f or 'HIFI_' in f or 'hifi_' in f)
                          and 'qHiFi' not in f and 'QHIFI' not in f and 'qhifi' not in f]
            qhifi_files = [f for f in output_files if 'qHiFi_' in f or 'QHIFI_' in f or 'qhifi_' in f]
            other_files = [f for f in output_files if f not in hifi_files and f not in qhifi_files]
            for f in hifi_files:
                short_name = os.path.basename(f)
                self.sim_file_combo.addItem(short_name, f)

            for f in qhifi_files:
                short_name = os.path.basename(f)
                self.qhifi_file_combo.addItem(short_name, f)

            for f in other_files:
                short_name = os.path.basename(f)
                self.sim_file_combo.addItem(short_name, f)

            new_sim_selection = None
            new_qhifi_selection = None

            if hifi_files:
                new_sim_selection = hifi_files[-1]
            elif current_sim and current_sim in [self.sim_file_combo.itemText(i) for i in
                                                 range(self.sim_file_combo.count())]:
                index = self.sim_file_combo.findText(current_sim)
                if index >= 0:
                    new_sim_selection = self.sim_file_combo.itemData(index)
            elif output_files:
                new_sim_selection = output_files[0]

            if qhifi_files:
                new_qhifi_selection = qhifi_files[-1]
            elif current_qhifi and current_qhifi in [self.qhifi_file_combo.itemText(i) for i in
                                                     range(self.qhifi_file_combo.count())]:
                index = self.qhifi_file_combo.findText(current_qhifi)
                if index >= 0:
                    new_qhifi_selection = self.qhifi_file_combo.itemData(index)

            if new_sim_selection:
                for i in range(self.sim_file_combo.count()):
                    if self.sim_file_combo.itemData(i) == new_sim_selection:
                        self.sim_file_combo.blockSignals(True)
                        self.sim_file_combo.setCurrentIndex(i)
                        self.sim_file_combo.blockSignals(False)
                        self.update_sim_preview_from_combo(new_sim_selection)
                        break
                else:
                    self.show_placeholder('left')
            else:
                self.show_placeholder('left')

            if new_qhifi_selection:
                for i in range(self.qhifi_file_combo.count()):
                    if self.qhifi_file_combo.itemData(i) == new_qhifi_selection:
                        self.qhifi_file_combo.blockSignals(True)
                        self.qhifi_file_combo.setCurrentIndex(i)
                        self.qhifi_file_combo.blockSignals(False)
                        self.update_qhifi_preview_from_combo(new_qhifi_selection)
                        break
                else:
                    self.show_placeholder('right')
            else:
                self.show_placeholder('right')
        else:
            self.show_placeholder('both')
