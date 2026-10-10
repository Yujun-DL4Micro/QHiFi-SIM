import os
from PyQt6.QtWidgets import *
from util.gui_control_panel import ControlPanel
from util.gui_display_panel import DisplayPanel
from util.hifi_reconstruction import HifiReconstructionWorker
from util.nn_reconstruction import NNReconstructionWorker
from PyQt6.QtCore import QTimer
from util.quant_worker import QuantAnalysisWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.raw_files = []
        self.output_files = []
        self.current_display_mode = 'gray'
        self.hifi_worker = None
        self.nn_worker = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("QHiFi-SIM Reconstruction")
        self.setGeometry(80, 80, 1500, 700)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        self.control_panel = ControlPanel(self)
        main_layout.addWidget(self.control_panel, 1)
        self.display_panel = DisplayPanel(self)
        main_layout.addWidget(self.display_panel, 2)
        self.connect_signals()

    def connect_signals(self):
        self.control_panel.add_files_btn.clicked.connect(self.add_files)
        self.control_panel.remove_file_btn.clicked.connect(self.remove_selected_file)
        self.control_panel.clear_files_btn.clicked.connect(self.clear_files)
        self.control_panel.output_path_btn.clicked.connect(self.browse_output_path)

        self.control_panel.nn_wf_browse_btn.clicked.connect(self.browse_nn_wf_dir)
        self.control_panel.nn_sim_browse_btn.clicked.connect(self.browse_nn_sim_dir)
        self.control_panel.nn_model_browse_btn.clicked.connect(self.browse_nn_model_path)
        self.control_panel.nn_output_browse_btn.clicked.connect(self.browse_nn_output_path)

        self.control_panel.start_btn.clicked.connect(self.start_unified_reconstruction)
        self.control_panel.nn_mode_combo.currentTextChanged.connect(self.on_nn_mode_changed)
        self.control_panel.file_list_widget.itemSelectionChanged.connect(
            self.control_panel.update_process_button_state
        )
        self.control_panel.nn_wf_dir_edit.textChanged.connect(
            self.control_panel.update_process_button_state
        )
        self.control_panel.nn_sim_dir_edit.textChanged.connect(
            self.control_panel.update_process_button_state
        )

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Raw Data", "", "TIFF Files (*.tif *.tiff);;All Files (*.*)"
        )
        if files:
            for file_path in files:
                if file_path not in self.raw_files:
                    self.raw_files.append(file_path)
                    self.control_panel.file_list_widget.addItem(os.path.basename(file_path))
            self.control_panel.update_process_button_state()

    def remove_selected_file(self):
        selected_items = self.control_panel.file_list_widget.selectedItems()
        if not selected_items:
            return

        for item in selected_items:
            row = self.control_panel.file_list_widget.row(item)
            self.control_panel.file_list_widget.takeItem(row)
            file_name = item.text()
            for file_path in self.raw_files[:]:
                if os.path.basename(file_path) == file_name:
                    self.raw_files.remove(file_path)
                    break
        self.control_panel.update_process_button_state()

    def clear_files(self):
        self.control_panel.file_list_widget.clear()
        self.raw_files.clear()
        self.control_panel.update_process_button_state()

    def browse_output_path(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select HiFi-SIM Output Directory"
        )
        if dir_path:
            self.control_panel.output_path_edit.setText(dir_path)

    def browse_nn_wf_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select WF Directory")
        if dir_path:
            self.control_panel.nn_wf_dir_edit.setText(dir_path)

    def browse_nn_sim_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select SIM Directory")
        if dir_path:
            self.control_panel.nn_sim_dir_edit.setText(dir_path)

    def browse_nn_model_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "PyTorch Files (*.pth *.pth.tar);;All Files (*.*)"
        )
        if file_path:
            self.control_panel.nn_model_path_edit.setText(file_path)

    def browse_nn_output_path(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self.control_panel.nn_output_path_edit.setText(dir_path)

    def on_nn_mode_changed(self, mode):
        is_external = (mode == "From SIM images")

        self.control_panel.nn_wf_dir_edit.setEnabled(is_external)
        self.control_panel.nn_wf_browse_btn.setEnabled(is_external)
        self.control_panel.nn_sim_dir_edit.setEnabled(is_external)
        self.control_panel.nn_sim_browse_btn.setEnabled(is_external)
        self.control_panel.update_process_button_state()

    def start_hifi_reconstruction(self):
        try:
            import cupy
            gpu_available = True
        except ImportError:
            gpu_available = False
            reply = QMessageBox.question(
                self, "GPU Warning",
                "CuPy not available. Reconstruction will use CPU (slow). Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        if not self.raw_files:
            QMessageBox.warning(self, "Warning", "Please select at least one raw data file!")  # 修改文本
            return

        output_path = self.control_panel.output_path_edit.text()
        if not output_path:
            QMessageBox.warning(self, "Warning", "Please select an output directory!")
            return

        os.makedirs(output_path, exist_ok=True)

        self.control_panel.reset_progress()
        self.control_panel.show_progress_bar()
        self.control_panel.set_progress(0, "Starting HIFI-SIM reconstruction...")
        params = {
            'raw_files': self.raw_files,
            'output_path': output_path,
            'wavelength': self.control_panel.wavelength_spin.value(),
            'na': self.control_panel.na_spin.value(),
            'pixel_size': self.control_panel.pixel_size_spin.value(),
            'mode': 'hifi'
        }

        self.hifi_worker = HifiReconstructionWorker(params)
        self.hifi_worker.file_progress_signal.connect(self.update_hifi_progress)
        self.hifi_worker.finished_signal.connect(self.hifi_reconstruction_finished)
        self.control_panel.set_processing_state(True)
        self.hifi_worker.start()

    def hifi_reconstruction_finished(self, success, message, timestamp_dir=None):
        if not success:
            self.control_panel.set_processing_state(False)
            self.control_panel.hide_progress_bar()
            QMessageBox.critical(self, "Error", message)
            return

        if timestamp_dir:
            self.hifi_output_dir = timestamp_dir
        elif hasattr(self.hifi_worker, 'timestamp_dir') and self.hifi_worker.timestamp_dir:
            self.hifi_output_dir = self.hifi_worker.timestamp_dir
        else:
            self.hifi_output_dir = self.control_panel.output_path_edit.text()

        self.control_panel.set_progress(50, "HIFI-SIM reconstruction completed")
        self.update_output_files_list()
        mode = self.control_panel.nn_mode_combo.currentText()

        if mode == "From Raw Data":
            reply = QMessageBox.question(
                self,
                "Intensity Correction Confirmation",
                "HIFI-SIM reconstruction has completed successfully.\n\n"
                "Do you wish to proceed with intensity correction via QHiFi reconstruction?\n\n"
                "Note: This step will further enhance image quality by applying neural network-based intensity correction.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.control_panel.set_progress(50, "Preparing for QHiFi reconstruction...")
                QTimer.singleShot(1500, self.continue_with_nn_reconstruction)
            else:
                self.control_panel.set_progress(100, "Process completed (without QHiFi)")
                QTimer.singleShot(1000, lambda: self.control_panel.set_processing_state(False))
                QTimer.singleShot(1000, lambda: QMessageBox.information(
                    self,
                    "Process Complete",
                    "HIFI-SIM reconstruction completed successfully.\n\n"
                    "Intensity correction (neural network reconstruction) was skipped as requested.\n"
                    "You can view the HIFI-SIM results in the left panel."
                ))
        else:
            self.control_panel.set_processing_state(False)
            self.control_panel.hide_progress_bar()

    def update_hifi_progress(self, current, total):
        if total > 0:
            progress = int((current / total) * 50)
            self.control_panel.set_progress(
                progress,
                f"HIFI-SIM: Processing file {current} of {total}"
            )

    def continue_with_nn_reconstruction(self):
        self.control_panel.set_progress(55, "Starting QHiFi reconstruction...")

        if not hasattr(self, 'hifi_output_dir') or not self.hifi_output_dir:
            output_path = self.control_panel.output_path_edit.text()
            timestamp_dirs = []
            if os.path.exists(output_path):
                for item in os.listdir(output_path):
                    item_path = os.path.join(output_path, item)
                    if os.path.isdir(item_path) and item.startswith('Results_'):
                        timestamp_dirs.append(item_path)

            if timestamp_dirs:
                timestamp_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                self.hifi_output_dir = timestamp_dirs[0]
            else:
                self.hifi_output_dir = output_path

        wf_dir = os.path.join(self.hifi_output_dir, 'WF')
        sim_dir = os.path.join(self.hifi_output_dir, 'HiFi-SIM')
        if not (os.path.exists(wf_dir) and os.path.exists(sim_dir)):
            QMessageBox.warning(self, "Warning",
                                f"HIFI-SIM reconstruction output not found!\nExpected at: {self.hifi_output_dir}")
            self.control_panel.set_processing_state(False)
            self.control_panel.hide_progress_bar()
            return

        wf_files = len([f for f in os.listdir(wf_dir) if f.endswith(('.tif', '.tiff'))])
        sim_files = len([f for f in os.listdir(sim_dir) if f.endswith(('.tif', '.tiff'))])
        total_pairs = min(wf_files, sim_files)

        if total_pairs == 0:
            QMessageBox.warning(self, "Warning", "No HIFI-SIM output files found!")
            self.nn_progress_dialog.close()
            self.control_panel.set_processing_state(False)
            return

        original_pixel_size = self.control_panel.pixel_size_spin.value()
        adjusted_pixel_size = original_pixel_size / 2

        params = {
            'mode': 'hifi_results',
            'wf_dir': wf_dir,
            'sim_dir': sim_dir,
            'hifi_params': {
                'pixel_size': adjusted_pixel_size,
                'na': self.control_panel.na_spin.value(),
                'wavelength': self.control_panel.wavelength_spin.value()
            },
            'output_path': self.control_panel.nn_output_path_edit.text(),
            'model_path': self.control_panel.nn_model_path_edit.text()
        }

        self.nn_worker = NNReconstructionWorker(params)
        self.nn_worker.progress_signal.connect(self.update_nn_progress)
        self.nn_worker.finished_signal.connect(self.nn_reconstruction_finished)
        self.nn_worker.start()

    def start_nn_reconstruction(self):
        self.control_panel.reset_progress()
        self.control_panel.show_progress_bar()
        self.control_panel.set_progress(0, "Starting QHiFi reconstruction...")

        original_pixel_size = self.control_panel.pixel_size_spin.value()
        adjusted_pixel_size = original_pixel_size / 2

        params = {
            'mode': 'external',
            'wf_dir': self.control_panel.nn_wf_dir_edit.text(),
            'sim_dir': self.control_panel.nn_sim_dir_edit.text(),
            'output_path': self.control_panel.nn_output_path_edit.text(),
            'model_path': self.control_panel.nn_model_path_edit.text(),
            'hifi_params': {
                'pixel_size': adjusted_pixel_size,
                'na': self.control_panel.na_spin.value(),
                'wavelength': self.control_panel.wavelength_spin.value()
            }
        }
        os.makedirs(self.control_panel.nn_output_path_edit.text(), exist_ok=True)

        if not (os.path.exists(params['wf_dir']) and os.path.exists(params['sim_dir'])):
            QMessageBox.warning(self, "Warning", "Please select valid WF and SIM directories!")
            self.control_panel.hide_progress_bar()
            return

        if not os.path.exists(params['model_path']):
            QMessageBox.warning(self, "Warning", "Model file does not exist!")
            self.control_panel.hide_progress_bar()
            return

        self.nn_worker = NNReconstructionWorker(params)
        self.nn_worker.progress_signal.connect(self.update_external_nn_progress)
        self.nn_worker.finished_signal.connect(self.nn_reconstruction_finished)
        self.control_panel.set_processing_state(True)
        self.nn_worker.start()

    def update_nn_progress(self, progress_value):
        nn_progress = 55 + int((progress_value / 100) * 45)
        self.control_panel.set_progress(
            nn_progress,
            f"QHiFi reconstruction: {progress_value}%"
        )

    def update_external_nn_progress(self, progress_value):
        self.control_panel.set_progress(
            progress_value,
            f"Processing: {progress_value}%"
        )
    def nn_reconstruction_finished(self, success, message):

        if success:
            self.control_panel.set_progress(100, "QHiFi reconstruction completed successfully")

        if success:
            if hasattr(self.nn_worker, 'output_dir'):
                output_dir = self.nn_worker.output_dir
            else:
                output_dir = self.nn_worker.output_dir or os.path.join(
                    self.control_panel.nn_output_path_edit.text(), 'QHiFi_Results')

            self.output_files = []

            hifi_dir = os.path.join(output_dir, 'HiFi')
            if os.path.exists(hifi_dir):
                for file in os.listdir(hifi_dir):
                    if file.lower().endswith('.tif'):
                        rel_path = os.path.relpath(os.path.join(hifi_dir, file),
                                                   self.control_panel.output_path_edit.text())
                        self.output_files.append(rel_path)

            qhifi_dir = os.path.join(output_dir, 'QHiFi')
            if os.path.exists(qhifi_dir):
                for file in os.listdir(qhifi_dir):
                    if file.lower().endswith('.tif'):
                        rel_path = os.path.relpath(os.path.join(qhifi_dir, file),
                                                   self.control_panel.output_path_edit.text())
                        self.output_files.append(rel_path)

            if hasattr(self.display_panel, 'update_file_combos'):
                self.display_panel.update_file_combos(self.output_files)

            hifi_files = [f for f in self.output_files if 'HiFi_' in f and 'qHiFi' not in f]
            qhifi_files = [f for f in self.output_files if 'qHiFi_' in f]

            if hifi_files:
                latest_hifi = hifi_files[-1]
                self.display_panel.sim_file_combo.blockSignals(True)
                self.display_panel.sim_file_combo.setCurrentText(latest_hifi)
                self.display_panel.sim_file_combo.blockSignals(False)
                self.display_panel.update_sim_preview_from_combo(latest_hifi)

            if qhifi_files:
                latest_qhifi = qhifi_files[-1]
                self.display_panel.qhifi_file_combo.blockSignals(True)
                self.display_panel.qhifi_file_combo.setCurrentText(latest_qhifi)
                self.display_panel.qhifi_file_combo.blockSignals(False)
                self.display_panel.update_qhifi_preview_from_combo(latest_qhifi)

            reply = QMessageBox.question(
                self,
                "Quantitative Analysis",
                "Do you want to run quantitative intensity analysis?\n\n"
                "This will compute PSNR, SSIM, R2 linearity, and ROI-based metrics.\n"
                "You will need to select the WF and SIM directories.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.start_quant_analysis()

            QTimer.singleShot(2000, lambda: self.control_panel.set_processing_state(False))
        else:
            self.control_panel.set_processing_state(False)
            self.control_panel.hide_progress_bar()
            QMessageBox.critical(self, "Error", message)

    def start_quant_analysis(self):
        mode = self.control_panel.nn_mode_combo.currentText()

        if mode == "From Raw Data":
            if not hasattr(self, 'hifi_output_dir') or not self.hifi_output_dir:
                QMessageBox.warning(self, "Warning", "HIFI-SIM output directory not found.")
                return
            wf_dir = os.path.join(self.hifi_output_dir, 'WF')
            sim_dir = os.path.join(self.hifi_output_dir, 'HiFi-SIM')
            if not os.path.exists(wf_dir):
                QMessageBox.warning(self, "Warning", f"WF directory not found: {wf_dir}")
                return
            if not os.path.exists(sim_dir):
                QMessageBox.warning(self, "Warning", f"SIM directory not found: {sim_dir}")
                return
        else:
            wf_dir = self.control_panel.nn_wf_dir_edit.text().strip()
            sim_dir = self.control_panel.nn_sim_dir_edit.text().strip()
            if not wf_dir or not os.path.isdir(wf_dir):
                QMessageBox.warning(self, "Warning", "WF directory is invalid or not set.")
                return
            if not sim_dir or not os.path.isdir(sim_dir):
                QMessageBox.warning(self, "Warning", "SIM directory is invalid or not set.")
                return

        if hasattr(self, 'nn_worker') and self.nn_worker and self.nn_worker.output_dir:
            output_dir = self.nn_worker.output_dir
        else:
            output_dir = self.control_panel.nn_output_path_edit.text().strip()
            if not output_dir:
                output_dir = "./QHiFi_resultDir"

        self.quant_worker = QuantAnalysisWorker(wf_dir, sim_dir, output_dir)
        self.quant_worker.progress_signal.connect(self.update_quant_progress)
        self.quant_worker.finished_signal.connect(self.quant_analysis_finished)
        self.quant_worker.start()

        self.control_panel.set_progress(0, "Starting quantitative analysis...")
        self.control_panel.show_progress_bar()

    def update_quant_progress(self, value):
        self.control_panel.set_progress(value, f"Quantitative analysis: {value}%")

    def quant_analysis_finished(self, success, message):
        self.control_panel.set_processing_state(False)
        self.control_panel.hide_progress_bar()

        if success:
            self.control_panel.set_progress(100, "Quantitative analysis completed")
            QMessageBox.information(
                self,
                "Analysis Complete",
                f"Quantitative analysis completed successfully.\n\n{message}"
            )
        else:
            QMessageBox.critical(
                self,
                "Analysis Failed",
                f"Quantitative analysis failed:\n\n{message}"
            )

    def update_output_files_list(self):
        output_path = self.control_panel.output_path_edit.text()
        self.output_files = []
        scan_paths = []

        if hasattr(self, 'hifi_output_dir') and self.hifi_output_dir:
            scan_paths.append(self.hifi_output_dir)
            for item in os.listdir(self.hifi_output_dir):
                item_path = os.path.join(self.hifi_output_dir, item)
                if os.path.isdir(item_path):
                    if 'WF' in item.upper() or 'wf' in item.lower():
                        continue
                    scan_paths.append(item_path)
        else:
            print("No current HIFI output directory found, searching for latest timestamp directory")
            timestamp_dirs = []
            if os.path.exists(output_path):
                for item in os.listdir(output_path):
                    item_path = os.path.join(output_path, item)
                    if os.path.isdir(item_path) and item.startswith('Results_'):
                        timestamp_dirs.append(item_path)

            timestamp_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)

            if timestamp_dirs:
                latest_dir = timestamp_dirs[0]
                print(f"Using latest timestamp directory: {latest_dir}")
                scan_paths.append(latest_dir)

                for item in os.listdir(latest_dir):
                    item_path = os.path.join(latest_dir, item)
                    if os.path.isdir(item_path):
                        if 'WF' in item.upper() or 'wf' in item.lower():
                            continue
                        scan_paths.append(item_path)
            else:
                print("No timestamp directories found, using root directory")
                scan_paths.append(output_path)

        nn_output_dir = self.control_panel.nn_output_path_edit.text()
        if os.path.exists(nn_output_dir):
            scan_paths.append(nn_output_dir)

        for path in scan_paths:
            if os.path.exists(path):
                for file in os.listdir(path):
                    file_lower = file.lower()
                    if file_lower.endswith(('.tif', '.tiff')):
                        if ('_wf' in file_lower or 'wf_' in file_lower or
                                'wf.' in file_lower or '.wf.' in file_lower):
                            continue
                        try:
                            rel_path = os.path.relpath(os.path.join(path, file), output_path)
                            if '/WF/' in rel_path.upper() or '\\WF\\' in rel_path.upper():
                                continue
                        except ValueError:
                            continue

                        if rel_path and rel_path not in self.output_files:
                            self.output_files.append(rel_path)

        self.output_files.sort()

        if hasattr(self.display_panel, 'update_file_combos'):
            self.display_panel.update_file_combos(self.output_files)

    def start_unified_reconstruction(self):
        mode = self.control_panel.nn_mode_combo.currentText()

        if mode == "From Raw Data":
            self.start_hifi_reconstruction()
        else:
            self.start_nn_reconstruction()