from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import os


class ControlPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("")
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        mode_group = QGroupBox("")
        mode_group.setStyleSheet(
            """
            QGroupBox { 
                background-color: #f0f8ff;
                border: 2px solid #4682b4;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                color: #2c3e50;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: #000000;
                font-weight: bold;
                font-size: 12pt;
            }
            """
        )
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Reconstruction Mode:")
        self.nn_mode_combo = QComboBox()
        self.nn_mode_combo.addItems(["From Raw Data", "From SIM images"])
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.nn_mode_combo)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        hifi_params_group = self.create_hifi_parameters()
        layout.addWidget(hifi_params_group)

        self.file_group = self.create_file_settings()
        layout.addWidget(self.file_group)

        self.nn_params_group = self.create_nn_parameters()
        layout.addWidget(self.nn_params_group)

        start_widget = self.create_start_widget()
        layout.addWidget(start_widget)

        layout.addStretch()
        self.setLayout(layout)

        self.setMaximumWidth(450)

        self.on_nn_mode_changed(self.nn_mode_combo.currentText())
        self.nn_mode_combo.currentTextChanged.connect(self.on_nn_mode_changed)

    def show_progress_bar(self):
        self.progress_container.setFixedHeight(70)
        self.progress_bar.setVisible(True)

    def hide_progress_bar(self):
        self.progress_container.setFixedHeight(0)
        self.progress_bar.setVisible(False)

    def reset_progress(self):
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("     0%")

    def set_progress(self, value, message=""):
        self.progress_bar.setValue(value)
        if message:
            self.progress_bar.setFormat(f"     {value}% - {message}")
        else:
            self.progress_bar.setFormat(f"     {value}%")

    def update_message(self, message):
        current_value = self.progress_bar.value()
        self.progress_bar.setFormat(f"     {current_value}% - {message}")

    def create_file_settings(self):
        group = QGroupBox("")
        layout = QVBoxLayout()

        file_list_label = QLabel("Selected Raw Data:")
        self.file_list_widget = QListWidget()
        self.file_list_widget.setMaximumHeight(150)

        file_buttons_layout = QHBoxLayout()
        self.add_files_btn = QPushButton("Add Files")
        self.remove_file_btn = QPushButton("Remove Selected")
        self.clear_files_btn = QPushButton("Clear All")

        file_buttons_layout.addWidget(self.add_files_btn)
        file_buttons_layout.addWidget(self.remove_file_btn)
        file_buttons_layout.addWidget(self.clear_files_btn)

        output_layout = QHBoxLayout()
        output_label = QLabel("HiFi-SIM Output Directory:")
        self.output_path_edit = QLineEdit("./HiFi-SIM_resultDir")
        self.output_path_btn = QPushButton("Browse...")

        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(self.output_path_btn)

        layout.addWidget(file_list_label)
        layout.addWidget(self.file_list_widget)
        layout.addLayout(file_buttons_layout)
        layout.addLayout(output_layout)

        group.setLayout(layout)
        return group

    def create_hifi_parameters(self):
        group = QGroupBox("Parameters")
        layout = QHBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setStretch(0, 1)
        layout.setStretch(1, 1)
        layout.setStretch(2, 1)
        wavelength_layout = QHBoxLayout()
        wavelength_label = QLabel("λ (em)")
        wavelength_label.setStyleSheet("font-weight: bold")
        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setRange(300, 800)
        self.wavelength_spin.setValue(525)
        wavelength_unit = QLabel("nm")
        wavelength_layout.addWidget(wavelength_label)
        wavelength_layout.addWidget(self.wavelength_spin)
        wavelength_layout.addWidget(wavelength_unit)
        na_layout = QHBoxLayout()
        na_label = QLabel("NA")
        na_label.setStyleSheet("font-weight: bold")
        self.na_spin = QDoubleSpinBox()
        self.na_spin.setRange(0.1, 2.0)
        self.na_spin.setValue(1.41)
        self.na_spin.setSingleStep(0.01)
        na_layout.addSpacing(10)
        na_placeholder = QLabel("")
        na_layout.addWidget(na_label)
        na_layout.addWidget(self.na_spin)
        na_layout.addWidget(na_placeholder)
        pixel_size_layout = QHBoxLayout()
        pixel_size_label = QLabel("Pixel Size")
        pixel_size_label.setStyleSheet("font-weight: bold")
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(1.0, 100.0)
        self.pixel_size_spin.setValue(62.6)
        pixel_size_unit = QLabel("nm")
        pixel_size_layout.addWidget(pixel_size_label)
        pixel_size_layout.addWidget(self.pixel_size_spin)
        pixel_size_layout.addWidget(pixel_size_unit)
        layout.addLayout(wavelength_layout)
        layout.addLayout(na_layout)
        layout.addLayout(pixel_size_layout)
        group.setLayout(layout)
        return group

    def create_start_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 30, 0, 10)

        button_container = QWidget()
        button_container.setFixedHeight(100)
        button_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button_container_layout = QVBoxLayout(button_container)
        button_container_layout.setContentsMargins(0, 0, 0, 0)
        self.start_btn = QPushButton("START")
        button_width = 200
        font_size = 24
        self.start_btn.setFixedWidth(button_width)
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #ff6b6b, 
                    stop: 0.4 #e74c3c, 
                    stop: 0.6 #c0392b, 
                    stop: 1 #a93226  
                );

                color: white;
                font-weight: bold;
                font-size: {font_size}px;

                border: none;
                border-radius: 12px;

                box-shadow: 
                    0 6px 0 #7b241c, 
                    0 8px 12px rgba(231, 76, 60, 0.3);

                text-shadow: 1px 1px 1px rgba(0, 0, 0, 0.3);

                padding: 18px 40px;
                min-width: {button_width}px;
            }}

            QPushButton:hover {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #ff8e8e,  
                    stop: 0.4 #ff6b6b,
                    stop: 0.6 #e74c3c, 
                    stop: 1 #c0392b  
                );
                box-shadow: 
                    0 7px 0 #7b241c, 
                    0 10px 16px rgba(231, 76, 60, 0.4);
                transform: translateY(-2px); 
            }}

            QPushButton:pressed {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #a93226, 
                    stop: 0.4 #922b21, 
                    stop: 0.6 #7b241c, 
                    stop: 1 #641e16   
                );
                box-shadow: 
                    0 2px 0 #4a150f,  
                    0 4px 8px rgba(231, 76, 60, 0.2);
                padding: 16px 38px;
            }}

            QPushButton:disabled {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #bdc3c7, 
                    stop: 0.4 #95a5a6,
                    stop: 0.6 #7f8c8d,
                    stop: 1 #5d6d7e
                );
                color: #ecf0f1;
                box-shadow: 
                    0 6px 0 #5d6d7e,
                    0 8px 12px rgba(149, 165, 166, 0.3);
            }}
        """)

        self.start_btn.setMinimumHeight(65)
        font = QFont()
        font.setPointSize(font_size - 2)
        font.setBold(True)
        self.start_btn.setFont(font)
        self.start_btn.setToolTip(
            "Start reconstruction process\n"
            "Raw Data Mode: HIFI-SIM reconstruction followed by optional neural network intensity correction\n"
            "SIM Images Mode: Neural network intensity correction only"
        )

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.start_btn)
        button_layout.addStretch()
        button_container_layout.addLayout(button_layout)

        progress_container = QWidget()
        progress_container.setFixedHeight(0)
        progress_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        progress_container_layout = QVBoxLayout(progress_container)
        progress_container_layout.setContentsMargins(0, 0, 0, 0)
        progress_container_layout.setSpacing(0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(35)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 5px;
                background-color: #f8f9fa;
                text-align: left;
                padding: 6px 12px;   
                font-size: 11px;
                font-weight: bold;
                color: black;  
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 5px;
                border: 1px solid #45a049;
            }
        """)

        progress_container_layout.addWidget(self.progress_bar)
        self.progress_container = progress_container
        layout.addWidget(button_container)
        layout.addSpacing(15)
        layout.addWidget(self.progress_container)
        return widget

    def create_nn_parameters(self):

        group = QGroupBox("")
        layout = QVBoxLayout()
        wf_layout = QHBoxLayout()
        self.nn_wf_label = QLabel("WF Directory:")
        self.nn_wf_dir_edit = QLineEdit()
        self.nn_wf_browse_btn = QPushButton("Browse...")
        wf_layout.addWidget(self.nn_wf_label)
        wf_layout.addWidget(self.nn_wf_dir_edit)
        wf_layout.addWidget(self.nn_wf_browse_btn)
        sim_layout = QHBoxLayout()
        self.nn_sim_label = QLabel("SIM Directory:")
        self.nn_sim_dir_edit = QLineEdit()
        self.nn_sim_browse_btn = QPushButton("Browse...")
        sim_layout.addWidget(self.nn_sim_label)
        sim_layout.addWidget(self.nn_sim_dir_edit)
        sim_layout.addWidget(self.nn_sim_browse_btn)
        model_layout = QHBoxLayout()
        model_label = QLabel("Model File:")
        self.nn_model_path_edit = QLineEdit("./checkpoint/best_checkpoint.pth.tar")
        self.nn_model_browse_btn = QPushButton("Browse...")
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.nn_model_path_edit)
        model_layout.addWidget(self.nn_model_browse_btn)
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Directory:")
        self.nn_output_path_edit = QLineEdit("./QHiFi_resultDir")
        self.nn_output_browse_btn = QPushButton("Browse...")
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.nn_output_path_edit)
        output_layout.addWidget(self.nn_output_browse_btn)

        layout.addLayout(wf_layout)
        layout.addLayout(sim_layout)
        layout.addLayout(model_layout)
        layout.addLayout(output_layout)

        group.setLayout(layout)
        return group

    def create_nn_process(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 20, 0, 10)
        self.nn_process_btn = QPushButton("START")
        button_width = 180
        font_size = 20
        self.nn_process_btn.setFixedWidth(button_width)
        self.nn_process_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #d9534f;
                color: white;
                font-weight: bold;
                font-size: {font_size}px;
                padding: 15px 30px;
                border: none;
                border-radius: 10px;
                min-width: {button_width}px;
            }}
            QPushButton:hover {{
                background-color: #c9302c;
                border: 3px solid #a71e2a;
            }}
            QPushButton:pressed {{
                background-color: #ac2925;
                padding: 13px 28px;
            }}
            QPushButton:disabled {{
                background-color: #e0a6a4;
                color: #f5f5f5;
            }}
        """)
        self.nn_process_btn.setMinimumHeight(65)
        font = QFont()
        font.setPointSize(font_size - 2)
        font.setBold(True)
        self.nn_process_btn.setFont(font)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.nn_process_btn)
        button_layout.addStretch()
        self.nn_progress_bar = QProgressBar()
        self.nn_progress_bar.setVisible(False)
        self.nn_progress_bar.setFixedHeight(25)
        layout.addLayout(button_layout)
        layout.addWidget(self.nn_progress_bar)
        return widget

    def on_nn_mode_changed(self, mode):
        is_external = (mode == "From SIM images")
        self.file_group.setVisible(not is_external)
        self.nn_wf_label.setVisible(is_external)
        self.nn_wf_dir_edit.setVisible(is_external)
        self.nn_wf_browse_btn.setVisible(is_external)
        self.nn_sim_label.setVisible(is_external)
        self.nn_sim_dir_edit.setVisible(is_external)
        self.nn_sim_browse_btn.setVisible(is_external)

    def update_process_button_state(self):
        mode = self.nn_mode_combo.currentText()
        if mode == "From SIM images":
            wf_dir = self.nn_wf_dir_edit.text()
            sim_dir = self.nn_sim_dir_edit.text()
            has_wf_dir = bool(wf_dir and os.path.exists(wf_dir))
            has_sim_dir = bool(sim_dir and os.path.exists(sim_dir))
            self.start_btn.setEnabled(has_wf_dir and has_sim_dir)
        else:
            has_raw_files = len(self.parent.raw_files) > 0 if hasattr(self.parent, 'raw_files') else False
            self.start_btn.setEnabled(has_raw_files)

    def update_file_progress(self, current, total):
        self.file_progress_label.setText(f"HIFI-SIM: Processing file {current} of {total}")

    def set_processing_state(self, is_processing):
        self.start_btn.setEnabled(not is_processing)
        if is_processing:
            self.show_progress_bar()
            self.reset_progress()
        else:
            pass
