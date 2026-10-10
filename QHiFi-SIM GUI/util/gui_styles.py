def get_stylesheet():
    return """
    QMainWindow {
        background-color: #f0f0f0;  
    }
    QGroupBox {
        font-weight: bold;  
        font-size: 10pt;
        border: 2px solid #cccccc;  
        border-radius: 5px;  
        margin-top: 1ex; 
        padding-top: 12px;  
        background-color: #ffffff;  
    }
    QGroupBox::title {
        subcontrol-origin: margin;  
        left: 10px;  
        padding: 0 5px 0 5px;  
    }
    QPushButton {
        padding: 8px 16px;  
        border-radius: 4px;  
        font-size: 14px;  
        margin: 4px 2px;  
    }
    QPushButton:hover {
        opacity: 0.9;  
    }
    QPushButton:disabled {
        background-color: #cccccc;  
    }
    QLineEdit, QSpinBox, QDoubleSpinBox {
        padding: 5px;  
        border: 1px solid #ccc;  
        border-radius: 3px;  
        font-size: 12pt;
    }
    QListWidget {
        border: 1px solid #ccc;  
        border-radius: 3px;  
        background-color: white;  
        font-size: 12pt;
    }
    QTabWidget::pane {
        border: 1px solid #cccccc;  
        background-color: white;  
    }
    QLabel {
        font-size: 12pt;  
    }
    QComboBox {
        font-size: 12pt;  
        padding: 4px;
    }
    """