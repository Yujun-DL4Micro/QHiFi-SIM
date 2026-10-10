import sys
from PyQt6.QtWidgets import QApplication
from util.gui_main_window import MainWindow
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()

