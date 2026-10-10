from PyQt6.QtCore import QThread, pyqtSignal
from util.quant_analysis import run_quant_analysis

class QuantAnalysisWorker(QThread):
    finished_signal = pyqtSignal(bool, str)
    progress_signal = pyqtSignal(int)

    def __init__(self, wf_dir, sim_dir, output_dir):
        super().__init__()
        self.wf_dir = wf_dir
        self.sim_dir = sim_dir
        self.output_dir = output_dir

    def run(self):
        try:
            success, message = run_quant_analysis(
                self.wf_dir, self.sim_dir, self.output_dir
            )
            self.progress_signal.emit(100)
            self.finished_signal.emit(success, message)
        except Exception as e:
            self.finished_signal.emit(False, f"Analysis failed: {str(e)}")