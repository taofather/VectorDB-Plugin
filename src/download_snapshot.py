import sys
import os
import threading
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox, QCheckBox
from PySide6.QtCore import Qt, Signal, Slot, QObject
from huggingface_hub import snapshot_download, login, HfApi
from huggingface_hub.utils import disable_progress_bars

class DownloadWorker(QObject):
    finished = Signal(bool, str)
    progress = Signal(str)

    def __init__(self, repo_id, local_dir, allow_patterns, ignore_patterns, revision=None):
        super().__init__()
        self.repo_id = repo_id
        self.local_dir = local_dir
        self.allow_patterns = allow_patterns
        self.ignore_patterns = ignore_patterns
        self.revision = revision

    def run(self):
        try:
            snapshot_download(
                repo_id=self.repo_id,
                local_dir=self.local_dir,
                max_workers=3,
                allow_patterns=self.allow_patterns if self.allow_patterns else None,
                ignore_patterns=self.ignore_patterns if self.ignore_patterns else None,
                revision=self.revision if self.revision else None,
            )
            self.finished.emit(True, f"Download completed successfully to {self.local_dir}")
        except Exception as e:
            self.finished.emit(False, f"An error occurred during download: {str(e)}")

class HuggingFaceDownloader(QMainWindow):
   def __init__(self):
       super().__init__()
       self.setWindowTitle("Hugging Face Repository Downloader")
       self.setGeometry(100, 100, 500, 450)

       central_widget = QWidget()
       self.setCentralWidget(central_widget)
       layout = QVBoxLayout(central_widget)

       self.repo_label = QLabel("Repository ID:")
       self.repo_entry = QLineEdit()
       layout.addWidget(self.repo_label)
       layout.addWidget(self.repo_entry)

       self.revision_label = QLabel("Revision (branch/tag/commit, optional):")
       self.revision_entry = QLineEdit()
       layout.addWidget(self.revision_label)
       layout.addWidget(self.revision_entry)

       self.dir_label = QLabel("Download Directory (optional):")
       self.dir_entry = QLineEdit()
       self.dir_button = QPushButton("Browse")
       self.dir_button.clicked.connect(self.browse_directory)
       dir_layout = QHBoxLayout()
       dir_layout.addWidget(self.dir_entry)
       dir_layout.addWidget(self.dir_button)
       layout.addWidget(self.dir_label)
       layout.addLayout(dir_layout)

       self.standard_patterns_checkbox = QCheckBox("Use Standard Ignore/Allow Patterns")
       self.standard_patterns_checkbox.stateChanged.connect(self.toggle_pattern_inputs)
       layout.addWidget(self.standard_patterns_checkbox)

       self.pattern_widgets = []

       self.allow_label = QLabel("Allow Patterns (comma-separated):")
       self.allow_entry = QLineEdit()
       self.allow_examples = QLabel("Examples: *.txt, model.bin, configs/*.json")
       self.allow_examples.setStyleSheet("color: gray;")
       self.pattern_widgets.extend([self.allow_label, self.allow_entry, self.allow_examples])
       layout.addWidget(self.allow_label)
       layout.addWidget(self.allow_entry)
       layout.addWidget(self.allow_examples)

       self.ignore_label = QLabel("Ignore Patterns (comma-separated):")
       self.ignore_entry = QLineEdit()
       self.ignore_examples = QLabel("Examples: *.md, test_*, data/raw/*")
       self.ignore_examples.setStyleSheet("color: gray;")
       self.pattern_widgets.extend([self.ignore_label, self.ignore_entry, self.ignore_examples])
       layout.addWidget(self.ignore_label)
       layout.addWidget(self.ignore_entry)
       layout.addWidget(self.ignore_examples)

       self.standard_ignore_patterns = [
           ".gitattributes",
           "*.ckpt",
           "*.gguf",
           "*.h5",
           "*.ot",
           "*.md",
           "README*",
           "onnx/**",
           "coreml/**",
           "openvino/**",
           "demo/**"
       ]

       self.token_label = QLabel("Hugging Face Access Token:")
       self.token_entry = QLineEdit()
       self.token_entry.setEchoMode(QLineEdit.Password)
       self.token_entry.setText("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")
       self.login_button = QPushButton("Login")
       self.login_button.clicked.connect(self.perform_login)
       layout.addWidget(self.token_label)
       layout.addWidget(self.token_entry)
       layout.addWidget(self.login_button)

       self.download_button = QPushButton("Begin Download")
       self.download_button.clicked.connect(self.start_download)
       layout.addWidget(self.download_button)

       self.status_label = QLabel("")
       layout.addWidget(self.status_label)

       self.worker = None
       self.thread = None

   def browse_directory(self):
       directory = QFileDialog.getExistingDirectory(self, "Select Directory")
       if directory:
           self.dir_entry.setText(directory)

   def perform_login(self):
       token = self.token_entry.text().strip()
       if token:
           try:
               login(token=token)
               self.show_success("Successfully logged in to Hugging Face")
           except Exception as e:
               self.show_error(f"Login failed: {str(e)}")
       else:
           self.show_error("Please enter a token")

   def toggle_pattern_inputs(self, state):
       for widget in self.pattern_widgets:
           widget.setEnabled(not state)

   def get_patterns(self):
       if self.standard_patterns_checkbox.isChecked():
           repo_id = self.repo_entry.text().strip()
           try:
               api = HfApi()
               repo_files = list(api.list_repo_tree(repo_id, recursive=True))
               
               safetensors_files = [file for file in repo_files if file.path.endswith('.safetensors')]
               bin_files = [file for file in repo_files if file.path.endswith('.bin')]
               
               ignore_patterns = self.standard_ignore_patterns.copy()
               
               if safetensors_files and bin_files:
                   ignore_patterns.append("*.bin")
               
               if safetensors_files or bin_files:
                   ignore_patterns.append("*consolidated*")
               
               return [], ignore_patterns
           except Exception as e:
               self.show_error(f"Error checking repository files: {str(e)}")
               return [], self.standard_ignore_patterns
       else:
           allow_patterns = [p.strip() for p in self.allow_entry.text().split(',') if p.strip()]
           ignore_patterns = [p.strip() for p in self.ignore_entry.text().split(',') if p.strip()]
           return allow_patterns, ignore_patterns

   @Slot()
   def start_download(self):
       repo_id = self.repo_entry.text().strip()
       local_dir = self.dir_entry.text().strip()
       revision = self.revision_entry.text().strip()  # Get the revision value
       
       if not repo_id:
           self.show_error("Please enter a repository ID.")
           return

       if not local_dir:
           folder_name = repo_id.replace('/', '--')
           local_dir = os.path.join(os.getcwd(), folder_name)
           os.makedirs(local_dir, exist_ok=True)
           self.show_info(f"No directory specified. Created and downloading to: {local_dir}")

       allow_patterns, ignore_patterns = self.get_patterns()

       self.download_button.setEnabled(False)
       self.status_label.setText("Download in progress. Check command prompt for details.")

       self.worker = DownloadWorker(repo_id, local_dir, allow_patterns, ignore_patterns, revision)
       self.thread = threading.Thread(target=self.worker.run)
       self.worker.finished.connect(self.on_download_finished)
       self.thread.start()

   @Slot(bool, str)
   def on_download_finished(self, success, message):
       if success:
           self.show_success(message)
       else:
           self.show_error(message)
       self.download_button.setEnabled(True)
       self.status_label.setText("")

   def show_error(self, message):
       QMessageBox.critical(self, "Error", message)

   def show_info(self, message):
       QMessageBox.information(self, "Info", message)

   def show_success(self, message):
       QMessageBox.information(self, "Success", message)

if __name__ == "__main__":
   app = QApplication(sys.argv)
   app.setStyle('Fusion')
   downloader = HuggingFaceDownloader()
   downloader.show()
   sys.exit(app.exec())