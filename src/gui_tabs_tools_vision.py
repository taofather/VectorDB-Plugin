import sys
import textwrap
import subprocess
from pathlib import Path
import logging
import yaml
import tempfile
import os
import traceback
import gc
import time
from PIL import Image
import torch
from PySide6.QtCore import QThread, Signal as pyqtSignal, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QMessageBox, QFileDialog, QProgressDialog, QDialog, QCheckBox

import module_process_images
from module_process_images import choose_image_loader
from constants import VISION_MODELS
from config_manager import ConfigManager

class ModelSelectionDialog(QDialog):
    def __init__(self, models, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Vision Models")
        layout = QVBoxLayout()
        
        self.checkboxes = {}
        for model_name, info in models.items():
            checkbox = QCheckBox(f"{model_name} (VRAM: {info['vram']})")
            checkbox.setChecked(True)
            self.checkboxes[model_name] = checkbox
            layout.addWidget(checkbox)
        
        buttons_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(ok_button)
        buttons_layout.addWidget(cancel_button)
        
        layout.addLayout(buttons_layout)
        self.setLayout(layout)
    
    def get_selected_models(self):
        return [model for model, checkbox in self.checkboxes.items() if checkbox.isChecked()]

class ImageProcessorThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def run(self):
        try:
            documents = choose_image_loader()
            self.finished.emit(documents)
        except Exception as e:
            error_msg = f"Error in image processing: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)

class MultiModelProcessorThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, image_path, selected_models):
        super().__init__()
        self.image_path = image_path
        self.selected_models = selected_models
        self.is_cancelled = False
        self.config_manager = ConfigManager()

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        try:
            results = []
            with Image.open(self.image_path) as raw_image:
                for i, model_name in enumerate(self.selected_models):
                    if self.is_cancelled:
                        print("\nProcessing cancelled by user")
                        torch.cuda.empty_cache()
                        gc.collect()
                        break

                    try:
                        print(f"\nProcessing with {model_name}...")
                        config = self.config_manager.get_config()
                        config['vision'] = {'chosen_model': model_name}
                        self.config_manager.save_config(config)

                        loader_name = VISION_MODELS[model_name]['loader']
                        loader_class = getattr(module_process_images, loader_name)
                        loader = loader_class(config)

                        loader.model, loader.tokenizer, loader.processor = loader.initialize_model_and_tokenizer()
                        start_time = time.time()
                        description = loader.process_single_image(raw_image)
                        process_time = time.time() - start_time
                        description = textwrap.fill(description, width=10)
                        results.append((model_name, description, process_time))

                        if hasattr(loader, 'model') and loader.model is not None:
                            loader.model.cpu()
                            del loader.model
                        if hasattr(loader, 'tokenizer') and loader.tokenizer is not None:
                            del loader.tokenizer
                        if hasattr(loader, 'processor') and loader.processor is not None:
                            del loader.processor

                        torch.cuda.empty_cache()
                        gc.collect()

                        print(f"Completed {model_name}")
                        self.progress.emit(i + 1)

                    except Exception as e:
                        error_msg = f"Error processing with {model_name}: {str(e)}\n{traceback.format_exc()}"
                        results.append((model_name, error_msg, 0.0))
                        print(error_msg)
                        torch.cuda.empty_cache()
                        gc.collect()

            torch.cuda.empty_cache()
            gc.collect()
            self.finished.emit(results)
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            self.error.emit(str(e))

class VisionToolSettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()

        mainVLayout = QVBoxLayout()
        self.setLayout(mainVLayout)

        hBoxLayout = QHBoxLayout()
        mainVLayout.addLayout(hBoxLayout)

        processButton = QPushButton("Multiple Files + One Vision Model")
        hBoxLayout.addWidget(processButton)
        processButton.clicked.connect(self.confirmationBeforeProcessing)

        newButton = QPushButton("Single Image + All Vision Models")
        hBoxLayout.addWidget(newButton)
        newButton.clicked.connect(self.selectSingleImage)

        self.thread = None

    def confirmationBeforeProcessing(self):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(
            "1. Create Database Tab:\n"
            "Select files you theoretically want in the vector database.\n\n"
            "2. Settings Tab:\n"
            "Select the vision model you want to test.\n\n"
            "3. Click the 'Process' button.\n\n"
            "This will test the selected vison model before actually entering the images into the vector database.\n\n"
            "Do you want to proceed?"
        )
        msgBox.setWindowTitle("Confirm Processing")
        msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        returnValue = msgBox.exec()
        if returnValue == QMessageBox.Ok:
            self.startProcessing()

    def startProcessing(self):
        if self.thread is None:
            self.thread = ImageProcessorThread()
            self.thread.finished.connect(self.onProcessingFinished)
            self.thread.error.connect(self.onProcessingError)
            self.thread.start()

    def onProcessingFinished(self, documents):
        self.thread = None
        print(f"Processed {len(documents)} documents")
        contents = self.extract_page_content(documents)
        self.save_page_contents(contents)

    def onProcessingError(self, error_msg):
        self.thread = None
        logging.error(f"Processing error: {error_msg}")
        QMessageBox.critical(self, "Processing Error", f"An error occurred during image processing:\n\n{error_msg}")

    def extract_page_content(self, documents):
        contents = []
        total_length = 0

        for doc in documents:
            if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                content = doc.page_content
                filepath = doc.metadata.get('source', doc.metadata.get('file_path', doc.metadata.get('file_name', 'Unknown filepath')))
            elif isinstance(doc, dict):
                content = doc.get("page_content", "Document is missing 'page_content'.")
                filepath = doc.get("metadata", {}).get('source', 
                         doc.get("metadata", {}).get('file_path',
                         doc.get("metadata", {}).get('file_name', 'Unknown filepath')))
            else:
                continue

            total_length += len(content)
            contents.append((filepath, content))

        return contents

    def save_page_contents(self, contents):
        if not contents:
            QMessageBox.warning(self, "No Content", "No content was extracted from the documents.")
            return

        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
                for filepath, content in contents:
                    temp_file.write(f"File: {filepath}\n")
                    temp_file.write(f"Content: {content}\n")
                    temp_file.write("-" * 80 + "\n")

            self.open_file(temp_file.name)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save contents: {str(e)}")

    def save_comparison_results(self, image_path, results):
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
                temp_file.write(f"Image: {image_path}\n\n")
                for model_name, description, process_time in results:
                    temp_file.write(f"Model: {model_name}\n")
                    temp_file.write(f"Processing Time: {process_time:.2f} seconds\n")
                    temp_file.write("Description:\n")
                    temp_file.write(f"{description}\n")
                    temp_file.write("-" * 80 + "\n\n")

            self.open_file(temp_file.name)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")

    def selectSingleImage(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;All Files (*)"
        )
        if not file_name:
            return

        dialog = ModelSelectionDialog(VISION_MODELS, self)
        if dialog.exec() != QDialog.Accepted:
            return

        selected_models = dialog.get_selected_models()
        if not selected_models:
            QMessageBox.warning(self, "No Models Selected", "Please select at least one model.")
            return

        progress_dialog = QProgressDialog("Processing image with multiple models...", "Cancel", 0, len(selected_models), self)
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setAutoClose(False)

        self.thread = MultiModelProcessorThread(file_name, selected_models)
        self.thread.progress.connect(progress_dialog.setValue)
        self.thread.finished.connect(lambda results: self.onMultiModelProcessingFinished(file_name, results))
        self.thread.error.connect(self.onMultiModelProcessingError)
        progress_dialog.canceled.connect(self.cancelProcessing)
        self.thread.start()

    def cancelProcessing(self):
        if self.thread:
            self.thread.cancel()

    def onMultiModelProcessingFinished(self, image_path, results):
        self.thread = None
        if results:
            self.save_comparison_results(image_path, results)

    def onMultiModelProcessingError(self, error_msg):
        self.thread = None
        QMessageBox.critical(self, "Processing Error", f"An error occurred during processing:\n\n{error_msg}")

    def open_file(self, file_path):
        if sys.platform == 'win32':
            os.startfile(file_path)
        elif sys.platform == 'darwin':
            subprocess.run(['open', file_path])
        else:
            subprocess.run(['xdg-open', file_path])