import logging
from pathlib import Path
import json
import requests
from PySide6.QtCore import QThread, Signal, QObject
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextBrowser, 
    QLabel, QMessageBox, QSizePolicy, QSpinBox
)
from config_manager import ConfigManager


class RPGSignals(QObject):
    response_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()


class RPGThread(QThread):
    def __init__(self, context_content, thread_type="plot"):
        super().__init__()
        self.context_content = context_content
        self.thread_type = thread_type  # "plot" or "characters"
        self.signals = RPGSignals()
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        self.server_config = self.config.get('server', {})
        self.host = self.server_config.get('host', 'localhost')
        self.port = self.server_config.get('port', 1234)

    def run(self):
        try:
            url = f"http://{self.host}:{self.port}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            data = {
                "messages": [
                    {"role": "system", "content": self.context_content},
                    {"role": "user", "content": "Escriu el resum d'una història pel joc de rol Aquelarre. Tindrà lloc en diferents paratges (pobles, boscos, ports, ciutats) a la península ibèrica Medieval (1300-1400)."}
                ],
                "temperature": 0.8,
                "max_tokens": -1,
                "stream": True
            }

            response = requests.post(url, headers=headers, json=data, stream=True)
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            line_text = line_text[6:]  # Remove 'data: ' prefix
                        
                        if line_text.strip() == '[DONE]':
                            break
                            
                        json_response = json.loads(line_text)
                        if 'choices' in json_response and json_response['choices']:
                            content = json_response['choices'][0].get('delta', {}).get('content', '')
                            if content:
                                full_response += content
                                self.signals.response_signal.emit(content)
                    except json.JSONDecodeError:
                        continue

            # Save the generated content
            if self.thread_type == "plot":
                self.save_plot_to_file(full_response)
            elif self.thread_type == "characters":
                self.save_characters_to_file(full_response)
            self.signals.finished_signal.emit()

        except requests.exceptions.RequestException as e:
            error_message = f"Error connecting to LM Studio server: {str(e)}"
            logging.error(error_message)
            self.signals.error_signal.emit(error_message)
            self.signals.finished_signal.emit()

    def save_plot_to_file(self, plot_content):
        """Save the generated plot to contexts/plot directory"""
        try:
            plot_dir = Path(__file__).parent.parent / 'contexts' / 'plot'
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Find the next available plot file number
            existing_plots = list(plot_dir.glob('plot-*.md'))
            plot_number = len(existing_plots) + 1
            
            plot_file = plot_dir / f'plot-{plot_number:03d}.md'
            
            with open(plot_file, 'w', encoding='utf-8') as f:
                f.write(f"# Plot {plot_number}\n\n")
                f.write(plot_content)
            
            logging.info(f"Plot saved to: {plot_file}")
            
        except Exception as e:
            logging.error(f"Error saving plot: {e}")


class RPGTab(QWidget):
    def __init__(self):
        super().__init__()
        self.rpg_thread = None
        self.current_response = ""
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("🎲 Aquelarre RPG - Game Master Assistant")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("Generate new plots and manage RPG campaigns for the Spanish medieval RPG Aquelarre.")
        desc_label.setStyleSheet("margin: 5px; color: #666;")
        layout.addWidget(desc_label)
        
        # Start new plot button
        button_layout = QHBoxLayout()
        self.start_plot_button = QPushButton("🚀 Start a new plot")
        self.start_plot_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.start_plot_button.clicked.connect(self.start_new_plot)
        button_layout.addWidget(self.start_plot_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Response display area
        self.response_browser = QTextBrowser()
        self.response_browser.setPlaceholderText("Generated plot will appear here...")
        layout.addWidget(self.response_browser, 1)
        
        # Status label
        self.status_label = QLabel("Ready to generate a new plot")
        self.status_label.setStyleSheet("color: #666; margin: 5px;")
        layout.addWidget(self.status_label)

    def start_new_plot(self):
        """Start generating a new plot using the system-summarizer context"""
        try:
            # Load the system-summarizer context
            context_file = Path(__file__).parent.parent / 'contexts' / 'game' / 'system-summarizer.md'
            
            if not context_file.exists():
                QMessageBox.warning(
                    self,
                    "Context File Missing",
                    f"The system-summarizer.md file is missing at:\n{context_file}\n\nPlease ensure the contexts/game directory contains the required files."
                )
                return
            
            with open(context_file, 'r', encoding='utf-8') as f:
                context_content = f.read()
            
            if not context_content.strip():
                QMessageBox.warning(
                    self,
                    "Empty Context File",
                    "The system-summarizer.md file is empty. Please add the appropriate context content."
                )
                return
            
            # Check if server is configured
            config_manager = ConfigManager()
            config = config_manager.get_config()
            server_config = config.get('server', {})
            host = server_config.get('host', 'localhost')
            port = server_config.get('port', 1234)
            
            if not host or not port:
                QMessageBox.warning(
                    self,
                    "Server Configuration Missing",
                    "Please configure the LM Studio server settings in the config.yaml file."
                )
                return
            
            # Disable button and update status
            self.start_plot_button.setEnabled(False)
            self.start_plot_button.setText("🔄 Generating plot...")
            self.status_label.setText("Connecting to LM Studio and generating plot...")
            self.response_browser.clear()
            self.current_response = ""
            
            # Start the RPG thread
            self.rpg_thread = RPGThread(context_content)
            self.rpg_thread.signals.response_signal.connect(self.update_response)
            self.rpg_thread.signals.error_signal.connect(self.show_error)
            self.rpg_thread.signals.finished_signal.connect(self.on_finished)
            self.rpg_thread.start()
            
        except Exception as e:
            logging.error(f"Error starting new plot: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to start plot generation: {str(e)}"
            )
            self.reset_ui()

    def update_response(self, chunk):
        """Update the response display with new content chunks"""
        self.current_response += chunk
        self.response_browser.setPlainText(self.current_response)
        
        # Auto-scroll to bottom
        cursor = self.response_browser.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.response_browser.setTextCursor(cursor)

    def show_error(self, error_message):
        """Show error message to user"""
        self.status_label.setText(f"Error: {error_message}")
        QMessageBox.critical(
            self,
            "Error",
            f"Failed to generate plot:\n{error_message}"
        )
        self.reset_ui()

    def on_finished(self):
        """Called when plot generation is complete"""
        self.status_label.setText("Plot generation completed! Check contexts/plot directory for saved file.")
        self.reset_ui()
        
        if self.current_response.strip():
            QMessageBox.information(
                self,
                "Plot Generated",
                "New plot has been generated and saved to contexts/plot directory!"
            )

    def reset_ui(self):
        """Reset UI elements to initial state"""
        self.start_plot_button.setEnabled(True)
        self.start_plot_button.setText("🚀 Start a new plot")

    def cleanup(self):
        """Cleanup when tab is closed"""
        if self.rpg_thread and self.rpg_thread.isRunning():
            self.rpg_thread.terminate()
            self.rpg_thread.wait() 