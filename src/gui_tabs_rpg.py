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
            
            # The context_content is already a complete JSON structure for the API request
            data = json.loads(self.context_content)

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

    def save_characters_to_file(self, characters_content):
        """Save the generated characters to contexts/characters directory"""
        try:
            characters_dir = Path(__file__).parent.parent / 'contexts' / 'characters'
            characters_dir.mkdir(parents=True, exist_ok=True)
            
            # Find the next available characters file number
            existing_characters = list(characters_dir.glob('characters-*.md'))
            characters_number = len(existing_characters) + 1
            
            characters_file = characters_dir / f'characters-{characters_number:03d}.md'
            
            with open(characters_file, 'w', encoding='utf-8') as f:
                f.write(f"# Characters Set {characters_number}\n\n")
                f.write(characters_content)
            
            logging.info(f"Characters saved to: {characters_file}")
            
        except Exception as e:
            logging.error(f"Error saving characters: {e}")


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
        
        # Buttons section
        buttons_layout = QVBoxLayout()
        
        # Start new plot button
        plot_button_layout = QHBoxLayout()
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
        plot_button_layout.addWidget(self.start_plot_button)
        plot_button_layout.addStretch()
        buttons_layout.addLayout(plot_button_layout)
        
        # Generate characters section
        characters_layout = QHBoxLayout()
        
        # Number of characters input
        num_chars_label = QLabel("Number of characters:")
        num_chars_label.setStyleSheet("margin: 5px;")
        characters_layout.addWidget(num_chars_label)
        
        self.num_characters_spinbox = QSpinBox()
        self.num_characters_spinbox.setMinimum(1)
        self.num_characters_spinbox.setMaximum(8)
        self.num_characters_spinbox.setValue(3)
        self.num_characters_spinbox.setStyleSheet("margin: 5px; padding: 5px;")
        characters_layout.addWidget(self.num_characters_spinbox)
        
        # Generate characters button
        self.generate_characters_button = QPushButton("👥 Generate characters")
        self.generate_characters_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.generate_characters_button.clicked.connect(self.generate_characters)
        characters_layout.addWidget(self.generate_characters_button)
        characters_layout.addStretch()
        buttons_layout.addLayout(characters_layout)
        
        layout.addLayout(buttons_layout)
        
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
            context_file = Path(__file__).parent.parent / 'contexts' / 'game' / 'system-summarizer.json'
            
            if not context_file.exists():
                QMessageBox.warning(
                    self,
                    "Context File Missing",
                    f"The system-summarizer.json file is missing at:\n{context_file}\n\nPlease ensure the contexts/game directory contains the required files."
                )
                return
            
            with open(context_file, 'r', encoding='utf-8') as f:
                context_content = f.read()
            
            if not context_content.strip():
                QMessageBox.warning(
                    self,
                    "Empty Context File",
                    "The system-summarizer.json file is empty. Please add the appropriate context content."
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
        self.reset_characters_ui()

    def generate_characters(self):
        """Generate characters using the system-character-creation context with tag substitutions"""
        try:
            # Load the system character creation JSON template
            json_file = Path(__file__).parent.parent / 'contexts' / 'game' / 'system-character-creation.json'
            md_file = Path(__file__).parent.parent / 'contexts' / 'game' / 'system-character-creation.md'
            
            if not json_file.exists():
                QMessageBox.warning(
                    self,
                    "JSON File Missing",
                    f"The system-character-creation.json file is missing at:\n{json_file}\n\nPlease ensure the contexts/game directory contains the required files."
                )
                return
                
            if not md_file.exists():
                QMessageBox.warning(
                    self,
                    "Markdown File Missing",
                    f"The system-character-creation.md file is missing at:\n{md_file}\n\nPlease ensure the contexts/game directory contains the required files."
                )
                return
            
            # Load JSON template
            with open(json_file, 'r', encoding='utf-8') as f:
                json_content = f.read()
            
            # Load Markdown content
            with open(md_file, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            if not json_content.strip():
                QMessageBox.warning(
                    self,
                    "Empty JSON File",
                    "The system-character-creation.json file is empty. Please add the appropriate context content."
                )
                return
                
            if not md_content.strip():
                QMessageBox.warning(
                    self,
                    "Empty Markdown File",
                    "The system-character-creation.md file is empty. Please add the appropriate context content."
                )
                return
            
            # Get the number of characters from the spinbox
            num_characters = self.num_characters_spinbox.value()
            
            # Load the plot content for substitution
            plot_file = Path(__file__).parent.parent / 'contexts' / 'plot' / 'plot-001.md'
            plot_content = ""
            
            if plot_file.exists():
                with open(plot_file, 'r', encoding='utf-8') as f:
                    plot_content = f.read()
            else:
                QMessageBox.warning(
                    self,
                    "Plot File Missing",
                    f"No plot file found at:\n{plot_file}\n\nPlease generate a plot first before creating characters."
                )
                return
            
            # Substitute markdown content and tags
            context_content = self.substitute_markdown_and_tags(json_content, md_content, num_characters, plot_content)
            
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
            self.generate_characters_button.setEnabled(False)
            self.generate_characters_button.setText("🔄 Generating characters...")
            self.status_label.setText("Connecting to LM Studio and generating characters...")
            self.response_browser.clear()
            self.current_response = ""
            
            # Start the RPG thread for character generation
            self.rpg_thread = RPGThread(context_content, "characters")
            self.rpg_thread.signals.response_signal.connect(self.update_response)
            self.rpg_thread.signals.error_signal.connect(self.show_error)
            self.rpg_thread.signals.finished_signal.connect(self.on_characters_finished)
            self.rpg_thread.start()
            
        except Exception as e:
            logging.error(f"Error generating characters: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to generate characters: {str(e)}"
            )
            self.reset_characters_ui()

    def substitute_markdown_and_tags(self, json_content, md_content, num_characters, plot_content):
        """Substitute markdown content and XHTML tags in the JSON content"""
        try:
            # Parse the JSON content
            context_data = json.loads(json_content)
            
            # First, substitute the markdown content in the JSON
            json_str = json.dumps(context_data, ensure_ascii=False, indent=2)
            
            # Replace %markdown-child% with the markdown content
            # We need to properly escape the markdown content for JSON
            escaped_md_content = json.dumps(md_content, ensure_ascii=False)[1:-1]  # Remove surrounding quotes
            json_str = json_str.replace('"%markdown-child%"', f'"{escaped_md_content}"')
            
            # Parse back to modify tags
            context_data = json.loads(json_str)
            
            # Iterate through messages and substitute XHTML tags in content
            for message in context_data.get('messages', []):
                if 'content' in message:
                    # Replace <num_characters></num_characters> with the actual number
                    message['content'] = message['content'].replace('<num_characters></num_characters>', str(num_characters))
                    
                    # Replace <plot></plot> with the plot content
                    message['content'] = message['content'].replace('<plot></plot>', plot_content)
            
            # Return the modified JSON as string
            return json.dumps(context_data, ensure_ascii=False, indent=2)
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            raise Exception(f"Invalid JSON format: {e}")
        except Exception as e:
            logging.error(f"Error in substitute_markdown_and_tags: {e}")
            raise

    def on_characters_finished(self):
        """Called when character generation is complete"""
        self.status_label.setText("Character generation completed! Check contexts/characters directory for saved file.")
        self.reset_characters_ui()
        
        if self.current_response.strip():
            QMessageBox.information(
                self,
                "Characters Generated",
                "New characters have been generated and saved to contexts/characters directory!"
            )

    def reset_characters_ui(self):
        """Reset character generation UI elements to initial state"""
        self.generate_characters_button.setEnabled(True)
        self.generate_characters_button.setText("👥 Generate characters")

    def cleanup(self):
        """Cleanup when tab is closed"""
        if self.rpg_thread and self.rpg_thread.isRunning():
            self.rpg_thread.terminate()
            self.rpg_thread.wait() 