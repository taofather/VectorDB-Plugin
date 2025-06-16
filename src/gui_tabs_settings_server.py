from constants import TOOLTIPS
import yaml
from pathlib import Path

from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import (QWidget, QLabel, QLineEdit, QGridLayout, QMessageBox, QSizePolicy, QCheckBox)

from config_manager import ConfigManager

class ServerSettingsTab(QWidget):
    def __init__(self):
        super(ServerSettingsTab, self).__init__()
        self.config_manager = ConfigManager()
        config_data = self.config_manager.get_config()
        
        self.server_config = config_data.get('server', {})
        
        grid_layout = QGridLayout()
        
        # Host settings
        self.host_label = QLabel("Host:")
        grid_layout.addWidget(self.host_label, 0, 0)
        
        self.host_edit = QLineEdit()
        self.host_edit.setPlaceholderText("Enter host...")
        self.host_edit.setText(self.server_config.get('host', 'localhost'))
        grid_layout.addWidget(self.host_edit, 0, 1)
        
        # Port settings
        self.port_label = QLabel("Port:")
        grid_layout.addWidget(self.port_label, 1, 0)
        
        self.port_edit = QLineEdit()
        self.port_edit.setPlaceholderText("Enter port...")
        self.port_edit.setText(str(self.server_config.get('port', 5000)))
        grid_layout.addWidget(self.port_edit, 1, 1)
        
        self.setLayout(grid_layout)
        
        # Set tooltips after widget creation
        self.host_label.setToolTip(TOOLTIPS["SERVER_HOST"])
        self.host_edit.setToolTip(TOOLTIPS["SERVER_HOST"])
        self.port_label.setToolTip(TOOLTIPS["SERVER_PORT"])
        self.port_edit.setToolTip(TOOLTIPS["SERVER_PORT"])

    def update_config(self):
        try:
            config_data = self.config_manager.get_config()
            settings_changed = False
            
            # Update host
            new_host = self.host_edit.text().strip()
            if new_host and new_host != config_data['server'].get('host', 'localhost'):
                config_data['server']['host'] = new_host
                settings_changed = True
            
            # Update port
            new_port = self.port_edit.text().strip()
            if new_port:
                try:
                    port_num = int(new_port)
                    if port_num != config_data['server'].get('port', 5000):
                        config_data['server']['port'] = port_num
                        settings_changed = True
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Invalid Port",
                        "Port must be a valid number."
                    )
                    return False
            
            if settings_changed:
                self.config_manager.save_config(config_data)
            
            return settings_changed
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Updating Configuration",
                f"An error occurred while updating the configuration: {e}"
            )
            return False

    def update_show_thinking(self, state):
        try:
            with open('config.yaml', 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
            
            config_data['server']['show_thinking'] = bool(state)
            
            with open('config.yaml', 'w', encoding='utf-8') as file:
                yaml.safe_dump(config_data, file)
                
            self.show_thinking = bool(state)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Updating Configuration",
                f"An error occurred while updating show_thinking setting: {e}"
            )

    def create_label(self, setting, settings_dict):
        label_text = f"{setting.replace('_', ' ').capitalize()}: {settings_dict[setting]['current']}"
        label = QLabel(label_text)
        self.widgets[setting] = {"label": label}
        return label

    def create_edit(self, setting, settings_dict):
        edit = QLineEdit()
        edit.setPlaceholderText(settings_dict[setting]['placeholder'])
        edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if settings_dict[setting]['validator']:
            edit.setValidator(settings_dict[setting]['validator'])
        self.widgets[setting]['edit'] = edit
        return edit

    def update_config(self):
        config_file_path = Path('config.yaml')
        if config_file_path.exists():
            try:
                with config_file_path.open('r', encoding='utf-8') as file:
                    config_data = yaml.safe_load(file)
                    self.server_config = config_data.get('server', {})
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Loading Configuration",
                    f"An error occurred while loading the configuration: {e}"
                )
                return False
        else:
            QMessageBox.critical(
                self,
                "Configuration File Missing",
                "The configuration file 'config.yaml' does not exist."
            )
            return False

        settings_changed = False
        errors = []

        new_port_text = self.widgets['port']['edit'].text().strip()
        if new_port_text:
            try:
                new_port = int(new_port_text)
                if not (1 <= new_port <= 65535):
                    raise ValueError("Port must be between 1 and 65535.")
            except ValueError:
                errors.append("Port must be an integer between 1 and 65535.")
        else:
            new_port = self.current_port

        if errors:
            error_message = "\n".join(errors)
            QMessageBox.warning(
                self,
                "Invalid Input",
                f"The following errors occurred:\n{error_message}"
            )
            return False

        if new_port_text and new_port != self.current_port:
            if ':' in self.connection_str and '/' in self.connection_str:
                new_connection_str = self.connection_str.replace(self.current_port, str(new_port))
                config_data['server']['connection_str'] = new_connection_str
                settings_changed = True
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Connection String",
                    "The existing connection string format is invalid. Unable to update port."
                )
                return False

        if settings_changed:
            try:
                with config_file_path.open('w', encoding='utf-8') as file:
                    yaml.safe_dump(config_data, file)
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Saving Configuration",
                    f"An error occurred while saving the configuration: {e}"
                )
                return False

            if new_port_text:
                self.connection_str = config_data['server']['connection_str']
                self.current_port = str(new_port)
                self.widgets['port']['label'].setText(f"Port: {new_port}")

            self.widgets['port']['edit'].clear()

        return settings_changed