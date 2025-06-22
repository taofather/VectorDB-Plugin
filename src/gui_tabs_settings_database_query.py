import yaml
from PySide6.QtGui import QIntValidator, QDoubleValidator
from PySide6.QtWidgets import QWidget, QLabel, QLineEdit, QGridLayout, QSizePolicy, QComboBox, QPushButton, QMessageBox, QSpinBox, QCheckBox

from constants import TOOLTIPS
from config_manager import ConfigManager

class QuerySettingsTab(QWidget):
    def __init__(self):
        super(QuerySettingsTab, self).__init__()
        self.config_manager = ConfigManager()
        config_data = self.config_manager.get_config()
        
        self.database_config = config_data.get('database', {})
        self.created_databases = config_data.get('created_databases', {})
        self.database_to_search = self.database_config.get('database_to_search', '')
        
        grid_layout = QGridLayout()
        
        # Database selection and current setting
        self.database_label = QLabel("Database:")
        self.database_label.setToolTip(TOOLTIPS["DATABASE_SELECT"])
        grid_layout.addWidget(self.database_label, 0, 0)
        
        self.database_combo = QComboBox()
        self.database_combo.addItems(list(self.created_databases.keys()))
        self.database_combo.setToolTip(TOOLTIPS["DATABASE_SELECT"])
        if self.database_to_search in self.created_databases:
            self.database_combo.setCurrentIndex(list(self.created_databases.keys()).index(self.database_to_search))
        self.database_combo.setMinimumWidth(100)
        grid_layout.addWidget(self.database_combo, 0, 2)
        
        self.current_database_label = QLabel(f"{self.database_to_search}")
        self.current_database_label.setToolTip(TOOLTIPS["DATABASE_SELECT"])
        grid_layout.addWidget(self.current_database_label, 0, 1)
        
        # Number of results and current setting
        self.results_label = QLabel("Number of Results:")
        self.results_label.setToolTip(TOOLTIPS["CONTEXTS"])
        grid_layout.addWidget(self.results_label, 1, 0)
        
        self.results_spinbox = QSpinBox()
        self.results_spinbox.setRange(1, 100)
        self.results_spinbox.setValue(self.database_config.get('num_results', 3))
        self.results_spinbox.setToolTip(TOOLTIPS["CONTEXTS"])
        grid_layout.addWidget(self.results_spinbox, 1, 2)
        
        self.current_results_label = QLabel(f"{self.database_config.get('num_results', 3)}")
        self.current_results_label.setToolTip(TOOLTIPS["CONTEXTS"])
        grid_layout.addWidget(self.current_results_label, 1, 1)
        
        # Show thinking checkbox
        self.thinking_label = QLabel("Show Thinking:")
        self.thinking_label.setToolTip(TOOLTIPS["SHOW_THINKING_CHECKBOX"])
        grid_layout.addWidget(self.thinking_label, 2, 0)
        
        self.thinking_checkbox = QCheckBox()
        self.thinking_checkbox.setChecked(self.database_config.get('show_thinking', True))
        self.thinking_checkbox.setToolTip(TOOLTIPS["SHOW_THINKING_CHECKBOX"])
        grid_layout.addWidget(self.thinking_checkbox, 2, 2)
        
        self.setLayout(grid_layout)

    def update_config(self):
        try:
            config_data = self.config_manager.get_config()
            settings_changed = False
            
            # Update database selection
            new_database = self.database_combo.currentText()
            if new_database != self.database_to_search:
                config_data['database']['database_to_search'] = new_database
                self.database_to_search = new_database
                self.current_database_label.setText(f"{new_database}")
                settings_changed = True
            
            # Update number of results
            new_results = self.results_spinbox.value()
            if new_results != config_data['database'].get('num_results', 3):
                config_data['database']['num_results'] = new_results
                self.current_results_label.setText(f"{new_results}")
                settings_changed = True
            
            # Update show thinking
            new_thinking = self.thinking_checkbox.isChecked()
            if new_thinking != config_data['database'].get('show_thinking', True):
                config_data['database']['show_thinking'] = new_thinking
                settings_changed = True
            
            if settings_changed:
                self.config_manager.save_config(config_data)
            
            return settings_changed
            
        except Exception as e:
            print(f"Error updating config: {e}")
            return False

    def reset_search_term(self):
        try:
            with open('config.yaml', 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Configuration",
                f"An error occurred while loading the configuration: {e}"
            )
            return

        config_data['database']['search_term'] = ''

        try:
            with open('config.yaml', 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_data, f)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Saving Configuration",
                f"An error occurred while saving the configuration: {e}"
            )
            return

        self.search_term_label.setText("Search Term Filter: ")
        self.search_term_edit.clear()