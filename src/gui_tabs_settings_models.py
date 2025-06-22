from PySide6.QtWidgets import QWidget, QLabel, QComboBox, QGridLayout
import yaml
from constants import TOOLTIPS
from config_manager import ConfigManager

class ModelSettingsTab(QWidget):
    def __init__(self):
        super(ModelSettingsTab, self).__init__()
        self.config_manager = ConfigManager()
        config_data = self.config_manager.get_config()
        
        self.embedding_model_name = config_data.get('EMBEDDING_MODEL_NAME', '')
        self.vector_models = config_data.get('vector_models', {})
        
        grid_layout = QGridLayout()
        
        # Embedding model selection and current setting
        self.model_label = QLabel("Embedding Model:")
        self.model_label.setToolTip(TOOLTIPS["EMBEDDING_MODEL"])
        grid_layout.addWidget(self.model_label, 0, 0)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(self.vector_models.keys()))
        self.model_combo.setToolTip(TOOLTIPS["EMBEDDING_MODEL"])
        if self.embedding_model_name in self.vector_models:
            self.model_combo.setCurrentIndex(list(self.vector_models.keys()).index(self.embedding_model_name))
        self.model_combo.setMinimumWidth(100)
        grid_layout.addWidget(self.model_combo, 0, 2)
        
        self.current_model_label = QLabel(f"{self.embedding_model_name}")
        self.current_model_label.setToolTip(TOOLTIPS["EMBEDDING_MODEL"])
        grid_layout.addWidget(self.current_model_label, 0, 1)
        
        self.setLayout(grid_layout)

    def update_config(self):
        try:
            config_data = self.config_manager.get_config()
            
            new_model = self.model_combo.currentText()
            if new_model != self.embedding_model_name:
                config_data['EMBEDDING_MODEL_NAME'] = new_model
                self.embedding_model_name = new_model
                self.current_model_label.setText(f"{new_model}")
                self.config_manager.save_config(config_data)
                return True
            
            return False
            
        except Exception as e:
            print(f"Error updating config: {e}")
            return False
