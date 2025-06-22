import yaml
from pathlib import Path
import torch
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QGridLayout, QVBoxLayout, QComboBox, QWidget, QSpinBox, QCheckBox
from constants import VISION_MODELS, TOOLTIPS
from config_manager import ConfigManager

def is_cuda_available():
    return torch.cuda.is_available()

def get_cuda_capability():
    if is_cuda_available():
        return torch.cuda.get_device_capability(0)
    return (0, 0)

class VisionSettingsTab(QWidget):
    def __init__(self):
        super(VisionSettingsTab, self).__init__()
        self.config_manager = ConfigManager()
        config_data = self.config_manager.get_config()
        
        self.vision_config = config_data.get('vision', {})
        self.compute_device_options = config_data.get('Compute_Device', {}).get('available', [])
        self.vision_device = config_data.get('Compute_Device', {}).get('vision', '')
        
        grid_layout = QGridLayout()
        
        # Device selection and current setting
        self.device_label = QLabel("Device:")
        self.device_label.setToolTip(TOOLTIPS["VISION_DEVICE"])
        grid_layout.addWidget(self.device_label, 0, 0)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(self.compute_device_options)
        self.device_combo.setToolTip(TOOLTIPS["VISION_DEVICE"])
        if self.vision_device in self.compute_device_options:
            self.device_combo.setCurrentIndex(self.compute_device_options.index(self.vision_device))
        self.device_combo.setMinimumWidth(100)
        grid_layout.addWidget(self.device_combo, 0, 2)
        
        self.current_device_label = QLabel(f"{self.vision_device}")
        self.current_device_label.setToolTip(TOOLTIPS["VISION_DEVICE"])
        grid_layout.addWidget(self.current_device_label, 0, 1)
        
        # Half precision checkbox
        self.half_precision_label = QLabel("Half-Precision (2x speedup - GPU only):")
        self.half_precision_label.setToolTip(TOOLTIPS["HALF_PRECISION"])
        grid_layout.addWidget(self.half_precision_label, 1, 0)
        
        self.half_precision_checkbox = QCheckBox()
        self.half_precision_checkbox.setChecked(self.vision_config.get('half', False))
        self.half_precision_checkbox.setToolTip(TOOLTIPS["HALF_PRECISION"])
        grid_layout.addWidget(self.half_precision_checkbox, 1, 2)
        
        self.setLayout(grid_layout)

    def update_config(self):
        try:
            config_data = self.config_manager.get_config()
            settings_changed = False
            
            # Update device selection
            new_device = self.device_combo.currentText()
            if new_device != self.vision_device:
                config_data['Compute_Device']['vision'] = new_device
                self.vision_device = new_device
                self.current_device_label.setText(f"{new_device}")
                settings_changed = True
            
            # Update half precision
            new_half_precision = self.half_precision_checkbox.isChecked()
            if new_half_precision != self.vision_config.get('half', False):
                config_data['vision']['half'] = new_half_precision
                settings_changed = True
            
            if settings_changed:
                self.config_manager.save_config(config_data)
            
            return settings_changed
            
        except Exception as e:
            print(f"Error updating config: {e}")
            return False

    def populate_model_combobox(self):
        cuda_available = is_cuda_available()
        cuda_capability = get_cuda_capability()
        available_models = []
        
        for model, info in VISION_MODELS.items():
            requires_cuda = info.get('requires_cuda', True)
            precision = info.get('precision')
            
            if cuda_available:
                if requires_cuda:
                    if precision == 'bfloat16':
                        if cuda_capability >= (8, 6):
                            available_models.append(model)
                    else:
                        available_models.append(model)
                else:
                    available_models.append(model)
            else:
                if not requires_cuda:
                    available_models.append(model)
        
        self.modelComboBox.addItems(available_models)

    def set_initial_model(self):
        config = self.read_config()
        saved_model = config.get('vision', {}).get('chosen_model')

        if saved_model and saved_model in [self.modelComboBox.itemText(i) for i in range(self.modelComboBox.count())]:
            index = self.modelComboBox.findText(saved_model)
            self.modelComboBox.setCurrentIndex(index)
        else:
            self.modelComboBox.setCurrentIndex(0)
        
        self.updateModelInfo()

    def updateModelInfo(self):
        chosen_model = self.modelComboBox.currentText()
        self.updateConfigFile('chosen_model', chosen_model)
        
        model_info = VISION_MODELS[chosen_model]
        self.sizeLabel.setText(model_info['size'])
        self.precisionLabel.setText(model_info['precision'])
        self.vramLabel.setText(model_info['vram'])
        self.quantLabel.setText(model_info['quant'])

    def read_config(self):
        config_file_path = Path('config.yaml')
        if config_file_path.exists():
            try:
                with open(config_file_path, 'r', encoding='utf-8') as file:
                    return yaml.safe_load(file)
            except Exception:
                pass
        return {}

    def updateConfigFile(self, key, value):
        current_config = self.read_config()
        vision_config = current_config.get('vision', {})
        if vision_config.get(key) != value:
            vision_config[key] = value
            current_config['vision'] = vision_config
            config_file_path = Path('config.yaml')
            with open(config_file_path, 'w', encoding='utf-8') as file:
                yaml.dump(current_config, file)