# gui_tabs_settings_tts.py
import yaml
from pathlib import Path
from PySide6.QtWidgets import (
    QLabel, QComboBox, QWidget, QGridLayout, QMessageBox, QHBoxLayout, QSpinBox, QCheckBox
)

from constants import WHISPER_SPEECH_MODELS, TOOLTIPS
from config_manager import ConfigManager


class TTSSettingsTab(QWidget):
    """A single-row, space-saving TTS settings panel."""

    # 1. Describe every backend (easy to extend later)
    BACKENDS = {
        "bark": {
            "label": "Bark (GPU)",
            "extras": {
                "size": {
                    "label": "Model",
                    "options": ["normal", "small"],
                    "default": "small",
                },
                "speaker": {
                    "label": "Speaker",
                    "options": [
                        "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2",
                        "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
                        "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8",
                        "v2/en_speaker_9",
                    ],
                    "default": "v2/en_speaker_6",
                },
            },
        },
        "whisperspeech": {
            "label": "WhisperSpeech (GPU)",
            "extras": {
                "s2a": {
                    "label": "S2A Model",
                    "options": list(WHISPER_SPEECH_MODELS["s2a"].keys()),
                    "default": list(WHISPER_SPEECH_MODELS["s2a"].keys())[0],
                },
                "t2s": {
                    "label": "T2S Model",
                    "options": list(WHISPER_SPEECH_MODELS["t2s"].keys()),
                    "default": list(WHISPER_SPEECH_MODELS["t2s"].keys())[0],
                },
            },
        },
        "chattts": {
            "label": "ChatTTS (CPU/CPU)",
            "extras": {},
        },
        "chatterbox": {
            "label": "Chatterbox (CPU/GPU)",
            "extras": {},
        },
        "googletts": {
            "label": "Google TTS (CPU)",
            "extras": {},
        },
    }

    # 2. Qt-setup
    def __init__(self):
        super().__init__()
        self.widgets_for_backend: dict[str, dict[str, QWidget]] = {}
        self.config_manager = ConfigManager()
        config_data = self.config_manager.get_config()
        
        self.tts_config = config_data.get('tts', {})
        self.compute_device_options = config_data.get('Compute_Device', {}).get('available', [])
        self.tts_device = config_data.get('Compute_Device', {}).get('tts', '')
        
        self._build_ui()
        self._load_from_yaml()
        self._update_visible_extras()

    def _build_ui(self):
        layout = QGridLayout(self)

        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 1)

        # static backend picker (always column-1)
        layout.addWidget(QLabel("TTS Backend:"), 0, 0)
        self.backend_combo = QComboBox()
        for key, spec in self.BACKENDS.items():
            self.backend_combo.addItem(spec["label"], userData=key)
        layout.addWidget(self.backend_combo, 0, 1)

        # dynamic extras live in this container (always column-2)
        self._extras_box = QWidget()
        self._extras_layout = QHBoxLayout(self._extras_box)
        self._extras_layout.setContentsMargins(0, 0, 0, 0)
        self._extras_layout.setSpacing(10)
        layout.addWidget(self._extras_box, 0, 2)

        # build, but don't place, every backend's extra widgets
        self.widgets_for_backend: dict[str, dict[str, tuple[QLabel, QComboBox]]] = {}
        for key, spec in self.BACKENDS.items():
            wdict = {}
            for extra_key, meta in spec["extras"].items():
                lbl = QLabel(meta["label"])
                cmb = QComboBox()
                cmb.addItems(meta["options"])
                cmb.currentTextChanged.connect(self._save_to_yaml)
                wdict[extra_key] = (lbl, cmb)
            self.widgets_for_backend[key] = wdict

        self.backend_combo.currentIndexChanged.connect(self._update_visible_extras)

        # Device selection and current setting
        self.device_label = QLabel("Device:")
        self.device_label.setToolTip(TOOLTIPS["TTS_DEVICE"])
        layout.addWidget(self.device_label, 1, 0)
        
        self.device_combo = QComboBox()
        self.device_combo.addItems(self.compute_device_options)
        self.device_combo.setToolTip(TOOLTIPS["TTS_DEVICE"])
        if self.tts_device in self.compute_device_options:
            self.device_combo.setCurrentIndex(self.compute_device_options.index(self.tts_device))
        self.device_combo.setMinimumWidth(100)
        layout.addWidget(self.device_combo, 1, 1)
        
        self.current_device_label = QLabel(f"{self.tts_device}")
        self.current_device_label.setToolTip(TOOLTIPS["TTS_DEVICE"])
        layout.addWidget(self.current_device_label, 1, 2)
        
        # Half precision checkbox
        self.half_precision_label = QLabel("Half-Precision (2x speedup - GPU only):")
        self.half_precision_label.setToolTip(TOOLTIPS["HALF_PRECISION"])
        layout.addWidget(self.half_precision_label, 2, 0)
        
        self.half_precision_checkbox = QCheckBox()
        self.half_precision_checkbox.setChecked(self.tts_config.get('half', False))
        self.half_precision_checkbox.setToolTip(TOOLTIPS["HALF_PRECISION"])
        layout.addWidget(self.half_precision_checkbox, 2, 2)

    # 3. YAML helpers
    def _config_path(self) -> Path:
        return Path("config.yaml")

    def _load_from_yaml(self):
        """Read config.yaml and populate widgets."""
        config_data = self.config_manager.get_config()

        # ---------------- backend ---------------- #
        tts_cfg = config_data.get("tts", {})
        backend = tts_cfg.get("model", "whisperspeech")
        idx = self.backend_combo.findData(backend)
        self.backend_combo.setCurrentIndex(idx if idx != -1 else 0)

        # --------------- extras ------------------ #
        # Bark
        bark_cfg = config_data.get("bark", {})
        for (lbl, cmb) in self.widgets_for_backend["bark"].values():
            if cmb.objectName() == "size":
                cmb.setCurrentText(bark_cfg.get("size", "small"))
            else:  # speaker
                cmb.setCurrentText(bark_cfg.get("speaker", "v2/en_speaker_6"))

        # WhisperSpeech
        if tts_cfg.get("model") == "whisperspeech":
            self.widgets_for_backend["whisperspeech"]["s2a"][1].setCurrentText(
                self._find_key_by_value(
                    WHISPER_SPEECH_MODELS["s2a"], tts_cfg.get("s2a")
                )
            )
            self.widgets_for_backend["whisperspeech"]["t2s"][1].setCurrentText(
                self._find_key_by_value(
                    WHISPER_SPEECH_MODELS["t2s"], tts_cfg.get("t2s")
                )
            )

    def _save_to_yaml(self):
        config_data = self.config_manager.get_config()

        # ---------------- backend ---------------- #
        backend_key = self.backend_combo.currentData()
        tts_cfg = config_data.setdefault("tts", {})
        tts_cfg["model"] = backend_key

        # --------------- extras ------------------ #
        if backend_key == "bark":
            bark = config_data.setdefault("bark", {})
            bark["size"] = self.widgets_for_backend["bark"]["size"][1].currentText()
            bark["speaker"] = (
                self.widgets_for_backend["bark"]["speaker"][1].currentText()
            )
        elif backend_key == "whisperspeech":
            tts_cfg["s2a"] = WHISPER_SPEECH_MODELS["s2a"][
                self.widgets_for_backend["whisperspeech"]["s2a"][1].currentText()
            ][0]
            tts_cfg["t2s"] = WHISPER_SPEECH_MODELS["t2s"][
                self.widgets_for_backend["whisperspeech"]["t2s"][1].currentText()
            ][0]

        self.config_manager.save_config(config_data)

    # ------------------------------------------------------------------ #
    # 4. Helpers
    # ------------------------------------------------------------------ #
    def _update_visible_extras(self):
        # clear existing widgets from the extras container
        while self._extras_layout.count():
            item = self._extras_layout.takeAt(0)
            if (w := item.widget()):
                w.setParent(None)

        # insert the widgets that correspond to the selected backend
        chosen = self.backend_combo.currentData()
        for lbl, cmb in self.widgets_for_backend[chosen].values():
            self._extras_layout.addWidget(lbl)
            self._extras_layout.addWidget(cmb)
            lbl.show()
            cmb.show()

        self._save_to_yaml()

    @staticmethod
    def _find_key_by_value(d: dict, value: str | None):
        """Reverse lookup with fallback to first key."""
        for k, v in d.items():
            if v[0] == value:
                return k
        return next(iter(d))

    def update_config(self):
        try:
            config_data = self.config_manager.get_config()
            settings_changed = False
            
            # Update device selection
            new_device = self.device_combo.currentText()
            if new_device != self.tts_device:
                config_data['Compute_Device']['tts'] = new_device
                self.tts_device = new_device
                self.current_device_label.setText(f"{new_device}")
                settings_changed = True
            
            # Update half precision
            new_half_precision = self.half_precision_checkbox.isChecked()
            if new_half_precision != self.tts_config.get('half', False):
                config_data['tts']['half'] = new_half_precision
                settings_changed = True
            
            if settings_changed:
                self.config_manager.save_config(config_data)
            
            return settings_changed
            
        except Exception as e:
            print(f"Error updating config: {e}")
            return False
