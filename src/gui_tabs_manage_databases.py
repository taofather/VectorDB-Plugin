import shutil
import sqlite3
from pathlib import Path
import psycopg2
import yaml
from PySide6.QtCore import Qt, QAbstractTableModel
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QTableView, QMenu,
    QGroupBox, QLabel, QComboBox, QMessageBox, QHeaderView
)

from utilities import open_file
from config_manager import ConfigManager
from database.factory import VectorDBFactory


class SQLiteTableModel(QAbstractTableModel):
    def __init__(self, data=None):
        super().__init__()
        self._data = data or []
        self._headers = ["File Name"]

    def data(self, index, role):
        if role == Qt.DisplayRole:
            return self._data[index.row()][0]
        elif role == Qt.ForegroundRole:
            return QColor('white')
        return None

    def rowCount(self, index):
        return len(self._data)

    def columnCount(self, index):
        return 1

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._headers[section]
        return None


class RefreshingComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.addItem("Select a database...")
        self.setItemData(0, QColor('gray'), Qt.ForegroundRole)
        self.setCurrentIndex(0)

    def showPopup(self):
        current_text = self.currentText()
        self.blockSignals(True)
        self.clear()
        self.addItem("Select a database...")
        self.setItemData(0, QColor('gray'), Qt.ForegroundRole)
        databases = self.parent().load_created_databases()
        self.addItems(databases)
        if current_text and current_text in databases:
            index = self.findText(current_text)
            if index >= 0:
                self.setCurrentIndex(index)
            else:
                self.setCurrentIndex(0)
        else:
            self.setCurrentIndex(0)
        self.blockSignals(False)
        super().showPopup()


class ManageDatabasesTab(QWidget):
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.created_databases = self.load_created_databases()

        self.layout = QVBoxLayout(self)

        self.documents_group_box = self.create_group_box_with_table_view("Files in Selected Database")
        self.layout.addWidget(self.documents_group_box)

        self.database_info_layout = QHBoxLayout()
        self.database_info_label = QLabel("No database selected.")
        self.database_info_label.setTextFormat(Qt.RichText)
        self.database_info_layout.addWidget(self.database_info_label)
        self.layout.addLayout(self.database_info_layout)

        self.buttons_layout = QHBoxLayout()
        self.pull_down_menu = RefreshingComboBox(self)
        self.pull_down_menu.activated.connect(self.update_table_view_and_info_label)
        self.buttons_layout.addWidget(self.pull_down_menu)
        self.create_buttons()
        self.layout.addLayout(self.buttons_layout)

        self.groups = {self.documents_group_box: 1}

    def load_created_databases(self):
        config_data = self.config_manager.get_config()
        databases = list(config_data.get('created_databases', {}).keys())
        return [db for db in databases if db != "user_manual"]

    def display_no_databases_message(self):
        self.documents_group_box.hide()
        self.database_info_label.setText("No database selected.")

    def create_group_box_with_table_view(self, title):
        group_box = QGroupBox(title)
        layout = QVBoxLayout()
        self.table_view = QTableView()
        self.model = SQLiteTableModel()
        self.table_view.setModel(self.model)
        self.table_view.setSelectionMode(QTableView.SingleSelection)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.doubleClicked.connect(self.on_double_click)
        self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(self.show_context_menu)

        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        layout.addWidget(self.table_view)
        group_box.setLayout(layout)
        return group_box

    def get_database_type(self):
        config_data = self.config_manager.get_config()
        return config_data.get('database', {}).get('type', 'tiledb')

    def get_pgvector_data(self, selected_database):
        config_data = self.config_manager.get_config()
        pg_config = config_data.get('postgresql', {})
        
        try:
            conn = psycopg2.connect(
                host=pg_config.get('host', 'localhost'),
                port=pg_config.get('port', 5432),
                user=pg_config.get('user', 'postgres'),
                password=pg_config.get('password', ''),
                database=pg_config.get('database', 'vectordb')
            )
            
            with conn.cursor() as cur:
                table_name = f"vectors_{selected_database}"
                cur.execute(f"""
                    SELECT DISTINCT 
                        metadata->>'file_name' as file_name,
                        metadata->>'file_path' as file_path
                    FROM {table_name}
                    WHERE metadata->>'file_name' IS NOT NULL
                """)
                data = cur.fetchall()
            conn.close()
            return data
        except Exception as e:
            QMessageBox.warning(self, "Database Error", f"An error occurred while accessing PostgreSQL: {e}")
            return []

    def get_tiledb_data(self, selected_database):
        db_path = Path(__file__).resolve().parent.parent / "Vector_DB" / selected_database / "metadata.db"
        if not db_path.exists():
            return []
            
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT file_name, file_path FROM document_metadata")
            data = cursor.fetchall()
            conn.close()
            return data
        except sqlite3.Error as e:
            QMessageBox.warning(self, "Database Error", f"An error occurred while accessing SQLite: {e}")
            return []

    def update_table_view_and_info_label(self, index):
        selected_database = self.pull_down_menu.currentText()
        if selected_database == "Select a database...":
            self.display_no_databases_message()
            return

        if selected_database:
            self.documents_group_box.show()
            
            # Get data based on database type
            db_type = self.get_database_type()
            if db_type == 'pgvector':
                data = self.get_pgvector_data(selected_database)
            else:
                data = self.get_tiledb_data(selected_database)

            self.model._data = [(row[0], row[1]) for row in data]
            self.model.layoutChanged.emit()

            config_data = self.config_manager.get_config()
            db_config = config_data.get('created_databases', {}).get(selected_database, {})
            model_path = db_config.get('model', '')
            model_name = Path(model_path).name
            chunk_size = db_config.get('chunk_size', '')
            chunk_overlap = db_config.get('chunk_overlap', '')
            info_text = (
                f'<span style="color: #4CAF50;"><b>Name:</b></span> "{selected_database}" '
                f'<span style="color: #888;">|</span> '
                f'<span style="color: #2196F3;"><b>Model:</b></span> "{model_name}" '
                f'<span style="color: #888;">|</span> '
                f'<span style="color: #FF9800;"><b>Chunk size/overlap:</b></span> {chunk_size} / {chunk_overlap}'
            )
            self.database_info_label.setText(info_text)
        else:
            self.display_no_databases_message()

    def on_double_click(self, index):
        selected_database = self.pull_down_menu.currentText()
        if selected_database and selected_database != "Select a database...":
            file_path = self.model._data[index.row()][1]
            if Path(file_path).exists():
                open_file(file_path)
            else:
                QMessageBox.warning(self, "Error", f"File not found at the specified path: {file_path}")
        else:
            QMessageBox.warning(self, "Error", "No database selected.")

    def create_buttons(self):
        self.delete_database_button = QPushButton("Delete Database")
        self.buttons_layout.addWidget(self.delete_database_button)
        self.delete_database_button.clicked.connect(self.delete_selected_database)

    def delete_selected_database(self):
        selected_database = self.pull_down_menu.currentText()
        if not selected_database or selected_database == "Select a database...":
            QMessageBox.warning(self, "Delete Database", "No database selected.")
            return

        reply = QMessageBox.question(
            self, 'Delete Database',
            "This cannot be undone.\nClick OK to proceed or Cancel to back out.",
            QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel
        )

        if reply == QMessageBox.Ok:
            self.model.beginResetModel()
            self.model._data = []
            self.model.endResetModel()

            # Get database type and create appropriate instance
            db_type = self.get_database_type()
            config = self.config_manager.get_config()
            db = VectorDBFactory.create_database(config)
            db.initialize(config)
            
            try:
                # Delete the database using the appropriate implementation
                db.delete_database(selected_database)
                self.pull_down_menu.setCurrentIndex(0)
                self.display_no_databases_message()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to delete database: {e}")
            finally:
                db.cleanup()

    def refresh_pull_down_menu(self):
        self.pull_down_menu.showPopup()
        self.pull_down_menu.hidePopup()

    def show_context_menu(self, position):
        selected_database = self.pull_down_menu.currentText()
        if selected_database and selected_database != "Select a database...":
            menu = QMenu()
            delete_action = QAction("Delete File", self)
            delete_action.triggered.connect(self.delete_selected_file)
            menu.addAction(delete_action)
            menu.exec_(self.table_view.viewport().mapToGlobal(position))

    def delete_selected_file(self):
        # Placeholder function for delete functionality
        pass
