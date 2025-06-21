import gc
import logging
import json
import requests
from pathlib import Path
import torch
import yaml
from openai import OpenAI
from PySide6.QtCore import QThread, Signal, QObject

from database_interactions import QueryVectorDB
from utilities import format_citations, normalize_chat_text, my_cprint
from constants import system_message, rag_string, THINKING_TAGS
from config_manager import ConfigManager

ROOT_DIRECTORY = Path(__file__).resolve().parent

contexts_output_file_path = ROOT_DIRECTORY / "contexts.txt"
metadata_output_file_path = ROOT_DIRECTORY / "metadata.txt"

class LMStudioSignals(QObject):
    response_signal = Signal(str)
    error_signal = Signal(str)
    finished_signal = Signal()
    citations_signal = Signal(str)

class LMStudioChat:
    def __init__(self):
        self.signals = LMStudioSignals()
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        self.server_config = self.config.get('server', {})
        self.host = self.server_config.get('host', 'localhost')
        self.port = self.server_config.get('port', 5000)
        self.show_thinking = self.server_config.get('show_thinking', False)
        self.query_vector_db = None

    def send_message(self, message, database_name):
        try:
            url = f"http://{self.host}:{self.port}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            data = {
                "messages": [{"role": "user", "content": message}],
                "temperature": 0.7,
                "max_tokens": -1
            }

            response = requests.post(url, headers=headers, json=data, stream=True)
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line.decode('utf-8').replace('data: ', ''))
                        if 'choices' in json_response and json_response['choices']:
                            content = json_response['choices'][0].get('delta', {}).get('content', '')
                            if content:
                                full_response += content
                                self.signals.response_signal.emit(content)
                    except json.JSONDecodeError:
                        continue

            self.signals.finished_signal.emit()

        except requests.exceptions.RequestException as e:
            error_message = f"Error connecting to LM Studio server: {str(e)}"
            my_cprint(error_message, "red")
            self.signals.error_signal.emit(error_message)
            self.signals.finished_signal.emit()

    def connect_to_local_model(self, augmented_query):
        """Generator that yields response chunks from LM Studio"""
        try:
            url = f"http://{self.host}:{self.port}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            data = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": augmented_query}
                ],
                "temperature": 0.7,
                "max_tokens": -1,
                "stream": True
            }

            response = requests.post(url, headers=headers, json=data, stream=True)
            response.raise_for_status()

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
                                yield content
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.RequestException as e:
            error_message = f"Error connecting to LM Studio server: {str(e)}"
            my_cprint(error_message, "red")
            raise Exception(error_message)

    def handle_response_and_cleanup(self, full_response, metadata_list):
        citations = format_citations(metadata_list)
        
        if self.query_vector_db:
            self.query_vector_db.cleanup()

        if torch.cuda.empty_cache():
            torch.cuda.empty_cache()
        gc.collect()
        
        return citations

    def save_metadata_to_file(self, metadata_list):
        with metadata_output_file_path.open('w', encoding='utf-8') as output_file:
            for metadata in metadata_list:
                output_file.write(f"{metadata}\n")

    def ask_local_chatgpt(self, query, selected_database):
        logging.info(f"Starting ask_local_chatgpt with query: '{query[:50]}...' and database: '{selected_database}'")
        
        if self.query_vector_db is None or self.query_vector_db.selected_database != selected_database:
            logging.info(f"Getting QueryVectorDB instance for database: {selected_database}")
            self.query_vector_db = QueryVectorDB.get_instance(selected_database)

        logging.info(f"About to search database...")
        contexts, metadata_list = self.query_vector_db.search(query)
        logging.info(f"Search returned {len(contexts)} contexts")

        self.save_metadata_to_file(metadata_list)

        if not contexts:
            logging.warning(f"No contexts found! contexts length: {len(contexts)}")
            self.signals.error_signal.emit("No relevant contexts found.")
            self.signals.finished_signal.emit()
            return
        else:
            logging.info(f"Found {len(contexts)} contexts, proceeding to LM Studio...")

            # DEBUG: Log the actual contexts content
            logging.info("=== CONTEXTS RETRIEVED ===")
            for i, context in enumerate(contexts):
                logging.info(f"Context {i+1}: {context[:200]}...")
            logging.info("=== END CONTEXTS ===")

            augmented_query = f"{rag_string}\n\n---\n\n" + "\n\n---\n\n".join(contexts) + f"\n\n-----\n\n{query}"
            
            # DEBUG: Log the complete prompt being sent
            logging.info("=== COMPLETE PROMPT ===")
            logging.info(f"System Message: {system_message}")
            logging.info(f"User Message: {augmented_query[:500]}...")
            logging.info("=== END PROMPT ===")
            
            full_response = ""
            response_generator = self.connect_to_local_model(augmented_query)
            for response_chunk in response_generator:
                self.signals.response_signal.emit(response_chunk)
                full_response += response_chunk

            with open('chat_history.txt', 'w', encoding='utf-8') as f:
                normalized_response = normalize_chat_text(full_response)
                f.write(normalized_response)

            self.signals.response_signal.emit("\n")

            citations = self.handle_response_and_cleanup(full_response, metadata_list)
            self.signals.citations_signal.emit(citations)
            self.signals.finished_signal.emit()

class LMStudioChatThread(QThread):
    def __init__(self, question, database_name):
        super().__init__()
        self.question = question
        self.database_name = database_name
        self.lm_studio_chat = LMStudioChat()

    def run(self):
        self.lm_studio_chat.ask_local_chatgpt(self.question, self.database_name)

def is_lm_studio_available():
    try:
        response = requests.get("http://127.0.0.1:1234/v1/models/", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

"""
[Main Process]
    |
    |         DatabaseQueryTab (GUI)                 LMStudioChatThread
    |         ------------------                     -----------------
    |              |                                     |
    |        [Submit Button]                             |
    |              |                                     |
    |         on_submit_button_clicked()                 |
    |              |                                     |
    |              |---> LMStudioChatThread.start() ---->|
    |              |                                     |
    |                                          [LMStudioChat Instance]
    |                                                    |
    |                                         ask_local_chatgpt()
    |                                                    |
    |                                         [QueryVectorDB Search]
    |                                                    |
    |                                      connect_to_local_chatgpt()
    |                                                    |
    |    Signal Flow                            OpenAI API Stream
    |    -----------                            ----------------
    |         |                                        |
    |    Signals Received:                     Stream Chunks:
    |    - response_signal                     - chunk.choices[0].delta.content
    |    - error_signal                                |
    |    - finished_signal                             |
    |    - citations_signal                             |
    |         |                                        |
    |    GUI Updates:                          Cleanup Operations:
    |    - update_response_lm_studio()         - handle_response_and_cleanup()
    |    - show_error_message()                - save_metadata_to_file()
    |    - on_submission_finished()            - torch.cuda.empty_cache()
    |    - display_citations_in_widget()       - gc.collect()
    |                                                  |
    |                                          Emit Final Signals:
    |                                          - citations_signal
    |                                          - finished_signal
"""