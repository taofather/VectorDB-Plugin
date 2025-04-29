import time
import logging

import torch
from multiprocessing import Process, Pipe
from multiprocessing.connection import PipeConnection
from PySide6.QtCore import QObject, Signal

import module_chat
from database_interactions import QueryVectorDB
from utilities import format_citations, my_cprint, normalize_chat_text
from constants import rag_string

class MessageType:
    QUESTION = "question"
    RESPONSE = "response" 
    PARTIAL_RESPONSE = "partial_response"
    CITATIONS = "citations"
    ERROR = "error"
    FINISHED = "finished"
    EXIT = "exit"
    TOKEN_COUNTS = "token_counts"

class LocalModelSignals(QObject):
    response_signal = Signal(str)  # 7.
    citations_signal = Signal(str)  # 8.
    error_signal = Signal(str)  # 9.
    finished_signal = Signal()  # 10.
    model_loaded_signal = Signal()  # 3.
    model_unloaded_signal = Signal()  # 11.
    token_count_signal = Signal(str)

class LocalModelChat:
    def __init__(self):
        self.model_process = None
        self.model_pipe = None
        self.current_model = None
        self.signals = LocalModelSignals()

    def start_model_process(self, model_name):
        if self.current_model != model_name:
            if self.is_model_loaded():
                self.terminate_current_process()

            parent_conn, child_conn = Pipe()
            self.model_pipe = parent_conn
            self.model_process = Process(target=self._local_model_process, args=(child_conn, model_name), daemon=True)
            self.model_process.start()
            self.current_model = model_name
            self._start_listening_thread()
            # 3. signal that model is loaded
            self.signals.model_loaded_signal.emit()
        else:
            logging.warning(f"Model {model_name} is already loaded")

    def terminate_current_process(self):
        if self.model_process is not None:
            try:
                if self.model_pipe:
                    try:
                        self.model_pipe.send((MessageType.EXIT, None))
                    except (BrokenPipeError, OSError):
                        logging.warning("Pipe already closed")
                    finally:
                        self.model_pipe.close()
                        self.model_pipe = None
                
                process = self.model_process
                self.model_process = None
                
                if process.is_alive():
                    process.join(timeout=10)
                    if process.is_alive():
                        logging.warning("Process did not terminate, forcing termination")
                        process.terminate()
                        process.join(timeout=5)
            except Exception as e:
                logging.exception(f"Error during process termination: {e}")
        else:
            logging.warning("No process to terminate")

        self.model_pipe = None
        self.model_process = None
        self.current_model = None
        time.sleep(0.5)
        self.signals.model_unloaded_signal.emit()

    def start_chat(self, user_question, selected_model, selected_database):
        if not self.model_pipe:
            self.signals.error_signal.emit("Model not loaded. Please start a model first.")
            return

        self.model_pipe.send((MessageType.QUESTION, (user_question, selected_model, selected_database)))

    def is_model_loaded(self):
        return self.model_process is not None and self.model_process.is_alive()

    def eject_model(self):
        self.terminate_current_process()

    def _start_listening_thread(self):
        import threading
        self.listener_thread = threading.Thread(target=self._listen_for_response, daemon=True)
        self.listener_thread.start()

    def _listen_for_response(self):
        """
        Listens every second for messages coming through the pipe from the child process. When a message is received, the
        message type determines which signal is emitted.
        """
        while True:
            if not self.model_pipe or not isinstance(self.model_pipe, PipeConnection):
                break
            
            try:
                # checks every second for messages from "_local_model_process" that's being run in the child process
                if self.model_pipe.poll(timeout=1):
                    message_type, message = self.model_pipe.recv()
                    if message_type in [MessageType.RESPONSE, MessageType.PARTIAL_RESPONSE]:
                        self.signals.response_signal.emit(message)
                    elif message_type == MessageType.CITATIONS:
                        self.signals.citations_signal.emit(message)
                    elif message_type == MessageType.ERROR:
                        self.signals.error_signal.emit(message)
                    elif message_type == MessageType.FINISHED:
                        self.signals.finished_signal.emit()
                        if message == MessageType.EXIT:
                            break
                    elif message_type == MessageType.TOKEN_COUNTS:
                        self.signals.token_count_signal.emit(message)
                else:
                    time.sleep(0.1)
            except (BrokenPipeError, EOFError, OSError) as e:
                # inconsequential but i'll address later
                # logging.warning(f"Pipe communication error: {str(e)}")
                break
            except Exception as e:
                logging.warning(f"Unexpected error in _listen_for_response: {str(e)}")
                break

        self.cleanup_listener_resources()

    def cleanup_listener_resources(self):
        # Just clean up the resources without trying to join the thread
        self.model_pipe = None
        self.model_process = None
        self.current_model = None
        # Remove the self.listener_thread.join(timeout=1) line

    @staticmethod
    def _local_model_process(conn, model_name): # child process for local model's generation
        model_instance = module_chat.choose_model(model_name)
        query_vector_db = None
        current_database = None
        try:
            while True:
                try:
                    message_type, message = conn.recv()
                    if message_type == MessageType.QUESTION:
                        user_question, _, selected_database = message
                        if query_vector_db is None or current_database != selected_database:
                            query_vector_db = QueryVectorDB(selected_database)
                            current_database = selected_database
                        contexts, metadata_list = query_vector_db.search(user_question)
                        if not contexts:
                            conn.send((MessageType.ERROR, "No relevant contexts found."))
                            conn.send((MessageType.FINISHED, None))
                            continue
                        # exit early with message if contexts length comes within 100 of model's max context limit
                        max_context_tokens = model_instance.max_length - 100
                        context_tokens = len(model_instance.tokenizer.encode("\n\n---\n\n".join(contexts)))

                        if context_tokens > max_context_tokens:
                            logging.warning(f"Context tokens ({context_tokens}) exceed max context limit ({max_context_tokens})")
                            error_message = (
                                "The contexts received from the vector database exceed the chat model's context limit.\n\n"
                                "You can either:\n"
                                "1) Adjust the chunk size setting when creating the database;\n"
                                "2) Adjust the search settings (e.g. relevancy, number of contexts to return, etc.);\n"
                                "3) Choose a chat model with a larger context."
                            )
                            conn.send((MessageType.ERROR, error_message))
                            conn.send((MessageType.FINISHED, None))
                            continue

                        augmented_query = f"{rag_string}\n\n---\n\n" + "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + user_question
                        # DEBUG
                        # print(augmented_query)

                        # counts tokens using the chosen model's tokenizer
                        prepend_token_count = len(model_instance.tokenizer.encode(rag_string))
                        context_token_count = len(model_instance.tokenizer.encode("\n\n---\n\n".join(contexts)))
                        user_question_token_count = len(model_instance.tokenizer.encode(user_question))

                        full_response = ""
                        for partial_response in module_chat.generate_response(model_instance, augmented_query):
                            full_response += partial_response
                            conn.send((MessageType.PARTIAL_RESPONSE, partial_response))

                        response_token_count = len(model_instance.tokenizer.encode(full_response))
                        remaining_tokens = model_instance.max_length - (prepend_token_count + user_question_token_count + context_token_count + response_token_count)
                        total_tokens = prepend_token_count + context_token_count + user_question_token_count + response_token_count

                        token_count_string = (
                            f"<span style='color:#2ECC40;'>available tokens ({model_instance.max_length})</span>"
                            f"<span style='color:#FF4136;'> - rag instruction ({prepend_token_count})"
                            f" - query ({user_question_token_count})"
                            f" - contexts ({context_token_count})"
                            f" - response ({response_token_count})</span>"
                            f"<span style='color:white;'> = {remaining_tokens} remaining tokens.</span>"
                        )

                        conn.send((MessageType.TOKEN_COUNTS, token_count_string))

                        with open('chat_history.txt', 'w', encoding='utf-8') as f:
                            normalized_response = normalize_chat_text(full_response)
                            f.write(normalized_response)
                        citations = format_citations(metadata_list)
                        conn.send((MessageType.CITATIONS, citations))
                        conn.send((MessageType.FINISHED, None))
                    elif message_type == MessageType.EXIT:
                        break
                except EOFError:
                    logging.warning("Connection closed by main process.")
                    break
                except Exception as e:
                    logging.exception(f"Error in local_model_process: {e}")
                    conn.send((MessageType.ERROR, str(e)))
                    conn.send((MessageType.FINISHED, None))
        finally:
            try:
                if hasattr(model_instance, 'cleanup'):
                    model_instance.cleanup()
            finally:
                conn.close()
                my_cprint("Local chat model removed from memory.", "red")

def is_cuda_available():
    return torch.cuda.is_available()