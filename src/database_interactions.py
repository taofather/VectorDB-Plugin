# database_interactions.py

import gc
import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional
import threading
import re
import sqlite3
import torch
import yaml
import concurrent.futures
import queue
from collections import defaultdict, deque
import shutil
import random
import sys
import traceback

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import TileDB

from document_processor import load_documents, split_documents
from module_process_images import choose_image_loader
from utilities import my_cprint, get_model_native_precision, get_appropriate_dtype, supports_flash_attention
from constants import VECTOR_MODELS

logging.basicConfig(level=logging.CRITICAL, force=True)
# logging.basicConfig(level=logging.DEBUG, force=True)
logger = logging.getLogger(__name__)


# DEBUG - implement later to potentially see the size of objects
def get_memory_usage(obj, name):
    """Helper function to get memory usage of an object"""
    try:
        size_bytes = sys.getsizeof(obj)
        if hasattr(obj, '__len__'):
            if len(obj) > 0:
                item_size = sys.getsizeof(obj[0]) if len(obj) > 0 else 0
                total_size = size_bytes + (item_size * len(obj))
                return f"{name}: {total_size / (1024**2):.2f} MB ({len(obj)} items)"
        return f"{name}: {size_bytes / (1024**2):.2f} MB"
    except:
        return f"{name}: Unable to calculate size"

class BaseEmbeddingModel:
    def __init__(self, model_name, model_kwargs, encode_kwargs, is_query=False):
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.encode_kwargs = encode_kwargs
        self.is_query = is_query

    def prepare_kwargs(self):
        ready = deepcopy(self.model_kwargs)

        # 1) update model_kwargs
        ready.setdefault("model_kwargs", {}).setdefault("trust_remote_code", True)

        # 2) update tokenizer_kwargs
        tok_kw = ready.setdefault("tokenizer_kwargs", {})
        tok_kw.setdefault("trust_remote_code", True)
        tok_kw.setdefault("padding", True)
        tok_kw.setdefault("truncation", True)
        tok_kw.setdefault("return_token_type_ids", False)
        tok_kw.setdefault("use_fast", True)
        tok_kw.setdefault("max_length", 512)

        return ready

    def prepare_encode_kwargs(self):
        if self.is_query:
            self.encode_kwargs['batch_size'] = 1
        return self.encode_kwargs

    def create(self):
        prepared_kwargs = self.prepare_kwargs()
        prepared_encode_kwargs = self.prepare_encode_kwargs()

        hf = HuggingFaceEmbeddings(
            model_name=self.model_name,
            show_progress=not self.is_query,
            model_kwargs=prepared_kwargs,
            encode_kwargs=prepared_encode_kwargs
        )

        tok = hf._client.tokenizer
        if "token_type_ids" in tok.model_input_names:
            tok.model_input_names.remove("token_type_ids")

        return hf


class SnowflakeEmbedding(BaseEmbeddingModel):
    def prepare_kwargs(self):
        # 1) inherit all kwargs from the base class
        snow_kwargs = super().prepare_kwargs()

        # 2) update tokenizer_kwargs for large model
        if "large" in self.model_name.lower():
            tok_kw = snow_kwargs.setdefault("tokenizer_kwargs", {})
            tok_kw.update({"max_length": 8192})

            return snow_kwargs

        # 1) determine if xformers can be used
        compute_device = snow_kwargs.get("device", "").lower()
        is_cuda        = compute_device.startswith("cuda")
        use_xformers   = is_cuda and supports_flash_attention()

        # 2) update tokenizer_kwargs for medium model
        tok_kw = snow_kwargs.setdefault("tokenizer_kwargs", {})
        tok_kw.update({"max_length": 8192})

        # 3) update config_kwargs for medium model
        snow_kwargs["config_kwargs"] = {
            "use_memory_efficient_attention": use_xformers,
            "unpad_inputs": use_xformers,
            "attn_implementation": "eager" if use_xformers else "sdpa",
        }

        return snow_kwargs


# deprecated
class StellaEmbedding(BaseEmbeddingModel):
    def prepare_kwargs(self):
        # 1) inherit all kwargs from the base class
        stella_kwargs = super().prepare_kwargs()

        # 2) update tokenizer_kwargs
        tok_kw = stella_kwargs.setdefault("tokenizer_kwargs", {})
        tok_kw.update({
            "max_length": 512,
        })

        return stella_kwargs

    def prepare_encode_kwargs(self):
        encode_kwargs = super().prepare_encode_kwargs()
        # 1) add the appropriate prompt_name if a query is being embedded
        if self.is_query:
            encode_kwargs["prompt_name"] = "s2p_query"

        return encode_kwargs


# deprecated
class Stella400MEmbedding(BaseEmbeddingModel):
    def prepare_kwargs(self):
        # 1) inherit all kwargs from the base class
        stella_kwargs = super().prepare_kwargs()

        # 2) determine if xformers can be used
        compute_device = stella_kwargs.get("device", "")
        is_cuda = compute_device.lower().startswith("cuda")
        use_xformers = is_cuda and supports_flash_attention()

        # 3) update tokenizer_kwargs
        tok_kw = stella_kwargs.setdefault("tokenizer_kwargs", {})
        tok_kw.update({
            "max_length": 512,
        })

        # 4) update config_kwargs
        stella_kwargs["config_kwargs"] = {
            "use_memory_efficient_attention": use_xformers,
            "unpad_inputs": use_xformers,
            "attn_implementation": "eager",   # always "eager" even when not using xformers
        }

        return stella_kwargs

    def prepare_encode_kwargs(self):
        encode_kwargs = super().prepare_encode_kwargs()
        # 1) add the appropriate prompt_name if a query is being embedded
        if self.is_query:
            encode_kwargs["prompt_name"] = "s2p_query"

        return encode_kwargs


# deprecated, only needed for deprecated gte-large-en-v1.5 and gte-base-en-v1.5 models
class AlibabaEmbedding(BaseEmbeddingModel):
    def prepare_kwargs(self):
        # 1) inherit all kwargs from the base class
        ali_kwargs = super().prepare_kwargs()

        # 2) determine if xformers can be used
        compute_device = ali_kwargs.get("device", "").lower()
        is_cuda = compute_device.startswith("cuda")
        use_xformers = is_cuda and supports_flash_attention()

        # 3) update tokenizer_kwargs
        tok_kw = ali_kwargs.setdefault("tokenizer_kwargs", {})
        tok_kw.update({
            "max_length": 8192,
        })

        # 4) update config_kwargs
        ali_kwargs["config_kwargs"] = {
            "use_memory_efficient_attention": use_xformers,
            "unpad_inputs": use_xformers,
            "attn_implementation": "eager" if use_xformers else "sdpa",
        }

        return ali_kwargs


class BgeCodeEmbedding(BaseEmbeddingModel):
    def prepare_kwargs(self):
        # 1) inherit all kwargs from the base class
        bge_kwargs = super().prepare_kwargs()

        # 2) update tokenizer_kwargs
        tok_kw = bge_kwargs.setdefault("tokenizer_kwargs", {})
        tok_kw.update({
            "max_length": 4096,
        })

        return bge_kwargs


class InflyAndAlibabaEmbedding(BaseEmbeddingModel):
    def prepare_kwargs(self):
        # 1) inherit all kwargs from the base class
        infly_kwargs = super().prepare_kwargs()

        # 2) update tokenizer_kwargs
        tok_kw = infly_kwargs.setdefault("tokenizer_kwargs", {})
        tok_kw.update({
            "max_length": 8192,
        })

        return infly_kwargs


class QwenEmbedding(BaseEmbeddingModel):
    def prepare_kwargs(self):
        q_kwargs = super().prepare_kwargs()

        # 1) update model_kwargs
        q_kwargs.setdefault("model_kwargs", {}).update({
            "attn_implementation": "sdpa",
        })

        # 2) update tokenizer_kwargs
        q_kwargs["tokenizer_kwargs"] = {
            **q_kwargs.get("tokenizer_kwargs", {}),
            "padding_side": "left",
            "max_length": 8192,
        }

        return q_kwargs


def create_vector_db_in_process(database_name):
    try:
        # Load configuration
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # Create database instance
        from database.factory import VectorDBFactory
        db = VectorDBFactory.create_database(config)
        
        # Initialize database with configuration
        db.initialize(config)
        
        # Create vector database
        create_vector_db = CreateVectorDB(database_name=database_name, db_instance=db)
        create_vector_db.run()
    except Exception as e:
        logging.error(f"Error creating vector database: {str(e)}")
        raise

def process_chunks_only_query(database_name, query, result_queue):
    try:
        query_db = QueryVectorDB.get_instance(database_name)
        contexts, metadata_list = query_db.search(query)

        formatted_contexts = []
        for index, (context, metadata) in enumerate(zip(contexts, metadata_list), start=1):
            file_name = metadata.get('file_name', 'Unknown')
            cleaned_context = re.sub(r'\n[ \t]+\n', '\n\n', context)
            cleaned_context = re.sub(r'\n\s*\n\s*\n*', '\n\n', cleaned_context.strip())
            formatted_context = (
                f"{'-'*80}\n"
                f"CONTEXT {index} | {file_name}\n"
                f"{'-'*80}\n"
                f"{cleaned_context}\n"
            )
            formatted_contexts.append(formatted_context)

        result_queue.put("\n".join(formatted_contexts))
    except Exception as e:
        result_queue.put(f"Error querying database: {str(e)}")
    finally:
        if 'query_db' in locals():
            query_db.cleanup()


class CreateVectorDB:
    def __init__(self, database_name: str, db_instance=None):
        self.database_name = database_name
        self.root_dir = Path(__file__).resolve().parent.parent  # Point to project root, not src/
        self.config = self.load_config(self.root_dir)
        self.vector_model = None
        self.db_instance = db_instance

    def load_config(self, root_directory):
        with open(root_directory / "config.yaml", 'r', encoding='utf-8') as stream:
            return yaml.safe_load(stream)

    def load_configuration(self):
        config_path = Path(__file__).resolve().parent.parent / 'config.yaml'
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise

    @torch.inference_mode()
    def initialize_vector_model(self, embedding_model_name, config_data):
        compute_device = config_data['Compute_Device']['database_creation']
        use_half = config_data.get("database", {}).get("half", False)
        model_native_precision = get_model_native_precision(embedding_model_name, VECTOR_MODELS)
        torch_dtype = get_appropriate_dtype(compute_device, use_half, model_native_precision)

        model_kwargs = {
            "device": compute_device, 
            "trust_remote_code": True,
            "similarity_fn_name": "euclidean", # (str, optional); "cosine" (default), "dot", "euclidean", "manhattan"
            "model_kwargs": {
                "torch_dtype": torch_dtype if torch_dtype is not None else None
            }
        }

        encode_kwargs = {'batch_size': 8}

        if "bge" in embedding_model_name.lower():
            # bge prompt
            encode_kwargs["prompt"] = ("Represent this sentence for searching relevant passages: ")

        model = BaseEmbeddingModel(embedding_model_name, model_kwargs, encode_kwargs).create()

        # Get model precision
        try:
            # Access dtype through the underlying client model
            if hasattr(model, 'client') and hasattr(model.client, 'dtype'):
                model_dtype = model.client.dtype
            elif hasattr(model, '_client') and hasattr(model._client, 'dtype'):
                model_dtype = model._client.dtype
            elif hasattr(model, 'model') and hasattr(model.model, 'dtype'):
                model_dtype = model.model.dtype
            else:
                # Fallback to a default precision if we can't determine dtype
                model_dtype = torch.float32
                
            if model_dtype == torch.float32:
                precision = "fp32"
            elif model_dtype == torch.float16:
                precision = "fp16"
            elif model_dtype == torch.bfloat16:
                precision = "bf16"
            else:
                precision = str(model_dtype)
        except Exception as e:
            # If we can't determine precision, use a default
            precision = "unknown"

        model_name = os.path.basename(embedding_model_name)
        my_cprint(f"{model_name} ({precision}) loaded using a batch size of {encode_kwargs['batch_size']}.", "green")

        return model, encode_kwargs

    @torch.inference_mode()
    def create_database(self, texts, embeddings):
        my_cprint("\nComputing vectors...", "yellow")
        start_time = time.time()

        hash_id_mappings = []
        MAX_UINT64 = 18446744073709551615

        try:
            all_texts = []
            all_metadatas = []
            all_ids = []
            chunk_counters = defaultdict(int)

            # Process all texts and generate IDs
            for doc in texts:
                file_hash = doc.metadata.get('hash')
                chunk_counters[file_hash] += 1
                tiledb_id = str(random.randint(0, MAX_UINT64 - 1))

                # ensure page_content is a clean string
                if hasattr(doc, 'page_content'):
                    if doc.page_content is None:
                        text_str = ""
                    elif isinstance(doc.page_content, str):
                        text_str = doc.page_content.strip()
                    elif isinstance(doc.page_content, (list, tuple)):
                        text_str = "\n".join(str(item) for item in doc.page_content).strip()
                    elif isinstance(doc.page_content, bytes):
                        try:
                            text_str = doc.page_content.decode('utf-8', errors='ignore').strip()
                        except Exception:
                            text_str = str(doc.page_content).strip()
                    else:
                        text_str = str(doc.page_content).strip()
                else:
                    text_str = str(doc).strip()

                if not text_str:
                    continue

                if not isinstance(text_str, str):
                    logging.error(f"Failed to convert to string: {type(text_str)} - {str(text_str)[:100]}")
                    continue

                all_texts.append(text_str)
                all_metadatas.append(doc.metadata)
                all_ids.append(tiledb_id)
                hash_id_mappings.append((tiledb_id, file_hash))

            # Debug check – ensure no non-strings slipped through
            bad_chunks = [
                (idx, type(txt), str(txt)[:60])
                for idx, txt in enumerate(all_texts)
                if not isinstance(txt, str)
            ]

            if bad_chunks:
                logging.error("Found non-string chunks:")
                for idx, type_, preview in bad_chunks:
                    logging.error(f"  {idx}: {type_} - {preview}")
                raise ValueError("Found non-string chunks in text data")

            # Compute embeddings
            embedding_start_time = time.time()
            my_cprint("Computing embeddings...", "yellow")

            vectors = embeddings.embed_documents(all_texts)
            embedding_end_time = time.time()
            embedding_elapsed = embedding_end_time - embedding_start_time
            my_cprint(f"Embedding computation completed in {embedding_elapsed:.2f} seconds.", "cyan")

            # Create the database using the appropriate implementation
            self.db_instance.create_database(
                database_name=self.database_name,
                texts=all_texts,
                embeddings=vectors,
                metadatas=all_metadatas,
                ids=all_ids
            )

            my_cprint("Processed all chunks", "yellow")

            end_time = time.time()
            elapsed_time = end_time - start_time
            my_cprint(f"Database created. Elapsed time: {elapsed_time:.2f} seconds.", "green")

            return hash_id_mappings

        except Exception as e:
            # show full traceback from child process
            traceback.print_exc()
            logging.error(f"Error creating database: {str(e)}")
            raise

    def create_metadata_db(self, documents, hash_id_mappings):
        if not (self.root_dir / "Vector_DB" / self.database_name).exists():
            (self.root_dir / "Vector_DB" / self.database_name).mkdir(parents=True, exist_ok=True)

        sqlite_db_path = self.root_dir / "Vector_DB" / self.database_name / "metadata.db"
        conn = sqlite3.connect(sqlite_db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT,
                hash TEXT,
                file_path TEXT,
                page_content TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hash_chunk_ids (
                tiledb_id TEXT PRIMARY KEY,
                hash TEXT
            )
        ''')

        try:
            # Prepare batch data for documents
            doc_rows = [
                (
                    doc.metadata.get("file_name", ""),
                    doc.metadata.get("hash", ""),
                    doc.metadata.get("file_path", ""),
                    doc.page_content
                )
                for doc in documents
            ]
            cursor.executemany('''
                INSERT INTO document_metadata (file_name, hash, file_path, page_content)
                VALUES (?, ?, ?, ?)
            ''', doc_rows)

            # Batch insert hash–ID mappings
            cursor.executemany('''
                INSERT INTO hash_chunk_ids (tiledb_id, hash)
                VALUES (?, ?)
            ''', hash_id_mappings)

            conn.commit()
        finally:
            conn.close()

    def load_audio_documents(self, source_dir: Path = None) -> list:
        if source_dir is None:
            source_dir = self.root_dir / "Docs_for_DB"
        json_paths = [f for f in source_dir.iterdir() if f.suffix.lower() == '.json']
        docs = []

        for json_path in json_paths:
            try:
                with open(json_path, 'r', encoding='utf-8') as json_file:
                    json_str = json_file.read()
                    doc = Document.parse_raw(json_str)
                    docs.append(doc)
            except Exception as e:
                my_cprint(f"Error loading {json_path}: {e}", "red")

        return docs
    
    def clear_docs_for_db_folder(self):
        docs_dir = self.root_dir / "Docs_for_DB"
        if docs_dir.exists():
            for item in docs_dir.iterdir():
                if item.is_file() or item.is_symlink():
                    try:
                        item.unlink()
                    except Exception as e:
                        print(f"Failed to delete {item}: {e}")

    @torch.inference_mode()
    def run(self):
        config_data = self.load_config(self.root_dir)
        EMBEDDING_MODEL_NAME = config_data.get("EMBEDDING_MODEL_NAME")
        db_type = config_data.get('database', {}).get('type', 'tiledb')

        # list to hold document objects
        documents = []

        # load text document objects
        text_documents = load_documents(self.root_dir / "Docs_for_DB")
        if isinstance(text_documents, list) and text_documents:
            documents.extend(text_documents)

        # separate lists for pdf and non-pdf document objects
        text_documents_pdf = [doc for doc in documents if doc.metadata.get("file_type") == ".pdf"]
        documents = [doc for doc in documents if doc.metadata.get("file_type") != ".pdf"]

        # load image descriptions
        print("Loading any images...")
        image_documents = choose_image_loader()
        if isinstance(image_documents, list) and image_documents:
            if len(image_documents) > 0:
                documents.extend(image_documents)

        # load audio transcriptions
        print("Loading any audio transcripts...")
        audio_documents = self.load_audio_documents()
        if isinstance(audio_documents, list) and audio_documents:
            documents.extend(audio_documents)

        # create a list to save pre-split text for sqliteDB
        json_docs_to_save = []
        json_docs_to_save.extend(documents)
        json_docs_to_save.extend(text_documents_pdf)

        # blank list to hold all split document objects
        texts = []

        # split document objects and add to list
        if (isinstance(documents, list) and documents) or (isinstance(text_documents_pdf, list) and text_documents_pdf):
            texts = split_documents(documents, text_documents_pdf)
            print(f"Documents split into {len(texts)} chunks.")

        del documents, text_documents_pdf
        gc.collect()

        # create db
        if isinstance(texts, list) and texts:
            embeddings, encode_kwargs = self.initialize_vector_model(EMBEDDING_MODEL_NAME, config_data)

            # Get hash->ID mappings and create the vector database
            hash_id_mappings = self.create_database(texts, embeddings)

            del texts
            gc.collect()

            # For pgvector, we don't need to create a metadata db since metadata is stored in the table
            if db_type != 'pgvector':
                # Pass mappings to metadata db creation
                self.create_metadata_db(json_docs_to_save, hash_id_mappings)
            
            del json_docs_to_save
            gc.collect()
            self.clear_docs_for_db_folder()


class QueryVectorDB:
    _instance = None
    _instance_lock = threading.Lock()
    _initialized = False

    def __new__(cls, *args, **kwargs):
        raise RuntimeError("Use QueryVectorDB.get_instance() instead of direct instantiation")

    def _init_once(self, selected_database):
        if not self._initialized:
            self.config = self.load_configuration()

            # upfront validation of the request
            if not selected_database:
                raise ValueError("No vector database selected.")
            if selected_database not in self.config["created_databases"]:
                raise ValueError(f'Database "{selected_database}" not found in config.')
            db_path = Path(__file__).resolve().parent.parent / "Vector_DB" / selected_database
            if not db_path.exists():
                raise FileNotFoundError(f'Database folder "{selected_database}" is missing on disk.')

            self.selected_database = selected_database
            self.embeddings = None
            self.db = None
            self.model_name = None
            self._debug_id = id(self)
            self._initialized = True
            logging.debug(f"Created new QueryVectorDB instance {self._debug_id} for database {selected_database}")

    @classmethod
    def get_instance(cls, selected_database):
        with cls._instance_lock:
            if cls._instance is not None:
                if cls._instance.selected_database != selected_database:
                    print(f"Database changed from {cls._instance.selected_database} to {selected_database}")
                    cls._instance.cleanup()
                    cls._instance = None
                else:
                    logging.debug(f"Reusing existing instance {cls._instance._debug_id} for database {selected_database}")

            if cls._instance is None:
                cls._instance = object.__new__(cls)
                cls._instance._init_once(selected_database)

            return cls._instance

    def load_configuration(self):
        config_path = Path(__file__).resolve().parent.parent / 'config.yaml'
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise

    @torch.inference_mode()
    def initialize_vector_model(self):
        model_path   = self.config['created_databases'][self.selected_database]['model']
        self.model_name = os.path.basename(model_path)
        compute_device  = self.config['Compute_Device']['database_query']

        # outer kwargs passed to SentenceTransformer
        model_kwargs = {
            "device": compute_device,
            "trust_remote_code": True,
            "similarity_fn_name": "euclidean", # (str, optional); "cosine" (default), "dot", "euclidean", "manhattan"
            "model_kwargs": {
                "trust_remote_code": True,
            },
            "tokenizer_kwargs": {
                "use_fast": True,
                "trust_remote_code": True,
            },
        }

        encode_kwargs = {"batch_size": 1}

        mp_lower = model_path.lower()

        if "qwen3-embedding" in mp_lower:
            encode_kwargs["prompt"] = (
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: <query>"
            )

            embeddings = QwenEmbedding(model_path, model_kwargs, encode_kwargs, is_query=True).create()
        elif "snowflake" in mp_lower or "intfloat" in mp_lower:
            # snowflake and intfloat prompt
            encode_kwargs["prompt"] = "query: "
            if "snowflake" in mp_lower:
                embeddings = SnowflakeEmbedding(model_path, model_kwargs, encode_kwargs, is_query=True).create()
            else:
                embeddings = BaseEmbeddingModel(model_path, model_kwargs, encode_kwargs, is_query=True).create()

        elif "alibaba" in mp_lower or "infly" in mp_lower:
            # alibaba and infly prompt
            encode_kwargs["prompt"] = (
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
                "Query: <query>"
            )
            embeddings = InflyAndAlibabaEmbedding(model_path, model_kwargs, encode_kwargs, is_query=True).create()

        # elif "bge-code" in mp_lower:
            # #bge-code prompt, highly customizable
            # code_instruction = "Given a question in text, retrieve relevant code that is relevant."
            # encode_kwargs["prompt"] = f"<instruct>{code_instruction}\n<query>"
            # embeddings = BgeCodeEmbedding(model_path, model_kwargs, encode_kwargs, is_query=True).create()

        elif "stella" in mp_lower:
            # stella prompt
            encode_kwargs["prompt"] = ("Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ")
            if "400m" in mp_lower:
                embeddings = Stella400MEmbedding(model_path, model_kwargs, encode_kwargs, is_query=True).create()
            else:
                embeddings = StellaEmbedding(model_path, model_kwargs, encode_kwargs, is_query=True).create()

        else:
            if "bge" in mp_lower:
                # bge prompt
                encode_kwargs["prompt"] = ("Represent this sentence for searching relevant passages: ")
            embeddings = BaseEmbeddingModel(model_path, model_kwargs, encode_kwargs, is_query=True).create()

        return embeddings

    def initialize_database(self):
        config_data = self.config
        db_type = config_data.get('database', {}).get('type', 'tiledb')
        
        if db_type == 'pgvector':
            from database.factory import VectorDBFactory
            db = VectorDBFactory.create_database(config_data)
            db.initialize(config_data)
            return db
        else:  # tiledb
            persist_directory = Path(__file__).resolve().parent.parent / "Vector_DB" / self.selected_database
            return TileDB.load(index_uri=str(persist_directory), embedding=self.embeddings, allow_dangerous_deserialization=True)

    @torch.inference_mode()
    def search(self, query, k: Optional[int] = None, score_threshold: Optional[float] = None):
        if not self.embeddings:
            logging.info(f"Initializing embedding model for database {self.selected_database}")
            self.embeddings = self.initialize_vector_model()

        if not self.db:
            logging.info(f"Initializing database connection for {self.selected_database}")
            self.db = self.initialize_database()

        # Set default values if not provided
        if k is None:
            k = 5
        if score_threshold is None:
            score_threshold = 0.8

        # Handle different database types
        db_type = self.config.get('database', {}).get('type', 'tiledb')
        
        if db_type == 'pgvector':
            # For pgvector, we need to set the embeddings
            self.db.embeddings = self.embeddings
            results = self.db.search(query, k=k, score_threshold=score_threshold)
            
            # Convert results to the expected format
            contexts = []
            metadata_list = []
            for result in results:
                contexts.append(result['text'])
                metadata_list.append(result['metadata'])
            
            return contexts, metadata_list
        else:  # tiledb
            relevant_contexts = self.db.similarity_search_with_score(
                query,
                k=k,
                score_threshold=score_threshold
            )
            
            contexts = [document.page_content for document, _ in relevant_contexts]
            metadata_list = [document.metadata for document, _ in relevant_contexts]
            scores = [score for _, score in relevant_contexts]
            
            for metadata, score in zip(metadata_list, scores):
                metadata['similarity_score'] = score
            
            return contexts, metadata_list

    def cleanup(self):
        logging.info(f"Cleaning up QueryVectorDB instance {self._debug_id} for database {self.selected_database}")

        if self.embeddings:
            logging.debug(f"Unloading embedding model for database {self.selected_database}")
            del self.embeddings
            self.embeddings = None

        if self.db:
            logging.debug(f"Closing database connection for {self.selected_database}")
            del self.db
            self.db = None

        if torch.cuda.is_available():
            logging.debug("Clearing CUDA cache")
            torch.cuda.empty_cache()

        gc.collect()
        logging.debug(f"Cleanup completed for instance {self._debug_id}")

"""
╔══════════════════════════════════════════════════════════════════════════╗
║  DEVELOPMENT NOTES – xFormers flags, attention-impl, and tokenization    ║
╚══════════════════════════════════════════════════════════════════════════╝

────────────────────────────────────────────────────────────────────────────
1.  Which models can use xFormers memory-efficient attention?
────────────────────────────────────────────────────────────────────────────
• Snowflake-GTE family   (all sizes except the "-large" variants)
• Alibaba-GTE family
• Stella-400 M           (v5)

Stella-1.5 B **cannot** use xFormers kernels at the time of writing.

────────────────────────────────────────────────────────────────────────────
2.  Snowflake-GTE & Alibaba-GTE  (shared behaviour)
────────────────────────────────────────────────────────────────────────────
✔  Flags belong in ✧config_kwargs✧ (which LangChain forwards to AutoConfig):

    {
        "config_kwargs": {
            "use_memory_efficient_attention":  <bool>,   # enable xFormers
            "unpad_inputs":                    <bool>,   # strip padding tokens
            "attn_implementation":             "eager"   # MUST be "eager"
        }
    }

Implementation rules inside the GTE source:

    • If use_memory_efficient_attention is **True**
        – xFormers must be importable, otherwise an assertion fires.
        – attn_implementation is automatically coerced to "eager"
          (the code does this for you, but supplying "eager" is clearer).

    • If use_memory_efficient_attention is **False**
        – You may still set unpad_inputs=True. The model will unpad/re-pad
          tensors using pure-PyTorch helpers (slower but functional).
        – attn_implementation can be "sdpa" or "eager". Either works.

────────────────────────────────────────────────────────────────────────────
3.  Stella-400 M (v5)
────────────────────────────────────────────────────────────────────────────
✔  Same flag block, but with stricter rules:

    {
        "config_kwargs": {
            "use_memory_efficient_attention":  <bool>,   # optional
            "unpad_inputs":                    <bool>,   # should match the flag above
            "attn_implementation":             "eager"   # ALWAYS "eager"
        }
    }

• The 400 M code path **does not implement an SDPA class** yet, so
  "eager" is mandatory even when xFormers is disabled.

• If you set use_memory_efficient_attention=True while xFormers is
  missing, an assertion will raise at runtime.

────────────────────────────────────────────────────────────────────────────
4.  Flag placement summary
────────────────────────────────────────────────────────────────────────────
outer  model_kwargs   (passed to SentenceTransformer)
│
├── tokenizer_kwargs   → forwarded to AutoTokenizer   ← configure padding
│                                                          & truncation here
│
├── model_kwargs       → forwarded to AutoModel       ← runtime knobs
│                        (dtype, quantisation, ...)
│
└── config_kwargs      → forwarded to AutoConfig      ← put the three
                         (BEFORE weights load)           xFormers flags here
     • use_memory_efficient_attention
     • unpad_inputs
     • attn_implementation

────────────────────────────────────────────────────────────────────────────
5.  Tokenization vs. encode_kwargs  (common pit-fall)
────────────────────────────────────────────────────────────────────────────
• SentenceTransformer.encode() *never* forwards encode_kwargs into the
  tokenizer. It tokenizes first, then passes encode_kwargs into the model's
  forward() call.

• Therefore:
      – padding / truncation / max_length → tokenizer_kwargs
      – batch_size, convert_to_numpy, etc. → encode_kwargs

Putting padding flags in encode_kwargs is silently ignored.

────────────────────────────────────────────────────────────────────────────
6.  Runtime checklist
────────────────────────────────────────────────────────────────────────────
□ GPU build?           → supports_flash_attention() must confirm
□ xFormers installed?  → import xformers.ops succeeds
□ Flags consistent?    → unpad_inputs should mirror use_memory_efficient_attention
□ attn_implementation  → "eager" for 400 M; "eager"/"sdpa" for others
"""