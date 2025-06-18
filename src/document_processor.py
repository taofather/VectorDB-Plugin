import os
import sys
import io
import logging
import warnings
from contextlib import redirect_stdout
import yaml
import math
from pathlib import Path, PurePath
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from langchain_community.docstore.document import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import TextSplitter
# # ─ monkey-patch block starts
# _old_merge = TextSplitter._merge_splits
# def _debug_merge(self, splits, separator):
    # print(">>> _length_function TYPE *inside* _merge_splits:", type(self._length_function))
    # return _old_merge(self, splits, separator)
# TextSplitter._merge_splits = _debug_merge
# # ─ monkey-patch block ends
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    EverNoteLoader,
    UnstructuredEPubLoader,
    UnstructuredEmailLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredRTFLoader,
    UnstructuredODTLoader,
    UnstructuredMarkdownLoader,
    BSHTMLLoader
)

from typing import Optional, Any, Iterator, Union, List, Dict
from langchain_community.document_loaders.blob_loaders import Blob

from langchain_community.document_loaders.parsers import PyMuPDFParser
import fitz  # Aquest és el nom correcte del mòdul dins de PyMuPDF

from constants import DOCUMENT_LOADERS
from extract_metadata import extract_document_metadata, add_pymupdf_page_metadata, compute_content_hash
from utilities import my_cprint
from config_manager import ConfigManager

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_processor.log', mode='w')
    ]
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SOURCE_DIRECTORY = ConfigManager().docs_dir
INGEST_THREADS = max(4, os.cpu_count() - 4)

class CustomPyMuPDFParser(PyMuPDFParser):
    def _lazy_parse(self, blob: Blob, text_kwargs: Optional[dict[str, Any]] = None) -> Iterator[Document]:
        with PyMuPDFParser._lock:
            with blob.as_bytes_io() as file_path:
                doc = fitz.open(stream=file_path, filetype="pdf") if blob.data else fitz.open(file_path)

                full_content = []
                for page in doc:
                    page_content = self._get_page_content(doc, page, text_kwargs)
                    # Only add page marker and content if there's actual content
                    if page_content.strip():
                        full_content.append(f"[[page{page.number + 1}]]{page_content}")

                yield Document(
                    page_content="".join(full_content),
                    metadata=self._extract_metadata(doc, blob)
                )

class CustomPyMuPDFLoader(PyMuPDFLoader):
    def __init__(self, file_path: Union[str, PurePath], **kwargs: Any) -> None:
        super().__init__(file_path, **kwargs)
        self.parser = CustomPyMuPDFParser(
            text_kwargs=kwargs.get('text_kwargs'),
            extract_images=kwargs.get('extract_images', False)
        )

for ext, loader_name in DOCUMENT_LOADERS.items():
    DOCUMENT_LOADERS[ext] = globals()[loader_name]

def load_single_document(file_path: Path) -> Document:
    file_extension = file_path.suffix.lower()
    loader_class = DOCUMENT_LOADERS.get(file_extension)
    if not loader_class:
        print(f"\033[91mFailed---> {file_path.name} (extension: {file_extension})\033[0m")
        logging.error(f"Unsupported file type: {file_path.name} (extension: {file_extension})")
        return None

    loader_options = {}

    if file_extension in [".epub", ".rtf", ".odt", ".md"]:
        loader_options.update({
            "mode": "single",
            "unstructured_kwargs": {
                "strategy": "fast"
            }
        })
    elif file_extension == ".pdf":
        loader_options.update({
            "extract_images": False,
            "text_kwargs": {},  # Optional: passed to https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_text
        })
    elif file_extension in [".eml", ".msg"]:
        loader_options.update({
            "mode": "single",
            "process_attachments": False,
            "unstructured_kwargs": {
                "strategy": "fast"
            }
        })
    elif file_extension == ".html":
        loader_options.update({
            "open_encoding": "utf-8",
            "bs_kwargs": {
                "features": "lxml",  # Specify the parser to use (lxml is generally fast and lenient)
                # "parse_only": SoupStrainer("body"),  # Optionally parse only the body tag
                # "from_encoding": "utf-8",  # Specify a different input encoding if needed
            },
            "get_text_separator": "\n",  # Use newline as separator when extracting text
            # "file_path": "path/to/file.html",  # Usually set automatically by the loader
            # "open_encoding": None,  # Set to None to let BeautifulSoup detect encoding
            # "get_text_separator": " ",  # Use space instead of newline if preferred
        })
    elif file_extension in [".xlsx", ".xls", ".xlsm"]:
        loader_options.update({
            "mode": "single",
            "unstructured_kwargs": {
                "strategy": "fast"
            }
        })
    elif file_extension in [".csv", ".txt"]:
        loader_options.update({
            "encoding": "utf-8",
            "autodetect_encoding": True
        })

    try:
        if file_extension in [".epub", ".rtf", ".odt", ".md", ".eml", ".msg", ".xlsx", ".xls", ".xlsm"]:
            unstructured_kwargs = loader_options.pop("unstructured_kwargs", {})
            loader = loader_class(str(file_path), mode=loader_options.get("mode", "single"), **unstructured_kwargs)
        else:
            loader = loader_class(str(file_path), **loader_options)
        documents = loader.load()

        if not documents:
            print(f"\033[91mFailed---> {file_path.name} (No content extracted)\033[0m")
            logging.error(f"No content could be extracted from file: {file_path.name}")
            return None

        document = documents[0]

        if not isinstance(document.page_content, str):
            if isinstance(document.page_content, (list, tuple)):
                document.page_content = "\n".join(map(str, document.page_content))
            else:
                document.page_content = str(document.page_content)

        safe_metadata = {}
        for k, v in document.metadata.items():
            safe_metadata[k] = v if isinstance(v, (str, int, float, bool)) or v is None else str(v)
        document.metadata = safe_metadata

        content_hash = compute_content_hash(document.page_content)
        metadata = extract_document_metadata(file_path, content_hash)
        document.metadata.update(metadata)
        print(f"Loaded---> {file_path.name}")
        return document

    except (OSError, UnicodeDecodeError) as e:
        print(f"\033[91mFailed---> {file_path.name} (Access/encoding error)\033[0m")
        logging.error(f"File access/encoding error - File: {file_path.name} - Error: {str(e)}")
        return None
    except Exception as e:
        print(f"\033[91mFailed---> {file_path.name} (Unexpected error)\033[0m")
        logging.error(f"Unexpected error processing file: {file_path.name} - Error: {str(e)}")
        return None

def load_document_batch(filepaths, threads_per_process):
    with ThreadPoolExecutor(threads_per_process) as exe:
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        data_list = [future.result() for future in futures]
    return (data_list, filepaths)

def load_documents(source_dir: Path) -> list:
    valid_extensions = {ext.lower() for ext in DOCUMENT_LOADERS.keys()}
    doc_paths = [f for f in source_dir.iterdir() if f.suffix.lower() in valid_extensions]

    docs = []

    if doc_paths:
        n_workers = min(INGEST_THREADS, max(len(doc_paths), 1))

        total_cores = os.cpu_count()
        threads_per_process = 1

        with ProcessPoolExecutor(n_workers) as executor:
            chunksize = math.ceil(len(doc_paths) / n_workers)
            futures = []
            for i in range(0, len(doc_paths), chunksize):
                chunk_paths = doc_paths[i:i + chunksize]
                futures.append(executor.submit(load_document_batch, chunk_paths, threads_per_process))

            for future in as_completed(futures):
                contents, _ = future.result()
                docs.extend([d for d in contents if d is not None])

    return docs

def split_documents(documents=None, text_documents_pdf=None):
    try:
        print("\nSplitting documents into chunks.")
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(config_path, "r", encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file)
            chunk_size = config["database"]["chunk_size"]
            chunk_overlap = config["database"]["chunk_overlap"]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=False,
        )

        texts = []

        # Split non-PDF documents (no cleanup needed - handled in load_single_document)
        if documents:
            # Ensure all input documents have string content before splitting
            for doc in documents:
                if not isinstance(doc.page_content, str):
                    doc.page_content = str(doc.page_content or "")
            
            texts = text_splitter.split_documents(documents)
            
            # Ensure all split documents have string content
            for text_doc in texts:
                if not isinstance(text_doc.page_content, str):
                    text_doc.page_content = str(text_doc.page_content or "")

        """
        I customized langchain's pymupdfparser to add custom page markers as follows:
        
        [[page1]]This is the text content of the first page.
        It might contain multiple lines, paragraphs, or sections.

        [[page2]]This is the text content of the second page.
        Again, it could be as long as necessary, depending on the content.

        [[page3]]Finally, this is the text content of the third page.
        """

        # ------------------------------------------------------------------ #
        # 2. Split PDF documents (with custom page markers)                   #
        # ------------------------------------------------------------------ #
        if text_documents_pdf:
            # Ensure all PDF documents have string content before processing
            for pdf_doc in text_documents_pdf:
                if not isinstance(pdf_doc.page_content, str):
                    pdf_doc.page_content = str(pdf_doc.page_content or "")
            
            processed_pdf_docs = []
            for doc in text_documents_pdf:
                chunked_docs = add_pymupdf_page_metadata(
                    doc,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                processed_pdf_docs.extend(chunked_docs)
            
            # Ensure all PDF chunks have string content
            for pdf_doc in processed_pdf_docs:
                if not isinstance(pdf_doc.page_content, str):
                    pdf_doc.page_content = str(pdf_doc.page_content or "")
            
            texts.extend(processed_pdf_docs)

        # Final safety check: ensure ALL texts have string content
        for text_doc in texts:
            if not isinstance(text_doc.page_content, str):
                text_doc.page_content = str(text_doc.page_content or "")

        return texts

    except Exception as e:
        logging.exception("Error during document splitting")
        logging.error(f"Error type: {type(e)}")
        raise

def process_documents(documents_dir: str) -> List[Dict[str, Any]]:
    """Process documents in the specified directory."""
    try:
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # Get document types from config
        document_types = config.get('document_types', [])
        if not document_types:
            my_cprint("No document types specified in config", "red")
            return []

        # Process each document
        processed_docs = []
        for doc_type in document_types:
            docs = process_document_type(documents_dir, doc_type, config)
            processed_docs.extend(docs)

        return processed_docs

    except Exception as e:
        my_cprint(f"Error processing documents: {e}", "red")
        return []

def process_document_type(documents_dir: str, doc_type: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process documents of a specific type."""
    try:
        # Get type-specific settings
        type_settings = config.get(f'{doc_type}_settings', {})
        if not type_settings:
            my_cprint(f"No settings found for document type: {doc_type}", "red")
            return []

        # Get file patterns for this type
        patterns = type_settings.get('patterns', [])
        if not patterns:
            my_cprint(f"No file patterns specified for document type: {doc_type}", "red")
            return []

        # Process matching files
        processed_docs = []
        for pattern in patterns:
            docs = process_files_by_pattern(documents_dir, pattern, doc_type, type_settings)
            processed_docs.extend(docs)

        return processed_docs

    except Exception as e:
        my_cprint(f"Error processing document type {doc_type}: {e}", "red")
        return []

def process_files_by_pattern(documents_dir: str, pattern: str, doc_type: str, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process files matching a specific pattern."""
    try:
        # Find matching files
        matching_files = Path(documents_dir).glob(pattern)
        
        # Process each file
        processed_docs = []
        for file_path in matching_files:
            doc = process_single_file(file_path, doc_type, settings)
            if doc:
                processed_docs.append(doc)

        return processed_docs

    except Exception as e:
        my_cprint(f"Error processing files with pattern {pattern}: {e}", "red")
        return []

def process_single_file(file_path: Path, doc_type: str, settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single file."""
    try:
        # Basic file info
        doc = {
            'path': str(file_path),
            'name': file_path.name,
            'type': doc_type,
            'size': file_path.stat().st_size
        }

        # Add type-specific processing here
        if doc_type == 'text':
            process_text_file(doc, settings)
        elif doc_type == 'image':
            process_image_file(doc, settings)
        # Add more document types as needed

        return doc

    except Exception as e:
        my_cprint(f"Error processing file {file_path}: {e}", "red")
        return None

def process_text_file(doc: Dict[str, Any], settings: Dict[str, Any]) -> None:
    """Process a text file."""
    try:
        with open(doc['path'], 'r', encoding='utf-8') as f:
            content = f.read()
        doc['content'] = content
        doc['word_count'] = len(content.split())

    except Exception as e:
        my_cprint(f"Error processing text file {doc['path']}: {e}", "red")

def process_image_file(doc: Dict[str, Any], settings: Dict[str, Any]) -> None:
    """Process an image file."""
    try:
        # Add image-specific processing here
        # For example: extract dimensions, format, etc.
        pass

    except Exception as e:
        my_cprint(f"Error processing image file {doc['path']}: {e}", "red")

"""
The PyMUPDF parser was modified in langchain-community 0.3.15+

- Adds "producer" and "creator" metadata fields
- Adds thread safety features
- Adds support for encrypted PDFs, tables, and enhanced image extraction
- Added configurable page handling modes

+----------------------+---------------------------+---------------+-----------+
| Parameter            | Available Options         | Default Value | Required? |
+----------------------+---------------------------+---------------+-----------+
| mode                 | "single", "page"          | "page"        | No        |
+----------------------+---------------------------+---------------+-----------+
| password            | Any string                 | None          | No        |
+----------------------+---------------------------+---------------+-----------+
| pages_delimiter     | Any string                 | "\n\f"        | No        |
+----------------------+---------------------------+---------------+-----------+
| extract_images      | True, False                | False         | No        |
+----------------------+---------------------------+---------------+-----------+
| images_parser       | BaseImageBlobParser obj    | None          | No        |
+----------------------+---------------------------+---------------+-----------+
| images_inner_format | "text"                     | "text"        | No        |
|                     | "markdown-img"             |               |           |
|                     | "html-img"                 |               |           |
+----------------------+---------------------------+---------------+-----------+
| extract_tables      | "csv"                      | None          | No        |
|                     | "markdown"                 |               |           |
|                     | "html"                     |               |           |
|                     | None                       |               |           |
+----------------------+---------------------------+---------------+-----------+
| extract_tables      | Dictionary with settings   | None          | No        |
| _settings           | for table extraction       |               |           |
+----------------------+---------------------------+---------------+-----------+
| text_kwargs         | Dictionary with text       | None          | No        |
| (Parser only)       | extraction settings        |               |           |
+----------------------+---------------------------+---------------+-----------+

This table is ONLY RELEVANT if I do not use custom sub-classes.  Keep for possible future reference

Regarding the additional metadata fields, this won't interfere with extract_metadata.py because it:

1) Applies after the document is loaded; and
2) Uses document.metadata.update(metadata), which means that it will either:

* Add your metadata fields alongside the Langchain metadata; or
* Override any duplicate fields with your values
"""