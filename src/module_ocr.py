import fitz
import io
from pathlib import Path
from PIL import Image
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import psutil
import tqdm
import torch
from typing import Union, List, Dict, Tuple
from multiprocessing import Process, Queue, Value

class OCRProcessor(ABC):
    def __init__(self, zoom: int = 2, progress_queue: Queue = None):
        self.zoom = zoom
        self.show_progress = False
        self.progress_queue = progress_queue
        backend_name = self.__class__.__name__
        print(f"\033[92mUsing {backend_name} backend\033[0m")
        if backend_name == "TesseractOCR":
            thread_count = self.get_optimal_threads()
            print(f"\033[92mUsing {thread_count} threads\033[0m")

    def convert_page_to_image(self, page) -> Image.Image:
        """Convert PDF page to PIL Image with specified zoom"""
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        return Image.open(io.BytesIO(img_data))

    @abstractmethod
    def process_page(self, page_num: int, page) -> Tuple[int, str]:
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def clean_text(self, text: str) -> str:
        """Each OCR backend implements its own text cleaning method"""
        pass

    def validate_pdf(self, pdf_path: Path) -> bool:
        """
        Validates that the PDF is readable and contains pages.
        Returns True if valid, False otherwise.
        """
        try:
            with fitz.open(str(pdf_path)) as doc:
                if doc.page_count == 0:
                    return False
                _ = doc[0].get_text()
            return True
        except Exception as e:
            return False

    def process_document(self, pdf_path: Path, output_path: Path = None):
        """Process entire PDF document"""
        if not self.validate_pdf(pdf_path):
            raise ValueError(f"Invalid or corrupted PDF file: {pdf_path}")

        if output_path is None:
            output_path = pdf_path.with_suffix('.txt')

        results = {}

        with fitz.open(str(pdf_path)) as pdf_document:
            total_pages = len(pdf_document)
            
            if self.progress_queue:
                self.progress_queue.put(('total', total_pages))

            with ThreadPoolExecutor(max_workers=self.get_optimal_threads()) as executor:
                future_to_page = {
                    executor.submit(self.process_page, page_num, pdf_document[page_num]): page_num 
                    for page_num in range(total_pages)
                }

                for future in as_completed(future_to_page):
                    page_num, processed_text = future.result()
                    results[page_num] = processed_text
                    if self.progress_queue:
                        self.progress_queue.put(('update', 1))

        with open(output_path, 'w', encoding='utf-8') as f:
            for page_num in range(total_pages):
                if results[page_num].strip():
                    page_content = f"[[page{page_num + 1}]]{results[page_num]}"
                    f.write(page_content)

        if self.progress_queue:
            self.progress_queue.put(('done', None))

    @staticmethod
    def get_optimal_threads() -> int:
        return max(4, psutil.cpu_count(logical=False) - 2)

class TesseractOCR(OCRProcessor):
    def __init__(self, zoom: int = 2, progress_queue: Queue = None):
        super().__init__(zoom, progress_queue)
        self.tessdata_path = None
        self.show_progress = True

    def initialize(self):
        from tesserocr import PyTessBaseAPI, PSM
        script_dir = Path(__file__).resolve().parent
        self.tessdata_path = script_dir / 'share' / 'tessdata'
        os.environ['TESSDATA_PREFIX'] = str(self.tessdata_path)

    def clean_text(self, text: str) -> str:
        """Tesseract-specific text cleaning"""
        lines = text.split('\n')
        processed_lines = []
        i = 0
        while i < len(lines):
            current_line = lines[i].rstrip()
            if current_line.endswith('-') and i < len(lines) - 1:
                next_line = lines[i+1].lstrip()
                if next_line and next_line[0].islower():
                    joined_word = current_line[:-1] + next_line.split(' ', 1)[0]
                    remaining_next_line = ' '.join(next_line.split(' ')[1:])
                    processed_lines.append(joined_word)
                    if remaining_next_line:
                        lines[i+1] = remaining_next_line
                    else:
                        i += 1
                else:
                    processed_lines.append(current_line)
            else:
                processed_lines.append(current_line)
            i += 1
        return '\n'.join(processed_lines)

    def process_page(self, page_num: int, page) -> Tuple[int, str]:
        image = self.convert_page_to_image(page)
        from tesserocr import PyTessBaseAPI, PSM
        # create a new API instance for each page to ensure thread safety.
        with PyTessBaseAPI(psm=PSM.AUTO, path=str(self.tessdata_path)) as api:
            api.SetImage(image)
            text = api.GetUTF8Text()
        text = self.clean_text(text)
        return page_num, text

class GotOCR(OCRProcessor):
    def __init__(self, model_path: str, zoom: int = 2, progress_queue: Queue = None):
        super().__init__(zoom, progress_queue)
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.show_progress = False

    def initialize(self):
        import sys
        import logging
        from transformers import AutoModel, AutoTokenizer
        from transformers import logging as transformers_logging
        from utilities import set_cuda_paths

        set_cuda_paths()
        transformers_logging.set_verbosity_error()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map='cuda',
            use_safetensors=True,
            pad_token_id=self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        )
        self.model = self.model.eval().cuda()

    def clean_text(self, text: str) -> str:
        """GOT-OCR specific text cleaning"""
        lines = text.split('\n')
        cleaned_lines = []
        i = 0
        while i < len(lines):
            cleaned_lines.append(lines[i])
            repeat_count = 1
            j = i + 1
            while j < len(lines) and lines[j] == lines[i]:
                repeat_count += 1
                j += 1
            if repeat_count > 2:
                if i + 1 < len(lines):
                    cleaned_lines.append(lines[i + 1])
                i = j
            else:
                i += 1
        return '\n'.join(cleaned_lines)

    def process_page(self, page_num: int, page) -> Tuple[int, str]:
        image = self.convert_page_to_image(page)
        with torch.inference_mode():
            text = self.model.chat_crop(self.tokenizer, image, ocr_type='ocr', gradio_input=True)
        text = self.clean_text(text)
        return page_num, text

def _process_documents_worker(
    pdf_paths: List[Path],
    backend: str,
    model_path: str,
    output_dir: Path,
    progress_queue: Queue
):
    """Worker process that handles the actual OCR processing"""
    if backend.lower() == 'tesseract':
        processor = TesseractOCR(progress_queue=progress_queue)
    elif backend.lower() == 'got':
        if not model_path:
            raise ValueError("model_path is required for GOT-OCR backend")
        processor = GotOCR(model_path, progress_queue=progress_queue)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    processor.initialize()

    for pdf_path in pdf_paths:
        output_path = None
        if output_dir:
            output_path = output_dir / f"{pdf_path.stem}_ocr.txt"
        processor.process_document(pdf_path, output_path)

def process_documents(
    pdf_paths: Union[Path, List[Path]], 
    backend: str = 'tesseract',
    model_path: str = None,
    output_dir: Path = None
):
    """
    Process one or multiple PDF documents using specified OCR backend

    Args:
        pdf_paths: Single Path or list of Paths to PDF files
        backend: 'tesseract' or 'got'
        model_path: Required for GOT-OCR backend
        output_dir: Optional output directory for results
    """
    if isinstance(pdf_paths, Path):
        pdf_paths = [pdf_paths]

    progress_queue = Queue()

    process = Process(target=_process_documents_worker, 
                     args=(pdf_paths, backend, model_path, output_dir, progress_queue))
    process.start()

    total_pages = None
    pbar = None

    while True:
        msg = progress_queue.get()
        cmd, data = msg

        if cmd == 'total':
            total_pages = data
            pbar = tqdm.tqdm(total=total_pages, desc="Processing pages")
        elif cmd == 'update':
            pbar.update(data)
        elif cmd == 'done':
            break

    if pbar:
        pbar.close()

    process.join()
