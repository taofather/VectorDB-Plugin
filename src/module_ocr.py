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
            print(f"\033[92mUsing up to {thread_count} threads\033[0m")

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
        """Each OCR backend implements its own text cleaning method, if needed."""
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
        self.temp_dir = None
        self.show_progress = True

    def initialize(self):
        import tempfile

        script_dir = Path(__file__).resolve().parent

        # control temporary directory
        # necessary in case the default temp locations don't have write permission
        self.temp_dir = script_dir / "temp_ocr"
        self.temp_dir.mkdir(exist_ok=True)

        os.environ['TMP'] = str(self.temp_dir)
        os.environ['TEMP'] = str(self.temp_dir)
        tempfile.tempdir = str(self.temp_dir)

        self.tessdata_path = script_dir / 'share' / 'tessdata'
        os.environ['TESSDATA_PREFIX'] = str(self.tessdata_path)

    def clean_text(self, text: str) -> str:
        return text

    def process_page(self, page_num: int, page) -> Tuple[int, str]:
        import tempfile
        import tesserocr
        import time
        from io import BytesIO
        from ocrmypdf.hocrtransform import HocrTransform

        fd, temp_pdf_path = tempfile.mkstemp(suffix=".pdf", dir=self.temp_dir)
        os.close(fd)

        out_pdf = fitz.open()

        with tesserocr.PyTessBaseAPI(lang="eng", path=str(self.tessdata_path)) as api:
            # remove rotation if present
            page.remove_rotation()

            # 2x page zoom for better OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            pil_image = Image.open(BytesIO(pix.tobytes("png")))

            api.SetImage(pil_image)
            hocr_text = api.GetHOCRText(0)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".hocr", dir=self.temp_dir) as hocr_temp:
                hocr_output = hocr_temp.name
            Path(hocr_output).write_text(hocr_text, encoding="utf-8")

            fd, text_pdf = tempfile.mkstemp(suffix=".pdf", dir=self.temp_dir)
            os.close(fd)

            pdf_width_pts = page.rect.width
            pdf_height_pts = page.rect.height
            dpi_x = (pix.width * 72) / pdf_width_pts
            dpi_y = (pix.height * 72) / pdf_height_pts
            dpi = (dpi_x + dpi_y) / 2.0

            hocr_transform = HocrTransform(hocr_filename=hocr_output, dpi=dpi)
            hocr_transform.width = pdf_width_pts
            hocr_transform.height = pdf_height_pts
            hocr_transform.to_pdf(out_filename=text_pdf, invisible_text=True)

            out_pdf.insert_pdf(page.parent, from_page=page_num, to_page=page_num)

            with fitz.open(text_pdf) as text_page:
                out_pdf[0].show_pdf_page(
                    out_pdf[0].rect,
                    text_page,
                    0,
                    overlay=True
                )

            Path(hocr_output).unlink(missing_ok=True) # comment to keep the hocr file for DEBUG

            for _ in range(10):
                try:
                    Path(text_pdf).unlink()
                    break
                except PermissionError:
                    time.sleep(0.1)

            out_pdf.save(temp_pdf_path)
            out_pdf.close()

            return page_num, temp_pdf_path

    def cleanup_temp_pdfs(self):
        if self.temp_dir is None:
            return

        for temp_file in Path(self.temp_dir).glob("tmp*.pdf"):
            try:
                temp_file.unlink()
            except PermissionError:
                print(f"Could not remove {temp_file} - file may be in use")

    def process_document(self, pdf_path: Path, output_path: Path = None):
        if not self.validate_pdf(pdf_path):
            raise ValueError(f"Invalid or corrupted PDF file: {pdf_path}")

        if output_path is None:
            output_path = pdf_path.with_stem(f"{pdf_path.stem}_OCR")

        if self.temp_dir is None:
            self.initialize()

        self.cleanup_temp_pdfs()

        pdf_document = fitz.open(str(pdf_path))
        num_pages = len(pdf_document)
        pdf_document.close()

        if self.progress_queue:
            self.progress_queue.put(('total', num_pages))

        page_args = [(page_num, fitz.open(str(pdf_path))[page_num]) for page_num in range(num_pages)]

        results = []
        with ThreadPoolExecutor(max_workers=self.get_optimal_threads()) as executor:
            futures = {executor.submit(self.process_page, *args): args[0] for args in page_args}

            for future in as_completed(futures):
                page_num, temp_pdf_path = future.result()
                results.append((temp_pdf_path, page_num))
                if self.progress_queue:
                    self.progress_queue.put(('update', 1))

        results.sort(key=lambda x: x[1])

        output_pdf = fitz.open()
        for temp_pdf_path, _ in results:
            output_pdf.insert_pdf(fitz.open(temp_pdf_path))

            Path(temp_pdf_path).unlink(missing_ok=True)

        output_pdf.save(output_path)
        output_pdf.close()

        # final cleanup
        self.cleanup_temp_pdfs()

        output_file = Path(output_path).name

        if self.progress_queue:
            self.progress_queue.put(('done', None))

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
