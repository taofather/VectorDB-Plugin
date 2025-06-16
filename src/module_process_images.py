import traceback
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    GenerationConfig,
    AutoConfig,
    AutoModelForVision2Seq,
    CLIPProcessor,
    CLIPModel
)
from langchain_community.docstore.document import Document
from extract_metadata import extract_image_metadata
from utilities import my_cprint, has_bfloat16_support
from constants import VISION_MODELS
from config_manager import ConfigManager

warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff']

current_directory = Path(__file__).parent
CACHE_DIR = current_directory / "models" / "vision"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_best_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def check_for_images(image_dir):
    return any(file.suffix.lower() in ALLOWED_EXTENSIONS for file in Path(image_dir).iterdir())

def run_loader_in_process(loader_func):
    try:
        return loader_func()
    except Exception as e:
        error_message = f"Error processing images: {e}\n\nTraceback:\n{traceback.format_exc()}"
        my_cprint(error_message, "red")
        return []

def choose_image_loader():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    chosen_model = config["vision"]["chosen_model"]
    if chosen_model == 'Granite Vision - 2b':
        loader_func = loader_granite(config).process_images
    elif chosen_model == 'THUDM glm4v - 9b':
        loader_func = loader_glmv4(config).process_images
    elif chosen_model == 'Molmo-D-0924 - 8b':
        loader_func = loader_molmo(config).process_images
    elif chosen_model in ['Ovis2 - 1b', 'Ovis2 - 2b']:
        loader_func = loader_ovis(config).process_images
    elif chosen_model in ['InternVL3 - 1b', 'InternVL3 - 2b', 'InternVL3 - 8b', 'InternVL3 - 14b']:
        loader_func = loader_internvl(config).process_images
    elif chosen_model in ['Qwen VL - 3b', 'Qwen VL - 7b']:
        loader_func = loader_qwenvl(config).process_images
    else:
        my_cprint("No valid image model specified in config.yaml", "red")
        return []
    image_dir = Path(__file__).parent / "Docs_for_DB"
    if not check_for_images(image_dir):
        return []
    with ProcessPoolExecutor(1) as executor:
        future = executor.submit(run_loader_in_process, loader_func)
        try:
            processed_docs = future.result()
        except Exception as e:
            my_cprint(f"Error occurred during image processing: {e}", "red")
            return []
        return processed_docs or []


class BaseLoader:
    def __init__(self, config):
        self.config = config
        self.device = get_best_device()
        self.model = None
        self.tokenizer = None
        self.processor = None

    def initialize_model_and_tokenizer(self):
        raise NotImplementedError

    def process_images(self):
        image_dir = Path(__file__).parent / "Docs_for_DB"
        documents = []
        image_files = [file for file in image_dir.iterdir() if file.suffix.lower() in ALLOWED_EXTENSIONS]
        self.model, self.tokenizer, self.processor = self.initialize_model_and_tokenizer()
        print("Processing images.")
        start_time = time.time()
        with tqdm(total=len(image_files), unit="image") as progress_bar:
            for full_path in image_files:
                try:
                    with Image.open(full_path) as raw_image:
                        extracted_text = self.process_single_image(raw_image)
                        extracted_metadata = extract_image_metadata(full_path)
                        documents.append(Document(page_content=extracted_text, metadata=extracted_metadata))
                        progress_bar.update(1)
                except Exception as e:
                    print(f"{full_path.name}: Error processing image - {e}")
        total_time = time.time() - start_time
        print(f"Loaded {len(documents)} image(s).")
        print(f"Total image processing time: {total_time:.2f} seconds")
        my_cprint("Vision model removed from memory.", "red")
        return documents

    def process_single_image(self, raw_image):
        raise NotImplementedError


class loader_glmv4(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        info = VISION_MODELS[chosen_model]
        model_id = info['repo_id']
        save_dir = info["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda")
        use_bf16 = torch.cuda.get_device_capability()[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True).vision_config.update(image_size=448)
        model = AutoModelForCausalLM.from_pretrained(model_id, token=False, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True, quantization_config=quant_config, cache_dir=cache_dir).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=False, trust_remote_code=True, cache_dir=cache_dir)
        precision_str = "bfloat16" if use_bf16 else "float16"
        device_str = "CUDA" if self.device == "cuda" else "CPU"
        my_cprint(f"{chosen_model} loaded into memory on {device_str} ({precision_str})", "green")
        return model, tokenizer, None

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        query = "Describe this image in as much detail as possible but do not repeat yourself."
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "image": raw_image, "content": query}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=1024, do_sample=False)

        outputs = outputs[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


class loader_molmo(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        info = VISION_MODELS[chosen_model]
        source = info.get('model_path') or info['repo_id']
        cache_dir = CACHE_DIR / info.get('cache_dir','')
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.processor = AutoProcessor.from_pretrained(source, token=False, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto', cache_dir=cache_dir)
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(source, token=False, trust_remote_code=True, quantization_config=quant_config, torch_dtype=torch.bfloat16, device_map='auto', cache_dir=cache_dir)
        self.model.model.vision_backbone = self.model.model.vision_backbone.to(torch.float32)
        self.model.eval()
        device_str = "CUDA" if self.device == "cuda" else "CPU"
        my_cprint(f"{chosen_model} loaded into memory on {device_str} (bfloat16)", "green")
        return self.model, None, self.processor

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")
        prompt = "Describe this image in as much detail as possible but do not repeat yourself."
        inputs = self.processor.process(images=[raw_image], text=prompt)
        inputs = {k: (v.to(device=self.device, dtype=torch.long) if k in ['input_ids','image_input_idx'] else v.to(device=self.device,dtype=torch.float32)).unsqueeze(0) for k,v in inputs.items()}
        try:
            gen_cfg = GenerationConfig(max_new_tokens=1024, eos_token_id=self.processor.tokenizer.eos_token_id)
            out = self.model.generate_from_batch(inputs, gen_cfg, tokenizer=self.processor.tokenizer)
            tokens = out[0, inputs['input_ids'].size(1):]
            text = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
            return ' '.join(line.strip() for line in text.split('\n') if line.strip())
        except Exception as e:
            my_cprint(f"Error processing image: {e}", "red")
            return ""


class loader_ovis(BaseLoader):
    def __init__(self, config):
        super().__init__(config)

    def initialize_model_and_tokenizer(self):
        chosen_model = self.config["vision"]["chosen_model"]
        info = VISION_MODELS[chosen_model]

        cache_dir = CACHE_DIR / info["cache_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)

        if self.device == "cuda":
            use_bf16 = torch.cuda.get_device_capability()[0] >= 8
            dtype = torch.bfloat16 if use_bf16 else torch.float16
        else:
            dtype = torch.float32

        self.model_dtype = dtype

        model = AutoModelForCausalLM.from_pretrained(
            info["repo_id"],
            torch_dtype=dtype,
            trust_remote_code=True,
            multimodal_max_length=8192,
            cache_dir=cache_dir,
            token=False
        ).to(self.device)

        model.eval()

        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()

        self.model = model

        precision_str = "bfloat16" if dtype == torch.bfloat16 else "float16" if dtype == torch.float16 else "float32"
        device_str = "CUDA" if self.device == "cuda" else "CPU"
        my_cprint(f"{chosen_model} loaded into memory on {device_str} ({precision_str})", "green")
        return model, text_tokenizer, visual_tokenizer

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        prompt = (
            "Explain everything you see in this picture "
            "but your response should be no more than one paragraph."
        )
        query = f"<image>\n{prompt}"

        _, input_ids, pixel_values = self.model.preprocess_inputs(query, [raw_image])

        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)

        input_ids = input_ids.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)

        pixel_values = pixel_values.to(device=self.device, dtype=self.model_dtype)

        pixel_values = [pixel_values]

        gen_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }

        # **Pass input_ids positionally** so Ovis2's generate() sees it as text_input_ids
        output_ids = self.model.generate(
            input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            **gen_kwargs
        )[0]

        description = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return " ".join(line.strip() for line in description.split("\n") if line.strip())


class loader_internvl(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        info = VISION_MODELS[chosen_model]
        cache_dir = CACHE_DIR / info["cache_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.device == "cuda":
            use_bf16 = torch.cuda.get_device_capability()[0] >= 8
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            precision_str = "bfloat16" if use_bf16 else "float16"

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=[
                    "vision_model",
                    "language_model.model.norm", 
                    "language_model.output",
                    "language_model.model.rotary_emb",
                    "language_model.lm_head",
                    "mlp1"
                ]
            )
            model = AutoModel.from_pretrained(
                info['repo_id'],
                quantization_config=quant_config,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                cache_dir=cache_dir,
                token=False
            ).eval()
        else:
            # CPU fallback
            dtype = torch.float32
            precision_str = "float32"
            model = AutoModel.from_pretrained(
                info['repo_id'],
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                cache_dir=cache_dir,
                token=False,
                device_map={"": "cpu"}
            ).eval()

        self.model_dtype = dtype
        device_str = "CUDA" if self.device == "cuda" else "CPU"

        tokenizer = AutoTokenizer.from_pretrained(
            info['repo_id'],
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=False
        )
        my_cprint(f"{chosen_model} loaded into memory on {device_str} ({precision_str})", "green")
        return model, tokenizer, None

    def find_closest_aspect_ratio(self, aspect_ratio, ratios, w, h, sz):
        best_diff = float('inf')
        best = (1, 1)
        area = w * h
        for r in ratios:
            ar = r[0] / r[1]
            diff = abs(aspect_ratio - ar)
            if diff < best_diff or (diff == best_diff and area > 0.5 * sz * sz * r[0] * r[1]):
                best_diff = diff
                best = r
        return best

    def _build_transform(self, size):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((size, size), interpolation=InterpolationMode.LANCZOS, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    def dynamic_preprocess(self, img, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        w, h = img.size
        ar = w / h
        ratios = sorted(
            {(i, j)
             for n in range(min_num, max_num + 1)
             for i in range(1, n + 1)
             for j in range(1, n + 1)
             if i * j <= max_num and i * j >= min_num},
            key=lambda x: x[0] * x[1]
        )
        best = self.find_closest_aspect_ratio(ar, ratios, w, h, image_size)
        tw, th = image_size * best[0], image_size * best[1]
        resized = img.resize((tw, th))
        blocks = best[0] * best[1]
        cols = tw // image_size
        parts = []
        for i in range(blocks):
            x = (i % cols) * image_size
            y = (i // cols) * image_size
            parts.append(resized.crop((x, y, x + image_size, y + image_size)))
        if use_thumbnail and len(parts) != 1:
            parts.append(img.resize((image_size, image_size)))
        return parts

    def _prepare_image(self, raw_image, input_size=448, max_num=24):
        imgs = self.dynamic_preprocess(raw_image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        tf = self._build_transform(input_size)
        return torch.stack([tf(im) for im in imgs])

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        pv = self._prepare_image(raw_image).to(self.model_dtype).to(self.device)
        question = "<image>\nExplain everything you see in this picture but your response should be no more than one paragraph, but the paragraph can be as long as you want."
        gen_cfg = {
            'num_beams': 1,
            'max_new_tokens': 1024,
            'do_sample': False,
            'pad_token_id': self.tokenizer.pad_token_id
        }
        resp = self.model.chat(self.tokenizer, pv, question, gen_cfg)
        return ' '.join(line.strip() for line in resp.split('\n') if line.strip())


class loader_granite(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        model_id = VISION_MODELS[chosen_model]['repo_id']
        save_dir = VISION_MODELS[chosen_model]["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        processor = AutoProcessor.from_pretrained(
            model_id,
            use_fast=True,
            cache_dir=cache_dir,
            token=False
        )

        if self.device == "cuda" and torch.cuda.is_available():
            use_bf16 = torch.cuda.get_device_capability()[0] >= 8
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            precision_str = "bfloat16" if use_bf16 else "float16"
            
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=[
                    "vision_tower",
                    "multi_modal_projector", 
                    "language_model.embed_tokens",
                    "language_model.norm",
                    "lm_head"
                ]
            )

            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                quantization_config=config,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=False,
                device_map="auto"
            )
            precision_str = "bfloat16" if use_bf16 else "float16"
            my_cprint(f"{chosen_model} loaded into memory on CUDA ({precision_str})", "green")
            
        else:
            # CPU mode - no quantization
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                token=False,
                device_map={"": "cpu"}
            )
            my_cprint(f"{chosen_model} loaded into memory on CPU (float32)", "green")
        
        model.eval()
        return model, None, processor

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")
        msg = (
            "Describe in detail what this image depicts but limit your response "
            "to one paragraph with no line breaks in it."
        )
        prompt = f"<|user|>\n<image>\n{msg}\n<|assistant|>\n"
        inputs = self.processor(images=raw_image, text=prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False, num_beams=1)
        resp = self.processor.decode(output[0], skip_special_tokens=True).split('<|assistant|>')[-1].strip()
        return " ".join(line.strip() for line in resp.split("\n") if line.strip())


class loader_qwenvl(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        model_info = VISION_MODELS[chosen_model]
        model_id = model_info['repo_id']
        save_dir = model_info['cache_dir']
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        use_bf16 = torch.cuda.get_device_capability()[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        precision_str = "bfloat16" if use_bf16 else "float16"
        device_str = "CUDA" if self.device == "cuda" else "CPU"
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=[
                "lm_head",
                "merger",
                "visual.blocks.0.attn",
                "visual.blocks.0.mlp",
                "visual.blocks.1.attn",
                "visual.blocks.1.mlp",
                "visual.blocks.2.attn",
                "visual.blocks.2.mlp",
                "visual.blocks.3.attn",
                "visual.blocks.3.mlp",
                "visual.blocks.4.attn",
                "visual.blocks.5.mlp",
                "visual.blocks.7.attn",
                "visual.blocks.7.mlp",
                "visual.blocks.8.mlp",
                "visual.blocks.10.mlp",
                "visual.blocks.12.mlp",
                "visual.blocks.13.mlp",
                "visual.blocks.14.attn",
                "visual.blocks.14.mlp",
                "visual.blocks.15.attn",
                "visual.blocks.15.mlp",
                "visual.blocks.17.mlp",
                "visual.blocks.31.mlp.down_proj"
            ]
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            use_fast=True,
            min_pixels=28*28,
            max_pixels=1280*28*28,
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=False
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=False
        )
        model = model.to(self.device)
        model.eval()
        precision_str = "bfloat16" if use_bf16 else "float16"
        my_cprint(f"{chosen_model} loaded into memory on {device_str} ({precision_str})", "green")
        return model, None, processor

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        user_message = "Describe in as much detail as possible what this image depicts?"
        prompt = (
            "<|im_start|>user\n"
            f"{user_message} <|vis_start|><|image_pad|><|vis_end|>\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        inputs = self.processor(
            images=raw_image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            top_k=None,
            top_p=None,
            num_beams=1,
            temperature=None
        )
        response = self.processor.decode(output[0], skip_special_tokens=True)
        response = response.split('assistant')[-1].strip()
        return ' '.join(line.strip() for line in response.split('\n') if line.strip())


def process_image(image_path):
    try:
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        image_model = config.get('image_model', '')
        if not image_model:
            my_cprint("No valid image model specified in config.yaml", "red")
            return None, None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CLIPModel.from_pretrained(image_model).to(device)
        processor = CLIPProcessor.from_pretrained(image_model)

        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt", padding=True).to(device)

        image_features = model.get_image_features(**inputs)
        image_features = image_features.detach().cpu().numpy()

        return image_features, image

    except Exception as e:
        my_cprint(f"Error processing image: {e}", "red")
        return None, None
