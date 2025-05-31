import traceback
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

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
    AutoModelForVision2Seq
)
from langchain_community.docstore.document import Document
from extract_metadata import extract_image_metadata
from utilities import my_cprint, has_bfloat16_support
from constants import VISION_MODELS

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
    if chosen_model in ["Florence-2-large", "Florence-2-base"]:
        loader_func = loader_florence2(config).process_images
    elif chosen_model == 'Granite Vision - 2b':
        loader_func = loader_granite(config).process_images
    elif chosen_model == 'THUDM glm4v - 9b':
        loader_func = loader_glmv4(config).process_images
    elif chosen_model == 'Molmo-D-0924 - 8b':
        loader_func = loader_molmo(config).process_images
    elif chosen_model in ['Ovis2 - 1b', 'Ovis2 - 2b']:
        loader_func = loader_ovis(config).process_images
    elif chosen_model in ['InternVL2.5 - 1b', 'InternVL2.5 - 4b']:
        loader_func = loader_internvl2_5(config).process_images
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


class loader_florence2(BaseLoader):
    def __init__(self, config):
        super().__init__(config)
        self.my_cprint = my_cprint

    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        repo_id = VISION_MODELS[chosen_model]["repo_id"]
        save_dir = VISION_MODELS[chosen_model]["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(repo_id, token=False, trust_remote_code=True, low_cpu_mem_usage=True, cache_dir=cache_dir).eval()
        processor = AutoProcessor.from_pretrained(repo_id,  token=False, trust_remote_code=True, cache_dir=cache_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            if torch.cuda.get_device_capability()[0] >= 8:
                model = model.to(self.device).bfloat16()
                self.model_dtype = torch.bfloat16
            else:
                model = model.to(self.device).half()
                self.model_dtype = torch.float16
        else:
            model = model.to(self.device).float()
            self.model_dtype = torch.float32
        device_type = "CUDA" if self.device.type == "cuda" else "CPU"
        self.my_cprint(f"Running {chosen_model} on {device_type} in precision {self.model_dtype}", "green")
        return model, None, processor

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model_dtype)
        generated_ids = self.model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=1, do_sample=False, early_stopping=False)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text, task=prompt,
            image_size=(raw_image.width, raw_image.height)
        )
        return parsed.get('<MORE_DETAILED_CAPTION>', generated_text)


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
        prec = "bfloat16" if use_bf16 else "float16"
        my_cprint(f"Running {chosen_model} on CUDA in {prec}", "green")
        return model, tokenizer, None

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        query = "Describe this image in as much detail as possible but do not repeat yourself."
        inputs = self.tokenizer.apply_chat_template([{"role":"user","image":raw_image,"content":query}], add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=1024, do_sample=False)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        desc = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return ' '.join(line.strip() for line in desc.split('\n') if line.strip())


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
        my_cprint(f"{chosen_model} vision model loaded into memory", "green")
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


# class loader_ovis(BaseLoader):
    # def __init__(self, config):
        # super().__init__(config)
        # native = VISION_MODELS[self.config["vision"]["chosen_model"]]["precision"]
        # # Choose dtype on GPU: bfloat16 if supported, else float16; always float32 on CPU
        # if self.device == "cuda":
            # if native in ("float32", "bfloat16") and has_bfloat16_support():
                # self.dtype = torch.bfloat16
            # elif native == "float32":
                # self.dtype = torch.float16
            # else:
                # self.dtype = torch.float16
        # else:
            # self.dtype = torch.float32

    # def initialize_model_and_tokenizer(self):
        # chosen_model = self.config["vision"]["chosen_model"]
        # info = VISION_MODELS[chosen_model]

        # cache_dir = CACHE_DIR / info["cache_dir"]
        # cache_dir.mkdir(parents=True, exist_ok=True)

        # model = AutoModelForCausalLM.from_pretrained(
            # info["repo_id"],
            # torch_dtype=self.dtype,
            # trust_remote_code=True,
            # multimodal_max_length=8192,
            # cache_dir=cache_dir
        # ).to(self.device)
        # model.eval()

        # text_tokenizer = model.get_text_tokenizer()
        # visual_tokenizer = model.get_visual_tokenizer()

        # for module in visual_tokenizer.modules():
            # if isinstance(module, torch.nn.Linear):
                # module.to(device=self.device, dtype=self.dtype)

        # return model, text_tokenizer, visual_tokenizer

    # @torch.inference_mode()
    # def process_single_image(self, raw_image):
        # prompt = (
            # "Explain everything you see in this picture "
            # "but your response should be no more than one paragraph."
        # )
        # query = f"<image>\n{prompt}"

        # _, input_ids, pixel_values = self.model.preprocess_inputs(query, [raw_image])
        # attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)

        # # Batchify and move to the correct device & dtype
        # input_ids      = input_ids.unsqueeze(0).to(self.device)        # [1, seq_len]
        # attention_mask = attention_mask.unsqueeze(0).to(self.device)  # [1, seq_len]
        # pixel_values   = pixel_values.to(device=self.device, dtype=self.dtype)  # [num_patches,3,14,14]
        # pixel_values   = [pixel_values]  # wrap in list for generate()

        # gen_kwargs = {
            # "max_new_tokens": 1024,
            # "do_sample": False,
            # "pad_token_id": self.tokenizer.pad_token_id,
            # "eos_token_id": self.tokenizer.eos_token_id,
            # "use_cache": True,
        # }

        # # **Pass input_ids positionally** so Ovis2’s generate() sees it as text_input_ids
        # output_ids = self.model.generate(
            # input_ids,
            # pixel_values=pixel_values,
            # attention_mask=attention_mask,
            # **gen_kwargs
        # )[0]

        # description = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        # return " ".join(line.strip() for line in description.split("\n") if line.strip())


class loader_ovis(BaseLoader):
    def __init__(self, config):
        super().__init__(config)
        native = VISION_MODELS[self.config["vision"]["chosen_model"]]["precision"]
        # Choose dtype on GPU: bfloat16 if supported, else float16; always float32 on CPU
        if self.device == "cuda":
            if native in ("float32", "bfloat16") and has_bfloat16_support():
                self.dtype = torch.bfloat16
                # print(f"OVIS: Selected bfloat16 precision based on native={native}")
            elif native == "float32":
                self.dtype = torch.float16
                # print(f"OVIS: Selected float16 precision based on native={native}")
            else:
                self.dtype = torch.float16
                # print(f"OVIS: Selected float16 precision based on native={native}")
        else:
            self.dtype = torch.float32
            # print(f"OVIS: Selected float32 precision for CPU based on native={native}")
        
        # print(f"OVIS: Device={self.device}, Initial dtype selection={self.dtype}")

    def initialize_model_and_tokenizer(self):
        chosen_model = self.config["vision"]["chosen_model"]
        info = VISION_MODELS[chosen_model]

        cache_dir = CACHE_DIR / info["cache_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"OVIS: Loading model with dtype={self.dtype}")
        
        model = AutoModelForCausalLM.from_pretrained(
            info["repo_id"],
            torch_dtype=self.dtype,
            trust_remote_code=True,
            multimodal_max_length=8192,
            cache_dir=cache_dir,
            token=False
        ).to(self.device)
        
        # # Print model layers precision before eval
        # print("OVIS: Model layer precisions after loading:")
        # for name, module in model.named_modules():
            # if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm)):
                # if hasattr(module, "weight") and module.weight is not None:
                    # print(f"  Layer {name}: {module.weight.dtype}")
        
        model.eval()
        
        # # Print model layers precision after eval
        # print("OVIS: Model layer precisions after eval():")
        # for name, module in model.named_modules():
            # if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LayerNorm)):
                # if hasattr(module, "weight") and module.weight is not None:
                    # print(f"  Layer {name}: {module.weight.dtype}")

        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()

        # # Print visual tokenizer layer info before conversion
        # print("OVIS: Visual tokenizer layer precisions before conversion:")
        # for name, module in visual_tokenizer.named_modules():
            # if isinstance(module, torch.nn.Linear):
                # if hasattr(module, "weight") and module.weight is not None:
                    # print(f"  VT Layer {name}: {module.weight.dtype}")
        
        # # Count modules before conversion
        # linear_count = sum(1 for module in visual_tokenizer.modules() 
                          # if isinstance(module, torch.nn.Linear))
        # print(f"OVIS: Found {linear_count} Linear modules in visual_tokenizer")

        # for module in visual_tokenizer.modules():
            # if isinstance(module, torch.nn.Linear):
                # old_dtype = module.weight.dtype if hasattr(module, "weight") else "unknown"
                # module.to(device=self.device, dtype=self.dtype)
                # new_dtype = module.weight.dtype if hasattr(module, "weight") else "unknown"
                # print(f"OVIS: Converting module from {old_dtype} to {self.dtype}, result={new_dtype}")
        
        # # Print visual tokenizer layer info after conversion
        # print("OVIS: Visual tokenizer layer precisions after conversion:")
        # for name, module in visual_tokenizer.named_modules():
            # if isinstance(module, torch.nn.Linear):
                # if hasattr(module, "weight") and module.weight is not None:
                    # print(f"  VT Layer {name}: {module.weight.dtype}")

        # Save model for process_single_image
        self.model = model
        
        return model, text_tokenizer, visual_tokenizer

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        prompt = (
            "Explain everything you see in this picture "
            "but your response should be no more than one paragraph."
        )
        query = f"<image>\n{prompt}"

        # print("OVIS: Starting image processing")
        _, input_ids, pixel_values = self.model.preprocess_inputs(query, [raw_image])
        # print(f"OVIS: After preprocess_inputs - pixel_values dtype={pixel_values.dtype}")
        
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)

        # Batchify and move to the correct device & dtype
        input_ids = input_ids.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        
        # print(f"OVIS: Before pixel_values conversion - dtype={pixel_values.dtype}")
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        # print(f"OVIS: After pixel_values conversion - dtype={pixel_values.dtype}")
        
        pixel_values = [pixel_values]  # wrap in list for generate()

        # # Check model precision during inference
        # print("OVIS: Model layer precisions during inference:")
        # for name, module in self.model.named_modules():
            # if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                # if hasattr(module, "weight") and module.weight is not None:
                    # if name.startswith("transformer") or name.startswith("lm_head"):
                        # print(f"  Inference layer {name}: {module.weight.dtype}")

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


class loader_internvl2_5(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        info = VISION_MODELS[chosen_model]
        cache_dir = CACHE_DIR / info["cache_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
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
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=False
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            info['repo_id'],
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=False
        )
        my_cprint("InternVL2.5 vision model loaded into memory", "green")
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
        pv = self._prepare_image(raw_image).to(torch.bfloat16).to(self.device)
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
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=[
                "vision_tower",
                "multi_modal_projector",
                "language_model.lm_head"
            ]
        )
        processor = AutoProcessor.from_pretrained(
            model_id,
            use_fast=True,
            cache_dir=cache_dir,
            token=False
        )
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            quantization_config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            token=False
        ).eval()
        my_cprint("Granite Vision model loaded into memory", "green")
        return model, None, processor

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        msg = "Describe in detail what this image depicts but limit your response to one paragraph with no line breaks in it."
        prompt = f"<|user|>\n<image>\n{msg}\n<|assistant|>\n"
        inputs = self.processor(
            images=raw_image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=1
        )
        resp = self.processor.decode(
            output[0],
            skip_special_tokens=True
        ).split('<|assistant|>')[-1].strip()
        return ' '.join(line.strip() for line in resp.split('\n') if line.strip())


class loader_qwenvl(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        model_info = VISION_MODELS[chosen_model]
        model_id = model_info['repo_id']
        save_dir = model_info['cache_dir']
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
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
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir=cache_dir,
            token=False
        )
        model = model.to(self.device)
        model.eval()
        my_cprint("Qwen2.5-VL model loaded into memory", "green")
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
