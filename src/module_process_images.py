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
    AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoProcessor, BlipForConditionalGeneration, BlipProcessor,
    LlamaTokenizer, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration, LlavaNextProcessor, BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, GenerationConfig, Autoconfig
)

from langchain_community.docstore.document import Document

from extract_metadata import extract_image_metadata
from utilities import my_cprint
from constants import VISION_MODELS

warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

ALLOWED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff']

current_directory = Path(__file__).parent
CACHE_DIR = current_directory / "models" / "vision"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

current_directory = Path(__file__).parent
VISION_DIR = current_directory / "models" / "vision"
VISION_DIR.mkdir(parents=True, exist_ok=True)

def get_best_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def check_for_images(image_dir):
    return any(
        Path(file).suffix.lower() in ALLOWED_EXTENSIONS
        for file in Path(image_dir).iterdir()
    )

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
    elif chosen_model == 'MiniCPM-V-2_6 - 8b':
        loader_func = loader_minicpm_V_2_6(config).process_images
    elif chosen_model == 'Granite Vision - 2b':
        loader_func = loader_granite(config).process_images
    elif chosen_model in ['Llava 1.6 Vicuna - 13b']:
        loader_func = loader_llava_next(config).process_images
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

    script_dir = Path(__file__).parent
    image_dir = script_dir / "Docs_for_DB"

    if not check_for_images(image_dir):
        return []

    with ProcessPoolExecutor(1) as executor:
        future = executor.submit(run_loader_in_process, loader_func)
        try:
            processed_docs = future.result()
        except Exception as e:
            my_cprint(f"Error occurred during image processing: {e}", "red")
            return []

        if processed_docs is None:
            return []
        return processed_docs


class BaseLoader:
    def __init__(self, config):
        self.config = config
        self.device = get_best_device()
        self.model = None
        self.tokenizer = None
        self.processor = None

    def initialize_model_and_tokenizer(self):
        raise NotImplementedError("Subclasses must implement initialize_model_and_tokenizer method")

    def process_images(self):
        script_dir = Path(__file__).parent
        image_dir = script_dir / "Docs_for_DB"
        documents = []
        allowed_extensions = ALLOWED_EXTENSIONS

        image_files = [file for file in image_dir.iterdir() if file.suffix.lower() in allowed_extensions]

        self.model, self.tokenizer, self.processor = self.initialize_model_and_tokenizer()

        print("Processing images...")
        
        total_start_time = time.time()

        with tqdm(total=len(image_files), unit="image") as progress_bar:
            for file_name in image_files:
                full_path = image_dir / file_name
                try:
                    with Image.open(full_path) as raw_image:
                        extracted_text = self.process_single_image(raw_image)
                        extracted_metadata = extract_image_metadata(full_path)
                        document = Document(page_content=extracted_text, metadata=extracted_metadata)
                        documents.append(document)
                        progress_bar.update(1)
                except Exception as e:
                    print(f"{file_name}: Error processing image - {e}")

        total_end_time = time.time()
        total_time_taken = total_end_time - total_start_time
        print(f"Loaded {len(documents)} image(s)...")
        print(f"Total image processing time: {total_time_taken:.2f} seconds")

        my_cprint("Vision model removed from memory.", "red")

        return documents

    def process_single_image(self, raw_image):
        raise NotImplementedError("Subclasses must implement.")


class loader_llava_next(BaseLoader):
    def initialize_model_and_tokenizer(self):
        from transformers.image_utils import PILImageResampling
        chosen_model = self.config['vision']['chosen_model']

        model_info = VISION_MODELS[chosen_model]
        model_id = model_info['repo_id']
        save_dir = model_info["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir
        )
        model.eval()

        my_cprint(f"{chosen_model} vision model loaded into memory...", "green")

        processor = LlavaNextProcessor.from_pretrained(
            model_id, 
            cache_dir=cache_dir,
            images_kwargs={
                "size": {"shortest_edge": 672},  # must be multiple of 336
                "image_grid_pinpoints": [
                    [672, 672],
                    [336, 1344],
                    [1344, 336],
                    [672, 336],
                    [336, 672]
                ],
                "resample": PILImageResampling.LANCZOS,
                "do_pad": True
            }
        )

        return model, None, processor

    @ torch.inference_mode()
    def process_single_image(self, raw_image):
        user_prompt = "Explain everything you see in this picture but your response should be no more than one paragraph, but the paragraph can be as long as you want."
        prompt = f"USER: <image>\n{user_prompt} ASSISTANT:"
        inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
        inputs = inputs.to(self.device)

        output = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)

        response = self.processor.decode(output[0], skip_special_tokens=True)
        model_response = response.split("ASSISTANT:")[-1].strip()
        model_response = ' '.join(line.strip() for line in model_response.split('\n') if line.strip())

        return model_response


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

        model = AutoModelForCausalLM.from_pretrained(
            repo_id, 
            trust_remote_code=True, 
            low_cpu_mem_usage=True, 
            cache_dir=cache_dir
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(
            repo_id, 
            trust_remote_code=True, 
            cache_dir=cache_dir
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            if torch.cuda.get_device_capability()[0] >= 8:
                # For Ampere or newer GPUs (compute capability >= 8)
                model = model.to(self.device).to(torch.bfloat16)
                self.precision_type = "bfloat16"
                self.model_dtype = torch.bfloat16
            else:
                # For older CUDA GPUs
                model = model.to(self.device).half()
                self.precision_type = "float16"
                self.model_dtype = torch.float16
        else:
            # For CPU
            model = model.to(self.device).float()
            self.precision_type = "float32"
            self.model_dtype = torch.float32

        device_type_display = "CUDA" if self.device.type == "cuda" else "CPU"
        self.my_cprint(f"Running {chosen_model} on {device_type_display} in {self.precision_type} precision", color="green")

        self.model = model
        self.processor = processor
        return model, None, processor

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model_dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=1,
            do_sample=False,
            early_stopping=False,
            top_p=None,
            top_k=None,
            temperature=None,
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=prompt, image_size=(raw_image.width, raw_image.height))

        return parsed_answer['<MORE_DETAILED_CAPTION>']


class loader_glmv4(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        model_info = VISION_MODELS[chosen_model]
        model_id = model_info['repo_id']
        save_dir = model_info["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda")
        use_bfloat16 = torch.cuda.get_device_capability()[0] >= 8

        if use_bfloat16:
            compute_dtype = torch.bfloat16
            self.precision_type = "bfloat16"
        else:
            compute_dtype = torch.float16
            self.precision_type = "float16"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

        model_config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        model_config.vision_config['image_size'] = 448

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            quantization_config=quantization_config,
            cache_dir=cache_dir
        )

        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        my_cprint(f"Running GLM4V-9B on CUDA in {self.precision_type} precision", "green")

        return model, tokenizer, None

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        query = "Describe this image in as much detail as possible but do not repeat yourself."

        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "image": raw_image, "content": query}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        )
        inputs = inputs.to(self.device)
        gen_kwargs = {
            "max_length": 1024,
            "do_sample": False,
            "top_k": None,
            "top_p": None,
            "temperature": None
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        description = ' '.join(line.strip() for line in description.split('\n') if line.strip())

        return description


class loader_molmo(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        model_info = VISION_MODELS[chosen_model]
        model_path = model_info.get('model_path')
        model_id = model_info.get('repo_id')
        if model_path:
            model_source = model_path
        else:
            model_source = model_id
        cache_dir = CACHE_DIR / model_info.get('cache_dir', '')
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_source,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                cache_dir=cache_dir
            )

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_source,
                trust_remote_code=True,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                cache_dir=cache_dir
            )

            # convert vision backbone to float32
            self.model.model.vision_backbone = self.model.model.vision_backbone.to(torch.float32)

            self.model.eval()
            my_cprint(f"{chosen_model} vision model loaded into memory...", "green")
        except Exception as e:
            my_cprint(f"Error loading {chosen_model} model: {str(e)}", "red")
            raise

        return self.model, None, self.processor

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        if raw_image.mode != "RGB":
            raw_image = raw_image.convert("RGB")

        user_prompt = "Describe this image in as much detail as possible but do not repeat yourself."

        inputs = self.processor.process(images=[raw_image], text=user_prompt)

        inputs = {
            k: (v.to(device=self.device, dtype=torch.long) if k in ['input_ids', 'image_input_idx'] else 
               v.to(device=self.device, dtype=torch.float32)).unsqueeze(0)
            for k, v in inputs.items()
        }

        try:
            generation_config = GenerationConfig(
                max_new_tokens=1024,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )

            output = self.model.generate_from_batch(
                inputs,
                generation_config,
                tokenizer=self.processor.tokenizer
            )

            generated_tokens = output[0, inputs['input_ids'].size(1):]
            generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_text = ' '.join(line.strip() for line in generated_text.split('\n') if line.strip())

        except Exception as e:
            my_cprint(f"Error processing image: {str(e)}", "red")
            return ""

        return generated_text


class loader_ovis(BaseLoader):
    def __init__(self, config):
        super().__init__(config)

    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        model_info = VISION_MODELS[chosen_model]
        repo_id = model_info['repo_id']
        save_dir = model_info["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        config = AutoConfig.from_pretrained(
            repo_id, 
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            multimodal_max_length=8192,
            cache_dir=cache_dir
        ).eval().cuda()

        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()

        return model, text_tokenizer, visual_tokenizer

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        text = "Explain everything you see in this picture but your response should be no more than one paragraph, but the paragraph can be as long as you want."
        query = f'<image>\n{text}'

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, [raw_image])
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)

        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        pixel_values = [pixel_values.to(dtype=self.processor.dtype, device=self.processor.device)]

        gen_kwargs = {
            'max_new_tokens': 1024,
            'do_sample': False,
            'top_p': None,
            'top_k': None,
            'temperature': None,
            'repetition_penalty': 1.0,
            'eos_token_id': self.model.generation_config.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'use_cache': True
        }

        output_ids = self.model.generate(
            input_ids, 
            pixel_values=pixel_values, 
            attention_mask=attention_mask, 
            **gen_kwargs
        )[0]

        output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        output = ' '.join(line.strip() for line in output.split('\n') if line.strip())

        return output


class loader_internvl2_5(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        model_info = VISION_MODELS[chosen_model]
        model_id = model_info['repo_id']
        save_dir = model_info["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=[
                "vision_model", # big impact
                "language_model.model.norm", # low impact
                "language_model.output", # low impact
                "language_model.model.rotary_emb", # low impact
                "language_model.lm_head", # medium impact
                "mlp1", # big impact
            ]
        )

        model = AutoModel.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        my_cprint(f"InternVL2.5 {model_info['size']} vision model loaded into memory...", "green")

        return model, tokenizer, None

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _build_transform(self, input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), 
                interpolation=InterpolationMode.LANCZOS,
                antialias=True),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) 
            for i in range(1, n + 1) 
            for j in range(1, n + 1) 
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def _prepare_image(self, raw_image, input_size=448, max_num=24):
        transform = self._build_transform(input_size=input_size)
        images = self.dynamic_preprocess(raw_image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        pixel_values = self._prepare_image(raw_image).to(torch.bfloat16).to(self.device)
        
        question = '<image>\nExplain everything you see in this picture but your response should be no more than one paragraph, but the paragraph can be as long as you want.'
        
        generation_config = dict(
            num_beams=1,
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config
        )
        response = ' '.join(line.strip() for line in response.split('\n') if line.strip())

        return response


class loader_granite(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        model_info = VISION_MODELS[chosen_model]
        model_id = model_info['repo_id']
        save_dir = model_info["cache_dir"]
        cache_dir = CACHE_DIR / save_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=[
                "vision_tower",
                "multi_modal_projector",
                "language_model.lm_head",
            ]
        )

        processor = AutoProcessor.from_pretrained(
            model_id,
            use_fast=True,
            cache_dir=cache_dir
        )

        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir
        ).eval()

        my_cprint(f"Granite Vision 3.2-2b model loaded into memory...", "green")

        return model, None, processor

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        user_message = "Describe in detail what this image depicts but limit your response to one paragraph with no line breaks in it."

        prompt = f"""<|user|>
<image>
{user_message}
<|assistant|>
"""

        inputs = self.processor(
            images=raw_image,
            text=prompt,
            return_tensors="pt"
        )

        inputs = inputs.to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=1,
        )

        response = self.processor.decode(output[0], skip_special_tokens=True)
        response = response.split('<|assistant|>')[-1].strip()
        response = ' '.join(line.strip() for line in response.split('\n') if line.strip())
        
        return response


class loader_qwenvl(BaseLoader):
    def initialize_model_and_tokenizer(self):
        chosen_model = self.config['vision']['chosen_model']
        model_info = VISION_MODELS[chosen_model]
        model_id = model_info['repo_id']
        precision = model_info['precision']
        save_dir = model_info["cache_dir"]
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
            ],
        )

        processor = AutoProcessor.from_pretrained(
            model_id,
            use_fast=True,
            min_pixels=28*28,
            max_pixels=1280*28*28,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir=cache_dir
        )

        model = model.to(self.device)
        model.eval()

        my_cprint(f"Qwen2.5-VL model loaded into memory...", "green")

        return model, None, processor

    @torch.inference_mode()
    def process_single_image(self, raw_image):
        user_message = "Describe in as much detail as possible what this image depicts?"

        prompt = f"""<|im_start|>user
{user_message} <|vis_start|><|image_pad|><|vis_end|>
<|im_end|>
<|im_start|>assistant
"""

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
            temperature=None,
        )

        response = self.processor.decode(output[0], skip_special_tokens=True)
        response = response.split('assistant')[-1].strip()
        response = ' '.join(line.strip() for line in response.split('\n') if line.strip())
        
        return response


"""
* need to update
+----------------------+-----------------------------------------+-----------+----------------+----------+
| Sub-Class            | Config Details                          | Attention | Precision      | Device   |
+----------------------+-----------------------------------------+-----------+----------------+----------+
| loader_llava_next    | do_sample=False, no temperature control | SDPA      | float16 4-bit  | CUDA     |
| loader_florence2     | Comprehensive beam settings, no         | SDPA      | autoselect     | CPU/CUDA |
|                      | sampling                                |           |                |          |
| loader_glmv4         | do_sample=False, no temperature         | SDPA      | bfloat16 4-bit | CUDA     |
| loader_molmo         | Uses GenerationConfig class             | SDPA      | float32 4-bit  | CUDA     |
| loader_ovis          | repetition_penalty=1.0, use_cache=True  | SDPA      | autoselect     | CUDA     |
+----------------------+-----------------------------------------+-----------+----------------+----------+
"""