from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import disable_progress_bars, RepositoryNotFoundError, GatedRepoError
from huggingface_hub.hf_api import RepoFile
from PySide6.QtCore import QObject, Signal
import fnmatch
import humanfriendly
import atexit
import yaml
import functools
from config_manager import ConfigManager

class ModelDownloadedSignal(QObject):
   downloaded = Signal(str, str)

model_downloaded_signal = ModelDownloadedSignal()

MODEL_DIRECTORIES = {
   "vector": "vector",
   "chat": "chat", 
   "tts": "tts",
   "jeeves": "jeeves",
   "ocr": "ocr"
}

@functools.lru_cache(maxsize=1)
def get_hf_token():
   config_path = Path("config.yaml")
   if config_path.exists():
       try:
           with open(config_path, 'r', encoding='utf-8') as config_file:
               config = yaml.safe_load(config_file)
               return config.get('hf_access_token')
       except Exception as e:
           print(f"Warning: Could not load config: {e}")
   return None

class ModelDownloader(QObject):
   def __init__(self, model_info, model_type):
       super().__init__()
       self.model_info = model_info
       self.model_type = model_type
       self._model_directory = None
       
       self.hf_token = get_hf_token()
       self.config = ConfigManager()
       
       self.api = HfApi(token=False)
       self.api.timeout = 60
       disable_progress_bars()
       self.local_dir = self.get_model_directory()

   def cleanup_incomplete_download(self):
       if self.local_dir.exists():
           import shutil
           shutil.rmtree(self.local_dir)

   def get_model_url(self):
       if isinstance(self.model_info, dict):
           return self.model_info['repo_id']
       else:
           return self.model_info

   def check_repo_type(self, repo_id):
       try:
           repo_info = self.api.repo_info(repo_id, timeout=60, token=False)
           if repo_info.private:
               return "private"
           elif getattr(repo_info, 'gated', False):
               return "gated"
           else:
               return "public"
       except Exception as e:
           if self.hf_token and ("401" in str(e) or "Unauthorized" in str(e)):
               try:
                   api_with_token = HfApi(token=self.hf_token)
                   repo_info = api_with_token.repo_info(repo_id, timeout=60)
                   if repo_info.private:
                       return "private"
                   elif getattr(repo_info, 'gated', False):
                       return "gated"
                   else:
                       return "public"
               except Exception as e2:
                   return f"error: {str(e2)}"
           elif "404" in str(e):
               return "not_found"
           else:
               return f"error: {str(e)}"

   def get_model_directory_name(self):
       if isinstance(self.model_info, dict):
           return self.model_info['cache_dir']
       else:
           return self.model_info.replace("/", "--")

   def get_model_directory(self):
       return self.config.models_dir / self.model_type / self.get_model_directory_name()

   def download_model(self, allow_patterns=None, ignore_patterns=None):
       repo_id = self.get_model_url()

       repo_type = self.check_repo_type(repo_id)
       if repo_type not in ["public", "gated"]:
           if repo_type == "private":
               print(f"Repository {repo_id} is private and requires a token.")
               if not self.hf_token:
                   print("No Hugging Face token found. Please add one through the credentials menu.")
               return
           elif repo_type == "not_found":
               print(f"Repository {repo_id} not found. Aborting download.")
               return
           else:
               print(f"Error checking repository {repo_id}: {repo_type}. Aborting download.")
               return

       if repo_type == "gated" and not self.hf_token:
           print(f"Repository {repo_id} is gated. Please add a Hugging Face token and request access through the web interface.")
           return

       local_dir = self.get_model_directory()
       local_dir.mkdir(parents=True, exist_ok=True)

       atexit.register(self.cleanup_incomplete_download)

       try:
           if repo_type == "gated" and self.hf_token:
               api_for_listing = HfApi(token=self.hf_token)
               repo_files = list(api_for_listing.list_repo_tree(repo_id, recursive=True))
           else:
               repo_files = list(self.api.list_repo_tree(repo_id, recursive=True, token=False))
           
           if allow_patterns is not None:
               final_ignore_patterns = None
           elif ignore_patterns is not None:
               final_ignore_patterns = ignore_patterns
           else:
               safetensors_files = [file for file in repo_files if file.path.endswith('.safetensors')]
               bin_files = [file for file in repo_files if file.path.endswith('.bin')]

               final_ignore_patterns = [
                   ".gitattributes",
                   "*.ckpt",
                   "*.gguf",
                   "*.h5", 
                   "*.ot",
                   "*.md",
                   "README*",
                   "onnx/**",
                   "coreml/**",
                   "openvino/**",
                   "demo/**"
               ]

               if safetensors_files and bin_files:
                   final_ignore_patterns.append("*.bin")

               if safetensors_files or bin_files:
                   final_ignore_patterns.append("*consolidated*")

           total_size = 0
           included_files = []
           ignored_files = []

           for file in repo_files:
               if not isinstance(file, RepoFile):
                   continue

               should_include = True

               if allow_patterns is not None:
                   should_include = any(fnmatch.fnmatch(file.path, pattern) for pattern in allow_patterns)
               elif final_ignore_patterns is not None:
                   should_include = not any(fnmatch.fnmatch(file.path, pattern) for pattern in final_ignore_patterns)

               if should_include:
                   total_size += file.size
                   included_files.append(file.path)
               else:
                   ignored_files.append(file.path)

           readable_total_size = humanfriendly.format_size(total_size)
           print(f"\nTotal size to be downloaded: {readable_total_size}")
           print("\nFiles to be downloaded:")
           for file in included_files:
               print(f"- {file}")
           print(f"\nDownloading to {local_dir}...")

           download_kwargs = {
               'repo_id': repo_id,
               'local_dir': str(local_dir),
               'max_workers': 4,
               'ignore_patterns': final_ignore_patterns,
               'allow_patterns': allow_patterns,
               'etag_timeout': 60
           }
           
           if repo_type == "gated" and self.hf_token:
               download_kwargs['token'] = self.hf_token
           elif repo_type == "public":
               download_kwargs['token'] = False

           snapshot_download(**download_kwargs)

           print("\033[92mModel downloaded and ready to use.\033[0m")
           atexit.unregister(self.cleanup_incomplete_download)
           model_downloaded_signal.downloaded.emit(self.get_model_directory_name(), self.model_type)

       except Exception as e:
           print(f"An error occurred during download: {str(e)}")
           if local_dir.exists():
               import shutil
               shutil.rmtree(local_dir)

"""
Needs to be doublechecked...

+----------------+------------------+------------------------+---------------+
| Source         | Parameters       | Behavior             | Repo Structure? |
+----------------+------------------+------------------------+---------------+
| "huggingface"  | (default)        | ~/.cache/huggingface | No              |
| "huggingface"  | cache_dir="path" | Downloads to path    | No              |
| "huggingface"  | local_dir="path" | Downloads to path    | Yes             |
+----------------+------------------+----------------------+-----------------+
| "local"        | (default)        | Current directory    | No              |
| "local"        | cache_dir="path" | Downloads to path    | No              |
| "local"        | local_dir="path" | Downloads to path    | Yes             |
+----------------+------------------+----------------------+-----------------+
| "custom"       | custom_path      | Uses existing files  | n/a             |
| "custom"       | +cache_dir       | cache_dir ignored    | n/a             |
| "custom"       | +local_dir       | local_dir ignored    | n/a             |
+----------------+------------------+----------------------+-----------------+

1. `local_dir` takes precedence over everything else:
```python
chat.load(source="huggingface", local_dir="path/to/dir", cache_dir="path/to/cache")  # local_dir wins
```

2. If `local_dir` is not specified but `cache_dir` is, cache_dir is used:
```python
chat.load(source="huggingface", cache_dir="path/to/cache")  # cache_dir used
```

3. If neither is specified, the default location is used:
```python
chat.load(source="huggingface")  # Uses ~/.cache/huggingface
chat.load(source="local")        # Uses current directory
```
"""