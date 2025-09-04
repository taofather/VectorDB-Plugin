import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Union, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {
    ".txt",
    ".eml",
    ".msg",
    ".csv",
    ".yaml",
    ".md",
    ".yml",
}

def _create_single_symlink(args):
    source_path, target_dir, source_root = args
    logger.debug(f"Processing file: {source_path}")
    try:
        source_path_obj = Path(source_path)
        source_root_obj = Path(source_root) if source_root else source_path_obj.parent

        # Create a relative path from source root to preserve directory structure partially
        try:
            relative_path = source_path_obj.relative_to(source_root_obj)
            # Replace directory separators with underscores to create a flat but unique name
            unique_name = str(relative_path).replace('/', '_').replace('\\', '_')
            logger.debug(f"Using relative path unique name: {unique_name}")
        except ValueError:
            # If relative path fails, create a unique name with parent directory
            unique_name = f"{source_path_obj.parent.name}_{source_path_obj.name}"
            logger.debug(f"Using parent+name unique name: {unique_name}")

        link_path = Path(target_dir) / unique_name
        logger.debug(f"Target symlink path: {link_path}")

        if not link_path.exists():
            link_path.symlink_to(source_path)
            logger.info(f"Created symlink: {unique_name} -> {source_path}")
            return True, None
        else:
            logger.warning(f"Symlink already exists: {link_path}")
    except Exception as e:
        error_msg = f"Error creating symlink for {Path(source_path).name}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
    return False, None

def create_symlinks_parallel(source: Union[str, Path, List[str], List[Path]], 
                           target_dir: Union[str, Path] = "Docs_for_DB") -> Tuple[int, list]:
    """
    Create symbolic links using multiprocessing if the number of files exceeds 500.

    Args:
        source: Can be either:
            - str or Path: Path to the source directory
            - List[str] or List[Path]: List of file paths
        target_dir: Path to the directory to store symlinks (default: 'Docs_for_DB')

    Returns:
        tuple: (number of links created, list of errors)
    """
    target_dir = Path(target_dir)
    if not target_dir.exists():
        print(f"Target directory does not exist: {target_dir}")
        return 0, []

    try:
        if isinstance(source, (str, Path)) and not isinstance(source, list):
            source_dir = Path(source)
            if not source_dir.exists():
                raise ValueError(f"Source directory does not exist: {source_dir}")

            logger.info(f"Starting recursive search in directory: {source_dir}")
            logger.info(f"Directory exists: {source_dir.exists()}")
            logger.info(f"Directory is readable: {source_dir.is_dir()}")

            # First, let's see what's in the root directory
            root_items = list(source_dir.iterdir()) if source_dir.exists() else []
            logger.info(f"Items in root directory: {len(root_items)}")
            for item in root_items[:10]:  # Log first 10 items
                if item.is_dir():
                    logger.debug(f"Directory: {item.name}")
                else:
                    logger.debug(f"File: {item.name} (extension: {item.suffix})")

            # Now try rglob
            all_files = list(source_dir.rglob("*"))
            logger.info(f"Found {len(all_files)} total items with rglob")

            # Show directory structure
            directories = [p for p in all_files if p.is_dir()]
            logger.info(f"Found {len(directories)} directories")
            for i, d in enumerate(directories[:5]):
                logger.debug(f"Directory {i+1}: {d}")

            only_files = [p for p in all_files if p.is_file()]
            logger.info(f"Found {len(only_files)} files (excluding directories)")

            # Show all files found
            logger.debug("All files found:")
            for i, f in enumerate(only_files[:10]):
                logger.debug(f"File {i+1}: {f} (extension: {f.suffix})")
            if len(only_files) > 10:
                logger.debug(f"... and {len(only_files)-10} more files")

            # Log file extensions found
            extensions_found = {}
            for f in only_files:
                ext = f.suffix.lower()
                extensions_found[ext] = extensions_found.get(ext, 0) + 1
            logger.info(f"File extensions found: {extensions_found}")

            files = [(str(p), str(target_dir), str(source_dir)) for p in only_files 
                    if p.suffix.lower() in ALLOWED_EXTENSIONS]
            logger.info(f"Found {len(files)} files with allowed extensions in directory: {source_dir}")

            # Log some examples of excluded files
            excluded_files = [p for p in only_files if p.suffix.lower() not in ALLOWED_EXTENSIONS]
            if excluded_files:
                logger.info(f"Excluded {len(excluded_files)} files with non-allowed extensions")
                for i, excluded in enumerate(excluded_files[:3]):
                    logger.debug(f"Excluded file {i+1}: {excluded} (extension: {excluded.suffix})")
                if len(excluded_files) > 3:
                    logger.debug(f"... and {len(excluded_files)-3} more excluded files")

            # Log first few files for debugging
            for i, (file_path, _, _) in enumerate(files[:5]):
                logger.debug(f"File {i+1}: {file_path}")
            if len(files) > 5:
                logger.debug(f"... and {len(files)-5} more files")

        elif isinstance(source, list):
            files = [(str(Path(p)), str(target_dir), None) for p in source]
            logger.info(f"Processing {len(files)} individual files from list")

        else:
            raise ValueError("Source must be either a directory path or a list of file paths")

        file_count = len(files)
        if file_count <= 1000:
            # For 1000 or fewer files, don't use multiprocessing
            results = [_create_single_symlink(file) for file in files]
        else:
            # For 501-10000 files, use single process
            # For >10000 files, scale up processes
            if file_count <= 10000:
                processes = 1
            else:
                processes = min((file_count // 10000) + 1, cpu_count())

            print(f"Processing {file_count} files using {processes} processes")

            with Pool(processes=processes) as pool:
                results = pool.map(_create_single_symlink, files)

        count = sum(1 for success, _ in results if success)
        errors = [error for _, error in results if error is not None]
        
        print(f"\nComplete! Created {count} symbolic links")
        if errors:
            print("\nErrors occurred:")
            for error in errors:
                print(error)

        return count, errors

    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}")