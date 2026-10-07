"""
Dataset download script for Privacy-Preserving Federated Symptom Checker.
Downloads and extracts:
1. Kaggle Disease-Symptom-Description
2. HAM10000 (Skin Lesions)
3. ICBHI 2017 (Respiratory Sounds)
"""

import os
import zipfile
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm
import logging
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url: str, output_path: Path):
    """Download a file from URL with a progress bar."""
    if output_path.exists():
        logger.info(f"File {output_path} already exists. Skipping download.")
        return
    logger.info(f"Downloading from {url} to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_archive(archive_path: Path, extract_dir: Path):
    """Extract zip or tar archive."""
    logger.info(f"Extracting {archive_path} to {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    if str(archive_path).endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif str(archive_path).endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        logger.warning(f"Unknown archive format for {archive_path}")

def download_kaggle_dataset(dataset_name: str, extract_dir: Path):
    """Download Kaggle dataset using kaggle CLI."""
    try:
        import kaggle
    except (ImportError, OSError):
        logger.error("Kaggle API not configured. Please install kaggle library and set up ~/.kaggle/kaggle.json")
        logger.error(f"Fallback: Manually download from https://kaggle.com/datasets/{dataset_name} and extract to {extract_dir}")
        return

    logger.info(f"Downloading Kaggle dataset {dataset_name} to {extract_dir}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(extract_dir), "--unzip"], check=True)
        logger.info(f"Successfully downloaded and extracted {dataset_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download Kaggle dataset {dataset_name}: {e}")

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Kaggle Disease-Symptom-Description
    disease_dir = RAW_DIR / "disease_symptom"
    download_kaggle_dataset("itachi9604/disease-symptom-description-dataset", disease_dir)

    # 2. HAM10000
    ham_dir = RAW_DIR / "ham10000"
    ham_dir.mkdir(parents=True, exist_ok=True)
    download_kaggle_dataset("kmader/skin-cancer-mnist-ham10000", ham_dir)

    # 3. ICBHI 2017 Respiratory Sound Database
    icbhi_dir = RAW_DIR / "icbhi"
    icbhi_dir.mkdir(parents=True, exist_ok=True)
    download_kaggle_dataset("vbookshelf/respiratory-sound-database", icbhi_dir)

if __name__ == '__main__':
    main()
