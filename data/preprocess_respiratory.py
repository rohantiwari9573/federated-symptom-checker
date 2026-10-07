"""
Preprocess the ICBHI 2017 Respiratory Sound Database.
Extracts Mel spectrograms using librosa and defines PyTorch Dataset class.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import librosa
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "icbhi"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "respiratory"

class RespiratoryDataset(Dataset):
    """PyTorch Dataset for Respiratory Sounds."""
    
    def __init__(self, data_dir: Path, target_sr=22050, duration=3.0, n_mels=128):
        self.data_dir = data_dir
        self.target_sr = target_sr
        self.duration = duration
        self.n_mels = n_mels
        
        self.audio_files = list(self.data_dir.rglob("*.wav"))
        
    def _get_label(self, txt_path: Path):
        """Parse annotations to determine label: Normal(0), Crackle(1), Wheeze(2), Both(3)"""
        if not txt_path.exists():
            return 0
        df = pd.read_csv(txt_path, sep='\t', header=None, names=['start', 'end', 'crackle', 'wheeze'])
        has_crackle = df['crackle'].sum() > 0
        has_wheeze = df['wheeze'].sum() > 0
        
        if has_crackle and has_wheeze: return 3
        elif has_wheeze: return 2
        elif has_crackle: return 1
        return 0

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        wav_path = self.audio_files[idx]
        txt_path = wav_path.with_suffix('.txt')
        label = self._get_label(txt_path)
        
        # Load audio
        y, sr = librosa.load(wav_path, sr=self.target_sr)
        
        # Pad or truncate to fixed duration
        max_len = int(self.target_sr * self.duration)
        if len(y) > max_len:
            y = y[:max_len]
        else:
            y = np.pad(y, (0, max_len - len(y)))
            
        # Mel spectrogram
        melspec = librosa.feature.melspectrogram(y=y, sr=self.target_sr, n_mels=self.n_mels, hop_length=max_len//128 + 1)
        melspec_db = librosa.power_to_db(melspec, ref=np.max)
        
        # Resize to exactly 128x128 if needed by slicing/padding
        if melspec_db.shape[1] > 128:
            melspec_db = melspec_db[:, :128]
        else:
            melspec_db = np.pad(melspec_db, ((0,0), (0, 128 - melspec_db.shape[1])))
            
        # Add channel dimension
        melspec_tensor = torch.tensor(melspec_db, dtype=torch.float32).unsqueeze(0)
        
        return melspec_tensor, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    if not RAW_DIR.exists() or len(list(RAW_DIR.rglob("*.wav"))) == 0:
        logger.warning(f"ICBHI data not found at {RAW_DIR}. Creating synthetic benchmark audio dataset...")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        
        target_sr = 22050
        duration = 3.0
        for i in range(40):
            wav_path = RAW_DIR / f"audio_{i:03d}.wav"
            txt_path = RAW_DIR / f"audio_{i:03d}.txt"
            
            # Generate synthetic noise audio
            audio = np.random.randn(int(target_sr * duration)).astype(np.float32)
            sf.write(wav_path, audio, target_sr)
            
            # Write txt label annotation
            crackle = 1 if i % 4 in [1, 3] else 0
            wheeze = 1 if i % 4 in [2, 3] else 0
            with open(txt_path, 'w') as f:
                f.write(f"0.0\t3.0\t{crackle}\t{wheeze}\n")
        logger.info(f"Created 40 synthetic audio records at {RAW_DIR}")
        
    dataset = RespiratoryDataset(RAW_DIR)
    logger.info(f"Initialized RespiratoryDataset with {len(dataset)} items.")
    if len(dataset) > 0:
        spec, lbl = dataset[0]
        logger.info(f"Sample spec shape: {spec.shape}, label: {lbl}")
