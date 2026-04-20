import torch
import torch.nn as nn
import torch.optim as optim
from utils.visualizer import compare_result, export_event_flash_mp4
import math
import torch.nn.functional as F

import os
from configs.config import get_config
cfg = get_config()

import sys
sys.path.append(str(cfg.dataset_read_py_path))

from read13 import AudioDataset, collate_fn
from torch.utils.data import DataLoader
dataset = AudioDataset(root_dir=cfg.dataset_data_path,
                       min_pitch=cfg.min_midi,
                       max_pitch=cfg.max_midi)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    # num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)

from models.detr2 import PitchTransformer
from spec import wav2cqt, wav2spec
from utils.equipTarget import get_target_map, to_device, embed_text

# tokenizer = MusicDetrTokenizer()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

# ---------- target ----------
# target_pitchMap = get_target_map([events], pitch_centre)


for step, batch in enumerate(loader):
    audio, target = batch
    break
