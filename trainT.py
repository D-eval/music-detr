"""
用来训练ALUnion
"""

import torch
import torch.nn as nn
import torch.optim as optim
from utils.visualizer import compare_result_3

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

from models.tokenizer import build_corpus, train_bpe
from spec import wav2cqt, wav2spec
from models.teacher import Teacher
from utils.equipTarget import get_target_map, get_sustain_map, get_sustain_map_textwise, normalize_targets_pitch, render_pred_pitch_map, render_pred_group_pitch_map, to_device, embed_text


corpus = build_corpus(dataset)
train_bpe(corpus)

from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(cfg.tokenizer.save_path)

all_len = []
for c in corpus:
    out = tokenizer.encode(c)
    all_len.append(len(out.ids))

