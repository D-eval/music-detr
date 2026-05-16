"""
baby
detr6

多类别事件

nohup python3 train6_stage1.py > train6_stage1.log 2>&1 &
"""
import warnings
warnings.simplefilter("ignore")

def to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {
            k: to_device(v, device)
            for k, v in batch.items()
        }
    elif isinstance(batch, list):
        return [to_device(x, device) for x in batch]
    else:
        return batch


import torch
import torch.nn as nn
import torch.optim as optim

from utils.trainRecorder import TrainingRecorder
from utils.visualizer import plot_pianoroll_event

import os
from configs.config6 import get_config
cfg = get_config()

import sys
sys.path.append(str(cfg.dataset_read_py_path_stage1_ytb))

from read import StackDataset, collate_fn
from torch.utils.data import DataLoader
# dataset = RandomChordSynthDataset(prototype_dir=cfg.dataset_read_py_path_stage1 / "prototype",
#                                   soundfont_dir=cfg.dataset_read_py_path_stage1 / "soundfonts",
#                                   sample_rate=cfg.sr,
#                                   min_midi=cfg.min_midi,
#                                   max_midi=cfg.max_midi)

dataset = StackDataset(cfg.sr, cfg.min_midi, cfg.max_midi)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)


from models.detr6 import PitchDetr
from spec import wav2cqt_2C, wav2spec_2C
from spec.cqt import MultiWindowCQT, get_freqs
from models.teacher import Teacher
from utils.equipTarget import get_target_map, get_sustain_map, get_sustain_map_textwise, normalize_targets_pitch, render_pred_pitch_map, render_pred_group_pitch_map, embed_text

freqs = get_freqs(cfg.min_midi, cfg.max_midi)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

preprocessor = MultiWindowCQT(freqs, cfg.sr, cfg.window_num, cfg.min_cycle, stride=cfg.cqt_stride).to(device)
model = PitchDetr(cfg).to(device)

# for audio, target in loader:
#     x, _, freqs = preprocessor(audio.to(device))
#     assert 0


# teacher = Teacher()

current_state = model.state_dict()

checkpoint_path = "../params/detr6/pupil1.pt"
state_dict = torch.load(checkpoint_path)

# 过滤不匹配的参数
filtered_state = {}
for k, v in state_dict.items():
    if k not in current_state:
        continue
    if v.shape != current_state[k].shape:
        print("skip shape mismatch:", k)
        continue
    filtered_state[k] = v

# 装载
model.load_state_dict(state_dict=filtered_state, strict=False)

# -------- optimizer --------
optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
recorder = TrainingRecorder(cfg, "pupil1")
recorder.load()
# -------- 混合精度（强烈建议）--------
scaler = torch.cuda.amp.GradScaler()

# -------- 训练 --------

model.eval()

total_tp = 0
total_fp = 0
total_fn = 0

total_exist_correct = 0
total_exist = 0

for i,batch in enumerate(loader):
    audio, targets = batch

    audio = audio.to(device)

    outputs = model(audio)

    b = 0

    infer_output = model.infer(
        outputs[b][None,...],
        threshold=0.5
    )

    pred_midis = infer_output['midi'].tolist()

    gt_exist = bool(targets[b]['exist'].item())

    if not gt_exist:
        gt_midis = []
    else:
        gt_midis = targets[b]['midi']

    pred_set = set(pred_midis)
    gt_set = set(gt_midis)

    # -------------------
    # exist accuracy
    # -------------------

    pred_exist = len(pred_set) > 0

    if pred_exist == gt_exist:
        total_exist_correct += 1

    total_exist += 1

    # -------------------
    # note metrics
    # -------------------

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    total_tp += tp
    total_fp += fp
    total_fn += fn

# =========================
# metrics
# =========================

precision = total_tp / (total_tp + total_fp + 1e-8)

recall = total_tp / (total_tp + total_fn + 1e-8)

f1 = 2 * precision * recall / (
    precision + recall + 1e-8
)

exist_acc = total_exist_correct / total_exist

print("\n========== Evaluation ==========")

print(f"exist acc: {exist_acc:.4f}")

print(f"precision: {precision:.4f}")
print(f"recall:    {recall:.4f}")
print(f"f1:        {f1:.4f}")

print(f"TP: {total_tp}")
print(f"FP: {total_fp}")
print(f"FN: {total_fn}")


# loss = model.get_loss(audio, target)

# with torch.amp.autocast("cuda"):
#     loss = model.get_loss(audio, target)
# with torch.amp.autocast("cuda"):
#     loss = model.get_loss(audio, target)


