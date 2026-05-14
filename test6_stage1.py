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

dataset = StackDataset(cfg.sr)
assert 0
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)

from models.detr6 import CQTEncoder
from spec import wav2cqt_2C, wav2spec_2C
from spec.cqt import MultiWindowCQT, get_freqs
from models.teacher import Teacher
from utils.equipTarget import get_target_map, get_sustain_map, get_sustain_map_textwise, normalize_targets_pitch, render_pred_pitch_map, render_pred_group_pitch_map, embed_text

freqs = get_freqs(cfg.min_midi, cfg.max_midi)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

preprocessor = MultiWindowCQT(freqs, cfg.sr, cfg.window_num, cfg.min_cycle, stride=cfg.cqt_stride).to(device)
model = CQTEncoder(cfg).to(device)

checkpoint_path = "../params/detr6/baby4.pt"
state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict=state_dict, strict=False)


# for step, batch in enumerate(loader):
#     audio, target = batch
#     audio = audio.to(device)
#     target = to_device(target, device)
#     with torch.no_grad():
#         audio = audio[0,...][None,...]
#         x, _, freqs = preprocessor(audio.to(device))
#         model.eval()
#         output = model(x)
#         infer_output = model.infer(output)
#         model.train()
#         # infer_output, target : Dict{
#         #     "root": root_pred, # (M) 0~11
#         #     "chord": chord_pred, # (M, 12)
#         #     "tonic": tonic_pred, # (M) 0~11
#         #     "start": start_pred, # (M)
#         #     "sustain": sustain_pred, # (M)
#         #     "exist": exist_pred, # (M)
#         # }
#         target = to_device(target, torch.device("cpu"))
#         infer_output = to_device(infer_output, torch.device("cpu"))
#         # plot_pianoroll_event(infer_output, target[0])
        
#         symbol_gt = str(target[0]['symbol']) # eg: C:maj...
#         symbol_pred = str(infer_output['symbol'])
        
#         if ":" in symbol_gt: # 不是 N
#             root_gt = target[0]['root_idx']
#             root_pred = infer_output['root_name']
            
#             quality_gt = target[0]['chord_idx']
#             quality_pred = infer_output['chord_name']

from collections import Counter, defaultdict
import csv
import os
import torch

def norm_scalar(x):
    if torch.is_tensor(x):
        return x.detach().cpu().view(-1)[0].item()
    return x

def parse_symbol(symbol):
    """
    return:
    {
        "exist": 0/1,
        "root": str or None,
        "quality": str or None,
        "bass": str or None,
        "symbol": str
    }
    """
    symbol = str(symbol)

    if ":" not in symbol:
        return {
            "exist": 0,
            "root": None,
            "quality": None,
            "bass": None,
            "symbol": symbol,
        }

    if "/" in symbol:
        chord_part, bass = symbol.split("/")
    else:
        chord_part = symbol
        bass = None

    root, quality = chord_part.split(":")

    if bass is None:
        bass = root

    return {
        "exist": 1,
        "root": root,
        "quality": quality,
        "bass": bass,
        "symbol": symbol,
    }


stats = Counter()
errors = []

model.eval()

for step, batch in enumerate(loader):
    audio, target = batch
    audio = audio.to(device)
    target = to_device(target, device)

    with torch.no_grad():
        audio = audio[0, ...][None, ...]
        x, _, freqs = preprocessor(audio)

        output = model(x)
        infer_output = model.infer(output)

    target_cpu = to_device(target, torch.device("cpu"))
    infer_cpu = to_device(infer_output, torch.device("cpu"))

    symbol_gt = str(target_cpu[0]["symbol"])
    symbol_pred = str(infer_cpu["symbol"])

    gt = parse_symbol(symbol_gt)
    pred = parse_symbol(symbol_pred)

    stats["total"] += 1

    # exist / N 判断
    if gt["exist"] == pred["exist"]:
        stats["exist_correct"] += 1
    else:
        stats["exist_wrong"] += 1

    # symbol 完全匹配
    if symbol_gt == symbol_pred:
        stats["symbol_correct"] += 1
    else:
        stats["symbol_wrong"] += 1

    # 只在 GT 是和弦时评估 root / quality / bass
    if gt["exist"] == 1:
        stats["chord_total"] += 1

        if pred["exist"] == 1:
            stats["pred_chord_when_gt_chord"] += 1

            if gt["root"] == pred["root"]:
                stats["root_correct"] += 1
            else:
                stats["root_wrong"] += 1

            if gt["quality"] == pred["quality"]:
                stats["quality_correct"] += 1
            else:
                stats["quality_wrong"] += 1

            if gt["bass"] == pred["bass"]:
                stats["bass_correct"] += 1
            else:
                stats["bass_wrong"] += 1

            if gt["root"] == pred["root"] and gt["quality"] == pred["quality"]:
                stats["root_quality_correct"] += 1
            else:
                stats["root_quality_wrong"] += 1

        else:
            stats["miss_chord"] += 1

    # 只在 GT 是 N 时评估 N
    else:
        stats["N_total"] += 1

        if pred["exist"] == 0:
            stats["N_correct"] += 1
        else:
            stats["N_wrong"] += 1

    # 记录错误样本
    if symbol_gt != symbol_pred:
        errors.append({
            "step": step,
            "symbol_gt": symbol_gt,
            "symbol_pred": symbol_pred,
            "root_gt": gt["root"],
            "root_pred": pred["root"],
            "quality_gt": gt["quality"],
            "quality_pred": pred["quality"],
            "bass_gt": gt["bass"],
            "bass_pred": pred["bass"],
        })


def safe_acc(correct, total):
    if total == 0:
        return 0.0
    return correct / total


print()
print("========== Evaluation ==========")
print("total:", stats["total"])

print()
print("[Exist / N]")
print("exist acc:", safe_acc(stats["exist_correct"], stats["total"]))
print("N acc:", safe_acc(stats["N_correct"], stats["N_total"]))
print("N total:", stats["N_total"])
print("chord total:", stats["chord_total"])
print("miss chord:", stats["miss_chord"])

print()
print("[Symbol]")
print("symbol acc:", safe_acc(stats["symbol_correct"], stats["total"]))

print()
print("[Chord only]")
print("root acc:", safe_acc(stats["root_correct"], stats["pred_chord_when_gt_chord"]))
print("quality acc:", safe_acc(stats["quality_correct"], stats["pred_chord_when_gt_chord"]))
print("bass acc:", safe_acc(stats["bass_correct"], stats["pred_chord_when_gt_chord"]))
print("root+quality acc:", safe_acc(stats["root_quality_correct"], stats["pred_chord_when_gt_chord"]))

print()
print("[Counts]")
for k, v in stats.items():
    print(k, v)


# 保存错误样本
error_file = "./tiny_save/eval_errors.csv"

with open(error_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "step",
            "symbol_gt",
            "symbol_pred",
            "root_gt",
            "root_pred",
            "quality_gt",
            "quality_pred",
            "bass_gt",
            "bass_pred",
        ]
    )
    writer.writeheader()
    writer.writerows(errors)

print()
print("saved error file:", error_file)

# python3 test6_stage1.py > ./tiny_save/baby4_eval.txt
