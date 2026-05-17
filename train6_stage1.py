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
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)
assert 0

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

# for audio, target in loader:
#     x, _, freqs = preprocessor(audio.to(device))
#     assert 0


# teacher = Teacher()
current_state = model.state_dict()

checkpoint_path = "../params/detr6/baby4.pt"
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
recorder = TrainingRecorder(cfg, "baby5")
recorder.load()
# -------- 混合精度（强烈建议）--------
scaler = torch.cuda.amp.GradScaler()

# -------- 训练 --------
model.train()
model.set_mode("midi")
model.retain_P = True

num_epochs = 50000

hist_len = recorder.history["loss"].__len__()
start_epoch = cfg.save_epoch * (hist_len-1) if hist_len!=0 else 0
for epoch in range(start_epoch+1, num_epochs):
    total_loss = 0
    for step, batch in enumerate(loader):
        audio, target = batch
        audio = audio.to(device)
        target = to_device(target, device)
        # ---------- forward + loss（AMP）----------
        with torch.amp.autocast("cuda"):
            x, _, freqs = preprocessor(audio.to(device))
            output = model(x)
            loss = model.get_loss(output, target)
            assert ~torch.isnan(loss)
        # ---------- backward ----------
        optimizer.zero_grad()

        scaler.scale(loss).backward()

        # 梯度裁剪（防炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        
        # ---------- log ----------
        if step % 10 == 0:
            print(f"[Epoch {epoch}] step {step} loss: {loss.item():.4f}")
            with torch.no_grad():
                audio = audio[0,...][None,...]
                x, _, freqs = preprocessor(audio.to(device))
                model.eval()
                output = model(x)
                infer_output = model.infer(output)
                model.train()
            # infer_output, target : Dict{
            #     "root": root_pred, # (M) 0~11
            #     "chord": chord_pred, # (M, 12)
            #     "tonic": tonic_pred, # (M) 0~11
            #     "start": start_pred, # (M)
            #     "sustain": sustain_pred, # (M)
            #     "exist": exist_pred, # (M)
            # }
            target = to_device(target, torch.device("cpu"))
            infer_output = to_device(infer_output, torch.device("cpu"))
            # plot_pianoroll_event(infer_output, target[0])
            with open("./tiny_save/temp.txt", "w") as f:
                f.write("target:\n")
                # f.write("chord_cls:"+str(target[0]['chord_cls'])+"\n")
                # f.write("root:"+str(target[0]['root'])+"\n")
                # f.write("cls:"+str(target[0]['chord_cls_name'])+"\n")
                # f.write("pitch:"+str(target[0]['pitch_cls'])+"\n")
                # f.write("chord:"+str(target[0]['symbol'])+"\n")
                f.write("midi:"+str(target[0]['midi'])+"\n")

                f.write("infer:\n")
                # f.write("root:"+str(infer_output['root_idx'])+"\n")
                # f.write("cls:"+str(infer_output['chord_name'])+"\n")
                # f.write("pitch:"+str(infer_output['pitch_cls'])+"\n")
                f.write("midi:"+str(infer_output['midi'])+"\n")
                
    print(f"==== Epoch {epoch} avg loss: {total_loss / (step+1):.4f} ====")
    # ---------- 保存 ----------
    if epoch % cfg.save_epoch == 0:
        os.makedirs(cfg.large_save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(cfg.large_save_dir, f"baby5.pt"))
        recorder.update(total_loss / (step+1), cfg.lr)
        recorder.save()


# loss = model.get_loss(audio, target)

# with torch.amp.autocast("cuda"):
#     loss = model.get_loss(audio, target)
# with torch.amp.autocast("cuda"):
#     loss = model.get_loss(audio, target)


