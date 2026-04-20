"""
chord framewise

nohup python3 train4.py > trainAL.log 2>&1 &
"""

import torch
import torch.nn as nn
import torch.optim as optim

from utils.trainRecorder import TrainingRecorder
from utils.visualizer import plot_pianoroll_timewise

import os
from configs.config import get_config22 as get_config
cfg = get_config()

import sys
sys.path.append(str(cfg.dataset_read_py_path))

from read0 import AudioDataset, collate_fn
from torch.utils.data import DataLoader
dataset = AudioDataset(root_dir=cfg.dataset_data_path)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    # num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)


from models.conv import BTF
from spec import wav2cqt_2C, wav2spec_2C
from models.teacher import Teacher
from utils.equipTarget import get_target_map, get_sustain_map, get_sustain_map_textwise, normalize_targets_pitch, render_pred_pitch_map, render_pred_group_pitch_map, to_device, embed_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

model = BTF().to(device)

# teacher = Teacher()

# checkpoint_path = "/home/vipuser/wby/proj_params/params/al/ckpt_epoch_90.pt"
# state_dict = torch.load(checkpoint_path)
# model.load_state_dict(state_dict=state_dict)

# -------- optimizer --------
optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
recorder = TrainingRecorder()
recorder.load()
# -------- 混合精度（强烈建议）--------
scaler = torch.cuda.amp.GradScaler()

# -------- 训练 --------
# model.train()


num_epochs = 500

hist_len = recorder.history["loss"].__len__()
start_epoch = cfg.save_epoch * (hist_len-1) if hist_len!=0 else 0
for epoch in range(start_epoch+1, num_epochs):
    total_loss = 0
    for step, batch in enumerate(loader):
        audio, target = batch
        audio = audio.to(device)
        with torch.amp.autocast("cuda"):
            output = model(audio)
            loss = model.get_loss(output, target)
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
            plot_pianoroll_timewise(output, target[0])
            assert 0
    print(f"==== Epoch {epoch} avg loss: {total_loss / (step+1):.4f} ====")
    # ---------- 保存 ----------
    if epoch % cfg.save_epoch == 0:
        torch.save(model.state_dict(), os.path.join(cfg.large_save_dir, f"ckpt_epoch_{epoch}.pt"))
        recorder.update(total_loss / (step+1), cfg.lr)
        recorder.save()


# loss = model.get_loss(audio, target)

# with torch.amp.autocast("cuda"):
#     loss = model.get_loss(audio, target)
# with torch.amp.autocast("cuda"):
#     loss = model.get_loss(audio, target)
