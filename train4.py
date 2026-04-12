"""
用来训练ALUnion
"""

import torch
import torch.nn as nn
import torch.optim as optim
from utils.visualizer import show_al_result

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
    batch_size=1,
    shuffle=True,
    # num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)


from models.AL import ALUnion
from spec import wav2cqt, wav2spec
from models.teacher import Teacher
from utils.equipTarget import get_target_map, get_sustain_map, get_sustain_map_textwise, normalize_targets_pitch, render_pred_pitch_map, render_pred_group_pitch_map, to_device, embed_text

if cfg.map_type == "target_map":
    get_map = get_target_map
elif cfg.map_type == "sustain_map":
    get_map = get_sustain_map_textwise
else:
    raise NotImplementedError("wtf")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

model = ALUnion().to(device)

teacher = Teacher()

# checkpoint_path = "/home/vipuser/wby/proj_params/params/detr2/ckpt_epoch_20.pt"
# state_dict = torch.load(checkpoint_path)
# model.load_state_dict(state_dict=state_dict)

# -------- optimizer --------
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# -------- 混合精度（强烈建议）--------
scaler = torch.cuda.amp.GradScaler()

# -------- 训练 --------
model.train()

start_epoch = 0
num_epochs = 500

for epoch in range(start_epoch, num_epochs):
    total_loss = 0
    for step, batch in enumerate(loader):
        audio, target = batch
        embed_text(target, teacher)
        target = to_device(target, device)
        
        loss = model.get_loss(audio, target)
        print("success")
    
        # assert 0
        # ---------- forward + loss（AMP）----------
        with torch.amp.autocast("cuda"):
            loss = model.get_loss(audio, target)
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
                infer_output = model.infer(audio[0, :])
                cqt, times = wav2cqt(audio[0,:].unsqueeze(0))
            show_al_result(infer_output,
                           target[0],
                           cqt[0],
                           times)
    print(f"==== Epoch {epoch} avg loss: {total_loss / (step+1):.4f} ====")

    # ---------- 保存 ----------
    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join(cfg.large_save_dir, f"ckpt_epoch_{epoch}.pt"))


# pitchmap = render_pred_pitch_map(event_pred[0], pitch_centre)

# compare_result_3(pitchmap.detach().cpu().numpy()[...,1],
#                 pitchmap.detach().cpu().numpy()[...,1],
#                 pitchmap.detach().cpu().numpy()[...,1],
#                 "wtf")