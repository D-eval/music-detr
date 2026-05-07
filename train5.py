"""
read5
detr5

多类别事件

nohup python3 train5.py > train5.log 2>&1 &
"""

import torch
import torch.nn as nn
import torch.optim as optim

from utils.trainRecorder import TrainingRecorder
from utils.visualizer import plot_pianoroll_event

import os
from configs.config5 import get_config
cfg = get_config()

import sys
sys.path.append(str(cfg.dataset_read_py_path))

from read5 import AudioDataset, collate_fn, to_device
from torch.utils.data import DataLoader
dataset = AudioDataset(root_dir=cfg.dataset_data_path)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    # num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)


from models.detr5 import PitchTransformer
from spec import wav2cqt_2C, wav2spec_2C
from models.teacher import Teacher
from utils.equipTarget import get_target_map, get_sustain_map, get_sustain_map_textwise, normalize_targets_pitch, render_pred_pitch_map, render_pred_group_pitch_map, embed_text

if cfg.map_type == "target_map":
    get_map = get_target_map
elif cfg.map_type == "sustain_map":
    get_map = get_sustain_map_textwise
else:
    raise NotImplementedError("wtf")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

model = PitchTransformer().to(device)

# teacher = Teacher()

checkpoint_path = "../params/detr5/ckpt_epoch.pt"
state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict=state_dict)

# -------- optimizer --------
optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
recorder = TrainingRecorder(cfg)
recorder.load()
# -------- 混合精度（强烈建议）--------
scaler = torch.cuda.amp.GradScaler()

# -------- 训练 --------
model.train()


num_epochs = 50000

hist_len = recorder.history["loss"].__len__()
start_epoch = cfg.save_epoch * (hist_len-1) if hist_len!=0 else 0
for epoch in range(start_epoch+1, num_epochs):
    total_loss = 0
    for step, batch in enumerate(loader):
        audio, target = batch
        target = to_device(target, device)
        audio = audio.to(device)
        # ---------- forward + loss（AMP）----------
        with torch.amp.autocast("cuda"):
            output = model(audio)
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
                model.eval()
                output = model(audio)
                infer_output = model.infer(output[0])
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
                f.write(str(target[0])+"\n")

                f.write("infer:\n")
                f.write(str(infer_output)+"\n")

            # assert 0
    print(f"==== Epoch {epoch} avg loss: {total_loss / (step+1):.4f} ====")
    # ---------- 保存 ----------
    if epoch % cfg.save_epoch == 0:
        torch.save(model.state_dict(), os.path.join(cfg.large_save_dir, f"ckpt_epoch.pt"))
        recorder.update(total_loss / (step+1), cfg.lr)
        recorder.save()


# loss = model.get_loss(audio, target)

# with torch.amp.autocast("cuda"):
#     loss = model.get_loss(audio, target)
# with torch.amp.autocast("cuda"):
#     loss = model.get_loss(audio, target)


