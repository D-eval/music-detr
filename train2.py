"""textwise"""

import torch
import torch.nn as nn
import torch.optim as optim
from utils.visualizer import compare_result_3

import os
from configs.config import get_config
cfg = get_config()

import sys
sys.path.append(str(cfg.dataset_read_py_path))

from read import AudioDataset, collate_fn, to_device
from torch.utils.data import DataLoader
dataset = AudioDataset(cfg.dataset_data_path)


loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    # num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)


from models.detr import PitchTransformer
from spec import wav2cqt, wav2spec
from models.tokenizer import MusicDetrTokenizer
from utils.equipTarget import get_target_map, get_sustain_map, get_sustain_map_textwise, normalize_targets_pitch, render_pred_pitch_map

if cfg.map_type == "target_map":
    get_map = get_target_map
elif cfg.map_type == "sustain_map":
    get_map = get_sustain_map_textwise
else:
    raise NotImplementedError("wtf")


device = torch.device("cuda" if torch.cuda.is_available else "cpu")
print("device:",device)

model = PitchTransformer().to(device)

tokenizer = MusicDetrTokenizer() # .to(device)

checkpoint_path = "/home/vipuser/wby/proj_params/params/detr/ckpt_epoch_100.pt"
state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict=state_dict)

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
        target = normalize_targets_pitch(target)
        target = to_device(target, device)
        
        # ---------- spec ----------
        pitch_spec, pitch_centre, pitchs = wav2cqt(audio)
        freq_spec, freq_centre, freqs = wav2spec(audio)

        input_dict = {
            "pitch_spec": pitch_spec.to(device), # (B, T, F)
            "pitchs": pitchs.to(device),
            "pitch_centre": pitch_centre.to(device),
            "freq_spec": freq_spec.to(device), # (B, T, F)
            "freqs": freqs.to(device),
            "freq_centre": freq_centre.to(device),
        }
        # assert 0
        # ---------- forward + loss（AMP）----------
        with torch.amp.autocast("cuda"):
            output = model(**input_dict)
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
            with torch.no_grad():
                event_pred = model.infer(output[0])
                pitch_centre = pitch_centre.to(device)
                pred_pitchmap = render_pred_pitch_map(event_pred, pitch_centre)
                gt_pitchmap = render_pred_pitch_map(target[0], pitch_centre)
            
            compare_result_3(pred_pitchmap.detach().cpu().numpy()[...,1],
                           gt_pitchmap.detach().cpu().numpy()[...,1],
                           pitch_spec[0].detach().cpu().numpy(),
                           "compare")
    print(f"==== Epoch {epoch} avg loss: {total_loss / (step+1):.4f} ====")

    # ---------- 保存 ----------
    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join(cfg.large_save_dir, f"ckpt_epoch_{epoch}.pt"))



