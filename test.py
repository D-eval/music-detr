import torch
import torch.nn as nn
import torch.optim as optim
from utils.visualizer import compare_result, export_event_flash_mp4

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
    batch_size=2,
    shuffle=True,
    # num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)

from models.detr import PitchTransformer
from spec import wav2cqt, wav2spec
from models.tokenizer import MusicDetrTokenizer
from utils.equipTarget import get_target_map

# tokenizer = MusicDetrTokenizer()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

# ---------- target ----------
# target_pitchMap = get_target_map([events], pitch_centre)

model = PitchTransformer().to(device)

for step, batch in enumerate(loader):
    audio, target = batch
    target = to_device(target, device)
    break

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

# ---------- forward + loss（AMP）----------
with torch.amp.autocast("cuda"):
    output = model(**input_dict)
    loss = model.get_loss(output, target)


# compare_result(torch.sigmoid(pitch_spec).detach().cpu().numpy(),
#                 target_pitchMap[...,0].detach().cpu().numpy().sum(0))

# 把 event 对齐到

# import soundfile as sf

# audio_np = audio.detach().cpu().numpy()
# events_np = events[:,0].detach().cpu().numpy()

# sf.write("output.wav", audio_np, cfg.sr)

# export_event_flash_mp4(audio_np, events_np, sr=cfg.sr, save_path="align.mp4")

# >>> events[:,0].shape
# torch.Size([34])
# 34 个开始时间
# >>> audio.shape
# torch.Size([132300])

# import matplotlib.pyplot as plt

# plt.close()

# plt.imshow(freq_spec[0].T, aspect='auto', origin='lower')
# plt.colorbar()
# plt.show()


