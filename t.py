import torch
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
print("device:",device)

import os
from configs.config import get_config
cfg = get_config()

import sys
sys.path.append(str(cfg.dataset_read_py_path))
# 帮我看看为啥我test.py里from read import AudioDataset, collate_fn找不到read，我已经sys.path.append了

from read1 import AudioDataset, collate_fn
from torch.utils.data import DataLoader
dataset = AudioDataset(cfg.dataset_data_path)
loader = DataLoader(
    dataset,
    batch_size=1, # 必须是1
    shuffle=True,
    # num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True
)

for batch in loader:
    audio, events, texts = batch
    break

# from datasets_al import get_dummy
# x = get_dummy()
# x = x[None,:]

from utils import wav2cqt, wav2spec

pitch_spec, pitch_centre, pitchs = wav2cqt(audio)
freq_spec, freq_centre, freqs = wav2spec(audio)


from models.tokenizer import MusicDetrTokenizer
tokenizer = MusicDetrTokenizer()


audio_emb, text_emb = tokenizer(audio, texts)

from utils.equipTarget import get_target_map
target_pitchMap = get_target_map(events, pitch_centre)

# >>> target_pitchMap.shape
# torch.Size([3, 117, 85, 2])
# 这个是 (N, T, P, 2), 
# 其中 N 表示 N 个文本描述 （可变）
# 其中2表示是否触发0/1 和 持续时间 /秒

from models.model import apply_freq_time_encoding

pos_encoding = apply_freq_time_encoding(freqs, freq_centre, 512)

from utils.visualizer import show_attn_alpha
show_attn_alpha(pos_encoding, 1, 1)

from models.model import PitchTransformer

model = PitchTransformer().to(device)

N = text_emb[0].shape[0]
input_dict = {
    "pitch_spec": pitch_spec.expand(N, -1, -1).to(device),
    "pitchs": pitchs.to(device),
    "pitch_centre": pitch_centre.to(device),
    "freq_spec": freq_spec.expand(N, -1, -1).to(device),
    "freqs": freqs.to(device),
    "freq_centre": freq_centre.to(device),
    "text_emb": text_emb[0][:,None,:].to(device)
}

checkpoint_path = "/home/vipuser/wby/proj_params/params/ckpt_epoch_90.pt"
state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict=state_dict)

output = model(**input_dict)
loss = model.get_loss(output, target_pitchMap.to(device))

onset_logits = torch.sigmoid(output[:,:,:,0])
onset_gt = target_pitchMap[:,:,:,0]

onset_gt_idx = torch.where(onset_gt)


import matplotlib.pyplot as plt
import torch

pred = torch.sigmoid(onset_logits[0]).detach().cpu().numpy()
gt = onset_gt[0].detach().cpu().numpy()

plt.figure(figsize=(12,5))

# 预测
plt.subplot(1,2,1)
plt.imshow(pred.T, aspect='auto', origin='lower')
plt.title("Prediction (sigmoid)")
plt.colorbar()

# GT
plt.subplot(1,2,2)
plt.imshow(gt.T, aspect='auto', origin='lower')
plt.title("Ground Truth")
plt.colorbar()

plt.tight_layout()
plt.savefig(os.path.join(cfg.save_dir,"compare.pdf"))

# events[2]-= 24

# 24, 107

# target[0]['boxes'].shape >> (33,2)
# target[0]['tones'].shape >> (33) midiIdx, -1表示瞬态无音高音色，例如鼓
# target[0]['text_emb'].shape >> (33,512) text句子向量
# target[0]['text'].__len__() >> 34 List[str] text

# cqt.shape >> (B, P, T), (B, 84, 117)
# h_pitch.shape >> (B, T, P)
# h_