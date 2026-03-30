import torch
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



# events[2]-= 24

# 24, 107

# target[0]['boxes'].shape >> (33,2)
# target[0]['tones'].shape >> (33) midiIdx, -1表示瞬态无音高音色，例如鼓
# target[0]['text_emb'].shape >> (33,512) text句子向量
# target[0]['text'].__len__() >> 34 List[str] text

# cqt.shape >> (B, P, T), (B, 84, 117)
# h_pitch.shape >> (B, T, P)
# h_