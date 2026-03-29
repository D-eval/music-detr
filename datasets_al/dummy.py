
import torch
from configs.config import get_config

def get_dummy():
    cfg = get_config()
    wav_len = cfg.wav_len
    return torch.randn((wav_len))

