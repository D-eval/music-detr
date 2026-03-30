"""
计算音符频率
到128个midi的频率投影
"""
from utils.midi import midi2freq, freq2midi
from configs.config import get_config
import numpy as np
import torch
import math

def get_freqs(min_midi, max_midi):
    # (F,)
    midi = torch.arange(min_midi, max_midi + 1)
    freqs = midi2freq(midi)
    return freqs


def wav2cqt(wav):
    """
    wav: (B, L)
    return:
    spec: (B, T, F) # 余弦的投影
    pos: (1, T) # 中心位置
    """
    
    cfg = get_config()
    
    window_len = cfg.window_len
    assert window_len % 2 == 0, "window_len 必须是偶数"
    stride = cfg.stride
    window_type = cfg.window_type # hann
    
    min_midi, max_midi = cfg.min_midi, cfg.max_midi
    freqs = get_freqs(min_midi, max_midi)
    
    wav_len = cfg.wav_len
    _, L = wav.shape
    assert L==wav_len, f"长度不匹配，预期长度{wav_len}，实际获得{L}"
    
    F = freqs.shape[0]
    device = wav.device
    # window
    if cfg.window_type == "hann":
        window = torch.hann_window(window_len, device=device)
    elif cfg.window_type == "hamming":
        window = torch.hamming_window(window_len, device=device)
    else:
        # window = torch.ones(window_len, device=device)
        raise ValueError(f"wtf {cfg.window_type}")
    
    x = wav
    x_unfold = x.unfold(dimension=1, size=window_len, step=stride)
    B, T, _ = x_unfold.shape
    
    x_unfold = x_unfold * window # 广播
    
    
    t = torch.arange(-window_len//2, window_len//2, device=device) / cfg.sr
    cos_basis = torch.cos(2 * math.pi * freqs[:, None] * t[None, :])

    spec = torch.einsum("btw,fw->btf", x_unfold, cos_basis) # (B,T,F)
    
    # 时间中心
    centers = torch.arange(T, device=device) * stride + window_len // 2
    pos = centers # (T)
    
    return spec, pos, freqs


