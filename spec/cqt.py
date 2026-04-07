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


def wav2cqt(wav, shift=0):
    """
    wav: (B, L)
    return:
    spec: (B, T, F) # 余弦的投影
    pos: (T) # 中心位置
    """
    
    cfg = get_config()
    
    window_len = cfg.window_len
    assert window_len % 2 == 0, "window_len 必须是偶数"
    stride = cfg.stride
    window_type = cfg.window_type # hann
    
    min_midi, max_midi = cfg.min_midi, cfg.max_midi
    freqs = get_freqs(min_midi, max_midi)
    
    # 🔥 加 shift（核心）
    if shift != 0:
        freqs = freqs * (2 ** (shift / 1200.0))

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
    
    t = torch.arange(-window_len//2, window_len//2, device=device) / cfg.sr
    
    x_unfold = x_unfold * window # 广播
    
    scale = cfg.cqt_scale
    sigma = scale / (freqs + 1e-6)  # (F,)
    # (F, W)
    mask = torch.exp(- (t[None, :] ** 2) / (2 * (sigma[:, None] ** 2)))
    # normalize（防止能量偏移）
    mask = mask / (mask.sum(dim=-1, keepdim=True) + 1e-8)
    
    sin_basis = torch.sin(2 * math.pi * freqs[:, None] * t[None, :])
    cos_basis = torch.cos(2 * math.pi * freqs[:, None] * t[None, :])
    
    sin_basis = sin_basis * mask
    cos_basis = cos_basis * mask

    spec_cos = torch.einsum("btw,fw->btf", x_unfold, cos_basis) # (B,T,F)
    spec_sin = torch.einsum("btw,fw->btf", x_unfold, sin_basis) # (B,T,F)
    spec = torch.sqrt(spec_cos**2 + spec_sin**2) # (B,T,F)
    
    # 时间中心
    centers = torch.arange(T, device=device) * stride + window_len // 2
    pos = centers # (T)
    
    return spec, pos, freqs



def estimate_shift(wav, shift_range=(-50, 50), step=1):
    """
    wav: (1, L) torch tensor
    return: best_shift (cents)
    """

    shifts = np.arange(shift_range[0], shift_range[1] + 1, step)

    scores = []

    for shift in shifts:
        spec, _, _ = wav2cqt(wav, shift=shift)  # (B,T,F)

        spec = spec.detach().cpu().numpy()

        # ----------------------
        # 1. 时间平均
        # ----------------------
        energy_f = spec.mean(axis=1)[0]  # (F,)

        # ----------------------
        # 2. mask（核心）
        # ----------------------
        # 👉 方法 A：平方增强 peak
        weight = energy_f ** 2

        # 👉 可选：频率范围限制
        F = len(energy_f)
        mask = np.ones(F)

        # 比如去掉极低频（前10%）和极高频（后10%）
        low = int(0.5 * F)
        high = int(0.9 * F)
        mask[:low] = 0
        mask[high:] = 0

        # ----------------------
        # 3. 计算 score
        # ----------------------
        score = np.sum(weight * mask)

        scores.append(score)

    scores = np.array(scores)

    best_idx = np.argmax(scores)
    best_shift = shifts[best_idx]

    return best_shift, shifts, scores