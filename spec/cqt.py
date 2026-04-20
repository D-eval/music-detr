"""
计算音符频率
到128个midi的频率投影
"""
from utils.midi import midi2freq, freq2midi
from configs.config import get_config
import numpy as np
import torch
import torch.nn.functional as F
import math

def get_freqs(min_midi, max_midi):
    # (F,)
    midi = torch.arange(min_midi, max_midi + 1)
    freqs = midi2freq(midi)
    return freqs


def wav2cqt_2C(wav, shift=0):
    """
        wav: (B, L, 2)
    """
    spec1, pos, freqs = wav2cqt(wav[:,:,0], shift)
    spec2, pos, freqs = wav2cqt(wav[:,:,1], shift)
    spec = torch.stack([spec1, spec2], dim=-1)
    return spec, pos, freqs


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
    freqs = get_freqs(min_midi, max_midi).to(wav.device)
    
    # 🔥 加 shift（核心）
    if shift != 0:
        freqs = freqs * (2 ** (shift / 1200.0))

    wav_len = cfg.wav_len
    _, L = wav.shape
    # assert L==wav_len, f"长度不匹配，预期长度{wav_len}，实际获得{L}"
    
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



freq2numT = lambda freq: 20 * (math.log(freq) - math.log(50)) / math.log(16000/50) +\
    1 * (math.log(16000) - math.log(freq)) / math.log(16000/50)

def cqt1(audio, temp_freq = 440):
    """
    audio: (T,)
    return: (T,)
    """
    cfg = get_config()
    sr = cfg.sr
    L = int(1/temp_freq * sr * freq2numT(temp_freq))
    L = L+1 if L % 2 == 0 else L  # ✅ 保证奇数
    window = torch.hann_window(L)

    phase = torch.arange(0, L) / sr * 2*math.pi
    cos = torch.cos(phase) * window
    sin = torch.sin(phase) * window

    cos = cos.view(1, 1, -1)
    sin = sin.view(1, 1, -1)

    # pad_left = L // 2
    # pad_right = L // 2 - 1

    # x_pad = F.pad(audio, (pad_left, pad_right))

    # x = x_pad.unfold(dimension=0, size=L, step=1) # (T-L, L)

    # cos_proj = x @ cos
    # sin_proj = x @ sin
    # freq_E = cos_proj **2 + sin_proj **2

    audio = audio.view(1, 1, -1)

    cos_proj = F.conv1d(audio, cos, padding=L//2)
    sin_proj = F.conv1d(audio, sin, padding=L//2)
    freq_E = cos_proj **2 + sin_proj **2
    return freq_E[0,0,:]
