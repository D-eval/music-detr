"""
连续频率 spectrogram（log/linear/mel 可选）
与 pitch_spec 对齐
"""

import torch
import math
from configs.config import get_config


# ===== 频率网格 =====
def get_spec_freqs():
    cfg = get_config()

    f_min = cfg.min_freq
    f_max = cfg.max_freq
    N = cfg.num_freqs
    scale = cfg.freq_scale

    if scale == "linear":
        freqs = torch.linspace(f_min, f_max, N)

    elif scale == "log":
        freqs = torch.exp(
            torch.linspace(
                torch.log(torch.tensor(f_min)),
                torch.log(torch.tensor(f_max)),
                N
            )
        )

    elif scale == "mel":
        def hz_to_mel(f):
            return 2595 * torch.log10(1 + f / 700)

        def mel_to_hz(m):
            return 700 * (10**(m / 2595) - 1)

        m_min = hz_to_mel(torch.tensor(f_min))
        m_max = hz_to_mel(torch.tensor(f_max))

        m = torch.linspace(m_min, m_max, N)
        freqs = mel_to_hz(m)

    else:
        raise ValueError(f"Unknown freq_scale {scale}")

    return freqs


def wav2spec_2C(wav):
    """
        wav: (B, L, 2)
    """
    spec1, pos, freqs = wav2spec(wav[:,:,0])
    spec2, pos, freqs = wav2spec(wav[:,:,1])
    spec = torch.stack([spec1, spec2], dim=-1)
    return spec, pos, freqs


# ===== 主函数 =====
def wav2spec(wav):
    """
    wav: (B, L)

    return:
        spec: (B, T, F)
        pos:  (1, T)
    """

    cfg = get_config()

    window_len = cfg.window_len
    assert window_len % 2 == 0
    stride = cfg.stride

    wav_len = cfg.wav_len
    _, L = wav.shape
    # assert L == wav_len

    device = wav.device

    # ===== window =====
    if cfg.window_type == "hann":
        window = torch.hann_window(window_len, device=device)
    elif cfg.window_type == "hamming":
        window = torch.hamming_window(window_len, device=device)
    else:
        raise ValueError(f"wtf {cfg.window_type}")

    # ===== unfold =====
    x = wav
    x_unfold = x.unfold(1, window_len, stride)  # (B, T, W)
    B, T, _ = x_unfold.shape

    x_unfold = x_unfold * window

    # ===== 频率 =====
    freqs = get_spec_freqs().to(device)
    F = freqs.shape[0]

    # ===== 时间轴 =====
    t = torch.arange(-window_len//2, window_len//2, device=device) / cfg.sr

    # ===== cos basis =====
    cos_basis = torch.cos(2 * math.pi * freqs[:, None] * t[None, :])

    # ===== 投影 =====
    spec = torch.einsum("btw,fw->btf", x_unfold, cos_basis) # (B, T, F)

    # ===== 时间中心 =====
    centers = torch.arange(T, device=device) * stride + window_len // 2
    pos = centers  # (T)

    return spec, pos, freqs