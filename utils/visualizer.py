import matplotlib.pyplot as plt
import os
from configs.config import get_config
import torch
import cv2
import numpy as np

"""
import matplotlib.font_manager as fm
fonts = [f.name for f in fm.fontManager.ttflist if 'Hei' in f.name or 'Song' in f.name or 'Wen' in f.name]
print(fonts)
"""

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  

def show_attn_alpha(pos_encoding, num_time=1, num_freq=1):
    cfg = get_config()
    for time_idx in range(num_time):
        freqs_matrix = pos_encoding[time_idx, :, :] @ pos_encoding[time_idx, :, :].T
        plt.figure(figsize=(6, 5))
        plt.imshow(freqs_matrix.cpu().numpy(), aspect='auto')
        plt.colorbar()
        plt.title(f"inner product={time_idx}")
        plt.xlabel("j")
        plt.ylabel("i")
        plt.savefig(os.path.join(cfg.save_dir, f"freq_time_{time_idx}.pdf"))
        plt.close()

    for freq_idx in range(num_freq):
        times_matrix = pos_encoding[:, freq_idx, :] @ pos_encoding[:, freq_idx, :].T
        plt.figure(figsize=(6, 5))
        plt.imshow(times_matrix.cpu().numpy(), aspect='auto')
        plt.colorbar()
        plt.title(f"inner product={time_idx}")
        plt.xlabel("j")
        plt.ylabel("i")
        plt.savefig(os.path.join(cfg.save_dir, f"time_freq{freq_idx}.pdf"))
        plt.close()



def compare_result(onset_logits, onset_gt, name="compare", title=None):
    # onset_logits onset_gt: (T, P)
    cfg = get_config()
    
    pred = onset_logits
    gt = onset_gt

    plt.figure(figsize=(12,5))

    # 预测
    plt.subplot(1,2,1)
    plt.imshow(pred.T, aspect='auto', origin='lower')
    if title is not None:
        plt.title(f"{title}")
    else:
        plt.title("Prediction")
    plt.colorbar()

    # GT
    plt.subplot(1,2,2)
    plt.imshow(gt.T, aspect='auto', origin='lower')
    plt.title("Ground Truth")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, name + ".pdf"))



def compare_result_3(onset_logits, onset_gt, cqt, name="compare", title=None):
    # onset_logits onset_gt: (T, P)
    cfg = get_config()
    
    pred = onset_logits
    gt = onset_gt

    plt.figure(figsize=(12,5))

    # 预测
    plt.subplot(1,3,1)
    plt.imshow(pred.T, aspect='auto', origin='lower')
    if title is not None:
        plt.title(f"{title}")
    else:
        plt.title("Prediction")
    plt.colorbar()

    # GT
    plt.subplot(1,3,2)
    plt.imshow(gt.T, aspect='auto', origin='lower')
    plt.title("Ground Truth")
    plt.colorbar()

    # CQT
    plt.subplot(1,3,3)
    plt.imshow(cqt.T, aspect='auto', origin='lower')
    plt.title("CQT")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, name + ".pdf"))


import cv2
import numpy as np
import soundfile as sf
import subprocess
import sys
import cv2
import numpy as np
import soundfile as sf
import subprocess
import os

def export_event_flash_mp4(audio, events, sr=44100, save_path="output.mp4"):
    """
    audio: (L,)
    events: (E,) sample index

    效果：
        平时：灰色背景
        event：闪蓝色
    """

    # ----------------------
    # 0. numpy + normalize
    # ----------------------
    if "torch" in str(type(audio)):
        audio = audio.detach().cpu().numpy()
    if "torch" in str(type(events)):
        events = events.detach().cpu().numpy()

    audio = audio.astype(np.float32)
    audio = audio / (np.abs(audio).max() + 1e-8)

    # ----------------------
    # 1. 参数
    # ----------------------
    H, W = 240, 240
    fps = 25
    duration = len(audio) / sr
    total_frames = int(duration * fps)

    # event 转成时间（秒）
    event_times = events # / sr

    # 每次闪持续时间（秒）
    flash_duration = 0.05

    # ----------------------
    # 2. 写视频
    # ----------------------
    tmp_video = "tmp_video.mp4"
    tmp_audio = "tmp_audio.wav"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(tmp_video, fourcc, fps, (W, H))

    for frame_idx in range(total_frames):
        t = frame_idx / fps

        # 判断是否在某个 event 附近
        is_event = np.any(np.abs(event_times - t) < flash_duration)

        if is_event:
            color = (255, 0, 0)   # 蓝（BGR）
        else:
            color = (200, 200, 200)  # 灰

        img = np.ones((H, W, 3), dtype=np.uint8)
        img[:] = color

        # 时间文字（可选）
        cv2.putText(img, f"{t:.2f}s", (20, H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

        out.write(img)

    out.release()

    # ----------------------
    # 3. 写音频
    # ----------------------
    sf.write(tmp_audio, audio, sr)

    # ----------------------
    # 4. ffmpeg 合成
    # ----------------------
    subprocess.run([
        "ffmpeg", "-y",
        "-i", tmp_video,
        "-i", tmp_audio,
        "-c:v", "copy",
        "-c:a", "aac",
        save_path
    ], check=True)

    os.remove(tmp_video)
    os.remove(tmp_audio)

    print(f"✅ 已生成: {save_path}")