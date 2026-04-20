import matplotlib.pyplot as plt
import os
from configs.config import get_config, get_config22
import torch
import cv2
import numpy as np

"""
import matplotlib.font_manager as fm
fonts = [f.name for f in fm.fontManager.ttflist if 'Hei' in f.name or 'Song' in f.name or 'Wen' in f.name]
print(fonts)
"""

# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  

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
    plt.savefig(os.path.join(cfg.save_dir, name + ".png"))
    
    plt.close()


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




def render_roll(events, T, P, use_prob=False):
    """
    events: infer_events 输出
    T: 时间长度
    P: pitch_vocab_size（含 pitchless）

    return:
        roll: (T, P)
    """

    roll = torch.zeros(T, P)

    for ev in events:
        s = int(ev["start"] * T)
        e = int(ev["end"] * T)

        s = max(0, min(T-1, s))
        e = max(s+1, min(T, e))

        p = ev["pitch"]

        if p >= P:
            continue  # 可选：忽略 pitchless

        if use_prob:
            # 用 soft 分布
            roll[s:e] += ev["pitch_prob"][:P]
        else:
            roll[s:e, p] += ev["confidence"]

    return roll

def plot_roll(roll, name="roll"):
    """roll: (T, P)"""
    cfg = get_config()
    
    plt.figure(figsize=(12, 4))
    plt.imshow(roll.T, aspect='auto', origin='lower')
    plt.xlabel("Time")
    plt.ylabel("Pitch")
    plt.colorbar()
    plt.title("Piano Roll")
    
    plt.savefig(os.path.join(cfg.save_dir, name + ".png"))
    plt.close()
    

# def show_al_result(output, target, cqt, times, name="al_result", title=None):
#     """
#     output: List[
#         Dict{
#             'text_desc': str,
#             'start': (num_events, ), sec 注意转换成float64，否则会数值溢出
#             'sustain': (num_events, ), sec
#             'pitch': (num_events, ), int 0 ~ P-2 是 pitch, P-1 是 模糊音高
#         }
#     ] * num_timbre
#     target: Dict{
#         "text_emb": (Nt, C_text),
#         "start": (Ne,), sec
#         "sustain": (Ne,), sec
#         "pitch": (Ne,) # -1 ~ P-1, 需要首先把 -1 变成 P-1
#         "text": List[str] Nt
#         "text_idx": (Ne,) int 0 ~ Nt-1
#     }
#     cqt: (T, P)
#     times: (T, ) long, sec * sr
#     保存为 save_dir 的多个文件
#     用2个子图 (T, P) 画出 每个 output Dict，并用 text_desc 作为 plt title
#     target 也是
#     每个png有2个子图，明明为 gt1.png, gt2.png, ..., pred1.png, pred2.png, ...
#     """
#     cfg = get_config()
#     sr = cfg.sr
    
#     save_dir = cfg.save_dir
#     save_dir = os.path.join(save_dir, name)
#     os.makedirs(save_dir) # 设置成覆盖模式，如果存在就覆盖

def show_al_result(output, target, cqt, times, name="al_result", title=None):
    cfg = get_config()
    sr = cfg.sr

    save_dir = os.path.join(cfg.save_dir, name)
    os.makedirs(save_dir, exist_ok=True)

    T, Pm1 = cqt.shape
    P = Pm1 + 1

    # ===== 工具函数：event → roll =====
    def events_to_roll(starts, sustains, pitch, text=None):
        roll = np.zeros((T, P))

        # 转 numpy + float64（防溢出）
        starts = starts.detach().cpu().numpy().astype(np.float64)
        sustains = sustains.detach().cpu().numpy().astype(np.float64)
        pitch = pitch.detach().cpu().numpy()

        for i in range(len(starts)):
            t0 = starts[i]
            dur = sustains[i]
            p = pitch[i]

            # pitch -1 → 模糊音高
            if p < 0:
                p = P - 1

            # 时间 → frame
            start_idx = np.argmin(np.abs(times - t0 * sr))
            end_time = (t0 + dur) * sr
            end_idx = np.argmin(np.abs(times - end_time))

            end_idx = max(start_idx + 1, end_idx)

            # clip
            start_idx = np.clip(start_idx, 0, T-1)
            end_idx = np.clip(end_idx, 0, T)

            roll[start_idx:end_idx, p] = 1.0

        return roll

    # =========================================================
    # ====================== GT ================================
    # =========================================================
    gt_starts = target["start"]
    gt_sustain = target["sustain"]
    gt_pitch = target["pitch"].clone()

    # -1 → P-1
    gt_pitch[gt_pitch < 0] = P - 1

    # 按 text_idx 分组
    text_idx = target["text_idx"]
    texts = target["text"]

    for i in range(len(texts)):
        mask = (text_idx == i)

        if mask.sum() == 0:
            continue

        roll = events_to_roll(
            gt_starts[mask],
            gt_sustain[mask],
            gt_pitch[mask]
        )

        plt.figure(figsize=(10, 4))
        plt.imshow(roll.T, aspect='auto', origin='lower')
        plt.title(f"GT: {texts[i]}")
        plt.xlabel("Time")
        plt.ylabel("Pitch")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"gt_{i}.png"))
        plt.close()

    # =========================================================
    # ====================== PRED ==============================
    # =========================================================
    for i, d in enumerate(output):
        starts = d["start"]
        sustain = d["sustain"]
        pitch = d["pitch"]

        roll = events_to_roll(starts, sustain, pitch)

        plt.figure(figsize=(10, 4))
        plt.imshow(roll.T, aspect='auto', origin='lower')

        desc = d.get("text_desc", f"pred_{i}")
        plt.title(f"PRED: {desc}")

        plt.xlabel("Time")
        plt.ylabel("Pitch")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"pred_{i}.png"))
        plt.close()
        
        
import numpy as np
import matplotlib.pyplot as plt

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']


def plot_pianoroll_event(pred, target, title="event_pianoroll", name="chord_pred"):

    def draw(ax, data, name):
        root = np.asarray(data["root"])
        chord = np.asarray(data["chord"])
        tonic = np.asarray(data["tonic"])
        start = np.asarray(data["start"])
        sustain = np.asarray(data["sustain"])
        # exist = np.asarray(data["exist"])
        before = np.asarray(data.get("before", np.zeros_like(start)))

        M = len(start)

        for i in range(M):
            # if exist[i] < 0.5:
            #     continue

            # ===== 处理 before =====
            if before[i] > 0.5:
                t0 = -1.0
            else:
                t0 = start[i]

            t1 = t0 + sustain[i]

            # 🎹 chord
            for p in range(12):
                if chord[i][p] > 0.5:
                    ax.add_patch(
                        plt.Rectangle(
                            (t0, p - 0.4),
                            sustain[i],
                            0.8,
                            color="skyblue",
                            alpha=0.6
                        )
                    )

            # 🔴 root
            ax.add_patch(
                plt.Rectangle(
                    (t0, root[i] - 0.4),
                    sustain[i],
                    0.8,
                    color="red",
                    alpha=0.9
                )
            )

            # 🟢 tonic（边框）
            ax.add_patch(
                plt.Rectangle(
                    (t0, tonic[i] - 0.4),
                    sustain[i],
                    0.8,
                    fill=False,
                    edgecolor="green",
                    linewidth=2
                )
            )

        # ===== 坐标轴 =====
        ax.set_title(name)
        ax.set_ylim(-0.5, 11.5)
        ax.set_yticks(range(12))
        ax.set_yticklabels(NOTE_NAMES)

        # 🎯 网格
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.grid(True, axis='x', linestyle=':', alpha=0.3)

        # ===== 标记 t=0 =====
        ax.axvline(0, color='black', linestyle='--', alpha=0.7)

        # ===== x范围（包含 before 区域）=====
        min_t = -1.5
        max_t = np.max(start + sustain) + 0.5 if len(start) > 0 else 1
        ax.set_xlim(min_t, max_t)

    # ===== 绘图 =====
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    draw(axs[0], pred, "Prediction")
    draw(axs[1], target, "Ground Truth")

    axs[-1].set_xlabel("Time")

    plt.suptitle(title)
    plt.tight_layout()

    cfg = get_config()
    plt.savefig(os.path.join(cfg.save_dir, name + ".png"))
    plt.close()
    


def equip_cls(event, start, sustain, t):
    """
    event: (N) int cls_num
    start: (N) float
    sustain: (N) float
    t: (T) float
    """
    result = 12 * torch.ones_like(t, device=t.device).long()
    N = event.shape[0]
    for n in range(N):
        choice = (start[n] <= t) * (t <= sustain[n]) # (T,)
        result[choice] = event[n]
    return result

def equip_chord(chord, start, sustain, t):
    """
    chord: (N, 12) 0~1
    start: (N) float
    sustain: (N) float
    t: (T) float
    """
    C = chord.shape[-1]
    result = 12 * torch.ones((t.shape[0], C), device=t.device).float()
    N = chord.shape[0]
    for n in range(N):
        choice = (start[n] <= t) * (t <= sustain[n]) # (T,)
        result[choice, :] = chord[n, :].float()
    return result

def plot_pianoroll_timewise(output, target, title="timewise_compare", name="timewise_pred", chord_threshold=0.5):
    """
    output: Dict{
        "output": (B, T, 38)
        "time": (T,)
    }
    target: Dict{
        start: (N,)
        sustain: (N,)
        root: (N,)
        tonic: (N,)
        chord: (N, 12)
        before: (N,) bool   # 可有可无
    }

    上图：forward 输出 -> 合并成 event 后画
    下图：target 原始 event 直接画
    """

    cfg = get_config22()

    x = output["output"]
    t = output["time"]

    if x.dim() != 3:
        raise ValueError(f"output['output'] 应为 (B,T,38)，实际得到 {x.shape}")

    x = x[0]  # 只画 batch 第一个, (T, 38)

    root_logits = x[:, :13]       # (T, 13)
    chord_logits = x[:, 13:25]    # (T, 12)
    tonic_logits = x[:, 25:]      # (T, 13)

    root_pred = torch.argmax(root_logits, dim=-1)                 # (T,)
    tonic_pred = torch.argmax(tonic_logits, dim=-1)               # (T,)
    chord_pred = (torch.sigmoid(chord_logits) > chord_threshold)  # (T,12)

    t = t.detach().cpu()
    root_pred = root_pred.detach().cpu()
    tonic_pred = tonic_pred.detach().cpu()
    chord_pred = chord_pred.detach().cpu().bool()

    def timewise_to_event(root_seq, chord_seq, tonic_seq, t_axis):
        """
        把逐帧预测合并成 event，保持和原来的 plot_pianoroll_event 风格一致
        None 类 = 12，不画
        """
        T = len(t_axis)
        if T == 0:
            return {
                "root": torch.empty(0, dtype=torch.long),
                "tonic": torch.empty(0, dtype=torch.long),
                "chord": torch.empty(0, 12, dtype=torch.float32),
                "start": torch.empty(0, dtype=torch.float32),
                "sustain": torch.empty(0, dtype=torch.float32),
            }

        roots = []
        tonics = []
        chords = []
        starts = []
        sustains = []

        def same_state(i, j):
            return (
                int(root_seq[i]) == int(root_seq[j])
                and int(tonic_seq[i]) == int(tonic_seq[j])
                and torch.equal(chord_seq[i], chord_seq[j])
            )

        seg_start = 0
        for i in range(1, T):
            if not same_state(i - 1, i):
                r = int(root_seq[seg_start].item())
                tn = int(tonic_seq[seg_start].item())
                ch = chord_seq[seg_start].float()

                if r < 12 or ch.any():
                    t0 = float(t_axis[seg_start].item())
                    t1 = float(t_axis[i].item())
                    roots.append(r)
                    tonics.append(tn)
                    chords.append(ch)
                    starts.append(t0)
                    sustains.append(max(t1 - t0, 1e-6))

                seg_start = i

        # 最后一段
        r = int(root_seq[seg_start].item())
        tn = int(tonic_seq[seg_start].item())
        ch = chord_seq[seg_start].float()
        if r < 12 or ch.any():
            t0 = float(t_axis[seg_start].item())
            if T > 1:
                dt = float((t_axis[-1] - t_axis[-2]).item())
            else:
                dt = 0.05
            t1 = float(t_axis[-1].item() + dt)
            roots.append(r)
            tonics.append(tn)
            chords.append(ch)
            starts.append(t0)
            sustains.append(max(t1 - t0, 1e-6))

        if len(roots) == 0:
            return {
                "root": torch.empty(0, dtype=torch.long),
                "tonic": torch.empty(0, dtype=torch.long),
                "chord": torch.empty(0, 12, dtype=torch.float32),
                "start": torch.empty(0, dtype=torch.float32),
                "sustain": torch.empty(0, dtype=torch.float32),
            }

        return {
            "root": torch.tensor(roots, dtype=torch.long),
            "tonic": torch.tensor(tonics, dtype=torch.long),
            "chord": torch.stack(chords, dim=0).float(),
            "start": torch.tensor(starts, dtype=torch.float32),
            "sustain": torch.tensor(sustains, dtype=torch.float32),
        }

    pred_event = timewise_to_event(root_pred, chord_pred, tonic_pred, t)

    # target 直接用原始 event，不再 equip
    gt_event = {
        "root": target["root"].detach().cpu().long(),
        "tonic": target["tonic"].detach().cpu().long(),
        "chord": target["chord"].detach().cpu().float(),
        "start": target["start"].detach().cpu().float(),
        "sustain": target["sustain"].detach().cpu().float(),
    }

    def draw(ax, data, subtitle):
        root = data["root"]
        chord = data["chord"]
        tonic = data["tonic"]
        start = data["start"]
        sustain = data["sustain"]

        M = len(start)

        for i in range(M):
            t0 = float(start[i].item())
            dur = float(sustain[i].item())

            # chord: 蓝色块
            for p in range(12):
                if float(chord[i, p].item()) > 0.5:
                    ax.add_patch(
                        plt.Rectangle(
                            (t0, p - 0.4),
                            dur,
                            0.8,
                            color="skyblue",
                            alpha=0.6
                        )
                    )

            # root: 红色块；12 表示 None，不画
            r = int(root[i].item())
            if 0 <= r < 12:
                ax.add_patch(
                    plt.Rectangle(
                        (t0, r - 0.4),
                        dur,
                        0.8,
                        color="red",
                        alpha=0.9
                    )
                )

            # tonic: 绿色边框；12 表示 None，不画
            tn = int(tonic[i].item())
            if 0 <= tn < 12:
                ax.add_patch(
                    plt.Rectangle(
                        (t0, tn - 0.4),
                        dur,
                        0.8,
                        fill=False,
                        edgecolor="green",
                        linewidth=2
                    )
                )

        ax.set_title(subtitle)
        ax.set_ylim(-0.5, 11.5)
        ax.set_yticks(range(12))
        ax.set_yticklabels(NOTE_NAMES)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.grid(True, axis='x', linestyle=':', alpha=0.3)

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    draw(axs[0], pred_event, "Prediction")
    draw(axs[1], gt_event, "Ground Truth")

    # 统一 x 范围
    all_starts = []
    all_ends = []
    for data in [pred_event, gt_event]:
        if len(data["start"]) > 0:
            all_starts.append(float(data["start"].min().item()))
            all_ends.append(float((data["start"] + data["sustain"]).max().item()))
    if len(all_starts) > 0:
        axs[-1].set_xlim(min(all_starts), max(all_ends))

    axs[-1].set_xlabel("Time")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, name + ".png"))
    plt.close()