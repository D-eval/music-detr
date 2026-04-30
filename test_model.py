import torch
import librosa
import numpy as np
from mido import Message, MidiFile, MidiTrack

from configs.config import get_config21 as get_config
from models.detr4 import PitchTransformer
from spec import wav2cqt_2C, wav2spec_2C

cfg = get_config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =======================
# 🎧 音频加载
# =======================
def load_audio(path, sr=44100):
    wav, _ = librosa.load(path, sr=sr, mono=False)
    wav = wav.T
    return torch.tensor(wav, dtype=torch.float32)


# =======================
# 🎹 chord → MIDI notes
# =======================
def chord_to_notes(root, chord_vec, tonic):
    notes = [60 + root - 12]
    if tonic > 0:
        notes += [60 + tonic + 12]
    for i in range(12):
        if chord_vec[i]:
            pitch = 60 + i
            notes.append(pitch)
    return notes


# =======================
# 🔁 merge events
# =======================
def merge_events(events, time_th=0.3, conf_th=0.7):
    if len(events) == 0:
        return events

    # 先过滤低置信度
    events = [e for e in events if e["conf"] > conf_th]

    # 按时间排序
    events = sorted(events, key=lambda x: x["start"])

    merged = []

    for e in events:
        if len(merged) == 0:
            merged.append(e)
            continue

        last = merged[-1]

        # 时间重叠（关键：不是只看 start）
        overlap = (
            abs(e["start"] - last["start"]) < time_th
        )

        if overlap:
            # 🔥 选更高置信度
            if e["conf"] > last["conf"]:
                merged[-1] = e
        else:
            merged.append(e)

    return merged

# =======================
# 🎯 推理函数
# =======================
def infer_wav(model, wav, sr=44100, window_sec=5.0, hop_sec=0.5):
    model.eval()

    T = wav.shape[0]
    window_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)

    results = []

    for start in range(0, T - window_len, hop_len):
        end = start + window_len
        wav_seg = wav[start:end].to(device)[None, ...]

        with torch.no_grad():
            cqt, cqt_pos, cqt_freqs = wav2cqt_2C(wav_seg)
            spec, spec_pos, spec_freqs = wav2spec_2C(wav_seg)

            output = model(cqt, cqt_freqs, cqt_pos, spec, spec_freqs, spec_pos)[0]
            pred = model.infer(output)

        # ===== 坐标还原 =====
        start_pred = pred["start"].cpu() + start / sr
        sustain_pred = pred["sustain"].cpu()
        before = pred["before"].cpu()
        print(len(start_pred))
        for i in range(len(start_pred)):
            if before[i]:
                continue  # 🔥忽略 before

            results.append({
                "start": start_pred[i].item(),
                "sustain": sustain_pred[i].item(),
                "root": pred["root"][i].item(),
                "chord": pred["chord"][i].cpu(),
                "tonic": pred["tonic"][i].item(),
                "conf": pred["exist"][i].item(),  # ✔ 直接用
            })

    results = merge_events(results)
    return results


# =======================
# 🎼 转 MIDI
# =======================
def events_to_midi(events, out_path="output.mid"):
    from mido import Message, MidiFile, MidiTrack

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    ticks_per_beat = 480
    bpm = 120

    def sec_to_tick(sec):
        return int(sec * ticks_per_beat * bpm / 60)

    events_all = []

    # ===== 1️⃣ 转成 note 事件（绝对时间）=====
    for e in events:
        notes = chord_to_notes(e["root"], e["chord"], e['tonic'])

        start = sec_to_tick(e["start"])
        end = sec_to_tick(e["start"] + e["sustain"])

        for note in notes:
            events_all.append(("on", note, start))
            events_all.append(("off", note, end))

    # ===== 2️⃣ 按时间排序 =====
    events_all.sort(key=lambda x: x[2])

    # ===== 3️⃣ 转 delta time =====
    last_time = 0
    for typ, note, t in events_all:
        delta = t - last_time
        assert delta >= 0, f"负时间: {delta}"

        if typ == "on":
            track.append(Message("note_on", note=note, velocity=64, time=delta))
        else:
            track.append(Message("note_off", note=note, velocity=64, time=delta))

        last_time = t

    mid.save(out_path)
    print("Saved MIDI:", out_path)


from read0 import AudioDataset, collate_fn, to_device
from torch.utils.data import DataLoader
dataset = AudioDataset(root_dir=cfg.dataset_data_path, cfg=cfg)



mp3_path = "/home/vipuser/wby/proj_params/musicNotebook/save/music_note/audio/Dirty Androids,ぷにぷに電機 - On The West Coastline.mp3"

print("Loading model...")
model = PitchTransformer().to(device)

checkpoint = torch.load("../params/detr21/ckpt_epoch.pt", map_location=device)
model.load_state_dict(checkpoint)

print("Loading audio...")
wav = load_audio(mp3_path)

print("Running inference...")
events = infer_wav(model, wav)

print("Events:", len(events))

print("Exporting MIDI...")
events_to_midi(events)

