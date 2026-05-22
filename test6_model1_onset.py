import torch
import librosa
import numpy as np
from mido import Message, MidiFile, MidiTrack

from configs.config6 import get_config
from models.detr6 import CQTEncoder as Model
from spec.cqt import MultiWindowCQT, get_freqs


ticks_per_beat = 480
bpm = 120
def sec_to_tick(sec):
    return sec * ticks_per_beat * bpm / 60


audio_path = '/Users/broyou/Music/网易云音乐/Dirty Androids,ぷにぷに電機 - On The West Coastline.mp3'

checkpoint_path = "../params/detr6/baby7.pt"

sr = 44100

window_len = int(sr * 3) # 3s

stride = window_len // 3


cfg = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(cfg).to(device)
model.set_mode("onset")
model.outputShape = "BTD"

freqs = get_freqs(cfg.min_midi, cfg.max_midi)
preprocessor = MultiWindowCQT(freqs, cfg.sr, cfg.window_num, cfg.min_cycle, stride=cfg.cqt_stride).to(device)

state_dict = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(state_dict=state_dict, strict=False)
model.eval()

audio, sr = librosa.load(audio_path, mono=False, sr=sr)
audio = torch.tensor(audio).T[None,...]
audio = audio.to(device)[:,:,:]
total_time = audio.shape[1]

time_starts_idx = np.arange(0, total_time, window_len)
time_ends_idx = time_starts_idx + window_len

time_starts = time_starts_idx / sr
ticks_starts = sec_to_tick(time_starts).astype(int)
ticks_interval = np.pad((ticks_starts[1:] - ticks_starts[:-1]), (1,0))
all_onsets = []

for i, start_idx in enumerate(time_starts_idx):
    temp_audio = audio[:, start_idx:start_idx+window_len, :]
    if (temp_audio**2).sum()<1e-3:
        all_onsets.append([])
        continue

    temp_time = start_idx / sr
    x, _, freqs = preprocessor(temp_audio.to(device))
    model.eval()
    output = model(x)
    infer_output = model.infer(x=output)
    if len(infer_output['onsets']) > 0:
        times = [temp['time'] for temp in infer_output['onsets']]
        times = [time + temp_time for time in times]
    else:
        times = []
    print(times)
    all_onsets.append(times)
    
    # output = model(temp_audio)
    # temp_infer = model.infer(output)
    # pitchs = temp_infer['midi'].cpu().tolist()
    # all_pitchs.append(pitchs)

# mid = MidiFile()
# track = MidiTrack()
# mid.tracks.append(track)

# for dt, ps in zip(ticks_interval, all_pitchs):
#     if len(ps)==0:
#         track.append(Message("note_on", note=None, velocity=64, time=delta))
#     for i, p in enumerate(ps):
#         delta = dt if i==0 else 0
#         track.append(Message("note_on", note=p, velocity=64, time=delta))
# mid.save("./temp.midi")

all_onsets = sum(all_onsets, [])

merge_threshold = 0.05  # 50ms

merged_onsets = []

for t in all_onsets:
    if len(merged_onsets) == 0:
        merged_onsets.append(t)
        continue
    # 太近
    if (
        t
        - merged_onsets[-1]
        < merge_threshold
    ):
        # average merge
        merged_onsets[-1] = (
            merged_onsets[-1]
            + t
        ) / 2
    else:
        merged_onsets.append(t)


# mid
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)
prev_tick = 0
for sec in merged_onsets:
    tick = int(
        sec_to_tick(sec)
    )
    dt = tick - prev_tick
    track.append(
        Message(
            "note_on",
            note=42,
            velocity=100,
            time=dt
        )
    )
    track.append(
        Message(
            "note_off",
            note=42,
            velocity=0,
            time=30
        )
    )
    prev_tick = tick + 30
mid.save("./onset.mid")