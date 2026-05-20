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

checkpoint_path = "../params/detr6/baby6.pt"

sr = 44100

window_len = int(sr * 0.5) # 0.5s
stride = window_len

cfg = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(cfg).to(device)
model.set_mode("midi")

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
all_pitchs = []

for i, start_idx in enumerate(time_starts_idx):
    temp_audio = audio[:, start_idx:start_idx+window_len, :]
    if (temp_audio**2).sum()<1e-3:
        all_pitchs.append([])
        continue

    x, _, freqs = preprocessor(temp_audio.to(device))
    model.eval()
    output = model(x)
    infer_output = model.infer(x=output)
    pitchs = infer_output['midi']
    print(pitchs)
    all_pitchs.append(pitchs)
    
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

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

prev_ps = set()

for dt, ps in zip(ticks_interval, all_pitchs):

    ps = set(ps)

    note_on = ps - prev_ps
    note_off = prev_ps - ps

    first = True

    # note off
    for p in note_off:

        delta = dt if first else 0

        track.append(
            Message(
                "note_off",
                note=int(p),
                velocity=0,
                time=int(delta)
            )
        )

        first = False

    # note on
    for p in note_on:

        delta = dt if first else 0

        track.append(
            Message(
                "note_on",
                note=int(p),
                velocity=64,
                time=int(delta)
            )
        )

        first = False

    # nothing changed
    if first:
        track.append(
            Message(
                "note_off",
                note=0,
                velocity=0,
                time=int(dt)
            )
        )

    prev_ps = ps

# 最后关闭所有 note
for p in prev_ps:

    track.append(
        Message(
            "note_off",
            note=int(p),
            velocity=0,
            time=0
        )
    )

mid.save("./temp.midi")