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


audio_path = '/Users/broyou/Music/Music/Media.localized/Music/tokyona/Cold Lilac Bloom Core, Pt. 1/Perpetual Descent.mp3'

checkpoint_path = "../params/detr6/baby4.pt"

sr = 44100

window_len = int(sr * 0.5) # 0.5s
stride = window_len

cfg = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(cfg).to(device)
state_dict = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(state_dict=state_dict, strict=False)
model.eval()
model.set_mode("symbol")

freqs = get_freqs(cfg.min_midi, cfg.max_midi)
preprocessor = MultiWindowCQT(freqs, cfg.sr, cfg.window_num, cfg.min_cycle, stride=cfg.cqt_stride).to(device)


audio, sr = librosa.load(audio_path, mono=False, sr=sr)
audio = torch.tensor(audio).T[None,...]
audio = audio.to(device)[:,:44100*10,:]
total_time = audio.shape[1]

time_starts_idx = np.arange(0, total_time, window_len)
time_ends_idx = time_starts_idx + window_len

time_starts = time_starts_idx / sr
ticks_starts = sec_to_tick(time_starts).astype(int)
ticks_interval = np.pad((ticks_starts[1:] - ticks_starts[:-1]), (1,0))
all_pitchs = []

chord_note_lst = [
    np.array([0, 4, 7]),
    np.array([0, 3, 7]),
    np.array([0, 4, 7, 10]),
    np.array([0, 3, 6]),
    np.array([0, 4, 8]),
    np.array([0, 7]),
]

for i, start_idx in enumerate(time_starts_idx):
    temp_audio = audio[:, start_idx:start_idx+window_len, :]
    if (temp_audio**2).sum()<1e-3:
        all_pitchs.append([])
        continue
    cqt, _, _ = preprocessor(temp_audio)
    output = model(cqt)
    
    # list pitch
    temp_infer = model.infer(x=output)
    if not temp_infer["exist"]:
        pitchs = []
    else:
        root = temp_infer['root_idx']
        chord_idx = temp_infer['chord_idx']

        chord_note = chord_note_lst[chord_idx] + root
        pitchs = [root + 72 - 12] + (chord_note + 72).tolist()
        
    all_pitchs.append(pitchs)

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

for dt, ps in zip(ticks_interval, all_pitchs):
    for i, p in enumerate(ps):
        delta = dt if i==0 else 0
        track.append(Message("note_on", note=p, velocity=64, time=delta))
mid.save("./temp.midi")
