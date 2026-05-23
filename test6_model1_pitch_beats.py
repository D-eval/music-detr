import torch
import librosa
import numpy as np

from mido import (
    Message,
    MidiFile,
    MidiTrack
)

from configs.config6 import get_config
from models.detr6 import CQTEncoder as Model
from spec.cqt import MultiWindowCQT, get_freqs


# =========================================================
# Config
# =========================================================

audio_path = '/Users/broyou/Music/网易云音乐/Dirty Androids,ぷにぷに電機 - On The West Coastline.mp3'

checkpoint_path = "../params/detr6/baby8.pt"

output_midi = "./temp.mid"

sr = 44100

ticks_per_beat = 480
bpm = 120

# 1 sec -> ticks
TICKS_PER_SECOND = ticks_per_beat * bpm / 60

HIHAT = 42

# =========================================================
# Utils
# =========================================================

def sec_to_tick(sec):
    return sec * TICKS_PER_SECOND


# =========================================================
# Load Model
# =========================================================

cfg = get_config()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = Model(cfg).to(device)

model.set_mode("pitch")
model.outputShape = "BT12D"

freqs = get_freqs(
    cfg.min_midi,
    cfg.max_midi
)

preprocessor = MultiWindowCQT(
    freqs,
    cfg.sr,
    cfg.window_num,
    cfg.min_cycle,
    stride=cfg.cqt_stride
).to(device)

state_dict = torch.load(
    checkpoint_path,
    map_location="cpu"
)

model.load_state_dict(
    state_dict=state_dict,
    strict=False
)

model.eval()

# =========================================================
# Load Audio
# =========================================================

print("loading audio...")

audio_np, sr = librosa.load(
    audio_path,
    mono=False,
    sr=sr
)

# mono for beat tracking
if audio_np.ndim == 2:
    mono_audio = audio_np.mean(0)
else:
    mono_audio = audio_np

# stereo tensor for model
audio = torch.tensor(audio_np).T[None, ...]

audio = audio.to(device)

# =========================================================
# Beat Tracking
# =========================================================

print("beat tracking...")

tempo, beat_samples = librosa.beat.beat_track(
    y=mono_audio,
    sr=sr,
    units="samples"
)

beat_samples = np.asarray(beat_samples)

print("tempo:", tempo)
print("beats:", len(beat_samples))

# 至少两个 beat 才能形成 interval
if len(beat_samples) < 2:
    raise RuntimeError("Not enough beats detected.")

segment_starts = beat_samples[:-1]
segment_ends = beat_samples[1:]

beat_times = segment_starts / sr

ticks_starts = sec_to_tick(
    beat_times
).astype(int)

ticks_interval = np.diff(
    np.pad(ticks_starts, (1, 0))
)

# =========================================================
# Infer Pitch
# =========================================================

all_pitchs = []

print("infer pitches...")

for i, (start_idx, end_idx) in enumerate(
    zip(segment_starts, segment_ends)
):

    temp_audio = audio[
        :,
        start_idx:end_idx,
        :
    ]

    # silence skip
    if (temp_audio ** 2).mean() < 1e-6:

        all_pitchs.append([])

        continue

    with torch.no_grad():

        x, _, _ = preprocessor(
            temp_audio.to(device)
        )

        output = model(x)

        infer_output = model.infer(
            x=output
        )

        pitchs = infer_output['midi']

    pitchs = list(set([
        int(p)
        for p in pitchs
        if 0 <= int(p) <= 127
    ]))

    print(i, pitchs)

    all_pitchs.append(pitchs)

# =========================================================
# MIDI
# =========================================================

mid = MidiFile()

# ---------------------------------------------------------
# Track 1 : Beat
# ---------------------------------------------------------

beat_track = MidiTrack()

mid.tracks.append(beat_track)

# ---------------------------------------------------------
# Track 2 : Notes
# ---------------------------------------------------------

note_track = MidiTrack()

mid.tracks.append(note_track)

# =========================================================
# Beat Track
# =========================================================

for i, dt in enumerate(ticks_interval):

    beat_track.append(
        Message(
            "note_on",
            channel=9,
            note=HIHAT,
            velocity=100,
            time=int(dt)
        )
    )

    beat_track.append(
        Message(
            "note_off",
            channel=9,
            note=HIHAT,
            velocity=0,
            time=20
        )
    )

# =========================================================
# Note Track
# =========================================================

prev_ps = set()

for dt, ps in zip(
    ticks_interval,
    all_pitchs
):

    ps = set(ps)

    note_on = ps - prev_ps

    note_off = prev_ps - ps

    first = True

    # -----------------------------------------------------
    # note off
    # -----------------------------------------------------

    for p in note_off:

        delta = dt if first else 0

        note_track.append(
            Message(
                "note_off",
                note=int(p),
                velocity=0,
                time=int(delta)
            )
        )

        first = False

    # -----------------------------------------------------
    # note on
    # -----------------------------------------------------

    for p in note_on:

        delta = dt if first else 0

        note_track.append(
            Message(
                "note_on",
                note=int(p),
                velocity=64,
                time=int(delta)
            )
        )

        first = False

    # -----------------------------------------------------
    # nothing changed
    # -----------------------------------------------------

    if first:

        note_track.append(
            Message(
                "note_off",
                note=0,
                velocity=0,
                time=int(dt)
            )
        )

    prev_ps = ps

# =========================================================
# Final Note Off
# =========================================================

for p in prev_ps:

    note_track.append(
        Message(
            "note_off",
            note=int(p),
            velocity=0,
            time=0
        )
    )

# =========================================================
# Save
# =========================================================

mid.save(output_midi)

print("saved:", output_midi)