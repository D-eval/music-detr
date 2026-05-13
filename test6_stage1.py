import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

def to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {
            k: to_device(v, device)
            for k, v in batch.items()
        }
    elif isinstance(batch, list):
        return [to_device(x, device) for x in batch]
    else:
        return batch


import torch
import torch.nn as nn
import torch.optim as optim

from utils.trainRecorder import TrainingRecorder
from utils.visualizer import plot_pianoroll_event

import os
from configs.config6 import get_config
cfg = get_config()

import sys
sys.path.append(str(cfg.dataset_read_py_path_stage1))

from read0 import RandomChordSynthDataset, chord_collate_fn
from torch.utils.data import DataLoader
dataset = RandomChordSynthDataset(prototype_dir=cfg.dataset_read_py_path_stage1 / "prototype",
                                  soundfont_dir=cfg.dataset_read_py_path_stage1 / "soundfonts",
                                  sample_rate=cfg.sr,
                                  min_midi=cfg.min_midi,
                                  max_midi=cfg.max_midi)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=chord_collate_fn,
    pin_memory=True
)

from models.detr6 import CQTEncoder, dilation_pool
from spec import wav2cqt_2C, wav2spec_2C
from spec.cqt import MultiWindowCQT, get_freqs
from models.teacher import Teacher
from utils.equipTarget import get_target_map, get_sustain_map, get_sustain_map_textwise, normalize_targets_pitch, render_pred_pitch_map, render_pred_group_pitch_map, embed_text

freqs = get_freqs(cfg.min_midi, cfg.max_midi)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)

preprocessor = MultiWindowCQT(freqs, cfg.sr, cfg.window_num, cfg.min_cycle).to(device)
model = CQTEncoder(cfg).to(device)

# for audio, target in loader:
#     x, _, freqs = preprocessor(audio.to(device))
#     assert 0


# teacher = Teacher()
checkpoint_path = "../params/detr6/baby3.pt"
state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict=state_dict)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
)

all_fea = []
all_label = []
for i, (audio, target) in enumerate(loader):
    if i >= 200:
        break
    audio = audio.to(device)
    with torch.no_grad():
        x, _, _ = preprocessor(audio)
        fea = model(x) # (1, T, D)
        fea = fea.mean(1) # (1, D)
    fea = fea[0]  # (12, 512)
    root = target["root"].item()
    chord = target["chord_cls"].item()

    all_fea.append(
        fea.detach().cpu()
    )
    all_label.append(
        (root, chord)
    )


import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

X = torch.stack(all_fea,dim=0).numpy()  # (N, 512)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=0,
)

X_2d = tsne.fit_transform(X) # (N,2)

# 两个subplot，root和chord
# root和chord用不同的color map
root_names = [
    "C","C#","D","D#","E","F",
    "F#","G","G#","A","A#","B"
]

chord_names = [
    "maj",
    "min",
    "dom",
    "dim",
    "aug"
]

# color maps
root_cmap = cm.get_cmap("tab20", 12)
chord_cmap = cm.get_cmap("Set1", 5)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(18, 8)
)

# =========================================
# root
# =========================================

ax = axes[0]

for i, (root, chord) in enumerate(all_label):

    x = X_2d[i, 0]
    y = X_2d[i, 1]

    color = root_cmap(root)

    ax.scatter(
        x,
        y,
        color=color,
        s=40,
        alpha=0.8,
    )

    ax.text(
        x,
        y,
        root_names[root],
        fontsize=7,
    )

ax.set_title("t-SNE colored by ROOT")

# =========================================
# chord
# =========================================

ax = axes[1]

for i, (root, chord) in enumerate(all_label):

    x = X_2d[i, 0]
    y = X_2d[i, 1]

    color = chord_cmap(chord)

    ax.scatter(
        x,
        y,
        color=color,
        s=40,
        alpha=0.8,
    )

    ax.text(
        x,
        y,
        chord_names[chord],
        fontsize=7,
    )

ax.set_title("t-SNE colored by CHORD")

plt.tight_layout()

plt.savefig("./tiny_save/wtf.pdf")
plt.close()
