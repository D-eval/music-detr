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
checkpoint_path = "../params/detr6/baby.pt"
state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict=state_dict)

# (K, 1, N)
a = model.layers[0].harmony_conv.kernels[2,0,:].cpu().detach().numpy()
plt.plot(a)
plt.savefig("./tiny_save/wtf.pdf")
# plt.close()

all_fea = []
all_label = []
for i in range(200):
    audio, target = dataset.getitem(i)
    audio = audio.to(device)
    with torch.no_grad():
        x, _, _ = preprocessor(audio.unsqueeze(0))
        fea = model(x)
        fea = fea.mean(1)
        fea = dilation_pool(fea, 12)  # (1, 12, 512)
    fea = fea[0]  # (12, 512)
    root = target["root"].item()
    for pc in range(12):
        all_fea.append(
            fea[pc].detach().cpu()
        )
        # label:
        # 当前 sample 的真实 root
        # 以及 feature 属于哪个 pitch class
        all_label.append(
            (root, pc)
        )


import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X = torch.stack(all_fea).numpy()  # (N, 512)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=0,
)

X_2d = tsne.fit_transform(X)



plt.figure(figsize=(10, 10))

for i, (root, pc) in enumerate(all_label):

    x = X_2d[i, 0]
    y = X_2d[i, 1]

    color = "red" if root==pc else "blue"

    plt.scatter(x, y, color=color)

    plt.text(
        x,
        y,
        f"{pc}",
        fontsize=8,
    )

plt.title("Pitch-class feature t-SNE")
plt.savefig("./tiny_save/wtf.pdf")