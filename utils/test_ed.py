import matplotlib.pyplot as plt
import os
import torch
import cv2
import numpy as np
import math
from pitchDist import build_euler_cost_matrix

from midi import midi2freq

midis = torch.arange(24, 100)
freqs = midi2freq(midis)
ed_matrix = build_euler_cost_matrix(freqs)

import matplotlib.pyplot as plt
plt.imshow(ed_matrix)
plt.show()