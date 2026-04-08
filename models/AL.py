"""
带有group query和训练策略
"""

import torch
from configs.config import get_config
from torch import nn
import math
import torch.nn.functional as F
from typing import Callable, Optional, Union, Dict

from .detr2 import PitchTransformer

from .llm import LanguageModel
from spec import wav2cqt, wav2spec

class ALUnion(nn.Module):
    def __init__(self):
        super().__init__()
        self.detr = PitchTransformer()
        self.lm = LanguageModel()
    def forward(self, audio):
        """
            audio: (B, T)
        """
        detr_output = self.detr_forward(audio)
        
    def detr_forward(self, audio):
        """
            audio: (B, T)
        """
        pitch_spec, pitch_centre, pitchs = wav2cqt(audio)
        freq_spec, freq_centre, freqs = wav2spec(audio)
        input_dict = {
            "pitch_spec": pitch_spec, # (B, T, F)
            "pitchs": pitchs,
            "pitch_centre": pitch_centre,
            "freq_spec": freq_spec, # (B, T, F)
            "freqs": freqs,
            "freq_centre": freq_centre,
        }
        detr_output = self.detr(**input_dict)
        return detr_output
    