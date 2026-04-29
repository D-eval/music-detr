"""
linear attention

Cell
inter: (L_inter, C)
inner: (L_inner, C)

ecosystem
cells: List[Cell]
envir: (..., C)
"""


import torch
from torch import nn
from typing import List

class Cell:
    def __init__(self,
                 inner_state,
                 inter_state):
        super().__init__()
        self.inner_state = inner_state # (B, L_inner, C)
        self.inter_state = inter_state # (B, L_inter, C)
        assert self.inner_state.shape[2] == self.inter_state.shape[2]


def inner_attn(cells: List[Cell],
            inner_decoder_layer: nn.Module):
    new_cells = []
    for cell in cells:
        temp_data = torch.cat([cell.inter_state, cell.inner_state], dim=1) # (B, L_inner+L_inter, C)
        new_cells.append(inner_decoder_layer(temp_data))
    return new_cells
