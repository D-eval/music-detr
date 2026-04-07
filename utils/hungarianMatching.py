"""
hungarianMatching:
events: (N, 4) [start_time, sustain_time, pitch, text_id]
output: (T, P, 2) [是否触发，sustain时间 /s]

output = {
    'text_out': query_out_text, # (B, Q, C_text)
    'event_out': query_out_event, # (B, Q, 2)
    'pitch_logits': query_out_pitch, # (B, Q, P+1)
    'hidden': hidden_text, # (B, Q, C)
    'exist': query_out_exist # (B, Q, 1)
}
"""

import torch
import math
from configs.config import get_config

from .pitchDist import pitch_dist_euler


def pitch_dist(p1, p2):
    # 对于标注不准，半音偏差
    dp = abs(p1 - p2)
    _dp = pitch_dist_euler(p1, p2)
    if dp <= 2:
        return dp
    else:
        return _dp


def concat_output(output, times):
    """
    output: (T, P, 2) [是否触发，log sustain /s]
    times: (T) [time /s]
    return: (T, P, 4) [触发概率，time，log sustain /s，, pitch]
    """
    cfg = get_config()
    sr = cfg.sr
    T, P, _ = output.shape
    pitch = torch.arange(P, device=output.device) + cfg.min_midi # (P)
    res = torch.cat([output, times.view(T, 1, 1).repeat(1, P, 1), pitch.view(1, P, 1).repeat(T, 1, 1)], dim=-1)
    return res.permute(0,2,1,3)



def cost(event_pred, event_gt):
    """
    event_pred: (4,) [logits, start_time, log_sustain, pitch]
    event_gt: (3,) [start_time, log_sustain, pitch]
    prob 越大，cost 越小
    IoU 越高，cost 越小
    """
    logits, start_pred, log_sustain_pred, pitch_pred = event_pred
    start_gt, log_sustain_gt, pitch_gt = event_gt
    
    d_pitch = pitch_dist(pitch_pred, pitch_gt)
    d_time = abs(start_pred - start_gt)
    d_sustain = abs(log_sustain_pred - log_sustain_gt)
    d_prob = -torch.sigmoid(logits)

    return (
        2.0 * d_prob +
        1.0 * d_time +
        0.1 * d_sustain +
        0.3 * d_pitch
    )

