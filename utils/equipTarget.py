import torch

from configs.config import get_config
from typing import List

def get_target_map(events, pitch_centre):
    # temp_events: (num_event, 4)
    target_pitchMap_lst = []
    for text_id in range(events[0][:,-1].unique().shape[0]):
        text_id_idx = (events[0][:, 3] == text_id)
        cfg = get_config()
        sr = cfg.sr
        temp_events = events[0][text_id_idx]
        starts = temp_events[:,0]
        starts_idx = starts * sr
        window_len = cfg.window_len
        total_len = cfg.wav_len
        valid = (starts_idx > window_len//2) & \
                (starts_idx < total_len - window_len//2)
        temp_events = temp_events[valid,:]
        starts = temp_events[:,0]
        starts_idx = starts * sr
        time_idx = torch.argmin(torch.abs(pitch_centre[:,None] - starts_idx[None,:]), dim=0)
        pitch_idx = (temp_events[:, 2] - cfg.min_midi).to(int)
        target_pitchMap = torch.zeros((pitch_centre.shape[0], cfg.pitch_vocab_size + 1, 2))
        for i in range(starts.shape[0]):
            target_pitchMap[time_idx[i], pitch_idx[i], 0] = 1
            target_pitchMap[time_idx[i], pitch_idx[i], 1] = temp_events[i, 1]
        target_pitchMap_lst.append(target_pitchMap)
    # (T,P,2): 是否触发，sustain时间 /s
    return torch.stack(target_pitchMap_lst)



def get_sustain_map(events, pitch_centre):
    # temp_events: (num_event, 4)
    # (T, P, 2)
    target_pitchMap_lst = []
    for text_id in range(events[0][:,-1].unique().shape[0]):
        text_id_idx = (events[0][:, 3] == text_id)
        cfg = get_config()
        sr = cfg.sr
        temp_events = events[0][text_id_idx]
        starts = temp_events[:,0]
        starts_idx = starts * sr
        window_len = cfg.window_len
        total_len = cfg.wav_len
        valid = (starts_idx > window_len//2) & \
                (starts_idx < total_len - window_len//2)
        temp_events = temp_events[valid,:]
        starts = temp_events[:,0]
        starts_idx = starts * sr
        time_idx = torch.argmin(torch.abs(pitch_centre[:,None] - starts_idx[None,:]), dim=0)
        pitch_idx = (temp_events[:, 2] - cfg.min_midi).to(int)
        target_pitchMap = torch.zeros((pitch_centre.shape[0], cfg.pitch_vocab_size + 1, 2))
        for i in range(starts.shape[0]):
            target_pitchMap[time_idx[i], pitch_idx[i], 0] = 1 # start
            target_pitchMap[time_idx[i], pitch_idx[i], 1] = 1 # sustain
            
            time_sustain = temp_events[i, 1]
            time_sustain_frames_num = int(time_sustain * sr // cfg.stride)

            start = max(0, time_idx[i])
            end = min(pitch_centre.shape[0], start + time_sustain_frames_num)

            if end >= start:
                target_pitchMap[start:end, pitch_idx[i], 1] = 1
            else:
                raise ValueError("wtf")
            
        target_pitchMap_lst.append(target_pitchMap)
    # (T,P,2): 是否触发，sustain时间 /s
    return torch.stack(target_pitchMap_lst)



def get_sustain_map_textwise(events, pitch_centre):
    # events: List[(N, 3)]
    # (T, P, 2)
    target_pitchMap_lst = []
    cfg = get_config()
    for text_id in range(len(events)):
        temp_events = events[text_id]
        pitchless = temp_events[0, 2] < 0

        sr = cfg.sr

        starts = temp_events[:,0]
        starts_idx = starts * sr
        window_len = cfg.window_len
        total_len = cfg.wav_len
        valid = (starts_idx > window_len//2) & \
                (starts_idx < total_len - window_len//2)
        temp_events = temp_events[valid,:]
        starts = temp_events[:,0]
        starts_idx = starts * sr
        time_idx = torch.argmin(torch.abs(pitch_centre[:,None] - starts_idx[None,:]), dim=0)
        pitch_idx = (temp_events[:, 2] - cfg.min_midi).to(int)
        pitch_idx = -1 if pitchless else pitch_idx
        
        target_pitchMap = torch.zeros((pitch_centre.shape[0], cfg.pitch_vocab_size + 1, 2))
        for i in range(starts.shape[0]):
            if pitchless:
                target_pitchMap[time_idx[i], -1, 0] = 1 # start
                target_pitchMap[time_idx[i], -1, 1] = 1 # sustain
            else:
                target_pitchMap[time_idx[i], pitch_idx[i], 0] = 1 # start
                target_pitchMap[time_idx[i], pitch_idx[i], 1] = 1 # sustain
            
            time_sustain = temp_events[i, 1]
            time_sustain_frames_num = int(time_sustain * sr // cfg.stride)

            start = max(0, time_idx[i])
            end = min(pitch_centre.shape[0], start + time_sustain_frames_num)

            if end >= start:
                if pitchless:
                    target_pitchMap[start:end, -1, 1] = 1
                else:
                    target_pitchMap[start:end, pitch_idx[i], 1] = 1
            else:
                raise ValueError("wtf")
            
        target_pitchMap_lst.append(target_pitchMap)
    # (T,P,2): 是否触发，sustain时间 /s
    return torch.stack(target_pitchMap_lst)

