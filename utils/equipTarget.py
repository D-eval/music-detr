import torch

from configs.config import get_config


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
        time_idx = torch.argmin(torch.abs(pitch_centre[0,:][:,None] - starts_idx[None,:]), dim=0)
        pitch_idx = (temp_events[:, 2] - cfg.min_midi).to(int)
        target_pitchMap = torch.zeros((pitch_centre.shape[1], cfg.pitch_vocab_size + 1, 2))
        for i in range(starts.shape[0]):
            target_pitchMap[time_idx[i], pitch_idx[i], 0] = 1
            target_pitchMap[time_idx[i], pitch_idx[i], 1] = temp_events[i, 1]
        target_pitchMap_lst.append(target_pitchMap)
    # (T,P,2): 是否触发，sustain时间 /s
    return torch.stack(target_pitchMap_lst)
