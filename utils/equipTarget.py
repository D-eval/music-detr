import torch

from configs.config import get_config
from typing import List

# for detr
def to_device(batch, device):
    # List[Dict]
    if torch.is_tensor(batch):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {
            k: to_device(v, device)
            for k, v in batch.items()
            if k != "text"  # ⚠️ 非 tensor 跳过
        } | {
            "text": batch["text"]
        }
    elif isinstance(batch, list):
        return [to_device(x, device) for x in batch]
    else:
        return batch


def embed_text(targets, tokenizer):
    # targets: list
    for sample in targets:
        texts = sample['text']
        _, text_emb = tokenizer(None, texts)
        sample['text_emb'] = torch.concat(text_emb, dim=0)
        # 原地操作


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



def normalize_targets_pitch(targets):
    """
    targets: list[dict]
    每个 target["pitch"]: (N,) tensor (long)

    处理规则：
    1. 如果 pitch == -1 → 设为 vocab_size（最后一类）
    2. 否则：
        - 用 ±12 折叠到 [min_midi, max_midi]
        - 再减去 min_midi → 映射到 [0, vocab_size-1]
    """
    cfg = get_config()

    min_midi = cfg.min_midi
    max_midi = cfg.max_midi
    vocab_size = cfg.pitch_vocab_size  # = max - min + 1

    for t in targets:
        pitch = t["pitch"]  # (N,)

        # ===== 1. 处理 -1 =====
        neg_mask = (pitch == -1)

        # ===== 2. fold 到合法区间 =====
        valid_mask = ~neg_mask
        p = pitch[valid_mask]

        # 折叠到区间（按12循环）
        # while版 → vector版
        # 公式：((p - min) % 12) + min + k*12  → 再 clamp
        # 更简单：循环逼近
        while True:
            too_low = p < min_midi
            too_high = p > max_midi

            if not (too_low.any() or too_high.any()):
                break

            p[too_low] += 12
            p[too_high] -= 12

        pitch[valid_mask] = p

        # ===== 3. 映射到 index =====
        pitch[valid_mask] = pitch[valid_mask] - min_midi  # → [0, vocab_size-1]

        # ===== 4. -1 → 最后一类 =====
        pitch[neg_mask] = vocab_size

        # ===== 5. 写回 =====
        t["pitch"] = pitch.long()

    return targets


def render_pred_group_pitch_map(events, pitch_centre):
    """
    events: List[Dict[str, Tensor]]
    pitch_centre: (T,)  每帧对应 sample index

    return:
        (T, P+1, 2)
    """
    pred_map = 0
    for event in events:
        pred_map += render_pred_pitch_map(event, pitch_centre)
    return pred_map
    
def render_pred_pitch_map(events, pitch_centre):
    """
    events: Dict[str, Tensor]
    pitch_centre: (T,)  每帧对应 sample index

    return:
        (T, P+1, 2)
    """

    cfg = get_config()
    sr = cfg.sr
    stride = cfg.stride

    T = pitch_centre.shape[0]
    P = cfg.pitch_vocab_size + 1

    device = pitch_centre.device

    # ===== 1. 取数据 =====
    start = events["start"]        # (M,)
    sustain = events["sustain"]    # (M,)
    pitch = events["pitch"]        # (M,)

    if start.numel() == 0:
        return torch.zeros((T, P, 2), device=device)

    # ===== 2. clamp =====
    # start = torch.clamp(start, 0, 1)
    # sustain = torch.clamp(sustain, 0, 1)

    # ===== 3. 转 sample =====
    start_idx_sample = start * sr  # (M,)

    # ===== 4. 找最近时间帧（vectorized）=====
    # (T, M)
    diff = torch.abs(pitch_centre[:, None] - start_idx_sample[None, :])
    time_idx = torch.argmin(diff, dim=0)  # (M,)

    # ===== 5. sustain → frame数 =====
    sustain_frames = (sustain * sr // stride).long()  # (M,)

    # ===== 6. pitch index =====
    pitch_idx = torch.clamp(pitch, 0, P-1)

    # ===== 7. 初始化 =====
    pred_map = torch.zeros((T, P, 2), device=device)

    # ===== 8. trigger（scatter）=====
    pred_map[time_idx, pitch_idx, 0] = 1
    pred_map[time_idx, pitch_idx, 1] = 1

    # ===== 9. sustain（向量化区间填充）=====
    # 构造 (T, M)

    start_t = time_idx                # (M)
    end_t = time_idx + sustain_frames # (M)
    end_t = torch.clamp(end_t, 0, T)

    for m in range(start_idx_sample.shape[0]):
        pred_map[start_t[m]:end_t[m], pitch_idx[m], 1] = 1

    return pred_map