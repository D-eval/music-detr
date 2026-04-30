import torch

def infer_wav(model, wav, sr=44100, window_sec=5.0, hop_sec=1.0, device="cuda"):
    """
    wav: (T,)
    return: merged events
    """
    model.eval()

    T = wav.shape[0]
    window_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)

    results = []

    for start in range(0, T, hop_len):
        end = start + window_len

        if end > T:
            break  # 或 padding，看你需求

        wav_seg = wav[start:end]

        # ==== 这里你要接你自己的前处理 ====
        # pitch_spec, freq_spec, ...
        inputs = preprocess(wav_seg, sr)  # 你自己已有的

        with torch.no_grad():
            output = model(**inputs)[0]
            pred = model.infer(output)

        # ==== 坐标还原到全局 ====
        start_pred = pred["start"] + start / sr

        # ==== 过滤 before ====
        mask = ~pred["before"]  # 只保留新事件

        for i in range(len(start_pred)):
            if not mask[i]:
                continue

            results.append({
                "start": start_pred[i].item(),
                "sustain": pred["sustain"][i].item(),
                "root": pred["root"][i].item(),
                "tonic": pred["tonic"][i].item(),
                "chord": pred["chord"][i].cpu(),
            })

    # ==== 合并重复事件 ====
    results = merge_events(results)

    return results

def merge_events(events, time_th=0.2, pitch_th=0):
    """
    events: List[dict]
    """

    if len(events) == 0:
        return events

    events = sorted(events, key=lambda x: x["start"])

    merged = [events[0]]

    for e in events[1:]:
        last = merged[-1]

        # 时间接近 + root一致
        if abs(e["start"] - last["start"]) < time_th and e["root"] == last["root"]:
            
            # 合并 sustain（取更大的）
            last["sustain"] = max(last["sustain"], e["sustain"])

            # chord 取 OR（更稳）
            last["chord"] = last["chord"] | e["chord"]

        else:
            merged.append(e)

    return merged

