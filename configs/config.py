
import argparse
import numpy as np
from pathlib import Path

def midi2freq(midi):
    """
    midi: int or np.array
    return: frequency (Hz)
    """
    midi = np.asarray(midi)
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))

def freq2midi(freq):
    """
    freq: float or np.array
    return: midi (float, not rounded)
    """
    freq = np.asarray(freq)
    return 69 + 12 * np.log2(freq / 440.0)

def get_config():
    cfg = argparse.Namespace()

    cfg.sr = 44100
    
    # cfg.min_len = int(cfg.sr * 0.5)
    # cfg.max_len = int(cfg.sr * 1)
    cfg.wav_len = int(cfg.sr * 3)
    
    cfg.window_len = int(cfg.sr * 0.2)
    cfg.stride = int(cfg.window_len * 0.125)
    cfg.window_type = "hann"
    
    cfg.cqt_scale = 7
    
    # cfg.min_midi_freq = 50
    # cfg.max_midi_freq = 5000
    cfg.min_midi = 24 # freq2midi(cfg.min_midi_freq)
    cfg.max_midi = 107 # freq2midi(cfg.max_midi_freq)
    
    cfg.pitch_vocab_size = cfg.max_midi - cfg.min_midi + 1
    cfg.music_scale = "12tone"
    
    cfg.min_freq = 50
    cfg.max_freq = 16000
    cfg.freq_scale = "mel" # mel, linear, log
    cfg.num_freqs = 128
    
    cfg.dataset_read_py_path = Path("../musicNotebook/web")
    cfg.dataset_data_path = Path("../musicNotebook/preprocess2")
    
    cfg.batch_size = 1
    
    cfg.text_encoder_name = "BAAI/bge-small-zh-v1.5"
    cfg.audio_encoder_name = "facebook/wav2vec2-base-960h"
    
    cfg.text_input_dim = 512 # 参考你使用的文本编码器
    cfg.audio_input_dim = 512 # 参考你使用的音频编码器
    
    # 模型设置
    cfg.d_model = 64
    cfg.intermediate_size = 128
    
    cfg.num_decoder_layer = 6
    
    cfg.n_attn_heads = 8
    cfg.n_kv_heads = 4
    cfg.head_dim = 16
    
    cfg.attention_dropout = 0.1
    cfg.rms_norm_eps = 1e-6
    
    cfg.attn_type = "flash"
    
    # 笑容部分
    cfg.use_same_pitch_freq = True
    
    cfg.abs_pos_encoding = argparse.Namespace()
    cfg.use_abs_pos_encoding = True
    cfg.abs_pos_encoding.ref_freq = 50
    cfg.abs_pos_encoding.ref_time = cfg.wav_len
    cfg.abs_pos_encoding.sigma = 1
    
    cfg.save_dir = "./tiny_save"
    cfg.large_save_dir = "../params/detr"
    
    cfg.use_diff_input = True
    cfg.output_mode = "sustain_only" # "Exclusion_MuteTriggerSustain"
    cfg.output_dim_dict = {
            "TriggerBool_ConditionalSustain": 2,
            "Exclusion_MuteSustain": 2,
            "sustain_only": 1
        }
    
    cfg.use_diff_input = False
    
    cfg.distinguish_pitch_freq = True
    
    cfg.pos_weight = 10.0
    cfg.map_type = "sustain_map"

    cfg.time_mask_len = 5 # None
    
    # detr
    cfg.num_querys = 100
    cfg.detr_output_dim_dict = {"text":cfg.text_input_dim,
                                "event":2, # [start, log sustain]
                                "pitch":cfg.pitch_vocab_size + 1,
                                "exist": 1}
    cfg.detr_num_decoder_layers = 12
    cfg.detr_d_model_list = [64] * 3 + [128] * 3 + [256] * 3 + [512] * 2 + [1024]
    cfg.pool_stride = [None, None, 4, None, None, 3, None, None, 2, None, None, 5]
    cfg.head_dim_list = [16] * 3 + [32] * 3 + [64] * 3 + [128] * 2 + [256]
    assert len(cfg.detr_d_model_list) == cfg.detr_num_decoder_layers
    cfg.ffn_dim_up = [1,1,2, 1,1,2, 1,1,2, 1,2,1]
    cfg.ffn_intermediate_up_list = [2,2,4, 2,2,4, 2,2,2, 2,2,2]
    
    cfg.detr_cost_weight = {
        "pitch": 1.0,
        "start": 1.0,
        "logSustain": 1.0,
        "text": 1.0
        # "IoU": 2.0,
    }
    
    cfg.detr_loss_weight = {
        "pitch": 1.0,
        "start": 1.0,
        "logSustain": 1.0,
        "text": 1.0,
        "exist": 1.0
    }
    
    cfg.detr_pos_weight = 1
    
    return cfg

