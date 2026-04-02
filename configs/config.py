
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
    cfg.dataset_data_path = Path("../musicNotebook/preprocess11")
    
    cfg.batch_size = 1
    
    cfg.text_encoder_name = "BAAI/bge-small-zh-v1.5"
    cfg.audio_encoder_name = "facebook/wav2vec2-base-960h"
    
    cfg.text_input_dim = 512 # 参考你使用的文本编码器
    cfg.audio_input_dim = 512 # 参考你使用的音频编码器
    
    # 模型设置
    cfg.d_model = 1
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
    cfg.large_save_dir = "../params"
    
    cfg.use_diff_input = True
    cfg.output_mode = "TriggerBool_ConditionalSustain" # "Exclusion_MuteTriggerSustain"
    cfg.output_dim_dict = {
            "TriggerBool_ConditionalSustain": 2,
            "Exclusion_MuteSustain": 2
        }
    
    cfg.distinguish_pitch_freq = True
    
    cfg.pos_weight = 1000.0

    return cfg

