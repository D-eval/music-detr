
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
    cfg.large_save_dir = "../params/detr2"
    
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
    cfg.num_cell = 20
    cfg.cell = argparse.Namespace()
    cfg.cell.num_receptor_tokens = 5 # 细胞膜受体，和外部信息做注意力
    cfg.cell.num_distillation_tokens = 1 # 蒸馏token，用于接近 text_emb
    cfg.cell.num_prompt_tokens = 16 # 用于输入 llm，预测 文本描述
    cfg.cell.num_event_tokens = 20 # 用于预测事件
    cfg.cell.share_params = True # 共享细胞内的参数，只有细胞膜不同
    
    cfg.num_prompt_querys = 9
    
    # assert cfg.num_querys % cfg.num_cls_querys == 0
    
    
    cfg.detr_output_dim_dict = {"text":cfg.text_input_dim,
                                "event":2, # [start, log sustain]
                                "pitch":cfg.pitch_vocab_size + 1,
                                "exist": 1,
                                "prompt": 16}
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
    
    cfg.detr_pos_weight_text = 1
    cfg.detr_pos_weight_event = 1
    
    cfg.detr2_loss_weight = {
        "sub": 1,
        "exist_text": 1,
        "text": 1,
        "start": 1,
        "sustain": 1,
        "pitch": 1,
        "exist_event": 1
    }
    
    cfg.detr2_cost_weight = {
        "start": 1,
        "sustain": 1,
        "pitch": 1,
        "exist": 1,
    }
    
    cfg.sustain_ref = 0.1
    
    cfg.text_cost_dist = "cosine" # cosine, euclidean
    cfg.text_loss_dist = "cosine" # cosine, euclidean
    
    cfg.infer_text_exist_threshold = 0.5
    cfg.infer_event_exist_threshold = 0.5
    
    cfg.llm = argparse.Namespace()
    
    cfg.llm.num_hidden_layers = 2 # 16
    
    cfg.llm.hidden_size = 16 # 128
    cfg.llm.intermediate_size = 16 # 256
    cfg.llm.rms_norm_eps = 1e-6

    cfg.llm.head_dim = 8 # 64
    cfg.llm.num_attention_heads = 8
    cfg.llm.num_key_value_heads = 4
    cfg.llm.attention_dropout = 0.1
    cfg.llm.attn_type = "flash"

    cfg.llm.padding_idx = 0 # embedding 的 0
    cfg.llm.ignore_index = -100 # 忽略的 label idx
    
    cfg.llm.max_length = 20

    cfg.union_loss_weights = {
        "lm": 0.5,
        "detr": 0.5
    }
    
    cfg.tokenizer = argparse.Namespace()
    cfg.tokenizer.only_last_detail = False
    cfg.tokenizer.save_path = "./tiny_save/tokenizer.json"
    
    cfg.tokenizer.pad = "<pad>"
    cfg.tokenizer.role = "<role>"
    cfg.tokenizer.inst = "<inst>"
    cfg.tokenizer.desc = "<desc>"
    cfg.tokenizer.begin = "<bos>"
    cfg.tokenizer.end = "<eos>"
    cfg.tokenizer.unk = "<unk>"
    
    cfg.llm.max_length = 22
    cfg.llm.rope_base = 20
    
    return cfg

