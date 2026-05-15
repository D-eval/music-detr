
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

    cfg.dataset_read_py_path = Path("../musicNotebook/web")
    cfg.dataset_data_path = Path("../musicNotebook/preprocess0")

    cfg.dataset_read_py_path_stage1 = Path("../dataset/baby")
    cfg.dataset_read_py_path_stage1_ytb = Path("../dataset/ytbRand")
    
    # cfg.min_len = int(cfg.sr * 0.5)
    # cfg.max_len = int(cfg.sr * 1)
    cfg.wav_len = int(cfg.sr * 5)
    
    cfg.window_len = int(cfg.sr * 0.2)
    cfg.stride = int(cfg.window_len * 0.125)
    cfg.window_type = "hann"
    
    cfg.cqt_scale = 7
    
    # cfg.min_midi_freq = 50
    # cfg.max_midi_freq = 5000
    cfg.min_midi = 24 # freq2midi(cfg.min_midi_freq)
    cfg.max_midi = 131 # freq2midi(cfg.max_midi_freq)
    cfg.num_P = cfg.max_midi - cfg.min_midi + 1
    
    cfg.cqt_stride = 0.1
    
    cfg.harmony_conv = argparse.Namespace()
    cfg.harmony_conv.kernel_size = 60
    cfg.harmony_conv.cycles = [5,7,12, 5,7,12, 5,7,12]
    cfg.harmony_conv.trainable = False
    cfg.harmony_conv.taus = [7,7,7, 12,12,12, 19,19,19]
    cfg.harmony_conv.num_layers = 4
    
    cfg.harmony_conv.backbone_output_dim = 512
    
    cfg.harmony_conv.loss_weight = {
        "chord":1,
        "root":1,
        "exist":1,
    }
    cfg.harmony_conv.share_pitch_affine = True
    
    cfg.harmony_conv.infer_threshold = 0.3
    
    cfg.envelope_conv = argparse.Namespace()
    cfg.envelope_conv.receptive_field = 0.05
    cfg.envelope_conv.rep_dim = 512
    
    cfg.pitchDetr = argparse.Namespace()
    cfg.pitchDetr.query_num = 20 # max pitch
    cfg.pitchDetr.num_layers = 3
    cfg.pitchDetr.detr_d_model_list = [512] * 3
    cfg.pitchDetr.pool_stride = [None] * 3
    cfg.pitchDetr.head_dim_list = [128] * 3
    assert len(cfg.pitchDetr.detr_d_model_list) == cfg.pitchDetr.num_layers
    cfg.pitchDetr.ffn_dim_up = [1] * 3
    cfg.pitchDetr.ffn_intermediate_up_list = [2] * 3
    cfg.pitchDetr.n_attn_heads = 8
    cfg.pitchDetr.attention_dropout = 0.1
    cfg.pitchDetr.d_model = 512
    cfg.pitchDetr.intermediate_size = 128
    cfg.pitchDetr.n_kv_heads = 4
    cfg.pitchDetr.head_dim = 64
    cfg.pitchDetr.rms_norm_eps = 1e-6
    cfg.pitchDetr.attn_type = "flash"
    cfg.pitchDetr.time_mask_len = 13
    cfg.pitchDetr.cost_weights = {
        "exist":0.1,
        "pitch":1,
    }
    cfg.pitchDetr.loss_weights = {
        "exist":0.5,
        "pitch":1,
    }
    
    cfg.pitch_vocab_size = cfg.max_midi - cfg.min_midi + 1
    cfg.music_scale = "12tone"

    # augmentation switches
    cfg.aug_noise = True
    cfg.aug_distortion = True
    cfg.aug_reverb = True
    cfg.aug_pitch_shift = True

    cfg.aug_pitch_shift_range = (-5, 5)  # 半音
    cfg.aug_noise_level = 0.01
    cfg.aug_distortion_gain = 5.0
    cfg.aug_reverb_decay = 0.3

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
    cfg.large_save_dir = "../params/detr6"
    
        
    cfg.use_diff_input = True
    cfg.output_mode = "sustain_only" # "Exclusion_MuteTriggerSustain"

    cfg.time_mask_len = 5 # None
    
    cfg.num_prompt_querys = 9
    
    # assert cfg.num_querys % cfg.num_cls_querys == 0
    
    cfg.window_num = 8
    cfg.input_dim = cfg.window_num * 2 # 双声道
    cfg.min_cycle = 2
    
    cfg.detr_num_decoder_layers = 12
    cfg.detr_d_model_list = [512] * 3 + [1024] * 3 + [1024] * 3 + [2048] * 2 + [2048]
    cfg.pool_stride = [None, None, 2, None, None, 3, None, None, 2, None, None, 5]
    cfg.head_dim_list = [128] * 3 + [512] * 3 + [512] * 3 + [512] * 2 + [512]
    assert len(cfg.detr_d_model_list) == cfg.detr_num_decoder_layers
    cfg.ffn_dim_up = [1,1,2, 1,1,1, 1,1,2, 1,1,1]
    cfg.ffn_intermediate_up_list = [2,2,4, 2,2,2, 2,2,4, 2,2,2]
    
    
    cfg.detr_pos_weight_text = 1
    cfg.detr_pos_weight_event = 1
    
    cfg.detr2_loss_weight = {
        "chord": {
            "start": 1,
            "sustain": 0.5,
            "exist": 1,
            "root": 1,
            "chord": 1,
            "tonic": 1,
        },
        "beat": {
            "beat": 1,
            "is_downbeat": 1,
            "exist": 1,
        },
        "metronome": {
            "bpm": 1,
            "offset": 1,
            "is_4beat": 1,
            "exist": 0,
        },
        "chord_before": {
            "sustain": 0.3,
            "exist": 1,
            "root": 1,
            "chord": 1,
            "tonic": 1,
        }
    }
    
    cfg.detr2_cost_weight = {
        "chord": {
            "start": 1,
            "sustain": 0.2,
            "exist": 0.5,
            "root": 0.1,
            "chord": 0.1,
            "tonic": 0.1,
        },
        "beat": {
            "beat": 1,
            "is_downbeat": 0.1,
            "exist": 1,
        }
    }
    
    cfg.sustain_ref = 0.1
    
    cfg.text_cost_dist = "cosine" # cosine, euclidean
    cfg.text_loss_dist = "cosine" # cosine, euclidean
    
    cfg.infer_threshold = 0.5
    cfg.infer_chord_threshold = 0.5
    cfg.infer_before_threshold = 0.5
    
    cfg.llm = argparse.Namespace()
    
    cfg.llm.num_hidden_layers = 16
    
    cfg.llm.hidden_size = 128
    cfg.llm.intermediate_size = 256
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
    
    cfg.lr = 1e-4
    cfg.save_epoch = 1
    
    cfg.cell_structure = [
        ("chord", 10),
        ("chord_before", 1),
        ("beat", 20),
        ("metronome", 1),
    ]

    
    return cfg

