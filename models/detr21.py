"""
带有group query和训练策略
event 39
root + chord + tonic
数据: read0.py
"""

import torch
from configs.config import get_config21 as get_config
from torch import nn
import math
import torch.nn.functional as F
from typing import Callable, Optional, Union, Dict
from scipy.optimize import linear_sum_assignment


def hungarian_match(cost_matrix):
    """
    cost_matrix: (N, Q)
    return:
        row_ind: (M,)  GT index
        col_ind: (M,)  query index
    """
    cost = cost_matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind


def cal_pitch_cost(gt, pred):
    """
    gt: (N,) long, 取值 0~P 或 -1（pitchless 音高）
    pred: (Q, P+1) logits

    return: (N, Q)
    """
    Pa1 = pred.shape[1]
    neg_idx = (gt < 0)
    gt[neg_idx] = Pa1 - 1
    log_prob = F.log_softmax(pred, dim=-1)  # (Q, P+1)
    # gather
    cost = -log_prob[:, gt].T  # (N, Q)
    return cost

def cal_start_cost(gt, pred):
    """
    gt: (N,)
    pred: (Q,)
    return: (N, Q)
    """
    return torch.abs(gt[:, None] - pred[None, :])

def cal_logSustain_cost(gt, pred):
    """
    gt: (N,)
    pred: (Q,)
    return: (N, Q)
    """
    return torch.abs(gt[:, None] - pred[None, :])

def cal_text_cost(gt, pred):
    """
    gt: (N, C)
    pred: (Q, C)

    return: (N, Q)
    """
    gt_norm = F.normalize(gt, dim=-1)
    pred_norm = F.normalize(pred, dim=-1)

    sim = torch.matmul(gt_norm, pred_norm.T)  # (N, Q)

    cost = 1 - sim
    return cost

class Qwen2MLP(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        cfg = get_config()
        
        d_input = cfg.detr_d_model_list[layer_idx]
        d_up = cfg.ffn_dim_up[layer_idx]
        d_output = d_input * d_up
        intermediate_size = cfg.ffn_intermediate_up_list[layer_idx]
        
        self.d_input = d_input
        self.d_up = d_up
        self.d_output = d_output
        self.intermediate_size = intermediate_size
        
        self.gate_proj = nn.Linear(self.d_input, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.d_input, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.d_output, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, x):
        # 用gate, 可以表达条件计算
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    (batch, num_key_value_heads, seqlen, head_dim) -> (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0
):
    # q: (B, T, C * H) k,v: (B, T, C * H_kv)
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output

def low_mem_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # fp16 计算
    query = query.to(torch.float16)
    key_states = key_states.to(torch.float16)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # inplace softmax（关键）
    attn_weights = torch.softmax(attn_weights, dim=-1)

    # 关闭 dropout（节省显存）
    # attn_weights = F.dropout(...)

    attn_output = torch.matmul(attn_weights, value_states.to(torch.float16))

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


def flash_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask=None,
    scaling: float = 1.0,
    dropout: float = 0.0
    ):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    # PyTorch 会自动用 Flash / MemEff kernel
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=dropout if module.training else 0.0,
        is_causal=False
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


def chunk_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    chunk_size=512
    ):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    B, H, T, D = query.shape
    outputs = []

    for i in range(0, T, chunk_size):
        q_chunk = query[:, :, i:i+chunk_size, :]  # (B,H,chunk,D)

        attn_weights = torch.matmul(q_chunk, key_states.transpose(2,3)) * scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, i:i+chunk_size, :]

        attn_weights = torch.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_weights, value_states)
        outputs.append(out)

    attn_output = torch.cat(outputs, dim=2)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output


AttentionType = {
    "flash": flash_attention_forward,
    "eager": eager_attention_forward,
    "fp16": low_mem_attention_forward,
    "chunk": chunk_attention_forward
}

class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        cfg = get_config()
        self.attn_type = cfg.attn_type
        
        d_model = cfg.detr_d_model_list[layer_idx]
        head_dim = cfg.head_dim_list[layer_idx]
        self.d_model = d_model
        self.head_dim = head_dim
        
        self.num_key_value_groups = cfg.n_attn_heads // cfg.n_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = cfg.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(self.d_model, cfg.n_attn_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.d_model, cfg.n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.d_model, cfg.n_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(cfg.n_attn_heads * self.head_dim, self.d_model, bias=False)
        print("初始化注意力模块")
    def forward(
        self,
        hidden_states: torch.Tensor,
        # position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        # past_key_values: Optional[Cache] = None,
        # cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_state: (N, All, C)
        # return: (N, All, C)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # [text,freq,pitch]
        attention_interface = AttentionType[self.attn_type]
        attn_output = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            # sliding_window=self.sliding_window,  # main diff with Llama
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


def rope_1d(x, pos):
    """
    x: (..., C)
    pos: (...,)
    """
    dim = x.shape[-1]
    half = dim // 2

    freqs = torch.arange(half, device=x.device)
    freqs = 1.0 / (10000 ** (freqs / half))

    angles = pos[..., None] * freqs  # (..., half)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    x1, x2 = x[..., :half], x[..., half:]

    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)
    
def generate_crossAttn_mask(Tc, T):
    pass

def rope_2d():
    pass

def split_head(x, H):
    """
        x: (..., C)
        return: (..., H, C//H)
    """
    *shape, C = x.shape
    assert C % H == 0, f"{C}不能被{H}整除"
    
    d = C//H
    return x.reshape(*shape, H, d)



class Qwen2RMSNorm(nn.Module):
    def __init__(self, d_model, eps: float = 1e-6) -> None:
        super().__init__()
        hidden_size = d_model
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 平方根倒数 rsqrt
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        cfg = get_config()
        
        self.layer_idx = layer_idx
        self.self_attn = Qwen2Attention(layer_idx=layer_idx)
        self.mlp = Qwen2MLP(layer_idx=layer_idx)
        self.input_layernorm = Qwen2RMSNorm(cfg.detr_d_model_list[layer_idx])
        self.post_attention_layernorm = Qwen2RMSNorm(cfg.detr_d_model_list[layer_idx])
        
        self.ffn_dim_up = cfg.ffn_dim_up[layer_idx]
    def forward(
        self,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_state: (N_cells, L_cell, C)
        
        _hidden_state = self.input_layernorm(hidden_state)
        _hidden_state = self.self_attn(
            hidden_states= _hidden_state,
            attention_mask=None,
        )
        
        # == 残差连接
        hidden_state = hidden_state + _hidden_state
        
        ffn_dim_up = self.ffn_dim_up
        
        # 升维
        _hidden_state = self.post_attention_layernorm(hidden_state)
        _hidden_state = self.mlp(_hidden_state)
        
        hidden_state = hidden_state.unsqueeze(-1).expand(-1,-1,-1,ffn_dim_up)
        hidden_state = torch.flatten(hidden_state, -2, -1)
        hidden_state = hidden_state + _hidden_state
        
        return hidden_state


class TFDecoderLayer(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        cfg = get_config()
        
        self.layer_idx = layer_idx
        self.self_attn_F = Qwen2Attention(layer_idx=layer_idx)
        self.self_attn_T = Qwen2Attention(layer_idx=layer_idx)
        self.mlp = Qwen2MLP(layer_idx=layer_idx)
        self.input_layernorm_F = Qwen2RMSNorm(cfg.detr_d_model_list[layer_idx])
        self.input_layernorm_T = Qwen2RMSNorm(cfg.detr_d_model_list[layer_idx])
        self.post_attention_layernorm = Qwen2RMSNorm(cfg.detr_d_model_list[layer_idx])
        
        self.time_mask_len = cfg.time_mask_len
        
        self.ffn_dim_up = cfg.ffn_dim_up[layer_idx]
        self.pool_stride = cfg.pool_stride[layer_idx]
    def forward(
        self,
        modal_dict: Dict[str, torch.Tensor],
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[Cache] = None,
        # use_cache: Optional[bool] = False,
        # cache_position: Optional[torch.LongTensor] = None,
        # position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        # **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        freq = modal_dict['freq'] # (B, T, F, C)
        pitch = modal_dict['pitch'] # (B, T, P, C)
        text = modal_dict['text'] # (B, Q, C)
        
        assert text.shape[-1]==freq.shape[-1]
        
        assert pitch.shape[1] == freq.shape[1], "无法对齐时间"
        T = freq.shape[1]
        B = text.shape[0]
        F = freq.shape[2]
        P = pitch.shape[2]
        Q = text.shape[1]
        C = freq.shape[-1]
        
        QT = Q * T
        FaP = F + P
        FaPaQ = FaP + Q
        
        BF = B * F
        BP = B * P
        TaQ = T + Q
        
        # Freq Attn
        _text = text.unsqueeze(1).expand(-1, T, -1, -1) # (N, T, L, C)
        _freq = torch.flatten(freq, 0, 1) # (NT, F, C)
        _pitch = torch.flatten(pitch, 0, 1) # (NT, P, C)
        _text = torch.flatten(_text, 0, 1) # (NT, L, C)
        
        hidden_states_F = torch.concat([_pitch, _freq, _text], dim=1) # (NT, FaPaL, C)
        
        hidden_states_F = self.input_layernorm_F(hidden_states_F)
        # Self Attention
        

        hidden_states_F = self.self_attn_F(
            hidden_states=hidden_states_F,
            attention_mask=None,
            # position_ids=position_ids,
            # past_key_values=past_key_values,
            # use_cache=use_cache,
            # cache_position=cache_position,
            # position_embeddings=position_embeddings,
            # **kwargs,
        )
        
        _freq = freq.permute(0,2,1,3) # (N, F, T, C)
        _freq = torch.flatten(_freq, 0, 1) # (NF, T, C)
        _text = text.unsqueeze(1).expand(-1, F, -1, -1) # (N, F, L, C)
        _text = torch.flatten(_text, 0, 1) # (NF, L, C)
        _freq = torch.concat([_freq, _text], dim=1) # (NF, TaL, C)
        
        _pitch = pitch.permute(0,2,1,3) # (N, P, T, C)
        _pitch = torch.flatten(_pitch, 0, 1) # (NP, T, C)
        _text = text.unsqueeze(1).expand(-1, P, -1, -1) # (N, P, L, C)
        _text = torch.flatten(_text, 0, 1) # (NP, L, C)
        _pitch = torch.concat([_pitch, _text], dim=1) # (NP, TaL, C)
        
        hidden_states_T = torch.concat([_freq, _pitch], dim=0) # (NF+NP, TaL, C)
        
        hidden_states_T = self.input_layernorm_T(hidden_states_T)
        

        if self.time_mask_len is None:
            attention_mask = None
        else:
            attention_mask = torch.ones((TaQ, TaQ), device=freq.device) * float("-inf")
            # 1. 每个 T 都能看见 L 和前后 self.time_mask_len 的 T
            # 2. 每个 L 都能看见所有
            text_idx = torch.arange(T, T+Q)
            time_idx = torch.arange(T)
            # 每个 time token 能看到自身 ± time_mask_len + 所有文本
            for t in time_idx:
                start = max(0, t - self.time_mask_len)
                end = min(T, t + self.time_mask_len + 1)
                attention_mask[t, start:end] = 0  # 允许看到的 T
                attention_mask[t, text_idx] = 0   # 允许看到所有文本
            # 文本 token 全可见
            attention_mask[text_idx[:, None], :] = 0

        hidden_states_T = self.self_attn_T(
            hidden_states=hidden_states_T,
            attention_mask=attention_mask,
        )
        
        # hidden_states_F (NT, FaPaL, C)
        # hidden_states_T (NF+NP, TaL, C)
        
        # == 解析 F 输出
        pitch_F = hidden_states_F[:, :P, :] # (NT, P, C)
        freq_F = hidden_states_F[:, P:FaP, :] # (NT, F, C)
        text_F = hidden_states_F[:, FaP:FaPaQ, :] # (NT, L, C)
        
        text_F = text_F.reshape(B, T, Q, C) # 要聚合 T 维度为 (B, Q, C)
        freq_F = freq_F.reshape(B, T, F, C)
        pitch_F = pitch_F.reshape(B, T, P, C)
        
        # == 解析 T 输出
        text_T_and_freq_T = hidden_states_T[:BF, :, :] # (NF, TaL, C)
        text_T_and_pitch_T = hidden_states_T[BF:BF+BP, :, :] # (NP, TaL, C)
        
        freq_T = text_T_and_freq_T[:, :T, :] # (NF, T, C)
        text_T_from_freq = text_T_and_freq_T[:, T:TaQ, :] # (NF, L, C)
        
        pitch_T = text_T_and_pitch_T[:, :T, :] # (NP, T, C)
        text_T_from_pitch = text_T_and_pitch_T[:, T:TaQ, :] # (NP, L, C)
        
        text_T_from_freq = text_T_from_freq.reshape(B, F, Q, C) # 聚合 F 维度
        text_T_from_pitch = text_T_from_pitch.reshape(B, P, Q, C) # 聚合 P 维度
        freq_T = freq_T.reshape(B, F, T, C)
        pitch_T = pitch_T.reshape(B, P, T, C)
        
        freq_T = freq_T.permute(0,2,1,3)
        pitch_T = pitch_T.permute(0,2,1,3)
        
        # query_F = text_F
        # query_T_from_freq = text_T_from_freq
        # query_T_from_pitch = text_T_from_pitch
        
        # query_F # (B, T, Q, C) # 要聚合 T 维度为 (B, Q, C)
        # query_T_from_freq # (B, F, Q, C) # 聚合 F 维度 (B, Q, C)
        # query_T_from_pitch # (B, P, Q, C) # 聚合 P 维度 (B, Q, C)
        
        # == 残差连接
        text = text + text_F.mean(1) + text_T_from_freq.mean(1) + text_T_from_pitch.mean(1) # (N, L, C)
        freq = freq + freq_F + freq_T # (N, T, F, C)
        pitch = pitch + pitch_F + pitch_T # (N, T, P, C)
        
        # 先池化，再升维
        if self.pool_stride is not None:
            freq = temporal_pool(freq, stride=self.pool_stride)
            pitch = temporal_pool(pitch, stride=self.pool_stride)
            
        ffn_dim_up = self.ffn_dim_up
        
        # 升维残差连接
        # _text = self.post_attention_layernorm(text)
        # _text = self.mlp(_text)
        # text_repeated = text.unsqueeze(-1).expand(-1,-1,-1,ffn_dim_up)
        # text_repeated = torch.flatten(text_repeated, -2, -1)
        # text = text_repeated + _text
        text = text # text 不走 mlp升维
        
        _freq = self.post_attention_layernorm(freq)
        _freq = self.mlp(_freq)
        freq_repeated = freq.unsqueeze(-1).expand(-1,-1,-1,-1,ffn_dim_up)
        freq_repeated = torch.flatten(freq_repeated, -2, -1)
        freq = freq_repeated + _freq
        
        _pitch = self.post_attention_layernorm(pitch)
        _pitch = self.mlp(_pitch)
        pitch_repeated = pitch.unsqueeze(-1).expand(-1,-1,-1,-1,ffn_dim_up)
        pitch_repeated = torch.flatten(pitch_repeated, -2, -1)
        pitch = pitch_repeated + _pitch
        
        modal_dict = {
            "text": text,
            "pitch": pitch,
            "freq": freq
        }
        
        return modal_dict



def apply_freq_time_encoding(freqs, times, d_model):
    # freqs: (F)
    # times: (T)
    # return: (T, F, C)
    T = times.shape[0]
    F = freqs.shape[0]
    assert d_model % 2 == 0, "wtf"
    
    cfg = get_config()
    
    # freqs_rel 表示在 C 上震荡了几下
    freqs_rel = freqs / cfg.abs_pos_encoding.ref_freq
    
    half = d_model // 2
    half_arange = torch.arange(half, device=freqs.device) / half
    
    # times 作为相位的偏移
    # 2 * pi * f * x + t
    # 这样，t相同时，如果波形和谐，内积就小
    # 相邻的 t 会互相看到
    times_rel = times / cfg.abs_pos_encoding.ref_time
    
    phase = 2 * math.pi * (freqs_rel[None,:,None] * half_arange[None,None,:] + times_rel[:,None,None])
    
    amps = torch.exp(-(half_arange[None,None,:] - times_rel[:,None,None])**2 / cfg.abs_pos_encoding.sigma**2) \
        * 1/cfg.abs_pos_encoding.sigma / math.sqrt(2 * math.pi)
    cos, sin = torch.cos(phase), torch.sin(phase)
    
    cos = cos * amps
    sin = sin * amps
    
    pos_encoding = torch.zeros((T, F, d_model), device=freqs.device)
    pos_encoding[:,:,::2] = cos
    pos_encoding[:,:,1::2] = sin
    
    return pos_encoding


# def temporal_pool(x, stride=4):
#     # x: (B, T, N, C)
#     B, T, N, C = x.shape
#     x = x.view(B, T // stride, stride, N, C)
#     x = x.mean(dim=2)  # 或 max
#     return x  # (B, T//stride, N, C)


def temporal_pool(x, stride=4):
    # x: (B, T, N, C)
    B, T, N, C = x.shape
    # reshape → (B*N, C, T)
    x = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
    # 自动处理 padding（ceil_mode=True）
    x = F.avg_pool1d(
        x,
        kernel_size=stride,
        stride=stride,
        ceil_mode=True  # 🔥关键
    )
    T_new = x.shape[-1]
    # reshape 回去 → (B, T_new, N, C)
    x = x.view(B, N, C, T_new).permute(0, 3, 1, 2)
    return x


class PitchTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = get_config()
        
        self.d_model_list = cfg.detr_d_model_list
        self.dim_up = cfg.ffn_dim_up
        
        self.pitch_num = cfg.pitch_vocab_size
        
        input_dim = cfg.input_dim

        self.pitch_embed = nn.Linear(input_dim, self.d_model_list[0])
        self.pitchless_embedding = nn.Parameter(torch.randn((self.d_model_list[0])))
        if cfg.use_same_pitch_freq:
            self.freq_embed = self.pitch_embed
        else:
            self.freq_embed = nn.Linear(input_dim, self.d_model_list[0])
        
        self.distinguish_pitch_freq = cfg.distinguish_pitch_freq
        if self.distinguish_pitch_freq:
            self.pitch_token_embedding = nn.Parameter(torch.randn((self.d_model_list[0])))
            self.freq_token_embedding = nn.Parameter(torch.randn((self.d_model_list[0])))
        
        self.use_abs_pos_encoding = cfg.use_abs_pos_encoding
        if self.use_abs_pos_encoding:
            self.freq_time_encoding = apply_freq_time_encoding
        
        # 细胞数
        self.num_cell = cfg.num_cell
        
        self.num_receptor_tokens = cfg.cell.num_receptor_tokens # 每个cell要有多个受体，接受多种信息因子
        self.num_distillation_tokens = cfg.cell.num_distillation_tokens
        assert self.num_distillation_tokens == 1, "wtf"
        self.num_prompt_tokens = cfg.cell.num_prompt_tokens
        self.num_event_tokens = cfg.cell.num_event_tokens
        
        self.receptor_tokens = nn.Parameter(torch.randn((self.num_cell, self.num_receptor_tokens, self.d_model_list[0])))
        if cfg.cell.share_params:
            self.distillation_tokens = nn.Parameter(torch.randn((1, self.d_model_list[0])))
            self.prompt_tokens = nn.Parameter(torch.randn((self.num_prompt_tokens, self.d_model_list[0])))
            self.event_tokens = nn.Parameter(torch.randn((self.num_event_tokens, self.d_model_list[0])))
        else:
            raise NotImplementedError("wtf")
                
        self.audio_embed = nn.Linear(cfg.audio_input_dim, self.d_model_list[0])
        self.text_input_dim = cfg.text_input_dim
        
        self.num_layers = cfg.detr_num_decoder_layers
        self.inter_decoder_layers = nn.ModuleList([
            TFDecoderLayer(i)
            for i in range(self.num_layers)
        ])
        self.inner_decoder_layers = nn.ModuleList([
            Qwen2DecoderLayer(i)
            for i in range(self.num_layers)
        ])
        
        self.output_mode = cfg.output_mode
        
        output_dim_dict = cfg.detr_output_dim_dict
        self.output_dim_dict = output_dim_dict
        
        output_dim_event = output_dim_dict["event"]
        output_dim_pitch = output_dim_dict["pitch"]
        output_dim_exist = output_dim_dict["exist"]

        self.output_dim_event = output_dim_event
        self.output_dim_pitch = output_dim_pitch
        self.output_dim_exist = output_dim_exist
        
        output_Qe_dim =  output_dim_event + output_dim_pitch + output_dim_exist
        # 要进行两次 hungarian matching
        # 第一次对 text 进行匹配
        # 第二次在 text 内部对 event 和 pitch 进行匹配
        
        self.events_head = nn.Linear(self.d_model_list[-1], output_Qe_dim)
        
        self.cost_weight = cfg.detr2_cost_weight
        self.loss_weight = cfg.detr2_loss_weight
        self.sustain_ref = cfg.sustain_ref
        
        self.pos_weight_exist_text = cfg.detr_pos_weight_text
        self.pos_weight_exist_event = cfg.detr_pos_weight_event
        
        self.text_cost_dist = cfg.text_cost_dist
        self.text_loss_dist = cfg.text_loss_dist
        
        self.infer_threshold = cfg.infer_threshold
        self.infer_chord_threshold = cfg.infer_chord_threshold
        self.infer_before_threshold = cfg.infer_before_threshold
        
    def forward(self,
                pitch_spec,
                pitchs,
                pitch_centre,
                freq_spec,
                freqs,
                freq_centre):
        """
        inputs
            pitch_spec: (B, T, P, C)
            freq_spec: (B, T, F, C)
        return: 
            output: (B, Q, C)
        """
        B, T, P, C = pitch_spec.shape
        _B, _T, F, _C = freq_spec.shape
        assert B == _B and T == _T, "无法对齐时间"
        assert C == _C == 2
            
        pitch_embedding = self.pitch_embed(pitch_spec)
        freq_embedding = self.freq_embed(freq_spec)

        if self.use_abs_pos_encoding:
            pitch_pos_encoding = self.freq_time_encoding(pitchs, pitch_centre, self.d_model_list[0])
            pitch_embedding = pitch_embedding + pitch_pos_encoding[None,...] # (B, T, F, C)

            freq_pos_encoding = self.freq_time_encoding(freqs, freq_centre, self.d_model_list[0])
            freq_embedding = freq_embedding + freq_pos_encoding[None,...]
        
        
        pitchless = self.pitchless_embedding[None, None, None, :].expand(B, T, 1, self.d_model_list[0])
        pitch_embedding = torch.concat([pitch_embedding, 
                                        pitchless],
                                       dim=2) # (B, T, P, C)
        
        if self.distinguish_pitch_freq:
            pitch_embedding = pitch_embedding + self.pitch_token_embedding[None,None,None,:]
            freq_embedding = freq_embedding + self.freq_token_embedding[None,None,None,:]
        
        # 组装 query_emb (B, N_cell * N_receptor, C)
        query_emb = self.receptor_tokens.flatten(0,1)[None,:,:].expand(B,-1,-1)
        
        N_cell = self.num_cell
        assert N_cell==1
        N_rec = self.num_receptor_tokens
        N_teacher = self.num_distillation_tokens
        N_prompt = self.num_prompt_tokens
        N_event = self.num_event_tokens
        
        # 组装 cell (B, N_cell, L_cell, C)
        receptors = self.receptor_tokens[None,:,:,:].expand(B, -1, -1, -1)
        teachers = self.distillation_tokens[None,None,:,:].expand(B, N_cell, -1, -1)
        prompts = self.prompt_tokens[None,None,:,:].expand(B, N_cell, -1, -1)
        events = self.event_tokens[None,None,:,:].expand(B, N_cell, -1, -1)
        
        # (B, N_cell, L_cell, C)
        cells = torch.concat([receptors, teachers, prompts, events], dim=2)
        
        modal_dict = {
            "text": query_emb, # (B, Q, C)
            "pitch": pitch_embedding, # (B, T, P+1, C)
            "freq": freq_embedding, # (B, T, F, C)
        }
        
        for i in range(self.num_layers):
            modal_dict = self.inter_decoder_layers[i](modal_dict) # (B, ..., C2)
            # 此时 receptors 已经融合了信息
            new_receptors = modal_dict['text'] # (B, Q, C1)
            # 注意，我们要避免 inter_decoder_layers 中 new_receptors 升维
            new_receptors = new_receptors.reshape(B, N_cell, -1, self.d_model_list[i])
            cells[:,:,:N_rec,:] = new_receptors # 因为 inter_decoder_layers 的 receptors 有残差连接
            cells = cells.flatten(0,1) # (B*N_cell, L_cell, C)
            cells = self.inner_decoder_layers[i](cells)
            cells = cells.reshape(B, N_cell, -1, self.d_model_list[i]*self.dim_up[i])
            # 更新 modal_dict
            new_query_emb = cells[:,:,:N_rec,:] # (B, N_cell, N_rec, C)
            new_query_emb = new_query_emb.flatten(1,2) # (B, N_cell*N_rec, C)
            modal_dict['text'] = new_query_emb # (B, Q, C2)

        output_distillation = cells[:,:,N_rec:N_rec+N_teacher,:] # (B, N, 1, C)
        output_prompts = cells[:,:,N_rec+N_teacher:N_rec+N_teacher+N_prompt,:] # (B, N, prompt, C)
        output_events = cells[:,:,N_rec+N_teacher+N_prompt:N_rec+N_teacher+N_prompt+N_event,:] # (B, N, E, C)

        # output_distillation = self.distllation_head(output_distillation) # (B, N, 1, C_text)
        # output_prompts = self.prompt_head(output_prompts) # (B, N, prompt, C_prompts)
        output_events = self.events_head(output_events) # (B, N, E, C_events)

        # assert output_distillation.shape[2]==1
        # output_distillation = output_distillation.squeeze(2) # (B, N, C)

        outputs = [{
            "text_distillation": None, # output_distillation[b,...], # (Qt, C)
            "text_prompt": None, # output_prompts[b,...], # (Qt, prompt, C)
            "event_out": output_events[b,...], # (Qt, Qe, C)
        } for b in range(B)] # chord 只关注 event_out[0] 就好了

        return outputs
        
    def get_sample_loss(self, output, target):
        """
            output: Dict
            target: Dict Ne
        """
        output = output['event_out'][0, ...] # (Qe, Ce)
        gt_idxs, pred_idxs, loss_matrix = self.match_event(output, target)
        M = gt_idxs.shape[0]
        loss = 0
        loss_dict = {}
        for k, v in loss_matrix.items():
            if k=="exist":
                continue
            loss += self.loss_weight[k] * v[gt_idxs, pred_idxs].mean()
            loss_dict[k] = v[gt_idxs, pred_idxs].mean().item()
            
        # 2) exist loss: 所有 query 都参与
        exist_pred = output[:, -1]  # (Qe,)
        exist_target = torch.zeros_like(exist_pred)  # (Qe,)
        exist_target[pred_idxs] = 1.0

        loss_exist = F.binary_cross_entropy_with_logits(
            exist_pred,
            exist_target,
            reduction="mean"
        )
        loss_dict['exist'] = loss_exist.item()

        loss = loss + self.loss_weight["exist"] * loss_exist
        return loss, loss_dict

    def infer(self, output):
        """
        output: Dict
        {
            "text_distillation": output_distillation[b,...], # (Qt, C)
            "text_prompt": output_prompts[b,...], # (Qt, prompt, C)
            "event_out": output_events[b,...], # (Qt, Qe, C)
        }
        """
        output = output['event_out'][0, ...] # (Qe, Ce)

        threshold = self.infer_threshold
        chord_threshold = self.infer_chord_threshold
        before_threshold = self.infer_before_threshold

        start_pred = output[:,0] # (Q)
        sustain_pred = output[:,1] # (Q,)
        before_pred = output[:,2] # (Q,)
        exist_pred = output[:,-1] # (Q)
        
        pitch_pred = output[:,3:3+self.output_dim_pitch]
        root_logits = pitch_pred[:,:12] # (Q,12)
        chord_logits = pitch_pred[:, 12:24] # (Q,12)
        tonic_logits = pitch_pred[:, 24:36] # (Q,12)

        choice = torch.sigmoid(exist_pred) > threshold
        before_pred = torch.sigmoid(before_pred) > before_threshold
        
        start_pred = start_pred[choice]
        sustain_pred = torch.exp(sustain_pred[choice]) * self.sustain_ref
        exist_pred = exist_pred[choice]
        root_pred = torch.argmax(root_logits[choice], dim=1)
        chord_pred = torch.sigmoid(chord_logits[choice]) > chord_threshold
        tonic_pred = torch.argmax(tonic_logits[choice], dim=1)
        
        result = {
            "root": root_pred, # (M) 0~11
            "chord": chord_pred, # (M, 12)
            "tonic": tonic_pred, # (M) 0~11
            "start": start_pred, # (M)
            "sustain": sustain_pred, # (M)
            "exist": exist_pred, # (M)
            "before": before_pred, # (M) 0~1
        }

        return result
        

    def get_loss(self, outputs, targets):
        loss = 0
        loss_dict = {}
        B = len(outputs)
        for b in range(B):
            temp_loss, temp_loss_dict = self.get_sample_loss(outputs[b], targets[b])
            loss += temp_loss / B
            for k in temp_loss_dict.keys():
                if loss_dict.get(k):
                    loss_dict[k] += temp_loss_dict[k] /B
                else:
                    loss_dict[k] = temp_loss_dict[k] /B
        return loss, loss_dict
 
    def get_cost_or_loss_matrix(self, output, target):
        """
            output: (Qe, Ce)
            target: Dict
        """
        Ne = target['chord'].shape[0]
        
        start_gt = target['start'] # (N,)
        sustain_gt = target['sustain'] # (N,)
        before_gt = target['before'] # (N,)
        root_gt = target['root'] # (N,) int
        tonic_gt = target['tonic'] # (N,) int
        chord_gt = target['chord'] # (N, 12) 0~1


        Qe = output.shape[0]
        assert output.shape[1]==3+36+1
        
        start_pred = output[:,0] # (Q)
        sustain_pred = output[:,1] # (Q,)
        before_pred = output[:,2] # (Q,)
        
        exist_pred = output[:,-1] # (Q)

        pitch_pred = output[:,3:3+self.output_dim_pitch]
        root_logits = pitch_pred[:,:12] # (Q,12)
        chord_logits = pitch_pred[:, 12:24] # (Q,12)
        tonic_logits = pitch_pred[:, 24:36] # (Q,12)

        
        cost_before = F.binary_cross_entropy_with_logits(before_pred[None,:].expand(Ne, Qe),
                                                         before_gt[:,None].expand(Ne, Qe),
                                                         reduction="none")
        
        prob_before = F.sigmoid(before_pred) # (Q,)
        mask = (1 - before_gt[:, None]) # * (1 - prob_before[None, :])
        
        diff_start = start_gt[:,None] - start_pred[None,:] # (N, Q)
        cost_start = diff_start**2 * mask # (N, Q)
        
        diff_sustain = (torch.log(sustain_gt[:, None] + 1e-6)-math.log(self.sustain_ref)) - sustain_pred[None,:] # (N, Q)
        cost_sustain = diff_sustain**2 * mask # (N, Q)
        
        log_prob_root = F.log_softmax(root_logits, dim=-1) # (Q, 12)
        cost_root = -log_prob_root[:, root_gt].T # (N, Q)
        
        log_prob_tonic = F.log_softmax(tonic_logits, dim=-1) # (Q, 12)
        cost_tonic = -log_prob_tonic[:, tonic_gt].T # (N, Q)
        
        # 扩展维度对齐
        chord_logits_exp = chord_logits[None, :, :]  # (1, Q, 12)
        chord_gt = chord_gt.float()
        chord_gt_exp = chord_gt[:, None, :]          # (N, 1, 12)

        # BCE cost
        cost_chord = F.binary_cross_entropy_with_logits(
            chord_logits_exp.expand(Ne, Qe, 12),
            chord_gt_exp.expand(Ne, Qe, 12),
            reduction='none'
        ).mean(dim=-1)  # (N, Q)
        
        # exist_prob = torch.sigmoid(exist_pred) # (Q,)
        # exist_logprob = - torch.log(exist_prob + 1e-6)[None, :].expand(Ne, Qe) # (N, Q)

        cost_exist = F.binary_cross_entropy_with_logits(
            exist_pred[None, :].expand(Ne, Qe),
            torch.ones((Ne, Qe), device=output.device),
            reduction="none"
        )
        
        return {
            "start": cost_start,
            "sustain": cost_sustain,
            "root": cost_root,
            "tonic": cost_tonic,
            "chord": cost_chord,
            "exist": cost_exist,
            "before": cost_before
        }

    def match_event(self, output, target):
        """
            output: (Qe, Ce)
            target: Dict
        """
        cost_matrix = self.get_cost_or_loss_matrix(output, target)

        cost = 0
        for k, v in cost_matrix.items():
            weight = self.cost_weight[k]
            cost = cost + weight * v

        gt_idxs, pred_idxs = hungarian_match(cost)
        return gt_idxs, pred_idxs, cost_matrix


# def cal_cost_matrix(labors, tasks):
#     # labors: 

# def cal_cost_of_assign_a_to_b(a, b):

# class PitchSpecEmbedding(nn.Module):
#     def __init__(self):
#         super().__init__()
#         cfg = get_config()
#         pitch_num = cfg.pitch_vocab_size
#         assert cfg.music_scale == "12tone", "须采用12平均律"
#         distance_metric = [
#             0, 7, 2
#         ]
        

# class ConditionalConv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, context_dim, group):
#         super().__init__()

#     def forward(self, x, context):
#         # x: (B, C, T)
#         # context: (B, D)



# class DETRPitch(nn.Module):
#     def __init__(self):
#         super().__init__()
#         cfg = get_config()
        

        
        