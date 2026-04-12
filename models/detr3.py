"""
text 维度的 hungarian matching 的 cost 计算 需要应用 event 信息的总和
"""

import torch
from configs.config import get_config
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
        
        self.use_diff_input = cfg.use_diff_input
        if self.use_diff_input:
            input_dim = 2
        else:
            input_dim = 1

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
        
        output_dim_text = output_dim_dict["text"]
        output_dim_event = output_dim_dict["event"]
        output_dim_pitch = output_dim_dict["pitch"]
        output_dim_exist = output_dim_dict["exist"]
        output_dim_prompt = output_dim_dict['prompt']
        
        self.output_dim_event = output_dim_event
        self.output_dim_pitch = output_dim_pitch
        self.output_dim_exist = output_dim_exist
        
        output_Qe_dim =  output_dim_event + output_dim_pitch + output_dim_exist
        output_Qt_dim = output_dim_text + output_dim_exist # 每个 text 也需要 confidence
        output_Qp_dim = output_dim_prompt # 用于生成句子
        # 要进行两次 hungarian matching
        # 第一次对 text 进行匹配
        # 第二次在 text 内部对 event 和 pitch 进行匹配
        
        self.distllation_head = nn.Linear(self.d_model_list[-1], output_Qt_dim)
        self.events_head = nn.Linear(self.d_model_list[-1], output_Qe_dim)
        self.prompt_head = nn.Linear(self.d_model_list[-1], output_Qp_dim)
        
        self.cost_weight = cfg.detr3_cost_weight
        self.loss_weight = cfg.detr3_loss_weight
        self.sustain_ref = cfg.sustain_ref
        
        self.pos_weight_exist_text = cfg.detr_pos_weight_text
        self.pos_weight_exist_event = cfg.detr_pos_weight_event
        
        self.text_cost_dist = cfg.text_cost_dist
        self.text_loss_dist = cfg.text_loss_dist
        
        self.infer_text_exist_threshold = cfg.infer_text_exist_threshold
        self.infer_event_exist_threshold = cfg.infer_event_exist_threshold
        
    def forward(self,
                pitch_spec,
                pitchs,
                pitch_centre,
                freq_spec,
                freqs,
                freq_centre):
        """
        inputs
            pitch_spec: (B, T, P)
            freq_spec: (B, T, F)
        return: 
            output: (B, Q, C)
        """
        B, T, P = pitch_spec.shape
        _B, _T, F = freq_spec.shape
        assert B == _B and T == _T, "无法对齐时间"
        
        pitch_output_size = [T, P+1]
        
        pitch_len = T * P
        freq_len = T * F
        
        if self.use_diff_input:
            d_pitch_spec = pitch_spec[:,1:,:] - pitch_spec[:,:-1,:]
            d_freq_spec = freq_spec[:1:,:] - freq_spec[:,:-1,:]
            pitch_spec = torch.stack([pitch_spec, d_pitch_spec], dim=-1)
            freq_spec = torch.stack([freq_spec, d_freq_spec], dim=-1)
        else:
            pitch_spec = pitch_spec.unsqueeze(-1)
            freq_spec = freq_spec.unsqueeze(-1)
            
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

        output_distillation = self.distllation_head(output_distillation) # (B, N, 1, C_text)
        output_prompts = self.prompt_head(output_prompts) # (B, N, prompt, C_prompts)
        output_events = self.events_head(output_events) # (B, N, E, C_events)

        assert output_distillation.shape[2]==1
        output_distillation = output_distillation.squeeze(2) # (B, N, C)

        outputs = [{
            "text_distillation": output_distillation[b,...], # (Qt, C)
            "text_prompt": output_prompts[b,...], # (Qt, prompt, C)
            "event_out": output_events[b,...], # (Qt, Qe, C)
        } for b in range(B)]

        return outputs
        
    def get_sample_loss(self, output, target):
        """
            output: Dict
            target: {
                "text_emb": (Nt, C_text),
                "start": (Ne,),
                "sustain": (Ne,),
                "pitch": (Ne,) # -1 ~ 84
                "text": List[str] Nt
                "text_idx": (Ne,)
            }
        """
        text_distillation = output["text_distillation"]
        text_exist = text_distillation[:, -1]
        text_pred = text_distillation[:, :-1]
        text_gt = target['text_emb']
        gt_idxs, pred_idxs = self.match_text(output, target)
        
        exist_bool = torch.zeros_like(text_exist, device=text_exist.device, dtype=bool)
        exist_bool[gt_idxs] = True
        loss_exist = F.binary_cross_entropy_with_logits(text_exist, exist_bool.float(),
                                                        pos_weight=torch.tensor([self.pos_weight_exist_text],
                                                                                device=text_exist.device))
        
        text_pred_choiced = text_pred[pred_idxs] # (M, C)
        text_gt_choiced = text_gt[gt_idxs] # (M, C)
        if self.text_loss_dist == "euclidean":
            loss_text = F.mse_loss(text_pred_choiced, text_gt_choiced)
        elif self.text_loss_dist == "cosine":
            text_pred_choiced_normed = F.normalize(text_pred_choiced, dim=-1)
            text_gt_choiced_normed = F.normalize(text_gt_choiced, dim=-1)
            cosine = text_pred_choiced_normed @ text_gt_choiced_normed.T # (M,)
            loss_text = torch.mean(1 - cosine)
        else:
            raise NotImplementedError("wtf")
        
        total_event_loss = 0
        for i in range(pred_idxs.shape[0]):
            pred_idx = pred_idxs[i]
            gt_idx = gt_idxs[i]
            
            if pred_idxs.shape[0] == 0:
                return 0

            temp_events_idxs = target['text_idx'] == gt_idx
            
            sub_target = {
                "start" : target['start'][temp_events_idxs], # (n,)
                "sustain" : target['sustain'][temp_events_idxs], # (n,)
                "pitch" : target['pitch'][temp_events_idxs], # (n,) long
                }
            sub_output = output['event_out'][pred_idx, ...]
            sub_output_event = sub_output[:, :self.output_dim_event]
            sub_output_pitch = sub_output[:, self.output_dim_event:self.output_dim_event+self.output_dim_pitch]
            sub_output_exist = sub_output[:, -1]
            sub_output = {
                "start": sub_output_event[:,0], # (Qe)
                "sustain": sub_output_event[:,1], # (Qe,)
                "pitch_logits": sub_output_pitch, # (Qe, P)
                "exist": sub_output_exist, # (Qe,)
            }
            event_loss = self.get_event_loss(output = sub_output, target = sub_target)
            total_event_loss += event_loss / (pred_idxs.shape[0] + 1e-6)
        
        loss =  self.loss_weight['sub'] * total_event_loss +\
                self.loss_weight['exist_text'] * loss_exist +\
                self.loss_weight['text'] * loss_text

        return loss

    def infer(self, output):
        """
        output: Dict
        {
            "text_distillation": output_distillation[b,...], # (Qt, C)
            "text_prompt": output_prompts[b,...], # (Qt, prompt, C)
            "event_out": output_events[b,...], # (Qt, Qe, C)
        }
        """
        threshold_text = self.infer_text_exist_threshold
        threshold_event = self.infer_event_exist_threshold
        
        text_exist_logits = output["text_distillation"][:, -1]
        text_exist_prob = torch.sigmoid(text_exist_logits)
        text_exist_bool = text_exist_prob > threshold_text
        
        text_exist_idxs = torch.where(text_exist_bool)[0]
        
        results = []
        for text_id in text_exist_idxs:
            text_distillation = output["text_distillation"][text_id, :-1]
            text_prompt = output["text_prompt"][text_id, :]
            temp_event = output["event_out"][text_id, :, :] # (Qe, C)
            
            event_exist_logits = temp_event[:,-1]
            event_exist_prob = torch.sigmoid(event_exist_logits)
            event_exist_bool = event_exist_prob > threshold_event
            
            event_exist_idxs = torch.where(event_exist_bool)[0]
            
            if event_exist_idxs.shape[0] == 0:
                continue
            event_choiced = temp_event[event_exist_idxs, :-1] # (Ne, C)
            
            starts = event_choiced[:,0]
            sustains = torch.exp(event_choiced[:,1]) * self.sustain_ref
            pitch_logits = event_choiced[:, self.output_dim_event:self.output_dim_event+self.output_dim_pitch]
            pitchs = torch.argmax(pitch_logits, dim=-1) # (Ne,)
            results.append({
                "text_distillation": text_distillation,
                "text_prompt": text_prompt,
                "start": starts.float(),
                "sustain": sustains.float(),
                "pitch": pitchs
            })
        return results
        

    def get_loss(self, outputs, targets):
        loss = 0
        for b in range(len(outputs)):
            loss += self.get_sample_loss(outputs[b], targets[b])
        return loss

    def get_event_loss(self, output, target):
        """
            output: {
                'start': (Q,)
                'sustain': (Q,) # 这里实际上是 log sustain
                'pitch_logits': (Q, P)
                'exist': (Q,)
            }
            target: {
                'start': (N,)
                'sustain': (N,)
                'pitch': (N,) long
            }
        """
        (gt_idx, pred_idx), _ = self.match_event(output, target)
        
        gt_start_choiced = target['start'][gt_idx]
        gt_sustain_choiced = target['sustain'][gt_idx]
        gt_pitch_choiced = target['pitch'][gt_idx]
        
        pred_start_choiced = output['start'][pred_idx]
        pred_sustain_choiced = output['sustain'][pred_idx]
        pred_pitch_choiced = output['pitch_logits'][pred_idx, :]
        
        loss_start = F.mse_loss(pred_start_choiced, gt_start_choiced)
        loss_sustain = F.mse_loss(pred_sustain_choiced, torch.log(gt_sustain_choiced + 1e-6)-math.log(self.sustain_ref))
        loss_pitch = F.cross_entropy(pred_pitch_choiced, gt_pitch_choiced)
        
        exist_bool = torch.zeros_like(output['exist'], device=output['exist'].device, dtype=bool)
        exist_bool[pred_idx] = True
        loss_exist = F.binary_cross_entropy_with_logits(output['exist'], exist_bool.float(),
                                                        pos_weight=torch.tensor([self.pos_weight_exist_event],
                                                        device=exist_bool.device))
        
        loss = self.loss_weight['start'] * loss_start +\
               self.loss_weight['sustain'] * loss_sustain +\
               self.loss_weight['pitch'] * loss_pitch +\
               self.loss_weight['exist_event'] * loss_exist
    
        return loss
    
    def match_event(self, output, target):
        """
            output: {
                'start': (Q,)
                'sustain': (Q,) # 这里实际上是 log sustain
                'pitch_logits': (Q, P)
                'exist': (Q,)
            }
            target: {
                'start': (N,)
                'sustain': (N,)
                'pitch': (N,) long
            }
        """
        diff_start = target['start'][:,None] - output['start'][None,:] # (N, Q)
        cost_start = diff_start**2 # (N, Q)
        
        diff_sustain = (torch.log(target['sustain'][:,None] + 1e-6)-math.log(self.sustain_ref)) - output['sustain'][None,:] # (N, Q)
        cost_sustain = diff_sustain**2 # (N, Q)
        
        log_prob = F.log_softmax(output['pitch_logits'], dim=-1) # (Q, P)
        cost_pitch = -log_prob[:, target['pitch']].T # (N, Q)
        
        exist_prob = torch.sigmoid(output['exist']) # (Q,)
        exist_logprob = - torch.log(exist_prob)[None, :]
        
        cost = self.cost_weight['start'] * cost_start +\
                self.cost_weight['sustain'] * cost_sustain +\
                self.cost_weight['pitch'] * cost_pitch +\
                self.cost_weight['exist'] * exist_logprob
                
        gt_idxs, pred_idxs = hungarian_match(cost)
        total_cost = cost[gt_idxs, pred_idxs].sum()
        return (gt_idxs, pred_idxs), total_cost

    def match_text(self, output, target):
        """
        output: Dict{
            "event_out": (Qt, Qe, C_e),
            "text_distillation": (Qt, C_text),
        }
        target: Dict{
            "text_emb": (Nt, C_text),
            "start": (Ne,),
            "sustain": (Ne,),
            "pitch": (Ne,), # -1 ~ 84
            "text_idx": (Ne,), # long 0 ~ Ne - 1
        }
        """
        # == 构造cost矩阵
        gt_timbre_idxs = target['text_idx']
        
        Nt = target['text_emb'].shape[0]
        gt_timbre_idxs_unique = gt_timbre_idxs.unique()
        assert len(gt_timbre_idxs_unique)==Nt
        
        Qt = output['event_out'].shape[0]
        
        cost_matrix = torch.zeros((Nt, Qt), device=output['event_out'].device)
        
        for gt_idx in gt_timbre_idxs_unique:
            Ne_choice_bool = (gt_timbre_idxs==gt_idx)
            
            gt_event = {
                'start': target['start'][Ne_choice_bool],
                'sustain': target['sustain'][Ne_choice_bool],
                'pitch': target['pitch'][Ne_choice_bool]
            }
            
            for pred_idx in range(Qt):
                event_out = output['event_out'][pred_idx] # (Qe, C_e)

                pred_event = {
                    'start': event_out[:,0],
                    'sustain': event_out[:,1],
                    'pitch_logits': event_out[:,2:2+self.output_dim_pitch],
                    'exist': event_out[:,-1]
                }
                
                _, total_cost = self.match_event(pred_event, gt_event)
                cost_matrix[gt_idx, pred_idx] = total_cost

        cost_text = self.get_text_cost_matrix_with_confidence(output['text_distillation'], target['text_emb'])

        # te: text event union cost
        cost_te_matrix = self.cost_weight['te_text'] * cost_text +\
            self.cost_weight['te_event'] * cost_matrix

        gt_idxs, pred_idxs = hungarian_match(cost_te_matrix)
        return gt_idxs, pred_idxs

    def get_text_cost_matrix_with_confidence(self, output, target):
        """
            target: (N, C)
            output: (Q, C+1)
        """
        exist = output[:, -1]
        pred = output[:, :-1]
        assert pred.shape[1] == target.shape[1]
        
        exist_prob = torch.sigmoid(exist) # (Q,)
        exist_logprob = - torch.log(exist_prob)[None, :]
        
        cost_exist = exist_logprob
        cost_text = self.get_text_cost_matrix(pred, target)
        
        cost = self.cost_weight['text'] * cost_text +\
            self.cost_weight['text_exist'] * cost_exist
        return cost
        

    def get_text_cost_matrix(self, output, target):
        """
            target: (N, C)
            output: (Q, C)
        """
        assert output.shape[1]==target.shape[1]
        if self.text_cost_dist == "euclidean":
            diff = target[:,None,:] - output[None,:,:] # (N, Q, C)
            cost = torch.sum(diff**2, dim=-1) # (N, Q)
        elif self.text_cost_dist == "cosine":
            target = F.normalize(target, dim=-1) # (N,C)
            output = F.normalize(output, dim=-1) # (Q,C)
            cosine = target @ output.T # (N, Q)
            cost = 1 - cosine
        else:
            raise NotImplementedError("wtf")
        return cost

    def match_text_old(self, output, target):
        """
            target: (N, C)
            output: (Q, C)
        """
        cost = self.get_text_cost_matrix(output, target)
        gt_idxs, pred_idxs = hungarian_match(cost)
        return gt_idxs, pred_idxs


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
        

        
        