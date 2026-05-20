"""
patience and love for our baby model
from simple task to complex
establish a simple dataset for baby model (CQTEncoder)


用原始的 maj, min, N, dom, dim, aug 和弦表示，适配主流数据集
而且 model 的 modal 输出要能预测和弦，作为pretrain
detr作为finetune
"""
from einops import rearrange
import torch
from configs.config6 import get_config
from torch import nn
import math
import torch.nn.functional as F
from typing import Callable, Optional, Union, Dict
from scipy.optimize import linear_sum_assignment
from spec.cqt import MultiWindowCQT, get_freqs
from .cell import Cells

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
    def __init__(self, layer_idx, cfg):
        super().__init__()
        
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

    def __init__(self, layer_idx, cfg):
        super().__init__()
        self.layer_idx = layer_idx
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
    def __init__(self, layer_idx, cfg):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Qwen2Attention(layer_idx=layer_idx, cfg=cfg)
        self.mlp = Qwen2MLP(layer_idx=layer_idx, cfg=cfg)
        self.input_layernorm = Qwen2RMSNorm(cfg.detr_d_model_list[layer_idx])
        self.post_attention_layernorm = Qwen2RMSNorm(cfg.detr_d_model_list[layer_idx])
        
        self.time_mask_len = cfg.time_mask_len
        
        self.ffn_dim_up = cfg.ffn_dim_up[layer_idx]
        self.pool_stride = cfg.pool_stride[layer_idx]
    def forward(
        self,
        modal_dict: Dict[str, torch.Tensor],
        only_object = False, # 只当作 transformer encoder
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[Cache] = None,
        # use_cache: Optional[bool] = False,
        # cache_position: Optional[torch.LongTensor] = None,
        # position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        # **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        pitch = modal_dict['object'] # (B, T, C)
        text = modal_dict['subject'] # (B, Q, C)
        
        B, T, C = pitch.shape
        _B, Q, _C = text.shape
        assert _C==C
        
        if only_object:
            Q=0
        
        TaQ = T + Q
        
        if only_object:
            hidden = pitch
        else:
            hidden = torch.concat([pitch, text], dim=1) # (B, TaQ, C)

        residual = hidden
        
        hidden = self.input_layernorm(hidden)
        
        if self.time_mask_len is None:
            attention_mask = None
        else:
            attention_mask = torch.ones((TaQ, TaQ), device=pitch.device) * float("-inf")
            # 1. 每个 T 都能看见 L 和前后 self.time_mask_len 的 T
            # 2. 每个 L 都能看见所有
            text_idx = torch.arange(T, TaQ)
            time_idx = torch.arange(T)
            # 每个 time token 能看到自身 ± time_mask_len + 所有文本
            for t in time_idx:
                start = max(0, t - self.time_mask_len)
                end = min(T, t + self.time_mask_len + 1)
                attention_mask[t, start:end] = 0  # 允许看到的 T
                attention_mask[t, text_idx] = 0   # 允许看到所有文本
            # 文本 token 全可见
            attention_mask[text_idx[:, None], :] = 0

        hidden = self.self_attn(
            hidden_states=hidden,
            attention_mask=attention_mask,
        )
        hidden = hidden + residual
        
        pitch = hidden[:, :T, :]
        text = hidden[:, T:, :]
        
        # 先升维，再池化

        ffn_dim_up = self.ffn_dim_up

        text = text # text 不走 mlp升维 因为它接下来要去 Cell 里面做 inner，保证信息不编造
        
        _pitch = self.post_attention_layernorm(pitch) # (B, T, C)
        _pitch = self.mlp(_pitch)
        pitch_repeated = pitch.unsqueeze(-1).expand(-1,-1,-1,ffn_dim_up)
        pitch_repeated = torch.flatten(pitch_repeated, -2, -1)
        pitch = pitch_repeated + _pitch
        
        if self.pool_stride is not None:
            pitch = temporal_pool(pitch, stride=self.pool_stride)
            
        modal_dict = {
            "subject": text,
            "object": pitch,
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


def build_harmony_kernel(kernel_size=36,
                       cycle=7,
                       tau=12):
    """
    return: (kernel_size,), (kernel_size,)
    """
    ts = torch.arange(kernel_size)
    harmony = torch.cos(2 * math.pi / cycle * ts)
    harmony = harmony ** 8 # 瘦，但是偶数
    envelope = torch.exp(- ts / tau)
    return harmony, envelope

class HarmonyConv(nn.Module):
    def __init__(self,
                 cycles,
                 kernel_size,
                 trainable,
                 taus):
        super().__init__()

        kernels = []
        envelopes = []
        for cycle, tau in zip(cycles, taus):
            kernel, envelope = build_harmony_kernel(kernel_size, cycle, tau)
            kernels.append(kernel)
            envelopes.append(envelope)

        kernels = torch.stack(kernels, dim=0) # (C_out, kernel_size)
        kernels = kernels[:,None,:] # (C_out, 1, kernel_size) 方便后续卷积
        if trainable:
            self.kernels = nn.Parameter(kernels)
        else:
            self.register_buffer("kernels", kernels)
        self.register_buffer("envelopes", torch.stack(envelopes, dim=0)[:,None,:]) # (C_out, 1, kernel_size)
    def forward(self, cqt):
        """
            cqt: (B*D*T, 1, P)
            return: (B*D*T, H, P)
        """
        kernel_size = self.kernels.shape[2]
        padding_len = kernel_size - 1
        
        kernels = self.kernels * self.envelopes # (harmony, kernel_size)

        x = F.pad(cqt, (0, padding_len))
        y = F.conv1d(x, kernels, padding=0) # (B*T, harmony, P)
        return y



# 300 epoch 即可收敛
# 归纳不同包络特质的音色
# 另外，提升frame wise任务的时间分辨率
class EnvelopeConv(nn.Module):
    def __init__(self,
                 kernel_size,
                 E, # 每个 P 的模式
                 P, # P 的个数
                 ):
        super().__init__()
        self.P = P
        self.O = math.ceil(P / 12)
        self.kernel_size = kernel_size
        kernels = torch.empty(self.O, E, 1, kernel_size)
        nn.init.kaiming_normal_(kernels, mode="fan_in", nonlinearity="linear")
        self.kernels = nn.Parameter(kernels)
        
    def forward(self, x):
        """
        x: (P, B*D, 1, T)
        P 分布在 T 上没有差异
        但是 T 分布在 P 上有差异
        所以要区分 P 个通道
        每个通道有 E 种模式 (envelope)
        
        return: (P, B*D, E, T)
        """
        P,B,c,T = x.shape
        assert c==1
        assert P==self.P
        ys = []
        for p in range(P):
            o = math.floor(p / 12)
            y = F.conv1d(x[p,...], self.kernels[o,...], padding=self.kernel_size // 2) # (B*D, E, T)
            ys.append(y)
        ys = torch.stack(ys,dim=0) # (P, B*D, E, T)
        return ys


class MLP(nn.Module):
    def __init__(self, d_input, d_output, intermediate_size):
        super().__init__()
        cfg = get_config()
        
        self.d_input = d_input
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


class HarmonyBlock(nn.Module):
    def __init__(self, D, cycles, taus, kernel_size):
        super().__init__()
        H = len(cycles)
        self.before_conv_layernorm = Qwen2RMSNorm(D)
        self.harmony_conv = HarmonyConv(
            cycles = cycles,
            kernel_size=kernel_size,
            trainable=True,
            taus = taus
        )
        self.mix_proj = nn.Linear(D*H, D) # inductive bias
        # self.before_mlp_layernorm = Qwen2RMSNorm(D)
        
        # self.mlp = MLP(D, D, D*4)
    def forward(self, x):
        """
        x: (B, T, P, D)
        """
        B, T, P, D = x.shape
        
        # residual = x
        x = self.before_conv_layernorm(x)
        
        x = x.permute(0,1,3,2) # (B, T, D, P)
        x = x.flatten(0,2)
        x = x[:,None,:] # (B*T*D, 1, P)
        x = self.harmony_conv(x) # (B*T*D, H, P)
        x = x.reshape(B,T,D,-1,P) # (B, T, D, H, P)
        x = x.permute(0,1,4,2,3) # (B, T, P, D, H)
        x = x.flatten(-2,-1) # (B, T, P, D*H)
        x = self.mix_proj(x) # (B, T, P, D)
        # x = x + residual # inductive bias by conv
        
        # residual = x
        # x = self.before_mlp_layernorm(x)
        # x = self.mlp(x) # (B, T, P, D)
        # x = x + residual # inductive bias by mlp
        return x


class EnvelopeBlock(nn.Module):
    def __init__(self, D, kernel_size, E, P):
        super().__init__()
        self.E=E
        self.before_conv_layernorm = Qwen2RMSNorm(D)
        self.envelope_conv = EnvelopeConv(kernel_size, E, P)
        self.mix_proj = nn.Linear(D*E, D) # inductive bias
        self.before_mlp_layernorm = Qwen2RMSNorm(D)
        # self.mlp = MLP(D, D, D*4)
    def forward(self, x):
        """
        x: (B, T, P, D)
        """
        B, T, P, D = x.shape
        E=self.E
        
        # residual = x
        x = self.before_conv_layernorm(x) # (B, T, P, D)
        x = x.permute(2, 0, 3, 1) # (P, B, D, T)
        x = x.flatten(1,2) # (P, B*D, T)
        x = x.unsqueeze(2) # (P, B*D, 1, T)
        x = self.envelope_conv(x) # (P, B*D, E, T)
        x = x.reshape(P,B,D,E,T) # (P,B,D,E,T)
        x = x.permute(1,4,0,2,3) # (B, T, P, D, E)
        x = x.flatten(-2,-1) # (B, T, P, D*E)
        x = self.mix_proj(x) # (B, T, P, D)
        # x = x + residual # inductive bias by conv
        
        # residual = x
        # x = self.before_mlp_layernorm(x)
        # x = self.mlp(x) # (B, T, P, D)
        # x = x + residual # inductive bias by mlp
        return x



def dilation_pool(x, dilation):
    # x: (B, P, C) -> (B, dilation, C)
    assert x.dim()==3
    P = x.shape[1]
    arange = torch.arange(P, device=x.device)
    res = []
    for d in range(dilation):
        choice = (arange % dilation)==d
        res.append(x[:,choice,:].mean(1))
    return torch.stack(res, dim=1)


# def build_pitch_encoding(D, N):
#     freqs = 2**(torch.arange(12)/12)
#     phase = 2*math.pi * torch.arange(D)[None,:]/D * freqs[:,None]
#     cosine = torch.cos(phase) # (12, D)
#     return cosine

def pitch_encoding_init(N, D):
    """
    return:
        (12, D)
    """
    pitch = torch.arange(N).float()  # (12,)
    theta = 2 * math.pi * pitch / N  # (12,)
    emb = []
    num_freq = D // 2
    for k in range(1, num_freq + 1):
        emb.append(torch.cos(k * theta))
        emb.append(torch.sin(k * theta))
    emb = torch.stack(emb, dim=-1)  # (12, 2*num_freq)
    # trim
    emb = emb[:, :D]
    # normalize
    emb = emb / emb.std()
    return emb


class Affine(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))
    def forward(self, x):
        return self.scale * x + self.bias
        
NoExistValue = 5
ExistValue = 0
ChordClassifierMixin_modes = ['symbol', 'pitch', 'midi', "onset"]
class ChordClassifierMixin:
    def set_mode(self, mode):
        assert mode in ChordClassifierMixin_modes
        self.mode = mode
    def get_loss(self, **kw_args):
        if self.mode=="symbol":
            return self.get_symbol_loss(**kw_args)
        elif self.mode=="pitch":
            return self.get_pitch_loss(**kw_args)
        elif self.mode=="midi":
            return self.get_midi_loss(**kw_args)
        elif self.mode=="onset":
            return self.get_onset_loss(**kw_args)
        else:
            raise ValueError('wtf')

    def get_onset_loss(self, x, target, sigma=0.1, radius=1):
        """
        target: List Dict{
            "start": (N,) float
        }
        x: B, T, D
        """
        B, T, D = x.shape
        
        stride = self.stride
        sr = self.sr
        
        onset_target = torch.zeros((B, T, 1), device=x.device)
        for b in range(B):
            for t in target[b]['start']:
                idx = int(t / stride)
                                
                for dt in range(-radius, radius+1):

                    j = idx + dt

                    if 0 <= j < T:

                        v = math.exp(
                            -0.5 * (dt / sigma) ** 2
                        )

                        onset_target[b, j, 0] = max(
                            onset_target[b, j, 0],
                            v
                        )
        
        onset_logits = self.head_onset(x) # (B, T, 1)
        
        loss = F.binary_cross_entropy_with_logits(onset_logits, onset_target)
        
        return loss
        
    def get_midi_loss(self, x, target, eps=1e-20):
        """
        target: List Dict{
            "midi_manyhot": (P,) 0~1
        }
        x: (B, T, P, D)
        """
        B, T, P, D = x.shape
        assert P==self.P
        x = x.mean(1) # (B, P, D)
        
        midi_tar = [temp['midi_manyhot'] for temp in target]
        midi_tar = torch.stack(midi_tar, dim=0) # (B, P)
        
        midi_logits = self.head_midi(x).squeeze(-1) # (B, P)
        # 偏向 neg, 提升 precision
        loss_midi = F.binary_cross_entropy_with_logits(midi_logits, midi_tar,
                                                        pos_weight=torch.tensor(0.5 ,device=x.device))
        
        
        pitch_tar = dilation_pool(midi_tar.unsqueeze(-1), 12).squeeze(-1).bool().float() # (B, 12)
        pitch_logits = dilation_pool(midi_logits.unsqueeze(-1), 12).squeeze(-1) # (B, 12)
        loss_pitch = F.binary_cross_entropy_with_logits(pitch_logits, pitch_tar)
        
        # octave_tar, 被激活的 octave
        # avg_pool1d (B, C, T) -> (B, C, T1)
        octave_tar = F.avg_pool1d(midi_tar.unsqueeze(1), 12, 12).squeeze(1).bool().float()
        octave_logits = F.avg_pool1d(midi_logits.unsqueeze(1), 12, 12).squeeze(1)
        loss_octave = F.binary_cross_entropy_with_logits(octave_logits, octave_tar)
        
        # root_tar, 对于 root_cls 进行一次，增加权重
        loss_root = 0
        for b in range(B):
            root_tar = torch.tensor(1., device=x.device)
            root_idx = torch.where(midi_tar[b,:])[0]
            if root_idx.numel() != 0:
                root_idx = root_idx.min()
                root_idx = root_idx % 12
                root_logits = pitch_logits[b, root_idx]
                loss_root += F.binary_cross_entropy_with_logits(root_logits, root_tar) / B
            
        loss = self.loss_weight['midi'] * loss_midi \
            + self.loss_weight['pitch'] * loss_pitch \
            + self.loss_weight['octave'] * loss_octave \
            + self.loss_weight['root'] * loss_root
        
        return loss

    def _get_midi_loss(self, x, target, eps=1e-20):
        """
        target: List Dict{
            "midi": (N,) int
            "midi_manyhot": (P,) 0~1
        }
        x: (B, T, P, D)
        """
        B, T, P, D = x.shape
        assert P==self.P
        x = x.mean(1) # (B, P, D)
        
        midi_tar = [temp['midi_manyhot'] for temp in target]
        midi_tar = torch.stack(midi_tar, dim=0) # (B, P)
        midi_tar_norm = torch.norm(
            midi_tar,
            dim=-1,
            keepdim=False
        ) # (B, 1)
        
        exists = [temp['exist'] for temp in target]
        exists = torch.stack(exists, dim=0).squeeze(-1).long() # (B,)
        exists = exists.bool() & midi_tar_norm.bool() # (B,)
        
        midi_logits = self.head_midi(x).squeeze(-1) # (B, P)
        midi_logits = torch.sigmoid(midi_logits)
        # midi_logits = F.softplus(midi_logits) # (B, P) float 0~1
        print(midi_logits[0,:])

        harmony_prior = self.harmony_prior # (P, P) 确保 (diag()==0).all()
        
        grad_x = midi_tar @ harmony_prior # (B, P)
        x_dot_y = (grad_x * midi_logits).sum(-1) # (B,)
        
        # y_dot_y = ((midi_tar @ harmony_prior) * midi_tar).sum(-1) # (B, P)
        assert ~torch.isnan(x_dot_y).any(), f"{midi_logits}"
        
        loss_midi = x_dot_y
        
        loss = (loss_midi.sum(-1)).mean()
        
        # pitch_tar = dilation_pool(midi_tar, 12) # (B, 12)
        
                
        return loss

    def get_pitch_loss(self, x, target):
        """
        target: List Dict{
            "root": (1,), # 0~11
            "pitch_cls_vec": (12,), # 0~1
            "chord_idx": (1,), # 0~4, maj,min,dom,dim,aug
        }
        x: (B, T, D)
        """
        exists = [temp['exist'] for temp in target]
        exists = torch.stack(exists, dim=0).squeeze(-1).long() # (B,)
        
        exist_tar = exists.unsqueeze(-1).float() # (B,1)
        exist_logits = self.head_exist(x) # (B, T, 1)
        exist_logits = exist_logits.mean(1) # (B,1)
        
        loss_exist = F.binary_cross_entropy_with_logits(exist_logits, exist_tar)
        
        target = [
            t for t, e in zip(target, exists)
            if e == 1
        ]
        
        x = x[exists.bool(), :, :]
        
        B, T, D = x.shape
        
        # pitch
        pitch_tar = [temp['pitch_cls_vec'] for temp in target]
        pitch_tar = torch.stack(pitch_tar, dim=0) # (B,12)
        
        pitch_logits = self.head_pitch(x) # (B, T, 12)
        pitch_logits = pitch_logits.mean(1) # (B, 12)
        
        loss_pitch = F.binary_cross_entropy_with_logits(pitch_logits, pitch_tar)
        
        loss = self.loss_weight['pitch'] * loss_pitch + \
                self.loss_weight['exist'] * loss_exist
        return loss
        
    def get_symbol_loss(self, x, target):
        """
        target: List Dict{
            "root": (1,), # 0~11
            "pitch_cls_vec": (12,), # 0~1
            "chord_idx": (1,), # 0~4, maj,min,dom,dim,aug
        }
        x: (B, T, D)
        """
        exists = [temp['exist'] for temp in target]
        exists = torch.stack(exists, dim=0).squeeze(-1).long() # (B,)
        
        exist_tar = exists.unsqueeze(-1).float() # (B,1)
        exist_logits = self.head_exist(x) # (B, T, 1)
        exist_logits = exist_logits.mean(1) # (B,1)
        
        loss_exist = F.binary_cross_entropy_with_logits(exist_logits, exist_tar)
        
        target = [
            t for t, e in zip(target, exists)
            if e == 1
        ]
        
        x = x[exists.bool(), :, :]
        
        B, T, D = x.shape
        
        root_tar = [temp['root_idx'] for temp in target]
        root_tar = torch.stack(root_tar, dim=0).squeeze(-1) # (B,)
        chord_tar = [temp['chord_idx'] for temp in target]
        chord_tar = torch.stack(chord_tar, dim=0).squeeze(-1) # (B,)
        
        # root
        root_logits = self.head_root(x) # (B, T, 12)
        root_logits = root_logits.mean(1) # (B, 12)
        loss_root = F.cross_entropy(root_logits, root_tar)
        
        # chord
        chord_logits = self.head_chord(x) # (B, T, 5)
        B, T, C = chord_logits.shape
        chord_logits = chord_logits.mean(1) # (B, 5)
        loss_chord = F.cross_entropy(chord_logits, chord_tar)
        
        loss = self.loss_weight['root'] * loss_root + \
                self.loss_weight['chord'] * loss_chord + \
                self.loss_weight['exist'] * loss_exist
        
        return loss

    @torch.no_grad()
    def infer(self, **kw_args):
        if self.mode=="symbol":
            return self.infer_symbol(**kw_args)
        elif self.mode=="pitch":
            return self.infer_pitch(**kw_args)
        elif self.mode=="midi":
            return self.infer_midi(**kw_args)
        elif self.mode=="onset":
            return self.infer_onset(**kw_args)
        else:
            raise ValueError("wtf")

    @torch.no_grad()
    def infer_onset(
        self,
        x,
        threshold=0.5
    ):
        """
        x: (B,T,D)
        """

        B,T,D = x.shape
        assert B==1

        stride = self.stride
        sr = self.sr

        onset_logits = self.head_onset(x)

        onset_prob = torch.sigmoid(
            onset_logits
        ).squeeze(-1) # (B,T)

        results = []

        for b in range(B):

            prob = onset_prob[b]

            peaks = []

            for t in range(1, T-1):

                if (
                    prob[t] > threshold
                    and
                    prob[t] >= prob[t-1]
                    and
                    prob[t] >= prob[t+1]
                ):

                    sec = (
                        t
                        * stride
                        / sr
                    )

                    peaks.append({
                        "frame": t,
                        "time": sec,
                        "conf": float(
                            prob[t].item()
                        )
                    })

            results.append({
                "onsets": peaks,
                "prob": prob.detach().cpu(),
            })

        if B == 1:
            return results[0]

        return results

    @torch.no_grad()
    def infer_midi(
        self,
        x,
        exist_threshold=0.3,
        midi_threshold=0.15
    ):
        """
        x: (B, T, P, D)
        """

        B, T, P, D = x.shape
        x = x.mean(1) # (B, P, D)
        # =========================
        # harmonic field
        # =========================

        midi_logits = self.head_midi(x).squeeze(-1) # (B,P)
        midi_prob = torch.sigmoid(midi_logits)

        # threshold on direction
        midi_binary = midi_prob > midi_threshold

        results = []

        for b in range(B):

            active_midi_idx = (
                torch.nonzero(
                    midi_binary[b],
                    as_tuple=False
                ).squeeze(-1)
                + self.min_midi
            ).tolist()

            if isinstance(active_midi_idx, int):
                active_midi_idx = [active_midi_idx]

            symbol = (
                "[" + ",".join(map(str, active_midi_idx)) + "]"
            )

            results.append({

                "midi_binary":
                    midi_binary[b].detach().cpu(),

                "midi":
                    active_midi_idx,

                "symbol":
                    symbol,
            })

        if B == 1:
            return results[0]

        return results

    @torch.no_grad()
    def _infer_midi(
        self,
        x,
        exist_threshold=0.5,
        midi_threshold=0.15,
        eps=1e-8
    ):
        """
        x: (B, T, P, D)
        """

        B, T, P, D = x.shape
        x = x.mean(1) # (B, P, D)
        # =========================
        # logits
        # =========================

        midi_logits = self.head_midi(x).squeeze(-1) # (B,T,P)

        midi_logits = midi_logits # (B,P)

        midi_logits = F.softplus(midi_logits) # (B, P)

        midi_logits_norm = torch.norm(midi_logits, dim=-1, keepdim=True) # (B, 1)
        noExist_logits = torch.exp(-midi_logits_norm)
        # midi_logits = F.pad(midi_logits, (1, 0), value=1) # (B, 1+P)
        midi_logits = torch.concat([noExist_logits, midi_logits], dim=-1) # (B, 1+P)
        # 归一化
        # 由于 1 的存在，导致 logits 需要累积才能使得 no_exist 维度几乎为 0
        midi_logits_norm = torch.norm(midi_logits, dim=-1, keepdim=True) # (B, 1)
        midi_direction = midi_logits / midi_logits_norm # (B, 1+P)
        
        # =========================
        # no-exist dim
        # =========================

        no_exist_score = midi_direction[:,0]

        exist_prob = 1 - no_exist_score

        exist = exist_prob > exist_threshold

        # =========================
        # midi dims
        # =========================

        midi_prob = midi_direction[:,1:] # (B,P)

        midi_binary = (
            midi_prob > midi_threshold
        )

        # 如果 no-exist 很高
        # 强制清空
        for b in range(B):

            if not exist[b]:
                midi_binary[b] *= False

        # =========================
        # build results
        # =========================

        results = []

        for b in range(B):

            active_midi_idx = (
                torch.nonzero(
                    midi_binary[b],
                    as_tuple=False
                ).squeeze(-1)
                + self.min_midi
            ).tolist()

            if isinstance(
                active_midi_idx,
                int
            ):
                active_midi_idx = [
                    active_midi_idx
                ]

            symbol = (
                "[" + ",".join(
                    map(str, active_midi_idx)
                ) + "]"
                if exist[b]
                else "N"
            )

            results.append({

                "exist":
                    bool(exist[b].item()),

                "exist_conf":
                    float(exist_prob[b].item()),

                "no_exist_score":
                    float(no_exist_score[b].item()),

                "midi_prob":
                    midi_prob[b]
                    .detach()
                    .cpu(),

                "midi_binary":
                    midi_binary[b]
                    .detach()
                    .cpu(),

                "midi":
                    active_midi_idx,

                "symbol":
                    symbol,
            })

        if B == 1:
            return results[0]

        return results

    @torch.no_grad()
    def infer_pitch(self, x, exist_threshold=0.5, pitch_threshold=0.5):
        """
        x: (B, T, D)
        """

        B, T, D = x.shape

        pitch_names = [
            "C", "C#", "D", "D#", "E", "F",
            "F#", "G", "G#", "A", "A#", "B"
        ]

        # pitch
        pitch_logits = self.head_pitch(x)      # (B,T,12)
        pitch_logits = pitch_logits.mean(1)    # (B,12)

        pitch_prob = torch.sigmoid(pitch_logits)  # (B,12)

        pitch_binary = pitch_prob > pitch_threshold  # (B,12)

        # exist
        exist_logits = self.head_exist(x)      # (B,T,1)
        exist_logits = exist_logits.mean(1).squeeze(-1)  # (B,)

        exist_prob = torch.sigmoid(exist_logits)
        exist = exist_prob > exist_threshold

        results = []

        for b in range(B):

            active_pitch_idx = torch.nonzero(
                pitch_binary[b],
                as_tuple=False
            ).squeeze(-1).tolist()

            if isinstance(active_pitch_idx, int):
                active_pitch_idx = [active_pitch_idx]

            active_pitch_names = [
                pitch_names[i]
                for i in active_pitch_idx
            ]

            if exist[b]:
                symbol = "[" + ",".join(active_pitch_names) + "]"
            else:
                symbol = "N"

            results.append({
                "exist": bool(exist[b].item()),
                "exist_conf": float(exist_prob[b].item()),

                "pitch_prob": pitch_prob[b].detach().cpu(),
                "pitch_binary": pitch_binary[b].detach().cpu(),

                "pitch_idx": active_pitch_idx,
                "pitch_names": active_pitch_names,

                "symbol": symbol,
            })

        if B == 1:
            return results[0]

        return results

    @torch.no_grad()
    def infer_symbol(self, x, exist_threshold=0.5):
        """
        x: (B, T, D)
        """

        B, T, D = x.shape

        root_names = [
            "C", "C#", "D", "D#", "E", "F",
            "F#", "G", "G#", "A", "A#", "B"
        ]

        chord_names = [
            "maj",
            "min",
            "dom",
            "dim",
            "aug",
            "N",
        ]

        root_logits = self.head_root(x)      # (B,T,12)
        root_logits = root_logits.mean(1)    # (B,12)
        root_prob = torch.softmax(root_logits, dim=-1)
        root_idx = root_prob.argmax(dim=-1)  # (B,)

        chord_logits = self.head_chord(x)    # (B,T,5)
        chord_logits = chord_logits.mean(1)  # (B,5)
        chord_prob = torch.softmax(chord_logits, dim=-1)
        chord_idx = chord_prob.argmax(dim=-1)  # (B,)

        exist_logits = self.head_exist(x) # (B,T,1)
        exist_logits = exist_logits.mean(1).squeeze(-1) # (B) # 如果 < threshold, 返回 N
        exist_prob = torch.sigmoid(exist_logits)
        exist = exist_prob > exist_threshold
        results = []

        for b in range(B):
            r = int(root_idx[b].item())
            c = int(chord_idx[b].item())
            e = exist[b]
            
            symbol = f"{root_names[r]}:{chord_names[c]}" if e else "N"

            results.append({
                "exist": exist,
                
                "root_idx": r,
                "root_name": root_names[r],
                "root_conf": float(root_prob[b, r].item()),

                "chord_idx": c,
                "chord_name": chord_names[c],
                "chord_conf": float(chord_prob[b, c].item()),

                "root_prob": root_prob[b].detach().cpu(),
                "chord_prob": chord_prob[b].detach().cpu(),

                "symbol": symbol,
            })

        if B == 1:
            return results[0]

        return results


# baby model
class CQTEncoder(ChordClassifierMixin, nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.harmony_conv.backbone_output_dim
        self.D = D
        
        self.min_midi = cfg.min_midi
        
        self.stride = cfg.cqt_stride
        self.sr = cfg.sr
        
        self.num_layers = cfg.harmony_conv.num_layers

        self.prior_affine = nn.Linear(cfg.input_dim, D)
        self.Players = nn.ModuleList([
            HarmonyBlock(
                D=D,
                cycles=cfg.harmony_conv.cycles,
                taus=cfg.harmony_conv.taus,
                kernel_size=cfg.harmony_conv.kernel_size
            )
            for _ in range(cfg.harmony_conv.num_layers)
        ]) # 不同乐器的泛音列归纳音符，预测 pitch_cls
        
        kernel_size = int(cfg.envelope_conv.receptive_field / cfg.cqt_stride)
        # 保证奇数
        kernel_size = kernel_size+1 if kernel_size%2==0 else kernel_size
        E = cfg.envelope_conv.rep_dim
        P = cfg.max_midi - cfg.min_midi + 1
        self.Tlayers = nn.ModuleList([
            EnvelopeBlock(
                D=D,
                kernel_size=kernel_size,
                E=E,
                P=P,
            )
            for _ in range(cfg.harmony_conv.num_layers)
        ])
        
        self.FFNNorms = nn.ModuleList([
            Qwen2RMSNorm(D)
            for _ in range(cfg.harmony_conv.num_layers)
        ])
        
        self.FFNlayers = nn.ModuleList([
            MLP(D, D, 4*D)
            for _ in range(cfg.harmony_conv.num_layers)
        ])

        # pitch pooling
        # self.pitch_head = nn.Sequential(
        #     nn.Linear(D, D),
        #     nn.GELU(),
        #     nn.Linear(D, 1)
        # )
        # self.chord_attn = nn.Sequential(
        #     nn.Linear(D, D),
        #     nn.GELU(),
        #     nn.Linear(D, 1)
        # )

        # self.out_proj = nn.Sequential(
        #     nn.Linear(2*D, 2*D),
        #     nn.GELU(),
        #     nn.Linear(2*D, D)
        # )
        
        freqs = get_freqs(cfg.min_midi, cfg.max_midi)
        self.register_buffer("freqs", freqs)
        
        harmony_prior = build_euler_cost_matrix(freqs.numpy(), return_log=cfg.use_log_euler_dissonance) # (P,P)
        self.register_buffer("harmony_prior", harmony_prior)
        
        num_freqs = freqs.shape[0]
        self.num_freqs=num_freqs
        self.P=num_freqs
        
        num_octave = math.ceil(num_freqs / 12)
        self.num_octave = num_octave
        # assert D % num_octave == 0
        
        self.register_buffer("pitchs",torch.arange(cfg.min_midi, cfg.max_midi+1)-cfg.min_midi)
        self.register_buffer("octave_affines_choice",self.pitchs // 12)

        out_dim_each_octave = D//num_octave
        out_dim = out_dim_each_octave * num_octave
        
        self.octave_affines = nn.ModuleList([nn.Linear(D, out_dim_each_octave) for _ in range(num_octave)])
        self.post_affine = nn.Linear(out_dim, D) # 初始值需要近似恒等映射
        
        out_dim_each_pitch = D//12
        out_dim_pitch = out_dim_each_pitch * 12
        
        self.share_pitch_affine = cfg.harmony_conv.share_pitch_affine
        
        if not self.share_pitch_affine:
            self.pitch_affines = nn.ModuleList([nn.Linear(D, out_dim_each_pitch) for _ in range(12)])
        else:
            self.pitch_affines = nn.Linear(D, out_dim_each_pitch)

        self.post_pitch_affine = nn.Linear(out_dim_pitch, D)
        # self.pitch_class_emb = nn.Embedding(12, D)
        # self.octave_emb = nn.Embedding(num_octave, D)
        # self.pitch_embedding = nn.Parameter(pitch_encoding_init(12, D))
        # self.register_buffer("pitch_embedding", pitch_encoding_init(12, D))
        
        self.head_root = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 12) # 0~11, N
        )
        
        self.head_chord = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 6) # maj,min,dom,dim,aug,N
        )
        
        self.head_exist = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 1) # N, chord
        )
        
        self.head_pitch = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 1)
        )
        
        self.head_midi = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 1)
        )
        
        self.head_onset = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 1)
        )
        
        self.exist_affine = Affine()
        
        self.loss_weight = cfg.harmony_conv.loss_weight
        self.threshold = cfg.harmony_conv.infer_threshold
        
        self.validShapes = ["BTPD", "BTD"]
        self.outputShape = "BTPD"

    def forward(self, x):
        """
            x: (B, T, P, C)
            return: (B, T, D)
            retain_P: (B, T, P, D)
        """
        B, T, P, C = x.shape
        assert P==self.num_freqs
        x = self.prior_affine(x) # (B, T, P, D)
        # 对于 harmony 和 envelop 应该用 2个 affine
 
        # # pitch embedding
        # midi = torch.arange(P, device=x.device)
        # pitch_class = midi % 12
        # octave = midi // 12
        # pc_emb = self.pitch_class_emb(pitch_class) # (P,D)
        # oct_emb = self.octave_emb(octave) # (P,D)
        # pitch_emb = pc_emb + oct_emb
        # x = x + pitch_emb[None,None,:,:]
        
        # layer
        for i in range(self.num_layers):
            residual = x
            x_p = self.Players[i](x) # (B, T, P, D)
            x_t = self.Tlayers[i](x) # (B, T, P, D)
            x = x_p + x_t + residual
            residual = x
            x = self.FFNNorms[i](x)
            x = self.FFNlayers[i](x)
            x = x + residual # (B, T, P, D)
        
        if self.outputShape=="BTPD":
            return x # (B, T, P, D)
        
        # x12 = dilation_pool(x.flatten(0,1), 12) # (B*T, 12, D)
        # x12 = x12.reshape(B,T,12,self.D) # (B, T, 12, D)
        
        # # attention pooling
        # attn = self.root_attn(x12)  # (B,T,12,1)
        # attn = torch.softmax(attn, dim=2) # (B,T,12,1)
        # root_identity = attn.squeeze(-1) @ self.pitch_embedding # (B, T, D)
        # root_content = (attn * x12).sum(2) # (B, T, D)
        # root_rep = root_identity + root_content

        # attn = self.chord_attn(x) # (B,T,P,1)
        # attn = torch.softmax(attn, dim=2) # (B,T,P,1)
        # chord_rep = (x * attn).sum(2) # (B, T, D)
        
        # rep = torch.concat([root_rep, chord_rep], dim=2) # (B,T,2*D)
        # rep = self.out_proj(rep) # (B, T, D)
        # rep = chord_rep
        
        # merge octave
        y = []
        for octave in range(self.num_octave):
            start_p = 12*octave
            end_p = min(12*(octave+1), P)
            xp = x[:,:,start_p:end_p,:] # (B,T,12,D)
            yp = self.octave_affines[octave](xp) # (B,T,12,d)
            y.append(yp)
        y = torch.concat(y, dim=-1)
        y = self.post_affine(y) # (B,T,12,D)
        
        if self.share_pitch_affine:
            z = self.pitch_affines(y) # (B,T,12,d)
        else:
            # merge pitch cls
            z = []
            for p in range(12):
                yp = y[:,:,p,:] # (B,T,D)
                zp = self.pitch_affines[p](yp) # (B,T,d)
                z.append(zp)
            z = torch.stack(z, dim=-2) # (B,T,12,d)
        z = z.flatten(-2,-1)
        z = self.post_pitch_affine(z) # (B,T,D)
        if self.outputShape=="BTD":
            return z
        else:
            raise ValueError("wtf")

    # @torch.no_grad()
    # def infer(self, x):
    #     """
    #     x: (B, T, P, D)

    #     return:
    #         List[Dict]
    #     """

    #     B, T, P, D = x.shape
    #     assert B==1
        
    #     # =====================================
    #     # pitch collapse
    #     # =====================================

    #     # x12 = dilation_pool(
    #     #     x.flatten(0, 1),
    #     #     12
    #     # )  # (B*T, 12, D)

    #     # x12 = x12.reshape(B, T, 12, D)
    #     x12 = x

    #     # =====================================
    #     # pitch prediction
    #     # =====================================

    #     pitch_logits = self.pitch_head(
    #         x12
    #     ).squeeze(-1)  # (B,T,12)

    #     pitch_logits = pitch_logits.mean(1)  # (B,12)

    #     pitch_prob = torch.sigmoid(
    #         pitch_logits
    #     )  # (B,12)

    #     pitch_pred = (
    #         pitch_prob > self.threshold
    #     ).float()
        


    #     # =====================================
    #     # chord template matching
    #     # =====================================

    #     chord_templates = {

    #         "maj": [0,4,7],
    #         "min": [0,3,7],
    #         "dim": [0,3,6],
    #         "aug": [0,4,8],
    #         "dom": [0,4,7,10],

    #     }

    #     root_names = [
    #         "C","C#","D","D#","E","F",
    #         "F#","G","G#","A","A#","B"
    #     ]

    #     results = []

    #     # for b in range(B):
    #     b = 0
    #     pitch_cls = torch.where(
    #         pitch_pred[b]
    #     )[0]
        
    #     pred_vec = pitch_prob[b]  # (12,)

    #     best_score = -1e9
    #     best_root = 0
    #     best_chord = "N"

    #     # exhaustive search
    #     for root in range(12):

    #         for chord_name, intervals in chord_templates.items():

    #             template = torch.zeros(
    #                 12,
    #                 device=x.device
    #             )

    #             for i in intervals:
    #                 template[(root + i) % 12] = 1.0

    #             # score:
    #             # inside high
    #             # outside low

    #             score = (
    #                 pred_vec * template
    #             ).sum()

    #             score -= (
    #                 pred_vec * (1-template)
    #             ).sum() * 0.5

    #             score = score.item()

    #             if score > best_score:
    #                 best_score = score
    #                 best_root = root
    #                 best_chord = chord_name

    #     symbol = f"{root_names[best_root]}:{best_chord}"

    #     results = {

    #         "pitch_prob":
    #             pitch_prob[b].detach().cpu(),

    #         "pitch_cls_vec":
    #             pitch_pred[b].detach().cpu(),

    #         "pitch_cls":
    #             pitch_cls.detach().cpu(),

    #         "root_idx":
    #             best_root,

    #         "root_name":
    #             root_names[best_root],

    #         "chord_name":
    #             best_chord,

    #         "score":
    #             float(best_score),

    #         "symbol":
    #             symbol,
    #     }

    #     return results


# tiny model for compare
class Tiny(ChordClassifierMixin, nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.harmony_conv.backbone_output_dim
        self.D = D
        
        self.prior_affine = nn.Sequential(
            nn.Linear(cfg.input_dim, D),
            nn.GELU(),
            nn.Linear(D, 1) # 0~11, N
        )
        
        freqs = get_freqs(cfg.min_midi, cfg.max_midi)
        self.register_buffer("freqs", freqs)
        num_freqs = freqs.shape[0]
        self.num_freqs=num_freqs
        
        self.post_affine = nn.Sequential(
            nn.Linear(num_freqs, D),
            nn.GELU(),
            nn.Linear(D, D)
        )
        
        self.head_root = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 12) # 0~11, N
        )
        
        self.head_chord = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 6) # maj,min,dom,dim,aug,N
        )
        
        self.head_exist = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 1) # N, chord
        )
        
        self.loss_weight = cfg.harmony_conv.loss_weight
        self.threshold = cfg.harmony_conv.infer_threshold

    def forward(self, x):
        """
            x: (B, T, P, C)
            return: (B, T, D)
        """
        B, T, P, C = x.shape
        assert P==self.num_freqs
        x = self.prior_affine(x).squeeze(-1) # (B, T, P)
        x = self.post_affine(x) # (B, T, D)
        return x




from utils.pitchDist import build_euler_cost_matrix

class PitchDetr(nn.Module):
    """
    induction through 0.5 s time
    """
    def __init__(self, cfg):
        super().__init__()
        # preprocessor
        min_midi, max_midi = cfg.min_midi, cfg.max_midi
        freqs = get_freqs(min_midi, max_midi)
        self.preprocessor = MultiWindowCQT(freqs, cfg.sr, cfg.window_num, cfg.min_cycle,
                                           trainable=False, stride=cfg.cqt_stride)
        
        # backbone
        self.backbone = CQTEncoder(cfg)
        Q = cfg.pitchDetr.query_num
        D = cfg.harmony_conv.backbone_output_dim

        # detr
        self.querys = nn.Parameter(torch.randn(1, Q, D))
        self.num_layers = cfg.pitchDetr.num_layers
        self.inter_decoder_layers = nn.ModuleList([
            TFDecoderLayer(i, cfg.pitchDetr)
            for i in range(self.num_layers)
        ])
        
        P = cfg.max_midi - cfg.min_midi + 1
        
        # num_octave = math.ceil(num_pitch / 12)
        # octave + pitch_cls
        self.exist_head = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 1)
        )
        self.pitch_head = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, P)
        )
        
        self.pitch_head_object = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, 1)
        )
        
        harmony_prior = build_euler_cost_matrix(freqs.numpy()) # (P,P)
        self.register_buffer("harmony_prior", harmony_prior)
        # pitch_vec (B, P), harmony_prior (P, P), pitch_tar, (B, P) one hot
        # cost // loss (pitch_vec, pitch_tar) = pitch_vec @ harmony_prior @ pitch_tar.T

        self.pitch_pos_embedding = nn.Parameter(apply_freq_time_encoding(freqs, torch.tensor([1]), D)) # (1, P, D)

        self.loss_weights = cfg.pitchDetr.loss_weights
        self.cost_weights = cfg.pitchDetr.cost_weights

        self.Q = Q
        self.D = D
        self.P = P
        
        self.min_midi = min_midi
        
    def forward(self, audio, only_object=False):
        """
        audio: (B, T1, 2)
        
        only_object:
        忽略query, 只当作 transformer encoder 使用
        只有 modal_dict['object'] 有意义
        """
        cqt, _, _ = self.preprocessor(audio) # (B, T, P, 2*W)
        fea = self.backbone(cqt, retain_P=True) # (B, T, P, D)
        fea = fea.mean(1) # (B, P, D)
        B, P, D = fea.shape
        assert P==self.P
        
        fea = fea + self.pitch_pos_embedding # (B, P, D)
        query = self.querys.expand(B,-1,-1) # (B, Q, D)
        
        modal_dict = {
            "subject": query, # (B, Q, C)
            "object": fea, # (B, P, C)
        }
        
        for i in range(self.num_layers):
            modal_dict = self.inter_decoder_layers[i](modal_dict, only_object=only_object) # (B, ..., C2)
        
        if only_object:
            return modal_dict['object'] # (B, P, C)
        
        # query = modal_dict['subject'] # (B, Q, C)
        return modal_dict

    # def object_mode(self):
    #     # 设置为
    #     super(self, ChordClassifierMixin).__init__()
    #     self.pitch_head = self.pitch_head_object

    def get_pitch_cost(self, pitch_logits, pitch_tar):
        """
            pitch_prob: (Q, P) prob, after softmax
            pitch_tar: (N, P) one_hot
        """
        pitch_prob = torch.softmax(pitch_logits, dim=-1)
        harmony_prior = self.harmony_prior # (P, P)
        cost_matrix = pitch_tar @ harmony_prior @ pitch_prob.T
        return cost_matrix
        
    def get_exist_cost(self, exist_logits, N):
        """
            exist_logits: (Q, 1)
            N: int
        """
        exist_neg_log_prob = - F.logsigmoid(exist_logits) # (Q, 1)
        return exist_neg_log_prob.expand(-1,N).T # (N, Q)
    
    def get_sample_loss(self, x, target):
        """
        x: (Q, C)
        target: {
            "exist": (1,) 0~1
            "midi": (N,) int
            "midi_vec": (N, P) 0~1
        }
        """
        exist_logits = self.exist_head(x) # (Q, 1)
        
        exist_gt = target['exist']
        if not exist_gt.item():
            # 所有 query 预测 0
            exist_gt = exist_gt[None,:].expand(self.Q, -1)
            loss = F.binary_cross_entropy_with_logits(exist_logits, exist_gt.float())
            return loss
        
        pitch_logits = self.pitch_head(x) # (Q, P)
        pitch_tar = target['midi_vec'] # (N, P)
        pitch_cost = self.get_pitch_cost(pitch_logits, pitch_tar)
        
        N = pitch_tar.shape[0]
        exist_cost = self.get_exist_cost(exist_logits, N)
        
        total_cost = pitch_cost * self.cost_weights['pitch'] +\
                    exist_cost * self.cost_weights['exist']
            
        cost_np = total_cost.detach().cpu().numpy() # (N, Q)
        N_ids, Q_ids = linear_sum_assignment(cost_np)
        
        assert Q_ids.shape[0]==N, f"too much pitch num: {Q_ids.shape[0]}, with Query {self.Q}"
        
        Q_exist_target = torch.zeros((self.Q,1), device=x.device) # (Q,1)
        Q_exist_target[Q_ids] = 1
        
        loss_exist = F.binary_cross_entropy_with_logits(exist_logits, Q_exist_target)
        
        # loss_pitch = pitch_cost[N_ids, Q_ids].mean() # 这样确保 query n 以不和谐度的路径靠近 pitch
        pitch_logits_select = pitch_logits[Q_ids, :] # (M, P)
        pitch_tar_select = pitch_tar[N_ids, :] # (M, P)
        pitch_tar_idx = pitch_tar_select.argmax(-1) # (M)
        loss_pitch = F.cross_entropy(pitch_logits_select, pitch_tar_idx)
        
        loss = self.loss_weights['exist'] * loss_exist +\
                self.loss_weights['pitch'] * loss_pitch
        return loss
    def get_loss(self, x, targets):
        """
        targets: List Dict
        x: (B, Q, C)
        """
        B, Q, C = x.shape
        total_loss = 0
        for b in range(B):
            loss = self.get_sample_loss(x[b,...], targets[b])
            total_loss += loss
        total_loss /= B
        return total_loss

    @torch.no_grad()
    def infer(self, x, threshold=0.5):
        """
        (1, Q, C)
        return: Dict {
            "midi_idx": (N,)
        }
        """
        assert x.shape[0]==1
        
        exist_logits = self.exist_head(x)
        exist_prob = torch.sigmoid(exist_logits) # (1, Q, 1)
        exist = exist_prob > threshold # (1, Q, 1)
        
        pitch_logits = self.pitch_head(x) # (1, Q, P)
        pitch_idx = torch.softmax(pitch_logits, -1) # (1, Q, P)
        pitch_idx = pitch_logits.argmax(-1) # (1, Q)
        
        exist = exist[0,:,0] # (Q)
        pitch_idx = pitch_idx[0,:]
        pitch_exist = pitch_idx[exist] # (M)
        result = {
            "midi": pitch_exist.detach().cpu() + self.min_midi,
            "exist_prob": exist_prob.detach().cpu(),
            "pitch_prob": pitch_idx.detach().cpu(),
        }
        return result

    def load_pretrain(self, checkpoint_path):
        current_state = self.backbone.state_dict()
        state_dict = torch.load(checkpoint_path)

        # 过滤不匹配的参数
        filtered_state = {}
        for k, v in state_dict.items():
            if k not in current_state:
                continue
            if v.shape != current_state[k].shape:
                print("skip shape mismatch:", k)
                continue
            filtered_state[k] = v
        # 装载
        self.backbone.load_state_dict(state_dict=filtered_state, strict=False)
        print(f"成功加载预训练权重: {checkpoint_path}")



from spec import wav2cqt_2C, wav2spec_2C
from configs.cell_cls import CellCls
class PitchTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()        
        self.d_model_list = cfg.detr_d_model_list
        self.dim_up = cfg.ffn_dim_up
        
        self.pitch_num = cfg.pitch_vocab_size
        
        input_dim = cfg.input_dim

        self.num_layers = cfg.detr_num_decoder_layers
        self.inter_decoder_layers = nn.ModuleList([
            TFDecoderLayer(i)
            for i in range(self.num_layers)
        ])
        
        self.inner_decoder_layers = nn.ModuleList([
            Qwen2DecoderLayer(i)
            for i in range(self.num_layers)
        ])
        
        # query(cell) setting
        self.cells = Cells(cfg.cell_structure, self.d_model_list[0], self.d_model_list[-1])
        
        self.cost_weight = cfg.detr2_cost_weight
        self.loss_weight = cfg.detr2_loss_weight
        
        self.infer_threshold = cfg.infer_threshold
        min_midi, max_midi = cfg.min_midi, cfg.max_midi
        
        freqs = get_freqs(min_midi, max_midi)
        self.preprocessor = MultiWindowCQT(freqs, cfg.sr, cfg.window_num, cfg.min_cycle,
                                           trainable=False, stride=cfg.cqt_stride)
        self.register_buffer("freqs", freqs)
        self.backbone = CQTEncoder(cfg)
        assert cfg.backbone_output_dim == self.d_model_list[0], "backbone输出维度必须等于decoder输入维度"

        self.pretrain_head = nn.Linear(cfg.backbone_output_dim, 5 + 1) # maj,min,dom,dim,arg + None

    def load_pretrain(self, checkpoint_path):
        current_state = self.backbone.state_dict()
        state_dict = torch.load(checkpoint_path)

        # 过滤不匹配的参数
        filtered_state = {}
        for k, v in state_dict.items():
            if k not in current_state:
                continue
            if v.shape != current_state[k].shape:
                print("skip shape mismatch:", k)
                continue
            filtered_state[k] = v
        # 装载
        self.backbone.load_state_dict(state_dict=filtered_state, strict=False)
        print(f"成功加载预训练权重: {checkpoint_path}")
       
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("已冻结backbone参数")
        
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("已解冻backbone参数")
        
    def forward(self,
                audio,
                only_pretrain=False):
        """
        audio: (B, T, 2)
        """
        pitch_spec, times = self.preprocessor(audio)
        """
        inputs
            pitch_spec: (B, T, P, C)
        return: 
            output: (B, Q, C)
        """
        B, T, P, C = pitch_spec.shape

        pitch_embedding = self.backbone(pitch_spec) # (B, T, C)
        if only_pretrain:
            return pitch_embedding, times

        cell_state = self.cells.build_state(B)
        cell_inter_state = self.cells.get_flatten_inter(cell_state) # (B, L_inter_all, C)
        
        modal_dict = {
            "subject": cell_inter_state, # (B, Q, C)
            "object": pitch_embedding, # (B, T, C)
        }
        
        for i in range(self.num_layers):
            modal_dict = self.inter_decoder_layers[i](modal_dict) # (B, ..., C2)
            # 此时 inter 已经融合了信息
            new_inter = modal_dict['subject'] # (B, Q, C1)
            self.cells.update_inter(new_inter, cell_state) # 更新 cell_state
            
            cell_state = self.cells.inner_decode(
                self.inner_decoder_layers[i],
                cell_state,
            )
            
            modal_dict['subject'] = self.cells.get_flatten_inter(cell_state) # (B, Q, C2)

        output_list = self.cells.extract_output(cell_state)
        
        """
        List B [
            {cls_name:
                fea_name: (Q, C_fea)
            }
        ]
        """
        
        return output_list

    def infer(self, output_dict_dict):
        """
        output: Dict cls_name Dict token_name (N, dim)
        """
        threshold = self.infer_threshold
        result = {}

        for cls_name, output_dict in output_dict_dict.items():
            if cls_name in CellCls.not_need_match_cls:
                result[cls_name] = self._infer_no_match(output_dict, threshold)
            else:
                result[cls_name] = self._infer_match(output_dict, threshold)

        return result

    def _infer_no_match(self, output_dict, threshold):
        assert output_dict['exist'].numel() == 1

        exist_prob = torch.sigmoid(output_dict['exist'][0, 0])
        result = {}

        if exist_prob > threshold:
            result.update(output_dict)
            result['exist'] = exist_prob

            if "sustain" in result:
                result['sustain'] = torch.exp(result['sustain']) * CellCls.sustain_ref
            
            if "root" in result:
                result['root'] = torch.argmax(result['root'], dim=-1)
            if "tonic" in result:
                result['tonic'] = torch.argmax(result['tonic'], dim=-1)
            if "chord" in result:
                result['chord'] = (result['chord'] > 0).float()

        else:
            result['exist'] = exist_prob

        return result
    
    def _infer_match(self, output_dict, threshold):
        exist_prob = torch.sigmoid(output_dict['exist'][:, 0])
        choice = exist_prob > threshold
        result = {}

        for token_name, output in output_dict.items():
            result[token_name] = self._process_token(token_name, output, choice)

        if "root" in result:
            result['root'] = torch.argmax(result['root'], dim=-1)
        if "tonic" in result:
            result['tonic'] = torch.argmax(result['tonic'], dim=-1)
        if "chord" in result:
            result['chord'] = (result['chord'] > 0).float()

        result["exist"] = exist_prob[choice]

        return result
            
    def _process_token(self, token_name, output, choice):
        if token_name == "sustain":
            return torch.exp(output[choice, :]) * CellCls.sustain_ref

        return output[choice, :]
        
        
    def get_loss(self, outputs, targets):
        loss = 0
        loss_dict = {}
        B = len(outputs)
        for b in range(B):
            temp_loss = self.cells.get_sample_loss(outputs[b],
                                                   targets[b],
                                                   self.cost_weight,
                                                   self.loss_weight)
            loss += temp_loss
        loss /= B
        return loss