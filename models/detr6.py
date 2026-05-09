"""
用原始的 maj, min, N, dom, dim, aug 和弦表示，适配主流数据集
而且 model 的 modal 输出要能预测和弦，作为pretrain
detr作为finetune
"""

import torch
from configs.config5 import get_config
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
        self.self_attn = Qwen2Attention(layer_idx=layer_idx)
        self.mlp = Qwen2MLP(layer_idx=layer_idx)
        self.input_layernorm = Qwen2RMSNorm(cfg.detr_d_model_list[layer_idx])
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
        pitch = modal_dict['object'] # (B, T, C)
        text = modal_dict['subject'] # (B, Q, C)
        
        B, T, C = pitch.shape
        _B, Q, _C = text.shape
        assert _C==C
        
        TaQ = T + Q
        
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
            cqt: (B, 1, P)
            return: (B, H, P)
        """
        kernel_size = self.kernels.shape[2]
        padding_len = kernel_size - 1
        
        kernels = self.kernels * self.envelopes # (harmony, kernel_size)

        x = F.pad(cqt, (0, padding_len))
        y = F.conv1d(x, kernels, padding=0) # (B*T, harmony, P)
        return y


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
        self.before_mlp_layernorm = Qwen2RMSNorm(D)
        
        self.mlp = MLP(D, D, D*4)
    def forward(self, x):
        """
        x: (B, T, P, D)
        """
        B, T, P, D = x.shape
        
        residual = x
        x = self.before_conv_layernorm(x)
        
        x = x.permute(0,1,3,2) # (B, T, D, P)
        x = x.flatten(0,2)
        x = x[:,None,:] # (B*T*D, 1, P)
        x = self.harmony_conv(x) # (B*T*D, H, P)
        x = x.reshape(B,T,D,-1,P) # (B, T, D, H, P)
        x = x.permute(0,1,4,2,3) # (B, T, P, D, H)
        x = x.flatten(-2,-1) # (B, T, P, D*H)
        x = self.mix_proj(x) # (B, T, P, D)
        x = x + residual # inductive bias by conv
        
        residual = x
        x = self.before_mlp_layernorm(x)
        x = self.mlp(x) # (B, T, P, D)
        x = x + residual # inductive bias by mlp
        return x


class CQTEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
                
        self.prior_affine = nn.Linear(cfg.input_dim, cfg.backbone_intermediate_dim)
        self.layers = nn.ModuleList([
            HarmonyBlock(
                D=cfg.backbone_intermediate_dim,
                cycles=cfg.harmony_conv.cycles,
                taus=cfg.harmony_conv.taus,
                kernel_size=cfg.harmony_conv.kernel_size
            )
            for _ in range(cfg.harmony_conv.num_layers)
        ])
        
    def forward(self, x):
        """
            x: (B, T, P, C)
            return: (B, T, D)
        """
        B, T, P, C = x.shape
        x = self.prior_affine(x) # (B, T, P, D)
        for layer in self.layers:
            x = layer(x) # (B, T, P, D)
        x = x.mean(dim=2) # (B, T, D)
        return x


from spec import wav2cqt_2C, wav2spec_2C
from configs.cell_cls import CellCls
class PitchTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = get_config()
        
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
        self.preprocessor = MultiWindowCQT(freqs, cfg.sr, cfg.window_num, cfg.min_cycle)
        self.register_buffer("freqs", freqs)
        
        self.backbone = CQTEncoder(cfg)
        
        assert cfg.backbone_output_dim == self.d_model_list[0], "backbone输出维度必须等于decoder输入维度"
        
        self.pretrain_head = nn.Linear(cfg.backbone_output_dim, 5 + 1) # maj,min,dom,dim,arg + None
        
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
    
    def pretrain_chord(self, audio, target):
        """
        target: Dict{
            start: (N), second
            root: (N), 0~11
            cls: (N), 0~4 maj,min,dom,dim,aug
            ...
        }
        """
        fea, times = self.forward(audio, only_pretrain=True)
        # fea: (B, T, C)
        # times: (T,)
        
        
    

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
