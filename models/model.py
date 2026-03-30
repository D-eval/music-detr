import torch
from configs.config import get_config
from torch import nn
import math
import torch.nn.functional as F
from typing import Callable, Optional, Union


class Qwen2MLP(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = get_config()
        self.hidden_size = cfg.d_model
        self.intermediate_size = cfg.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU

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
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights_before_softmax = attn_weights.clone().detach()
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights_before_softmax



class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        cfg = get_config()
        self.head_dim = getattr(cfg, "head_dim", cfg.d_model // cfg.n_attn_heads)
        self.num_key_value_groups = cfg.n_attn_heads // cfg.n_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = cfg.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_attn_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(cfg.n_attn_heads * self.head_dim, cfg.d_model, bias=False)
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
        attn_output, attn_weights = eager_attention_forward(
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
        return attn_output, attn_weights


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
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        cfg = get_config()
        hidden_size = cfg.d_model
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
        self.self_attn = Qwen2Attention(layer_idx=layer_idx)
        self.mlp = Qwen2MLP()
        self.input_layernorm = Qwen2RMSNorm()
        self.post_attention_layernorm = Qwen2RMSNorm()
    def forward(
        self,
        hidden_states: torch.Tensor,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[Cache] = None,
        # use_cache: Optional[bool] = False,
        # cache_position: Optional[torch.LongTensor] = None,
        # position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        # **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        
        residual = hidden_states
        # 我靠，原来attn前后都要加一个layernorm
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states_after_ffn = hidden_states.clone().detach()
        # Self Attention
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=None,
            # position_ids=position_ids,
            # past_key_values=past_key_values,
            # use_cache=use_cache,
            # cache_position=cache_position,
            # position_embeddings=position_embeddings,
            # **kwargs,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
                
        return hidden_states, attn_weights



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


class PitchTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = get_config()
        
        self.d_model = cfg.d_model
        
        pitch_num = cfg.pitch_vocab_size
        
        self.pitch_embed = nn.Linear(1, self.d_model)
        self.pitchless_embedding = nn.Parameter(torch.randn((self.d_model)))
        if cfg.use_same_pitch_freq:
            self.freq_embed = self.pitch_embed
        else:
            self.freq_embed = nn.Linear(1, self.d_model)
        
        self.distinguish_pitch_freq = cfg.distinguish_pitch_freq
        if self.distinguish_pitch_freq:
            self.pitch_token_embedding = nn.Parameter(torch.randn((self.d_model)))
            self.freq_token_embedding = nn.Parameter(torch.randn((self.d_model)))
        
        self.use_abs_pos_encoding = cfg.use_abs_pos_encoding
        if self.use_abs_pos_encoding:
            self.freq_time_encoding = apply_freq_time_encoding
        
        self.text_embed = nn.Linear(cfg.text_input_dim, cfg.d_model)
        self.audio_embed = nn.Linear(cfg.audio_input_dim, cfg.d_model)
        
        self.decoder_layers = nn.ModuleList([
            Qwen2DecoderLayer(i)
            for i in range(cfg.num_decoder_layer)
        ])
        
        self.output_mode = cfg.output_mode
        output_dim = cfg.output_dim_dict[cfg.output_mode]
        self.cls_head = nn.Linear(self.d_model, output_dim)
        
    def forward(self,
                pitch_spec,
                pitchs,
                pitch_centre,
                freq_spec,
                freqs,
                freq_centre,
                text_emb):
        """
        inputs
            pitch_spec: (N, T, P)
            freq_spec: (N, T, F)
            text_emb: (N, L, C)
        return: 
            output: (N, T, P+1, cls)
        """
        pitch_size = pitch_spec.shape[1:]
        freq_size = freq_spec.shape[1:]
        
        pitch_output_size = list(pitch_size)
        pitch_output_size[1] += 1 # 加一个 pitchless
        
        pitch_len = pitch_output_size[0] * pitch_output_size[1]
        freq_len = freq_size[0] * freq_size[1]
        
        N, L, _ = text_emb.shape
        assert pitch_spec.shape[0]==freq_spec.shape[0]==N, "N不一样"
        
        pitch_embedding = self.pitch_embed(pitch_spec.unsqueeze(-1))
        freq_embedding = self.freq_embed(freq_spec.unsqueeze(-1))

        if self.use_abs_pos_encoding:
            pitch_pos_encoding = self.freq_time_encoding(pitchs, pitch_centre, self.d_model)
            pitch_embedding = pitch_embedding + pitch_pos_encoding[None,...] # (N, Tf, F, C)

            freq_pos_encoding = self.freq_time_encoding(freqs, freq_centre, self.d_model)
            freq_embedding = freq_embedding + freq_pos_encoding[None,...]
        
        
        pitchless = self.pitchless_embedding[None, None, None, :].expand(N, pitch_size[0], 1, self.d_model)
        pitch_embedding = torch.concat([pitch_embedding, 
                                        pitchless],
                                       dim=2) # (N, Tp, P, C)
        
        if self.distinguish_pitch_freq:
            pitch_embedding = pitch_embedding + self.pitch_token_embedding[None,None,None,:]
            freq_embedding = freq_embedding + self.freq_token_embedding[None,None,None,:]
        
        text_embedding = self.text_embed(text_emb) # (N, L, C)
        
        hidden_state = torch.concat([
            text_embedding,
            torch.flatten(pitch_embedding, 1,2),
            torch.flatten(freq_embedding, 1,2)
        ], dim=1) # (N, L+Tf*F+Tp*P, C)
        
        for layer in self.decoder_layers:
            hidden_state, _ = layer(hidden_state)
        
        # hidden_text = hidden_state[:, :L, :]
        
        hidden_pitch_freq = hidden_state[:, L:, :]
        
        hidden_pitch = hidden_pitch_freq[:, :pitch_len, :]
        # hidden_freq = hidden_pitch_freq[:, pitch_len:pitch_len+freq_len, :]
        
        hidden_pitch = hidden_pitch.reshape((N, pitch_output_size[0], pitch_output_size[1], -1)) # (N, T, P+1, C)
  
        output = self.cls_head(hidden_pitch)
        return output


    def get_loss(self, output, target):
        """
            output: (N, T, P+1, cls)
            target: (N, T, P+1, 2) TriggerBool_ConditionalSustain
        """
        if self.output_mode == "TriggerBool_ConditionalSustain":
            assert output.shape[-1] == 2
            
            weights = torch.tensor([0.1, 1.0], device=output.device)

            output = torch.flatten(output, 1, 2) # (N, All, 2)
            target = torch.flatten(target, 1, 2).long() # (N, All, 2)
            loss_start = F.cross_entropy(output[:, :, 0], target[:, :, 0], weight=weights)
            
            target_sustain = torch.log(target[:, :, 1] + 1e-3)
            loss_sustain = F.smooth_l1_loss(output[:, :, 1]*target[:, :, 0], target_sustain)
            
            return loss_start + loss_sustain
        else:
            raise NotImplementedError("非常抱歉")


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
        

        
        