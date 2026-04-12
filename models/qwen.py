
from typing import Callable, Optional, Union

import torch
from torch import nn

from configs.config import get_config

import torch.nn.functional as F


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


class Qwen2MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        cfg = get_config()
        
        self.hidden_size = cfg.llm.hidden_size
        self.intermediate_size = cfg.llm.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, x):
        # 用gate, 可以表达条件计算
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def build_prefix_causal_mask(hidden_states, Lp):
    """
    hidden_states: (B, T, C)
    return: (B, 1, T, T)  # 适配 attention
    """
    device = hidden_states.device
    B, T, _ = hidden_states.shape
    L = T - Lp
    assert L > 0
    # ===== 1. 初始化全可见 =====
    mask = torch.ones((T, T), device=device)
    
    mask[:Lp, Lp:] = 0 # Lp 不能看到后面的
    # mask[Lp:, :Lp] = 1 # 后面的能看到 Lp
    
    # ===== 2. suffix 内部改为 causal =====
    causal = torch.tril(torch.ones((L, L), device=device))
    mask[Lp:, Lp:] = causal
    
    # ===== 3. reshape 成 attention 用的格式 =====
    mask = mask[None, None, :, :]  # (1,1,T,T)
    mask = mask.expand(B, -1, -1, -1)  # (B,1,T,T)

    return mask

class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, layer_idx: int):
        super().__init__()
        cfg = get_config()
        self.layer_idx = layer_idx
        self.attn_type = cfg.llm.attn_type
        self.head_dim = getattr(cfg.llm, "head_dim", cfg.llm.hidden_size // cfg.llm.num_attention_heads)
        self.num_key_value_groups = cfg.llm.num_attention_heads // cfg.llm.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = cfg.llm.attention_dropout
        self.is_causal = True
        
        self.external_modal_len = cfg.cell.num_prompt_tokens
        
        self.q_proj = nn.Linear(cfg.llm.hidden_size, cfg.llm.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.llm.hidden_size, cfg.llm.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.llm.hidden_size, cfg.llm.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(cfg.llm.num_attention_heads * self.head_dim, cfg.llm.hidden_size, bias=False)
    def forward(
        self,
        hidden_states: torch.Tensor, # (B, L, C)
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        Lp = self.external_modal_len
        
        input_shape = hidden_states.shape[:-1] # (B, L)
        hidden_shape = (*input_shape, -1, self.head_dim) # (B, L, -1, h)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (B, H, L, h)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_audio = query_states[:, :, :Lp, :]
        key_audio = key_states[:, :, :Lp, :]
        
        query_to_rotate = query_states[:, :, Lp:, :]
        key_to_rotate = key_states[:, :, Lp:, :]

        cos, sin = position_embeddings
        query_to_rotate, key_to_rotate = apply_rotary_pos_emb(query_to_rotate, key_to_rotate, cos, sin)

        query_states = torch.concat([query_audio, query_to_rotate], dim=2)
        key_states = torch.concat([key_audio, key_to_rotate], dim=2)

        attention_interface = AttentionType[self.attn_type]

        # 前 Lp 位置只能互相看到
        # 后 L 位置是因果的
        attention_mask = build_prefix_causal_mask(hidden_states, Lp)

        attn_output = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen2RMSNorm(nn.Module):
    def __init__(self) -> None:
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        cfg = get_config()
        hidden_size = cfg.llm.hidden_size
        eps = cfg.llm.rms_norm_eps
        
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 平方根倒数 rsqrt
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"



class Qwen2DecoderLayer(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        cfg = get_config()
        self.hidden_size = cfg.llm.hidden_size

        self.self_attn = Qwen2Attention(layer_idx=layer_idx)

        self.mlp = Qwen2MLP()
        self.input_layernorm = Qwen2RMSNorm()
        self.post_attention_layernorm = Qwen2RMSNorm()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states



class Qwen2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, device=None):
        super().__init__()
        cfg = get_config()
        # BC: "rope_type" was originally "type"
        
        dim = cfg.llm.head_dim
        base = cfg.llm.rope_base
        
        # 低维高频，i=0 freq=1, 高维低频，i=dim freq=1/base
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device).float() / dim)
        )
        
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        # position_ids: (B, T)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen2Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        cfg = get_config()
        self.num_hidden_layers = cfg.llm.num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = cfg.llm.hidden_size
        self.padding_idx = cfg.llm.padding_idx
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(layer_idx) for layer_idx in range(self.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm()
        self.rotary_emb = Qwen2RotaryEmbedding()
        self.external_modal_len = cfg.cell.num_prompt_tokens

    def forward(
        self,
        prompt_emb: torch.Tensor, # (B, Lp, C)
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        assert prompt_emb.shape[1]==self.external_modal_len
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = torch.concat([prompt_emb, inputs_embeds], dim=1)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        for decoder_layer in self.layers[: self.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        cfg = get_config()
        self.model = Qwen2Model(vocab_size)
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(cfg.llm.hidden_size, self.vocab_size, bias=False)

        self.ignore_index = cfg.llm.ignore_index

    def forward(
        self,
        prompt_emb: torch.Tensor, # (B, Lp, C)
        input_ids: Optional[torch.LongTensor] = None, # (B, L)
        labels: Optional[torch.LongTensor] = None,
        # logits_to_keep: Union[int, torch.Tensor] = 0, # 训练时逐渐加长序列
    ):
        """
            label: (B, T_select)
        """
        B = prompt_emb.shape[0]
        L = input_ids.shape[1] + 1
        
        position_ids = torch.arange(L-1, dtype=torch.long, device=prompt_emb.device)[None, :].expand(B, -1) # (Lp,)
        L = input_ids.shape[1]
        
        hidden_states = self.model(
            prompt_emb=prompt_emb,
            input_ids=input_ids, # (B, L)
            position_ids=position_ids,
        ) # (B, T, C)
        
        logits_to_keep = L
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        # logits: (B, T_select, V)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size)

        return {
            "loss":loss,
            "logits":logits,
            "hidden_states":hidden_states,
        }

    def loss_function(self, logits, labels, vocab_size):
        # logits: (B, T, V)
        # labels: (B, T) long

        logits = logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1)

        return F.cross_entropy(logits, labels, ignore_index=self.ignore_index)
