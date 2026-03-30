import torch
from configs.config import get_config
from torch import nn
import math

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
    

class PitchAttention(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = get_config()
        
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_attn_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

        self.pitch_embedding = nn.Embedding(cfg.pitch_vocab_size, self.head_dim)

    def repeat_kv(self, x):
        """
        x: (B, T, P, H_kv, d)
        → (B, T, P, H, d)
        """
        B, T, P, H_kv, d = x.shape
        repeat = self.n_heads // H_kv
        x = x.unsqueeze(3).repeat(1, 1, 1, repeat, 1, 1)
        return x.view(B, T, P, self.n_heads, d)

    def forward(self, hidden_states):
        """
        hidden_states: (B, T, P, C)
        """
        B, T, P, _ = hidden_states.shape

        # ===== QKV =====
        q = self.q_proj(hidden_states).view(B, T, P, self.n_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, P, self.n_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, P, self.n_kv_heads, self.head_dim)

        # ===== GQA =====
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        # ===== 加 pitch embedding（关键）=====
        pitch_ids = torch.arange(P, device=hidden_states.device)
        pitch_emb = self.pitch_embedding(pitch_ids)  # (P, d)

        q = q + pitch_emb.view(1, 1, P, 1, self.head_dim)
        k = k + pitch_emb.view(1, 1, P, 1, self.head_dim)

        # ===== reshape for attention =====
        # (B, H, T*P, d)
        q = q.permute(0, 3, 1, 2, 4).reshape(B, self.n_heads, T * P, self.head_dim)
        k = k.permute(0, 3, 1, 2, 4).reshape(B, self.n_heads, T * P, self.head_dim)
        v = v.permute(0, 3, 1, 2, 4).reshape(B, self.n_heads, T * P, self.head_dim)

        # ===== attention =====
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, H, TP, d)

        # ===== reshape back =====
        out = out.view(B, self.n_heads, T, P, self.head_dim)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, T, P, -1)

        return self.o_proj(out)
        


class TimeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = get_config()

        self.d_model = cfg.d_model
        self.n_heads = cfg.n_attn_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

    # ===== GQA =====
    def repeat_kv(self, x):
        """
        x: (B, P, T, H_kv, d)
        → (B, P, T, H, d)
        """
        B, P, T, H_kv, d = x.shape
        repeat = self.n_heads // H_kv
        x = x.unsqueeze(3).repeat(1, 1, 1, repeat, 1, 1)
        return x.view(B, P, T, self.n_heads, d)

    # ===== 标准 RoPE（时间）=====
    def build_rope(self, T, device):
        d = self.head_dim
        assert d % 2 == 0

        half = d // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(half, device=device) / half))

        pos = torch.arange(T, device=device)
        angles = pos[:, None] * inv_freq[None, :]  # (T, d/2)

        return torch.cos(angles), torch.sin(angles)

    def apply_rope(self, x, cos, sin):
        """
        x: (B, P, T, H, d)
        cos/sin: (T, d/2)
        """
        B, P, T, H, d = x.shape
        half = d // 2

        x1 = x[..., :half]
        x2 = x[..., half:]

        cos = cos.view(1, 1, T, 1, half)
        sin = sin.view(1, 1, T, 1, half)

        return torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

    def forward(self, hidden_states):
        """
        hidden_states: (B, T, P, C)
        """
        B, T, P, _ = hidden_states.shape

        # ===== 转换维度：按 pitch 分组 =====
        # (B, P, T, C)
        x = hidden_states.permute(0, 2, 1, 3)

        # ===== QKV =====
        q = self.q_proj(x).view(B, P, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, P, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, P, T, self.n_kv_heads, self.head_dim)

        # ===== GQA =====
        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        # ===== RoPE（时间）=====
        cos_t, sin_t = self.build_rope(T, x.device)
        q = self.apply_rope(q, cos_t, sin_t)
        k = self.apply_rope(k, cos_t, sin_t)

        # ===== reshape for attention =====
        # (B, P, H, T, d)
        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        # ===== attention =====
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, P, H, T, d)

        # ===== reshape back =====
        out = out.permute(0, 1, 3, 2, 4).reshape(B, P, T, -1)

        # (B, T, P, C)
        out = out.permute(0, 2, 1, 3)

        return self.o_proj(out)


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
    half_arange = torch.arange(half) / half
    
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
    
    pos_encoding = torch.zeros((T, F, d_model))
    pos_encoding[:,:,::2] = cos
    pos_encoding[:,:,1::2] = sin
    
    return pos_encoding

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        pass

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
            DecoderLayer()
            for _ in range(cfg.num_decoder_layer)
        ])
        
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
            pitch_bar: (N, T, P+1, cls)
        """
        pitch_size = pitch_spec.shape[1:]
        freq_size = freq_spec.shape[1:]
        
        pitch_output_size = list(pitch_size)
        pitch_output_size[1] += 1 # 加一个 pitchless
        
        pitch_len = pitch_output_size[0] * pitch_output_size[1]
        freq_len = freq_size[0] * freq_size[1]
        
        N, L, _ = text_emb.shape
        assert pitch_spec.shape[0]==freq_spec.shape[0]==N
        
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
            hidden_state = layer(hidden_state)
        
        # hidden_text = hidden_state[:, :L, :]
        
        hidden_pitch_freq = hidden_state[:, L:, :]
        
        hidden_pitch = hidden_pitch_freq[:, :pitch_len, :]
        # hidden_freq = hidden_pitch_freq[:, pitch_len:pitch_len+freq_len, :]
        
        hidden_pitch = hidden_pitch.reshape((N, pitch_output_size[0], pitch_output_size[1], -1)) # (N, T, P+1, C)
  
        output = self.cls_head(hidden_pitch)
        return output

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
        

        
        