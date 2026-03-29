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


class PitchPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = get_config()
        
        pitch_num = cfg.pitch_vocab_size
        
        
    def forward(self,
                pitch_spec,
                pitchs,
                spec,
                freqs,
                text_embedding):
        """
        inputs
            pitch_spec: (B, T, P)
        return: 
            pitch_bar: (B, T, P, 3)
        """
        


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
        
#         self.pitch_embed = nn.Linear(cfg.pitch_vocab_size, cfg.d_model)
#         self.text_embed = nn.Linear(cfg.text_input_dim, cfg.d_model)
#         self.audio_embed = nn.Linear(cfg.audio_input_dim, cfg.d_model)
        
        