"""
纯粹 F 维度，弱跨 T
"""

import torch
from configs.config import get_config22 as get_config
from torch import nn
import math
import torch.nn.functional as F
from typing import Callable, Optional, Union, Dict
from spec.cqt import wav2cqt_2C
from spec.spec import wav2spec_2C



class Qwen2MLP(nn.Module):
    def __init__(self, intermediate_size, d_input, d_output):
        super().__init__()
        
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

def equip_cls(event, start, sustain, t):
    """
    event: (N) int cls_num
    start: (N) float
    sustain: (N) float
    t: (T) float
    """
    result = 12 * torch.ones_like(t, device=t.device).long()
    N = event.shape[0]
    for n in range(N):
        choice = (start[n] <= t) * (t <= sustain[n]) # (T,)
        result[choice] = event[n].long()
    return result

def equip_chord(chord, start, sustain, t):
    """
    chord: (N, 12) 0~1
    start: (N) float
    sustain: (N) float
    t: (T) float
    """
    C = chord.shape[-1]
    result = torch.zeros((t.shape[0], C), device=t.device).float()
    N = chord.shape[0]
    for n in range(N):
        choice = (start[n] <= t) * (t <= sustain[n]) # (T,)
        result[choice, :] = chord[n, :].float()
    return result


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


class ResidualCausalBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1, dropout=0.1, use_causal=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_causal = use_causal
        self.norm = Qwen2RMSNorm(dim)
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0  # ❗手动 pad
        )

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, C, T)
        """
        residual = x # (B, C, T)
        
        x = x.permute(0, 2, 1) # (B, T, C)
        x = self.norm(x) # (B, T, C)
        x = x.permute(0, 2, 1)   # (B, C, T)
        
        if self.use_causal:
            pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (pad, 0))  # causal pad
        else:
            pad = (self.kernel_size - 1) * self.dilation
            pad_left = pad // 2
            pad_right = pad - pad_left
            x = F.pad(x, (pad_left, pad_right))  # 对称 padding
            
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)

        return x + residual

class BTF(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = get_config()
        self.sr = cfg.sr
        self.num_pitch = cfg.pitch_vocab_size
        self.num_freq = cfg.num_freqs
        self.use_causal = cfg.conv.use_causal
        input_fea_size = 2 * (cfg.num_freqs + cfg.pitch_vocab_size)
        # 卷积，把 0.005 s 作为一个听感单元
        kernel_size = int(cfg.conv.sound_union_time * cfg.sr) # 220
        self.kernel_size = kernel_size
        self.embedding = nn.Conv1d(input_fea_size,
                              cfg.conv.embedding_dim,
                              kernel_size=kernel_size,
                              padding=0)

        layers = []
        for i in range(cfg.conv.num_layers):
            dilation = 3*(i + 1)
            layers.append(
                ResidualCausalBlock(
                    cfg.conv.embedding_dim,
                    kernel_size=10,
                    dilation=dilation,
                    dropout=0.1,
                    use_causal=self.use_causal
                )
            )
        self.layers = nn.ModuleList(layers)
        self.post_layer_norm = Qwen2RMSNorm(cfg.conv.embedding_dim)
        
        # self.head = Qwen2MLP(cfg.conv.intermediate_size, cfg.conv.embedding_dim, cfg.conv.output_dim)
        self.head = nn.Linear(cfg.conv.embedding_dim, cfg.conv.output_dim)
        self.N_weight = cfg.conv.N_weight
        self.loss_weight = cfg.conv.loss_weight
    def forward(self, audio):
        """
        audio: (B, T, 2)
        """
        B, T, pan = audio.shape
        assert pan == 2, "我要双声道"

        audio = 0.5 * torch.stack([
            audio[:, :, 0] + audio[:, :, 1],
            audio[:, :, 0] - audio[:, :, 1]
        ], dim=-1)  # (B, T, 2)
        
        pitch, t, _ = wav2cqt_2C(audio) # (B, T, P, 2)
        t = t / self.sr
        spec = wav2spec_2C(audio)[0] # (B, T, F, 2)
        
        pitch = torch.flatten(pitch, 2, 3)
        spec = torch.flatten(spec, 2, 3)
        
        x = torch.concat([pitch, spec], dim=-1) # (B, T, C)
        x = x.permute(0, 2, 1) # (B, C, T)
                
        if self.use_causal:
            pad = self.kernel_size - 1
            x = F.pad(x, (pad, 0))  # 左边 pad
            x = self.embedding(x) # (B, D, T)
        else:
            pad = self.kernel_size - 1
            pad_left = pad//2
            pad_right = pad - pad_left
            x = F.pad(x, (pad_left, pad_right))  # 左边 pad
            x = self.embedding(x) # (B, D, T)
            
        for layer in self.layers:
            x = layer(x) # (B, D, T)

        x = x.permute(0, 2, 1)
        x = self.post_layer_norm(x)
        x = x.permute(0, 2, 1)
        
        assert x.shape[-1]==t.shape[-1], f"x:{x.shape[-1]}, t:{t.shape[-1]}"
        x = x.permute(0, 2, 1) # (B, T, D)
        output = self.head(x) # (B, T, 38)
        return {"output":output,
                "time":t} # (B, T, 38)
    
    def get_loss(self, output, targets):
        """
        output: Dict{
            output: (B, T, 38)
            time: (T)
        }
        target: List [
            Dict{
            start: (N)
            sustain: (N)
            root: (N)
            chord: (N, 12)
            before: (N) bool
            }
        ] * B
        """
        t = output['time']
        
        root_logits = output['output'][:,:,:13] # softmax
        chord_logits = output['output'][:,:,13:25] # sigmoid
        tonic_logits = output['output'][:,:,25:] # softmax
        
        root_gts = []
        tonic_gts = []
        chord_gts = []
        for target in targets:
            root_gt = equip_cls(target['root'], target['start'], target['sustain'], t) # (T,) long, 0~12, 12表示N
            tonic_gt = equip_cls(target['tonic'], target['start'], target['sustain'], t) # (T,) long, 0~12, 12表示N
            chord_gt = equip_chord(target['chord'], target['start'], target['sustain'], t) # (T, 12) 0~1

            root_gts.append(root_gt)
            tonic_gts.append(tonic_gt)
            chord_gts.append(chord_gt)
            
        root_gt = torch.stack(root_gts, dim=0) # (B, T)
        tonic_gt = torch.stack(tonic_gts, dim=0) # (B, T)
        chord_gt = torch.stack(chord_gts, dim=0) # (B, T, 12)

        class_weight = torch.ones(13, device=t.device)
        class_weight[-1] = 0.2   # None 类权重 ↓↓↓（关键）

        root_loss = F.cross_entropy(
            root_logits.reshape(-1, 13),
            root_gt.reshape(-1),
            weight=class_weight,
            reduction='mean'
        )

        # ===== TONIC LOSS =====
        tonic_loss = F.cross_entropy(
            tonic_logits.reshape(-1, 13),
            tonic_gt.reshape(-1),
            weight=class_weight,
            reduction='mean'
        )
            
        # ===== CHORD LOSS =====
        # 👉 正样本加权（关键）
        pos_weight = torch.ones(12, device=t.device) * 3.0

        chord_loss = F.binary_cross_entropy_with_logits(
            chord_logits,
            chord_gt,
            reduction='mean',
            pos_weight=pos_weight
        )

        # ===== TOTAL =====
        loss = (
            self.loss_weight["root"] * root_loss +
            self.loss_weight["tonic"] * tonic_loss +
            self.loss_weight["chord"] * chord_loss
        )

        return loss
    
    def infer(self, output, threshold=0.5):
        """
        output: model forward 输出
        return: List[Dict] * B
        """

        x = output["output"]  # (B, T, 38)
        t = output["time"]    # (T,)

        root_logits = x[:, :, :13]
        chord_logits = x[:, :, 13:25]
        tonic_logits = x[:, :, 25:]

        # ===== frame-level =====
        root = torch.argmax(root_logits, dim=-1)        # (B, T)
        tonic = torch.argmax(tonic_logits, dim=-1)
        chord = torch.sigmoid(chord_logits)            # (B, T, 12)

        B, T = root.shape
        results = []

        for b in range(B):
            root_b = root[b].cpu().numpy()
            tonic_b = tonic[b].cpu().numpy()
            chord_b = chord[b].cpu().numpy()

            events = []

            # ===== segmentation =====
            start_idx = 0

            for i in range(1, T):
                # 判断是否变化（root 或 chord 变化）
                if (
                    root_b[i] != root_b[i-1] or
                    (chord_b[i] > threshold).astype(int).tolist()
                    != (chord_b[i-1] > threshold).astype(int).tolist()
                ):
                    # 生成 event
                    if root_b[start_idx] != 12:  # 忽略 None
                        t0 = t[start_idx].item()
                        t1 = t[i].item()

                        events.append({
                            "root": int(root_b[start_idx]),
                            "tonic": int(tonic_b[start_idx]),
                            "chord": (chord_b[start_idx] > threshold).astype(float),
                            "start": t0,
                            "sustain": t1 - t0,
                        })

                    start_idx = i

            # 最后一段
            if root_b[start_idx] != 12:
                t0 = t[start_idx].item()
                t1 = t[-1].item()

                events.append({
                    "root": int(root_b[start_idx]),
                    "tonic": int(tonic_b[start_idx]),
                    "chord": (chord_b[start_idx] > threshold).astype(float),
                    "start": t0,
                    "sustain": t1 - t0,
                })

            # 转 dict
            if len(events) > 0:
                result = {
                    "root": torch.tensor([e["root"] for e in events]),
                    "tonic": torch.tensor([e["tonic"] for e in events]),
                    "chord": torch.tensor([e["chord"] for e in events]),
                    "start": torch.tensor([e["start"] for e in events]),
                    "sustain": torch.tensor([e["sustain"] for e in events]),
                }
            else:
                result = {
                    "root": torch.empty(0),
                    "tonic": torch.empty(0),
                    "chord": torch.empty(0, 12),
                    "start": torch.empty(0),
                    "sustain": torch.empty(0),
                }

            results.append(result)

        return results