"""
detr-lan 联合
"""

import torch
from configs.config import get_config
from torch import nn
import math
import torch.nn.functional as F
from typing import Callable, Optional, Union, Dict

from .detr2 import PitchTransformer

from .qwen import Qwen2ForCausalLM
from spec import wav2cqt, wav2spec

from .tokenizer import ALTokenizer

"""
联合模型
同时训练llm的文本描述和detr的检测
"""

class ALUnion(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = get_config()
        
        self.detr = PitchTransformer()
        
        self.lm_tokenizer = ALTokenizer()
        self.lm = Qwen2ForCausalLM(vocab_size=self.lm_tokenizer.tokenizer.get_vocab_size())
        
        self.loss_weights = cfg.union_loss_weights
        
    def get_loss(self,
                audio, # (B, T)
                targets, # List[Dict] B
                ):
        """
            audio: (B, T)
        """
        detr_outputs = self.detr_forward(audio)
        loss_detr = self.detr.get_loss(detr_outputs, targets)
        
        prompts_emb = [output['text_prompt'] for output in detr_outputs] # List[ (Qt, num_prompt, C) ]
    
        all_prompts = []
        all_sentences = []
        
        for b in range(len(prompts_emb)):
            detr_output = detr_outputs[b]
            text_distillation = detr_output["text_distillation"]
            # text_exist = text_distillation[:, -1]
            text_pred = text_distillation[:, :-1] # (Q, C)
            text_gt = targets[b]['text_emb'] # (N, C)
            gt_idxs, pred_idxs = self.detr.match_text(text_pred, text_gt)
            for i in range(len(gt_idxs)):
                gt_idx = gt_idxs[i]
                pred_idx = pred_idxs[i]
                sentence_gt = targets[b]['text'][gt_idx] # str
                sentence_ids = self.lm_tokenizer.encode_and_pad(sentence_gt) # (L,) long
                all_sentences.append(torch.tensor(sentence_ids, dtype=torch.long, device=text_pred.device))
                prompt_pred = detr_output['text_prompt'][pred_idx] # (num_prompt, C)
                all_prompts.append(prompt_pred)
                
        all_prompts = torch.stack(all_prompts, dim=0) # (M, Lp, C)
        all_sentences = torch.stack(all_sentences, dim=0) # (M, L) long
        
        lm_outputs = self.lm_forward(all_prompts, all_sentences)
        
        loss_lm = lm_outputs['loss']
        
        loss = self.loss_weights['detr'] * loss_detr +\
            self.loss_weights['lm'] * loss_lm
        
        return loss

    def lm_forward(self, prompts, sentences):
        """
            prompts: (M, Lp, C)
            sentences: (M, L) long
        """
        B = prompts.shape[0]
        L = sentences.shape[1]
        Lp = prompts.shape[1]
        
        text_ids = sentences # (M, L)
        
        position_ids = torch.arange(L-1, dtype=torch.long, device=prompts.device)[None, :].expand(B, -1) # (Lp,)
        
        input_ids = text_ids[:,:-1]
        labels = text_ids[:,1:]
        
        output = self.lm(
            prompt_emb=prompts,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
        )
        
        return output        
        
    def detr_forward(self, audio):
        """
            audio: (B, T)
        """
        pitch_spec, pitch_centre, pitchs = wav2cqt(audio)
        freq_spec, freq_centre, freqs = wav2spec(audio)
        input_dict = {
            "pitch_spec": pitch_spec, # (B, T, F)
            "pitchs": pitchs,
            "pitch_centre": pitch_centre,
            "freq_spec": freq_spec, # (B, T, F)
            "freqs": freqs,
            "freq_centre": freq_centre,
        }
        detr_output = self.detr(**input_dict)
        return detr_output
    