
from configs.config import get_config
import torch
import torchaudio
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from torch import nn


class MusicDetrTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.text_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
        self.text_encoder = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5")

        self.audio_tokenizer = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_encoder = AutoModel.from_pretrained("facebook/wav2vec2-base-960h")

        cfg = get_config()
        self.source_sr = cfg.sr
        self.target_sr = 16000
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=self.source_sr,
            new_freq=self.target_sr
        )
    def forward(self, audio, texts):
        """预处理

        Args:
            texts List[str]: _description_
            audio (B, T): _description_
        """
        if texts is None:
            text_emb = None
        else:
            # assert texts.__len__() == audio.shape[0], "batch 必须相同"
            text_emb = self.extract_text_encoding(texts)
        # audio = self.resampler(audio)
        # audio = self.normalize(audio)
        # audio_emb = self.extract_audio_encoding(audio)
        return None, text_emb
    def normalize(self, audio):
        """ 归一化
        audio: (B, T)
        """
        mu = torch.mean(audio, dim=1, keepdim=True)
        sigma = torch.std(audio, dim=1, keepdim=True)
        audio = (audio - mu) / (sigma + 1e-6)
        return audio
    def extract_text_encoding(self, texts):
        """
        texts: List[List[str]]
        """
        texts_emb = []
        for text in texts:
            inputs = self.text_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.text_encoder(**inputs)
            emb = outputs.last_hidden_state[:, 0]  # CLS # (B, L, D)
            emb = emb.detach()
            texts_emb.append(emb)
        return texts_emb
    def extract_audio_encoding(self, audio):
        """
        Args:
            audio (B, T)
        """
        # inputs = self.audio_tokenizer(audio, sampling_rate=self.target_sr, return_tensors="pt") # 归一化
        inputs = {"input_values":audio}
        with torch.no_grad():
            outputs = self.audio_encoder(**inputs)
        features = outputs.extract_features
        hidden_state = outputs.last_hidden_state
        # print(f"features:{features.shape}")
        return hidden_state
