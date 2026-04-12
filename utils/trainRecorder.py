
import torch
from configs.config import get_config
import os

class TrainingRecorder:
    def __init__(self):
        self.history = {
            "loss": [],
            "lr": [],
        }
        cfg = get_config()
        
        save_dir = cfg.save_dir
        name = cfg.large_save_dir.split("/")[-1]
        self.path = os.path.join(save_dir, name+".pt")

    def update(self, loss, lr):
        self.history["loss"].append(loss)
        self.history["lr"].append(lr)

    def save(self):
        torch.save(self.history, self.path)

    def load(self):
        if os.path.exists(self.path):
            self.history = torch.load(self.path)
        else:
            print("没有训练记录")
    def latest(self):
        return {k: v[-1] for k, v in self.history.items()}