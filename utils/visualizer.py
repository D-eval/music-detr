import matplotlib.pyplot as plt
import os
from configs.config import get_config
import torch

def show_attn_alpha(pos_encoding, num_time=1, num_freq=1):
    cfg = get_config()
    for time_idx in range(num_time):
        freqs_matrix = pos_encoding[time_idx, :, :] @ pos_encoding[time_idx, :, :].T
        plt.figure(figsize=(6, 5))
        plt.imshow(freqs_matrix.cpu().numpy(), aspect='auto')
        plt.colorbar()
        plt.title(f"inner product={time_idx}")
        plt.xlabel("j")
        plt.ylabel("i")
        plt.savefig(os.path.join(cfg.save_dir, f"freq_time_{time_idx}.pdf"))
        plt.close()

    for freq_idx in range(num_freq):
        times_matrix = pos_encoding[:, freq_idx, :] @ pos_encoding[:, freq_idx, :].T
        plt.figure(figsize=(6, 5))
        plt.imshow(times_matrix.cpu().numpy(), aspect='auto')
        plt.colorbar()
        plt.title(f"inner product={time_idx}")
        plt.xlabel("j")
        plt.ylabel("i")
        plt.savefig(os.path.join(cfg.save_dir, f"time_freq{freq_idx}.pdf"))
        plt.close()



def compare_result(onset_logits, onset_gt):
    # onset_logits onset_gt: (T, P)
    cfg = get_config()
    
    pred = onset_logits
    gt = onset_gt

    plt.figure(figsize=(12,5))

    # 预测
    plt.subplot(1,2,1)
    plt.imshow(pred.T, aspect='auto', origin='lower')
    plt.title("Prediction (sigmoid)")
    plt.colorbar()

    # GT
    plt.subplot(1,2,2)
    plt.imshow(gt.T, aspect='auto', origin='lower')
    plt.title("Ground Truth")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir,"compare.pdf"))
