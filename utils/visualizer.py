import matplotlib.pyplot as plt
import os
from configs.config import get_config

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
