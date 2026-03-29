
import torch

def midi2freq(midi):
    """
    midi: int or np.array
    return: frequency (Hz)
    """
    # midi = torch.Tensor(midi)
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))

def freq2midi(freq):
    """
    freq: float or np.array
    return: midi (float, not rounded)
    """
    # freq = torch.Tensor(freq)
    return 69 + 12 * torch.log2(freq / 440.0)
