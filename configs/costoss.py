import torch
import torch.nn.functional as F
import math

cls_token_cost_dict = {}
cls_token_cost_dict["chord"] = {}
cls_token_cost_dict["beat"] = {}
not_need_match_cls = ["chord_before", "metronome"]
not_need_match_cls_token_loss_dict = {
    "chord_before": {
        "sustain": lambda out, tar: (((torch.log(tar + 1e-6) - math.log(sustain_ref))) - out) ** 2,
        "exist": lambda out, tar: F.binary_cross_entropy_with_logits(out, tar, reduction="none"),
        "root": lambda out, tar: F.cross_entropy(out, tar),
        "chord": lambda out, tar: F.binary_cross_entropy_with_logits(out, tar, reduction="none").mean(),
        "tonic": lambda out, tar: F.cross_entropy(out, tar),
    },
    "metronome": {
        "exist": lambda out, tar: F.binary_cross_entropy_with_logits(out, tar, reduction="none"),
        "bpm": lambda out, tar: F.l1_loss(out, tar),
        "offset": lambda out, tar: F.l1_loss(out, tar),
        "is_4beat": lambda out, tar: F.binary_cross_entropy_with_logits(out, tar, reduction="none"),
    }
}

def exist_loss(out, gt_idx):
    """
    out: (Q,)
    """
    tar = torch.zeros_like(out, device=out.device)
    tar[gt_idx] = 1.0
    loss = F.binary_cross_entropy_with_logits(
        out,
        tar,
        reduction="mean"
    )
    return loss


# 匹配后需要重新计算loss，而不是使用cost
posterior_cls_token_loss_dict = {
    "chord": {"exist": exist_loss},
    "beat": {"exist": exist_loss}
}

sustain_ref = 0.1
def anchor_cost(out, tar):
    """
    (Q or N, 2)
    """
    Q = out.shape[0]
    N = tar.shape[0]
    tar[:, 1] = (torch.log(tar[:, 1] + 1e-6) - math.log(sustain_ref))
    diff = out[None, :, :] - tar[:, None, :]
    cost = (diff**2).sum(dim=-1)
    return cost
cls_token_cost_dict["chord"]["anchor"] = anchor_cost


def exist_cost(out):
    """
    (Q,)
    """
    cost = F.binary_cross_entropy_with_logits(
        out,
        torch.ones_like(out, device=out.device),
        reduction="none"
    )
    return cost[None, :]
cls_token_cost_dict["chord"]["exist"] = exist_cost


def root_cost(out, tar):
    """
    out: (Q, 13)
    tar: (N,) int
    """
    log_prob_root = F.log_softmax(out, dim=-1) # (Q, 13)
    cost_root = -log_prob_root[:, tar].T # (N, Q)
    return cost_root
cls_token_cost_dict["chord"]["root"] = root_cost
cls_token_cost_dict["chord"]["tonic"] = root_cost

def chord_cost(out, tar):
    """
    out: (Q, 12)
    tar: (N, 12) 0~1
    """
    chord_logits_exp = out[None, :, :]  # (1, Q, 12)
    chord_gt = tar.float()
    chord_gt_exp = chord_gt[:, None, :]          # (N, 1, 12)

    Qe = out.shape[0]
    Ne = tar.shape[0]

    # BCE cost
    cost_chord = F.binary_cross_entropy_with_logits(
        chord_logits_exp.expand(Ne, Qe, 12),
        chord_gt_exp.expand(Ne, Qe, 12),
        reduction='none'
    ).mean(dim=-1)  # (N, Q)
    return cost_chord
cls_token_cost_dict["chord"]["chord"] = chord_cost


def beat_cost(out, tar):
    """
    out: (Q,)
    tar: (N,)
    """
    diff = out[None,:] - tar[:,None]
    cost = diff**2
    return cost
cls_token_cost_dict["beat"]["beat"] = beat_cost

def is_downbeat_cost(out, tar):
    """
    out: (Q,)
    tar: (N,) {0,1}
    """
    tar = tar.float()
    
    Qe = out.shape[0]
    Ne = tar.shape[0]
    
    out = out[None, :].expand(Ne, Qe)
    tar = tar[:, None].expand(Ne, Qe)
    
    cost = F.binary_cross_entropy_with_logits(
        out,
        tar,
        reduction="none"
    )
    
    return cost
cls_token_cost_dict["beat"]["is_downbeat"] = is_downbeat_cost



def before_sustain_loss(out, tar):
    """
    (1,)
    """
    tar = (torch.log(tar[:, 1] + 1e-6) - math.log(sustain_ref))
    diff = out - tar
    loss = (diff**2)
    return loss
