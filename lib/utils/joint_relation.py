"""Quan he token dung chung cho loss KD va script chan doan Teacher."""
import torch.nn.functional as F


def joint_relation_matrix(tokens):
    """(B,J,C) -> cosine(LN(token)) (B,J,J), FP32, khong cat gradient.
    Tra ca duong cheo; ben goi tu ap mask GT/bo duong cheo theo muc dich.
    """
    if tokens.ndim != 3:
        raise ValueError('joint_relation_matrix can token (B,J,C)')
    z = F.normalize(F.layer_norm(tokens.float(), tokens.shape[-1:]), dim=-1, eps=1e-6)
    return z @ z.transpose(1, 2)
