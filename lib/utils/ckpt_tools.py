"""Nap checkpoint Teacher / Student cho cac script danh gia / chan doan (dung chung)."""
import os
import sys

import torch

from core.config import cfg


def inspect_ckpt(mode, ckpt_path):
    """Doc checkpoint -> info {'kind': 'teacher' | 'student' | 'khong ro', 'has_probe': bool}.
    Chi ho tro kien truc hien tai (head goc xoay tung khop); checkpoint kien truc cu -> bao loi."""
    from core.base import _PickleShim
    if not os.path.isfile(ckpt_path):
        print(f"[LOI] Khong thay checkpoint {mode}: {ckpt_path}")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location='cpu', pickle_module=_PickleShim, weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in ckpt:
                state = ckpt[k]
                break
    keys = list(state.keys())
    if any('hint_encoder.' in k or 'hint_pred.' in k for k in keys):
        sys.exit(f'[LOI] {ckpt_path} la checkpoint T-hint / S-hint (da bo khoi code)')
    if not any('per_joint_head.' in k for k in keys):
        sys.exit(f'[LOI] {ckpt_path} la kien truc cu (VPoser), code hien tai chi ho tro kien truc v2')
    if any('fusion.projector_student.' in k for k in keys):
        kind = 'student'
    elif any('fusion.joint_proj.' in k for k in keys):
        kind = 'teacher'
    else:
        kind = 'khong ro'
    info = {'kind': kind, 'has_probe': any('probe_' in k for k in keys)}
    print(f"  [{mode}] {ckpt_path} -> loai: {kind}" + (" (co head tham do anh)" if info['has_probe'] else ""))
    if kind != 'khong ro' and kind != mode:
        sys.exit(f"[LOI] Checkpoint {kind.upper()} dat vao vi tri {mode.upper()}: {ckpt_path}")
    return info


def build_and_load(mode, ckpt_path, device='cuda'):
    """Tao ARTS(mode) theo cfg hien tai (Student: DiffLifter / gate tu YAML), nap checkpoint, dong bang.
    Teacher: img_probe lay theo checkpoint. Tra (model, info)."""
    from core.base import load_model_weights
    from models.ARTS import ARTS
    info = inspect_ckpt(mode, ckpt_path)
    saved = {k: cfg.MODEL[k] for k in ('name', 'img_probe')}
    cfg.MODEL.name = mode
    if mode == 'teacher':
        cfg.MODEL.img_probe = info['has_probe']
    model = ARTS(num_joint=17, embed_dim=cfg.MODEL.hpe_dim)
    cfg.MODEL.update(saved)
    model = load_model_weights(model.to(device), ckpt_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, info
