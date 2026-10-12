"""
content_ablation_test.py — Model co DUNG NOI DUNG ANH khong? (FuseKin, eval-only)
=================================================================================

Thay feature anh (dau ra backbone) truoc khi vao fusion bang:
  - blind: torch.randn_like  -> xoa sach noi dung (de soc out-of-distribution, chi tham khao)
  - mean : trung binh feature ca batch, ap cho moi mau -> van 'hop le' nhung mat thong tin RIENG tung mau
Con so chinh: (mean - that) = so mm bi te di khi mat thong tin anh rieng -> cang lon = cang dung anh.
Do ca MPJPE (17 khop H36M, root-rel) va MPVPE (mesh, can theo Pelvis H36M).

Ho tro:
  - Teacher GT cu (input = (1-a)*GT + a*MotionBERT, --teacher_alpha)
  - Teacher T-hint (PLAN_TEACHER_HINT.md): tu nhan dien tu checkpoint; do 2 dieu kien CO goi y / KHONG goi y
  - Student thuong va Student S-hint (tu nhan dien; S-hint chay fusion 2 luot nhu luc test)

Cach chay:
    python main/content_ablation_test.py --cfg <config> \
        [--teacher_ckpt <ckpt_teacher>]   # mac dinh: cfg.MODEL.TEACHER
        [--student_ckpt <ckpt_student>]   # bo trong -> chi test Teacher
        [--teacher_alpha 0 0.5 1.0]       # chi cho Teacher GT cu
        [--max_batches 200] [--batch_size 16]
Can CUDA.
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # cho Apple Accelerate/vecLib
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Content-ablation test (anh = randn / TB batch) cho Teacher/Student (FuseKin)')
parser.add_argument('--cfg', type=str, required=True, help='config yaml')
parser.add_argument('--student_ckpt', type=str, default='', help='bo trong -> chi test Teacher')
parser.add_argument('--teacher_ckpt', type=str, default='', help='mac dinh: cfg.MODEL.TEACHER')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_batches', type=int, default=30)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--teacher_alpha', type=float, nargs='+', default=[0.0],
                    help='(Teacher GT cu) input = (1-a)*GT + a*MotionBERT_lift; nhieu gia tri -> moi gia tri 1 dieu kien')
parser.add_argument('--use_thresh', type=float, default=3.0,
                    help='(mm) mean - that duoi nguong nay -> coi nhu KHONG dung thong tin anh rieng tung mau')
args = parser.parse_args()

from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401
from models.ARTS import ARTS
from core.base import load_model_weights, _PickleShim
from utils.jotr_dataset import get_test_dataset

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not torch.cuda.is_available():
    print("\n[LOI] Can CUDA de chay script nay.\n")
    sys.exit(1)

DEVICE = torch.device('cuda')


def hr(title=''):
    print('\n' + '=' * 78)
    if title:
        print(title)
        print('=' * 78)


def inspect_ckpt(mode, ckpt_path):
    """Doc checkpoint -> (kind, info). kind: 'teacher' | 'student' | 'khong ro'.
    info: has_probe, hint ('teacher' | 'student' | None), hint_dim."""
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
    has_hint_enc = any('hint_encoder.' in k for k in keys)
    has_hint_pred = any('hint_pred.' in k for k in keys)
    hint_dim = 0
    for k in keys:
        if k.endswith('hint_proj.weight'):
            hint_dim = int(state[k].shape[1])
    if has_hint_enc:
        kind = 'teacher'          # T-hint: kien truc Student + HintEncoder
    elif any('fusion.projector_student.' in k for k in keys):
        kind = 'student'
    elif any('fusion.joint_proj.' in k for k in keys):
        kind = 'teacher'
    else:
        kind = 'khong ro'
    # Kien truc v2 (PLAN_STUDENT_V2.md): nhan dien tu ten key -> dung lai dung kien truc, khong phu thuoc yml
    pe2d_key = next((k for k in keys if k.endswith('fusion.pe2d.proj.weight')), None)
    arch = {
        'joint_pe2d': pe2d_key is not None,
        'pe2d_num_freq': int(state[pe2d_key].shape[1]) // 4 if pe2d_key else 6,
        'fusion_joint_self_attn': any('.joint_sa.' in k for k in keys),
        'pose_head_mode': 'per_joint' if any('per_joint_head.' in k for k in keys) else 'vposer',
        'pose_query_bone': any('bone_proj.' in k for k in keys),
        'shape_attn_mode': 'tokens' if any('fuse_shape_tok.' in k for k in keys) else 'global',
    }
    info = {'has_probe': any('probe_' in k for k in keys),
            'hint': 'teacher' if has_hint_enc else ('student' if has_hint_pred else None),
            'hint_dim': hint_dim, 'arch': arch}
    tag = f" (T-hint k={hint_dim})" if has_hint_enc else (f" (S-hint k={hint_dim})" if has_hint_pred else "")
    v2 = {k: v for k, v in arch.items() if v not in (False, 'vposer', 'global') and k != 'pe2d_num_freq'}
    if v2:
        tag += f" (v2: {v2})"
    print(f"  [{mode}] {ckpt_path} -> loai: {kind}{tag}" + (" (co head tham do anh)" if info['has_probe'] else ""))
    if kind != 'khong ro' and kind != mode:
        print(f"\n[LOI] Checkpoint {kind.upper()} dat vao vi tri {mode.upper()}: {ckpt_path}")
        sys.exit(1)
    return info


def build_and_load(mode, ckpt_path, need_lift=False):
    info = inspect_ckpt(mode, ckpt_path)
    keys = ('name', 'teacher_lift_alpha_max', 'img_probe', 'teacher_hint', 'student_hint', 'hint_dim') \
        + tuple(info['arch'].keys())
    saved = {k: cfg.MODEL[k] for k in keys if k in cfg.MODEL}
    cfg.MODEL.name = mode
    for k, v in info['arch'].items():   # kien truc v2 lay tu checkpoint (Teacher cu va Student v2 khac nhau)
        cfg.MODEL[k] = v
    if mode == 'teacher':
        cfg.MODEL.teacher_hint = info['hint'] == 'teacher'
        cfg.MODEL.teacher_lift_alpha_max = 1.0 if (need_lift and not cfg.MODEL.teacher_hint) else 0.0
        cfg.MODEL.img_probe = info['has_probe']
    else:
        cfg.MODEL.student_hint = info['hint'] == 'student'
    if info['hint_dim'] > 0:
        cfg.MODEL.hint_dim = info['hint_dim']
    model = ARTS(num_joint=17, embed_dim=cfg.MODEL.hpe_dim)
    for k in keys:   # tra cfg ve nhu cu (key khong co tu dau thi xoa)
        if k in saved:
            cfg.MODEL[k] = saved[k]
        elif k in cfg.MODEL:
            del cfg.MODEL[k]
    model = model.to(DEVICE)
    model = load_model_weights(model, ckpt_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, info


def h36m_from_mesh(mesh, regressor):
    return torch.einsum('jk,bkc->bjc', regressor, mesh)


def errors_mm(mesh_pred, mesh_gt, reg):
    """MPJPE (17 khop H36M, root-rel) va MPVPE (mesh can theo Pelvis H36M), mm, theo tung mau."""
    jp, jg = h36m_from_mesh(mesh_pred, reg), h36m_from_mesh(mesh_gt, reg)
    rp, rg = jp[:, 0:1], jg[:, 0:1]
    mpjpe = torch.norm((jp - rp) - (jg - rg), dim=-1).mean(-1) * 1000.0
    mpvpe = torch.norm((mesh_pred - rp) - (mesh_gt - rg), dim=-1).mean(-1) * 1000.0
    return mpjpe.cpu(), mpvpe.cpu()


def paired(d):
    d = np.asarray(d, dtype=np.float64)
    if d.size < 2:
        return float(d.mean()), float('nan')
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(d.size))


def new_cond():
    return {f: {'j': [], 'v': []} for f in ('real', 'blind', 'mean')}


def run_three(net, joints_in, feats, store, reg, gt_mesh, **kw):
    """Chay net.smpl_model voi feature anh that / randn / TB batch, luu MPJPE + MPVPE."""
    variants = {
        'real': feats,
        'blind': torch.randn_like(feats),
        'mean': feats.mean(dim=0, keepdim=True).expand_as(feats).contiguous(),
    }
    for name, f in variants.items():
        out = net.smpl_model(joints=joints_in, img_feats=f, is_train=False, return_features=True, **kw)
        j, v = errors_mm(out['smpl_mesh_cam'], gt_mesh, reg)
        store[name]['j'].append(j)
        store[name]['v'].append(v)


def main():
    # Student scratch (vd Student v2) khong co Teacher: cho phep chi test Student khi co --student_ckpt
    # va khong dat --teacher_ckpt (yml scratch khong co MODEL.TEACHER).
    teacher_ckpt = args.teacher_ckpt or cfg.MODEL.get('TEACHER', '')
    if not teacher_ckpt and not args.student_ckpt:
        print("[LOI] Can it nhat --teacher_ckpt (hoac cfg.MODEL.TEACHER) hoac --student_ckpt")
        sys.exit(1)

    print(f"Student ckpt : {args.student_ckpt or '(khong dung - chi test Teacher)'}")
    print(f"Teacher ckpt : {teacher_ckpt or '(khong dung - chi test Student)'}")
    alphas = sorted(set(float(a) for a in args.teacher_alpha))

    hr("Dang build va nap checkpoint...")
    student = build_and_load('student', args.student_ckpt)[0] if args.student_ckpt else None
    teacher, t_hint = None, False
    if teacher_ckpt:
        teacher, t_info = build_and_load('teacher', teacher_ckpt, need_lift=any(a > 0 for a in alphas))
        t_hint = t_info['hint'] == 'teacher'

    hr("Dang chuan bi tap test...")
    test_name = cfg.DATASET.test_list[0]
    dataset = get_test_dataset(test_name, None)
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=gen,
                        num_workers=args.workers, pin_memory=False)
    print(f"Dataset: {test_name} | so sample: {len(dataset)} | se duyet toi da {args.max_batches} batch")
    reg = torch.as_tensor(dataset.h36m_joint_regressor, dtype=torch.float32, device=DEVICE)

    results = {}       # ten dieu kien -> new_cond()
    inp_err = {}       # ten dieu kien -> loi joint dau vao (mm)
    n_batches = 0
    with torch.no_grad():
        for inputs, targets, meta in loader:
            if n_batches >= args.max_batches:
                break
            n_batches += 1
            img = inputs['img'].to(DEVICE).float()
            joints2d = inputs['joints'].to(DEVICE).float()
            joints_mask = inputs['joints_mask'].to(DEVICE).float()
            gt_mesh = targets['smpl_mesh_cam'].to(DEVICE).float()
            gt_h36m = h36m_from_mesh(gt_mesh, reg)
            gt_h36m = gt_h36m - gt_h36m[:, 0:1]

            # ---------------- Teacher ----------------
            if teacher is None:
                pass
            elif t_hint:
                feat_t, _ = teacher.get_image_features(img)
                lifted = teacher.lift_2d_to_3d(joints2d, joints_mask=joints_mask)
                lifted = (lifted - lifted[:, 0:1]).float()
                z = teacher.smpl_model.hint_encoder(gt_h36m, lifted)
                for name, zz in (('Teacher T-hint [CO goi y]', z), ('Teacher T-hint [KHONG goi y]', torch.zeros_like(z))):
                    results.setdefault(name, new_cond())
                    inp_err.setdefault(name, []).append(
                        (torch.norm(lifted - gt_h36m, dim=-1).mean(-1) * 1000.0).cpu())
                    run_three(teacher, lifted, feat_t, results[name], reg, gt_mesh, hint_z=zz,
                              **teacher._pe2d_kwargs(joints2d, joints_mask))
            else:
                feat_t, _ = teacher.get_image_features(img)
                lifted = None
                if any(a > 0 for a in alphas):
                    lifted = teacher.lift_2d_to_3d(joints2d, joints_mask=joints_mask)
                    lifted = (lifted - lifted[:, 0:1]).to(gt_h36m.dtype)
                for a in alphas:
                    name = f'Teacher a={a} [' + ('GT sach' if a == 0 else f'(1-{a})*GT + {a}*MotionBERT') + ']'
                    t_in = gt_h36m if a == 0 else (1.0 - a) * gt_h36m + a * lifted
                    results.setdefault(name, new_cond())
                    inp_err.setdefault(name, []).append((torch.norm(t_in - gt_h36m, dim=-1).mean(-1) * 1000.0).cpu())
                    run_three(teacher, t_in, feat_t, results[name], reg, gt_mesh)

            # ---------------- Student ----------------
            if student is not None:
                feat_s, _ = student.get_image_features(img)
                pose_3d = student.lift_2d_to_3d(joints2d, joints_mask=joints_mask)
                pose_3d = pose_3d - pose_3d[:, 0:1]
                name = 'Student' + (' S-hint' if getattr(student, 'student_hint', False) else '') + ' (joint MotionBERT)'
                results.setdefault(name, new_cond())
                inp_err.setdefault(name, []).append((torch.norm(pose_3d - gt_h36m, dim=-1).mean(-1) * 1000.0).cpu())
                # Student v2 (joint_pe2d): truyen CUNG 2D dua vao MotionBERT cho PE vi tri trong fusion
                run_three(student, pose_3d, feat_s, results[name], reg, gt_mesh,
                          **student._pe2d_kwargs(joints2d, joints_mask))

            if n_batches % 5 == 0:
                print(f"  ... da xu ly {n_batches}/{args.max_batches} batch")

    hr("KET QUA (mm): anh that vs anh = nhieu trang (blind) vs anh = TB batch (mean)")
    summary = []
    for name, r in results.items():
        cat = {f: {m: torch.cat(r[f][m]).numpy() for m in ('j', 'v')} for f in r}
        n = len(cat['real']['j'])
        print(f"  {name}   (n={n}, loi joint dau vao: {torch.cat(inp_err[name]).mean().item():.1f} mm)")
        print(f"    {'':14s} {'MPJPE':>8s} {'MPVPE':>8s}")
        for f, lab in (('real', 'anh that'), ('blind', 'blind (randn)'), ('mean', 'mean (TB)')):
            print(f"    {lab:14s} {cat[f]['j'].mean():8.2f} {cat[f]['v'].mean():8.2f}")
        row = {'name': name}
        for f in ('mean', 'blind'):
            for m, lab in (('j', 'MPJPE'), ('v', 'MPVPE')):
                d = cat[f][m] - cat['real'][m]
                dm, ci = paired(d)
                row[f'{f}_{m}'] = dm
                print(f"    {f:5s} - that ({lab}): {dm:+.2f} +/- {ci:.2f} mm; {(d > 0).mean():.1%} mau te hon")
        row['real_j'] = cat['real']['j'].mean()
        summary.append(row)
    print("  (khoang tin cay tinh theo mau doc lap -> lac quan voi 3DPW)")

    hr("TOM TAT: muc dung thong tin anh rieng tung mau = (mean - that)")
    print(f"  {'dieu kien':46s} {'MPJPE that':>10s} {'dMPJPE':>8s} {'dMPVPE':>8s}  ket luan")
    for row in summary:
        uses = row['mean_j'] >= args.use_thresh or row['mean_v'] >= args.use_thresh
        verdict = 'DUNG anh' if uses else 'gan nhu KHONG dung anh'
        print(f"  {row['name']:46s} {row['real_j']:10.2f} {row['mean_j']:+8.2f} {row['mean_v']:+8.2f}  {verdict}")
    if t_hint:
        on = next(r for r in summary if 'CO goi y' in r['name'])
        off = next(r for r in summary if 'KHONG goi y' in r['name'])
        print(f"\n  T-hint: goi y giup {off['real_j'] - on['real_j']:+.2f} mm MPJPE "
              f"(tieu chi PLAN: >= 10 mm); dung anh khi CO goi y: {on['mean_j']:+.2f} mm (tieu chi: >= 8 mm)")


if __name__ == '__main__':
    main()
