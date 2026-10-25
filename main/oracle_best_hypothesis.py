"""
oracle_best_hypothesis.py -- Neu biet dap an (GT) de chon phuong an tot nhat cua DiffLifter thi con sai bao nhieu?
Chi eval, khong train. Dung GT CHI DE CHON (oracle) -> day la CAN TREN, khong phai ket qua trien khai.

Phuong an (moi mau): pose mac dinh S=1/K=1 cua Student + K gia thuyet DiffLifter (--K, --steps, nhieu ngau nhien co dinh).
Cac muc chon (tu de den kho):
  L0  mac dinh (S=1, K=1)                  -- moc hien tai
  L1  trung binh K gia thuyet
  L2  oracle CA POSE: 1 phuong an tot nhat / mau
  L3  oracle THEO NHOM: moi nhom (chan P, chan T, tay T, tay P, than-dau) lay phuong an tot nhat rieng
  L4  oracle THEO KHOP: moi khop lay phuong an tot nhat rieng
  L5  oracle THEO NHOM + LAT CHI: moi chi chon trong (K+1) phuong an x 4 trang thai lat do sau (doan tren / duoi)
  L6  L5 + them phuong an phan xa do sau toan than (mat <-> lung)
Bao cao moi tap: MPJPE17 / PA-MPJPE pose (mm), loi tung khop L0 vs L3/L5; neu co --student_ckpt / --teacher_ckpt:
mesh (MPJPE17 tu mesh / MPVPE) khi dua pose da chon vao Student / Teacher (anh that).

Vi du:
  python main/oracle_best_hypothesis.py --cfg config/train_student_kd_e1_v2_diff_jotr_l2last.yml \
      --student_ckpt experiment/student_kd_e1_v2_diff_jotr_l2last/checkpoint/best.pth.tar \
      --teacher_ckpt experiment/teacher_gt_v2_jotr/checkpoint/best.pth.tar \
      --sets 3dpw,3dpw-oc,3dpw-pc,3dpw-crowd --K 20 --max_samples 4000 --out_dir experiment/_eval/oracle_best_l2last
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

parser = argparse.ArgumentParser(description='Can tren oracle: chon phuong an DiffLifter tot nhat bang GT')
parser.add_argument('--cfg', type=str, required=True, help='YAML Student (lifter diffusion + loader test)')
parser.add_argument('--student_ckpt', type=str, required=True, help='Student best.pth.tar (chua trong so lifter)')
parser.add_argument('--teacher_ckpt', type=str, default='', help='(tuy chon) Teacher GT de giai ma pose da chon')
parser.add_argument('--no_mesh', action='store_true', help='chi do pose, khong chay Student/Teacher')
parser.add_argument('--sets', type=str, default='3dpw,3dpw-oc,3dpw-pc,3dpw-crowd')
parser.add_argument('--K', type=int, default=20)
parser.add_argument('--steps', type=int, default=5, help='so buoc DDIM khi sinh K gia thuyet')
parser.add_argument('--max_samples', type=int, default=0, help='so mau ngau nhien moi tap; 0 = toan bo')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--out_dir', type=str, required=True)
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

# ============================================================
JN = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top',
      'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
LIMBS = ((1, 2, 3), (4, 5, 6), (11, 12, 13), (14, 15, 16))      # (goc, giua, ngon)
GROUPS = [list(l) for l in LIMBS] + [[7, 8, 9, 10]]              # 4 chi + than-dau (Pelvis luon = 0)
LEVELS = ['L0 mac dinh S1/K1', 'L1 trung binh K', 'L2 oracle ca pose', 'L3 oracle theo nhom',
          'L4 oracle theo khop', 'L5 oracle nhom + lat chi', 'L6 L5 + phan xa toan than']


def procrustes_err(pred, gt):
    """PA-MPJPE theo mau (m). pred, gt (B, 17, 3)."""
    mp, mg = pred.mean(1, keepdim=True), gt.mean(1, keepdim=True)
    p, g = pred - mp, gt - mg
    u, s, vt = torch.linalg.svd(p.transpose(1, 2) @ g)
    d = torch.sign(torch.det(vt.transpose(1, 2) @ u.transpose(1, 2)))
    dm = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    r = vt.transpose(1, 2) @ dm @ u.transpose(1, 2)
    sc = (s * torch.diagonal(dm, dim1=1, dim2=2)).sum(-1) / (p ** 2).sum((1, 2)).clamp_min(1e-12)
    al = sc[:, None, None] * p @ r.transpose(1, 2) + mg
    return torch.norm(al - gt, dim=-1).mean(-1)


def limb_states(p, limb):
    """p (B, N, 17, 3) -> 4 trang thai lat do sau cua 1 chi: (B, N, 4, 3, 3) toa do (goc, giua, ngon)."""
    a, b, c = limb
    out = []
    for up in (False, True):
        for lo in (False, True):
            q = p[:, :, [a, b, c]].clone()
            if up:
                nb = 2 * q[..., 0, 2] - q[..., 1, 2]
                q[..., 2, 2] = q[..., 2, 2] + (nb - q[..., 1, 2])
                q[..., 1, 2] = nb
            if lo:
                q[..., 2, 2] = 2 * q[..., 1, 2] - q[..., 2, 2]
            out.append(q)
    return torch.stack(out, 2)


def select(cands, gt):
    """cands (B, N, 17, 3) -> dict muc -> pose (B, 17, 3). Dung GT de chon (oracle)."""
    B, N = cands.shape[:2]
    ar = torch.arange(B, device=cands.device)
    err_j = torch.norm(cands - gt[:, None], dim=-1)                         # (B, N, 17)
    res = {}
    res['L2 oracle ca pose'] = cands[ar, err_j.mean(-1).argmin(1)]
    g_pose = torch.zeros_like(gt)
    for g in GROUPS:
        bi = err_j[:, :, g].mean(-1).argmin(1)
        g_pose[:, g] = cands[ar, bi][:, g]
    res['L3 oracle theo nhom'] = g_pose
    bj = err_j.argmin(1)                                                     # (B, 17)
    res['L4 oracle theo khop'] = torch.gather(cands, 1, bj[:, None, :, None].expand(-1, 1, -1, 3))[:, 0]
    # nhom + lat chi: moi chi chon trong N x 4 trang thai; than-dau chon trong N
    f_pose = g_pose.clone()
    for limb in LIMBS:
        st = limb_states(cands, limb)                                        # (B, N, 4, 3, 3)
        e = torch.norm(st - gt[:, None, None, list(limb)], dim=-1).mean(-1)   # (B, N, 4), ca 3 khop cua chi
        best = e.reshape(B, -1).argmin(1)
        f_pose[:, list(limb)] = st.reshape(B, -1, 3, 3)[ar, best]
    res['L5 oracle nhom + lat chi'] = f_pose
    return res


def main():
    out_dir = Path(args.out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        print(f"[LOI] Thu muc ket qua khong rong: {out_dir}")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg=''):
        print(msg)
        lines.append(msg)
        (out_dir / 'log.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    hr('Nap checkpoint')
    student, _ = build_and_load('student', args.student_ckpt)
    if not getattr(student, 'use_diff', False):
        print('[LOI] Student phai dung pose_lifter=diffusion')
        sys.exit(1)
    teacher = build_and_load('teacher', args.teacher_ckpt)[0] if (args.teacher_ckpt and not args.no_mesh) else None
    use_s = not args.no_mesh
    log(f'cfg={args.cfg} | student={args.student_ckpt} | teacher={args.teacher_ckpt or "-"} | K={args.K} S={args.steps} '
        f'| max_samples={args.max_samples}')
    log('Pose: MPJPE17 / PA (H36M-17 root-rel, mm). Mesh: MPJPE17 tu mesh / MPVPE (mm), anh that. Oracle dung GT de chon.')
    summary = {}
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        try:
            ds = get_test_dataset(name, None)
        except Exception as e:
            log(f'[{name}] LOI: {type(e).__name__}: {e}')
            continue
        n_all = len(ds)
        if 0 < args.max_samples < n_all:
            idx = np.sort(np.random.default_rng(args.seed).choice(n_all, args.max_samples, replace=False)).tolist()
            ds_run = Subset(ds, idx)
        else:
            ds_run = ds
        loader = DataLoader(ds_run, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        reg = torch.as_tensor(ds.h36m_joint_regressor, dtype=torch.float32, device=DEVICE)
        acc = {lv: {'pose': [], 'pa': [], 'joint': [], 's_mpjpe': [], 's_mpvpe': [], 't_mpjpe': [], 't_mpvpe': []}
               for lv in LEVELS}
        seen = 0
        with torch.no_grad():
            for bi, (inputs, targets, meta) in enumerate(loader):
                img = inputs['img'].to(DEVICE).float()
                j2 = inputs['joints'].to(DEVICE).float()
                jm = inputs['joints_mask'].to(DEVICE).float()
                cam = inputs.get('cam_bbox', torch.zeros(img.shape[0], 3)).to(DEVICE).float()
                gt_mesh = targets['smpl_mesh_cam'].to(DEVICE).float()
                gt17 = h36m_from_mesh(gt_mesh, reg)
                gt17 = gt17 - gt17[:, 0:1]
                base = student.lift_diffusion(j2, jm, cam).float()
                base = base - base[:, 0:1]
                torch.manual_seed(args.seed * 100003 + bi)
                hyps = student._diff_hyp(j2, jm, cam, args.steps, args.K, True).float()   # (B, K, 17, 3)
                cands = torch.cat([base[:, None], hyps], 1)
                poses = {'L0 mac dinh S1/K1': base, 'L1 trung binh K': hyps.mean(1)}
                poses.update(select(cands, gt17))
                refl = cands.clone()
                refl[..., 2] = -refl[..., 2]
                poses['L6 L5 + phan xa toan than'] = select(torch.cat([cands, refl], 1), gt17)['L5 oracle nhom + lat chi']
                feat_s = student.get_image_features(img)[0] if use_s else None
                feat_t = teacher.get_image_features(img)[0] if teacher is not None else None
                for lv in LEVELS:
                    p = poses[lv]
                    a = acc[lv]
                    ej = torch.norm(p - gt17, dim=-1)
                    a['pose'].append(ej.mean(-1).cpu())
                    a['pa'].append(procrustes_err(p, gt17).cpu())
                    a['joint'].append(ej.cpu())
                    for tag, net, f in (('s', student, feat_s), ('t', teacher, feat_t)):
                        if net is None or f is None:
                            continue
                        out = net.smpl_model(joints=p, img_feats=f, is_train=False, return_features=False,
                                             **net._pe2d_kwargs(j2, jm))
                        jp = h36m_from_mesh(out['smpl_mesh_cam'], reg)
                        jg = h36m_from_mesh(gt_mesh, reg)
                        rp, rg = jp[:, 0:1], jg[:, 0:1]
                        a[f'{tag}_mpjpe'].append((torch.norm((jp - rp) - (jg - rg), dim=-1).mean(-1)).cpu())
                        a[f'{tag}_mpvpe'].append((torch.norm((out['smpl_mesh_cam'] - rp) - (gt_mesh - rg), dim=-1).mean(-1)).cpu())
                seen += img.shape[0]
                if (bi + 1) % 20 == 0:
                    print(f'  [{name}] {seen}/{len(ds_run)}')
        res = {}
        log(f'\n==================== {name} (n={seen}/{n_all}) ====================')
        log(f"{'muc':30s} {'pose':>7s} {'PA':>7s}" + ("  | Student mesh MPJPE/MPVPE" if use_s else '')
            + ("  | Teacher mesh MPJPE/MPVPE" if teacher is not None else ''))
        for lv in LEVELS:
            a = acc[lv]
            r = {'pose_mm': float(torch.cat(a['pose']).mean() * 1000), 'pa_mm': float(torch.cat(a['pa']).mean() * 1000),
                 'per_joint_mm': (torch.cat(a['joint']).mean(0) * 1000).tolist()}
            s_ = f"{lv:30s} {r['pose_mm']:7.2f} {r['pa_mm']:7.2f}"
            for tag, lab in (('s', 'Student'), ('t', 'Teacher')):
                if a[f'{tag}_mpjpe']:
                    r[f'{lab}_mesh_mpjpe_mm'] = float(torch.cat(a[f'{tag}_mpjpe']).mean() * 1000)
                    r[f'{lab}_mesh_mpvpe_mm'] = float(torch.cat(a[f'{tag}_mpvpe']).mean() * 1000)
                    s_ += f"  | {r[f'{lab}_mesh_mpjpe_mm']:7.2f} / {r[f'{lab}_mesh_mpvpe_mm']:7.2f}"
            res[lv] = r
            log(s_)
        for lv in ('L3 oracle theo nhom', 'L5 oracle nhom + lat chi'):
            log(f"  Tung khop (mm) L0 -> {lv[:2]}: " + ', '.join(
                f"{JN[j]} {res['L0 mac dinh S1/K1']['per_joint_mm'][j]:.0f}->{res[lv]['per_joint_mm'][j]:.0f}"
                for j in range(1, 17)))
        summary[name] = res
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    log(f'\nDa luu: {out_dir}')


if __name__ == '__main__':
    main()
