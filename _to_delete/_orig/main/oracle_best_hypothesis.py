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
Do GIONG HET luc test (main/test_student_3dpw_sets.py): pose da chon -> mang sinh mesh (Student / Teacher, anh that)
-> mesh -> joint H36M -> dataset.evaluate: MPJPE / PA-MPJPE (14 khop) va MPVPE (mm), toan bo tap.
OC / PC chay ca 2 ban: chi seq test (pw3d_ocpc_test_only) va toan bo (--ocpc_mode).
GT chi dung de CHON phuong an (so pose 17 khop voi GT), sai so bao cao la sai so MESH cuoi cung.
Dong REF: dua pose GT vao mang (can duoi cua mang giai ma).

Vi du:
  python main/oracle_best_hypothesis.py --cfg config/train_student_kd_e1_v2_diff_jotr_l2last.yml \
      --student_ckpt experiment/student_kd_e1_v2_diff_jotr_l2last/checkpoint/best.pth.tar \
      --teacher_ckpt experiment/teacher_gt_v2_jotr/checkpoint/best.pth.tar \
      --sets 3dpw,3dpw-oc,3dpw-pc,3dpw-crowd --K 20 --out_dir experiment/_eval/oracle_best_l2last_eval14
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
parser.add_argument('--ocpc_mode', type=str, default='both', choices=['test', 'all', 'both'],
                    help='OC/PC: test = chi seq test, all = toan bo (ro ri), both = ca hai')
parser.add_argument('--K', type=int, default=20)
parser.add_argument('--steps', type=int, default=5, help='so buoc DDIM khi sinh K gia thuyet')
parser.add_argument('--max_samples', type=int, default=0, help='so mau ngau nhien moi tap; 0 = toan bo')
parser.add_argument('--batch_size', type=int, default=32)
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
REF = 'REF pose GT (tran)'
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
    teacher = build_and_load('teacher', args.teacher_ckpt)[0] if args.teacher_ckpt else None
    nets = [('Student', student)] + ([('Teacher', teacher)] if teacher is not None else [])
    log(f'cfg={args.cfg} | student={args.student_ckpt} | teacher={args.teacher_ckpt or "-"} | K={args.K} S={args.steps} '
        f'| max_samples={args.max_samples or "toan bo"}')
    log('Sai so = dataset.evaluate tren MESH dau ra (MPJPE / PA-MPJPE 14 khop, MPVPE; mm) - giong test_student_3dpw_sets.py')
    log('Pose da chon (bang GT, oracle) -> mang sinh mesh (anh that) -> mesh -> joint -> sai so.')

    runs = []
    for name in [s_.strip() for s_ in args.sets.split(',') if s_.strip()]:
        if name in ('3dpw-oc', '3dpw-pc'):
            if args.ocpc_mode in ('test', 'both'):
                runs.append((f'{name} (chi seq test)', name, True))
            if args.ocpc_mode in ('all', 'both'):
                runs.append((f'{name} (toan bo, ro ri)', name, False))
        else:
            runs.append((name, name, False))

    rows_all = [REF] + LEVELS
    summary = {}
    for label, name, test_only in runs:
        cfg.DATASET.pw3d_ocpc_test_only = test_only
        try:
            ds = get_test_dataset(name, None)
        except Exception as e:
            log(f'[{label}] LOI: {type(e).__name__}: {e}')
            continue
        n_all = len(ds)
        if 0 < args.max_samples < n_all:
            idx = np.sort(np.random.default_rng(args.seed).choice(n_all, args.max_samples, replace=False)).tolist()
            ds_run = Subset(ds, idx)
        else:
            ds_run = ds
        loader = DataLoader(ds_run, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        reg = torch.as_tensor(ds.h36m_joint_regressor, dtype=torch.float32, device=DEVICE)
        acc = {(lv, tag): {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': []} for lv in rows_all for tag, _ in nets}
        seen = 0
        with torch.no_grad():
            for bi, (inputs, targets, meta) in enumerate(loader):
                img = inputs['img'].to(DEVICE)
                j2 = inputs['joints'].to(DEVICE).float()
                jm = inputs['joints_mask'].to(DEVICE).float()
                cam = inputs.get('cam_bbox', torch.zeros(img.shape[0], 3)).to(DEVICE).float()
                gt_mesh = targets['smpl_mesh_cam'].to(DEVICE).float()
                gt17 = h36m_from_mesh(gt_mesh, reg)
                gt17 = gt17 - gt17[:, 0:1]
                base = student.lift_diffusion(j2, jm, cam).float()          # giong forward_student (S1/K1, seed 0)
                base = base - base[:, 0:1]
                torch.manual_seed(args.seed * 100003 + bi)
                hyps = student._diff_hyp(j2, jm, cam, args.steps, args.K, True).float()   # (B, K, 17, 3)
                cands = torch.cat([base[:, None], hyps], 1)
                poses = {REF: gt17, 'L0 mac dinh S1/K1': base, 'L1 trung binh K': hyps.mean(1)}
                poses.update(select(cands, gt17))
                refl = cands.clone()
                refl[..., 2] = -refl[..., 2]
                poses['L6 L5 + phan xa toan than'] = select(torch.cat([cands, refl], 1), gt17)['L5 oracle nhom + lat chi']
                gt_np = gt_mesh.cpu().numpy()
                for tag, net in nets:
                    feat = net.get_image_features(img)[0]
                    pe = net._pe2d_kwargs(j2, jm)
                    for lv in rows_all:
                        out = net.smpl_model(joints=poses[lv], img_feats=feat, is_train=False,
                                             return_features=False, **pe)
                        pm = out['smpl_mesh_cam'].detach().float().cpu().numpy()
                        outs = [{'smpl_mesh_cam': pm[k], 'smpl_mesh_cam_target': gt_np[k].copy()}
                                for k in range(pm.shape[0])]
                        r = ds.evaluate(outs, min(seen, n_all - pm.shape[0]))
                        for key in ('mpjpe', 'pa_mpjpe', 'mpvpe'):
                            acc[(lv, tag)][key].extend(r[key])
                seen += img.shape[0]
                if (bi + 1) % 50 == 0:
                    l0 = np.mean(acc[('L0 mac dinh S1/K1', 'Student')]['mpjpe'])
                    l5 = np.mean(acc[('L5 oracle nhom + lat chi', 'Student')]['mpjpe'])
                    print(f'  [{label}] {seen}/{len(ds_run)} | Student MPJPE L0 {l0:.2f} -> L5 {l5:.2f}')
        res = {}
        log(f'\n==================== {label} (n={seen}/{n_all}) ====================')
        head = f"{'muc':30s}"
        for tag, _ in nets:
            head += f" | {tag + ' MPJPE   PA  MPVPE':>27s}"
        log(head)
        for lv in rows_all:
            s_ = f'{lv:30s}'
            res[lv] = {}
            for tag, _ in nets:
                m = {k: float(np.mean(v)) for k, v in acc[(lv, tag)].items()}
                res[lv][tag] = m
                s_ += f" | {m['mpjpe']:13.2f} {m['pa_mpjpe']:6.2f} {m['mpvpe']:6.2f}"
            log(s_)
            if lv == REF:
                log('-' * len(head))
        summary[label] = {'n': seen, 'n_all': n_all, 'res': res}

    # Bang gon: Student, L0 (= so test cua ban) vs cac muc oracle
    log('\n' + '=' * 100)
    log('TOM TAT Student MPJPE / PA / MPVPE (mm, 14 khop, dataset.evaluate)')
    pick = ['L0 mac dinh S1/K1', 'L2 oracle ca pose', 'L3 oracle theo nhom', 'L5 oracle nhom + lat chi',
            'L6 L5 + phan xa toan than', REF]
    short = ['L0 (hien tai)', 'L2 ca pose', 'L3 nhom', 'L5 nhom+lat', 'L6 +phan xa', 'REF GT']
    log(f"{'tap':28s} {'n':>6s} | " + ' | '.join(f'{x:>20s}' for x in short))
    for label, d in summary.items():
        log(f"{label:28s} {d['n']:6d} | " + ' | '.join(
            f"{d['res'][lv]['Student']['mpjpe']:6.2f}/{d['res'][lv]['Student']['pa_mpjpe']:5.2f}/{d['res'][lv]['Student']['mpvpe']:6.2f}"
            for lv in pick))
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    log(f'\nDa luu: {out_dir}')


if __name__ == '__main__':
    main()
