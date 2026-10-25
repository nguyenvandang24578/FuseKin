"""
analyze_kd_selection.py -- Phan tich truoc khi thiet ke KD chon loc / KD co trong so. Chi eval, khong train.

Tung KHOP cua tung MAU (tren tap train cua Student va tap test):
  e_T    : loi khop Teacher (mesh Teacher -> H36M-17, root-rel) so GT, mm        -> Teacher quality
  e_S    : loi khop Student (checkpoint KD), mm
  e_lift : loi pose dau vao cua Student (lifter) so GT, mm
  d      : 1 - cos(LN(token khop Student sau fusion), LN(token Teacher))        -> Teacher-Student discrepancy
  (tuy chon --init_ckpt: e_S0, d0 cua Student init-only -> loi ich KD = e_S0 - e_S)
Mau: d_glob (feature toan cuc), MPJPE_T, MPJPE_S.
Bao cao: phan phoi e_T / d; tuong quan Spearman (d ~ e_S, d ~ e_lift, d ~ e_T); bang 4 o (Teacher tot/kem x d nho/lon);
bang theo thap phan vi cua d va cua e_T; goi y tau cho trong so mem; luu samples.csv tung khop.
Tap train dung loader train cua YAML Student (co augmentation nhu luc KD), lay ngau nhien --max_samples mau moi bo.

Vi du:
  python main/analyze_kd_selection.py --cfg config/train_student_kd_e1_v2_diff_jotr_l2last.yml \
      --student_ckpt experiment/student_kd_e1_v2_diff_jotr_l2last/checkpoint/best.pth.tar \
      --teacher_ckpt experiment/teacher_gt_v2_jotr/checkpoint/best.pth.tar \
      --train_sets config --test_sets 3dpw,3dpw-pc,3dpw-crowd --max_samples 2000 \
      --out_dir experiment/_eval/kd_sel_analysis_jotr
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import csv
import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

parser = argparse.ArgumentParser(description='Phan tich Teacher quality / Teacher-Student discrepancy cho KD')
parser.add_argument('--cfg', type=str, required=True, help='YAML Student KD (du lieu train, lifter)')
parser.add_argument('--student_ckpt', type=str, required=True, help='Student KD best.pth.tar')
parser.add_argument('--teacher_ckpt', type=str, default='', help='mac dinh cfg.MODEL.TEACHER')
parser.add_argument('--init_ckpt', type=str, default='', help='(tuy chon) Student init-only de do loi ich KD')
parser.add_argument('--train_sets', type=str, default='config', help="'config' = DATASET.train_list; rong = bo qua")
parser.add_argument('--test_sets', type=str, default='3dpw', help='cach nhau dau phay; rong = bo qua')
parser.add_argument('--max_samples', type=int, default=2000, help='so mau ngau nhien moi bo; 0 = toan bo')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--teacher_good_mm', type=float, default=50.0, help='nguong Teacher "tot" cho bang 4 o')
parser.add_argument('--out_dir', type=str, required=True)
args = parser.parse_args()

from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401
from models.ARTS import ARTS
from core.base import load_model_weights, _PickleShim
from utils.jotr_dataset import get_train_dataset, get_test_dataset

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



# ============================================================
JN = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top',
      'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')


def cos_dist(s, t):
    return 1.0 - F.cosine_similarity(F.layer_norm(s, s.shape[-1:]), F.layer_norm(t, t.shape[-1:]), dim=-1)


def joints17(out, reg):
    j = h36m_from_mesh(out['smpl_mesh_cam'], reg)
    return j - j[:, 0:1]


def lift(student, j2, jm, inputs, img):
    if getattr(student, 'use_diff', False):
        cam_bbox = inputs.get('cam_bbox', torch.zeros(img.shape[0], 3)).to(DEVICE).float()
        p = student.lift_diffusion(j2, jm, cam_bbox)
    elif getattr(student, 'use_pmce', False):
        jr = inputs.get('joints_raw', j2).to(DEVICE).float()
        p = student.lift_pmce(jr, jm, student.get_image_features(img)[1])
    else:
        p = student.lift_2d_to_3d(j2, joints_mask=jm)
    return (p - p[:, 0:1]).float()


def spearman(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return float('nan')
    rx = np.argsort(np.argsort(x[ok]))
    ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def collect(name, ds, is_train, teacher, student, init_s, reg):
    n = len(ds)
    if args.max_samples > 0 and args.max_samples < n:
        idx = np.sort(np.random.default_rng(args.seed).choice(n, args.max_samples, replace=False)).tolist()
        ds = Subset(ds, idx)
    else:
        idx = list(range(n))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    J = {k: [] for k in ('e_T', 'e_S', 'e_lift', 'd', 'e_S0', 'd0', 'valid')}
    Sm = {k: [] for k in ('d_glob', 'mpjpe_T', 'mpjpe_S', 'd_glob0', 'mpjpe_S0')}
    seen = 0
    with torch.no_grad():
        for inputs, targets, meta in loader:
            img = inputs['img'].to(DEVICE).float()
            j2 = inputs['joints'].to(DEVICE).float()
            jm = inputs['joints_mask'].to(DEVICE).float()
            if 'orig_joint_cam' in targets:          # tap train / unified: giong dau vao Teacher luc KD
                gt17 = targets['orig_joint_cam'].to(DEVICE).float()
            else:
                gt17 = h36m_from_mesh(targets['smpl_mesh_cam'].to(DEVICE).float(), reg)
            gt17 = gt17 - gt17[:, 0:1]
            if is_train and 'fit_joint_trunc' in meta:
                valid = meta['fit_joint_trunc'].to(DEVICE).float().reshape(gt17.shape[0], 17, -1)[..., 0]
                valid = valid * meta['is_valid_fit'].to(DEVICE).float().reshape(-1, 1)
            else:
                valid = torch.ones(gt17.shape[:2], device=DEVICE)

            def run(net, joints_in, feats):
                return net.smpl_model(joints=joints_in, img_feats=feats, is_train=False, return_features=True,
                                      **net._pe2d_kwargs(j2, jm))
            ft, _ = teacher.get_image_features(img)
            t = run(teacher, gt17, ft)
            fs, _ = student.get_image_features(img)
            lifted = lift(student, j2, jm, inputs, img)
            s = run(student, lifted, fs)
            jt, js = joints17(t, reg), joints17(s, reg)
            e_T = torch.norm(jt - gt17, dim=-1) * 1000
            e_S = torch.norm(js - gt17, dim=-1) * 1000
            J['e_T'].append(e_T); J['e_S'].append(e_S)
            J['e_lift'].append(torch.norm(lifted - gt17, dim=-1) * 1000)
            J['d'].append(cos_dist(s['feat_joint'], t['feat_joint']))
            J['valid'].append(valid)
            nv = valid.sum(-1).clamp_min(1)
            Sm['d_glob'].append(cos_dist(s['feat_global'], t['feat_global']))
            Sm['mpjpe_T'].append((e_T * valid).sum(-1) / nv)
            Sm['mpjpe_S'].append((e_S * valid).sum(-1) / nv)
            if init_s is not None:
                f0, _ = init_s.get_image_features(img)
                l0 = lift(init_s, j2, jm, inputs, img)
                s0 = run(init_s, l0, f0)
                e0 = torch.norm(joints17(s0, reg) - gt17, dim=-1) * 1000
                J['e_S0'].append(e0)
                J['d0'].append(cos_dist(s0['feat_joint'], t['feat_joint']))
                Sm['d_glob0'].append(cos_dist(s0['feat_global'], t['feat_global']))
                Sm['mpjpe_S0'].append((e0 * valid).sum(-1) / nv)
            seen += img.shape[0]
            if (seen // args.batch_size) % 20 == 0:
                print(f'  [{name}] {seen}/{len(idx)}')
    J = {k: torch.cat(v).cpu().numpy() for k, v in J.items() if v}
    Sm = {k: torch.cat(v).cpu().numpy() for k, v in Sm.items() if v}
    return J, Sm, idx


def report(name, J, Sm, log, rows_writer, idx):
    v = J['valid'] > 0
    eT, eS, eL, d = J['e_T'][v], J['e_S'][v], J['e_lift'][v], J['d'][v]
    jid = np.tile(np.arange(17), (J['valid'].shape[0], 1))[v]
    res = {'n_samples': int(J['valid'].shape[0]), 'n_joints': int(v.sum())}
    log(f'\n==================== {name} (mau {res["n_samples"]}, khop hop le {res["n_joints"]}) ====================')
    # 1. Teacher quality
    q = np.percentile(eT, [50, 75, 90, 99])
    res['teacher'] = {'mean': float(eT.mean()), 'p50': q[0], 'p75': q[1], 'p90': q[2], 'p99': q[3],
                      'pct_gt30': float((eT > 30).mean() * 100), 'pct_gt50': float((eT > 50).mean() * 100),
                      'pct_gt80': float((eT > 80).mean() * 100),
                      'sample_mpjpe_T': float(Sm['mpjpe_T'].mean()), 'sample_mpjpe_S': float(Sm['mpjpe_S'].mean())}
    log(f"1. Teacher quality: e_T TB {eT.mean():.1f} | p50 {q[0]:.1f} p75 {q[1]:.1f} p90 {q[2]:.1f} p99 {q[3]:.1f} mm | "
        f">30mm {res['teacher']['pct_gt30']:.1f}% >50mm {res['teacher']['pct_gt50']:.1f}% >80mm {res['teacher']['pct_gt80']:.1f}%")
    log(f"   MPJPE mau: Teacher {Sm['mpjpe_T'].mean():.2f} | Student {Sm['mpjpe_S'].mean():.2f} | lifter {eL.mean():.2f} mm")
    per_j = [(JN[j], float(eT[jid == j].mean()), float(eS[jid == j].mean()), float(d[jid == j].mean()))
             for j in range(1, 17) if (jid == j).any()]
    log('   Theo khop (e_T / e_S / d): ' + ', '.join(f'{n_} {a:.0f}/{b:.0f}/{c:.3f}' for n_, a, b, c in per_j))
    res['per_joint'] = {n_: {'e_T': a, 'e_S': b, 'd': c} for n_, a, b, c in per_j}
    # 2. Discrepancy
    qd = np.percentile(d, [10, 50, 90])
    res['disc'] = {'p10': qd[0], 'p50': qd[1], 'p90': qd[2], 'glob_p50': float(np.median(Sm['d_glob'])),
                   'rho_d_eS': spearman(d, eS), 'rho_d_elift': spearman(d, eL), 'rho_d_eT': spearman(d, eT),
                   'rho_eT_eS': spearman(eT, eS),
                   'rho_dglob_mpjpeS': spearman(Sm['d_glob'], Sm['mpjpe_S'])}
    r = res['disc']
    log(f"2. Discrepancy d: p10 {qd[0]:.3f} p50 {qd[1]:.3f} p90 {qd[2]:.3f} | global p50 {r['glob_p50']:.3f}")
    log(f"   Spearman: d~e_S {r['rho_d_eS']:+.3f} | d~e_lift {r['rho_d_elift']:+.3f} | d~e_T {r['rho_d_eT']:+.3f} | "
        f"e_T~e_S {r['rho_eT_eS']:+.3f} | d_glob~MPJPE_S {r['rho_dglob_mpjpeS']:+.3f}")
    # 4 o
    tg = eT <= args.teacher_good_mm
    ds_ = d <= qd[1]
    log(f"   Bang 4 o (Teacher tot = e_T <= {args.teacher_good_mm:.0f} mm; d nho = <= trung vi):")
    log(f"     {'o':28s} {'%khop':>6s} {'e_S':>7s} {'e_lift':>7s} {'e_T':>7s}" + (f" {'loiichKD':>9s}" if 'e_S0' in J else ''))
    res['quad'] = {}
    for lab, m in (('Teacher tot & d nho', tg & ds_), ('Teacher tot & d lon', tg & ~ds_),
                   ('Teacher kem & d nho', ~tg & ds_), ('Teacher kem & d lon', ~tg & ~ds_)):
        if not m.any():
            continue
        row = {'pct': float(m.mean() * 100), 'e_S': float(eS[m].mean()), 'e_lift': float(eL[m].mean()),
               'e_T': float(eT[m].mean())}
        extra = ''
        if 'e_S0' in J:
            g = (J['e_S0'][v] - eS)[m]
            row['kd_gain'] = float(g.mean())
            extra = f" {row['kd_gain']:+9.2f}"
        res['quad'][lab] = row
        log(f"     {lab:28s} {row['pct']:6.1f} {row['e_S']:7.1f} {row['e_lift']:7.1f} {row['e_T']:7.1f}{extra}")
    # thap phan vi
    for key, arr in (('d', d), ('e_T', eT)):
        edges = np.percentile(arr, np.linspace(0, 100, 11))
        b = np.clip(np.searchsorted(edges, arr, side='right') - 1, 0, 9)
        log(f"   Theo thap phan vi {key}: [bin: {key} TB | e_S | e_lift | e_T" + (' | loi ich KD]' if 'e_S0' in J else ']'))
        res[f'dec_{key}'] = []
        for k in range(10):
            m = b == k
            if not m.any():
                continue
            it = {'bin': k + 1, key: float(arr[m].mean()), 'e_S': float(eS[m].mean()), 'e_lift': float(eL[m].mean()),
                  'e_T': float(eT[m].mean())}
            s_ = f"     D{k + 1:<2d} {it[key]:8.3f} | {it['e_S']:6.1f} | {it['e_lift']:6.1f} | {it['e_T']:6.1f}"
            if 'e_S0' in J:
                it['kd_gain'] = float((J['e_S0'][v] - eS)[m].mean())
                s_ += f" | {it['kd_gain']:+6.2f}"
            res[f'dec_{key}'].append(it)
            log(s_)
    if 'e_S0' in J:
        res['kd_gain_total'] = float((J['e_S0'][v] - eS).mean())
        res['d0_p50'] = float(np.median(J['d0'][v]))
        log(f"   Loi ich KD tong (e_S init - e_S KD): {res['kd_gain_total']:+.2f} mm | d init p50 {res['d0_p50']:.3f} vs KD {qd[1]:.3f}")
    # goi y tau cho trong so mem (chi y nghia tren tap train)
    res['suggest'] = {'tau_T_p75_mm': float(q[1]), 'tau_d_median': float(qd[1])}
    log(f"   Goi y (neu la tap train): tau_T ~ p75 e_T = {q[1]:.1f} mm | tau_d ~ trung vi d = {qd[1]:.3f}")
    # csv
    nS = J['valid'].shape[0]
    for i in range(nS):
        for j in range(17):
            if J['valid'][i, j] <= 0:
                continue
            row = {'set': name, 'index': idx[i], 'joint': JN[j], 'e_T': f"{J['e_T'][i, j]:.2f}",
                   'e_S': f"{J['e_S'][i, j]:.2f}", 'e_lift': f"{J['e_lift'][i, j]:.2f}", 'd': f"{J['d'][i, j]:.5f}",
                   'd_glob': f"{Sm['d_glob'][i]:.5f}"}
            if 'e_S0' in J:
                row['e_S0'] = f"{J['e_S0'][i, j]:.2f}"
                row['d0'] = f"{J['d0'][i, j]:.5f}"
            rows_writer(row)
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

    teacher_ckpt = args.teacher_ckpt or cfg.MODEL.get('TEACHER', '')
    hr('Nap checkpoint')
    teacher, _ = build_and_load('teacher', teacher_ckpt)
    student, _ = build_and_load('student', args.student_ckpt)
    init_s = build_and_load('student', args.init_ckpt)[0] if args.init_ckpt else None
    reg = None
    log(f'cfg={args.cfg}\nteacher={teacher_ckpt}\nstudent={args.student_ckpt}\ninit={args.init_ckpt or "-"}\n'
        f'max_samples/bo={args.max_samples} | seed={args.seed}')
    train_names = list(cfg.DATASET.train_list) if args.train_sets == 'config' else \
        [s.strip() for s in args.train_sets.split(',') if s.strip()]
    test_names = [s.strip() for s in args.test_sets.split(',') if s.strip()]
    summary = {}
    fields = ['set', 'index', 'joint', 'e_T', 'e_S', 'e_lift', 'd', 'd_glob'] + (['e_S0', 'd0'] if init_s is not None else [])
    with (out_dir / 'samples.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, is_train in [(n_, True) for n_ in train_names] + [(n_, False) for n_ in test_names]:
            try:
                ds = get_train_dataset(name, None) if is_train else get_test_dataset(name, None)
            except Exception as e:
                log(f'[{name}] LOI tao dataset: {type(e).__name__}: {e}')
                continue
            reg = torch.as_tensor(ds.h36m_joint_regressor, dtype=torch.float32, device=DEVICE)
            tag = ('train/' if is_train else 'test/') + name
            if is_train:
                log(f'\n[{tag}] loader train (CO augmentation nhu luc KD)')
            J, Sm, idx = collect(tag, ds, is_train, teacher, student, init_s, reg)
            summary[tag] = report(tag, J, Sm, log, w.writerow, idx)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                          encoding='utf-8')
    log(f'\nDa luu: {out_dir}')


if __name__ == '__main__':
    main()
