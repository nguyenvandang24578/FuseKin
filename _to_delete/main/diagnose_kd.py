"""
diagnose_kd.py — Chan doan vi sao Student (S-hint / scratch) chua vuot Teacher T-hint [khong goi y]
====================================================================================================
Eval-only (KHONG train). Duyet split (mac dinh 3dpw-val, thu tu co dinh), chay Teacher T-hint + 1..n Student
trong nhieu dieu kien, roi in 9 muc:

  [1] Do chinh xac: MPJPE / PA-MPJPE / MPVPE theo DUNG giao thuc 14 khop cua dataset.evaluate (= so trong log)
  [2] Teacher: goi y z giup bao nhieu; z co dac thu tung mau khong (z cua mau khac);
      z_hat cua Student dua vao Teacher co giup khong
  [3] Student S-hint: z_hat vs z_T (MSE, R2, cos) + Student nhan z_T (oracle) / z = 0
  [4] Dung anh / dung joint: feature anh = TB batch, joint 3D = TB batch, joint 3D = GT
  [5] Quan he voi loi MotionBERT lift: tuong quan, % mau tot hon lift, 4 nhom theo loi lift
  [6] Dac trung Student vs Teacher (z=0 va z_T) tung tang: LN-MSE, cos, CKA; huong 'buoc goi y'
  [7] Attention khop -> anh: entropy, % attention roi quanh vi tri 2D cua khop
  [8] Loi theo tung khop
  [9] Goi y doc ket qua (nguong kinh nghiem, chi tham khao)

Cach chay (tu thu muc goc repo, can CUDA):
    python main/diagnose_kd.py --cfg ./config/train_student_hint_init_v2.yml \
        --student_ckpt experiment/student_hint_init_v2/checkpoint/best.pth.tar \
                       experiment/student_hint_z_v2/checkpoint/best.pth.tar
  - Teacher mac dinh = cfg.MODEL.TEACHER (T-hint v2). Ten Student lay tu thu muc experiment/<ten>/checkpoint.
  - Tu nhan dien Student thuong (vd student_v2 scratch) hay S-hint -> them duong dan vao --student_ckpt.
  - So voi tap train (khoang cach train/val): them --split 3dpw-train-noaug --max_samples 3000
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
import math
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Chan doan Student vs Teacher T-hint (FuseKin, eval-only)')
parser.add_argument('--cfg', type=str, required=True, help='yml S-hint (lay MODEL.TEACHER, MotionBERT, dataset)')
parser.add_argument('--student_ckpt', type=str, nargs='*', default=[], help='1..n checkpoint Student (S-hint hoac thuong)')
parser.add_argument('--student_name', type=str, nargs='*', default=[], help='ten hien thi (mac dinh: ten thu muc experiment)')
parser.add_argument('--teacher_ckpt', type=str, default='', help='mac dinh: cfg.MODEL.TEACHER')
parser.add_argument('--split', type=str, default='3dpw-val', help='3dpw-val | 3dpw | 3dpw-train-noaug ...')
parser.add_argument('--max_samples', type=int, default=0, help='0 = ca split; >0 = lay deu N mau (thu tu co dinh)')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--cka_samples', type=int, default=2048, help='so mau dung tinh CKA')
parser.add_argument('--attn_radius', type=float, default=0.3,
                    help='ban kinh quanh khop 2D, don vi toa do [-1, 1] (1 o cua luoi 8x8 = 0.25)')
parser.add_argument('--save', type=str, default='', help='file .npz luu loi tung mau (bo trong = khong luu)')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401
from models.ARTS import ARTS
from core.base import load_model_weights, _PickleShim
from utils.jotr_dataset import get_test_dataset

if not torch.cuda.is_available():
    print("\n[LOI] Can CUDA de chay script nay.\n")
    sys.exit(1)
DEVICE = torch.device('cuda')
torch.manual_seed(0)
np.random.seed(0)

LIFT = 'lift (MotionBERT)'


def hr(title):
    print('\n' + '=' * 100)
    print(title)
    print('=' * 100)


# ============================================================
# Nap model (giong content_ablation_test.py: kien truc v2 + loai goi y lay tu checkpoint)
# ============================================================
def inspect_ckpt(mode, ckpt_path):
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
    epoch = ckpt.get('epoch') if isinstance(ckpt, dict) else None
    keys = list(state.keys())
    has_hint_enc = any('hint_encoder.' in k for k in keys)
    has_hint_pred = any('hint_pred.' in k for k in keys)
    hint_dim = 0
    for k in keys:
        if k.endswith('hint_proj.weight'):
            hint_dim = int(state[k].shape[1])
    if has_hint_enc:
        kind = 'teacher'
    elif any('fusion.projector_student.' in k for k in keys):
        kind = 'student'
    elif any('fusion.joint_proj.' in k for k in keys):
        kind = 'teacher'
    else:
        kind = 'khong ro'
    pe2d_key = next((k for k in keys if k.endswith('fusion.pe2d.proj.weight')), None)
    arch = {
        'joint_pe2d': pe2d_key is not None,
        'pe2d_num_freq': int(state[pe2d_key].shape[1]) // 4 if pe2d_key else 6,
        'fusion_joint_self_attn': any('.joint_sa.' in k for k in keys),
        'pose_head_mode': 'per_joint' if any('per_joint_head.' in k for k in keys) else 'vposer',
        'pose_query_bone': any('bone_proj.' in k for k in keys),
        'shape_attn_mode': 'tokens' if any('fuse_shape_tok.' in k for k in keys) else 'global',
    }
    info = {'hint': 'teacher' if has_hint_enc else ('student' if has_hint_pred else None),
            'hint_dim': hint_dim, 'arch': arch, 'epoch': epoch}
    tag = f" T-hint k={hint_dim}" if has_hint_enc else (f" S-hint k={hint_dim}" if has_hint_pred else " (khong goi y)")
    print(f"  [{mode}] {ckpt_path} -> {kind}{tag} | epoch {epoch} | v2: "
          f"{all([arch['joint_pe2d'], arch['fusion_joint_self_attn'], arch['pose_head_mode'] == 'per_joint'])}")
    if kind != 'khong ro' and kind != mode:
        print(f"\n[LOI] Checkpoint {kind.upper()} dat vao vi tri {mode.upper()}: {ckpt_path}")
        sys.exit(1)
    return info


def build_and_load(mode, ckpt_path):
    info = inspect_ckpt(mode, ckpt_path)
    keys = ('name', 'teacher_lift_alpha_max', 'img_probe', 'teacher_hint', 'student_hint', 'hint_dim') \
        + tuple(info['arch'].keys())
    saved = {k: cfg.MODEL[k] for k in keys if k in cfg.MODEL}
    cfg.MODEL.name = mode
    for k, v in info['arch'].items():
        cfg.MODEL[k] = v
    if mode == 'teacher':
        cfg.MODEL.teacher_hint = info['hint'] == 'teacher'
        cfg.MODEL.teacher_lift_alpha_max = 0.0
        cfg.MODEL.img_probe = False
    else:
        cfg.MODEL.student_hint = info['hint'] == 'student'
    if info['hint_dim'] > 0:
        cfg.MODEL.hint_dim = info['hint_dim']
    model = ARTS(num_joint=17, embed_dim=cfg.MODEL.hpe_dim)
    for k in keys:
        if k in saved:
            cfg.MODEL[k] = saved[k]
        elif k in cfg.MODEL:
            del cfg.MODEL[k]
    model = load_model_weights(model.to(DEVICE), ckpt_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, info


def student_label(path, i):
    if i < len(args.student_name):
        return args.student_name[i]
    parts = os.path.normpath(path).split(os.sep)
    name = parts[parts.index('checkpoint') - 1] if 'checkpoint' in parts[1:] else os.path.splitext(parts[-1])[0]
    if name.startswith('student_'):
        name = name[len('student_'):]
    return 'S:' + name


# ============================================================
# Do loi (giong PW3D.evaluate: 14 khop H36M root-rel, PA co ti le, MPVPE can theo khop 0 SMPL)
# ============================================================
def rigid_align_batch(A, B):
    """Procrustes co ti le (= rigid_align numpy cua dataset), canh A (N, J, 3) vao B (N, J, 3)."""
    A64, B64 = A.double(), B.double()
    muA, muB = A64.mean(1, keepdim=True), B64.mean(1, keepdim=True)
    A0, B0 = A64 - muA, B64 - muB
    H = A0.transpose(1, 2) @ B0
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(1, 2)
    d = torch.where(torch.det(V @ U.transpose(1, 2)) < 0, -1.0, 1.0).to(S.dtype)
    D = torch.ones_like(S)
    D[:, -1] = d
    R = V @ torch.diag_embed(D) @ U.transpose(1, 2)
    scale = (S * D).sum(-1) / (A0 ** 2).sum(dim=(1, 2)).clamp_min(1e-12)
    return (scale[:, None, None] * (A0 @ R.transpose(1, 2)) + muB).to(A.dtype)


class Metric:
    def __init__(self, ds):
        jr = ds.joint_regressor
        jr = jr.toarray() if hasattr(jr, 'toarray') else np.asarray(jr)
        self.reg = torch.as_tensor(np.asarray(ds.h36m_joint_regressor), dtype=torch.float32, device=DEVICE)  # (17, V)
        self.reg_smpl0 = torch.as_tensor(jr[0], dtype=torch.float32, device=DEVICE)                            # (V,)
        self.root = int(ds.h36m_root_joint_idx)
        self.eval_idx = list(ds.h36m_eval_joint)
        self.names = [ds.h36m_joints_name[i] for i in self.eval_idx]

    def h36m17(self, mesh):
        j = torch.einsum('jv,bvc->bjc', self.reg, mesh)
        return j - j[:, self.root:self.root + 1]

    def _joint_err(self, j17_pred, j17_gt):
        p, g = j17_pred[:, self.eval_idx], j17_gt[:, self.eval_idx]
        pj = torch.norm(p - g, dim=-1) * 1000.0                                   # (B, 14)
        pa = torch.norm(rigid_align_batch(p, g) - g, dim=-1).mean(-1) * 1000.0    # (B,)
        return pj, pa

    def from_mesh(self, mesh_pred, mesh_gt, j17_gt):
        pj, pa = self._joint_err(self.h36m17(mesh_pred), j17_gt)
        mp = mesh_pred - torch.einsum('v,bvc->bc', self.reg_smpl0, mesh_pred)[:, None]
        mg = mesh_gt - torch.einsum('v,bvc->bc', self.reg_smpl0, mesh_gt)[:, None]
        mpvpe = torch.norm(mp - mg, dim=-1).mean(-1) * 1000.0
        return {'mpjpe': pj.mean(-1), 'pa': pa, 'mpvpe': mpvpe, 'pj': pj}

    def from_joints(self, j17_pred, j17_gt):
        pj, pa = self._joint_err(j17_pred - j17_pred[:, self.root:self.root + 1], j17_gt)
        return {'mpjpe': pj.mean(-1), 'pa': pa, 'mpvpe': torch.full_like(pa, float('nan')), 'pj': pj}


# ============================================================
# Tich luy
# ============================================================
class Acc:
    """Trung binh co trong so: key -> [tong, so luong]."""
    def __init__(self):
        self.s = {}

    def add(self, key, val, w=None):
        val = val.detach().float()
        if w is None:
            tot, n = float(val.sum()), float(val.numel())
        else:
            w = w.float()
            tot, n = float((val * w).sum()), float(w.sum())
        a = self.s.setdefault(key, [0.0, 0.0])
        a[0] += tot
        a[1] += n

    def get(self, key):
        a = self.s.get(key)
        return a[0] / a[1] if a and a[1] > 0 else float('nan')


def fwd(net, joints, feats, pe_kw, hint_z=None, features=False):
    """Goi smpl_model. S-hint + hint_z=None -> fusion 2 luot (z_hat) nhu luc test; hint_z != None -> dung z do."""
    kw = dict(pe_kw)
    if hint_z is not None:
        kw['hint_z'] = hint_z
    return net.smpl_model(joints=joints, img_feats=feats, is_train=False, return_features=features, **kw)


def feat_points(out):
    """Cac diem lay dac trung (da LayerNorm theo kenh, khong affine - giong KD feat): dict ten -> (B, N, C)."""
    d = {'joint_in': out.get('feat_joint_in')}
    layers = out.get('feat_layers') or []
    for i, l in enumerate(layers):
        d[f'fus_L{i + 1}'] = l['joint']
    if layers:
        d[f'img_L{len(layers)}'] = layers[-1]['img']
    d['hyper'] = out.get('feat_hyper')
    d['pose'] = out.get('feat_pose')
    d['global'] = out['feat_global'].unsqueeze(1)
    return {k: F.layer_norm(v.float(), (v.shape[-1],)) for k, v in d.items() if v is not None}


def compare_feats(acc, tag, fa, fb):
    for k in fa:
        if k in fb:
            x, y = fa[k], fb[k]
            acc.add((tag, k, 'lnmse'), ((x - y) ** 2).mean(dim=(1, 2)))
            acc.add((tag, k, 'cos'), F.cosine_similarity(x, y, dim=-1).mean(1))


def hint_direction(acc, src, fS, fT1, fT0):
    """Buoc goi y cua Teacher dT = f(T|zT) - f(T|z0); Student dS = f(S) - f(T|z0).
    cos(dS, dT) > 0 va proj ~ 1: Student di dung huong ma goi y day Teacher."""
    for k in fS:
        if k in fT1 and k in fT0:
            dT = (fT1[k] - fT0[k]).mean(1)
            dS = (fS[k] - fT0[k]).mean(1)
            acc.add((src, k, 'dir_cos'), F.cosine_similarity(dS, dT, dim=-1, eps=1e-8))
            acc.add((src, k, 'dir_proj'), (dS * dT).sum(-1) / (dT * dT).sum(-1).clamp_min(1e-8))


def attn_stats(acc, src, out, j2d, jmask, cells, radius):
    layers = out.get('feat_layers') or []
    B, J = j2d.shape[:2]
    xy = j2d[..., :2].float()
    m = (jmask.reshape(B, J) > 0).float()                       # chi khop co 2D hop le
    near = (torch.cdist(xy, cells.expand(B, -1, -1)) <= radius).float()   # (B, J, HW)
    base = near.mean(-1)                                       # attention deu -> ti le o gan khop
    for li, l in enumerate(layers):
        a = l['attn_joint_to_img'].float()                     # (B, h, J, HW)
        hw = a.shape[-1]
        ent = -(a * a.clamp_min(1e-12).log()).sum(-1) / math.log(hw)     # 1 = deu, 0 = 1 o
        am = a.mean(1)                                         # TB cac head (B, J, HW)
        cen = am @ cells[0]                                    # tam attention (B, J, 2)
        peak = cells[0][am.argmax(-1)]                         # o attention lon nhat (B, J, 2)
        L = f'L{li + 1}'
        acc.add((src, L, 'ent'), ent.mean(1), m)
        acc.add((src, L, 'mass'), (am * near).sum(-1), m)
        acc.add((src, L, 'mass_best'), (a * near.unsqueeze(1)).sum(-1).max(1).values, m)
        acc.add((src, L, 'base'), base, m)
        acc.add((src, L, 'd_cen'), (cen - xy).norm(dim=-1), m)
        acc.add((src, L, 'd_peak'), (peak - xy).norm(dim=-1), m)
        acc.add((src, L, 'd_center'), xy.norm(dim=-1), m)


def linear_cka(X, Y):
    X = X.double() - X.double().mean(0, keepdim=True)
    Y = Y.double() - Y.double().mean(0, keepdim=True)
    num = (Y.t() @ X).norm() ** 2
    den = (X.t() @ X).norm() * (Y.t() @ Y).norm()
    return float(num / den.clamp_min(1e-12))


def pdiff(a, b):
    """TB va khoang tin cay 95% (theo mau doc lap -> lac quan voi 3DPW) cua a - b theo tung mau."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    ci = 1.96 * d.std(ddof=1) / np.sqrt(d.size) if d.size > 1 else float('nan')
    return float(d.mean()), float(ci)


def nmean(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float('nan')


# ============================================================
def main():
    teacher_ckpt = args.teacher_ckpt or cfg.MODEL.get('TEACHER', '')
    if not teacher_ckpt:
        print('[LOI] Can --teacher_ckpt hoac cfg.MODEL.TEACHER (checkpoint T-hint)')
        sys.exit(1)

    hr('Nap checkpoint')
    teacher, t_info = build_and_load('teacher', teacher_ckpt)
    if t_info['hint'] != 'teacher':
        print('[LOI] Teacher phai la T-hint (co hint_encoder).')
        sys.exit(1)
    students = []
    for i, p in enumerate(args.student_ckpt):
        net, info = build_and_load('student', p)
        students.append({'name': student_label(p, i), 'net': net, 'hint': info['hint'] == 'student',
                         'epoch': info['epoch']})
    for s in students:
        s['main'] = s['name'] + ('|zhat' if s['hint'] else '')
    print('  Student: ' + ', '.join(f"{s['name']} ({'S-hint' if s['hint'] else 'thuong'}, ep {s['epoch']})"
                                    for s in students))

    hr(f'Du lieu: {args.split}')
    ds = get_test_dataset(args.split, None)
    data = ds
    if 0 < args.max_samples < len(ds):
        idx = np.unique(np.linspace(0, len(ds) - 1, args.max_samples).round().astype(int))
        data = Subset(ds, idx.tolist())
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"  so mau: {len(data)} / {len(ds)} | batch {args.batch_size}")
    metric = Metric(ds)

    results = {}                        # dieu kien -> {mpjpe, pa, mpvpe, pj}: list tensor theo batch
    facc, aacc = Acc(), Acc()           # dac trung / attention
    cka = {}                            # nguon -> diem -> list (B, C)
    n_cka = 0
    zstore = {s['name']: {'zh': [], 'zT': []} for s in students if s['hint']}
    ablate = {}                         # dieu kien chinh -> {loai ablation: ten dieu kien}
    point_order = []
    cells = None
    checked = False

    def rec(name, gt_mesh, gt17, out=None, joints17=None):
        r = metric.from_mesh(out['smpl_mesh_cam'], gt_mesh, gt17) if out is not None \
            else metric.from_joints(joints17, gt17)
        acc = results.setdefault(name, {'mpjpe': [], 'pa': [], 'mpvpe': [], 'pj': []})
        for k in acc:
            acc[k].append(r[k].detach().float().cpu())

    hr('Dang chay...')
    with torch.no_grad():
        for bi, (inputs, targets, meta) in enumerate(loader):
            img = inputs['img'].to(DEVICE).float()
            j2d = inputs['joints'].to(DEVICE).float()
            jm = inputs['joints_mask'].to(DEVICE).float()
            gt_mesh = targets['smpl_mesh_cam'].to(DEVICE).float()
            gt17 = metric.h36m17(gt_mesh)
            B = img.shape[0]

            # ---------------- Teacher T-hint ----------------
            fT, _ = teacher.get_image_features(img)
            if cells is None:
                h, w = fT.shape[-2:]
                cells = teacher.smpl_model.fusion._cell_centers(h, w, DEVICE, torch.float32)   # (1, HW, 2)
            liftT = teacher.lift_2d_to_3d(j2d, joints_mask=jm)
            liftT = (liftT - liftT[:, 0:1]).float()
            pkT = teacher._pe2d_kwargs(j2d, jm)
            zT = teacher.smpl_model.hint_encoder(gt17, liftT)                  # eval: khong nhieu
            z0 = torch.zeros_like(zT)
            rec(LIFT, gt_mesh, gt17, joints17=liftT)

            oT1 = fwd(teacher, liftT, fT, pkT, zT, features=True)
            oT0 = fwd(teacher, liftT, fT, pkT, z0, features=True)
            rec('T|zT', gt_mesh, gt17, oT1)
            rec('T|z0', gt_mesh, gt17, oT0)
            rec('T|zT-mau-khac', gt_mesh, gt17, fwd(teacher, liftT, fT, pkT, zT.roll(1, 0)))
            fT_mean = fT.mean(0, keepdim=True).expand_as(fT).contiguous()
            j_mean = liftT.mean(0, keepdim=True).expand_as(liftT).contiguous()
            rec('T|zT|anh-TB', gt_mesh, gt17, fwd(teacher, liftT, fT_mean, pkT, zT))
            rec('T|z0|anh-TB', gt_mesh, gt17, fwd(teacher, liftT, fT_mean, pkT, z0))
            rec('T|z0|joint-TB', gt_mesh, gt17, fwd(teacher, j_mean, fT, pkT, z0))
            rec('T|z0|joint-GT', gt_mesh, gt17, fwd(teacher, gt17, fT, pkT, z0))
            ablate['T|zT'] = {'anh-TB': 'T|zT|anh-TB'}
            ablate['T|z0'] = {'anh-TB': 'T|z0|anh-TB', 'joint-TB': 'T|z0|joint-TB', 'joint-GT': 'T|z0|joint-GT'}

            fT1, fT0 = feat_points(oT1), feat_points(oT0)
            if not point_order:
                point_order = list(fT0.keys())
            compare_feats(facc, ('T|zT', 'T|z0'), fT1, fT0)
            attn_stats(aacc, 'T|zT', oT1, j2d, jm, cells, args.attn_radius)
            attn_stats(aacc, 'T|z0', oT0, j2d, jm, cells, args.attn_radius)
            take_cka = n_cka < args.cka_samples
            if take_cka:
                for src, f in (('T|zT', fT1), ('T|z0', fT0)):
                    for k, v in f.items():
                        cka.setdefault(src, {}).setdefault(k, []).append(v.mean(1).cpu())

            # Kiem tra 1 lan: do loi cua script == dataset.evaluate (so trong log)
            if not checked:
                checked = True
                outs = [{'smpl_mesh_cam': m, 'smpl_mesh_cam_target': g}
                        for m, g in zip(oT0['smpl_mesh_cam'].cpu().numpy(), gt_mesh.cpu().numpy())]
                try:
                    ref = ds.evaluate(outs, 0)
                    mine = metric.from_mesh(oT0['smpl_mesh_cam'], gt_mesh, gt17)
                    print('  Kiem tra do loi (batch 1, T|z0): script / dataset.evaluate = '
                          f"MPJPE {mine['mpjpe'].mean():.3f}/{np.mean(ref['mpjpe']):.3f} | "
                          f"PA {mine['pa'].mean():.3f}/{np.mean(ref['pa_mpjpe']):.3f} | "
                          f"MPVPE {mine['mpvpe'].mean():.3f}/{np.mean(ref['mpvpe']):.3f}")
                except Exception as e:   # khong chan chay chinh
                    print(f'  (bo qua kiem tra voi dataset.evaluate: {e})')

            # ---------------- Student ----------------
            for s in students:
                net, nm = s['net'], s['name']
                fS, _ = net.get_image_features(img)
                liftS = net.lift_2d_to_3d(j2d, joints_mask=jm)
                liftS = (liftS - liftS[:, 0:1]).float()
                pkS = net._pe2d_kwargs(j2d, jm)
                oS = fwd(net, liftS, fS, pkS, None, features=True)          # S-hint: 2 luot, z_hat
                rec(s['main'], gt_mesh, gt17, oS)
                fS_mean = fS.mean(0, keepdim=True).expand_as(fS).contiguous()
                jS_mean = liftS.mean(0, keepdim=True).expand_as(liftS).contiguous()
                rec(f'{nm}|anh-TB', gt_mesh, gt17, fwd(net, liftS, fS_mean, pkS))
                rec(f'{nm}|joint-TB', gt_mesh, gt17, fwd(net, jS_mean, fS, pkS))
                rec(f'{nm}|joint-GT', gt_mesh, gt17, fwd(net, gt17, fS, pkS))
                ablate[s['main']] = {'anh-TB': f'{nm}|anh-TB', 'joint-TB': f'{nm}|joint-TB',
                                     'joint-GT': f'{nm}|joint-GT'}
                if s['hint']:
                    zh = oS['hint_pred']
                    zstore[nm]['zh'].append(zh.float().cpu())
                    zstore[nm]['zT'].append(zT.float().cpu())
                    rec(f'{nm}|z0', gt_mesh, gt17, fwd(net, liftS, fS, pkS, z0))
                    rec(f'{nm}|zT', gt_mesh, gt17, fwd(net, liftS, fS, pkS, zT))
                    rec(f'T|zhat({nm})', gt_mesh, gt17, fwd(teacher, liftT, fT, pkT, zh))

                fSp = feat_points(oS)
                compare_feats(facc, (s['main'], 'T|z0'), fSp, fT0)
                compare_feats(facc, (s['main'], 'T|zT'), fSp, fT1)
                hint_direction(facc, s['main'], fSp, fT1, fT0)
                attn_stats(aacc, s['main'], oS, j2d, jm, cells, args.attn_radius)
                if take_cka:
                    for k, v in fSp.items():
                        cka.setdefault(s['main'], {}).setdefault(k, []).append(v.mean(1).cpu())
            if take_cka:
                n_cka += B

            if (bi + 1) % 20 == 0:
                print(f'  ... {bi + 1}/{len(loader)} batch')

    # ============================================================
    # Tong hop
    # ============================================================
    R = {name: {k: torch.cat(v).numpy() for k, v in d.items()} for name, d in results.items()}
    N = len(R[LIFT]['mpjpe'])
    lift = R[LIFT]['mpjpe']
    M = {name: float(r['mpjpe'].mean()) for name, r in R.items()}
    mains = ['T|zT', 'T|z0'] + [s['main'] for s in students]

    # ---------------- [1] ----------------
    hr(f'[1] DO CHINH XAC tren {args.split} (n={N}) - 14 khop, giong dataset.evaluate (= so trong log)')
    print(f"  {'dieu kien':34s} {'MPJPE':>7s} {'PA':>7s} {'MPVPE':>7s} {'so voi lift':>12s}")
    for name, r in R.items():
        mpv = nmean(r['mpvpe'])
        mpv_s = f'{mpv:7.2f}' if np.isfinite(mpv) else f"{'-':>7s}"
        print(f"  {name:34s} {r['mpjpe'].mean():7.2f} {r['pa'].mean():7.2f} {mpv_s} {M[name] - lift.mean():+12.2f}")
    print("  Chu thich: zT = goi y tu GT (Teacher), z0 = khong goi y, zhat = Student tu doan;\n"
          "             anh-TB = feature anh thay bang TB batch (mat thong tin anh rieng tung mau);\n"
          "             joint-TB = joint 3D vao = TB batch; joint-GT = joint 3D vao = GT (2D van la OpenPose);\n"
          "             T|zT-mau-khac = Teacher nhan z cua mau khac trong batch; T|zhat(S) = Teacher nhan z_hat cua S.")

    # ---------------- [2] ----------------
    hr('[2] TEACHER: goi y z')
    d, ci = pdiff(R['T|z0']['mpjpe'], R['T|zT']['mpjpe'])
    print(f"  z_T giup Teacher:                {d:+7.2f} +/- {ci:.2f} mm   (T|z0 - T|zT)")
    d, ci = pdiff(R['T|zT-mau-khac']['mpjpe'], R['T|z0']['mpjpe'])
    print(f"  z cua mau khac so voi z = 0:     {d:+7.2f} +/- {ci:.2f} mm   (> 0: z dac thu tung mau, z sai con hai)")
    for s in students:
        if s['hint']:
            d, ci = pdiff(R[f"T|zhat({s['name']})"]['mpjpe'], R['T|z0']['mpjpe'])
            print(f"  Teacher nhan z_hat cua {s['name']:18s} {d:+7.2f} +/- {ci:.2f} mm so voi z = 0 "
                  f"(< 0: z_hat co thong tin huu ich)")

    # ---------------- [3] ----------------
    zinfo = {}
    if zstore:
        hr('[3] STUDENT S-HINT: z_hat (Student tu doan) vs z_T (Teacher tu GT)')
        for nm, zz in zstore.items():
            zh, zt = torch.cat(zz['zh']).double(), torch.cat(zz['zT']).double()
            err = (zh - zt) ** 2
            var = zt.var(0, unbiased=False)
            r2_dim = 1 - err.mean(0) / var.clamp_min(1e-12)
            r2 = float(1 - err.sum() / ((zt - zt.mean(0)) ** 2).sum().clamp_min(1e-12))
            zc, tc = zh - zh.mean(0), zt - zt.mean(0)
            corr_dim = (zc * tc).sum(0) / (zc.norm(dim=0) * tc.norm(dim=0)).clamp_min(1e-12)
            cosv = float(F.cosine_similarity(zh, zt, dim=-1).mean())
            zinfo[nm] = {'r2': r2}
            print(f"  {nm}")
            print(f"    MSE(z_hat, z_T) = {float(err.mean()):.4f} | MSE(0, z_T) = {float((zt ** 2).mean()):.4f} "
                  f"| R2 gop = {r2:+.3f} | R2 TB tung chieu = {float(r2_dim.mean()):+.3f} "
                  f"| tuong quan TB tung chieu = {float(corr_dim.mean()):+.3f} | cos = {cosv:+.3f}")
            print(f"    do lon RMS: |z_hat| = {float(zh.pow(2).mean().sqrt()):.3f} | |z_T| = {float(zt.pow(2).mean().sqrt()):.3f}")
            main = f'{nm}|zhat'
            d1, c1 = pdiff(R[main]['mpjpe'], R[f'{nm}|zT']['mpjpe'])
            d2, c2 = pdiff(R[f'{nm}|z0']['mpjpe'], R[main]['mpjpe'])
            print(f"    Student nhan z_T (oracle) tot hon z_hat: {d1:+7.2f} +/- {c1:.2f} mm  (lon -> Student BIET dung z)")
            print(f"    z_hat giup Student so voi z = 0:         {d2:+7.2f} +/- {c2:.2f} mm")
            zinfo[nm].update(oracle=d1, zhat_gain=d2)

    # ---------------- [4] ----------------
    hr('[4] DUNG ANH / DUNG JOINT (so mm te them, > 0 = phu thuoc vao thanh phan bi thay)')
    print(f"  {'dieu kien':26s} {'MPJPE':>7s} {'anh-TB':>15s} {'joint-TB':>9s} {'joint-GT':>9s}")
    for c in mains:
        ab = ablate.get(c, {})
        row = f"  {c:26s} {M[c]:7.2f}"
        if 'anh-TB' in ab:
            dj = M[ab['anh-TB']] - M[c]
            dv = nmean(R[ab['anh-TB']]['mpvpe']) - nmean(R[c]['mpvpe'])
            row += f" {dj:+7.2f}/{dv:+6.2f}v"
        else:
            row += f" {'-':>15s}"
        for k in ('joint-TB', 'joint-GT'):
            row += f" {M[ab[k]] - M[c]:+9.2f}" if k in ab else f" {'-':>9s}"
        print(row)
    print("  anh-TB: MPJPE/MPVPE(v). joint-GT < 0 = neu joint vao dung GT thi tot hon bay nhieu mm.")

    # ---------------- [5] ----------------
    hr(f'[5] QUAN HE VOI LOI MOTIONBERT LIFT (lift TB {lift.mean():.2f} mm)')
    q = np.quantile(lift, [0, 0.25, 0.5, 0.75, 1.0])
    bins = [(lift >= q[i]) & ((lift < q[i + 1]) if i < 3 else (lift <= q[i + 1])) for i in range(4)]
    print(f"  {'dieu kien':26s} {'corr':>6s} {'he so':>6s} {'%tot hon lift':>14s}   4 nhom theo loi lift: lift -> model (mm)")
    for c in mains:
        y = R[c]['mpjpe']
        corr = float(np.corrcoef(lift, y)[0, 1])
        slope = float(np.polyfit(lift, y, 1)[0])
        better = float((y < lift).mean()) * 100
        bs = '  '.join(f'{lift[b].mean():5.1f}->{y[b].mean():5.1f}' for b in bins)
        print(f"  {c:26s} {corr:6.3f} {slope:6.3f} {better:13.1f}%   {bs}")
    print("  corr/he so cao (~1) = loi model bam theo loi lift (chu yeu chep lift); nhom lift te nhat cho biet\n"
          "  model co SUA duoc lift sai nhieu khong.")

    # ---------------- [6] ----------------
    hr(f'[6] DAC TRUNG (LayerNorm theo kenh) - CKA tren {min(n_cka, N)} mau')

    def cka_of(a, b, k):
        if a in cka and b in cka and k in cka[a] and k in cka[b]:
            return linear_cka(torch.cat(cka[a][k]), torch.cat(cka[b][k]))
        return float('nan')

    print('  Teacher: goi y lam thay doi dac trung o dau (T|zT vs T|z0)')
    print(f"    {'diem':10s} {'LN-MSE':>8s} {'cos':>7s} {'CKA':>7s}")
    for k in point_order:
        print(f"    {k:10s} {facc.get((('T|zT', 'T|z0'), k, 'lnmse')):8.4f} "
              f"{facc.get((('T|zT', 'T|z0'), k, 'cos')):7.4f} {cka_of('T|zT', 'T|z0', k):7.4f}")
    for s in students:
        src = s['main']
        print(f"\n  {src}: so voi Teacher [z=0] va [z_T]; huong = Student co di theo 'buoc goi y' cua Teacher khong")
        print(f"    {'diem':10s} {'MSE/z0':>8s} {'cos/z0':>7s} {'CKA/z0':>7s} {'MSE/zT':>8s} {'cos/zT':>7s} "
              f"{'CKA/zT':>7s} {'huong cos':>9s} {'huong ti le':>11s}")
        for k in point_order:
            print(f"    {k:10s} {facc.get(((src, 'T|z0'), k, 'lnmse')):8.4f} {facc.get(((src, 'T|z0'), k, 'cos')):7.4f} "
                  f"{cka_of(src, 'T|z0', k):7.4f} {facc.get(((src, 'T|zT'), k, 'lnmse')):8.4f} "
                  f"{facc.get(((src, 'T|zT'), k, 'cos')):7.4f} {cka_of(src, 'T|zT', k):7.4f} "
                  f"{facc.get((src, k, 'dir_cos')):9.3f} {facc.get((src, k, 'dir_proj')):11.3f}")
    print("\n  MSE = LN-MSE (cung dang KD feat trong log). huong cos ~0 / ti le ~0: Student khong tien ve phia\n"
          "  Teacher co goi y; ti le ~1: Student da di het 'buoc goi y'.")

    # ---------------- [7] ----------------
    hr(f'[7] ATTENTION khop -> anh (ban kinh {args.attn_radius} quanh vi tri 2D OpenPose cua khop, chi khop hop le)')
    print(f"  {'nguon':26s} {'tang':4s} {'entropy':>8s} {'% gan khop':>11s} {'deu':>6s} {'gap':>5s} "
          f"{'head tot':>9s} {'d tam':>7s} {'d dinh':>7s} {'d giua':>7s}")
    for src in mains:
        for li in range(1, 10):
            L = f'L{li}'
            if (src, L, 'ent') not in aacc.s:
                break
            mass, base = aacc.get((src, L, 'mass')), aacc.get((src, L, 'base'))
            print(f"  {src:26s} {L:4s} {aacc.get((src, L, 'ent')):8.3f} {100 * mass:10.1f}% {100 * base:5.1f}% "
                  f"{mass / max(base, 1e-8):5.2f} {100 * aacc.get((src, L, 'mass_best')):8.1f}% "
                  f"{aacc.get((src, L, 'd_cen')):7.3f} {aacc.get((src, L, 'd_peak')):7.3f} "
                  f"{aacc.get((src, L, 'd_center')):7.3f}")
    print("  entropy 1 = attention deu ca anh. '% gan khop' (TB cac head) so voi 'deu' (attention deu); gap = ti so.\n"
          "  head tot = head tap trung quanh khop nhat. d tam / d dinh = khoang cach tam / o dinh attention toi khop;\n"
          "  d giua = khoang cach tu khop toi giua anh (moc so sanh).")

    # ---------------- [8] ----------------
    hr('[8] LOI THEO KHOP (MPJPE mm)')
    cols = [LIFT] + mains
    labels = ['lift'] + [c if len(c) <= 10 else f'#{i}' for i, c in enumerate(mains)]
    legend = [f'#{i} = {c}' for i, c in enumerate(mains) if len(c) > 10]
    print('  ' + f"{'khop':12s}" + ''.join(f'{lb:>11s}' for lb in labels))
    for j, jn in enumerate(metric.names):
        print('  ' + f'{jn:12s}' + ''.join(f"{R[c]['pj'][:, j].mean():11.1f}" for c in cols))
    print('  ' + f"{'TB':12s}" + ''.join(f"{R[c]['mpjpe'].mean():11.1f}" for c in cols))
    if legend:
        print('  ' + ' | '.join(legend))

    # ---------------- [9] ----------------
    hr('[9] GOI Y DOC KET QUA (nguong kinh nghiem - chi tham khao, xem so lieu o tren)')
    g = M['T|z0'] - M['T|zT']
    print(f"  - Teacher: goi y giup {g:.1f} mm -> " + ('z mang nhieu thong tin GT.' if g >= 10 else 'z it thong tin.'))
    print(f"  - Lift MotionBERT tren {args.split}: {lift.mean():.1f} mm; Teacher [z=0] {M['T|z0'] - lift.mean():+.1f} mm so voi lift.")
    for s in students:
        nm, main = s['name'], s['main']
        gap = M[main] - M['T|z0']
        print(f"  - {nm}: {M[main]:.2f} mm, so voi Teacher [z=0] {gap:+.2f} mm"
              + (' -> CHUA vuot diem xuat phat.' if gap > -0.5 else ' -> da vuot Teacher [z=0].'))
        if s['hint'] and nm in zinfo:
            zi = zinfo[nm]
            if zi['oracle'] >= 5:
                print(f"      cho z_T dung: tot hon {zi['oracle']:.1f} mm -> Student BIET dung z; nut that la DOAN z "
                      f"(R2 = {zi['r2']:+.2f}).")
            elif zi['oracle'] >= 2:
                print(f"      cho z_T dung: tot hon {zi['oracle']:.1f} mm -> Student chi dung z mot phan.")
            else:
                print(f"      cho z_T dung: chi {zi['oracle']:+.1f} mm -> Student gan nhu KHONG dung z (ke ca z dung).")
            tz = M[f'T|zhat({nm})'] - M['T|z0']
            print(f"      z_hat dua vao Teacher: {tz:+.1f} mm so voi z = 0 -> "
                  + ('z_hat CO thong tin huu ich.' if tz <= -1 else 'z_hat KHONG mang thong tin huu ich.'))
            if zi['r2'] < 0.1:
                print(f"      R2 = {zi['r2']:+.2f} < 0.1 -> z_hat gan nhu khong doan duoc z_T tren {args.split}.")
        di = M[f'{nm}|anh-TB'] - M[main]
        dti = M['T|z0|anh-TB'] - M['T|z0']
        print(f"      mat anh rieng tung mau: te them {di:+.1f} mm (Teacher [z=0]: {dti:+.1f}) -> "
              + ('gan nhu KHONG dung anh.' if di < 3 else 'CO dung anh.'))
        dg = M[main] - M[f'{nm}|joint-GT']
        print(f"      joint vao = GT: tot hon {dg:.1f} mm -> " + ('chat luong joint vao la nut that lon.' if dg >= 10
                                                                   else 'joint vao khong phai nut that chinh.'))
        corr = float(np.corrcoef(lift, R[main]['mpjpe'])[0, 1])
        if corr >= 0.8:
            print(f"      tuong quan voi loi lift {corr:.2f} -> loi Student bam sat loi MotionBERT.")
    if args.split != '3dpw-train-noaug':
        print("  - Chay them --split 3dpw-train-noaug --max_samples 3000 de so khoang cach train/val "
              "(z_hat R2, loi lift, dung anh).")

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        pack = {f'mpjpe::{k}': v['mpjpe'] for k, v in R.items()}
        pack.update({f'pa::{k}': v['pa'] for k, v in R.items()})
        for nm, zz in zstore.items():
            pack[f'zhat::{nm}'] = torch.cat(zz['zh']).numpy()
            pack[f'zT::{nm}'] = torch.cat(zz['zT']).numpy()
        np.savez(args.save, **pack)
        print(f'\n  Da luu loi tung mau: {args.save}')


if __name__ == '__main__':
    main()
