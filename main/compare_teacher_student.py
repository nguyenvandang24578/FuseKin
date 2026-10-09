"""
compare_teacher_student.py - So sanh "khong gian hoc" giua Teacher va Student (FuseKin)
===========================================================================================

Dung 2 checkpoint (Teacher + Student) chay tren TAP VAL THAT de xem Student hoc duoc gi so voi
Teacher: ve ket qua cuoi (MPJPE) lan ve khong gian dac trung ben trong model.

Cac muc in ra (va luu vao results.json):

  [1] MPJPE (mm) cua 3 moc: MotionBERT lift (can duoi), Student, Teacher (can tren).
      Co them: trung vi, khoang tin cay 95% cua hieu cap (paired), ti le mau Student tot hon MotionBERT.
  [2] MPJPE tung khop (17 khop H36M) + khoang cach dac trung tung khop.
  [3] Do giong nhau o DAU RA: MPJPE giua khop du doan cua Student va cua Teacher.
  [4] Khoang cach dac trung Student-Teacher tai 3 diem noi (cung cong thuc KD luc train)
      + linear CKA (do giong nhau ve HINH HOC khong gian dac trung, bat bien phep xoay), kem gia tri nen.
  [5] Tung block fusion: khoang cach dac trung, entropy attention, KL(Teacher || Student) cua attention.
  [6] Shuffle test: tron ANH giua cac mau, tron JOINT giua cac mau -> model dua vao nguon nao.
  [7] Hinh: PCA feat_global (3 panel), MPJPE/khoang cach tung khop, tung block.

Cach chay (tu thu muc goc repo):
    python main/compare_teacher_student.py \
        --cfg <config_cua_student> \
        --student_ckpt <checkpoint_student>.pth.tar \
        [--teacher_ckpt <checkpoint_teacher>.pth.tar]   # mac dinh: cfg.MODEL.TEACHER
        [--max_batches 30] [--batch_size 16] [--workers 0] [--out_dir ./compare_out] [--save_feats]

Can CUDA. Luu y thong ke: 3DPW co tuong quan giua cac frame lien tiep nen khoang tin cay tinh theo
"mau doc lap" la LAC QUAN (thuc te rong hon); dung chung nhu thuoc do tuong doi.
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

H36M_JOINTS = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck',
               'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')

parser = argparse.ArgumentParser(description='So sanh khong gian hoc Teacher vs Student (FuseKin)')
parser.add_argument('--cfg', type=str, required=True, help='config yaml (dung config cua Student)')
parser.add_argument('--student_ckpt', type=str, required=True, help='checkpoint cua Student (.pth.tar)')
parser.add_argument('--teacher_ckpt', type=str, default='', help='checkpoint cua Teacher (mac dinh: cfg.MODEL.TEACHER)')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_batches', type=int, default=30, help='so batch val toi da se duyet qua')
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--out_dir', type=str, default='./compare_out')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save_feats', action='store_true', help='luu dac trung + loi tung mau ra features.npz')
parser.add_argument('--improve_thresh', type=float, default=5.0,
                    help='(mm) nguong de goi y "Student cai thien it so voi MotionBERT"')
parser.add_argument('--shuffle_thresh', type=float, default=3.0,
                    help='(mm) nguong de goi y "model bo qua mot nguon dau vao" khi tron anh/joint')
args = parser.parse_args()

from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401
import models
from models.ARTS import ARTS
from core.base import load_model_weights
from utils.jotr_dataset import get_test_dataset

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not torch.cuda.is_available():
    print("\n[LOI] Can CUDA de chay script nay (Vposer/SMPL layer trong repo goi .cuda() "
          "cung trong code).\n")
    sys.exit(1)

DEVICE = torch.device('cuda')
os.makedirs(args.out_dir, exist_ok=True)


def hr(title=''):
    print('\n' + '=' * 78)
    if title:
        print(title)
        print('=' * 78)


# --------------------------------------------------------------------------
# Build + load 2 model
# --------------------------------------------------------------------------
def build_and_load(mode, ckpt_path):
    old_mode = cfg.MODEL.name
    cfg.MODEL.name = mode
    model = ARTS(num_joint=17, embed_dim=cfg.MODEL.hpe_dim)
    cfg.MODEL.name = old_mode

    model = model.to(DEVICE)
    model = load_model_weights(model, ckpt_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# --------------------------------------------------------------------------
# Helpers hinh hoc / dac trung (torch)
# --------------------------------------------------------------------------
def h36m_from_mesh(mesh, regressor):
    """mesh: (B,6890,3) - tuyet doi hay root-relative deu duoc, ham tu tru pelvis (idx 0) sau khi regress."""
    j = torch.einsum('jk,bkc->bjc', regressor, mesh)
    return j - j[:, 0:1, :]


def per_joint_err_mm(j_pred, j_gt):
    """(B,17,3) x 2 -> (B,17) sai so tung khop, mm (gia dinh don vi met)."""
    return torch.norm(j_pred - j_gt, dim=-1) * 1000.0


def center(x):
    return F.layer_norm(x, x.shape[-1:])


def cos_dist(a, b):
    """Khoang cach cosine da center - giong het cong thuc KD luc train. Tra ve shape = a.shape[:-1]."""
    return 1.0 - F.cosine_similarity(center(a), center(b.detach()), dim=-1)


# --------------------------------------------------------------------------
# Helpers thong ke / so hoc (numpy, khong phu thuoc torch)
# --------------------------------------------------------------------------
_EPS = 1e-8


def derangement(B, rng):
    """Hoan vi KHONG co diem bat dong (moi mau deu bi doi) - randperm thuong co the giu nguyen vai mau."""
    if B < 2:
        return None
    shift = int(rng.randint(1, B))
    return (np.arange(B) + shift) % B


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    return {'mean': float(x.mean()), 'median': float(np.median(x)), 'std': float(x.std()), 'n': int(x.size)}


def paired(d):
    """Hieu cap d (N,) -> (mean, nua-do-rong khoang tin cay 95% theo mau doc lap)."""
    d = np.asarray(d, dtype=np.float64)
    if d.size < 2:
        return float(d.mean()), float('nan')
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(d.size))


def linear_cka(X, Y):
    """Linear CKA giua hai bieu dien cua CUNG N mau: X (N,Dx), Y (N,Dy). 1 = cung hinh hoc, 0 = khong lien quan."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    Kx, Ky = X @ X.T, Y @ Y.T
    hsic = (Kx * Ky).sum()
    return float(hsic / (np.sqrt((Kx * Kx).sum()) * np.sqrt((Ky * Ky).sum()) + 1e-12))


def pca_fit(X, k=2):
    mu = X.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    return mu, Vt[:k], var[:k]


def pca_project(X, mu, comps):
    return (X - mu) @ comps.T


def attn_entropy(p):
    """p (B,H,Q,K), tong theo K = 1 -> entropy trung binh moi mau (B,)."""
    return (-(p * np.log(p + _EPS)).sum(-1)).mean(axis=(1, 2))


def attn_kl(p_t, p_s):
    """KL(Teacher || Student) tren chieu cuoi, trung binh theo (head, hang) -> (B,)."""
    return (p_t * (np.log(p_t + _EPS) - np.log(p_s + _EPS))).sum(-1).mean(axis=(1, 2))


# --------------------------------------------------------------------------
# Ve hinh (numpy + matplotlib)
# --------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_pca(feat_t, feat_s, err_s, path):
    plt = _plt()
    n = len(feat_t)
    mu, comps, var = pca_fit(np.concatenate([feat_t, feat_s], axis=0))
    ct, cs = pca_project(feat_t, mu, comps), pca_project(feat_s, mu, comps)
    mu_t, comps_t, var_t = pca_fit(feat_t)
    ct2, cs2 = pca_project(feat_t, mu_t, comps_t), pca_project(feat_s, mu_t, comps_t)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    ax.scatter(ct[:, 0], ct[:, 1], s=8, alpha=0.5, c='tab:blue', label='Teacher')
    ax.scatter(cs[:, 0], cs[:, 1], s=8, alpha=0.5, c='tab:orange', label='Student')
    for i in range(min(n, 60)):
        ax.plot([ct[i, 0], cs[i, 0]], [ct[i, 1], cs[i, 1]], c='gray', lw=0.4, alpha=0.5)
    ax.set_title(f'PCA chung (PC1 {var[0]:.1%}, PC2 {var[1]:.1%}); duong xam noi cung 1 mau')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend()

    ax = axes[1]
    ax.scatter(ct2[:, 0], ct2[:, 1], s=8, alpha=0.5, c='tab:blue', label='Teacher')
    ax.scatter(cs2[:, 0], cs2[:, 1], s=8, alpha=0.5, c='tab:orange', label='Student (chieu len truc Teacher)')
    ax.set_title(f'PCA chi fit tren Teacher (PC1 {var_t[0]:.1%}, PC2 {var_t[1]:.1%})')
    ax.set_xlabel('PC1 (Teacher)'); ax.set_ylabel('PC2 (Teacher)'); ax.legend()

    ax = axes[2]
    sc = ax.scatter(cs[:, 0], cs[:, 1], s=10, c=err_s, cmap='viridis')
    fig.colorbar(sc, ax=ax, label='Student MPJPE (mm)')
    ax.set_title('Student tren PCA chung, mau = MPJPE tung mau')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')

    fig.suptitle('feat_global: Teacher vs Student')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_per_joint(names, err_mb, err_s, err_t, kd_in, kd_jt, path):
    plt = _plt()
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    w = 0.27
    axes[0].bar(x - w, err_mb, w, label='MotionBERT lift')
    axes[0].bar(x, err_s, w, label='Student')
    axes[0].bar(x + w, err_t, w, label='Teacher')
    axes[0].set_ylabel('MPJPE (mm)'); axes[0].set_title('Sai so tung khop'); axes[0].legend()
    axes[1].bar(x - w / 2, kd_in, w, label='feat_joint_in')
    axes[1].bar(x + w / 2, kd_jt, w, label='feat_joint')
    axes[1].set_ylabel('khoang cach cosine (da center)')
    axes[1].set_title('Khoang cach dac trung Student-Teacher tung khop'); axes[1].legend()
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=60, ha='right')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_per_block(blk, path):
    """blk: dict ten -> list theo block."""
    plt = _plt()
    nb = len(blk['cos_joint'])
    x = np.arange(nb)
    labels = [f'block {i}' for i in range(nb)]
    w = 0.35
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.bar(x - w / 2, blk['cos_joint'], w, label='token joint')
    ax.bar(x + w / 2, blk['cos_img'], w, label='token anh')
    ax.set_title('Khoang cach dac trung Student-Teacher'); ax.legend()
    ax = axes[0, 1]
    ax.bar(x - w / 2, blk['ent_j2i_t'], w, label='Teacher')
    ax.bar(x + w / 2, blk['ent_j2i_s'], w, label='Student')
    ax.set_title('Entropy attention joint -> anh (cao = nhin rong)'); ax.legend()
    ax = axes[1, 0]
    ax.bar(x - w / 2, blk['ent_i2j_t'], w, label='Teacher')
    ax.bar(x + w / 2, blk['ent_i2j_s'], w, label='Student')
    ax.set_title('Entropy attention anh -> joint'); ax.legend()
    ax = axes[1, 1]
    ax.bar(x - w / 2, blk['kl_j2i'], w, label='joint -> anh')
    ax.bar(x + w / 2, blk['kl_i2j'], w, label='anh -> joint')
    ax.set_title('KL(Teacher || Student) cua attention'); ax.legend()
    for ax in axes.ravel():
        ax.set_xticks(x); ax.set_xticklabels(labels)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    teacher_ckpt = args.teacher_ckpt or cfg.MODEL.get('TEACHER', '')
    if not teacher_ckpt:
        print("[LOI] Thieu teacher checkpoint: dung --teacher_ckpt hoac dat cfg.MODEL.TEACHER")
        sys.exit(1)
    for tag, p in (('Student', args.student_ckpt), ('Teacher', teacher_ckpt)):
        if not os.path.isfile(p):
            print(f"[LOI] Khong thay checkpoint {tag}: {p}")
            sys.exit(1)
    print(f"Student ckpt : {args.student_ckpt}")
    print(f"Teacher ckpt : {teacher_ckpt}")

    hr("Dang build va nap checkpoint cho 2 model...")
    student = build_and_load('student', args.student_ckpt)
    teacher = build_and_load('teacher', teacher_ckpt)

    hr("Dang chuan bi tap val (3DPW test split)...")
    test_name = cfg.DATASET.test_list[0]
    dataset = get_test_dataset(test_name, None)
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=gen,
                        num_workers=args.workers, pin_memory=False)
    print(f"Dataset: {test_name}  |  so sample: {len(dataset)}  |  se duyet toi da "
          f"{args.max_batches} batch (batch_size={args.batch_size})")

    h36m_reg = torch.as_tensor(dataset.h36m_joint_regressor, dtype=torch.float32, device=DEVICE)
    shuf_rng = np.random.RandomState(args.seed + 1)

    acc = defaultdict(list)

    def add(name, value):
        if torch.is_tensor(value):
            value = value.detach().float().cpu().numpy()
        acc[name].append(np.asarray(value))

    def cat(name):
        return np.concatenate(acc[name], axis=0)

    n_batches, n_blocks = 0, 0
    with torch.no_grad():
        for inputs, targets, meta in loader:
            if n_batches >= args.max_batches:
                break
            n_batches += 1

            img = inputs['img'].to(DEVICE).float()
            joints2d = inputs['joints'].to(DEVICE).float()
            joints_mask = inputs['joints_mask'].to(DEVICE).float()
            gt_mesh = targets['smpl_mesh_cam'].to(DEVICE).float()
            B = img.shape[0]

            gt_h36m = h36m_from_mesh(gt_mesh, h36m_reg)          # (B,17,3) root-relative, GT that

            # Teacher: GT 3D (regress tu GT mesh, giong luc eval that). Student: 2D -> MotionBERT -> fusion.
            t_out = teacher(img, gt_h36m, is_train=False)
            s_out = student(img, joints2d, is_train=False, joints_mask=joints_mask)

            # ---- [1][2][3] MPJPE ----
            mb_joints = s_out['lifted_joints_3d']                # da root-relative (xem forward_student)
            s_j = h36m_from_mesh(s_out['smpl_mesh_cam'], h36m_reg)
            t_j = h36m_from_mesh(t_out['smpl_mesh_cam'], h36m_reg)
            add('err_mb', per_joint_err_mm(mb_joints, gt_h36m))
            add('err_s', per_joint_err_mm(s_j, gt_h36m))
            add('err_t', per_joint_err_mm(t_j, gt_h36m))
            add('err_s_vs_t', per_joint_err_mm(s_j, t_j))

            # ---- [4] khoang cach dac trung (tung khop) + dac trung cho CKA / PCA ----
            add('kd_proj_pj', cos_dist(s_out['feat_joint_in'], t_out['feat_joint_in']))   # (B,17)
            add('kd_joint_pj', cos_dist(s_out['feat_joint'], t_out['feat_joint']))        # (B,17)
            add('kd_global', cos_dist(s_out['feat_global'], t_out['feat_global']))        # (B,)
            for tag, key in (('in', 'feat_joint_in'), ('jt', 'feat_joint'), ('gl', 'feat_global')):
                add(f'f_{tag}_t', t_out[key].flatten(1))
                add(f'f_{tag}_s', s_out[key].flatten(1))

            # ---- [5] tung block fusion ----
            n_blocks = len(s_out['feat_layers'])
            for bi, (ls, lt) in enumerate(zip(s_out['feat_layers'], t_out['feat_layers'])):
                add(f'blk{bi}_cos_joint', cos_dist(ls['joint'], lt['joint']).mean(-1))
                add(f'blk{bi}_cos_img', cos_dist(ls['img'], lt['img']).mean(-1))
                for nm, key in (('j2i', 'attn_joint_to_img'), ('i2j', 'attn_img_to_joint')):
                    ps = ls[key].detach().float().cpu().numpy()
                    pt = lt[key].detach().float().cpu().numpy()
                    add(f'blk{bi}_ent_{nm}_s', attn_entropy(ps))
                    add(f'blk{bi}_ent_{nm}_t', attn_entropy(pt))
                    add(f'blk{bi}_kl_{nm}', attn_kl(pt, ps))

            # ---- [6] shuffle test (hoan vi khong diem bat dong) ----
            perm_img = derangement(B, shuf_rng)
            perm_jnt = derangement(B, shuf_rng)
            if perm_img is not None:
                pi = torch.as_tensor(perm_img, device=DEVICE)
                pj = torch.as_tensor(perm_jnt, device=DEVICE)

                # cung tap mau con de so sanh (khi B<2 bi bo qua)
                add('sub_s_norm', per_joint_err_mm(s_j, gt_h36m).mean(1))
                add('sub_t_norm', per_joint_err_mm(t_j, gt_h36m).mean(1))

                # tron ANH, giu nguyen joint va GT
                s_im = student(img[pi], joints2d, is_train=False, joints_mask=joints_mask)
                t_im = teacher(img[pi], gt_h36m, is_train=False)
                add('s_img_shuf', per_joint_err_mm(h36m_from_mesh(s_im['smpl_mesh_cam'], h36m_reg), gt_h36m).mean(1))
                add('t_img_shuf', per_joint_err_mm(h36m_from_mesh(t_im['smpl_mesh_cam'], h36m_reg), gt_h36m).mean(1))

                # tron JOINT (va mask di kem), giu nguyen anh va GT
                s_jn = student(img, joints2d[pj], is_train=False, joints_mask=joints_mask[pj])
                t_jn = teacher(img, gt_h36m[pj], is_train=False)
                add('s_jnt_shuf', per_joint_err_mm(h36m_from_mesh(s_jn['smpl_mesh_cam'], h36m_reg), gt_h36m).mean(1))
                add('t_jnt_shuf', per_joint_err_mm(h36m_from_mesh(t_jn['smpl_mesh_cam'], h36m_reg), gt_h36m).mean(1))

            if n_batches % 5 == 0:
                print(f"  ... da xu ly {n_batches}/{args.max_batches} batch")

    if n_batches == 0:
        print("[LOI] Khong duyet duoc batch nao (dataset rong?).")
        sys.exit(1)

    results = {'n_batches': n_batches, 'student_ckpt': args.student_ckpt, 'teacher_ckpt': teacher_ckpt}

    # ======================================================================
    e_mb, e_s, e_t = cat('err_mb'), cat('err_s'), cat('err_t')      # (N,17)
    m_mb, m_s, m_t = e_mb.mean(1), e_s.mean(1), e_t.mean(1)         # (N,)
    N = len(m_s)
    d_mb_s, d_s_t = m_mb - m_s, m_s - m_t
    gm, gm_ci = paired(d_mb_s)
    gt_, gt_ci = paired(d_s_t)
    results['mpjpe'] = {'motionbert': summarize(m_mb), 'student': summarize(m_s), 'teacher': summarize(m_t),
                        'gain_vs_motionbert': {'mean': gm, 'ci95': gm_ci},
                        'student_minus_teacher': {'mean': gt_, 'ci95': gt_ci},
                        'frac_student_better_than_motionbert': float((d_mb_s > 0).mean())}

    hr("[1] MPJPE (mm) - 3 moc so sanh tren cung tap val")
    print(f"  {'':<48}{'mean':>8}{'median':>9}{'std':>8}")
    for lab, arr in (('MotionBERT lift (can duoi, input tho cua Student)', m_mb),
                     ('Student (sau fusion + HyperGCN + VPoser)', m_s),
                     ('Teacher (can tren, nhan GT 3D sach)', m_t)):
        s = summarize(arr)
        print(f"  {lab:<48}{s['mean']:8.2f}{s['median']:9.2f}{s['std']:8.2f}")
    print(f"\n  Student cai thien so voi MotionBERT : {gm:+.2f} +/- {gm_ci:.2f} mm (95%, n={N}); "
          f"{results['mpjpe']['frac_student_better_than_motionbert']:.1%} mau tot hon")
    print(f"  Student con cach Teacher            : {gt_:+.2f} +/- {gt_ci:.2f} mm")
    print("  (khoang tin cay tinh theo mau doc lap -> lac quan voi 3DPW)")
    if gm < 0:
        print(f"\n  [CANH BAO] Student TE HON joint MotionBERT dau vao {-gm:.1f} mm "
              f"({1 - results['mpjpe']['frac_student_better_than_motionbert']:.1%} mau te hon): "
              "pipeline dang lam xau input thay vi khu nhieu.")
    elif gm < args.improve_thresh:
        print(f"\n  [GOI Y] Student chi cai thien {gm:.1f} mm (< {args.improve_thresh} mm) so voi joint MotionBERT dau vao.")
        print("  Co the sai so cua joint dau vao dang quyet dinh ket qua; can doi chieu voi muc [5], [6] truoc khi ket luan.")

    # ======================================================================
    kd_in_pj = cat('kd_proj_pj').mean(0)       # (17,)
    kd_jt_pj = cat('kd_joint_pj').mean(0)      # (17,)
    results['per_joint'] = {n: {'motionbert': float(e_mb[:, j].mean()), 'student': float(e_s[:, j].mean()),
                                'teacher': float(e_t[:, j].mean()), 'kd_feat_joint_in': float(kd_in_pj[j]),
                                'kd_feat_joint': float(kd_jt_pj[j])} for j, n in enumerate(H36M_JOINTS)}
    hr("[2] Tung khop (mm; khoang cach dac trung: 0 = giong het Teacher)")
    print(f"  {'joint':<11}{'MotionBERT':>11}{'Student':>9}{'Teacher':>9}{'d(feat_in)':>12}{'d(feat_jt)':>12}")
    for j, n in enumerate(H36M_JOINTS):
        print(f"  {n:<11}{e_mb[:, j].mean():11.2f}{e_s[:, j].mean():9.2f}{e_t[:, j].mean():9.2f}"
              f"{kd_in_pj[j]:12.4f}{kd_jt_pj[j]:12.4f}")

    # ======================================================================
    agree = cat('err_s_vs_t').mean(1)
    results['output_agreement_student_vs_teacher_mm'] = summarize(agree)
    hr("[3] Do giong nhau o dau ra: MPJPE giua khop Student va khop Teacher")
    print(f"  mean {agree.mean():.2f} mm | median {np.median(agree):.2f} mm")
    print(f"  (so voi: Student-GT = {m_s.mean():.2f} mm, Teacher-GT = {m_t.mean():.2f} mm)")

    # ======================================================================
    kd_global = cat('kd_global')
    # CKA ghep dung cap mau, va CKA "nen" khi ghep SAI cap (Student mau i voi Teacher mau khac)
    cka_perm = derangement(N, shuf_rng)
    cka, cka_base = {}, {}
    for k, tag in (('feat_joint_in', 'in'), ('feat_joint', 'jt'), ('feat_global', 'gl')):
        ft, fs = cat(f'f_{tag}_t'), cat(f'f_{tag}_s')
        cka[k] = linear_cka(ft, fs)
        cka_base[k] = linear_cka(ft, fs[cka_perm]) if cka_perm is not None else float('nan')
    results['feature_distance'] = {'kd_proj': float(kd_in_pj.mean()), 'kd_joint': float(kd_jt_pj.mean()),
                                   'kd_global': float(kd_global.mean())}
    results['linear_cka'] = {'paired': cka, 'baseline_shuffled_pairs': cka_base}
    hr("[4] Khong gian dac trung Student vs Teacher")
    print("  Khoang cach cosine da center (0 = giong het; so voi log 'KD dist' luc TRAIN):")
    print(f"    kd_proj   (feat_joint_in) : {kd_in_pj.mean():.4f}")
    print(f"    kd_joint  (feat_joint)    : {kd_jt_pj.mean():.4f}")
    print(f"    kd_global (feat_global)   : {kd_global.mean():.4f}")
    print("  Linear CKA (1 = cung hinh hoc khong gian, bat bien phep xoay / doi truc):")
    print(f"    {'':<16}  {'ghep dung':>9}   {'nen (ghep sai cap)':>18}")
    for k, v in cka.items():
        print(f"    {k:<16}: {v:9.4f}   {cka_base[k]:18.4f}")
    print("  Luon doc CKA cung gia tri nen: khi so chieu >> so mau (vd feat_joint 8704 chieu), CKA cua du lieu")
    print("  khong lien quan cung co the rat cao, nen chi co y nghia khi 'ghep dung' >> 'nen'.")
    if N < 100:
        print(f"  [LUU Y] Chi co {N} mau, CKA kem on dinh khi N nho; tang --max_batches.")
    print("  Neu khoang cach tren val cao hon han luc train -> Student overfit KD vao tap train.")

    # ======================================================================
    if n_blocks:
        blk = {k: [] for k in ('cos_joint', 'cos_img', 'ent_j2i_s', 'ent_j2i_t', 'ent_i2j_s', 'ent_i2j_t', 'kl_j2i', 'kl_i2j')}
        for bi in range(n_blocks):
            for k in blk:
                blk[k].append(float(cat(f'blk{bi}_{k}').mean()))
        results['per_block'] = blk
        hr("[5] Tung block fusion (cross-attention 2 chieu)")
        print(f"  {'block':<7}{'d(joint)':>10}{'d(img)':>9}{'H j->i T':>10}{'H j->i S':>10}"
              f"{'H i->j T':>10}{'H i->j S':>10}{'KL j->i':>9}{'KL i->j':>9}")
        for bi in range(n_blocks):
            print(f"  {bi:<7}{blk['cos_joint'][bi]:10.4f}{blk['cos_img'][bi]:9.4f}"
                  f"{blk['ent_j2i_t'][bi]:10.3f}{blk['ent_j2i_s'][bi]:10.3f}"
                  f"{blk['ent_i2j_t'][bi]:10.3f}{blk['ent_i2j_s'][bi]:10.3f}"
                  f"{blk['kl_j2i'][bi]:9.4f}{blk['kl_i2j'][bi]:9.4f}")
        print("  H = entropy attention (T = Teacher, S = Student; toi da: joint->anh ln64 = 4.16, anh->joint ln17 = 2.83).")
        print("  Entropy cua Student cao hon Teacher nghia la Student nhin phan tan hon; KL lon nghia la khac cach dinh tuyen.")

    # ======================================================================
    shuffle = None
    if 's_img_shuf' in acc:
        s0, t0 = cat('sub_s_norm'), cat('sub_t_norm')
        shuffle = {
            'student_img': float(cat('s_img_shuf').mean() - s0.mean()),
            'teacher_img': float(cat('t_img_shuf').mean() - t0.mean()),
            'student_joint': float(cat('s_jnt_shuf').mean() - s0.mean()),
            'teacher_joint': float(cat('t_jnt_shuf').mean() - t0.mean()),
            'student_base': float(s0.mean()), 'teacher_base': float(t0.mean()), 'n': int(s0.size),
        }
        results['shuffle'] = shuffle
        hr("[6] Shuffle test - model dua vao anh hay dua vao joint? (MPJPE tang bao nhieu, mm)")
        print(f"  {'':<10}{'goc':>8}{'tron anh':>12}{'tron joint':>13}")
        print(f"  {'Student':<10}{s0.mean():8.2f}{shuffle['student_img']:+12.2f}{shuffle['student_joint']:+13.2f}")
        print(f"  {'Teacher':<10}{t0.mean():8.2f}{shuffle['teacher_img']:+12.2f}{shuffle['teacher_joint']:+13.2f}")
        print("  (hoan vi khong diem bat dong, moi mau deu bi doi; GT giu cua mau goc)")
        if abs(shuffle['student_img']) < args.shuffle_thresh:
            print(f"\n  [GOI Y] Tron anh chi doi MPJPE cua Student {shuffle['student_img']:+.2f} mm "
                  f"(< {args.shuffle_thresh} mm): Student co ve it dung nhanh anh.")
        if abs(shuffle['teacher_img']) < args.shuffle_thresh:
            print(f"  [GOI Y] Tron anh chi doi MPJPE cua Teacher {shuffle['teacher_img']:+.2f} mm: "
                  "Teacher co ve chu yeu dua vao GT joint.")
        if shuffle['student_joint'] < shuffle['student_img']:
            print("  [GOI Y] Student nhay cam voi anh hon voi joint -> dang dua vao anh nhieu hon joint.")
    else:
        print("\n[6] Bo qua shuffle test (moi batch chi co 1 mau).")

    # ======================================================================
    hr("[7] Hinh ve")
    plots = [('feat_global_pca.png', lambda p: plot_pca(cat('f_gl_t'), cat('f_gl_s'), m_s, p)),
             ('per_joint.png', lambda p: plot_per_joint(H36M_JOINTS, e_mb.mean(0), e_s.mean(0), e_t.mean(0),
                                                        kd_in_pj, kd_jt_pj, p))]
    if n_blocks:
        plots.append(('per_block.png', lambda p: plot_per_block(blk, p)))
    for fname, fn in plots:
        path = os.path.join(args.out_dir, fname)
        try:
            fn(path)
            print(f"  Da luu: {path}")
        except Exception as e:
            print(f"  [BO QUA] {fname}: {e}")
    print("  feat_global_pca.png: panel 1 = PCA chung (duong xam noi cap Teacher-Student cung mau);")
    print("    panel 2 = PCA chi fit tren Teacher, chieu Student len truc do; panel 3 = Student to mau theo MPJPE.")

    json_path = os.path.join(args.out_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Da luu so lieu: {json_path}")
    if args.save_feats:
        npz_path = os.path.join(args.out_dir, 'features.npz')
        np.savez_compressed(npz_path, f_global_teacher=cat('f_gl_t'), f_global_student=cat('f_gl_s'),
                            f_joint_teacher=cat('f_jt_t'), f_joint_student=cat('f_jt_s'),
                            err_motionbert=e_mb, err_student=e_s, err_teacher=e_t)
        print(f"  Da luu dac trung: {npz_path}")

    hr("TONG KET")
    print(f"So batch: {n_batches} (~{N} sample)")
    print(f"MPJPE: MotionBERT={m_mb.mean():.2f} | Student={m_s.mean():.2f} | Teacher={m_t.mean():.2f} mm")
    print(f"KD distance (val): proj={kd_in_pj.mean():.4f} | joint={kd_jt_pj.mean():.4f} | global={kd_global.mean():.4f}")
    print("Linear CKA (ghep dung / nen): " + " | ".join(f"{k}={v:.3f}/{cka_base[k]:.3f}" for k, v in cka.items()))
    if shuffle:
        print(f"Tron anh (mm): Student {shuffle['student_img']:+.2f} | Teacher {shuffle['teacher_img']:+.2f}; "
              f"tron joint: Student {shuffle['student_joint']:+.2f} | Teacher {shuffle['teacher_joint']:+.2f}")


if __name__ == '__main__':
    main()
