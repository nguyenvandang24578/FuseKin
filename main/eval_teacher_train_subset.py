"""
eval_teacher_train_subset.py — Teacher sai 46.7 mm tren val (san decoder <= 27 mm): do OVERFIT hay do THIEU KHA NANG?
=====================================================================================================================

Danh gia CUNG mot checkpoint, CUNG giao thuc (14 khop H36M, PA-MPJPE, MPVPE nhu PW3D.evaluate) tren:
    [train] tap con ngau nhien cua 3DPW-train
    [val]   tap con ngau nhien cua 3DPW test (cfg.DATASET.test_list[0])

Tap train duoc nap voi data_name='3dpw-train' (dung datalist train) nhung ep data_split='test' sau khi
nap -> __getitem__ di nhanh TEST: KHONG augmentation, bbox tinh tu OpenPose giong het val,
target = smpl_mesh_cam tuyet doi. Chi khac nhau o DU LIEU.

Ngoai trung binh, script in them BANG THEO TUNG KHOP va THEO TUNG XUONG:
  - MPJPE / PA-MPJPE tung khop: khop nao sai nhieu nhat.
  - Sai GOC tung xuong (do, giua huong xuong du doan va GT): KHONG bi cong don theo chuoi khop.
    -> Neu sai VI TRI tang dan ra dau chi nhung sai GOC tung xuong deu nhau: loi chu yeu do CONG DON.
    -> Neu sai GOC lon o vai xuong cu the (vd cang tay, cang chan): loi o CHINH cac khop do (xoay
       quanh truc xuong, VPoser kem o vung do...).

Cach chay:
    python main/eval_teacher_train_subset.py --cfg ./config/train_teacher_gt.yml \
        --teacher_ckpt <best.pth.tar> [--num 3000] [--batch_size 32] [--workers 4] [--skip_val]

Can CUDA. Khong train gi.
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Danh gia Teacher tren tap con 3DPW-train vs val (cung giao thuc) + theo tung khop')
parser.add_argument('--cfg', type=str, required=True, help='config cua Teacher (vd ./config/train_teacher_gt.yml)')
parser.add_argument('--teacher_ckpt', type=str, default='', help='mac dinh: cfg.MODEL.TEACHER')
parser.add_argument('--num', type=int, default=3000, help='so mau moi tap (<=0: toan bo)')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--skip_val', action='store_true', help='chi danh gia train')
args = parser.parse_args()

from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401
from models.ARTS import ARTS
from core.base import load_model_weights
from utils.jotr_dataset import Human36M17Dataset, get_test_dataset
from utils.transforms import rigid_align
from data_final.PW3D.dataset import PW3D

if not torch.cuda.is_available():
    print("\n[LOI] Can CUDA de chay script nay.\n")
    sys.exit(1)
DEVICE = torch.device('cuda')

H36M_NAMES = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck',
              'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
EVAL_JOINTS = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)   # giong PW3D.h36m_eval_joint
# (cha, con) theo cay H36M
BONES = ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
         (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16))
# Nhom de tom tat: than/goc -> giua chi -> dau chi
GROUPS = {
    'hong (Hip)':          ('R_Hip', 'L_Hip'),
    'goi (Knee)':          ('R_Knee', 'L_Knee'),
    'co chan (Ankle)':     ('R_Ankle', 'L_Ankle'),
    'co + dau (Neck/Head)': ('Neck', 'Head_top'),
    'vai (Shoulder)':      ('R_Shoulder', 'L_Shoulder'),
    'khuyu (Elbow)':       ('R_Elbow', 'L_Elbow'),
    'co tay (Wrist)':      ('R_Wrist', 'L_Wrist'),
}


def hr(title=''):
    print('\n' + '=' * 78)
    if title:
        print(title)
        print('=' * 78)


def build_teacher(ckpt_path):
    if cfg.MODEL.name != 'teacher':
        print(f"[LOI] cfg.MODEL.name = '{cfg.MODEL.name}', script nay chi danh gia Teacher. Dung config cua Teacher.")
        sys.exit(1)
    if not os.path.isfile(ckpt_path):
        print(f"[LOI] Khong thay checkpoint: {ckpt_path}")
        sys.exit(1)
    model = ARTS(num_joint=17, embed_dim=cfg.MODEL.hpe_dim).to(DEVICE)
    model = load_model_weights(model, ckpt_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def build_train_eval_dataset():
    """Datalist 3DPW-train nhung di nhanh TEST cua __getitem__ (khong augmentation, target mesh tuyet doi)."""
    base = PW3D(transforms.ToTensor(), data_name='3dpw-train')
    base.data_split = 'test'
    return Human36M17Dataset(base)


def make_loader(dataset, num, seed):
    n = len(dataset)
    if 0 < num < n:
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(n, size=num, replace=False))
        sub = Subset(dataset, idx.tolist())
    else:
        sub = dataset
    loader = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    return loader, len(sub)


@torch.no_grad()
def collect(model, dataset, loader):
    """Tra ve (N,17,3) joint du doan / GT (met, root-relative) va MPVPE tung mau (mm).
    Forward giong het nhanh Teacher trong evaluate_3dpw_subset."""
    reg17 = torch.as_tensor(dataset.h36m_joint_regressor, dtype=torch.float32, device=DEVICE)   # (17,6890)
    reg_smpl = torch.as_tensor(dataset.joint_regressor, dtype=torch.float32, device=DEVICE)     # SMPL joint regressor
    alpha_max = cfg.MODEL.get('teacher_lift_alpha_max', 0.0)
    preds, gts, mpvpe = [], [], []
    for inputs, targets, meta in tqdm(loader, leave=False):
        img = inputs['img'].to(DEVICE).float()
        gt_mesh = targets['smpl_mesh_cam'].to(DEVICE).float()
        gt_j = torch.einsum('jk,bkc->bjc', reg17, gt_mesh)
        gt_j = gt_j - gt_j[:, 0:1]
        kw = {}
        if alpha_max > 0:
            a = cfg.MODEL.get('teacher_lift_alpha_eval', None)
            a = alpha_max if a is None else float(a)
            kw = dict(pose_2d=inputs['joints'].to(DEVICE).float(),
                      joints_mask=inputs['joints_mask'].to(DEVICE).float(),
                      lift_alpha=torch.full((img.shape[0],), a, device=DEVICE))
        out = model(img, gt_j, is_train=False, **kw)
        pr_mesh = out['smpl_mesh_cam'].float()
        pr_j = torch.einsum('jk,bkc->bjc', reg17, pr_mesh)
        pr_j = pr_j - pr_j[:, 0:1]
        # MPVPE: giong PW3D.evaluate (tru joint 0 cua SMPL regressor)
        g = gt_mesh - torch.einsum('jk,bkc->bjc', reg_smpl, gt_mesh)[:, 0:1]
        p = pr_mesh - torch.einsum('jk,bkc->bjc', reg_smpl, pr_mesh)[:, 0:1]
        mpvpe.append((torch.norm(p - g, dim=-1).mean(-1) * 1000).cpu())
        preds.append(pr_j.cpu())
        gts.append(gt_j.cpu())
    return torch.cat(preds).numpy(), torch.cat(gts).numpy(), torch.cat(mpvpe).numpy()


def analyze(pred, gt):
    """pred, gt: (N,17,3) met, root-relative."""
    N = pred.shape[0]
    ev = list(EVAL_JOINTS)
    # vi tri tung khop (mm), chua can chinh
    err = np.linalg.norm(pred - gt, axis=-1) * 1000                       # (N,17)
    # PA: can chinh tren 14 khop eval (giong PW3D.evaluate), roi do tung khop
    pa = np.zeros((N, len(ev)))
    for i in range(N):
        al = rigid_align(pred[i, ev], gt[i, ev])
        pa[i] = np.linalg.norm(al - gt[i, ev], axis=-1) * 1000
    # sai goc tung xuong (do) — khong cong don theo chuoi
    ang = np.zeros((N, len(BONES)))
    for b, (pa_, ch) in enumerate(BONES):
        vp = pred[:, ch] - pred[:, pa_]
        vg = gt[:, ch] - gt[:, pa_]
        cos = (vp * vg).sum(-1) / (np.linalg.norm(vp, axis=-1) * np.linalg.norm(vg, axis=-1) + 1e-8)
        ang[:, b] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    # sai do dai xuong (mm) — phan "shape/ty le"
    blen = np.zeros((N, len(BONES)))
    for b, (pa_, ch) in enumerate(BONES):
        blen[:, b] = (np.linalg.norm(pred[:, ch] - pred[:, pa_], axis=-1)
                      - np.linalg.norm(gt[:, ch] - gt[:, pa_], axis=-1)) * 1000
    return dict(mpjpe=err[:, ev].mean(), pa=pa.mean(), err=err, pa_j=pa, ang=ang, blen=blen)


def print_tables(res):
    names = list(res.keys())
    ev = list(EVAL_JOINTS)

    hr("TRUNG BINH (mm) — giong PW3D.evaluate")
    print(f"  {'':8s} {'n':>6s} {'MPJPE':>9s} {'PA-MPJPE':>9s} {'MPVPE':>9s}")
    for k in names:
        r = res[k]
        print(f"  {k:8s} {r['n']:6d} {r['mpjpe']:9.2f} {r['pa']:9.2f} {r['mpvpe']:9.2f}")
    print(f"  {'san':8s} {'':>6s} {'<=27.3':>9s} {'~16':>9s} {'~30':>9s}   (vposer_floor_test, shape=0)")

    hr("THEO TUNG KHOP (mm): MPJPE / PA-MPJPE   (sap xep theo MPJPE val giam dan)")
    key_sort = 'val' if 'val' in res else names[0]
    order = sorted(range(len(ev)), key=lambda t: -res[key_sort]['err'][:, ev[t]].mean())
    header = "  {:12s}".format('khop') + ''.join(f" {k + ' MPJPE':>12s} {k + ' PA':>9s}" for k in names)
    print(header)
    for t in order:
        j = ev[t]
        row = f"  {H36M_NAMES[j]:12s}"
        for k in names:
            row += f" {res[k]['err'][:, j].mean():12.1f} {res[k]['pa_j'][:, t].mean():9.1f}"
        print(row)

    hr("THEO NHOM (trung binh trai/phai, mm): goc chi -> giua chi -> dau chi")
    print("  {:22s}".format('nhom') + ''.join(f" {k + ' MPJPE':>12s} {k + ' PA':>9s}" for k in names))
    for g, js in GROUPS.items():
        idx_j = [H36M_NAMES.index(n) for n in js]
        idx_t = [ev.index(j) for j in idx_j]
        row = f"  {g:22s}"
        for k in names:
            row += f" {res[k]['err'][:, idx_j].mean():12.1f} {res[k]['pa_j'][:, idx_t].mean():9.1f}"
        print(row)

    hr("THEO TUNG XUONG: sai GOC (do, khong cong don) va sai DO DAI (mm, du doan - GT)")
    print("  {:24s}".format('xuong') + ''.join(f" {k + ' goc':>10s} {k + ' dai':>9s}" for k in names))
    b_order = sorted(range(len(BONES)), key=lambda b: -res[key_sort]['ang'][:, b].mean())
    for b in b_order:
        pa_, ch = BONES[b]
        row = f"  {H36M_NAMES[pa_] + '->' + H36M_NAMES[ch]:24s}"
        for k in names:
            row += f" {res[k]['ang'][:, b].mean():10.1f} {res[k]['blen'][:, b].mean():+9.1f}"
        print(row)

    print("\n  Cach doc:")
    print("  - Sai VI TRI tang dan hong->goi->co chan / vai->khuyu->co tay, nhung sai GOC xuong deu nhau")
    print("    -> chu yeu la CONG DON theo chuoi khop.")
    print("  - Sai GOC lon han o vai xuong (vd Knee->Ankle, Elbow->Wrist)")
    print("    -> loi o CHINH cac khop do; kiem tra VPoser/head pose o vung do.")
    print("  - Sai DO DAI xuong lech he thong (+/- vai mm tro len) -> shape (beta) du doan sai.")


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt = args.teacher_ckpt or cfg.MODEL.get('TEACHER', '')
    if not ckpt:
        print("[LOI] Thieu checkpoint: dung --teacher_ckpt hoac dat cfg.MODEL.TEACHER")
        sys.exit(1)
    print(f"Teacher ckpt : {ckpt}")
    print(f"teacher_lift_alpha_max = {cfg.MODEL.get('teacher_lift_alpha_max', 0.0)}")

    hr("Build va nap Teacher")
    model = build_teacher(ckpt)

    sets = [('train', build_train_eval_dataset())]
    if not args.skip_val:
        sets.append(('val', get_test_dataset(cfg.DATASET.test_list[0], None)))

    res = {}
    for name, ds in sets:
        loader, n = make_loader(ds, args.num, args.seed)
        print(f"  [{name}] {n}/{len(ds)} mau")
        pred, gt, mpvpe = collect(model, ds, loader)
        r = analyze(pred, gt)
        r['mpvpe'] = float(mpvpe.mean())
        r['n'] = n
        res[name] = r
        print(f"  [{name}] MPJPE={r['mpjpe']:.2f}  PA-MPJPE={r['pa']:.2f}  MPVPE={r['mpvpe']:.2f}")

    print_tables(res)


if __name__ == '__main__':
    main()
