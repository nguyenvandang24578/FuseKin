"""
tools/fit_err_stats.py -- THONG KE sai so fit NeuralAnnot so voi nhan 2D gan tay (MSCOCO, CrowdPose train), kieu ARTS /
I2L-MeshNet (get_fitting_error). KHONG train, KHONG doc anh, KHONG sua dataset.

Buoc 1 (kiem regressor): J_regressor_coco.npy phai (17, 6890), tong moi hang ~1. In them do lech tung khop giua
  J_regressor_coco va J_regressor_coco_hip_smpl tren vai mesh fit (ky vong: lech chu yeu o 2 hong).
Buoc 2 (sai so fit), moi nguoi co fit, KHONG augmentation:
  mesh fit (get_smpl_coord, khong lat) -> J_regressor_coco -> 17 khop COCO 3D -> cam2pixel (camera NeuralAnnot)
  -> affine crop (bbox giong __getitem__, scale 1, rot 0) -> luoi output_hm_shape (64)
  so voi nhan tay qua CUNG phep bien doi, chi tren khop co nhan (joint_valid):
    MSCOCO   : 17 khop COCO
    CrowdPose: 12 khop chi (vai, khuyu, co tay, hong, goi, co chan); Head_top / Neck khong co trong regressor COCO -> bo
  sai so mau = TB khoang cach (px luoi 64) tren cac khop co nhan.
In: phan phoi (p10/p50/p90/p99), % mau bi loc o nguong 2/3/4/5/8 px, tach nhom num_overlap == 0 / > 0,
    sai so TB tung khop; CrowdPose: them so nguoi quy tac jotr_strict (fit_err < 5 -> bo) se loai.
Luu: <out_dir>/fit_err_<bo>.npz (ann_id, err, n_valid, num_overlap, per_joint, fit_err) de dung lai.

Chay (moi truong arts, tu thu muc goc repo):
  python tools/fit_err_stats.py --cfg config/train_img_lifter_v2_no3dpw.yaml
  python tools/fit_err_stats.py --cfg config/train_img_lifter_v2_no3dpw.yaml --sets MSCOCO --max_n 2000   # chay thu nhanh
"""
import os, sys
sys.path.insert(0, './lib')
sys.path.insert(0, './')
sys.path.insert(0, './main')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import os.path as osp

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True, help='YAML train (fusekin_unified + gt_posefix), vd lifter v2 no3dpw')
parser.add_argument('--sets', default='MSCOCO,CrowdPose')
parser.add_argument('--coco_reg', default='', help="'' = <jotr_data_root>/MSCOCO/J_regressor_coco.npy")
parser.add_argument('--max_n', type=int, default=0, help='0 = tat ca nguoi co fit; >0 = chi lay N nguoi dau (chay thu)')
parser.add_argument('--n_reg_check', type=int, default=200, help='so mesh dung de so 2 regressor (buoc 1)')
parser.add_argument('--out_dir', default='experiment/_fit_err_stats')
args = parser.parse_args()

from core.config import cfg, update_config
update_config(args.cfg)
# CrowdPose: TAT quy tac jotr_strict khi load de thay TOAN BO fit; anh huong cua quy tac duoc dem rieng qua fit_err
STRICT_ORIG = bool(cfg.DATASET.get('jotr_strict', False))
cfg.DATASET.jotr_strict = False
import __init_path  # noqa: F401
from torchvision import transforms
from tqdm import tqdm
from utils.jotr_dataset import TRAIN_DATASETS
from utils.preprocessing import process_bbox, augmentation_noimg
from utils.transforms import cam2pixel

COCO17 = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
          'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle')
LIMB12 = ('L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
          'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle')
THRS = (2.0, 3.0, 4.0, 5.0, 8.0)
CROWD_FIT_THR = 5.0      # nguong quy tac jotr_strict cua CrowdPose (self.fitting_thr)


def load_coco_reg():
    path = args.coco_reg or osp.join(cfg.DATASET.jotr_data_root, 'MSCOCO', 'J_regressor_coco.npy')
    reg = np.load(path).astype(np.float64)
    print(f'===> J_regressor_coco: {path} | shape {reg.shape}')
    assert reg.shape == (17, 6890), f'Ky vong (17, 6890), nhan {reg.shape}'
    rs = reg.sum(1)
    print(f'     tong moi hang: min {rs.min():.4f} | max {rs.max():.4f} (ky vong ~1)')
    path_hip = osp.join(cfg.DATASET.jotr_data_root, 'MSCOCO', 'J_regressor_coco_hip_smpl.npy')
    reg_hip = np.load(path_hip).astype(np.float64) if osp.isfile(path_hip) else None
    if reg_hip is not None:
        print(f'     J_regressor_coco_hip_smpl: {path_hip} | shape {reg_hip.shape}')
    return reg, reg_hip


def to_hm(xy_img, trans):
    """Toa do anh goc (N, 2) -> crop input_img_shape (affine trans) -> luoi output_hm_shape."""
    xy1 = np.concatenate([xy_img, np.ones_like(xy_img[:, :1])], 1)
    xy = np.dot(trans, xy1.T).T[:, :2]
    xy[:, 0] = xy[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
    xy[:, 1] = xy[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
    return xy


def get_fit(name, data):
    """Tra (smpl_param, cam_param, fit_err | nan) cua 1 nguoi; None neu khong co fit."""
    na = data.get('smpl_param') if name == 'MSCOCO' else data.get('neural_annot_result')
    if na is None:
        return None
    fe = na.get('fit_err', np.nan) if isinstance(na, dict) else np.nan
    return na['smpl_param'], na['cam_param'], float(fe)


def run_set(name, reg, reg_hip):
    ds = TRAIN_DATASETS[name](transforms.ToTensor(), 'train')
    if name == 'MSCOCO':
        lab_names, lab_idx = COCO17, list(range(17))                            # nhan COCO: 17 khop dau
    else:
        lab_names = LIMB12
        lab_idx = [ds.crowdpose_joints_name.index(n) for n in LIMB12]           # chi so trong nhan CrowdPose
    reg_idx = [COCO17.index(n) for n in lab_names]                              # chi so hang regressor COCO

    dl = ds.datalist if args.max_n <= 0 else ds.datalist[:args.max_n]
    N, J = len(dl), len(lab_names)
    err = np.full(N, np.nan, dtype=np.float32)
    per_joint = np.full((N, J), np.nan, dtype=np.float32)
    n_valid = np.zeros(N, dtype=np.int32)
    n_over = np.zeros(N, dtype=np.int32)
    fit_err = np.full(N, np.nan, dtype=np.float32)
    ann_id = np.full(N, -1, dtype=np.int64)
    has_fit = np.zeros(N, dtype=bool)
    hip_diff = []                                                              # buoc 1: lech 2 regressor (px luoi 64)

    for i, data in enumerate(tqdm(dl, desc=name)):
        f = get_fit(name, data)
        if f is None:
            continue
        sp, cp, fe = f
        has_fit[i] = True
        fit_err[i] = fe
        n_over[i] = int(data.get('num_overlap', 0))
        ann_id[i] = int(data.get('ann_id', -1))
        img_shape = data['img_shape']
        bbox = data['bbox']
        if cfg.update_bbox:                                                     # giong __getitem__
            bbox = process_bbox(data['tight_bbox'], img_shape[1], img_shape[0])
        if bbox is None:
            continue
        trans, _, _, _ = augmentation_noimg(bbox, 'test', img_shape)           # scale 1, rot 0, khong lat
        mesh, _, _, _ = ds.get_smpl_coord(sp, cp, False, img_shape)            # (6890, 3) he camera NeuralAnnot
        mesh = np.asarray(mesh, dtype=np.float64)

        j3 = np.dot(reg, mesh)                                                  # (17, 3) khop COCO 3D
        j2 = to_hm(cam2pixel(j3, cp['focal'], cp['princpt'])[:, :2], trans)     # (17, 2) luoi 64
        lab = np.asarray(data['joint_img'], dtype=np.float64)
        lab_xy = to_hm(lab[lab_idx, :2], trans)                                 # (J, 2) luoi 64
        v = np.asarray(data['joint_valid'], dtype=np.float32).reshape(-1)[lab_idx] > 0
        d = np.linalg.norm(j2[reg_idx] - lab_xy, axis=1)                        # (J,)
        per_joint[i, v] = d[v]
        n_valid[i] = int(v.sum())
        if n_valid[i] > 0:
            err[i] = float(d[v].mean())

        if reg_hip is not None and len(hip_diff) < args.n_reg_check:
            jh = to_hm(cam2pixel(np.dot(reg_hip, mesh), cp['focal'], cp['princpt'])[:, :2], trans)
            hip_diff.append(np.linalg.norm(jh - j2, axis=1))                    # (17,)

    # ---- Buoc 1: lech giua 2 regressor ----
    if hip_diff:
        hd = np.stack(hip_diff).mean(0)
        print(f'\n[{name}] Lech TB J_regressor_coco vs J_regressor_coco_hip_smpl ({len(hip_diff)} mesh, px luoi 64):')
        print('   ' + ' | '.join(f'{COCO17[j]} {hd[j]:.2f}' for j in range(17)))

    # ---- Buoc 2: thong ke sai so fit ----
    has = ~np.isnan(err)
    print(f'\n[{name}] nguoi trong datalist {N} | co fit va co >= 1 khop nhan: {int(has.sum())} '
          f'| co fit nhung 0 khop nhan: {int((has_fit & ~has).sum())} | khong co fit: {int((~has_fit).sum())}')
    groups = (('tat ca', has), ('num_overlap = 0', has & (n_over == 0)), ('num_overlap > 0', has & (n_over > 0)))
    for gname, m in groups:
        e = err[m]
        if e.size == 0:
            print(f'   {gname:16s}: 0 mau')
            continue
        q = np.percentile(e, [10, 50, 90, 99])
        drop = ' | '.join(f'>{t:g}px {100.0 * (e > t).mean():5.1f}%' for t in THRS)
        print(f'   {gname:16s}: n {e.size:6d} | p10 {q[0]:.2f} p50 {q[1]:.2f} p90 {q[2]:.2f} p99 {q[3]:.2f} | bi loc: {drop}')
    pj = np.nanmean(per_joint[has], axis=0)
    print('   sai so TB tung khop (px luoi 64): ' + ' | '.join(f'{lab_names[j]} {pj[j]:.2f}' for j in range(J)))
    if name == 'CrowdPose':
        fe_ok = ~np.isnan(fit_err) & has
        if fe_ok.any():
            rule = fit_err[fe_ok] < CROWD_FIT_THR
            print(f'   quy tac jotr_strict (fit_err < {CROWD_FIT_THR:g} -> bo; YAML dang {"BAT" if STRICT_ORIG else "TAT"}): '
                  f'loai {int(rule.sum())} / {int(fe_ok.sum())} ({100.0 * rule.mean():.1f}%) | '
                  f'sai so TB nhom bi loai {np.mean(err[fe_ok][rule]) if rule.any() else float("nan"):.2f} px, '
                  f'nhom giu {np.mean(err[fe_ok][~rule]) if (~rule).any() else float("nan"):.2f} px')
            c = np.corrcoef(fit_err[fe_ok], err[fe_ok])[0, 1] if fe_ok.sum() > 2 else float('nan')
            print(f'   tuong quan fit_err (file NeuralAnnot) vs sai so tu tinh: {c:.3f}')

    os.makedirs(args.out_dir, exist_ok=True)
    out = osp.join(args.out_dir, f'fit_err_{name}.npz')
    np.savez(out, ann_id=ann_id, err=err, n_valid=n_valid, num_overlap=n_over, per_joint=per_joint,
             fit_err=fit_err, joint_names=np.array(lab_names))
    print(f'   -> luu {out}')


def main():
    reg, reg_hip = load_coco_reg()
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        assert name in ('MSCOCO', 'CrowdPose'), f'Chi ho tro MSCOCO, CrowdPose (nhan {name})'
        run_set(name, reg, reg_hip)


if __name__ == '__main__':
    main()
