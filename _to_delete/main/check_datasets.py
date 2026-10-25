"""
check_datasets.py — Kiem tra du lieu TRUOC khi train full (H36M + MuCo + COCO + CrowdPose [+ 3DPW-train])
va test 3DPW / 3DPW-OC / 3DPW-PC / 3DPW-Crowd. Chi doc du lieu (khong can GPU, khong train).

In ra:
  [A] Moi tap train (qua wrapper Human36M17Dataset): co dung duoc khong, key + shape, ti le khop 2D hop le,
      is_3D / is_valid_fit, don vi orig_joint_cam, sai lech orig (3D that) vs fit (SMPL) theo khop,
      lech goc mesh (pelvis H36M vs khop 0 SMPL) neu co smpl_mesh_cam
  [B] Ghep batch giua cac tap (default_collate) -> co loi key / shape khong
  [C] Moi tap test: dung duoc khong, so mau, nguon 2D thuc su dung (Halpe / OpenPose / Higher-HRNet)
  [D] 3DPW-OC / PC: ann_id + sequence trung voi train / validation / test? co du trong file Halpe test, RootNet?

Chay (tu thu muc goc repo):
    python main/check_datasets.py --cfg ./config/train_teacher_hint_k16_v2.yml
    # tuy chon: --train Human36M MuCo MSCOCO CrowdPose 3dpw-train  --test 3dpw 3dpw-oc 3dpw-pc 3dpw-crowd  --n 64
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
import json
import random
import traceback
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Kiem tra dong nhat du lieu train/test (FuseKin)')
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--train', type=str, nargs='*', default=['Human36M', 'MuCo', 'MSCOCO', 'CrowdPose', '3dpw-train'])
parser.add_argument('--test', type=str, nargs='*', default=['3dpw', '3dpw-oc', '3dpw-pc', '3dpw-crowd'])
parser.add_argument('--n', type=int, default=64, help='so mau ngau nhien doc tu moi tap train')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from utils.jotr_dataset import get_train_dataset, get_test_dataset
from utils.h36m_adapter import HUMAN36M_JOINTS as H

random.seed(args.seed)
np.random.seed(args.seed)


def hr(t):
    print('\n' + '=' * 100)
    print(t)
    print('=' * 100)


def arr(v):
    return v.numpy() if torch.is_tensor(v) else np.asarray(v)


def shape_of(v):
    if torch.is_tensor(v) or isinstance(v, np.ndarray):
        return f'{tuple(v.shape)} {str(v.dtype).replace("torch.", "")}'
    return type(v).__name__


def per_joint(vals, mask=None, fmt='{:.2f}'):
    """vals (n, 17) -> chuoi 'Ten:gia_tri' theo khop H36M (TB co trong so mask)."""
    out = []
    for j in range(vals.shape[1]):
        if mask is None:
            v = vals[:, j].mean()
        else:
            m = mask[:, j] > 0
            v = vals[m, j].mean() if m.any() else float('nan')
        out.append(f'{H[j]}:' + fmt.format(v))
    return ' '.join(out)


# ============================================================
# [A] Tap train
# ============================================================
def check_train(name):
    hr(f'[A] TRAIN: {name}')
    try:
        ds = get_train_dataset(name, None)
    except Exception as e:
        print(f'  [LOI] Khong tao duoc dataset: {type(e).__name__}: {e}')
        traceback.print_exc(limit=4)
        return None
    print(f'  so mau: {len(ds)}')
    idx = random.sample(range(len(ds)), min(args.n, len(ds)))
    samples, n_err = [], 0
    for i in idx:
        try:
            samples.append(ds[i])
        except Exception as e:
            n_err += 1
            if n_err <= 3:
                print(f'  [LOI] doc mau {i}: {type(e).__name__}: {e}')
                if n_err == 1:
                    traceback.print_exc(limit=4)
    print(f'  doc duoc {len(samples)}/{len(idx)} mau')
    if not samples:
        return None
    inp, tgt, meta = samples[0]
    for tag, d in (('inputs', inp), ('targets', tgt), ('meta', meta)):
        print(f'  {tag:7s}: ' + ', '.join(f'{k}{shape_of(v)}' for k, v in d.items()))

    jm = np.stack([arr(s[0]['joints_mask']).reshape(-1) for s in samples])
    print('  2D vao - ti le khop hop le: ' + per_joint(jm))
    j2 = np.stack([arr(s[0]['joints'])[:, :2] for s in samples])
    print(f'  2D vao - mien gia tri (khop hop le): [{j2[jm > 0].min():.2f}, {j2[jm > 0].max():.2f}] (ky vong ~[-1, 1])')

    is3d = np.array([float(s[2].get('is_3D', np.nan)) for s in samples])
    fit = np.array([float(s[2].get('is_valid_fit', np.nan)) for s in samples])
    print(f'  is_3D TB = {np.nanmean(is3d):.2f} | is_valid_fit TB = {np.nanmean(fit):.2f}')

    if 'orig_joint_cam' in tgt:
        oc = np.stack([arr(s[1]['orig_joint_cam']) for s in samples])
        ov = np.stack([arr(s[2]['orig_joint_valid']).reshape(-1) for s in samples]) if 'orig_joint_valid' in meta \
            else np.ones(oc.shape[:2])
        print(f'  orig_joint_cam: max|x| = {np.abs(oc).max():.3f} (met neu < ~2) | |root| TB = '
              f'{np.linalg.norm(oc[:, 0], axis=-1).mean():.4f} | toan 0 o {np.mean(np.abs(oc).sum((1, 2)) == 0):.0%} mau')
        print('  orig_joint_valid theo khop: ' + per_joint(ov))
    if 'fit_joint_cam' in tgt:
        fc = np.stack([arr(s[1]['fit_joint_cam']) for s in samples])
        ft = np.stack([arr(s[2]['fit_joint_trunc']).reshape(-1) for s in samples])
        print(f'  fit_joint_cam: max|x| = {np.abs(fc).max():.3f} | fit_joint_trunc theo khop: ' + per_joint(ft))
        if 'orig_joint_cam' in tgt:
            m = (is3d[:, None] > 0.5) & (fit[:, None] > 0.5) & (ov > 0) & (ft > 0)
            if m.any():
                d = np.linalg.norm(oc - fc, axis=-1) * 1000
                print('  |orig - fit| (mm, mau 3D + fit hop le) theo khop: ' + per_joint(d, m, '{:.0f}'))
                print('    (PW3D = 0 vi fit lay tu chinh mesh qua regressor H36M; tap khac lon -> fit la khop SMPL, '
                      'KHONG cung dinh nghia voi khop H36M ma loss so sanh)')
    if 'smpl_mesh_cam' in tgt:
        mesh = np.stack([arr(s[1]['smpl_mesh_cam']) for s in samples])
        jr = ds.joint_regressor
        jr = jr.toarray() if hasattr(jr, 'toarray') else np.asarray(jr)
        r_smpl0 = np.einsum('v,nvc->nc', jr[0], mesh)
        r_h36m = np.einsum('v,nvc->nc', np.asarray(ds.joint_regressor_human36)[0], mesh)
        off = np.linalg.norm(r_smpl0 - r_h36m, axis=-1) * 1000
        print(f'  smpl_mesh_cam: goc dang dung ~ pelvis H36M (|pelvis H36M| TB = {np.linalg.norm(r_h36m, axis=-1).mean() * 1000:.1f} mm); '
              f'lech khop 0 SMPL - pelvis H36M = {off.mean():.1f} mm (max {off.max():.1f})')
        print('    (mesh du doan root-rel theo khop 0 SMPL -> mesh_loss bi lech hang so neu 2 goc khac nhau)')
    else:
        print('  [!] KHONG co targets["smpl_mesh_cam"] -> Teacher/Student_Trainer doc key nay -> KeyError khi train')
    return samples[0]


# ============================================================
# [C] Tap test
# ============================================================
def check_test(name):
    hr(f'[C] TEST: {name}')
    try:
        ds = get_test_dataset(name, None)
    except Exception as e:
        print(f'  [LOI] Khong tao duoc dataset: {type(e).__name__}: {e}')
        traceback.print_exc(limit=4)
        return
    base = ds.dataset
    dl = base.datalist
    n_halpe = sum(d.get('halpe_h36m') is not None for d in dl)
    n_hh = sum(d.get('hhrnetpose') is not None for d in dl)
    print(f'  so mau: {len(ds)} | split {base.data_split} | cfg.crowd sau khi tao = {cfg.crowd} | '
          f'co Halpe: {n_halpe} | co Higher-HRNet: {n_hh}')
    if n_halpe:
        src = 'Halpe-26'
    elif cfg.crowd and base.data_split == 'validation' and n_hh:
        src = 'Higher-HRNet'
    else:
        src = 'OpenPose'
    print(f'  nguon 2D se dung luc __getitem__ (voi cfg.crowd HIEN TAI): {src}')
    try:
        inp, tgt, meta = ds[0]
        m = arr(inp['joints_mask']).reshape(-1)
        print(f"  mau 0: joints{shape_of(inp['joints'])} | khop hop le {int(m.sum())}/17 | targets: {list(tgt)}")
    except Exception as e:
        print(f'  [LOI] doc mau 0: {type(e).__name__}: {e}')
        traceback.print_exc(limit=4)


# ============================================================
# [D] 3DPW-OC / PC: trung lap + do phu file
# ============================================================
def load_json(p):
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def ann_and_seq(js):
    imgs = {im['id']: im for im in js['images']}
    aids = {a['id'] for a in js['annotations']}
    seqs = {imgs[a['image_id']].get('sequence') for a in js['annotations'] if a['image_id'] in imgs}
    return aids, seqs


def check_overlap():
    hr('[D] 3DPW-OC / PC: trung voi split nao, co du file Halpe / RootNet khong')
    root = os.path.join(cfg.DATASET.jotr_data_root, 'PW3D')
    base = {}
    for sp in ('train', 'validation', 'test'):
        js = load_json(os.path.join(root, f'3DPW_latest_{sp}.json'))
        if js is None:
            print(f'  [!] thieu 3DPW_latest_{sp}.json')
            continue
        base[sp] = ann_and_seq(js)
    halpe_test = load_json(os.path.join(root, '3DPW_test_halpe26.json'))
    rootnet = load_json(os.path.join(root, 'rootnet_output', 'bbox_root_pw3d_output.json'))
    rootnet_ids = {str(a['ann_id']) for a in rootnet} if rootnet else None
    for nm in ('oc', 'pc'):
        js = load_json(os.path.join(root, f'3DPW_{nm}.json'))
        if js is None:
            print(f'  [!] thieu 3DPW_{nm}.json')
            continue
        aids, seqs = ann_and_seq(js)
        print(f'  3DPW-{nm.upper()}: {len(aids)} nguoi, {len(seqs)} sequence')
        for sp, (a2, s2) in base.items():
            print(f'    trung {sp:10s}: ann_id {len(aids & a2):6d} | sequence {len(seqs & s2):3d}'
                  + (f'  -> {sorted(seqs & s2)[:6]}' if sp == 'train' and (seqs & s2) else ''))
        if halpe_test is not None:
            miss = [a for a in aids if str(a) not in halpe_test]
            print(f'    thieu trong 3DPW_test_halpe26.json: {len(miss)}/{len(aids)}'
                  + (' -> tap test nay se LOI khi bat Halpe' if miss else ''))
        if rootnet_ids is not None:
            miss = [a for a in aids if str(a) not in rootnet_ids]
            print(f'    thieu trong RootNet bbox (split test dung bbox nay): {len(miss)}/{len(aids)}')
    print('  Neu OC/PC trung sequence voi train -> train co 3dpw-train thi KHONG duoc bao cao tren OC/PC do.')


def main():
    hr(f'Cau hinh: pw3d_2d_source = {cfg.DATASET.pw3d_2d_source} | lift_only = {cfg.DATASET.lift_only} | '
       f'data_root = {cfg.DATASET.jotr_data_root}')
    firsts = {}
    for name in args.train:
        s = check_train(name)
        if s is not None:
            firsts[name] = s

    if len(firsts) >= 2:
        hr('[B] GHEP BATCH GIUA CAC TAP (default_collate, nhu MultipleDatasets + DataLoader)')
        names = list(firsts)
        for a in names:
            for b in names:
                if a >= b:
                    continue
                for pair in ((a, b), (b, a)):
                    try:
                        default_collate([firsts[pair[0]], firsts[pair[1]]])
                        print(f'  {pair[0]:>10s} + {pair[1]:<10s}: OK')
                    except Exception as e:
                        print(f'  {pair[0]:>10s} + {pair[1]:<10s}: LOI {type(e).__name__}: {str(e)[:120]}')
        keys = {n: (set(s[0]), set(s[1]), set(s[2])) for n, s in firsts.items()}
        for i, tag in enumerate(('inputs', 'targets', 'meta')):
            common = set.intersection(*[k[i] for k in keys.values()])
            print(f'  {tag}: key chung = {sorted(common)}')
            for n, k in keys.items():
                extra = sorted(k[i] - common)
                if extra:
                    print(f'    {n} co them: {extra}')

    for name in args.test:
        check_test(name)
    check_overlap()


if __name__ == '__main__':
    main()
