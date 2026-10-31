"""
tools/check_jotr_equivalence.py -- KIEM TRA dau vao 2D cua lifter (che do JOTR) GIONG 100% JOTR (2026-10-31).
Chay dataset GOC cua JOTR (JOTR/dataloader/*.py, cfg JOTR/config.py) va dataset cua FuseKin (data_final/*, YAML) trong
2 TIEN TRINH RIENG (trung ten module utils / config), cung mau (khop theo anh + bbox; 3DPW theo ann_id), cung seed
(np.random / random dat lai truoc moi __getitem__ -> cung bbox ngau nhien, augmentation, PoseFix).
So 'joints' (khung hm 64) + 'joints_mask' tren 17 khop H36M (JOTR: SMPL-30 -> H36M-17 theo TEN, giong
utils.h36m_adapter.convert_smpl_to_human36m). Yeu cau: mask trung 100%, |lech toa do| ~ 0 tren khop mask > 0.

Chuan bi tren server (1 lan, tu thu muc goc FuseKin; JOTR/ = repo JOTR goc):
  R=$(pwd)/data_final
  for d in Human36M MuCo MSCOCO CrowdPose; do mkdir -p JOTR/data/$d; for f in $R/$d/*; do
      case "$f" in *.py|*__pycache__) continue;; esac; [ -e JOTR/data/$d/$(basename $f) ] || ln -s $f JOTR/data/$d/; done; done
  [ -e JOTR/data/3DPW ] || ln -s $R/PW3D JOTR/data/3DPW
  [ -e JOTR/data/J_regressor_extra.npy ] || ln -s $R/J_regressor_extra.npy JOTR/data/
  [ -e JOTR/common/utils/smplpytorch ] || ln -s $(pwd)/smplpytorch JOTR/common/utils/smplpytorch
Chay (moi truong arts):
  python tools/check_jotr_equivalence.py --cfg config/finetune_difflifter_d512_e40_jotr.yaml --jotr_root JOTR --n 200
  (--sets Human36M,MuCo,MSCOCO,CrowdPose,3dpw,3dpw-oc,3dpw-pc,3dpw-crowd mac dinh; ket qua: <out_dir>/report.txt)
"""
import argparse
import os
import os.path as osp
import pickle
import random
import subprocess
import sys

import numpy as np

H36M = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top',
        'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
TRAIN = ('Human36M', 'MuCo', 'MSCOCO', 'CrowdPose')
PW3D_SETS = ('3dpw', '3dpw-oc', '3dpw-pc', '3dpw-crowd')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='config/finetune_difflifter_d512_e40_jotr.yaml')
    p.add_argument('--jotr_root', default='JOTR')
    p.add_argument('--sets', default=','.join(TRAIN + PW3D_SETS))
    p.add_argument('--n', type=int, default=200, help='so mau ngau nhien moi bo')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tol', type=float, default=1e-3, help='nguong lech toa do (don vi khung hm 64)')
    p.add_argument('--out_dir', default='experiment/check_jotr_equivalence')
    p.add_argument('--side', default='all', choices=['all', 'ours', 'jotr', 'compare'])
    return p.parse_args()


def to_h36m(j, m, names):
    """Doi (N, >=2) + mask (N, 1) sang H36M-17 theo TEN; khop khong co -> 0."""
    j = np.asarray(j, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64).reshape(len(j), -1)[:, :1]
    if len(j) == len(H36M):
        return j[:, :2], m[:, 0]
    oj, om = np.zeros((len(H36M), 2)), np.zeros(len(H36M))
    for t, n in enumerate(H36M):
        if n in names:
            s = list(names).index(n)
            oj[t], om[t] = j[s, :2], m[s, 0]
    return oj, om


def key_of(name, d):
    if name in PW3D_SETS:
        return int(d['ann_id'])
    parts = osp.normpath(d['img_path']).split(os.sep)[-3:]
    return ('/'.join(parts), tuple(np.round(np.asarray(d['tight_bbox'], dtype=np.float64).reshape(-1), 2).tolist()))


def run_items(name, ds, items, joints_names):
    """items: [(key, idx, seed)] -> {key: (joints17 (17, 2), mask17 (17,))}. Dat seed truoc moi __getitem__."""
    out = {}
    for key, idx, seed in items:
        np.random.seed(seed)
        random.seed(seed)
        inputs = ds[idx][0]
        j = inputs['joints'].numpy() if hasattr(inputs['joints'], 'numpy') else inputs['joints']
        m = inputs['joints_mask'].numpy() if hasattr(inputs['joints_mask'], 'numpy') else inputs['joints_mask']
        out[key] = to_h36m(j, m, joints_names)
    return out


def side_ours(args):
    sys.path.insert(0, './lib')
    sys.path.insert(0, './')
    sys.path.insert(0, './main')
    from torchvision import transforms
    from core.config import cfg, update_config
    update_config(args.cfg)
    cfg.DATASET.pw3d_ocpc_test_only = False
    import __init_path  # noqa: F401  (them smplpytorch vao sys.path)
    from data_final.PW3D.dataset import PW3D
    from utils.jotr_dataset import TRAIN_DATASETS
    res = {}
    for name in [s for s in args.sets.split(',') if s]:
        ds = TRAIN_DATASETS[name](transforms.ToTensor(), 'train') if name in TRAIN else \
            PW3D(transforms.ToTensor(), data_name=name)
        rng = np.random.default_rng(args.seed)
        idxs = sorted(rng.choice(len(ds), min(args.n, len(ds)), replace=False).tolist())
        items = [(key_of(name, ds.datalist[i]), i, args.seed * 100000 + k) for k, i in enumerate(idxs)]
        names = ds.joints_name if name in PW3D_SETS else H36M      # lifter tra H36M-17 o bo train
        res[name] = {'items': [(k, s) for k, _, s in items], 'out': run_items(name, ds, items, names),
                     'len': len(ds)}
        print(f'[ours] {name}: {len(items)} mau / {len(ds)}', flush=True)
    with open(osp.join(args.out_dir, 'ours.pkl'), 'wb') as f:
        pickle.dump(res, f)


def side_jotr(args):
    with open(osp.join(args.out_dir, 'ours.pkl'), 'rb') as f:
        ours = pickle.load(f)
    root = osp.abspath(args.jotr_root)
    os.chdir(root)
    sys.path.insert(0, root)
    from config import cfg                     # JOTR/config.py (them common/ + data/* vao sys.path)
    cfg.update_bbox = True                     # giong JOTR assets/yaml/main_train.yml, fine_tune.yml
    import importlib
    from torchvision import transforms
    res = {}
    for name in [s for s in args.sets.split(',') if s]:
        if name in TRAIN:
            mod = importlib.import_module(f'dataloader.{name}')
            ds = getattr(mod, name)(transforms.ToTensor(), 'train')
        else:
            mod = importlib.import_module('dataloader.PW3D')
            ds = mod.PW3D(transforms.ToTensor(), name)
        index = {}
        for i, d in enumerate(ds.datalist):
            index.setdefault(key_of(name, d), i)
        items, missing = [], 0
        for key, seed in ours[name]['items']:
            if key not in index:
                missing += 1
                continue
            items.append((key, index[key], seed))
        res[name] = {'out': run_items(name, ds, items, ds.joints_name), 'missing': missing, 'len': len(ds)}
        print(f'[jotr] {name}: {len(items)} mau khop (thieu {missing}) / {len(ds)}', flush=True)
    with open(osp.join(args.out_dir, 'jotr.pkl'), 'wb') as f:
        pickle.dump(res, f)


def side_compare(args):
    with open(osp.join(args.out_dir, 'ours.pkl'), 'rb') as f:
        ours = pickle.load(f)
    with open(osp.join(args.out_dir, 'jotr.pkl'), 'rb') as f:
        jotr = pickle.load(f)
    lines, ok_all = [], True
    for name in ours:
        if name not in jotr:
            continue
        o, j = ours[name]['out'], jotr[name]['out']
        n, mask_bad, coord_bad, maxd, ex = 0, 0, 0, 0.0, []
        for k, (oj, om) in o.items():
            if k not in j:
                continue
            n += 1
            jj, jm = j[k]
            mb = (om > 0) != (jm > 0)
            both = (om > 0) & (jm > 0)
            d = np.abs(oj - jj).max(1) * both
            if mb.any():
                mask_bad += 1
            if (d > args.tol).any():
                coord_bad += 1
            maxd = max(maxd, float(d.max()) if d.size else 0.0)
            if (mb.any() or (d > args.tol).any()) and len(ex) < 5:
                bad = [H36M[t] + ('[mask %d vs %d]' % (om[t] > 0, jm[t] > 0) if mb[t] else '[lech %.3f]' % d[t])
                       for t in range(len(H36M)) if mb[t] or d[t] > args.tol]
                ex.append(f'      {k}: ' + ', '.join(bad))
        ok = n > 0 and mask_bad == 0 and coord_bad == 0 and jotr[name]['missing'] == 0
        ok_all &= ok
        lines.append(f'[{"OK " if ok else "LECH"}] {name:<11} | so {n} mau (JOTR thieu {jotr[name]["missing"]}) | '
                     f'mask lech {mask_bad} | toa do lech > {args.tol} {coord_bad} | lech max {maxd:.4f} | '
                     f'so mau tap: ours {ours[name]["len"]} / JOTR {jotr[name]["len"]}')
        lines += ex
    lines.append('KET LUAN: ' + ('KHOP 100% voi JOTR' if ok_all else 'CON LECH -> xem cac dong LECH'))
    txt = '\n'.join(lines)
    print(txt)
    with open(osp.join(args.out_dir, 'report.txt'), 'w') as f:
        f.write(txt + '\n')


def main():
    args = parse_args()
    args.out_dir = osp.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.side == 'ours':
        side_ours(args)
    elif args.side == 'jotr':
        side_jotr(args)
    elif args.side == 'compare':
        side_compare(args)
    else:
        base = [sys.executable, osp.abspath(__file__), '--cfg', args.cfg, '--jotr_root', args.jotr_root,
                '--sets', args.sets, '--n', str(args.n), '--seed', str(args.seed), '--tol', str(args.tol),
                '--out_dir', args.out_dir]
        subprocess.run(base + ['--side', 'ours'], check=True)
        subprocess.run(base + ['--side', 'jotr'], check=True)
        side_compare(args)


if __name__ == '__main__':
    main()
