"""
tools/oracle_2d_select_3dpw.py -- ORACLE (CHI DE DO MUC TRAN, KHONG DUNG LAM KET QUA CHINH: dung GT 2D cua tap test).
Moi nguoi: so 4 nguon 2D (Halpe cu, Halpe moi <--new_suffix>, OpenPose, Higher-HRNet [chi Crowd]) voi GT 2D (mesh SMPL
GT chieu len anh, regressor H36M) tren 12 khop tay chan, CUNG cach tinh voi main/compare_2d_sources.py:
  diem = so khop xa (> --far x kich thuoc nguoi) + 0.5 x so khop bo (conf <= nguong rieng cua nguon); hoa -> loi TB thap hon.
Chon nguon diem thap nhat -> ghi Halpe-26 (26, 3) ra 3DPW_<split>_halpe26_oracle.json (dung DATASET.pw3d_halpe_suffix '_oracle').
OpenPose / HHRNet -> Halpe-26: 17 khop dau = COCO-17; Neck / Hip = trung diem 2 vai / 2 hong (score = min); Head (dinh dau)
va 6 khop ban chan = 0 (mo hinh coi la thieu). Khop phat hien (> nguong rieng) bi nang score len >= halpe26_conf_thr + 0.01
de khong bi loai boi nguong Halpe.
Chay (moi truong arts):
  python tools/oracle_2d_select_3dpw.py --cfg ./config/train_student_v3_kd_jotr.yml --sets 3dpw,3dpw-oc,3dpw-pc,3dpw-crowd \\
      --new_suffix _sam
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
sys.path.append('./main')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import json

parser = argparse.ArgumentParser(description='ORACLE chon nguon 2D theo GT tren 3DPW (do muc tran)')
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--sets', type=str, default='3dpw,3dpw-oc,3dpw-pc,3dpw-crowd')
parser.add_argument('--new_suffix', type=str, default='_sam', help="Halpe moi: 3DPW_<tag>_halpe26<suffix>.json ('' = bo)")
parser.add_argument('--far', type=float, default=0.2)
parser.add_argument('--out_suffix', type=str, default='_oracle')
args = parser.parse_args()

import numpy as np
from tqdm import tqdm
from core.config import cfg, update_config
update_config(args.cfg)
cfg.DATASET.pw3d_2d_source = 'halpe26'
cfg.DATASET.pw3d_ocpc_test_only = False
cfg.DATASET.pw3d_halpe_suffix = ''          # dataset doc Halpe CU

import __init_path  # noqa: F401
from utils.jotr_dataset import get_test_dataset
from utils.transforms import cam2pixel
from utils.h36m_adapter import halpe26_to_human36m

LIMB = ('L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
        'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle')
COCO17 = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
          'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle')
SOURCES = ('halpe', 'halpe_new', 'openpose', 'hhrnet')


def coco17_to_halpe26(k, thr):
    """COCO-17 (17, 3) -> Halpe-26 (26, 3). Khop phat hien (> thr) nang score >= halpe26_conf_thr + 0.01."""
    k = np.asarray(k, dtype=np.float32).copy()
    hthr = float(cfg.DATASET.halpe26_conf_thr) + 0.01
    det = k[:, 2] > thr
    k[:, 2] = np.where(det, np.maximum(k[:, 2], hthr), 0.0)
    y = np.zeros((26, 3), dtype=np.float32)
    y[:17] = k
    for dst, (a, b) in ((18, (5, 6)), (19, (11, 12))):     # Neck, Hip
        if k[a, 2] > 0 and k[b, 2] > 0:
            y[dst, :2] = (k[a, :2] + k[b, :2]) / 2
            y[dst, 2] = min(k[a, 2], k[b, 2])
    return y


def run(name):
    ds = get_test_dataset(name, None).dataset
    h_idx = [ds.h36m_joints_name.index(n) for n in LIMB]
    op2coco = [ds.openpose_joints_name.index(n) for n in COCO17]
    thr = {'halpe': float(cfg.DATASET.halpe26_conf_thr), 'halpe_new': float(cfg.DATASET.halpe26_conf_thr),
           'openpose': float(ds.conf_thr), 'hhrnet': float(ds.conf_thr)}
    tag = {'3dpw-oc': 'oc', '3dpw-pc': 'pc'}.get(name, ds.data_split)
    root = os.path.join(cfg.DATASET.jotr_data_root, 'PW3D')

    def load(p):
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
        print(f'[CANH BAO] khong thay {p}')
        return None
    old = load(os.path.join(root, f'3DPW_{tag}_halpe26.json'))
    new = load(os.path.join(root, f'3DPW_{tag}_halpe26{args.new_suffix}.json')) if args.new_suffix else None
    hh = load(os.path.join(root, '3DPW_validation_crowd_hhrnet_result.json')) if name == '3dpw-crowd' else None

    out, cnt, far_b, far_a = {}, {s: 0 for s in SOURCES}, 0, 0
    for it in tqdm(ds.datalist, desc=name):
        aid = str(it['ann_id'])
        mesh, _ = ds.get_smpl_coord(it['smpl_param'])
        gt = cam2pixel(np.dot(ds.h36m_joint_regressor, mesh), it['cam_param']['focal'], it['cam_param']['princpt'])[:, :2]
        size = float(max(np.ptp(gt[:, 0]), np.ptp(gt[:, 1]), 1.0))
        gt12 = gt[h_idx]
        cand = {}                                             # nguon -> Halpe-26 (26, 3)
        if old is not None and aid in old:
            cand['halpe'] = np.asarray(old[aid], dtype=np.float32)
        if new is not None and aid in new:
            cand['halpe_new'] = np.asarray(new[aid], dtype=np.float32)
        op = np.asarray(it['openpose'], dtype=np.float32)
        if op.ndim == 2 and len(op) >= 18:
            cand['openpose'] = coco17_to_halpe26(op[op2coco], thr['openpose'])
        if hh is not None and aid in hh:
            cand['hhrnet'] = coco17_to_halpe26(np.asarray(hh[aid]['coco_joints'], dtype=np.float32).reshape(-1, 3)[:17],
                                               thr['hhrnet'])
        best, bkey = None, None
        for s in SOURCES:
            if s not in cand:
                continue
            k = halpe26_to_human36m(cand[s])[h_idx]
            t = thr[s] if s.startswith('halpe') else float(cfg.DATASET.halpe26_conf_thr)   # sau khi nang score
            det = k[:, 2] > t
            e = np.linalg.norm(k[:, :2] - gt12, axis=1) / size
            nfar = int((det & (e > args.far)).sum())
            key = (nfar + 0.5 * int((~det).sum()), float(e[det].mean()) if det.any() else 1e9)
            if s == 'halpe':
                far_b += int(nfar > 0)
            if bkey is None or key < bkey:
                best, bkey, bfar = s, key, nfar
        if best is None:
            continue
        cnt[best] += 1
        far_a += int(bfar > 0)
        out[aid] = np.round(cand[best], 3).tolist()
    p = os.path.join(root, f'3DPW_{tag}_halpe26{args.out_suffix}.json')
    with open(p, 'w') as f:
        json.dump(out, f)
    n = max(len(out), 1)
    print(f'[{name}] {len(out)} nguoi -> {p}')
    print('   chon: ' + ' | '.join(f'{s} {100 * c / n:.1f}%' for s, c in cnt.items()))
    print(f'   mau co khop xa: Halpe cu {100 * far_b / n:.1f}% -> oracle {100 * far_a / n:.1f}%')


if __name__ == '__main__':
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        run(name)
