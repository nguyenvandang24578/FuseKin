"""
tools/dump_pw3d_gt2d.py (2026-11-17, nguoi dung yeu cau) -- Xuat 2D GT H36M-17 (pixel anh goc) cho tung nguoi 3DPW,
de tools/extract_vitpose_3dpw.py --bbox_source gt2d tao bbox tu 2D GT.

Dinh nghia 2D GT GIONG FuseKin (data_final/PW3D/dataset.py, nhanh danh gia): mesh SMPL GT (get_smpl_coord, camera, met)
-> J_regressor_h36m_correct -> 17 khop H36M -> cam2pixel(focal, princpt). Gom ca khop bi che / ngoai anh.
Chay trong env FuseKin (jotr_new), tu thu muc goc FuseKin; KHONG doc anh, chi chay SMPL tren CPU.

Dau ra: <PW3D>/3DPW_{tag}_gt2d_h36m.json = { "<ann_id>": [[x, y, in_img] x 17], ... }
  tag: train (3dpw-train) | validation (3dpw-val, gom ca Crowd) | test (3dpw) | oc (3dpw-oc) | pc (3dpw-pc)
  in_img = 1 neu khop nam trong anh, 0 neu ngoai anh.
  Nguoi bi PW3D bo khoi datalist (process_bbox None) khong co trong file -> extract_vitpose_3dpw quay ve bbox GT.

Vi du:
  python tools/dump_pw3d_gt2d.py --tags train validation test oc pc
"""
import os
import os.path as osp
import sys
import json
import argparse

sys.path.insert(0, './lib')
sys.path.insert(0, './')

parser = argparse.ArgumentParser(description='Xuat 2D GT H36M-17 cho 3DPW')
parser.add_argument('--cfg', type=str, default='', help='(tuy chon) YAML de lay DATASET.jotr_data_root')
parser.add_argument('--tags', nargs='+', default=['train', 'validation', 'test', 'oc', 'pc'])
parser.add_argument('--out_dir', default='', help="mac dinh = <jotr_data_root>/PW3D")
args = parser.parse_args()

import numpy as np
from tqdm import tqdm

from core.config import cfg, update_config
if args.cfg:
    update_config(args.cfg)
cfg.DATASET.pw3d_ocpc_test_only = False     # OC / PC toan bo sequence (giong file Halpe / ViTPose)
cfg.DATASET.pw3d_2d_source = 'openpose'     # chi de khoi tao dataset; 2D detector khong dung o day

from data_final.PW3D.dataset import PW3D
from utils.transforms import cam2pixel

TAG2NAME = {'train': '3dpw-train', 'validation': '3dpw-val', 'test': '3dpw', 'oc': '3dpw-oc', 'pc': '3dpw-pc'}


def main():
    for tag in args.tags:
        assert tag in TAG2NAME, f'tag khong hop le: {tag} (chon trong {list(TAG2NAME)})'
        ds = PW3D(None, data_name=TAG2NAME[tag])
        out_dir = args.out_dir or ds.data_path
        os.makedirs(out_dir, exist_ok=True)
        res, n_out = {}, 0
        for d in tqdm(ds.datalist, desc=tag):
            mesh, _ = ds.get_smpl_coord(d['smpl_param'])                       # (6890, 3) met, toa do camera
            j3d = np.dot(ds.h36m_joint_regressor, mesh)                        # (17, 3)
            cp = d['cam_param']
            j2d = cam2pixel(j3d, cp['focal'], cp['princpt'])[:, :2]            # (17, 2) pixel anh goc
            H, W = d['img_shape']
            inside = ((j2d[:, 0] >= 0) & (j2d[:, 0] < W) & (j2d[:, 1] >= 0) & (j2d[:, 1] < H)).astype(np.float32)
            n_out += int((inside == 0).sum())
            res[str(d['ann_id'])] = np.concatenate([j2d, inside[:, None]], axis=1).round(2).tolist()
        out_path = osp.join(out_dir, f'3DPW_{tag}_gt2d_h36m.json')
        with open(out_path, 'w') as f:
            json.dump(res, f)
        print(f'[{tag}] {len(res)} nguoi -> {out_path} | khop ngoai anh {n_out} / {17 * max(len(res), 1)}')


if __name__ == '__main__':
    main()
