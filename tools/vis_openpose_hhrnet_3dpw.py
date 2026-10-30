"""
tools/vis_openpose_hhrnet_3dpw.py -- Ve NGAU NHIEN --num nguoi moi tap 3DPW: so 2D Halpe-26 (dang dung) voi OpenPose
(JOTR / 3DCrowdNet dung cho 3DPW / OC / PC) va Higher-HRNet (dung cho Crowd), tren anh goc.
Moi hinh 1 nguoi, cac o: [Halpe] | [OpenPose] | [Higher-HRNet (chi Crowd)]. Ca 3 deu doi sang H36M-17 de ve cung kieu
(OpenPose / HHRNet khong co Torso, Head_top). Do = xuong GT (H36M-17 hoi quy tu mesh SMPL GT chieu len anh), xanh la =
nguon 2D (khop conf > nguong rieng), tim = bbox annotation. Tieu de o: loi TB so GT (% kich thuoc nguoi, cac khop
nguon phat hien) + so khop phat hien.
Khong can GPU / model. Chay trong moi truong FuseKin:
  python tools/vis_openpose_hhrnet_3dpw.py --cfg ./config/train_student_v3_scratch_jotr.yml \\
      --sets 3dpw,3dpw-oc,3dpw-pc,3dpw-crowd --num 20
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
sys.path.append('./main')   # __init_path nam trong main/
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--sets', type=str, default='3dpw,3dpw-oc,3dpw-pc,3dpw-crowd')
parser.add_argument('--num', type=int, default=20, help='so nguoi ngau nhien moi tap')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

import cv2
import numpy as np
from core.config import cfg, update_config
update_config(args.cfg)
cfg.DATASET.pw3d_2d_source = 'halpe26'
cfg.DATASET.pw3d_ocpc_test_only = False

import __init_path  # noqa: F401
from utils.jotr_dataset import get_test_dataset
from utils.transforms import cam2pixel, transform_joint_to_other_db

H36M_SKELETON = ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
                 (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16))


def draw_skel(img, j, valid, color, r=4, t=2):
    for a, b in H36M_SKELETON:
        if valid[a] and valid[b]:
            cv2.line(img, tuple(int(v) for v in j[a, :2]), tuple(int(v) for v in j[b, :2]), color, t)
    for k in range(len(j)):
        if valid[k]:
            cv2.circle(img, (int(j[k, 0]), int(j[k, 1])), r, color, -1)


def put(img, txt):
    cv2.putText(img, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def run(name, rng):
    ds = get_test_dataset(name, None).dataset          # PW3D
    names = ds.h36m_joints_name
    hh = None
    if name == '3dpw-crowd':
        p = os.path.join(cfg.DATASET.jotr_data_root, 'PW3D', '3DPW_validation_crowd_hhrnet_result.json')
        if os.path.isfile(p):
            with open(p) as f:
                hh = json.load(f)
    out_dir = os.path.join(cfg.output_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    idx = np.sort(rng.choice(len(ds.datalist), min(args.num, len(ds.datalist)), replace=False))
    for rank, i in enumerate(idx):
        it = ds.datalist[i]
        img = cv2.imread(it['img_path'])
        if img is None:
            continue
        mesh, _ = ds.get_smpl_coord(it['smpl_param'])
        gt = cam2pixel(np.dot(ds.h36m_joint_regressor, mesh), it['cam_param']['focal'], it['cam_param']['princpt'])
        size = float(max(np.ptp(gt[:, 0]), np.ptp(gt[:, 1]), 1.0))
        src = []
        if it['halpe_h36m'] is not None:
            src.append(('Halpe', np.asarray(it['halpe_h36m'], dtype=np.float32), float(cfg.DATASET.halpe26_conf_thr)))
        src.append(('OpenPose', transform_joint_to_other_db(np.asarray(it['openpose'], dtype=np.float32),
                                                            ds.openpose_joints_name, names), float(ds.conf_thr)))
        if hh is not None and str(it['ann_id']) in hh:
            k = np.asarray(hh[str(it['ann_id'])]['coco_joints'], dtype=np.float32)
            k = ds.add_neck(ds.add_pelvis(k, ds.coco_joints_name), ds.coco_joints_name)
            src.append(('Higher-HRNet', transform_joint_to_other_db(k, ds.coco_joints_name, names), float(ds.conf_thr)))
        bx, by, bw, bh = [float(v) for v in it['tight_bbox']]
        H, W = img.shape[:2]
        x0, x1 = int(max(0, bx - 0.6 * bw)), int(min(W, bx + 1.6 * bw))
        y0, y1 = int(max(0, by - 0.15 * bh)), int(min(H, by + 1.15 * bh))
        panels = []
        for title, j, thr in src:
            p = img.copy()
            cv2.rectangle(p, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (255, 0, 255), 2)
            draw_skel(p, gt, np.ones(17, bool), (0, 0, 255))
            v = j[:, 2] > thr
            draw_skel(p, j, v, (0, 200, 0))
            err = np.linalg.norm(j[v, :2] - gt[v, :2], axis=1).mean() / size * 100 if v.any() else float('nan')
            c = np.ascontiguousarray(p[y0:y1, x0:x1])
            if c.size == 0:
                continue
            f = 420.0 / c.shape[0]
            c = cv2.resize(c, None, fx=f, fy=f)
            put(c, f'{title} | loi {err:.0f}% | {int(v.sum())}/17 khop')
            panels.append(c)
        if panels:
            h = min(q.shape[0] for q in panels)
            seq = os.path.basename(os.path.dirname(it['img_path']))
            cv2.imwrite(os.path.join(out_dir, f'{rank:02d}_{seq}_{it["ann_id"]}.jpg'),
                        cv2.hconcat([q[:h] for q in panels]))
    print(f'[{name}] ve {len(idx)} nguoi -> {out_dir}')


def main():
    rng = np.random.default_rng(args.seed)
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        run(name, rng)
    print('Do = GT, xanh la = nguon 2D, tim = bbox annotation; loi = % kich thuoc nguoi')


if __name__ == '__main__':
    main()
