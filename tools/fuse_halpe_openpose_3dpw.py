"""
tools/fuse_halpe_openpose_3dpw.py -- Thay CA POSE RTMPose (Halpe-26) bi NHAM NGUOI bang OpenPose cua chinh nguoi do.
Khong can GPU / model, chi doc json. Chay duoc trong moi truong bat ky co numpy + opencv + tqdm.

Nhan dien nham nguoi: mismatch_op (tools/extract_halpe26_3dpw.py) = ti le khop chung (ca 2 nguon phat hien) lech
  > 0.2 x kich thuoc nguoi (theo OpenPose). mismatch > --thr -> THAY ca pose bang OpenPose.
  Giu RTMPose khi: khong co OpenPose, OpenPose < --op_min_joints khop, hoac < 4 khop chung (mismatch = nan).
OpenPose-18 -> Halpe-26: map thang 17 khop co ten trung (mui, mat, tai, vai, khuyu, co tay, hong, goi, co chan) + Neck (18);
  Hip (19) = trung diem 2 hong; Head (17) va ban chan (20-25) = score 0 (OpenPose khong co).
  Score: khop OpenPose > OPENPOSE_THR -> --op_score (mac dinh 1.0), con lai 0 va toa do 0.
Dau vao : <data_root>/3DPW_<split>_halpe26<in_suffix>.json (mac dinh _sam = Halpe SAM) + annotation (3DPW_latest_<split>.json | 3DPW_<split>.json)
Dau ra  : <data_root>/3DPW_<split>_halpe26<out_suffix>.json (mac dinh _sam_opfix) -> DATASET.pw3d_halpe_suffix: '_sam_opfix'
          (chi ap dung split khong phai train). validation dung chung cho 3dpw-val va 3dpw-crowd.
Visual  : <vis_dir>/<split>/ TAT CA nguoi bi thay, 3 o: [Halpe goc] | [OpenPose goc] | [Selected (sau thay)]
          (vang = khop Halpe lech xa OpenPose; tim = bbox GT). --no_vis de tat.
Vi du:
  python tools/fuse_halpe_openpose_3dpw.py --data_root data_final/PW3D --splits test oc pc validation
  python main/compare_2d_sources.py --cfg ./config/train_student_v3_kd_jotr.yml --sets 3dpw-pc,3dpw-crowd \\
      --halpe_new_suffix _sam_opfix
"""
import argparse
import json
import os
import os.path as osp
import sys

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from extract_halpe26_3dpw import (draw, mismatch_op, op_joints, gt_box, HALPE26_SKELETON,  # noqa: E402
                                  OPENPOSE18_SKELETON, OPENPOSE_THR, HALPE_THR, HALPE_OP_PAIRS)
from sam_pointer_masks_3dpw import load_split  # noqa: E402

# Halpe-26 <- OpenPose-18 (chi so Halpe: chi so OpenPose)
HALPE_FROM_OP = {0: 0, 1: 15, 2: 14, 3: 17, 4: 16, 5: 5, 6: 2, 7: 6, 8: 3, 9: 7, 10: 4,
                 11: 11, 12: 8, 13: 12, 14: 9, 15: 13, 16: 10, 18: 1}
SWEEP = (0.3, 0.5, 0.7)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='data_final/PW3D')
    p.add_argument('--splits', nargs='+', default=['test', 'oc', 'pc', 'validation'])
    p.add_argument('--in_suffix', default='_sam', help="Halpe dau vao: 3DPW_<split>_halpe26<in_suffix>.json (mac dinh _sam; '' = Halpe cu)")
    p.add_argument('--out_suffix', default='_sam_opfix')
    p.add_argument('--thr', type=float, default=0.5, help='mismatch > thr -> thay bang OpenPose')
    p.add_argument('--op_min_joints', type=int, default=6, help='OpenPose it hon so khop nay -> giu RTMPose')
    p.add_argument('--op_score', type=float, default=1.0, help='score gan cho khop OpenPose hop le')
    p.add_argument('--vis_dir', default='debug_opfix_vis')
    p.add_argument('--no_vis', action='store_true')
    return p.parse_args()


def op_to_halpe(op, score):
    """OpenPose (18, 3) -> Halpe-26 (26, 3) [x, y, score]."""
    h = np.zeros((26, 3), dtype=np.float32)
    for hi, oi in HALPE_FROM_OP.items():
        if op[oi, 2] > OPENPOSE_THR:
            h[hi] = (op[oi, 0], op[oi, 1], score)
    if h[11, 2] > 0 and h[12, 2] > 0:
        h[19] = ((h[11, 0] + h[12, 0]) / 2, (h[11, 1] + h[12, 1]) / 2, score)
    return h


def far_joints(k, s, op):
    """Chi so khop Halpe lech xa OpenPose (cung dinh nghia voi mismatch_op) -> to vang khi ve."""
    v = op[op[:, 2] > OPENPOSE_THR]
    size = max(np.ptp(v[:, 0]), np.ptp(v[:, 1]), 1.0)
    return [h for h, o in HALPE_OP_PAIRS
            if s[h] > HALPE_THR and op[o, 2] > OPENPOSE_THR and np.linalg.norm(k[h] - op[o, :2]) > 0.2 * size]


def save_vis(img, ann, old, op, new, m, out_path, args_in=''):
    a, b, c = img.copy(), img.copy(), img.copy()
    g = [int(v) for v in gt_box(ann)]
    for im in (a, b, c):
        cv2.rectangle(im, (g[0], g[1]), (g[2], g[3]), (255, 0, 255), 2)
    draw(a, old[:, :2], old[:, 2], HALPE26_SKELETON, HALPE_THR, highlight=far_joints(old[:, :2], old[:, 2], op))
    draw(b, op[:, :2], op[:, 2], OPENPOSE18_SKELETON, OPENPOSE_THR)
    draw(c, new[:, :2], new[:, 2], HALPE26_SKELETON, HALPE_THR)
    # cat quanh bbox GT + khop cua ca 2 nguon (khop nham nguoi co the nam ngoai bbox GT)
    pts = [old[old[:, 2] > HALPE_THR, :2], op[op[:, 2] > OPENPOSE_THR, :2], np.array([g[:2], g[2:]], np.float32)]
    pts = np.concatenate([q for q in pts if len(q)], axis=0)
    H, W = img.shape[:2]
    x0, y0 = max(0, int(pts[:, 0].min()) - 40), max(0, int(pts[:, 1].min()) - 50)
    x1, y1 = min(W, int(pts[:, 0].max()) + 40), min(H, int(pts[:, 1].max()) + 40)
    titles = [f'Halpe{args_in} (lech OP {m:.0%})', 'OpenPose goc', 'Selected = OpenPose']
    crops = [np.ascontiguousarray(p[y0:y1, x0:x1]) for p in (a, b, c)]
    crops = [draw(cr, [], [], [], HALPE_THR, title=t) for cr, t in zip(crops, titles)]
    cv2.imwrite(out_path, np.concatenate(crops, axis=1))


def run_split(args, split):
    in_path = osp.join(args.data_root, f'3DPW_{split}_halpe26{args.in_suffix}.json')
    if not osp.isfile(in_path):
        raise FileNotFoundError(f'{in_path} khong ton tai')
    with open(in_path) as f:
        halpe = json.load(f)
    imgs, by_img, anns = load_split(args.data_root, split)
    if not any('openpose_result' in a for a in anns):
        raise KeyError(f'[{split}] annotation khong co truong openpose_result -> khong thay duoc')
    print(f'[{split}] Halpe: {in_path} ({len(halpe)} nguoi) | annotation: {len(anns)} nguoi')

    out = dict(halpe)          # giu nguyen moi nguoi khong bi thay
    vis_dir = osp.join(args.vis_dir, split)
    st = {'n': 0, 'no_op': 0, 'nan': 0, 'replaced': 0, 'missing': 0}
    sweep = {t: 0 for t in SWEEP}
    for image_id, a_list in tqdm(by_img.items(), desc=split):
        img = None
        for ann in a_list:
            key = str(ann['id'])
            if key not in halpe:
                st['missing'] += 1
                continue
            st['n'] += 1
            old = np.asarray(halpe[key], dtype=np.float32)
            op = op_joints(ann)
            if op is None or (op[:, 2] > OPENPOSE_THR).sum() < args.op_min_joints:
                st['no_op'] += 1
                continue
            m = mismatch_op(old[:, :2], old[:, 2], ann)
            if np.isnan(m):
                st['nan'] += 1
                continue
            for t in SWEEP:
                sweep[t] += m > t
            if m <= args.thr:
                continue
            new = op_to_halpe(op, args.op_score)
            out[key] = new.round(3).tolist()
            st['replaced'] += 1
            if not args.no_vis:
                if img is None:
                    im = imgs[image_id]
                    img = cv2.imread(osp.join(args.data_root, 'imageFiles', im['sequence'], im['file_name']))
                if img is not None:
                    os.makedirs(vis_dir, exist_ok=True)
                    im = imgs[image_id]
                    name = f'{m * 100:03.0f}_{im["sequence"]}_{key}_{osp.splitext(im["file_name"])[0]}.jpg'
                    save_vis(img, ann, old, op, new, m, osp.join(vis_dir, name), args.in_suffix)

    n = max(st['n'], 1)
    print(f'[{split}] n={st["n"]} | thay {st["replaced"]} ({100 * st["replaced"] / n:.2f}%, thr={args.thr}) | '
          f'khong OpenPose {st["no_op"]} | <4 khop chung {st["nan"]} | ann khong co trong Halpe {st["missing"]}')
    print(f'[{split}] so nguoi se bi thay theo nguong: '
          + ', '.join(f'>{t}: {sweep[t]} ({100 * sweep[t] / n:.2f}%)' for t in SWEEP))
    out_path = osp.join(args.data_root, f'3DPW_{split}_halpe26{args.out_suffix}.json')
    with open(out_path, 'w') as f:
        json.dump(out, f)
    print(f'[{split}] luu {len(out)} nguoi -> {out_path}' + ('' if args.no_vis else f' | hinh: {vis_dir}'))


def main():
    args = parse_args()
    if args.out_suffix == args.in_suffix:
        raise SystemExit('--out_suffix trung --in_suffix -> se ghi de file dau vao')
    for split in args.splits:
        run_split(args, split)


if __name__ == '__main__':
    main()
