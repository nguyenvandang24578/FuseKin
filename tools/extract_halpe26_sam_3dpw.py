"""
tools/extract_halpe26_sam_3dpw.py -- BUOC 2/2 (moi truong `pose2d`): trich Halpe-26 bang RTMPose tren anh da CHE
phan ngoai mask SAM cua dung nguoi (mask tu tools/sam_pointer_masks_3dpw.py). bbox dua vao RTMPose = bbox annotation
(nhu ban cu); --flip: trung binh voi anh lat ngang.
  --vis N > 0 : ve N nguoi (mask cua buoc 1 chay voi --vis N) -> <vis_dir>/<split>/: [Halpe cu] | [mask + Halpe moi]
                | [con tro OpenPose / Higher-HRNet]; ghi 3DPW_<split>_halpe26_samvis.json (chi cac nguoi do) de cham
                theo GT: main/compare_2d_sources.py --halpe_new_suffix _samvis
  --vis 0     : chay het, ghi 3DPW_<split>_halpe26_sam.json -> dung bang DATASET.pw3d_halpe_suffix: '_sam'
Vi du:
  python tools/extract_halpe26_sam_3dpw.py --data_root data_final/PW3D --splits test oc pc validation --vis 50 \\
      --mask_dir debug_sam_vis --flip --vis_dir debug_sam_vis/img
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
from extract_halpe26_3dpw import build_model, draw, HALPE26_SKELETON, OPENPOSE18_SKELETON, HALPE_THR  # noqa: E402
from sam_pointer_masks_3dpw import load_split, OPENPOSE_THR, HHRNET_THR  # noqa: E402

HALPE_FLIP = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (20, 21), (22, 23), (24, 25))
COCO17_SKELETON = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
                   (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='data_final/PW3D')
    p.add_argument('--splits', nargs='+', default=['test', 'oc', 'pc', 'validation'])
    p.add_argument('--mask_dir', default='data_final/PW3D/sam_masks')
    p.add_argument('--vis', type=int, default=0, help='> 0: ve cac nguoi co mask (buoc 1 --vis N), khong ghi json chinh')
    p.add_argument('--vis_dir', default='debug_sam_vis/img')
    p.add_argument('--flip', action='store_true')
    p.add_argument('--dilate', type=float, default=0.04, help='no mask theo ti le canh lon bbox')
    p.add_argument('--out_dir', default='', help='mac dinh = data_root')
    p.add_argument('--mode', default='performance', choices=['performance', 'balanced', 'lightweight'])
    p.add_argument('--onnx', default='')
    p.add_argument('--input_size', type=int, nargs=2, default=None)
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    return p.parse_args()


def load_masks(path):
    d = np.load(path)
    return {int(i): k for k, i in enumerate(d['ann_ids'])}, d


def get_mask(d, k):
    r = d['runs'][d['off'][k]:d['off'][k + 1]]
    vals = np.arange(len(r)) % 2
    return np.repeat(vals, r).astype(bool).reshape(int(d['H'][k]), int(d['W'][k]))


def masked(img, m, box, dil):
    s = max(3, int(dil * max(box[2] - box[0], box[3] - box[1])) | 1)
    md = cv2.dilate(m.astype(np.uint8), np.ones((s, s), np.uint8)).astype(bool)
    out = img.copy()
    out[~md] = 114
    return out


def rtm(model, img, box, flip):
    k, s = model(img, bboxes=[box])
    k, s = np.asarray(k)[0], np.asarray(s)[0]
    if not flip:
        return k, s
    W = img.shape[1]
    kf, sf = model(img[:, ::-1].copy(), bboxes=[[W - 1 - box[2], box[1], W - 1 - box[0], box[3]]])
    kf, sf = np.asarray(kf)[0].copy(), np.asarray(sf)[0].copy()
    kf[:, 0] = W - 1 - kf[:, 0]
    for a, b in HALPE_FLIP:
        kf[[a, b]] = kf[[b, a]]
        sf[[a, b]] = sf[[b, a]]
    return (k + kf) / 2, (s + sf) / 2


def vis_one(img, m, box, old, k, s, ann, hh, out_path):
    H, W = img.shape[:2]
    a, b, c = img.copy(), img.copy(), img.copy()
    for p in (a, b, c):
        cv2.rectangle(p, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 2)
    if old is not None:
        draw(a, old[:, :2], old[:, 2], HALPE26_SKELETON, HALPE_THR)
    b[m] = (0.5 * b[m] + 0.5 * np.array([255, 128, 0])).astype(np.uint8)
    draw(b, k, s, HALPE26_SKELETON, HALPE_THR)
    src = 'OpenPose'
    if hh is not None and str(ann['id']) in hh:
        kk = np.asarray(hh[str(ann['id'])]['coco_joints'], dtype=np.float32).reshape(-1, 3)
        draw(c, kk[:, :2], kk[:, 2], COCO17_SKELETON, HHRNET_THR)
        src = 'Higher-HRNet'
    else:
        op = np.asarray(ann.get('openpose_result', []), dtype=np.float32).reshape(-1, 3)
        if len(op) >= 18:
            draw(c, op[:18, :2], op[:18, 2], OPENPOSE18_SKELETON, OPENPOSE_THR)
    bw = box[2] - box[0]
    x0, x1 = int(max(0, box[0] - 0.6 * bw)), int(min(W, box[2] + 0.6 * bw))
    y0, y1 = int(max(0, box[1] - 40)), int(min(H, box[3] + 40))
    crops = []
    for p, t in ((a, 'Halpe cu (bbox GT)'), (b, 'SAM con tro + Halpe moi'), (c, f'Con tro: {src}')):
        q = np.ascontiguousarray(p[y0:y1, x0:x1])
        f = 480.0 / max(q.shape[0], 1)
        q = cv2.resize(q, None, fx=f, fy=f)
        draw(q, [], [], [], 0.3, title=t)
        crops.append(q)
    cv2.imwrite(out_path, np.concatenate(crops, axis=1))


def main():
    args = parse_args()
    model = build_model(args)
    out_dir = args.out_dir or args.data_root
    hh_path = osp.join(args.data_root, '3DPW_validation_crowd_hhrnet_result.json')
    for split in args.splits:
        idx, d = load_masks(osp.join(args.mask_dir, f'sam_masks_{split}.npz'))
        imgs, by_img, _ = load_split(args.data_root, split)
        old = {}
        p_old = osp.join(args.data_root, f'3DPW_{split}_halpe26.json')
        if osp.isfile(p_old):
            with open(p_old) as f:
                old = json.load(f)
        hh = None
        if split == 'validation' and osp.isfile(hh_path):
            with open(hh_path) as f:
                hh = json.load(f)
        if args.vis > 0:
            os.makedirs(osp.join(args.vis_dir, split), exist_ok=True)
        res, miss = {}, 0
        todo = [(k, v) for k, v in by_img.items() if any(a['id'] in idx for a in v)]
        for k, lst in tqdm(todo, desc=split):
            im = imgs[k]
            img = cv2.imread(osp.join(args.data_root, 'imageFiles', im['sequence'], im['file_name']))
            if img is None:
                miss += 1
                continue
            for a in lst:
                if a['id'] not in idx:
                    continue
                m = get_mask(d, idx[a['id']])
                x, y, w, h = [float(v) for v in a['bbox']]
                box = [x, y, x + w, y + h]
                kp, sc = rtm(model, masked(img, m, box, args.dilate), box, args.flip)
                res[str(a['id'])] = np.concatenate([kp, sc[:, None]], axis=1).round(3).tolist()
                if args.vis > 0:
                    o = old.get(str(a['id']))
                    o = np.asarray(o, dtype=np.float32) if o is not None else None
                    vis_one(img, m, box, o, kp, sc, a, hh,
                            osp.join(args.vis_dir, split, f'{im["sequence"]}_{a["id"]}_{im["file_name"]}'))
        suffix = '_samvis' if args.vis > 0 else '_sam'
        out = osp.join(out_dir, f'3DPW_{split}_halpe26{suffix}.json')
        with open(out, 'w') as f:
            json.dump(res, f)
        print(f'[{split}] {len(res)} nguoi -> {out} | anh loi: {miss}' + (f' | hinh: {args.vis_dir}/{split}' if args.vis else ''))


if __name__ == '__main__':
    main()
