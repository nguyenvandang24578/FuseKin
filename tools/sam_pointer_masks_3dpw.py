"""
tools/sam_pointer_masks_3dpw.py -- BUOC 1/2 (moi truong co torch, vd `arts`): mask SAM cua DUNG nguoi cho 3DPW.
Prompt SAM = bbox annotation + "con tro" danh tinh = khop 2D bottom-up DA GAN SAN cho nguoi do (cung nguon JOTR /
3DCrowdNet dung khi test): Higher-HRNet (neu co, tap Crowd) hoac OpenPose (annotation). Khong co con tro (< 3 khop)
-> chi bbox. Y tuong: BUCTD (pose bottom-up lam con tro) + PromptHMR (mask tot hon box khi nguoi chong nhau).
Ghi <out_dir>/sam_masks_<split>.npz: ann_ids, score, n_ptr (so diem con tro), H, W, runs + off (mask nen run-length,
hang truoc, bat dau bang so diem 0).
  --vis N : chi N nguoi NGAU NHIEN moi split (seed co dinh) -> de buoc 2 ve hinh; 0 = TAT CA nguoi.
Buoc 2 (moi truong pose2d): tools/extract_halpe26_sam_3dpw.py --mask_dir <out_dir> (cung --splits, cung --vis).
Cai SAM1 (Python 3.8): pip install --no-deps segment-anything ; checkpoint sam_vit_h_4b8939.pth
Vi du:
  python tools/sam_pointer_masks_3dpw.py --data_root data_final/PW3D --splits test oc pc validation --vis 50 \\
      --sam_ckpt checkpoints/sam_vit_h_4b8939.pth --out_dir debug_sam_vis
  (chay het: bo --vis, --out_dir data_final/PW3D/sam_masks)
"""
import argparse
import json
import os
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

OPENPOSE_THR = 0.05      # giong PW3D.conf_thr
HHRNET_THR = 0.05


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='data_final/PW3D')
    p.add_argument('--splits', nargs='+', default=['test', 'oc', 'pc', 'validation'])
    p.add_argument('--vis', type=int, default=0, help='N > 0: chi N nguoi ngau nhien moi split (de ve); 0 = tat ca')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--sam_type', default='vit_h')
    p.add_argument('--sam_ckpt', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out_dir', default='data_final/PW3D/sam_masks')
    return p.parse_args()


def load_split(data_root, split):
    f = osp.join(data_root, f'3DPW_latest_{split}.json')
    if not osp.exists(f):
        f = osp.join(data_root, f'3DPW_{split}.json')
    with open(f) as fh:
        db = json.load(fh)
    imgs = {im['id']: im for im in db['images']}
    by_img = defaultdict(list)
    for a in db['annotations']:
        by_img[a['image_id']].append(a)
    return imgs, by_img, db['annotations']


def pointer_points(ann, hh):
    """Khop bottom-up cua nguoi do: Higher-HRNet (neu co) roi toi OpenPose."""
    if hh is not None and str(ann['id']) in hh:
        k = np.asarray(hh[str(ann['id'])]['coco_joints'], dtype=np.float32).reshape(-1, 3)
        return [tuple(p[:2]) for p in k if p[2] > HHRNET_THR]
    op = np.asarray(ann.get('openpose_result', []), dtype=np.float32).reshape(-1, 3)
    return [tuple(p[:2]) for p in op[:18] if p[2] > OPENPOSE_THR] if len(op) >= 18 else []


def rle_encode(m):
    flat = m.reshape(-1).astype(np.int8)
    d = np.flatnonzero(np.diff(flat)) + 1
    runs = np.diff(np.concatenate([[0], d, [flat.size]]))
    if flat[0] == 1:
        runs = np.concatenate([[0], runs])
    return runs.astype(np.int32)


def main():
    args = parse_args()
    import torch  # noqa: F401
    from segment_anything import sam_model_registry, SamPredictor
    pred = SamPredictor(sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt).to(args.device))
    os.makedirs(args.out_dir, exist_ok=True)
    hh_path = osp.join(args.data_root, '3DPW_validation_crowd_hhrnet_result.json')
    for split in args.splits:
        imgs, by_img, anns = load_split(args.data_root, split)
        hh = None
        if split == 'validation' and osp.isfile(hh_path):
            with open(hh_path) as f:
                hh = json.load(f)
        if args.vis > 0:
            rng = np.random.default_rng(args.seed)
            keep = {anns[i]['id'] for i in rng.choice(len(anns), min(args.vis, len(anns)), replace=False)}
        else:
            keep = None
        ids, scores, nptr, hs, ws, runs, off = [], [], [], [], [], [], [0]
        todo = [(k, v) for k, v in by_img.items() if keep is None or any(a['id'] in keep for a in v)]
        for k, lst in tqdm(todo, desc=split):
            im = imgs[k]
            img = cv2.imread(osp.join(args.data_root, 'imageFiles', im['sequence'], im['file_name']))
            if img is None:
                continue
            pred.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            for a in lst:
                if keep is not None and a['id'] not in keep:
                    continue
                x, y, w, h = [float(v) for v in a['bbox']]
                pts = pointer_points(a, hh)
                kw = {'box': np.array([x, y, x + w, y + h], dtype=np.float32), 'multimask_output': False}
                if len(pts) >= 3:
                    kw['point_coords'] = np.asarray(pts, dtype=np.float32)
                    kw['point_labels'] = np.ones(len(pts), dtype=np.int32)
                masks, sc, _ = pred.predict(**kw)
                r = rle_encode(np.asarray(masks[0]).astype(bool))
                ids.append(a['id']); scores.append(float(np.asarray(sc).reshape(-1)[0]))
                nptr.append(len(pts) if len(pts) >= 3 else 0); hs.append(img.shape[0]); ws.append(img.shape[1])
                runs.append(r); off.append(off[-1] + len(r))
        out = osp.join(args.out_dir, f'sam_masks_{split}.npz')
        np.savez_compressed(out, ann_ids=np.asarray(ids, np.int64), score=np.asarray(scores, np.float32),
                            n_ptr=np.asarray(nptr, np.int32), H=np.asarray(hs, np.int32), W=np.asarray(ws, np.int32),
                            runs=np.concatenate(runs) if runs else np.zeros(0, np.int32), off=np.asarray(off, np.int64))
        print(f'[{split}] {len(ids)} mask -> {out} | khong co con tro (chi bbox): {int(np.sum(np.asarray(nptr) == 0))}')


if __name__ == '__main__':
    main()
