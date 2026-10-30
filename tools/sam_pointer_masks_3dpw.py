"""
tools/sam_pointer_masks_3dpw.py -- BUOC 1/2 (moi truong co torch, vd `arts`): mask SAM cua DUNG nguoi cho 3DPW.
Prompt SAM = "con tro" danh tinh = khop 2D bottom-up DA GAN SAN cho nguoi do (cung nguon JOTR / 3DCrowdNet dung khi
test): Higher-HRNet (neu co, tap Crowd) hoac OpenPose (annotation) -> DIEM DUONG; khop con tro cua NGUOI KHAC cung anh
-> DIEM AM. Mac dinh KHONG dua bbox (bbox chua ca 2 nguoi keo SAM ra 2 nguoi; --use_box de bat). Lay 3 mask ung vien,
chon mask chua nhieu diem duong / it diem am nhat, roi CHI GIU vung lien thong chua diem duong (bo mang roi).
Khong co con tro (< 3 khop) -> chi bbox. Y tuong: BUCTD (pose bottom-up lam con tro) + PromptHMR (mask).
Ghi <out_dir>/sam_masks_<split>.npz: ann_ids, score, n_ptr (so diem con tro), H, W, runs + off (mask nen run-length,
hang truoc, bat dau bang so diem 0).
  --prompt center : CHI 1-2 diem giua than (co = giua 2 vai, pelvis = giua 2 hong cua con tro), KHONG diem am,
      lay mask LON NHAT trong 3 mask (muc "ca nguoi"), khong loc lien thong. Thieu ca 2 diem -> chi bbox.
      --clip_box: cat mask trong bbox GT noi 10% (chi dung voi --prompt center).
  --vis N : chi N nguoi NGAU NHIEN moi split (seed co dinh) -> de buoc 2 ve hinh; 0 = TAT CA nguoi.
Buoc 2 (moi truong pose2d): tools/extract_halpe26_sam_3dpw.py --mask_dir <out_dir> (cung --splits, cung --vis).
SAM2 (mac dinh, can Python >= 3.10, moi truong rieng co torch chay duoc): pip install sam2 (tai
'facebook/sam2.1-hiera-large' tu HuggingFace). SAM1: --sam sam1 --sam_ckpt sam_vit_h_4b8939.pth.
Vi du:
  python tools/sam_pointer_masks_3dpw.py --data_root data_final/PW3D --splits test oc pc validation --vis 50 \\
      --out_dir debug_sam_vis
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
    p.add_argument('--sam', default='sam2', choices=['sam2', 'sam1'])
    p.add_argument('--sam_model', default='facebook/sam2.1-hiera-large')
    p.add_argument('--sam_type', default='vit_h')
    p.add_argument('--sam_ckpt', default='')
    p.add_argument('--use_box', action='store_true', help='dua them bbox annotation vao prompt (mac dinh tat)')
    p.add_argument('--prompt', default='skeleton', choices=['skeleton', 'center'],
                   help='skeleton: ca bo khop + diem am (cu); center: 1-2 diem giua than + mask lon nhat')
    p.add_argument('--clip_box', action='store_true', help='--prompt center: cat mask trong bbox GT noi 10%%')
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


def center_points(ann, hh):
    """1-2 diem giua than tu con tro: co = giua 2 vai, pelvis = giua 2 hong (ca 2 khop phai co)."""
    if hh is not None and str(ann['id']) in hh:
        k = np.asarray(hh[str(ann['id'])]['coco_joints'], dtype=np.float32).reshape(-1, 3)
        pairs, thr = [(5, 6), (11, 12)], HHRNET_THR          # COCO-17: vai L/R, hong L/R
    else:
        k = np.asarray(ann.get('openpose_result', []), dtype=np.float32).reshape(-1, 3)
        if len(k) < 18:
            return []
        pairs, thr = [(2, 5), (8, 11)], OPENPOSE_THR         # OpenPose-18: vai R/L, hong R/L
    out = []
    for i, j in pairs:
        if k[i, 2] > thr and k[j, 2] > thr:
            out.append(tuple(((k[i, :2] + k[j, :2]) / 2).tolist()))
    return out


def choose_mask(masks, scores, pos, neg):
    """Chon mask ung vien: max (ti le diem duong trong mask - ti le diem am trong mask), hoa -> diem SAM.
    Roi chi giu vung lien thong chua it nhat 1 diem duong."""
    H, W = masks.shape[-2:]

    def inside(m, pts):
        if not pts:
            return 0.0
        v = [m[min(max(int(y), 0), H - 1), min(max(int(x), 0), W - 1)] for x, y in pts]
        return float(np.mean(v))
    best, bs = 0, None
    for i in range(masks.shape[0]):
        m = masks[i].astype(bool)
        sc = (inside(m, pos) - inside(m, neg), float(scores[i]))
        if bs is None or sc > bs:
            best, bs = i, sc
    m = masks[best].astype(np.uint8)
    n, lab = cv2.connectedComponents(m)
    keep = set()
    for x, y in pos:
        xi, yi = min(max(int(x), 0), W - 1), min(max(int(y), 0), H - 1)
        if lab[yi, xi] > 0:
            keep.add(int(lab[yi, xi]))
    if keep:
        m = np.isin(lab, list(keep))
    return m.astype(bool), float(scores[best])


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
    if args.sam == 'sam2':
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        pred = SAM2ImagePredictor.from_pretrained(args.sam_model, device=args.device)
    else:
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
                box = np.array([x, y, x + w, y + h], dtype=np.float32)
                if args.prompt == 'center':
                    cp = center_points(a, hh)
                    kw = {'multimask_output': True}
                    if cp:
                        kw['point_coords'] = np.asarray(cp, dtype=np.float32)
                        kw['point_labels'] = np.ones(len(cp), dtype=np.int32)
                    else:
                        kw['box'] = box
                    masks, sc, _ = pred.predict(**kw)
                    masks, sc = np.asarray(masks).astype(bool), np.asarray(sc).reshape(-1)
                    i = int(np.argmax(masks.reshape(len(masks), -1).sum(1))) if cp else int(np.argmax(sc))
                    m, s_best = masks[i], float(sc[i])
                    if args.clip_box:
                        H, W = m.shape
                        x0, y0 = int(max(x - 0.05 * w, 0)), int(max(y - 0.05 * h, 0))
                        x1, y1 = int(min(x + 1.05 * w, W)), int(min(y + 1.05 * h, H))
                        c = np.zeros_like(m); c[y0:y1, x0:x1] = True
                        m = m & c
                    r = rle_encode(m)
                    ids.append(a['id']); scores.append(s_best)
                    nptr.append(len(cp)); hs.append(img.shape[0]); ws.append(img.shape[1])
                    runs.append(r); off.append(off[-1] + len(r))
                    continue
                pts = pointer_points(a, hh)
                neg = [q for b in lst if b is not a for q in pointer_points(b, hh)] if len(pts) >= 3 else []
                kw = {'multimask_output': True}
                if len(pts) >= 3:
                    kw['point_coords'] = np.asarray(pts + neg, dtype=np.float32)
                    kw['point_labels'] = np.asarray([1] * len(pts) + [0] * len(neg), dtype=np.int32)
                    if args.use_box:
                        kw['box'] = box
                else:
                    kw['box'] = box
                masks, sc, _ = pred.predict(**kw)
                masks, sc = np.asarray(masks), np.asarray(sc).reshape(-1)
                if len(pts) >= 3:
                    m, s_best = choose_mask(masks, sc, pts, neg)
                else:
                    i = int(np.argmax(sc))
                    m, s_best = masks[i].astype(bool), float(sc[i])
                r = rle_encode(m)
                ids.append(a['id']); scores.append(s_best)
                nptr.append(len(pts) if len(pts) >= 3 else 0); hs.append(img.shape[0]); ws.append(img.shape[1])
                runs.append(r); off.append(off[-1] + len(r))
        out = osp.join(args.out_dir, f'sam_masks_{split}.npz')
        np.savez_compressed(out, ann_ids=np.asarray(ids, np.int64), score=np.asarray(scores, np.float32),
                            n_ptr=np.asarray(nptr, np.int32), H=np.asarray(hs, np.int32), W=np.asarray(ws, np.int32),
                            runs=np.concatenate(runs) if runs else np.zeros(0, np.int32), off=np.asarray(off, np.int64))
        print(f'[{split}] {len(ids)} mask -> {out} | khong co con tro (chi bbox): {int(np.sum(np.asarray(nptr) == 0))}')


if __name__ == '__main__':
    main()
