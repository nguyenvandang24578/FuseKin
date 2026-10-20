"""
Trich 2D keypoint Halpe-26 cho MSCOCO (train2017) va CrowdPose (train) bang RTMPose (rtmlib, ONNX).
Cung model + cach lam voi tools/extract_halpe26_muco.py. Chay trong moi truong `pose2d`, KHONG can code FuseKin.

Chi trich nguoi CO pseudo-GT 3D NeuralAnnot (lifter chi dung duoc nhung nguoi nay):
  mscoco   : person_keypoints_train2017.json, iscrowd = 0, num_keypoints > 0, ann_id co trong MSCOCO_train_SMPL_NeuralAnnot.json
  crowdpose: crowdpose_train.json, keypoints khac 0, ann_id co trong CrowdPose_train_SMPL_NeuralAnnot.json va fit_err <= 5
Bbox    : bbox GT (xywh) cua tung nguoi (giong 3DPW / MuCo). Nhieu nguoi / anh -> chay 1 lan cho ca anh.
Dau ra  : <out_dir>/MSCOCO_train_halpe26.npz hoac CrowdPose_train_halpe26.npz
            ann_id : (N,)       int64
            kpts   : (N, 26, 3) float32 -- [x, y, score] Halpe-26, pixel ANH GOC
          Chia shard (--num_shards K --shard i) -> *.shard{i}of{K}.npz, gop bang --merge.

Vi du (tu thu muc goc FuseKin):
  conda activate pose2d
  python tools/extract_halpe26_coco.py --dataset mscoco --max_items 200 --vis_num 8        # thu nhanh
  for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python tools/extract_halpe26_coco.py --dataset mscoco --num_shards 4 --shard $i > log_halpe_coco_$i.txt 2>&1 & done; wait
  python tools/extract_halpe26_coco.py --dataset mscoco --num_shards 4 --merge
  (CrowdPose: thay --dataset crowdpose)
"""
import argparse
import json
import os
import os.path as osp
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from extract_halpe26_h36m import build_model, save_vis   # cung model + ham ve

CFG = {
    'mscoco': dict(name='MSCOCO', ann='annotations/person_keypoints_train2017.json',
                   neural='annotations/MSCOCO_train_SMPL_NeuralAnnot.json', img_dir='images/train2017', fit_thr=None),
    'crowdpose': dict(name='CrowdPose', ann='annotations/crowdpose_train.json',
                      neural='annotations/CrowdPose_train_SMPL_NeuralAnnot.json', img_dir='images', fit_thr=5.0),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=list(CFG))
    p.add_argument('--data_root', default='', help='Mac dinh <JOTR_DATA_ROOT hoac data_final>/<MSCOCO|CrowdPose>')
    p.add_argument('--out_dir', default='', help='Mac dinh = data_root')
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--merge', action='store_true', help='Chi gop cac shard da chay xong thanh 1 file')
    p.add_argument('--max_items', type=int, default=0, help='>0: chi chay N anh dau (thu nhanh, ghi file _debug)')
    p.add_argument('--io_threads', type=int, default=8)
    p.add_argument('--mode', default='performance', choices=['performance', 'balanced', 'lightweight'])
    p.add_argument('--onnx', default='')
    p.add_argument('--input_size', type=int, nargs=2, default=None, metavar=('W', 'H'))
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--vis_num', type=int, default=0)
    p.add_argument('--vis_dir', default='')
    return p.parse_args()


def out_name(out_dir, name, shard=None, num_shards=1, debug=False):
    base = osp.join(out_dir, f'{name}_train_halpe26' + ('_debug' if debug else ''))
    return f'{base}.shard{shard}of{num_shards}.npz' if num_shards > 1 else f'{base}.npz'


def collect_items(args, c):
    """[(file_name, [(ann_id, bbox_xyxy), ...]), ...] theo anh, chi nguoi co NeuralAnnot hop le."""
    ann_path, neu_path = osp.join(args.data_root, c['ann']), osp.join(args.data_root, c['neural'])
    print(f'Doc {ann_path} ...')
    with open(ann_path) as f:
        d = json.load(f)
    print(f'Doc {neu_path} ...')
    with open(neu_path) as f:
        neural = json.load(f)
    files = {im['id']: im['file_name'] for im in d['images']}
    by_img = defaultdict(list)
    n_all = 0
    for ann in d['annotations']:
        kp = ann.get('keypoints', [])
        if args.dataset == 'mscoco':
            if ann.get('iscrowd', 0) or ann.get('num_keypoints', 0) == 0:
                continue
        elif sum(kp) == 0:
            continue
        n_all += 1
        na = neural.get(str(ann['id']))
        if na is None:
            continue
        if c['fit_thr'] is not None and na.get('fit_err', 0.0) > c['fit_thr']:
            continue
        x, y, w, h = [float(v) for v in ann['bbox']]
        if w <= 1 or h <= 1:
            continue
        by_img[ann['image_id']].append((int(ann['id']), [x, y, x + w, y + h]))
    items = sorted(((files[i], v) for i, v in by_img.items()), key=lambda t: t[0])
    n = sum(len(v) for _, v in items)
    print(f'[{c["name"]}] {n_all} nguoi hop le -> {n} nguoi co NeuralAnnot / {len(items)} anh')
    return items


def run(model, args, c, out_dir):
    items = collect_items(args, c)
    if args.max_items > 0:
        items = items[:args.max_items]
    if args.num_shards > 1:
        items = items[args.shard::args.num_shards]
    print(f'[{c["name"]}] shard {args.shard}/{args.num_shards}: {len(items)} anh')
    vis_dir = args.vis_dir or f'debug_halpe26_{args.dataset}'
    vis_every = max(1, len(items) // args.vis_num) if args.vis_num else 0

    ids, kpts_all, n_missing = [], [], 0
    pool = ThreadPoolExecutor(max_workers=args.io_threads)
    chunk = 256
    for c0 in tqdm(range(0, len(items), chunk), desc=f'{c["name"]} (x{chunk} anh)'):
        part = items[c0:c0 + chunk]
        imgs = pool.map(lambda t: cv2.imread(osp.join(args.data_root, c['img_dir'], t[0])), part)
        for j, ((fname, persons), img) in enumerate(zip(part, imgs)):
            if img is None:
                n_missing += 1
                continue
            k, s = model(img, bboxes=[b for _, b in persons])
            k, s = np.asarray(k), np.asarray(s)
            assert k.shape[1] == 26, f'Model khong phai Halpe26: {k.shape}'
            for (aid, bbox), kk, ss in zip(persons, k, s):
                ids.append(aid)
                kpts_all.append(np.concatenate([kk, ss[:, None]], axis=1).astype(np.float32))
            gi = c0 + j
            if vis_every and gi % vis_every == 0 and gi // vis_every < args.vis_num:
                os.makedirs(vis_dir, exist_ok=True)
                save_vis(img, k[0], s[0], persons[0][1],
                         osp.join(vis_dir, f'{args.dataset}_{persons[0][0]}_' + osp.basename(fname)))
    pool.shutdown()

    path = out_name(out_dir, c['name'], args.shard, args.num_shards, debug=args.max_items > 0)
    kpts_arr = np.stack(kpts_all) if kpts_all else np.zeros((0, 26, 3), np.float32)
    np.savez_compressed(path, ann_id=np.array(ids, dtype=np.int64), kpts=kpts_arr)
    print(f'[{c["name"]}] luu {len(ids)} nguoi -> {path} | anh khong doc duoc: {n_missing}')
    if len(ids):
        sc = kpts_arr[:, :, 2]
        print(f'[{c["name"]}] score TB = {sc.mean():.3f} | ti le khop score > 0.3 = {(sc > 0.3).mean() * 100:.1f}%')


def merge(args, c, out_dir):
    parts = [out_name(out_dir, c['name'], i, args.num_shards) for i in range(args.num_shards)]
    missing = [p for p in parts if not osp.isfile(p)]
    if missing:
        raise SystemExit(f'Thieu shard: {missing}')
    ids, kpts = [], []
    for p in parts:
        z = np.load(p)
        ids.append(z['ann_id'])
        kpts.append(z['kpts'])
    ids, kpts = np.concatenate(ids), np.concatenate(kpts)
    assert len(np.unique(ids)) == len(ids), 'Trung ann_id giua cac shard'
    path = out_name(out_dir, c['name'])
    np.savez_compressed(path, ann_id=ids, kpts=kpts)
    print(f'[{c["name"]}] gop {len(parts)} shard -> {path} ({len(ids)} nguoi)')


def main():
    args = parse_args()
    c = CFG[args.dataset]
    if not args.data_root:
        args.data_root = osp.join(os.environ.get('JOTR_DATA_ROOT', 'data_final'), c['name'])
    out_dir = args.out_dir or args.data_root
    os.makedirs(out_dir, exist_ok=True)
    if args.merge:
        merge(args, c, out_dir)
        return
    assert 0 <= args.shard < args.num_shards
    model = build_model(args)
    run(model, args, c, out_dir)


if __name__ == '__main__':
    main()
