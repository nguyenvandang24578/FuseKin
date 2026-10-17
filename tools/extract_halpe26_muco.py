"""
Trich 2D keypoint Halpe-26 cho MuCo-3DHP (train) bang RTMPose (rtmlib, ONNX) -- ban MuCo cua extract_halpe26_h36m.py.
Chay trong moi truong rieng `pose2d` (rtmlib + onnxruntime-gpu), KHONG can code FuseKin.

Dau vao : <data_root>/MuCo-3DHP.json (COCO-style, cung file ma data_final/MuCo/dataset.py doc), anh <data_root>/<file_name>
Chon nguoi GIONG MuCo.load_data(): moi anh chi lay NGUOI GAN CAMERA NHAT (min do sau khop Pelvis trong keypoints_cam).
Bbox    : bbox GT cua nguoi do (xywh). Anh MuCo la anh ghep nhieu nguoi -> detector co the bi nguoi khac chen vao,
          day la loi thuc te (giong 3DPW-OC / Crowd), giu nguyen.
Dau ra  : <out_dir>/MuCo_train_halpe26.npz
            ann_id : (N,)       int64   -- khoa tra cuu (ann id cua nguoi duoc chon)
            kpts   : (N, 26, 3) float32 -- [x, y, score] Halpe-26, pixel ANH GOC
          Chia shard (--num_shards K --shard i) -> *.shard{i}of{K}.npz, gop bang --merge.

Vi du (chay tu thu muc goc FuseKin):
  conda activate pose2d
  python tools/extract_halpe26_muco.py --data_root data_final/MuCo --max_items 200 --vis_num 8   # thu nhanh
  for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python tools/extract_halpe26_muco.py --num_shards 4 --shard $i > log_halpe_muco_$i.txt 2>&1 & done; wait
  python tools/extract_halpe26_muco.py --num_shards 4 --merge
Kiem tra bang mat qua dataset FuseKin: tools/vis_h36m_halpe26.py --dataset MuCo
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

MUCO_PELVIS_IDX = 14   # muco_joints_name.index('Pelvis') trong data_final/MuCo/dataset.py


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=os.environ.get('JOTR_DATA_ROOT', 'data_final') + '/MuCo',
                   help='Thu muc MuCo (chua MuCo-3DHP.json va anh)')
    p.add_argument('--out_dir', default='', help='Mac dinh = data_root')
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--merge', action='store_true', help='Chi gop cac shard da chay xong thanh 1 file')
    p.add_argument('--max_items', type=int, default=0, help='>0: chi chay N nguoi dau (thu nhanh, ghi file _debug)')
    p.add_argument('--io_threads', type=int, default=8)
    p.add_argument('--mode', default='performance', choices=['performance', 'balanced', 'lightweight'])
    p.add_argument('--onnx', default='')
    p.add_argument('--input_size', type=int, nargs=2, default=None, metavar=('W', 'H'))
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--vis_num', type=int, default=0)
    p.add_argument('--vis_dir', default='debug_halpe26_muco')
    return p.parse_args()


def out_name(out_dir, shard=None, num_shards=1, debug=False):
    base = osp.join(out_dir, 'MuCo_train_halpe26' + ('_debug' if debug else ''))
    return f'{base}.shard{shard}of{num_shards}.npz' if num_shards > 1 else f'{base}.npz'


def collect_items(args):
    """(ann_id, file_name, bbox_xyxy) cua nguoi gan camera nhat moi anh -- giong MuCo.load_data()."""
    path = osp.join(args.data_root, 'MuCo-3DHP.json')
    print(f'Doc {path} ...')
    with open(path) as f:
        d = json.load(f)
    by_img = defaultdict(list)
    for ann in d['annotations']:          # giu thu tu xuat hien, giong COCO.getAnnIds
        by_img[ann['image_id']].append(ann)
    items = []
    for im in d['images']:
        anns = by_img.get(im['id'], [])
        if not anns:
            continue
        depths = [a['keypoints_cam'][MUCO_PELVIS_IDX][2] for a in anns]
        a = anns[depths.index(min(depths))]
        x, y, w, h = [float(v) for v in a['bbox']]
        items.append((int(a['id']), im['file_name'], [x, y, x + w, y + h]))
    items.sort(key=lambda t: t[0])
    print(f'[MuCo] {len(d["images"])} anh -> {len(items)} nguoi (gan camera nhat)')
    return items


def run(model, args, out_dir):
    items = collect_items(args)
    if args.max_items > 0:
        items = items[:args.max_items]
    if args.num_shards > 1:
        items = items[args.shard::args.num_shards]
    print(f'[MuCo] shard {args.shard}/{args.num_shards}: {len(items)} nguoi')
    vis_every = max(1, len(items) // args.vis_num) if args.vis_num else 0

    ids, kpts_all, n_missing = [], [], 0
    pool = ThreadPoolExecutor(max_workers=args.io_threads)
    chunk = 256
    for c0 in tqdm(range(0, len(items), chunk), desc=f'MuCo (x{chunk})'):
        part = items[c0:c0 + chunk]
        imgs = pool.map(lambda t: cv2.imread(osp.join(args.data_root, t[1])), part)
        for j, ((aid, fname, bbox), img) in enumerate(zip(part, imgs)):
            if img is None:
                n_missing += 1
                continue
            k, s = model(img, bboxes=[bbox])
            k, s = np.asarray(k)[0], np.asarray(s)[0]
            assert k.shape[0] == 26, f'Model khong phai Halpe26: {k.shape}'
            ids.append(aid)
            kpts_all.append(np.concatenate([k, s[:, None]], axis=1).astype(np.float32))
            gi = c0 + j
            if vis_every and gi % vis_every == 0 and gi // vis_every < args.vis_num:
                os.makedirs(args.vis_dir, exist_ok=True)
                save_vis(img, k, s, bbox, osp.join(args.vis_dir, f'muco_{aid}_' + osp.basename(fname)))
    pool.shutdown()

    path = out_name(out_dir, args.shard, args.num_shards, debug=args.max_items > 0)
    kpts_arr = np.stack(kpts_all) if kpts_all else np.zeros((0, 26, 3), np.float32)
    np.savez_compressed(path, ann_id=np.array(ids, dtype=np.int64), kpts=kpts_arr)
    print(f'[MuCo] luu {len(ids)} nguoi -> {path} | anh khong doc duoc: {n_missing}')
    if len(ids):
        sc = kpts_arr[:, :, 2]
        print(f'[MuCo] score TB = {sc.mean():.3f} | ti le khop score > 0.3 = {(sc > 0.3).mean() * 100:.1f}%')


def merge(args, out_dir):
    parts = [out_name(out_dir, i, args.num_shards) for i in range(args.num_shards)]
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
    path = out_name(out_dir)
    np.savez_compressed(path, ann_id=ids, kpts=kpts)
    print(f'[MuCo] gop {len(parts)} shard -> {path} ({len(ids)} nguoi)')


def main():
    args = parse_args()
    out_dir = args.out_dir or args.data_root
    os.makedirs(out_dir, exist_ok=True)
    if args.merge:
        merge(args, out_dir)
        return
    assert 0 <= args.shard < args.num_shards
    model = build_model(args)
    run(model, args, out_dir)


if __name__ == '__main__':
    main()
