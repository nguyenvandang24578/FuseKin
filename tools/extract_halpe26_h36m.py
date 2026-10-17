"""
Trich 2D keypoint Halpe-26 cho Human3.6M bang RTMPose (rtmlib, ONNX) -- ban H36M cua tools/extract_halpe26_3dpw.py.
Chay trong moi truong rieng `pose2d` (rtmlib + onnxruntime-gpu), KHONG can code FuseKin.

Dau vao : <data_root>/annotations/Human36M_subject{S}_data.json  (cung file ma data_final/Human36M/dataset.py doc)
          <data_root>/images/<file_name>
Loc frame GIONG Human36M.load_data():
  train: S1 S5 S6 S7 S8, frame_idx % 5 == 0, ca 4 camera
  test : S9 S11,         frame_idx % 64 == 0, chi camera 4
  (--all_frames: bo loc frame; --all_cams: test lay ca 4 camera)
Bbox    : bbox GT trong annotation (ann['bbox'], xywh) -> RTMPose tu mo rong ben trong. H36M moi anh 1 nguoi.
Dau ra  : <out_dir>/Human36M_{split}_halpe26.npz
            file_name : (N,)      str  -- khoa tra cuu (img['file_name'], duy nhat: subject/act/subact/cam/frame)
            kpts      : (N, 26, 3) float32 -- [x, y, score] Halpe-26, pixel ANH GOC
          Chia shard (--num_shards K --shard i) -> ghi *.shard{i}of{K}.npz, gop bang --merge.

Vi du (chay tu thu muc goc FuseKin):
  conda activate pose2d
  python tools/extract_halpe26_h36m.py --data_root data_final/Human36M --splits test --vis_num 8
  python tools/extract_halpe26_h36m.py --data_root data_final/Human36M --splits train test
  # song song 4 GPU:
  for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python tools/extract_halpe26_h36m.py --splits train --num_shards 4 --shard $i & done; wait
  python tools/extract_halpe26_h36m.py --splits train --num_shards 4 --merge
Kiem tra bang mat sau khi trich: tools/vis_h36m_halpe26.py (ve qua chinh dataset FuseKin, co augment / flip).
"""
import argparse
import json
import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

# Dung chung model + ham ve voi ban 3DPW (cung thu muc tools/)
from extract_halpe26_3dpw import build_model, draw, halpe2h36m, HALPE26_SKELETON, H36M_SKELETON

SPLIT_CFG = {
    'train': dict(subjects=[1, 5, 6, 7, 8], ratio=5),
    'test': dict(subjects=[9, 11], ratio=64),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=os.environ.get('JOTR_DATA_ROOT', 'data_final') + '/Human36M',
                   help='Thu muc Human36M (chua annotations/ va images/)')
    p.add_argument('--splits', nargs='+', default=['train', 'test'], choices=['train', 'test'])
    p.add_argument('--out_dir', default='', help='Mac dinh = data_root')
    p.add_argument('--all_frames', action='store_true', help='Bo loc frame (train %%5, test %%64)')
    p.add_argument('--all_cams', action='store_true', help='Test: lay ca 4 camera (mac dinh chi camera 4)')
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--merge', action='store_true', help='Chi gop cac shard da chay xong thanh 1 file')
    p.add_argument('--io_threads', type=int, default=8, help='So luong doc anh song song (doc anh la co chai)')
    # model (giong extract_halpe26_3dpw.py)
    p.add_argument('--mode', default='performance', choices=['performance', 'balanced', 'lightweight'])
    p.add_argument('--onnx', default='')
    p.add_argument('--input_size', type=int, nargs=2, default=None, metavar=('W', 'H'))
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--vis_num', type=int, default=0, help='So anh ve kiem tra moi split (0 = tat)')
    p.add_argument('--vis_dir', default='debug_halpe26_h36m')
    return p.parse_args()


def out_name(out_dir, split, shard=None, num_shards=1):
    base = osp.join(out_dir, f'Human36M_{split}_halpe26')
    return f'{base}.shard{shard}of{num_shards}.npz' if num_shards > 1 else f'{base}.npz'


def collect_items(args, split):
    """Danh sach (file_name, bbox_xyxy) theo dung bo loc cua Human36M.load_data()."""
    sc = SPLIT_CFG[split]
    ratio = 1 if args.all_frames else sc['ratio']
    items = []
    for s in sc['subjects']:
        path = osp.join(args.data_root, 'annotations', f'Human36M_subject{s}_data.json')
        with open(path) as f:
            d = json.load(f)
        imgs = {im['id']: im for im in d['images']}
        n0 = len(items)
        for ann in d['annotations']:
            im = imgs[ann['image_id']]
            if im['frame_idx'] % ratio != 0:
                continue
            if split == 'test' and not args.all_cams and str(im['cam_idx']) != '4':
                continue
            x, y, w, h = [float(v) for v in ann['bbox']]
            items.append((im['file_name'], [x, y, x + w, y + h]))
        print(f'[{split}] S{s}: {len(items) - n0} nguoi')
    items.sort(key=lambda t: t[0])          # thu tu co dinh -> chia shard on dinh giua cac lan chay
    return items


def save_vis(img, kpts, scores, bbox, out_path, thr=0.3):
    """2 o: [Halpe-26 goc] | [H36M-17 sau khi doi]. Cat quanh bbox."""
    a, b = img.copy(), img.copy()
    x0, y0, x1, y1 = [int(v) for v in bbox]
    for im in (a, b):
        cv2.rectangle(im, (x0, y0), (x1, y1), (255, 0, 255), 2)
    draw(a, kpts, scores, HALPE26_SKELETON, thr, highlight=(17, 18, 19))
    h36 = halpe2h36m(np.concatenate([kpts, scores[:, None]], axis=1))
    draw(b, h36[:, :2], h36[:, 2], H36M_SKELETON, thr, highlight=(7, 10))
    H, W = img.shape[:2]
    cx0, cx1 = max(0, x0 - 60), min(W, x1 + 60)
    cy0, cy1 = max(0, y0 - 60), min(H, y1 + 40)
    crops = [np.ascontiguousarray(p[cy0:cy1, cx0:cx1]) for p in (a, b)]
    titles = ['Halpe-26 (do: 17 dinh dau, 18 co, 19 hong)', 'H36M-17 (do: 7 Torso, 10 Head_top)']
    crops = [draw(cr, [], [], [], thr, title=t) for cr, t in zip(crops, titles)]
    cv2.imwrite(out_path, np.concatenate(crops, axis=1))


def run_split(model, args, split, out_dir):
    items = collect_items(args, split)
    if args.num_shards > 1:
        items = items[args.shard::args.num_shards]
    print(f'[{split}] shard {args.shard}/{args.num_shards}: {len(items)} nguoi')
    img_dir = osp.join(args.data_root, 'images')
    vis_every = max(1, len(items) // args.vis_num) if args.vis_num else 0

    names, kpts_all, n_missing = [], [], 0
    pool = ThreadPoolExecutor(max_workers=args.io_threads)
    chunk = 256
    for c0 in tqdm(range(0, len(items), chunk), desc=f'{split} (x{chunk})'):
        part = items[c0:c0 + chunk]
        imgs = pool.map(lambda t: cv2.imread(osp.join(img_dir, t[0])), part)   # BGR, dung dinh dang rtmlib
        for j, ((fname, bbox), img) in enumerate(zip(part, imgs)):
            if img is None:
                n_missing += 1
                continue
            k, s = model(img, bboxes=[bbox])          # (1, 26, 2), (1, 26)
            k, s = np.asarray(k)[0], np.asarray(s)[0]
            assert k.shape[0] == 26, f'Model khong phai Halpe26: {k.shape}'
            names.append(fname)
            kpts_all.append(np.concatenate([k, s[:, None]], axis=1).astype(np.float32))
            gi = c0 + j
            if vis_every and gi % vis_every == 0 and gi // vis_every < args.vis_num:
                os.makedirs(args.vis_dir, exist_ok=True)
                save_vis(img, k, s, bbox, osp.join(args.vis_dir, f'{split}_' + fname.replace('/', '_')))
    pool.shutdown()

    path = out_name(out_dir, split, args.shard, args.num_shards)
    kpts_arr = np.stack(kpts_all) if kpts_all else np.zeros((0, 26, 3), np.float32)
    np.savez_compressed(path, file_name=np.array(names), kpts=kpts_arr)
    print(f'[{split}] luu {len(names)} nguoi -> {path} | anh khong doc duoc: {n_missing}')
    if len(names):
        sc = kpts_arr[:, :, 2]
        print(f'[{split}] score TB = {sc.mean():.3f} | ti le khop score > 0.3 = {(sc > 0.3).mean() * 100:.1f}%')


def merge(args, split, out_dir):
    parts = [out_name(out_dir, split, i, args.num_shards) for i in range(args.num_shards)]
    missing = [p for p in parts if not osp.isfile(p)]
    if missing:
        raise SystemExit(f'Thieu shard: {missing}')
    names, kpts = [], []
    for p in parts:
        d = np.load(p)
        names.append(d['file_name'])
        kpts.append(d['kpts'])
    names, kpts = np.concatenate(names), np.concatenate(kpts)
    assert len(set(names.tolist())) == len(names), 'Trung file_name giua cac shard'
    path = out_name(out_dir, split)
    np.savez_compressed(path, file_name=names, kpts=kpts)
    print(f'[{split}] gop {len(parts)} shard -> {path} ({len(names)} nguoi)')


def main():
    args = parse_args()
    out_dir = args.out_dir or args.data_root
    os.makedirs(out_dir, exist_ok=True)
    if args.merge:
        for split in args.splits:
            merge(args, split, out_dir)
        return
    assert 0 <= args.shard < args.num_shards
    model = build_model(args)
    for split in args.splits:
        run_split(model, args, split, out_dir)


if __name__ == '__main__':
    main()
