"""
tools/vis_vitpose_lowconf.py (2026-11-18, nguoi dung duyet) -- Visual 2D ViTPose (Halpe-26) cua CrowdPose train,
tap trung vao cac mau confidence thap. Khong sua file nao co san, khong dung model, chi doc npz + anh.

Dau vao : <data_root>/CrowdPose_train_halpe26_vitpose.npz (ann_id, kpts (N, 26, 3) [x, y, score], pixel anh goc)
          <data_root>/annotations/crowdpose_train.json (lay file_name + bbox theo ann_id)
Dau ra  : <out_dir>/lowconf.jpg, random.jpg, highconf.jpg  (luoi cac nguoi, ve xuong + khop mau theo score)
          <out_dir>/stats.json (thong ke score tren khop CO THAT, khong tinh dinh dau + 6 diem ban chan score 0)
Mau     : do score < 0.3 | vang 0.3..0.6 | xanh > 0.6 ; khop score = 0 khong ve.
Khop co that = COCO-17 (0..16) + Neck (18) + Hip (19); bo 17 (dinh dau) va 20..25 (ban chan) vi luon score 0.

Chay (tu thu muc goc repo, env nao co numpy + opencv):
  python tools/vis_vitpose_lowconf.py --data_root data_final/CrowdPose --num 24
"""
import argparse
import json
import os
import os.path as osp

import cv2
import numpy as np

REAL = list(range(17)) + [18, 19]
# Xuong COCO-17 (chi vi tri 0..16 cua Halpe-26)
EDGES = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
         (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
TILE_W, TILE_H = 256, 320


def sc_color(s):
    return (0, 0, 255) if s < 0.3 else ((0, 200, 255) if s < 0.6 else (0, 200, 0))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='data_final/CrowdPose')
    p.add_argument('--npz', default='', help='mac dinh <data_root>/CrowdPose_train_halpe26_vitpose.npz')
    p.add_argument('--num', type=int, default=24, help='so nguoi moi nhom')
    p.add_argument('--cols', type=int, default=6)
    p.add_argument('--min_h', type=float, default=0, help='chi xet nguoi co bbox cao >= min_h px (0 = khong loc)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', default='output/vis_vitpose_crowdpose')
    return p.parse_args()


def make_tile(img, bbox, kp, aid, ms):
    """Cat quanh bbox x1.25, dat vao o TILE_W x TILE_H, ve xuong theo score."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 1.25, (y2 - y1) * 1.25
    a, b, c, d = int(max(0, cx - w / 2)), int(max(0, cy - h / 2)), int(min(W, cx + w / 2)), int(min(H, cy + h / 2))
    crop = img[b:d, a:c]
    sc = min(TILE_W / max(c - a, 1), TILE_H / max(d - b, 1))
    crop = cv2.resize(crop, (max(int((c - a) * sc), 1), max(int((d - b) * sc), 1)))
    canvas = np.full((TILE_H, TILE_W, 3), 60, np.uint8)
    canvas[:crop.shape[0], :crop.shape[1]] = crop
    pts = (kp[:, :2] - np.array([a, b])) * sc
    # bbox goc (xanh duong nhat)
    cv2.rectangle(canvas, (int((x1 - a) * sc), int((y1 - b) * sc)), (int((x2 - a) * sc), int((y2 - b) * sc)),
                  (255, 160, 0), 1)
    for i, j in EDGES:
        if kp[i, 2] > 0 and kp[j, 2] > 0:
            cv2.line(canvas, tuple(int(v) for v in pts[i]), tuple(int(v) for v in pts[j]),
                     sc_color(min(kp[i, 2], kp[j, 2])), 2)
    for i in REAL:
        if kp[i, 2] > 0:
            cv2.circle(canvas, tuple(int(v) for v in pts[i]), 3, sc_color(kp[i, 2]), -1)
    cv2.putText(canvas, f'id {aid} s {ms:.2f}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return canvas


def make_grid(idx, kpts, ids, ann_info, img_dir, cols, path):
    tiles = []
    for i in idx:
        fname, bbox = ann_info[int(ids[i])]
        img = cv2.imread(osp.join(img_dir, fname))
        if img is None:
            continue
        ms = kpts[i, REAL, 2].mean()
        tiles.append(make_tile(img, bbox, kpts[i], int(ids[i]), ms))
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[r:r + cols], axis=1) for r in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.concatenate(rows, axis=0))
    print('luu', path)


def main():
    args = parse_args()
    npz = args.npz or osp.join(args.data_root, 'CrowdPose_train_halpe26_vitpose.npz')
    z = np.load(npz)
    ids, kpts = z['ann_id'], z['kpts']
    print(f'{len(ids)} nguoi tu {npz}')
    with open(osp.join(args.data_root, 'annotations/crowdpose_train.json')) as f:
        d = json.load(f)
    files = {im['id']: im['file_name'] for im in d['images']}
    ann_info = {}
    for a in d['annotations']:
        x, y, w, h = [float(v) for v in a['bbox']]
        ann_info[int(a['id'])] = (files[a['image_id']], [x, y, x + w, y + h])
    bh = np.array([ann_info[int(i)][1][3] - ann_info[int(i)][1][1] for i in ids])

    sc = kpts[:, :, 2]
    real = sc[:, REAL]
    ms = real.mean(1)                                   # score TB tung nguoi tren khop co that
    st = {
        'n_nguoi': int(len(ids)),
        'score_TB_26khop': float(sc.mean()), 'ti_le_>0.3_26khop': float((sc > 0.3).mean()),
        'score_TB_khop_co_that': float(real.mean()), 'ti_le_>0.3_khop_co_that': float((real > 0.3).mean()),
        'ti_le_nguoi_TB<0.3': float((ms < 0.3).mean()), 'ti_le_nguoi_TB<0.5': float((ms < 0.5).mean()),
        'score_tung_khop_0_25': [float(v) for v in sc.mean(0)],
        'theo_chieu_cao_bbox': {},
    }
    qs = np.quantile(bh, [0, 0.25, 0.5, 0.75, 1.0])
    for q in range(4):
        m = (bh >= qs[q]) & (bh <= qs[q + 1])
        st['theo_chieu_cao_bbox'][f'{qs[q]:.0f}-{qs[q + 1]:.0f}px'] = {
            'n': int(m.sum()), 'score_TB': float(real[m].mean()), 'ti_le_>0.3': float((real[m] > 0.3).mean())}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(osp.join(args.out_dir, 'stats.json'), 'w') as f:
        json.dump(st, f, indent=1)
    print(json.dumps({k: v for k, v in st.items() if k != 'score_tung_khop_0_25'}, indent=1))

    pool = np.where(bh >= args.min_h)[0]
    order = pool[np.argsort(ms[pool])]
    rng = np.random.RandomState(args.seed)
    img_dir = osp.join(args.data_root, 'images')
    make_grid(order[:args.num], kpts, ids, ann_info, img_dir, args.cols, osp.join(args.out_dir, 'lowconf.jpg'))
    make_grid(rng.choice(pool, args.num, replace=False), kpts, ids, ann_info, img_dir, args.cols,
              osp.join(args.out_dir, 'random.jpg'))
    make_grid(order[::-1][:args.num], kpts, ids, ann_info, img_dir, args.cols, osp.join(args.out_dir, 'highconf.jpg'))


if __name__ == '__main__':
    main()
