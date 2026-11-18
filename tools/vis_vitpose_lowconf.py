"""
tools/vis_vitpose_lowconf.py (2026-11-18, nguoi dung duyet) -- Visual 2D ViTPose (Halpe-26) cua CrowdPose train,
tap trung vao cac mau confidence thap. Khong sua file nao co san, khong dung model, chi doc npz + anh.

Dau vao : <data_root>/CrowdPose_train_halpe26_vitpose.npz (ann_id, kpts (N, 26, 3) [x, y, score], pixel anh goc)
          <data_root>/annotations/crowdpose_train.json (lay file_name + bbox theo ann_id)
Dau ra  : <out_dir>/lowconf.jpg, random.jpg, highconf.jpg, worst.jpg (luoi cac nguoi, ve xuong + khop mau theo score)
          GT 2D CrowdPose (crowdpose_train.json, 14 khop) ve x trang + xuong trang mong (2026-11-18 bo sung).
          worst.jpg = nguoi lech GT nhieu nhat: loi = TB khoang cach 12 khop tay chan (GT co, ViTPose score > 0.3)
          / chieu cao bbox; > --err_thr thuong la nham nguoi (uoc luong tho, nguong tu chon).
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
# 12 khop tay chan CrowdPose (0..11: vai, khuyu, co tay, hong, goi, co chan, trai truoc phai) = COCO 5..16
CP2COCO = list(range(5, 17))
GT_EDGES = ((0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7), (6, 8), (8, 10), (7, 9), (9, 11))


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
    p.add_argument('--err_thr', type=float, default=0.2, help='nguong loi / chieu cao bbox de dem nghi nham nguoi')
    p.add_argument('--out_dir', default='output/vis_vitpose_crowdpose')
    return p.parse_args()


def make_tile(img, bbox, kp, aid, ms, gt=None, err=None):
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
    if gt is not None:                                   # GT 2D CrowdPose: xuong trang mong + dau x
        gp = (gt[:, :2] - np.array([a, b])) * sc
        for i, j in GT_EDGES:
            if gt[i, 2] > 0 and gt[j, 2] > 0:
                cv2.line(canvas, tuple(int(v) for v in gp[i]), tuple(int(v) for v in gp[j]), (255, 255, 255), 1)
        for i in range(len(gt)):
            if gt[i, 2] > 0:
                x, y = int(gp[i, 0]), int(gp[i, 1])
                cv2.line(canvas, (x - 4, y - 4), (x + 4, y + 4), (255, 255, 255), 2)
                cv2.line(canvas, (x - 4, y + 4), (x + 4, y - 4), (255, 255, 255), 2)
    for i, j in EDGES:
        if kp[i, 2] > 0 and kp[j, 2] > 0:
            cv2.line(canvas, tuple(int(v) for v in pts[i]), tuple(int(v) for v in pts[j]),
                     sc_color(min(kp[i, 2], kp[j, 2])), 2)
    for i in REAL:
        if kp[i, 2] > 0:
            cv2.circle(canvas, tuple(int(v) for v in pts[i]), 3, sc_color(kp[i, 2]), -1)
    txt = f'id {aid} s {ms:.2f}' + (f' e {err:.2f}' if err is not None and np.isfinite(err) else '')
    cv2.putText(canvas, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return canvas


def make_grid(idx, kpts, ids, ann_info, img_dir, cols, path, gts=None, errs=None):
    tiles = []
    for i in idx:
        fname, bbox = ann_info[int(ids[i])]
        img = cv2.imread(osp.join(img_dir, fname))
        if img is None:
            continue
        ms = kpts[i, REAL, 2].mean()
        tiles.append(make_tile(img, bbox, kpts[i], int(ids[i]), ms,
                               None if gts is None else gts[i], None if errs is None else errs[i]))
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
    cats = d.get('categories', [])
    if cats and 'keypoints' in cats[0]:
        print('Ten khop GT CrowdPose:', cats[0]['keypoints'])         # kiem tra thu tu khop (12 dau = tay chan)
    ann_info, ann_kp = {}, {}
    for a in d['annotations']:
        x, y, w, h = [float(v) for v in a['bbox']]
        ann_info[int(a['id'])] = (files[a['image_id']], [x, y, x + w, y + h])
        ann_kp[int(a['id'])] = np.array(a['keypoints'], dtype=np.float32).reshape(-1, 3)
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
    # Loi ViTPose so GT tren 12 khop tay chan, chia chieu cao bbox (NaN neu < 3 khop dung duoc)
    gts = np.stack([ann_kp[int(i)] for i in ids])                       # (N,14,3)
    g12, v12 = gts[:, :12, :2], gts[:, :12, 2] > 0
    p12, s12 = kpts[:, CP2COCO, :2], kpts[:, CP2COCO, 2] > 0.3
    m = v12 & s12
    d12 = np.linalg.norm(p12 - g12, axis=-1)
    nm = m.sum(1)
    err = np.where(nm >= 3, (d12 * m).sum(1) / np.maximum(nm, 1) / np.maximum(bh, 1.0), np.nan)
    ok = np.isfinite(err)
    st['loi_vs_GT'] = {'n_tinh_duoc': int(ok.sum()), 'trung_vi': float(np.median(err[ok])),
                       'TB': float(err[ok].mean()), f'ti_le_>{args.err_thr}': float((err[ok] > args.err_thr).mean()),
                       'ti_le_>0.1': float((err[ok] > 0.1).mean())}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(osp.join(args.out_dir, 'stats.json'), 'w') as f:
        json.dump(st, f, indent=1)
    print(json.dumps({k: v for k, v in st.items() if k != 'score_tung_khop_0_25'}, indent=1))

    pool = np.where(bh >= args.min_h)[0]
    order = pool[np.argsort(ms[pool])]
    rng = np.random.RandomState(args.seed)
    img_dir = osp.join(args.data_root, 'images')
    make_grid(order[:args.num], kpts, ids, ann_info, img_dir, args.cols, osp.join(args.out_dir, 'lowconf.jpg'), gts, err)
    make_grid(rng.choice(pool, args.num, replace=False), kpts, ids, ann_info, img_dir, args.cols,
              osp.join(args.out_dir, 'random.jpg'), gts, err)
    make_grid(order[::-1][:args.num], kpts, ids, ann_info, img_dir, args.cols, osp.join(args.out_dir, 'highconf.jpg'),
              gts, err)
    pe = pool[np.isfinite(err[pool])]
    make_grid(pe[np.argsort(-err[pe])][:args.num], kpts, ids, ann_info, img_dir, args.cols,
              osp.join(args.out_dir, 'worst.jpg'), gts, err)


if __name__ == '__main__':
    main()
