"""
tools/check_pw3d_halpe_ids.py
Kiem tra 3DPW_oc.json / 3DPW_pc.json co dung CHUNG ann_id voi 3DPW_latest_test.json khong.

Boi canh: PW3D.load_data doc 2D Halpe-26 tu 3DPW_test_halpe26.json (tra theo str(ann_id)) cho CA 3 tap
'3dpw', '3dpw-oc', '3dpw-pc'. Neu ann_id trong 3DPW_oc.json / 3DPW_pc.json KHAC voi 3DPW_latest_test.json
(danh so lai) thi 2D Halpe cua nguoi nay duoc ghep voi GT 3D cua nguoi khac -> MPJPE ~300 mm.

Chay (tu thu muc goc repo, khong can GPU):
  python tools/check_pw3d_halpe_ids.py
  python tools/check_pw3d_halpe_ids.py --root data_final/PW3D
"""
import argparse
import json
import os.path as osp

import numpy as np


def load_anns(path):
    """Tra ve {ann_id: (sequence, file_name, bbox)}."""
    with open(path) as f:
        d = json.load(f)
    imgs = {im['id']: im for im in d['images']}
    out = {}
    for a in d['annotations']:
        im = imgs[a['image_id']]
        out[int(a['id'])] = (im['sequence'], im['file_name'], np.asarray(a['bbox'], dtype=np.float64))
    return out


def key_of(seq, fname, bbox):
    return (seq, fname) + tuple(np.round(bbox, 1).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data_final/PW3D')
    args = ap.parse_args()

    test = load_anns(osp.join(args.root, '3DPW_latest_test.json'))
    with open(osp.join(args.root, '3DPW_test_halpe26.json')) as f:
        halpe_ids = set(int(k) for k in json.load(f).keys())
    key2test_id = {key_of(*v): k for k, v in test.items()}
    print(f'3DPW_latest_test : {len(test)} ann | khoang id {min(test)}..{max(test)}')
    print(f'3DPW_test_halpe26: {len(halpe_ids)} ann | khoang id {min(halpe_ids)}..{max(halpe_ids)}')

    for name in ('3DPW_oc.json', '3DPW_pc.json'):
        sub = load_anns(osp.join(args.root, name))
        n = len(sub)
        in_test = same_img = same_bbox = in_halpe = remap_ok = remap_same_id = 0
        bad = []
        for aid, (seq, fname, bbox) in sub.items():
            in_halpe += aid in halpe_ids
            if aid in test:
                in_test += 1
                tseq, tfname, tbbox = test[aid]
                ok_img = (tseq == seq and tfname == fname)
                ok_box = bool(np.abs(tbbox - bbox).max() < 1.0)
                same_img += ok_img
                same_bbox += (ok_img and ok_box)
                if not (ok_img and ok_box) and len(bad) < 5:
                    bad.append((aid, (seq, fname), (tseq, tfname)))
            k = key_of(seq, fname, bbox)
            if k in key2test_id:
                remap_ok += 1
                remap_same_id += (key2test_id[k] == aid)
        print(f'\n[{name}] {n} ann | khoang id {min(sub)}..{max(sub)}')
        print(f'  id co trong file Halpe test        : {in_halpe}/{n}')
        print(f'  id co trong 3DPW_latest_test       : {in_test}/{n}')
        print(f'  cung id + cung (sequence, anh)     : {same_img}/{n}')
        print(f'  cung id + cung anh + cung bbox     : {same_bbox}/{n}   <-- = n thi 2D dung, < n thi 2D SAI NGUOI')
        print(f'  tim duoc nguoi nay trong test (theo anh+bbox): {remap_ok}/{n} | trong do cung id: {remap_same_id}')
        for aid, mine, theirs in bad:
            print(f'    vi du lech: id {aid}: {name}={mine} | test={theirs}')


if __name__ == '__main__':
    main()
