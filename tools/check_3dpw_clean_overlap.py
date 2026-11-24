"""
tools/check_3dpw_clean_overlap.py -- Kiem tra muc do tach biet (separation) giua 3dpw-train (+ 3dpw-val clean)
  (khi bat DATASET.pw3d_train_exclude_eval_seqs: True va pw3d_train_include_val_clean: True) va 4 tap danh gia:
    - 3dpw (test, 35515 mau)
    - 3dpw-oc (toan bo, 20243 mau)
    - 3dpw-pc (toan bo, 2218 mau)
    - 3dpw-crowd (validation crowd, 1923 mau)

In ro:
  1) So sequence va so mau (samples) cua 3dpw-train goc, 3dpw-val goc, va sau khi loc sach
  2) Danh sach tung sequence duoc giu lai + so mau cua tung sequence
  3) Bang doi chieu truc tiep tren PW3D('3dpw-train') voi ca 4 tap danh gia (dam bao 0% trung lap)

Chay:
  python tools/check_3dpw_clean_overlap.py
"""
from collections import Counter
import json
import os
import os.path as osp
import sys

sys.path.insert(0, './lib')
sys.path.insert(0, './')

import torchvision.transforms as transforms
from core.config import cfg
from data_final.PW3D.dataset import PW3D


def _load_json_stats(json_path, crowd_only=False):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    img_by_id = {int(im['id']): im for im in data['images']}
    seqs = set()
    frames = set()
    seq_counts = Counter()
    for ann in data['annotations']:
        im = img_by_id[int(ann['image_id'])]
        seq = im['sequence']
        if crowd_only and ('courtyard_hug_00' not in seq and 'courtyard_dancing_00' not in seq):
            continue
        seqs.add(seq)
        frames.add((seq, im['file_name']))
        seq_counts[seq] += 1
    return seqs, frames, sum(seq_counts.values()), seq_counts


def main():
    pw3d_dir = osp.join(cfg.DATASET.jotr_data_root, 'PW3D')
    print('=' * 92)
    print(f'1. THONG KE TU FILE ANNOTATION JSON TRONG: {pw3d_dir}')
    print('=' * 92)

    tr_seqs, tr_frames, tr_anns, tr_cnt = _load_json_stats(osp.join(pw3d_dir, '3DPW_latest_train.json'))
    va_seqs, va_frames, va_anns, va_cnt = _load_json_stats(osp.join(pw3d_dir, '3DPW_latest_validation.json'))
    eval_sets = {
        '3dpw (test)': _load_json_stats(osp.join(pw3d_dir, '3DPW_latest_test.json')),
        '3dpw-oc (full)': _load_json_stats(osp.join(pw3d_dir, '3DPW_oc.json')),
        '3dpw-pc (full)': _load_json_stats(osp.join(pw3d_dir, '3DPW_pc.json')),
        '3dpw-crowd': _load_json_stats(osp.join(pw3d_dir, '3DPW_latest_validation.json'), crowd_only=True),
    }

    all_eval_seqs = set()
    for _, (eseqs, _, _, _) in eval_sets.items():
        all_eval_seqs |= eseqs

    clean_tr_seqs = sorted(tr_seqs - all_eval_seqs)
    clean_va_seqs = sorted(va_seqs - all_eval_seqs)
    n_clean_tr = sum(tr_cnt[s] for s in clean_tr_seqs)
    n_clean_va = sum(va_cnt[s] for s in clean_va_seqs)

    print(f'- 3dpw-train GOC : {len(tr_seqs):2d} seqs ({tr_anns:5d} mau) -> GIU LAI: {len(clean_tr_seqs):2d} seqs ({n_clean_tr:5d} mau)')
    print(f'- 3dpw-val   GOC : {len(va_seqs):2d} seqs ({va_anns:5d} mau) -> GIU LAI: {len(clean_va_seqs):2d} seqs ({n_clean_va:5d} mau)')
    print(f'=> TONG 3DPW CLEAN (train + val): {len(clean_tr_seqs) + len(clean_va_seqs)} sequences | {n_clean_tr + n_clean_va} mau')
    print('-' * 92)
    print('Chi tiet tung sequence SACH duoc giu lai:')
    for s in clean_tr_seqs:
        print(f'  [train] {s:<38} : {tr_cnt[s]:5d} mau')
    for s in clean_va_seqs:
        print(f'  [val]   {s:<38} : {va_cnt[s]:5d} mau')

    print('\n' + '=' * 92)
    print('2. KIEM TRA TRUC TIEP TREN PW3D("3dpw-train") VOI exclude_eval_seqs=True & include_val_clean=True')
    print('=' * 92)
    cfg.DATASET.pw3d_ocpc_test_only = False
    cfg.DATASET.pw3d_train_exclude_eval_seqs = True
    cfg.DATASET.pw3d_train_include_val_clean = True
    tf = transforms.ToTensor()

    ds_train_clean = PW3D(tf, '3dpw-train')
    train_paths = {d['img_path'] for d in ds_train_clean.datalist}
    train_seqs_ds = {osp.basename(osp.dirname(d['img_path'])) for d in ds_train_clean.datalist}

    print(f'\n=> TONG SO MAU SE TRAIN TRONG PW3D("3dpw-train"): {len(ds_train_clean.datalist)} mau | {len(train_seqs_ds)} sequences')
    print('-' * 92)
    print(f'{"Tap danh gia":<18} | {"So mau":>8} | {"So seq":>6} | {"Trung seq":>10} | {"Trung img_path":>14} | Ket luan')
    print('-' * 92)

    all_ok = True
    for ename in ('3dpw', '3dpw-oc', '3dpw-pc', '3dpw-crowd'):
        ds_e = PW3D(tf, ename)
        epaths = {d['img_path'] for d in ds_e.datalist}
        eseqs = {osp.basename(osp.dirname(d['img_path'])) for d in ds_e.datalist}
        n_ov_seq = len(train_seqs_ds & eseqs)
        n_ov_path = len(train_paths & epaths)
        status = 'TACH BIET 100% (0% LEAK)' if (n_ov_seq == 0 and n_ov_path == 0) else 'CO TRUNG LAP!'
        if n_ov_seq != 0 or n_ov_path != 0:
            all_ok = False
        print(f'{ename:<18} | {len(ds_e.datalist):8d} | {len(eseqs):6d} | {n_ov_seq:10d} | {n_ov_path:14d} | {status}')

    print('=' * 92)
    assert all_ok, 'PHAT HIEN TRUNG LAP GIUA 3DPW-TRAIN-CLEAN VA TAP TEST!'
    print(f'KET LUAN: {len(train_seqs_ds)} sequences ({len(ds_train_clean.datalist)} mau) TACH BIET 100% voi ca 4 tap test.')


if __name__ == '__main__':
    main()
