"""
tools/check_3dpw_clean_overlap.py -- Kiem tra muc do tach biet (separation) giua 3dpw-train
  (khi bat DATASET.pw3d_train_exclude_eval_seqs: True) va 4 tap danh gia:
    - 3dpw (test, 35515 mau)
    - 3dpw-oc (toan bo, 20243 mau)
    - 3dpw-pc (toan bo, 2218 mau)
    - 3dpw-crowd (validation crowd, 1923 mau)

Kiem tra o 4 cap do:
  1) Cap sequence (ten video sequence): giao nhau phai = 0
  2) Cap khung hinh (sequence/file_name va img_path): giao nhau phai = 0
  3) Cap tiền tố cảnh / chuỗi quay (scene prefix truoc '_00', '_01'): thong ke xem co cung dia diem quay khong
  4) Kiem tra truc tiep tren class PW3D (datalist that ma DataLoader doc khi train / test)

Chay:
  python tools/check_3dpw_clean_overlap.py
"""
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
    n_anns = 0
    for ann in data['annotations']:
        im = img_by_id[int(ann['image_id'])]
        seq = im['sequence']
        if crowd_only and ('courtyard_hug_00' not in seq and 'courtyard_dancing_00' not in seq):
            continue
        seqs.add(seq)
        frames.add((seq, im['file_name']))
        n_anns += 1
    return seqs, frames, n_anns


def _scene_prefix(seq_name):
    parts = seq_name.split('_')
    if parts[-1].isdigit():
        return '_'.join(parts[:-1])
    return seq_name


def main():
    pw3d_dir = osp.join(cfg.DATASET.jotr_data_root, 'PW3D')
    print('=' * 88)
    print(f'1. KIEM TRA TRUC TIEP TU FILE ANNOTATION JSON TRONG: {pw3d_dir}')
    print('=' * 88)

    tr_seqs, tr_frames, tr_anns = _load_json_stats(osp.join(pw3d_dir, '3DPW_latest_train.json'))
    eval_sets = {
        '3dpw (test)': _load_json_stats(osp.join(pw3d_dir, '3DPW_latest_test.json')),
        '3dpw-oc (full)': _load_json_stats(osp.join(pw3d_dir, '3DPW_oc.json')),
        '3dpw-pc (full)': _load_json_stats(osp.join(pw3d_dir, '3DPW_pc.json')),
        '3dpw-crowd': _load_json_stats(osp.join(pw3d_dir, '3DPW_latest_validation.json'), crowd_only=True),
    }

    print(f'3dpw-train GOC (chua loc): {len(tr_seqs)} sequences | {len(tr_frames)} frames | {tr_anns} mau nguoi')
    print('-' * 88)
    print('Giao nhau cua 3dpw-train GOC (CHUA LOC) voi 4 tap test:')
    all_eval_seqs = set()
    all_eval_frames = set()
    for name, (eseqs, eframes, eanns) in eval_sets.items():
        all_eval_seqs |= eseqs
        all_eval_frames |= eframes
        ov_s = sorted(tr_seqs & eseqs)
        ov_f = len(tr_frames & eframes)
        print(f'  - vs {name:<16} ({len(eseqs):2d} seqs, {eanns:5d} mau): '
              f'trung {len(ov_s):2d} seqs, {ov_f:5d} frames | seqs trung: {ov_s}')

    # Tap clean sau khi loai bo moi sequence/frame cua 4 tap danh gia
    clean_seqs = tr_seqs - all_eval_seqs
    removed_seqs = sorted(tr_seqs & all_eval_seqs)
    print('-' * 88)
    print(f'Sequences bi loai khoi 3dpw-train ({len(removed_seqs)} seqs): {removed_seqs}')
    print(f'Sequences GIU LAI trong 3dpw-train-clean ({len(clean_seqs)} seqs): {sorted(clean_seqs)}')

    # Kiem tra tien to canh quay (scene prefix)
    clean_prefixes = {_scene_prefix(s) for s in clean_seqs}
    eval_prefixes = {_scene_prefix(s) for s in all_eval_seqs}
    shared_prefixes = sorted(clean_prefixes & eval_prefixes)
    print('-' * 88)
    print(f'Kiem tra tien to canh (scene prefix truoc _00/_01):')
    print(f'  - Tien to trung voi 4 tap test ({len(shared_prefixes)}): {shared_prefixes}')

    print('\n' + '=' * 88)
    print('2. KIEM TRA TRUC TIEP TREN DATASET PW3D (pw3d_train_exclude_eval_seqs = True)')
    print('=' * 88)
    cfg.DATASET.pw3d_ocpc_test_only = False
    cfg.DATASET.pw3d_train_exclude_eval_seqs = True
    tf = transforms.ToTensor()

    ds_train_clean = PW3D(tf, '3dpw-train')
    train_paths = {d['img_path'] for d in ds_train_clean.datalist}
    train_seqs_ds = {osp.basename(osp.dirname(d['img_path'])) for d in ds_train_clean.datalist}

    print(f'\n=> PW3D("3dpw-train", clean): {len(ds_train_clean.datalist)} mau | {len(train_seqs_ds)} sequences')
    print('-' * 88)
    print(f'{"Tap danh gia":<18} | {"So mau":>8} | {"So seq":>6} | {"Trung seq":>10} | {"Trung img_path":>14} | Ket luan')
    print('-' * 88)

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

    print('=' * 88)
    assert all_ok, 'PHAT HIEN TRUNG LAP GIUA 3DPW-TRAIN-CLEAN VA TAP TEST!'
    print('KET LUAN: 3dpw-train-clean TACH BIET HOAN TOAN (0 sequence, 0 frame trung) voi ca 4 tap 3dpw, 3dpw-oc, 3dpw-pc, 3dpw-crowd.')


if __name__ == '__main__':
    main()
