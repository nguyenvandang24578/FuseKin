"""Do loi pose DiffLifter tren tung bo train/test, khong train.

Lenh cu van eval toan bo cac split 3DPW, sampler theo checkpoint.
--train_sets config: them cac bo DATASET.train_list, giu augmentation cua loader train.
--max_samples 512: lay ngau nhien khong hoan lai toi da 512 mau MOI bo; 0 = toan bo.
--student_input: dung YAML Student, GT/2D/conf nhu Student va reset noise seed=0 moi batch.
In MPJPE H36M-17, MPJPE16 bo pelvis, loi tung khop; khong phai MPJPE mesh 14 khop.
--out_dir: tuy chon luu log, summary.json, samples.csv va YAML vao thu muc moi.
"""
import os
import sys
sys.path.append('./lib')
sys.path.append('./')

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from core.config import update_config, cfg
from utils.jotr_dataset import TRAIN_DATASETS, get_train_dataset, get_test_dataset
from diag_difflifter import JOINT_NAMES, build_model, make_cond, get_gt


def parse_args():
    p = argparse.ArgumentParser(description='Do loi pose DiffLifter, khong chay Teacher/Student mesh')
    p.add_argument('--cfg', required=True)
    p.add_argument('--ckpt', required=True, help='checkpoint lifter best_epoch.bin, khong phai checkpoint mesh')
    p.add_argument('--gpu', default='0')
    p.add_argument('--sets', default='3dpw-train-noaug,3dpw-val,3dpw',
                   help='cac split eval cach nhau boi dau phay; config = DATASET.test_list; rong = bo qua')
    p.add_argument('--train_sets', default='',
                   help='cac bo train cach nhau boi dau phay; config = DATASET.train_list; mac dinh tat')
    p.add_argument('--max_samples', type=int, default=0, help='gioi han mau ngau nhien moi bo; 0 = toan bo nhu cu')
    p.add_argument('--student_input', action='store_true',
                   help='bat DIFFLIFT.eval_student_input: target/sampler/noise nhu pose chinh cua Student')
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--workers', type=int, default=None, help='mac dinh DATASET.workers')
    p.add_argument('--S', type=int, default=0, help='>0: ghi de buoc DDIM; Student dung MODEL, lenh cu dung checkpoint')
    p.add_argument('--K', type=int, default=0, help='>0: ghi de K gia thuyet; K>1 lay mean')
    p.add_argument('--seed', type=int, default=0, help='seed chon mau/augmentation; student_input van dung noise seed=0')
    p.add_argument('--out_dir', default='', help='thu muc moi de luu ket qua; rong = chi in nhu cu')
    args = p.parse_args()
    if args.batch_size < 1 or args.max_samples < 0 or args.S < 0 or args.K < 0 or args.seed < 0:
        p.error('batch_size phai >0; max_samples/S/K/seed phai >=0')
    if args.workers is not None and args.workers < 0:
        p.error('workers phai >=0')
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def seed_worker(worker_id):
    # Worker moi co RNG rieng, khong phu thuoc viec chon batch cua lifter.
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def expand_sets(value, defaults):
    names = list(defaults) if value == 'config' else [v.strip() for v in value.split(',') if v.strip()]
    return list(dict.fromkeys(names))


def target_label(name, is_train):
    if name in ('MSCOCO', 'CrowdPose') or (is_train and cfg.DATASET.fusekin_unified
                                             and not cfg.DATASET.lift_only and name in TRAIN_DATASETS):
        return 'NeuralAnnot pseudo-GT (H36M regressor x SMPL)'
    if name == 'Human36M':
        return 'mocap H36M; test dung GT 2D sach' if not is_train else 'mocap H36M'
    if name == 'MuCo':
        return 'GT 3D MuCo'
    return 'H36M regressor x 3DPW GT SMPL mesh'


def validate_student_input(train_names, ckpt):
    if cfg.MODEL.name != 'student':
        raise ValueError('--student_input can YAML Student (DiffLifter)')
    if cfg.DATASET.lift_only:
        raise ValueError('Do dau vao Student can lift_only=False nhu luc train mesh')
    if any(name in TRAIN_DATASETS for name in train_names) and not cfg.DATASET.fusekin_unified:
        raise ValueError('Do 4 bo train Student can fusekin_unified=True; dung YAML *_jotr.yml')
    saved_conf = ckpt['difflift_cfg'].get('soft_conf')
    if saved_conf is not None and bool(saved_conf) != bool(cfg.DATASET.lift_soft_conf):
        raise ValueError('lift_soft_conf trong YAML Student khac checkpoint DiffLifter')


def fit_weights(meta, batch, device):
    # Cung mask GT voi log lift_err_mm trong Student_Trainer; khong dung mask detector 2D.
    if 'fit_joint_trunc' not in meta or 'is_valid_fit' not in meta:
        return None
    weights = meta['fit_joint_trunc'].to(device).float().reshape(batch, 17, -1)[..., 0]
    valid_fit = meta['is_valid_fit'].to(device).float().reshape(batch, -1)[:, 0]
    return weights * valid_fit[:, None]


@torch.no_grad()
def eval_mpjpe(model, name, args, device, is_train=False, log=print, sample_writer=None):
    ds = get_train_dataset(name, None) if is_train else get_test_dataset(name, None)
    total = len(ds)
    if total == 0:
        raise ValueError('Dataset rong sau khi loc annotation/Halpe/SMPL')
    if args.max_samples > 0:
        n_select = min(args.max_samples, total)
        indices = np.sort(np.random.default_rng(args.seed).choice(total, n_select, replace=False)).tolist()
        selected = Subset(ds, indices)
    else:
        indices = range(total)
        selected = ds
    # Lay mau rieng tung bo, khong ap dung ratio/oversampling cua loader train.
    set_seed(args.seed)
    workers = cfg.DATASET.workers if args.workers is None else args.workers
    loader = DataLoader(selected, batch_size=args.batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == 'cuda', worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(args.seed))
    reg = torch.as_tensor(ds.h36m_joint_regressor, dtype=torch.float32, device=device)
    student_input = bool(cfg.DIFFLIFT.eval_student_input)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    split = 'train' if is_train else ('train-noaug' if name == '3dpw-train-noaug' else 'eval')
    source = target_label(name, is_train)
    log(f'\n[{split}/{name}] chon {len(selected)}/{total} mau | GT: {source}')
    if is_train:
        log('  Augmentation: giu nguyen loader/YAML train; day khong phai eval train-noaug.')
    n, n_skip, seen = 0, 0, 0
    joint_sum = np.zeros(17, dtype=np.float64)
    masked_sum, masked_count = 0.0, 0.0
    for inp, tgt, meta in tqdm(loader, desc=f'{split}/{name}'):
        batch = inp['joints'].shape[0]
        cond = make_cond(inp, device)
        if cond.shape != (batch, 17, 6) or not torch.isfinite(cond).all():
            raise ValueError('Dieu kien phai la (B,17,6) huu han: [x,y,conf,cam_bbox]')
        gt = get_gt(tgt, reg, device)
        if gt.shape != (batch, 17, 3) or not torch.isfinite(gt).all():
            raise ValueError('GT phai la H36M-17 huu han, met, root-relative')
        valid = gt.abs().flatten(1).sum(1) > 0
        n_skip += int((~valid).sum())
        errors = torch.full((batch, 17), float('nan'), device=device)
        if valid.any():
            if student_input:
                # Giong ARTS.lift_diffusion: seed=0 LAI moi batch, du doan truoc khi loc GT.
                cond[..., :2] *= (cond[..., 2:3] > 0)
                gen.manual_seed(0)
                pred, _, _ = model(cond, generator=gen)
                pred = pred - pred[:, 0:1]
                errors = torch.norm(pred - gt, dim=-1)
            else:
                # Giu lenh cu: RNG noi tiep giua batch, bo GT rong truoc khi sample.
                pred, _, _ = model(cond[valid], generator=gen)
                pred = pred - pred[:, 0:1]
                errors[valid] = torch.norm(pred - gt[valid], dim=-1)
            if not torch.isfinite(errors[valid]).all():
                raise ValueError('DiffLifter xuat NaN/Inf')
            joint_sum += errors[valid].double().sum(0).cpu().numpy()
            n += int(valid.sum())
        weights = fit_weights(meta, batch, device) if is_train and student_input else None
        if weights is not None:
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError('Mask GT khong hop le')
            masked_sum += float((errors[valid] * weights[valid]).sum())
            masked_count += float(weights[valid].sum())
        if sample_writer is not None:
            for i in range(batch):
                good = bool(valid[i])
                row = {'dataset': name, 'split': split, 'index': int(indices[seen + i]),
                       'img_path': str(meta.get('img_path', [''] * batch)[i]), 'gt_source': source,
                       'valid_gt': good, 'missing_2d': int((cond[i, :, 2] <= 0).sum()),
                       'mpjpe17_mm': float(errors[i].mean()) * 1000 if good else None,
                       'mpjpe16_mm': float(errors[i, 1:].mean()) * 1000 if good else None}
                sample_writer.writerow(row)
        seen += batch
    if n == 0:
        raise ValueError(f'Khong co GT hop le trong {seen} mau; khong bao MPJPE=0')
    per_joint = (joint_sum / n * 1000).tolist()
    result = {'dataset': name, 'split': split, 'dataset_n': total, 'selected_n': seen, 'n': n, 'n_skip': n_skip,
              'selection': 'random_without_replacement' if args.max_samples else 'all_in_order',
              'selected_indices': list(indices) if args.max_samples else None,
              'gt_source': source, 'mpjpe17_mm': float(np.mean(per_joint)),
              'mpjpe16_mm': float(np.mean(per_joint[1:])),
              'fit_mask_mpjpe_mm': masked_sum / masked_count * 1000 if masked_count > 0 else None,
              'fit_mask_weight_sum': masked_count, 'per_joint_mm': dict(zip(JOINT_NAMES, per_joint))}
    log(f'  MPJPE(17)={result["mpjpe17_mm"]:.3f} | MPJPE(16, bo pelvis)={result["mpjpe16_mm"]:.3f} mm '
        f'| hop le={n}, bo GT rong={n_skip}')
    if result['fit_mask_mpjpe_mm'] is not None:
        log(f'  MPJPE theo mask GT cua log Student={result["fit_mask_mpjpe_mm"]:.3f} mm '
            '(gom pelvis; tong loi / tong trong so khop tren tap con)')
    log('  Tung khop (mm): ' + ', '.join(f'{name}={value:.2f}' for name, value in zip(JOINT_NAMES, per_joint)))
    return result


def main():
    args = parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if ',' in args.gpu:
        raise ValueError('Dung mot GPU de giu dung cach chia batch/noise cua phep do')
    update_config(args.cfg)
    cfg.DIFFLIFT.eval_student_input = bool(args.student_input or cfg.DIFFLIFT.eval_student_input)
    train_names = expand_sets(args.train_sets, cfg.DATASET.train_list)
    test_names = expand_sets(args.sets, cfg.DATASET.test_list)
    if not train_names and not test_names:
        raise ValueError('Can it nhat mot dataset train hoac test')
    if any(name not in TRAIN_DATASETS and name not in ('3dpw-train', 'PW3D') for name in train_names):
        raise ValueError('--train_sets chi nhan 4 bo train, 3dpw-train, PW3D hoac config')
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    if cfg.DIFFLIFT.eval_student_input:
        validate_student_input(train_names, ckpt)
    model = build_model(ckpt, device)
    if cfg.DIFFLIFT.eval_student_input:
        model.sampling_timesteps = int(cfg.MODEL.difflift_steps)
        model.num_proposals = int(cfg.MODEL.difflift_k)
    if args.S > 0:
        model.sampling_timesteps = args.S
    if args.K > 0:
        model.num_proposals = args.K
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
            raise FileExistsError(f'Thu muc ket qua khong rong: {out_dir}; chon thu muc moi')
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'config.yml').write_text(Path(args.cfg).read_text(encoding='utf-8-sig'), encoding='utf-8')
    lines = []

    def log(message):
        print(message, flush=True)
        lines.append(message)
        if out_dir is not None:
            (out_dir / 'eval_log.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    summary = {'cfg': args.cfg, 'ckpt': args.ckpt, 'epoch': ckpt.get('epoch'), 'args': vars(args),
               'S': model.sampling_timesteps, 'K': model.num_proposals, 'student_input': cfg.DIFFLIFT.eval_student_input,
               'noise': 'seed=0 reset each batch' if cfg.DIFFLIFT.eval_student_input else 'seed stream across batches',
               'train_list_yaml': list(cfg.DATASET.train_list), 'select_dataset': cfg.TEST.select_dataset,
               'configured_lifter_checkpoint': cfg.MODEL.difflift_checkpoint,
               'data_settings': {key: cfg.DATASET[key] for key in ('lift_only', 'fusekin_unified', 'lift_soft_conf',
                                 'pw3d_2d_source', 'h36m_2d_source', 'muco_2d_source', 'halpe26_conf_thr', 'no_rot_aug')},
               'results': [], 'errors': []}
    log(f'ckpt={args.ckpt} | epoch={ckpt.get("epoch", "?")} | S={model.sampling_timesteps} K={model.num_proposals}')
    log(f'DATASET={summary["data_settings"]} | batch={args.batch_size} | noise={summary["noise"]}')
    log('MPJPE H36M-17 root-relative, mm; MPJPE16 bo pelvis; khong phai metric mesh 14 khop.')
    log('Tap con la chan doan, khong thay the eval toan tap. Moi bo dung GT rieng ghi trong log/summary.')
    if cfg.DIFFLIFT.eval_student_input:
        log('Che do Student: chi do pose chinh DiffLifter; khong chay Teacher, gate, KD hay mesh head.')
        if (model.sampling_timesteps, model.num_proposals) != (cfg.MODEL.difflift_steps, cfg.MODEL.difflift_k):
            log('CHU Y: --S/--K da ghi de sampler YAML, ket qua khong con la pose chinh cua run Student.')
    sample_file = None
    try:
        writer = None
        if out_dir is not None:
            sample_file = (out_dir / 'samples.csv').open('w', encoding='utf-8', newline='')
            writer = csv.DictWriter(sample_file, fieldnames=['dataset', 'split', 'index', 'img_path', 'gt_source',
                                                            'valid_gt', 'missing_2d', 'mpjpe17_mm', 'mpjpe16_mm'])
            writer.writeheader()
        for is_train, names in ((True, train_names), (False, test_names)):
            for name in names:
                try:
                    summary['results'].append(eval_mpjpe(model, name, args, device, is_train, log, writer))
                except Exception as exc:
                    summary['errors'].append({'dataset': name, 'train_loader': is_train,
                                              'type': type(exc).__name__, 'message': str(exc)})
                    log(f'[{name}] LOI: {type(exc).__name__}: {exc}')
                if sample_file is not None:
                    sample_file.flush()
                if out_dir is not None:
                    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                                    allow_nan=False), encoding='utf-8')
    finally:
        if sample_file is not None:
            sample_file.close()
    log('\nTOM TAT: MPJPE17 / MPJPE16, mm (thap hon tot hon)')
    for result in summary['results']:
        log(f'  {result["split"]}/{result["dataset"]}: n={result["n"]}/{result["selected_n"]} '
            f'| {result["mpjpe17_mm"]:.3f} / {result["mpjpe16_mm"]:.3f}')
    if out_dir is not None:
        log(f'Ket qua: {out_dir}')
    if summary['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
