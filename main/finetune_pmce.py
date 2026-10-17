"""Finetune PMCE (GraphormerNet, lift 2D -> 3D co dac trung anh) tren dataset FuseKin voi 2D Halpe-26.

Mac dinh: train h36m-train + muco-train + 3dpw-train (Setting 2: CO 3dpw-train -> bao rieng, ro ri sang 3DPW-OC/PC),
          theo doi 3dpw-val moi epoch, test cuoi 3dpw-test + h36m-test bang EPOCH CUOI (latest).
Giong PMCE goc (PMCE/base.py LiftTrainer + train_pose_h36m.yml): khoi tao tu checkpoint PMCE Human36M, L1 tren khop 3D
  (mm, root-relative), Adam lr 5e-5, 2D chuan hoa X/w*2 - [1, h/w] KHONG tru root, 1 anh nhan ban 16 frame,
  dac trung anh tinh san (tools/extract_img_feat.py, SPIN ResNet-50 crop 256), KHONG augment.
Chi so danh gia: MPJPE / PA-MPJPE 17 khop H36M root-relative (mm), giong main/compare_pmce_lift.py.

Chay (moi truong FuseKin, tu thu muc goc repo; can da trich dac trung cho moi tap dung o duoi):
  python main/finetune_pmce.py --cfg ./config/finetune_pmce_halpe26.yaml \
      --pmce_ckpt /path/to/pmce_human36m.pth.tar --gpu 0,1,2,3
  # chi danh gia 1 checkpoint:
  python main/finetune_pmce.py --cfg ./config/finetune_pmce_halpe26.yaml --pmce_ckpt <ckpt> --eval_only
"""
import os, sys, json, time
sys.path.append('./lib')
sys.path.append('./')
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from core.config import cfg, update_config
from utils.pmce_data import PMCESet, FEAT_SETS

SEQLEN = 16
JOINTS = ['Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose',
          'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist']


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, help='yml co DATASET.* (nguon 2D Halpe-26), MODEL.hpe_dim/hpe_dep')
    ap.add_argument('--pmce_ckpt', required=True, help='checkpoint PMCE khoi tao (model_state_dict) hoac ckpt finetune')
    ap.add_argument('--train_sets', nargs='+', default=['h36m-train', 'muco-train', '3dpw-train'], choices=FEAT_SETS)
    ap.add_argument('--val_sets', nargs='+', default=['3dpw-val'], choices=FEAT_SETS)
    ap.add_argument('--test_sets', nargs='+', default=['3dpw-test', 'h36m-test'], choices=FEAT_SETS)
    ap.add_argument('--feat_dir', default='data_final/feat_cache')
    ap.add_argument('--save_dir', default='experiment/finetune_pmce_halpe26')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--lr_decay', type=float, default=0.95, help='nhan lr moi epoch')
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--final_ckpt', default='latest', choices=['latest', 'best'],
                    help="test cuoi: 'latest' (epoch cuoi) | 'best' (theo val_sets[0])")
    ap.add_argument('--eval_only', action='store_true', help='Chi danh gia --pmce_ckpt tren val_sets + test_sets')
    ap.add_argument('--gpu', default='0')
    return ap.parse_args()


def load_pmce(path, device, n_gpu):
    from models.PoseEstimation import get_model
    model = get_model(num_joint=len(JOINTS), embed_dim=cfg.MODEL.hpe_dim, depth=cfg.MODEL.hpe_dep, pretrained=False)
    ck = torch.load(path, map_location='cpu')
    sd = ck.get('model_state_dict', ck)
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[ckpt] {path} | epoch={ck.get("epoch", "?")} | missing={len(missing)} {missing[:3]} | '
          f'unexpected={len(unexpected)} {unexpected[:3]}')
    if missing or unexpected:
        print('  CANH BAO: nap khong khop hoan toan (kiem tra ban 17 khop, hpe_dim=256, hpe_dep=3).')
    model = model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)
    return model


def forward(model, x, feat):
    """x (B, 17, 2), feat (B, 2048) -> pred (B, 17, 3) mm, root-relative."""
    xs = x.unsqueeze(1).repeat(1, SEQLEN, 1, 1)
    fs = feat.unsqueeze(1).repeat(1, SEQLEN, 1)
    pred = model(xs, fs)
    return pred - pred[:, :1]


@torch.no_grad()
def evaluate(model, loader, device, desc):
    from finetune_motionbert import batch_pa_errors
    model.eval()
    err, pa = [], []
    for x, mask, feat, gt, valid in tqdm(loader, desc=desc):
        x, feat, gt = x.to(device), feat.to(device), gt.to(device)
        pred = forward(model, x, feat) / 1000.0          # mm -> met
        err.append((torch.norm(pred - gt, dim=-1) * 1000).cpu().numpy())
        pa.append((batch_pa_errors(pred, gt) * 1000).cpu().numpy())
    err, pa = np.concatenate(err), np.concatenate(pa)
    return {'mpjpe': float(err.mean()), 'pa_mpjpe': float(pa.mean()), 'per_joint': err.mean(0).tolist(), 'n': len(pa)}


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    update_config(args.cfg)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda')
    n_gpu = torch.cuda.device_count()
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, 'log.txt')

    def log(msg):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    log(f'===== finetune_pmce {time.strftime("%Y-%m-%d %H:%M")} | {vars(args)}')
    log(f'[cfg] h36m_2d_source={cfg.DATASET.get("h36m_2d_source")} | muco_2d_source={cfg.DATASET.get("muco_2d_source")} '
        f'| pw3d_2d_source={cfg.DATASET.pw3d_2d_source} | conf_thr={cfg.DATASET.halpe26_conf_thr}')

    def make_loader(names, shuffle):
        sets = [PMCESet(n, args.feat_dir) for n in names]
        ds = sets[0] if len(sets) == 1 else ConcatDataset(sets)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                          pin_memory=True, drop_last=shuffle)

    model = load_pmce(args.pmce_ckpt, device, n_gpu)
    eval_loaders = {n: make_loader([n], False) for n in dict.fromkeys(args.val_sets + args.test_sets)}

    def run_eval(tag):
        out = {}
        for n, ld in eval_loaders.items():
            r = evaluate(model, ld, device, f'{tag} {n}')
            out[n] = r
            log(f'[{tag} {n}] MPJPE: {r["mpjpe"]:.2f} mm | PA-MPJPE: {r["pa_mpjpe"]:.2f} mm | n={r["n"]}')
        return out

    if args.eval_only:
        res = run_eval('Eval')
        with open(os.path.join(args.save_dir, 'eval_only.json'), 'w') as f:
            json.dump(res, f, indent=2)
        return

    train_loader = make_loader(args.train_sets, True)
    log(f'# train: {len(train_loader.dataset)} mau ({args.train_sets}) | {len(train_loader)} iter/epoch | GPU={n_gpu}')
    run_eval('Epoch 0')   # truoc finetune (diem xuat phat)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    best = float('inf')

    def save(path, epoch):
        torch.save({'epoch': epoch, 'model_state_dict': raw_model.state_dict(),
                    'fusekin_meta': {'train_sets': args.train_sets, 'lr': args.lr, 'init': args.pmce_ckpt}}, path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, run_err, nb = time.time(), 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for x, mask, feat, gt, valid in pbar:
            x, feat = x.to(device, non_blocking=True), feat.to(device, non_blocking=True)
            gt_mm, valid = gt.to(device) * 1000.0, valid.to(device)
            pred = forward(model, x, feat)
            # L1 co mat na (giong CoordLoss cua PMCE), chuan hoa theo so khop hop le
            loss = (torch.abs(pred - gt_mm) * valid).sum() / (valid.sum() * 3).clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                e = (torch.norm(pred - gt_mm, dim=-1) * valid[..., 0]).sum() / valid[..., 0].sum().clamp_min(1.0)
            run_loss += loss.item()
            run_err += e.item()
            nb += 1
            if nb % 50 == 0:
                pbar.set_postfix(loss=f'{run_loss / nb:.2f}', mpjpe=f'{run_err / nb:.1f}')
        lr_now = optimizer.param_groups[0]['lr']
        log(f'[Epoch {epoch}/{args.epochs}] {(time.time() - t0) / 60:.1f}m | lr {lr_now:.2e} | '
            f'train L1 {run_loss / nb:.2f} mm | train MPJPE {run_err / nb:.2f} mm')
        res = run_eval(f'Epoch {epoch}')
        save(os.path.join(args.save_dir, 'latest_epoch.bin'), epoch)
        save(os.path.join(args.save_dir, f'epoch_{epoch}.bin'), epoch)
        v = res[args.val_sets[0]]['mpjpe']
        if v < best:
            best = v
            save(os.path.join(args.save_dir, 'best_epoch.bin'), epoch)
            log(f'--> best theo {args.val_sets[0]}: {v:.2f} mm (epoch {epoch})')
        for g in optimizer.param_groups:
            g['lr'] *= args.lr_decay

    final = os.path.join(args.save_dir, f'{args.final_ckpt}_epoch.bin')
    ck = torch.load(final, map_location='cpu')
    raw_model.load_state_dict(ck['model_state_dict'])
    log(f'===== Test cuoi voi {args.final_ckpt} checkpoint (epoch {ck["epoch"]}) =====')
    res = run_eval('Final')
    with open(os.path.join(args.save_dir, 'final_test.json'), 'w') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
