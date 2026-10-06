import os, sys
sys.path.append('./lib')
sys.path.append('./')
import argparse
import numpy as np
import torch
from tqdm import tqdm

from core.config import cfg, update_config
from core.base import get_dataloader


def pa_align(pred, gt):
    """Procrustes align pred (J,3) to gt (J,3). Trả về pred đã căn chỉnh."""
    mu_p, mu_g = pred.mean(0), gt.mean(0)
    p, g = pred - mu_p, gt - mu_g
    var_p = (p ** 2).sum()
    U, S, Vt = np.linalg.svd(p.T @ g)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / max(var_p, 1e-8)
    return scale * p @ R + mu_g


def evaluate(lifter, loader, name, max_batches):
    mpjpe_sum, pa_sum, n = 0.0, 0.0, 0
    per_joint = np.zeros(17)
    per_joint_cnt = np.zeros(17)
    for i, (inputs, targets, meta) in enumerate(tqdm(loader, desc=name)):
        if max_batches > 0 and i >= max_batches:
            break
        if i == 0:
            print(f'[{name}] targets keys: {list(targets.keys())} | meta keys: {list(meta.keys())}')
        img_dummy = None
        pose2d = inputs['joints'].cuda().float()
        joints_mask = inputs['joints_mask'].cuda().float()
        key = 'orig_joint_cam' if 'orig_joint_cam' in targets else 'fit_joint_cam'
        gt = targets[key].float()
        gt = (gt - gt[:, 0:1, :]).numpy()  # (B,17,3) mét, root-relative

        with torch.no_grad():
            pred = lifter(pose2d, joints_mask=joints_mask)
            pred = (pred - pred[:, 0:1, :]).cpu().numpy()

        if 'orig_joint_valid' in meta:
            valid = meta['orig_joint_valid'].numpy().reshape(gt.shape[0], 17, -1)[..., 0]
        else:
            valid = np.ones(gt.shape[:2])

        for b in range(gt.shape[0]):
            v = valid[b] > 0
            if v.sum() < 3:
                continue
            err = np.linalg.norm(pred[b] - gt[b], axis=-1)
            mpjpe_sum += err[v].mean()
            per_joint[v] += err[v]
            per_joint_cnt[v] += 1
            aligned = pa_align(pred[b][v], gt[b][v])
            pa_sum += np.linalg.norm(aligned - gt[b][v], axis=-1).mean()
            n += 1
    mpjpe = mpjpe_sum / max(n, 1) * 1000
    pa = pa_sum / max(n, 1) * 1000
    print(f'==> [{name}] MotionBERT lift: MPJPE={mpjpe:.2f} mm | PA-MPJPE={pa:.2f} mm | samples={n}')
    pj = per_joint / np.maximum(per_joint_cnt, 1) * 1000
    names = ['Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck',
             'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist']
    print('    per-joint MPJPE (mm): ' + ', '.join(f'{k}={v:.0f}' for k, v in zip(names, pj)))
    return mpjpe, pa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/train_student.yml')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--max_batches', type=int, default=200, help='<=0: toàn bộ')
    parser.add_argument('--debug', action='store_true', default=False)
    args, _ = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    update_config(args.cfg)

    # Dựng ARTS ở mode student để có pose_lifter (MotionBERT) với đúng trọng số finetune
    from models.ARTS import ARTS
    old_mode = cfg.MODEL.name
    cfg.MODEL.name = 'student'
    model = ARTS(num_joint=17, embed_dim=cfg.MODEL.get('hpe_dim', 512)).cuda().eval()
    cfg.MODEL.name = old_mode
    lifter = model.lift_2d_to_3d

    cfg.TEST.batch_size = max(cfg.TEST.batch_size, 32)

    print('===== TRAIN (3dpw-train, có augmentation) =====')
    _, train_loader = get_dataloader(args, cfg.DATASET.train_list, is_train=True)
    evaluate(lifter, train_loader, 'train', args.max_batches)

    print('===== TEST =====')
    test_names = cfg.DATASET.test_list
    _, test_loaders = get_dataloader(args, test_names, is_train=False)
    for nm, loader in zip(test_names, test_loaders):
        evaluate(lifter, loader, f'test-{nm}', args.max_batches)


if __name__ == '__main__':
    main()
