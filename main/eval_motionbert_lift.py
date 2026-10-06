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


def get_gt_joints(targets, dataset, device='cuda'):
    """
    Ưu tiên orig_joint_cam (có sẵn ở train).
    Nếu không có (test set), regress H36M-17 từ smpl_mesh_cam.
    Trả về np.ndarray (B,17,3), root-relative, mét.
    """
    if 'orig_joint_cam' in targets:
        gt = targets['orig_joint_cam'].float().numpy()
        return gt - gt[:, 0:1, :]
    if 'fit_joint_cam' in targets:
        gt = targets['fit_joint_cam'].float().numpy()
        return gt - gt[:, 0:1, :]
    # Test set: chỉ có mesh → regress joint
    mesh = targets['smpl_mesh_cam'].to(device).float()  # (B,6890,3)
    h36m_reg = torch.as_tensor(
        dataset.h36m_joint_regressor, device=device, dtype=mesh.dtype
    )  # (17,6890)
    gt_joints = torch.matmul(
        h36m_reg.unsqueeze(0).expand(mesh.shape[0], -1, -1), mesh
    )  # (B,17,3)
    gt_joints = gt_joints - gt_joints[:, 0:1, :]
    return gt_joints.cpu().numpy()


def evaluate(lifter, loader, dataset, name, max_batches):
    mpjpe_sum, pa_sum, n = 0.0, 0.0, 0
    per_joint = np.zeros(17)
    per_joint_cnt = np.zeros(17)
    for i, (inputs, targets, meta) in enumerate(tqdm(loader, desc=name)):
        if max_batches > 0 and i >= max_batches:
            break
        if i == 0:
            print(f'\n[{name}] targets keys: {list(targets.keys())} | meta keys: {list(meta.keys())}')

        pose2d = inputs['joints'].cuda().float()
        joints_mask = inputs['joints_mask'].cuda().float()

        gt = get_gt_joints(targets, dataset)  # (B,17,3) mét, root-relative, numpy

        with torch.no_grad():
            pred = lifter(pose2d, joints_mask=joints_mask)
            pred = (pred - pred[:, 0:1, :]).cpu().numpy()

        # Mask hợp lệ (dùng khi train set có)
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
    joint_names = ['Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso',
                   'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
                   'R_Shoulder', 'R_Elbow', 'R_Wrist']
    pj = per_joint / np.maximum(per_joint_cnt, 1) * 1000
    print('    per-joint MPJPE (mm): ' + ', '.join(f'{k}={v:.0f}' for k, v in zip(joint_names, pj)))
    return mpjpe, pa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/train_student.yml')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--max_batches', type=int, default=200, help='<=0: toàn bộ dataset')
    parser.add_argument('--debug', action='store_true', default=False)
    args, _ = parser.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    update_config(args.cfg)

    # Dựng ARTS ở mode student để lấy pose_lifter (MotionBERT đã finetune)
    from models.ARTS import ARTS
    old_mode = cfg.MODEL.name
    cfg.MODEL.name = 'student'
    artnet = ARTS(num_joint=17, embed_dim=cfg.MODEL.get('hpe_dim', 512)).cuda().eval()
    cfg.MODEL.name = old_mode
    lifter = artnet.lift_2d_to_3d

    cfg.TEST.batch_size = max(cfg.TEST.batch_size, 32)

    print('===== TRAIN (3dpw-train, có augmentation) =====')
    train_datasets, train_loader = get_dataloader(args, cfg.DATASET.train_list, is_train=True)
    evaluate(lifter, train_loader, train_datasets[0], 'train', args.max_batches)

    print('\n===== TEST =====')
    test_names = cfg.DATASET.test_list
    test_datasets, test_loaders = get_dataloader(args, test_names, is_train=False)
    for nm, dataset, loader in zip(test_names, test_datasets, test_loaders):
        evaluate(lifter, loader, dataset, f'test-{nm}', args.max_batches)


if __name__ == '__main__':
    main()
