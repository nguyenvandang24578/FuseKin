import torch
from tqdm import tqdm


def evaluate_3dpw_subset(model, dataset, loader, device='cuda'):
    model.eval()
    accumulated = {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': []}
    sample_index = 0

    with torch.no_grad():
        progress = tqdm(loader, desc='Testing 3DPW', leave=False)
        for inputs, targets, meta in progress:
            model_inputs = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            target_mesh_tensor = targets['smpl_mesh_cam'].to(device).float()

            # Teacher evaluation uses GT 3D joints. PW3D test provides GT
            # mesh, so derive H36M-17 GT joints and root-center them to match
            # Teacher training input. Student evaluation uses 2D joints.
            from core.config import cfg
            use_diffusion = getattr(cfg.MODEL, 'REFINER', 'hypergcn') == 'diffusion'

            # --- Prepare kp2d condition for diffusion ---
            kp2d_eval, kp_conf_eval = None, None
            if use_diffusion:
                eval_mode = getattr(cfg.DIFF, 'eval_kp_mode', 'detector')
                if eval_mode == 'gt':
                    # Mode 1: clean GT 2D
                    kp2d_eval = targets['orig_joint_img'].to(device)[:, :, :2].clone()
                    kp2d_eval[:, :, 0] = kp2d_eval[:, :, 0] / cfg.output_hm_shape[2] * 2 - 1
                    kp2d_eval[:, :, 1] = kp2d_eval[:, :, 1] / cfg.output_hm_shape[1] * 2 - 1
                    kp_conf_eval = meta['orig_joint_trunc'].to(device).squeeze(-1).float()
                elif eval_mode == 'noisy_gt':
                    # Mode 2: GT + simulated noise (fixed seed for reproducibility)
                    from models.smpl_hyperdiff import make_noisy_kp2d
                    kp2d_gt = targets['orig_joint_img'].to(device)[:, :, :2].clone()
                    kp2d_gt[:, :, 0] = kp2d_gt[:, :, 0] / cfg.output_hm_shape[2] * 2 - 1
                    kp2d_gt[:, :, 1] = kp2d_gt[:, :, 1] / cfg.output_hm_shape[1] * 2 - 1
                    kp_conf_gt = meta['orig_joint_trunc'].to(device).squeeze(-1).float()
                    torch.manual_seed(42)
                    kp2d_eval, kp_conf_eval, _ = make_noisy_kp2d(kp2d_gt, kp_conf_gt)
                else:
                    # Mode 3: detector detections (default)
                    det = model_inputs['joints']           # (B, 17, 2 or 3)
                    kp2d_eval = det[:, :, :2]              # already in [-1, 1]
                    if det.shape[-1] >= 3:
                        kp_conf_eval = (det[:, :, 2] > 0.3).float()
                    else:
                        mask = model_inputs.get('joints_mask')
                        kp_conf_eval = mask.squeeze(-1).float() if mask is not None \
                            else torch.ones(det.shape[0], det.shape[1], device=device)

            # --- Forward ---
            if cfg.MODEL.name in ('teacher', 'ARTS'):
                # Phải dùng orig_joint_img đã chuẩn hóa [-1, 1] (2.5D) 
                # giống hệt lúc train ở Teacher_Trainer!
                teacher_gt_joints = targets['orig_joint_img'].to(device).clone()
                teacher_gt_joints[..., 0] = teacher_gt_joints[..., 0] / cfg.output_hm_shape[2] * 2 - 1
                teacher_gt_joints[..., 1] = teacher_gt_joints[..., 1] / cfg.output_hm_shape[1] * 2 - 1
                teacher_gt_joints[..., 2] = teacher_gt_joints[..., 2] / cfg.output_hm_shape[0] * 2
                teacher_gt_joints = teacher_gt_joints - teacher_gt_joints[:, 0:1, :]
                outputs = model(
                    model_inputs['img'], teacher_gt_joints,
                    is_train=False, use_gt_3d=True,
                    kp2d=kp2d_eval, kp_conf=kp_conf_eval,
                )
            else:
                outputs = model(
                    model_inputs['img'], model_inputs['joints'],
                    is_train=False,
                    kp2d=kp2d_eval, kp_conf=kp_conf_eval,
                )

            pred_mesh = outputs['smpl_mesh_cam'].detach().cpu().numpy()
            target_mesh = target_mesh_tensor.detach().cpu().numpy()
            batch_outputs = [
                {
                    'smpl_mesh_cam': pred_mesh[index],
                    'smpl_mesh_cam_target': target_mesh[index],
                }
                for index in range(pred_mesh.shape[0])
            ]
            batch_result = dataset.evaluate(batch_outputs, sample_index)
            for key, values in batch_result.items():
                accumulated[key].extend(values)
            sample_index += len(batch_outputs)

    return {
        key: sum(values) / len(values) if values else float('nan')
        for key, values in accumulated.items()
    }