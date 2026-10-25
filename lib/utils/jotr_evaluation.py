import torch
from tqdm import tqdm


def evaluate_3dpw_subset(model, dataset, loader, device='cuda'):
    """Danh gia mesh tren 1 tap 3DPW bang dataset.evaluate: MPJPE / PA-MPJPE (14 khop) va MPVPE, mm.
    Teacher: dau vao = GT 3D H36M-17 hoi quy tu mesh GT (root-relative, met) + 2D Halpe cho PE.
    Student: dau vao = 2D Halpe -> DiffLifter (can cam_bbox)."""
    from core.config import cfg
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

            if cfg.MODEL.name == 'teacher':
                # Giong Teacher_Trainer: GT 3D met, root-relative (H36M-17 hoi quy tu mesh GT,
                # cung nguon voi orig_joint_cam luc train).
                h36m_regressor = torch.as_tensor(
                    dataset.h36m_joint_regressor,
                    device=device,
                    dtype=target_mesh_tensor.dtype,
                )
                teacher_gt_joints = torch.matmul(
                    h36m_regressor.unsqueeze(0).expand(target_mesh_tensor.shape[0], -1, -1),
                    target_mesh_tensor,
                )
                teacher_gt_joints = teacher_gt_joints - teacher_gt_joints[:, 0:1, :]
                outputs = model(model_inputs['img'], teacher_gt_joints, is_train=False,
                                pose_2d=model_inputs['joints'].float(),
                                joints_mask=model_inputs.get('joints_mask'))
            else:
                outputs = model(
                    model_inputs['img'], model_inputs['joints'],
                    is_train=False,
                    joints_mask=model_inputs.get('joints_mask'),
                    cam_bbox=model_inputs.get('cam_bbox'),   # DiffLifter
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
