import torch
from tqdm import tqdm


def evaluate_3dpw_subset(model, dataset, loader, device='cuda', with_lifter=False):
    """Danh gia mesh tren 1 tap 3DPW bang dataset.evaluate: MPJPE / PA-MPJPE (14 khop) va MPVPE, mm.
    Teacher: dau vao = GT 3D H36M-17 hoi quy tu mesh GT (root-relative, met) + 2D Halpe cho PE.
    Student: dau vao = 2D Halpe -> DiffLifter (can cam_bbox)."""
    from core.config import cfg
    model.eval()
    accumulated = {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': []}
    # with_lifter (2026-11-05, chan doan): do pose 3D DAU VAO Student (lifted_joints_3d) bang DUNG thuoc do 14 khop
    # cua dataset.evaluate (H36M regressor x mesh GT, root Pelvis, h36m_eval_joint) -> so truc tiep voi mesh Student.
    if with_lifter:
        from utils.img_lifter_tools import pa_errors
        accumulated.update({'lift_mpjpe': [], 'lift_pa_mpjpe': []})
        eval_idx = torch.as_tensor(dataset.h36m_eval_joint, device=device)
        root_idx = int(dataset.h36m_root_joint_idx)
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
                                joints_mask=model_inputs.get('joints_mask'),
                                cam_bbox=model_inputs.get('cam_bbox'))
            else:
                outputs = model(
                    model_inputs['img'], model_inputs['joints'],
                    is_train=False,
                    joints_mask=model_inputs.get('joints_mask'),
                    cam_bbox=model_inputs.get('cam_bbox'),   # DiffLifter
                )

            if with_lifter and 'lifted_joints_3d' in outputs:
                reg = torch.as_tensor(dataset.h36m_joint_regressor, device=device, dtype=target_mesh_tensor.dtype)
                gt17 = torch.einsum('jk,bkc->bjc', reg, target_mesh_tensor)
                gt14 = (gt17 - gt17[:, root_idx:root_idx + 1])[:, eval_idx]

                def _acc14(key, j17):
                    # j17 (B,17,3) met -> MPJPE / PA 14 khop (cung thuoc do dataset.evaluate)
                    j14 = (j17 - j17[:, root_idx:root_idx + 1])[:, eval_idx]
                    accumulated.setdefault(key + '_mpjpe', []).extend(
                        (torch.norm(j14 - gt14, dim=-1).mean(-1) * 1000).cpu().tolist())
                    accumulated.setdefault(key + '_pa_mpjpe', []).extend(pa_errors(j14, gt14).cpu().tolist())

                _acc14('lift', outputs['lifted_joints_3d'].float())                       # tang 0: dau vao
                if 'privileged_3d' in outputs:                                            # tang 1: token pose
                    _acc14('token', outputs['privileged_3d'].float())
                for it, h in enumerate(outputs.get('refine_history', []) or []):          # tang 2-3: mesh truoc [D]
                    _acc14(f'mesh_it{it}', torch.einsum('jk,bkc->bjc', reg, h['smpl_mesh_cam'].float()))
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
