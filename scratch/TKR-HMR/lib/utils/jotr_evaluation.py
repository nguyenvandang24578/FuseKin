import torch


def evaluate_3dpw_subset(model, dataset, loader, device='cuda'):
    model.eval()
    accumulated = {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': []}
    sample_index = 0

    with torch.no_grad():
        for inputs, targets, _ in loader:
            model_inputs = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in inputs.items()
            }
            outputs = model(model_inputs['img'], model_inputs['joints'], is_train=False)
            pred_mesh = outputs['smpl_mesh_cam'].detach().cpu().numpy()
            target_mesh = targets['smpl_mesh_cam'].detach().cpu().numpy()
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