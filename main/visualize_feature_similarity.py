import argparse
import copy
import csv
import os
import sys

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
sys.path.append('./lib')
sys.path.append('./')

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import __init_path  # noqa: F401
from core.base import load_model_weights, prepare_network
from core.config import cfg, update_config


JOINT_NAMES = (
    'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle',
    'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose',
    'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
    'R_Shoulder', 'R_Elbow', 'R_Wrist',
)


def load_models(args):
    update_config(args.config)
    cfg.TRAIN.wandb = False

    print('Loading Student checkpoint...')
    val_loaders, val_datasets, student, _, _, _, _, _ = prepare_network(
        args, load_dir=args.student_checkpoint, is_train=False
    )

    teacher = copy.deepcopy(student)
    teacher.mode = 'teacher'
    print('Loading Teacher checkpoint...')
    load_model_weights(teacher, args.teacher_checkpoint)

    student = torch.nn.DataParallel(student).cuda().eval()
    teacher = torch.nn.DataParallel(teacher).cuda().eval()
    return val_loaders, val_datasets, student, teacher


def get_teacher_pose(target_mesh, dataset):
    regressor = torch.as_tensor(
        dataset.h36m_joint_regressor,
        device=target_mesh.device,
        dtype=target_mesh.dtype,
    )
    joints = torch.matmul(regressor.unsqueeze(0), target_mesh)
    return joints - joints[:, 0:1, :]


def pca_2d(features):
    features = features - features.mean(axis=0, keepdims=True)
    if features.shape[0] < 2:
        return np.zeros((features.shape[0], 2), dtype=np.float32)
    _, _, vh = np.linalg.svd(features, full_matrices=False)
    components = vh[:2].T
    projected = features @ components
    if projected.shape[1] == 1:
        projected = np.pad(projected, ((0, 0), (0, 1)))
    return projected.astype(np.float32)


def save_csv(path, confidence, joint_cosine, img_cosine, relation_error, attention_error):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'sample', 'mean_confidence', 'joint_cosine',
            'image_cosine', 'relation_error', 'attention_error',
        ])
        for index in range(len(confidence)):
            writer.writerow([
                index,
                float(confidence[index]),
                float(joint_cosine[index]),
                float(img_cosine[index]),
                float(relation_error[index]),
                float(attention_error[index]),
            ])


def save_visualizations(output_dir, confidence, joint_cosine, img_cosine,
                        relation_error, attention_error, student_joint, teacher_joint):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            'Visualization requires matplotlib. Install it with: '
            'python -m pip install matplotlib'
        ) from error

    os.makedirs(output_dir, exist_ok=True)
    sample_count = len(confidence)
    order = np.argsort(confidence)

    fig, axis = plt.subplots(figsize=(14, 6))
    image = axis.imshow(
        joint_cosine[order].T,
        aspect='auto',
        vmin=0.0,
        vmax=1.0,
        cmap='viridis',
    )
    axis.set_title('Student-Teacher cosine similarity per joint')
    axis.set_xlabel('Samples sorted by mean 2D confidence')
    axis.set_ylabel('Joint')
    axis.set_yticks(np.arange(len(JOINT_NAMES)))
    axis.set_yticklabels(JOINT_NAMES, fontsize=8)
    fig.colorbar(image, ax=axis, label='Cosine similarity')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'joint_cosine_heatmap.png'), dpi=180)
    plt.close(fig)

    bins = [0.0, 0.4, 0.8, 1.01]
    labels = ['low', 'medium', 'high']
    grouped = [
        relation_error[(confidence >= bins[i]) & (confidence < bins[i + 1])]
        for i in range(3)
    ]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(
        grouped,
        labels=[f'{label}\n(n={len(values)})' for label, values in zip(labels, grouped)],
        showfliers=False,
    )
    axis.set_xlabel('Mean 2D confidence group')
    axis.set_ylabel('Relational Distillation Error (MSE)')
    axis.set_title('Relation Error by confidence')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'relation_error_by_confidence.png'), dpi=180)
    plt.close(fig)

    grouped_attn = [
        attention_error[(confidence >= bins[i]) & (confidence < bins[i + 1])]
        for i in range(3)
    ]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(
        grouped_attn,
        labels=[f'{label}\n(n={len(values)})' for label, values in zip(labels, grouped_attn)],
        showfliers=False,
    )
    axis.set_xlabel('Mean 2D confidence group')
    axis.set_ylabel('Attention Distillation Error (KL Div)')
    axis.set_title('Attention Error by confidence')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'attention_error_by_confidence.png'), dpi=180)
    plt.close(fig)

    teacher_projection = pca_2d(teacher_joint)
    student_projection = pca_2d(student_joint)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    scatter_a = axes[0].scatter(
        teacher_projection[:, 0], teacher_projection[:, 1],
        c=confidence, cmap='plasma', s=18,
    )
    axes[0].set_title('Teacher joint feature PCA')
    scatter_b = axes[1].scatter(
        student_projection[:, 0], student_projection[:, 1],
        c=confidence, cmap='plasma', s=18,
    )
    axes[1].set_title('Student joint feature PCA')
    for axis in axes:
        axis.set_xlabel('PC1')
        axis.set_ylabel('PC2')
    fig.colorbar(scatter_b, ax=axes, label='Mean 2D confidence')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'joint_feature_pca.png'), dpi=180)
    plt.close(fig)

    mean_joint_cosine = joint_cosine.mean(axis=0)
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.bar(np.arange(len(JOINT_NAMES)), mean_joint_cosine)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(np.arange(len(JOINT_NAMES)))
    axis.set_xticklabels(JOINT_NAMES, rotation=45, ha='right')
    axis.set_ylabel('Mean cosine similarity')
    axis.set_title(f'Mean joint feature similarity ({sample_count} samples)')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'mean_joint_cosine.png'), dpi=180)
    plt.close(fig)


def main(args):
    val_loaders, datasets, student, teacher = load_models(args)
    loader = val_loaders[0]
    dataset = datasets[0]

    confidence_values = []
    joint_cosine_values = []
    image_cosine_values = []
    relation_error_values = []
    attention_error_values = []
    student_joint_values = []
    teacher_joint_values = []

    with torch.no_grad():
        for step, (inputs, targets, _) in enumerate(tqdm(loader, desc='Extracting features')):
            if args.max_batches > 0 and step >= args.max_batches:
                break

            image = inputs['img'].cuda().float()
            pose2d = inputs['joints'].cuda().float()
            if pose2d.shape[-1] < 3:
                raise ValueError('Expected pose2d input with (x, y, confidence).')

            target_mesh = targets['smpl_mesh_cam'].cuda().float()
            teacher_pose = get_teacher_pose(target_mesh, dataset)

            student_out = student(image, pose2d, is_train=False)
            teacher_out = teacher(image, teacher_pose, is_train=False)

            student_joint = student_out['feat'].flatten(1)
            teacher_joint = teacher_out['feat'].flatten(1)
            student_joint_pooled = student_out['feat'].mean(dim=1)
            teacher_joint_pooled = teacher_out['feat'].mean(dim=1)

            joint_cosine = F.cosine_similarity(
                student_out['feat'], teacher_out['feat'], dim=-1
            )
            img_cos_per_token = F.cosine_similarity(
                student_out['feat_img'], teacher_out['feat_img'], dim=-1
            )
            image_cosine = img_cos_per_token.mean(dim=-1)

            student_joint_rel = F.normalize(student_out['feat'], dim=-1) @ F.normalize(student_out['feat'], dim=-1).transpose(-1, -2)
            teacher_joint_rel = F.normalize(teacher_out['feat'], dim=-1) @ F.normalize(teacher_out['feat'], dim=-1).transpose(-1, -2)
            relation_error = (student_joint_rel - teacher_joint_rel).pow(2).flatten(1).mean(1)

            confidence = pose2d[:, :, 2].mean(dim=1)
            attention_error = torch.zeros_like(confidence)
            for student_layer, teacher_layer in zip(
                    student_out.get('feat_layers', []), teacher_out.get('feat_layers', [])):
                if 'attn_joint_to_img' in student_layer and 'attn_img_to_joint' in student_layer:
                    attention_error += (
                        F.kl_div((student_layer['attn_joint_to_img'] + 1e-8).log(), teacher_layer['attn_joint_to_img'], reduction='none').sum(-1).mean((-1, -2))
                        + F.kl_div((student_layer['attn_img_to_joint'] + 1e-8).log(), teacher_layer['attn_img_to_joint'], reduction='none').sum(-1).mean((-1, -2))
                    )

            confidence_values.append(confidence.cpu().numpy())
            joint_cosine_values.append(joint_cosine.cpu().numpy())
            image_cosine_values.append(image_cosine.cpu().numpy())
            relation_error_values.append(relation_error.cpu().numpy())
            attention_error_values.append(attention_error.cpu().numpy())
            student_joint_values.append(student_joint_pooled.cpu().numpy())
            teacher_joint_values.append(teacher_joint_pooled.cpu().numpy())

    confidence = np.concatenate(confidence_values)
    joint_cosine = np.concatenate(joint_cosine_values)
    image_cosine = np.concatenate(image_cosine_values)
    relation_error = np.concatenate(relation_error_values)
    attention_error = np.concatenate(attention_error_values)
    student_joint = np.concatenate(student_joint_values)
    teacher_joint = np.concatenate(teacher_joint_values)

    save_csv(
        os.path.join(args.output_dir, 'feature_similarity.csv'),
        confidence, joint_cosine.mean(axis=1), image_cosine, relation_error, attention_error,
    )
    save_visualizations(
        args.output_dir, confidence, joint_cosine, image_cosine,
        relation_error, attention_error, student_joint, teacher_joint,
    )

    print(f'Processed {len(confidence)} samples.')
    print(f'Mean joint cosine: {joint_cosine.mean():.4f}')
    print(f'Mean image cosine: {image_cosine.mean():.4f}')
    print(f'Mean relation error: {relation_error.mean():.4f}')
    print(f'Mean attention error: {attention_error.mean():.4f}')
    print(f'Outputs saved to: {args.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--student_checkpoint', required=True)
    parser.add_argument('--teacher_checkpoint', required=True)
    parser.add_argument('--config', default='config/train_student.yml')
    parser.add_argument('--output_dir', default='output/feature_visualization')
    parser.add_argument('--max_batches', type=int, default=-1)
    parser.add_argument('--resume_training', action='store_true')
    main(parser.parse_args())
