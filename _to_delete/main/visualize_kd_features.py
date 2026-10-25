import os, sys
sys.path.append('./lib')
sys.path.append('./')
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from core.config import cfg, update_config
from core.base import Student_Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/train_init_mesh.yaml', help='experiment configure file name')
    parser.add_argument('--debug', action='store_true', default=True, help='reduce dataset items')
    parser.add_argument('--gpu', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2')
    args, _ = parser.parse_known_args()
    
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        
    update_config(args.cfg)
    cfg.TRAIN.batch_size = 1 # Force batch size 1 for visualization
    
    print("Initializing Student_Trainer (Loading Student and Teacher models)...")
    # This will load both models based on the config file
    trainer = Student_Trainer(args, load_dir='')
    
    # Lấy đường dẫn checkpoint của Student từ config và nạp trọng số
    student_ckpt = "./experiment/student/checkpoint/best.pth.tar"
    if student_ckpt and os.path.exists(student_ckpt):
        print(f"Loading Student weights from {student_ckpt}...")
        from core.base import load_model_weights
        load_model_weights(trainer.model, student_ckpt)
    else:
        print(f"WARNING: Cannot find Student checkpoint at {student_ckpt}")
        
    trainer.model.eval()
    trainer.teacher.eval()
    
    print("Loading one batch...")
    for i, (inputs, targets, meta) in enumerate(trainer.batch_generator):
        print(f"Batch {i} loaded!")
        
        # Prepare inputs exactly as in base.py
        input_image = inputs['img'].cuda().float()
        input_pose2d = inputs['joints'].cuda().float()
        joints_mask = inputs['joints_mask'].cuda().float()
        gt_orig_joint_cam = targets['orig_joint_cam'].cuda()
        gt_pose_input = gt_orig_joint_cam - gt_orig_joint_cam[:, 0:1, :]

        with torch.no_grad():
            # Student forward
            model_output = trainer.model(
                input_image, input_pose2d, is_train=False,
                gt_joints_3d=gt_pose_input, joints_mask=joints_mask
            )
            s_feat_joint = model_output['feat'] # Shape: (B, J, D)

            # Teacher forward
            t_out = trainer.teacher(input_image, gt_pose_input, is_train=False)
            t_feat_joint = t_out['feat'].detach() # Shape: (B, J, D)

        # Take the first sample in batch
        s_feat = s_feat_joint[0] # (J, D)
        t_feat = t_feat_joint[0] # (J, D)

        # Normalize features to compute cosine similarity via dot product
        s_feat_n = F.normalize(s_feat, dim=-1) # (J, D)
        t_feat_n = F.normalize(t_feat, dim=-1) # (J, D)
        
        # Compute Cosine Similarity Matrix: S * T^T
        # Shape: (17, 17) where entry (i, j) is sim between student joint i and teacher joint j
        similarity_matrix = torch.matmul(s_feat_n, t_feat_n.transpose(0, 1)).cpu().numpy()

        joint_names = ['Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist']
        
        # ==========================================
        # Plot 1: Heatmap of Cosine Similarity
        # ==========================================
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, xticklabels=joint_names, yticklabels=joint_names, cmap="viridis", annot=False)
        plt.title('Cosine Similarity: Student vs Teacher Joint Features')
        plt.xlabel('Teacher Joints')
        plt.ylabel('Student Joints')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig('visualize_kd_heatmap.png')
        print("=> Saved Cosine Similarity Heatmap to visualize_kd_heatmap.png")
        
        # ==========================================
        # Plot 2: PCA Scatter Plot
        # ==========================================
        try:
            from sklearn.decomposition import PCA
            
            # Combine S and T features for PCA: shape (34, D)
            all_feats = torch.cat([s_feat_n, t_feat_n], dim=0).cpu().numpy()
            pca = PCA(n_components=2)
            feats_2d = pca.fit_transform(all_feats)
            
            s_feats_2d = feats_2d[:17]
            t_feats_2d = feats_2d[17:]
            
            plt.figure(figsize=(10, 10))
            plt.scatter(t_feats_2d[:, 0], t_feats_2d[:, 1], c='blue', marker='o', s=150, label='Teacher', alpha=0.6)
            plt.scatter(s_feats_2d[:, 0], s_feats_2d[:, 1], c='red', marker='x', s=150, label='Student', alpha=0.6)
            
            # Draw lines connecting corresponding joints
            for j in range(17):
                plt.plot([t_feats_2d[j, 0], s_feats_2d[j, 0]], [t_feats_2d[j, 1], s_feats_2d[j, 1]], 'gray', linestyle='--', alpha=0.5)
                # Annotate joint names
                plt.text(t_feats_2d[j, 0], t_feats_2d[j, 1], joint_names[j], fontsize=9, color='blue')
                plt.text(s_feats_2d[j, 0], s_feats_2d[j, 1], joint_names[j], fontsize=9, color='red')
                
            plt.title('PCA 2D Projection of Joint Features (Student vs Teacher)')
            plt.legend()
            plt.tight_layout()
            plt.savefig('visualize_kd_pca.png')
            print("=> Saved PCA plot to visualize_kd_pca.png")
        except ImportError:
            print("sklearn not installed, skipping PCA visualization.")

        break # Only process one image for visualization

if __name__ == '__main__':
    main()
