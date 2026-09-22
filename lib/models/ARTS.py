import torch
import torch.nn as nn
from functools import partial
from core.config import cfg as cfg
from models import Multimodel
from models.backbones.resnet import ResNetBackbone
from models.DSTformer import DSTformer
from models.teacher import Teacher

import os
os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"

# MotionBERT-Lite hyperparameters (matching MotionBERT/configs/pretrain/MB_lite.yaml)
_MB_DIM_IN   = 3
_MB_DIM_OUT  = 3       # Output 3D joint coords directly (same as MotionBERT Pose3D task)
_MB_DIM_FEAT   = 256     # Lite MotionBERT (Full = 512)
_MB_DIM_REP  = 512
_MB_DEPTH    = 5
_MB_NUM_HEADS = 8
_MB_MLP_RATIO  = 4       # Lite MotionBERT (Full = 2)
_MB_MAXLEN   = 243     # MotionBERT was trained with 243-frame sequences
_MB_NUM_JOINTS = 17
_MB_ATT_FUSE = True


class ARTS(nn.Module):
    def __init__(self, num_joint, embed_dim, depth):
        super(ARTS, self).__init__()

        self.num_joint = num_joint
        self.backbone = ResNetBackbone(cfg.MODEL.resnet_type)

        # Replace PoseEstimation with MotionBERT DSTformer
        # Constructed exactly as in MotionBERT/lib/utils/learning.py :: load_backbone()
        self.pose_lifter = DSTformer(
            dim_in=_MB_DIM_IN,
            dim_out=_MB_DIM_OUT,
            dim_feat=_MB_DIM_FEAT,
            dim_rep=_MB_DIM_REP,
            depth=_MB_DEPTH,
            num_heads=_MB_NUM_HEADS,
            mlp_ratio=_MB_MLP_RATIO,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            maxlen=_MB_MAXLEN,
            num_joints=_MB_NUM_JOINTS,
            att_fuse=_MB_ATT_FUSE,
        )

        # Load MotionBERT pretrained checkpoint if path is provided
        if cfg.MODEL.get('motionbert_pretrained', ''):
            chk_path = cfg.MODEL.motionbert_pretrained
            print(f'[ARTS] Loading MotionBERT checkpoint: {chk_path}')
            checkpoint = torch.load(chk_path, map_location='cpu')
            # MotionBERT saves weights under key 'model_pos'
            # (ref: MotionBERT/train.py :: save_checkpoint)
            state_dict = checkpoint['model_pos']
            # Strip DataParallel 'module.' prefix if present
            # (ref: MotionBERT/lib/utils/learning.py :: load_pretrained_weights)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dict[k] = v
            missing, unexpected = self.pose_lifter.load_state_dict(new_state_dict, strict=False)
            print(f'[ARTS] MotionBERT loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}')

        if cfg.MODEL.name == 'teacher':
            self.teacher_model = Teacher(num_joint=num_joint, embed_dim=embed_dim)
            # Freeze backbone + MotionBERT: chỉ train Teacher fusion
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.pose_lifter.parameters():
                param.requires_grad = False
            self.backbone.eval()
            self.pose_lifter.eval()
            print('[ARTS] Frozen backbone + MotionBERT for Teacher training')
        else:
            self.pose_mesh_coevo = Multimodel.get_model(num_joint, embed_dim*2)

    def forward(self, img, pose2d, is_train=True):
        # Nếu train Teacher, không cần gradient cho backbone + MotionBERT
        _no_grad = torch.no_grad() if (cfg.MODEL.name == 'teacher' and is_train) else torch.enable_grad()
        
        with _no_grad:
            legacy_feature_input = img.dim() not in (4, 5)
            if legacy_feature_input:
                img_feat = pose2d
                pose2d = img
                feature_map = None
                if img_feat.dim() == 2:
                    img_feat = img_feat.unsqueeze(1)
                batch_size = pose2d.shape[0]
                sequence_length = pose2d.shape[1] if pose2d.dim() == 4 else 1
            elif img.dim() == 5:
                batch_size, sequence_length = img.shape[:2]
                backbone_input = img.reshape(batch_size * sequence_length, *img.shape[2:])
                feature_map, img_feat = self.backbone(backbone_input)
                feature_map = feature_map.reshape(batch_size, sequence_length, *feature_map.shape[1:])
                img_feat = img_feat.reshape(batch_size, sequence_length, -1)
            else:
                batch_size, sequence_length = img.shape[0], 1
                backbone_input = img
                feature_map, img_feat = self.backbone(backbone_input)
                img_feat = img_feat.reshape(batch_size, 1, -1)

            if pose2d.dim() == 3:
                pose2d = pose2d.unsqueeze(1)  # (B, 1, J, C)

            # --- MotionBERT Lifting (2D -> 3D) ---
            # DSTformer expects exactly 3 channels: (x, y, confidence)
            # Always take first 2 channels then append confidence=1
            xy = pose2d[..., :2]                                    # (B, 1, J, 2)
            conf = torch.ones(*xy.shape[:-1], 1, device=xy.device)  # (B, 1, J, 1)
            pose2d_3ch = torch.cat([xy, conf], dim=-1)              # (B, 1, J, 3)
            # Duplicate single frame to maxlen=243 to preserve pretrained temporal embeddings
            mb_input = pose2d_3ch.repeat(1, _MB_MAXLEN, 1, 1)      # (B, 243, J, 3)
            pose3d_seq = self.pose_lifter(mb_input)         # (B, 243, J, 3)  -- DSTformer forward()
            # Take the centre frame (matching MotionBERT data_stride=81, centre idx=121)
            centre = _MB_MAXLEN // 2                        # = 121
            pose3d = pose3d_seq[:, centre, :, :]            # (B, J, 3)

        # Reshape to match what pose_mesh_coevo expects: (B, seqlen, J, 3)
        pose3d = pose3d.unsqueeze(1).repeat(1, cfg.DATASET.seqlen, 1, 1)  # (B, seqlen, J, 3)

        if cfg.MODEL.name == 'teacher':
            # teacher_model expect: joints=(B, 17, 3), img_feats=(B, 2048, H, W)
            if feature_map.dim() == 5:
                B, T, C, H, W = feature_map.shape
                feat_map_2d = feature_map.reshape(B * T, C, H, W)
                pose3d_1d = pose3d.reshape(B * T, pose3d.shape[2], 3)
            else:
                feat_map_2d = feature_map                 # (B, 2048, H, W)
                pose3d_1d = pose3d[:, 0]                  # (B, J, 3) - lấy frame đầu vì seqlen=1
            
            teacher_out_list = self.teacher_model(joints=pose3d_1d, img_feats=feat_map_2d, is_train=is_train)
            # RegressorSpin trả về list[dict], lấy phần tử cuối cùng
            teacher_out = teacher_out_list[-1]
            # Các tensor có shape (B, seqlen=1, ...), squeeze chiều seqlen ra
            theta = teacher_out['theta'][:, -1]          # (B, 85) = [cam(3), pose(72), shape(10)]
            verts = teacher_out['verts'][:, -1]           # (B, 6890, 3)
            kp_3d = teacher_out['kp_3d']                  # (B, J, 3) - đã không có seqlen
            output = {
                'joint_img': kp_3d,
                'smpl_mesh_cam': verts,
                'smpl_pose': theta[:, 3:75],
                'smpl_shape': theta[:, 75:],
            }
            return output

        evo_pose, init_smpl_pose, init_smpl_shape, final_mesh, smploutput = self.pose_mesh_coevo(pose3d / 1000, img_feat, pose2d, is_train=is_train)

        pose3d = pose3d[:, cfg.DATASET.seqlen // 2]   # (B, J, 3)
        final_theta = smploutput[-1]['theta']
        if final_theta.dim() == 3:
            final_theta = final_theta[:, -1]

        output = {
            'joint_img': pose3d,
            'smpl_mesh_cam': final_mesh,
            'smpl_pose': final_theta[:, 3:75],
            'smpl_shape': final_theta[:, 75:],
        }
        return output


def get_model(num_joint, embed_dim, depth):
    model = ARTS(num_joint, embed_dim, depth)
    return model


