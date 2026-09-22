import torch
import torch.nn as nn
from functools import partial
from core.config import cfg as cfg
from models import Multimodel
from models.backbones.resnet import ResNetBackbone
from models.DSTformer import DSTformer

import os
os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"

# MotionBERT-Lite hyperparameters (matching MotionBERT/configs/pretrain/MB_lite.yaml)
_MB_DIM_IN   = 3
_MB_DIM_OUT  = 3       # Output 3D joint coords directly (same as MotionBERT Pose3D task)
_MB_DIM_FEAT   = 256     # MB-Lite uses 256, full MotionBERT uses 512
_MB_DIM_REP  = 512
_MB_DEPTH    = 5
_MB_NUM_HEADS = 8
_MB_MLP_RATIO  = 4       # MB-Lite uses 4, full MotionBERT uses 2
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

        self.pose_mesh_coevo = Multimodel.get_model(num_joint, embed_dim*2)

    def forward(self, img, pose2d, is_train=True):
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
        # DSTformer expects input shape (B, T, J, C) with T up to maxlen=243
        # Since FuseKin is single-image (seqlen=1), we duplicate the frame to 243
        # to preserve all pretrained temporal embeddings.
        # (same trick used by MotionBERT authors for single-frame inference)
        mb_input = pose2d.repeat(1, _MB_MAXLEN, 1, 1)  # (B, 243, J, C)
        pose3d_seq = self.pose_lifter(mb_input)         # (B, 243, J, 3)  -- DSTformer forward()
        # Take the centre frame (matching MotionBERT data_stride=81, centre idx=121)
        centre = _MB_MAXLEN // 2                        # = 121
        pose3d = pose3d_seq[:, centre, :, :]            # (B, J, 3)

        # Reshape to match what pose_mesh_coevo expects: (B, seqlen, J, 3)
        pose3d = pose3d.unsqueeze(1).repeat(1, cfg.DATASET.seqlen, 1, 1)  # (B, seqlen, J, 3)

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

