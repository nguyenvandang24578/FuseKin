import torch
import torch.nn as nn
from core.config import cfg as cfg
from models import Multimodel, PoseEstimation
from models.backbones.resnet import ResNetBackbone

import os
os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"


class ARTS(nn.Module):
    def __init__(self, num_joint, embed_dim, depth):
        super(ARTS, self).__init__()

        self.num_joint = num_joint
        self.backbone = ResNetBackbone(cfg.MODEL.resnet_type)
        self.pose_lifter = PoseEstimation.get_model(num_joint, embed_dim, depth, pretrained=cfg.MODEL.posenet_pretrained)
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
            pose2d = pose2d.unsqueeze(1)
        pose3d = self.pose_lifter(pose2d, img_feat)
        pose3d = pose3d.reshape(-1, cfg.DATASET.seqlen, self.num_joint, 3)
        
        evo_pose, init_smpl_pose, init_smpl_shape, final_mesh, smploutput = self.pose_mesh_coevo(pose3d / 1000, img_feat, pose2d, is_train=is_train)
        
        pose3d = pose3d[:, cfg.DATASET.seqlen // 2]
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
