import os
from functools import partial

import torch
import torch.nn as nn

from core.config import cfg
from models import Multimodel
from models.backbones.resnet import ResNetBackbone
from models.DSTformer import DSTformer
from models.teacher import Teacher

os.environ.setdefault("WANDB_MODE", "offline")

MOTIONBERT_CONFIG = dict(
    dim_in=3,
    dim_out=3,
    dim_feat=512,
    dim_rep=512,
    depth=5,
    num_heads=8,
    mlp_ratio=2,
    maxlen=243,
    num_joints=17,
    att_fuse=True,
)
NUM_FRAMES = MOTIONBERT_CONFIG["maxlen"]


class ARTS(nn.Module):
    def __init__(self, num_joint, embed_dim):
        super().__init__()
        self.mode = cfg.MODEL.name

        self.backbone = ResNetBackbone(cfg.MODEL.resnet_type)
        self.pose_lifter = DSTformer(
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **MOTIONBERT_CONFIG,
        )
        self.load_pose_lifter_weights()

        if self.mode in ("teacher", "student"):
            self.smpl_model = Teacher(num_joint=num_joint, embed_dim=embed_dim)
        elif self.mode == "ARTS":
            self.pose_mesh_coevo = Multimodel.get_model(num_joint, embed_dim * 2)
        else:
            raise ValueError(f"Mode không hợp lệ: {self.mode}. Chọn teacher, student hoặc ARTS.")

        self.freeze_backbone_and_pose_lifter()

    def load_pose_lifter_weights(self):
        path = cfg.MODEL.get("motionbert_pretrained", "")
        if not path:
            print("Cảnh báo: chưa có weight pretrained cho MotionBERT.")
            return

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = {}
        for name, weight in checkpoint["model_pos"].items():
            state_dict[name.replace("module.", "", 1)] = weight

        missing, unexpected = self.pose_lifter.load_state_dict(state_dict, strict=False)
        print(f"Load MotionBERT xong. Thiếu: {missing}. Thừa: {unexpected}.")

    def freeze_backbone_and_pose_lifter(self):
        for module in (self.backbone, self.pose_lifter):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.pose_lifter.eval()
        return self

    def get_image_features(self, image):
        feature_map, global_feature = self.backbone(image)
        global_feature = global_feature.reshape(image.shape[0], 1, -1)
        return feature_map, global_feature

    def lift_2d_to_3d(self, pose_2d):
        if pose_2d.dim() == 3:
            pose_2d = pose_2d.unsqueeze(1)

        first_frame = pose_2d[:, 0]
        xy = first_frame[..., :2]
        if first_frame.shape[-1] >= 3:
            confidence = first_frame[..., 2:3]
        else:
            confidence = torch.ones_like(xy[..., :1])
        pose_xyc = torch.cat([xy, confidence], dim=-1)

        repeated_frames = pose_xyc.unsqueeze(1).repeat(1, NUM_FRAMES, 1, 1)
        pose_3d_all_frames = self.pose_lifter(repeated_frames)
        return pose_3d_all_frames[:, NUM_FRAMES // 2]

    def format_smpl_output(self, smpl_output):
        theta = smpl_output["theta"][:, -1]
        return {
            "joint_img": smpl_output["kp_3d"],
            "smpl_mesh_cam": smpl_output["verts"][:, -1],
            "smpl_pose": theta[:, 3:75],
            "smpl_shape": theta[:, 75:],
        }

    def forward_teacher(self, image, gt_pose_3d, is_train):
        with torch.no_grad():
            feature_map, _ = self.get_image_features(image)

        smpl_output = self.smpl_model(
            joints=gt_pose_3d,
            img_feats=feature_map,
            is_train=is_train,
        )[-1]
        return self.format_smpl_output(smpl_output)

    def forward_student(self, image, pose_2d, is_train):
        with torch.no_grad():
            feature_map, _ = self.get_image_features(image)
            pose_3d = self.lift_2d_to_3d(pose_2d) / 1000      # mm -> m
            pose_3d = pose_3d - pose_3d[:, 0:1, :]            # root-relative như đầu vào teacher

        smpl_output = self.smpl_model(
            joints=pose_3d,
            img_feats=feature_map,
            is_train=is_train,
        )[-1]
        return self.format_smpl_output(smpl_output)

    def forward_arts(self, image, pose_2d, is_train):
        with torch.no_grad():
            ft_map, global_feature = self.get_image_features(image)
            pose_3d = self.lift_2d_to_3d(pose_2d)

        seqlen = cfg.DATASET.seqlen

        output = self.pose_mesh_coevo(
            pose_3d / 1000,
            ft_map,
            is_train=is_train,
        )
        # MotionBERT outputs 3D joints in millimeters; convert to meters so
        # joint_img shares the unit of the GT joints used in the training loss.
        output["joint_img"] = pose_3d/1000
        return output

    def forward(self, image, joints, is_train=True):
        if self.mode == "teacher":
            return self.forward_teacher(image, joints, is_train)
        if self.mode == "student":
            return self.forward_student(image, joints, is_train)
        return self.forward_arts(image, joints, is_train)


def get_model(num_joint, embed_dim, depth=None):
    return ARTS(num_joint, embed_dim)