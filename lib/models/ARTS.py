import os
from functools import partial

import torch
import torch.nn as nn

from core.config import cfg as cfg
from models import Multimodel
from models.backbones.resnet import ResNetBackbone
from models.DSTformer import DSTformer
from models.teacher import Teacher

os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"

# MotionBERT-Lite hyperparameters (matching MotionBERT/configs/pretrain/MB_lite.yaml)
_MB_DIM_IN = 3
_MB_DIM_OUT = 3        # Output 3D joint coords directly (same as MotionBERT Pose3D task)
_MB_DIM_FEAT = 256     # Lite MotionBERT (Full = 512)
_MB_DIM_REP = 512
_MB_DEPTH = 5
_MB_NUM_HEADS = 8
_MB_MLP_RATIO = 4      # Lite MotionBERT (Full = 2)
_MB_MAXLEN = 243       # MotionBERT was trained with 243-frame sequences
_MB_NUM_JOINTS = 17
_MB_ATT_FUSE = True


class ARTS(nn.Module):
    """Top-level model wrapper.

    There are two explicit operating modes, selected by ``cfg.MODEL.name``:

    1. ``teacher``
       Input:  image + GT 3D joints, where the second argument to ``forward``
       must be ``gt_pose3d`` with shape (B, 17, 3), in meters and root-relative.
       MotionBERT is NOT used in this path. The frozen ResNet backbone provides
       image feature maps and ``Teacher`` fuses image + GT 3D joints to regress
       SMPL pose/shape/mesh.

    2. ``student`` / ``ARTS``
       Input: image + 2D joints. The 2D joints are lifted to 3D by MotionBERT,
       then the legacy co-evolution mesh model is run. This is the old student
       path.
    """

    TEACHER_NAME = 'teacher'
    STUDENT_NAMES = {'student', 'ARTS'}

    def __init__(self, num_joint, embed_dim, depth):
        super(ARTS, self).__init__()

        self.num_joint = num_joint
        self.mode = cfg.MODEL.name

        self.backbone = ResNetBackbone(cfg.MODEL.resnet_type)
        self.pose_lifter = self._build_motionbert()
        self._load_motionbert_if_configured()

        if self.mode == self.TEACHER_NAME:
            self.teacher_model = Teacher(num_joint=num_joint, embed_dim=embed_dim)
            self._freeze_teacher_inputs()
            print('[ARTS] Mode=teacher: using image + GT 3D joints. Backbone and MotionBERT are frozen.')
        elif self.mode in self.STUDENT_NAMES:
            self.pose_mesh_coevo = Multimodel.get_model(num_joint, embed_dim * 2)
            print(f'[ARTS] Mode={self.mode}: using image + 2D joints -> MotionBERT -> student mesh model.')
        else:
            raise ValueError(
                f'Unsupported cfg.MODEL.name={self.mode!r}. '
                f'Expected {self.TEACHER_NAME!r} or one of {sorted(self.STUDENT_NAMES)}.'
            )

    def _build_motionbert(self):
        """Build the MotionBERT/DSTformer 2D-to-3D lifter used by the student path."""
        return DSTformer(
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

    def _load_motionbert_if_configured(self):
        """Load pretrained MotionBERT weights when cfg.MODEL.motionbert_pretrained is set."""
        if not cfg.MODEL.get('motionbert_pretrained', ''):
            return

        chk_path = cfg.MODEL.motionbert_pretrained
        print(f'[ARTS] Loading MotionBERT checkpoint: {chk_path}')
        checkpoint = torch.load(chk_path, map_location='cpu')
        state_dict = checkpoint['model_pos']

        # Strip DataParallel 'module.' prefix if present.
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v

        missing, unexpected = self.pose_lifter.load_state_dict(new_state_dict, strict=False)
        print(f'[ARTS] MotionBERT loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}')

    def _freeze_teacher_inputs(self):
        """Freeze modules that only provide Teacher inputs."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.pose_lifter.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.pose_lifter.eval()

    def _extract_image_features(self, img):
        """Return spatial feature maps and global image tokens from image input.

        Args:
            img: (B, 3, H, W) or (B, T, 3, H, W)

        Returns:
            feature_map:
                (B, 2048, h, w) for image input, or
                (B, T, 2048, h, w) for sequence input.
            img_feat:
                (B, 1, 2048) or (B, T, 2048), global-pooled ResNet features.
        """
        if img.dim() == 5:
            batch_size, sequence_length = img.shape[:2]
            backbone_input = img.reshape(batch_size * sequence_length, *img.shape[2:])
            feature_map, img_feat = self.backbone(backbone_input)
            feature_map = feature_map.reshape(batch_size, sequence_length, *feature_map.shape[1:])
            img_feat = img_feat.reshape(batch_size, sequence_length, -1)
            return feature_map, img_feat

        if img.dim() == 4:
            batch_size = img.shape[0]
            feature_map, img_feat = self.backbone(img)
            img_feat = img_feat.reshape(batch_size, 1, -1)
            return feature_map, img_feat

        raise ValueError(
            f'ARTS expects image input with shape (B,3,H,W) or (B,T,3,H,W), got {tuple(img.shape)}'
        )

    def _lift_2d_to_3d_with_motionbert(self, pose2d):
        """Student path: lift 2D joints to one-frame 3D joints using MotionBERT.

        Args:
            pose2d: (B, J, 2/3) or (B, T, J, 2/3)

        Returns:
            pose3d: (B, J, 3)
        """
        if pose2d.dim() == 3:
            pose2d = pose2d.unsqueeze(1)  # (B, 1, J, C)
        elif pose2d.dim() != 4:
            raise ValueError(f'2D pose input must be (B,J,C) or (B,T,J,C), got {tuple(pose2d.shape)}')

        xy = pose2d[..., :2]
        if pose2d.shape[-1] >= 3:
            conf = pose2d[..., 2:3]
        else:
            conf = torch.ones(*xy.shape[:-1], 1, device=xy.device, dtype=xy.dtype)
        pose2d_3ch = torch.cat([xy, conf], dim=-1)

        # Duplicate the available frame to MotionBERT's pretraining length.
        mb_input = pose2d_3ch[:, :1].repeat(1, _MB_MAXLEN, 1, 1)
        pose3d_seq = self.pose_lifter(mb_input)      # (B, 243, J, 3)
        centre = _MB_MAXLEN // 2
        return pose3d_seq[:, centre, :, :]           # (B, J, 3)

    def _forward_teacher(self, img, gt_pose3d, is_train=True):
        """Teacher mode: image + GT 3D joints -> SMPL output."""
        if gt_pose3d.dim() != 3 or gt_pose3d.shape[-1] != 3:
            raise ValueError(
                'Teacher mode expects GT 3D joints as second input with shape (B,17,3), '
                f'got {tuple(gt_pose3d.shape)}'
            )

        # Backbone is frozen in Teacher mode, so avoid graph construction.
        with torch.no_grad():
            feature_map, _ = self._extract_image_features(img)

        if feature_map.dim() == 5:
            B, T, C, H, W = feature_map.shape
            feat_map_2d = feature_map.reshape(B * T, C, H, W)
            pose3d_1d = gt_pose3d.unsqueeze(1).repeat(1, T, 1, 1).reshape(B * T, gt_pose3d.shape[1], 3)
        else:
            feat_map_2d = feature_map
            pose3d_1d = gt_pose3d

        teacher_out = self.teacher_model(
            joints=pose3d_1d,
            img_feats=feat_map_2d,
            is_train=is_train,
        )[-1]

        theta = teacher_out['theta'][:, -1]       # (B, 85) = [cam(3), pose(72), shape(10)]
        verts = teacher_out['verts'][:, -1]       # (B, 6890, 3)
        kp_3d = teacher_out['kp_3d']              # (B, J, 3) when seqlen=1

        return {
            'joint_img': kp_3d,
            'smpl_mesh_cam': verts,
            'smpl_pose': theta[:, 3:75],
            'smpl_shape': theta[:, 75:],
        }

    def _forward_student(self, img, pose2d, is_train=True):
        """Student/legacy ARTS mode: image + 2D joints -> MotionBERT -> mesh."""
        feature_map, img_feat = self._extract_image_features(img)
        pose3d = self._lift_2d_to_3d_with_motionbert(pose2d)  # (B, J, 3)

        # Legacy student co-evolution module expects temporal pose input.
        pose3d_seq = pose3d.unsqueeze(1).repeat(1, cfg.DATASET.seqlen, 1, 1)

        # NOTE: kept from the old student code. If MotionBERT is trained to output
        # meters instead of millimeters, this /1000 should be revisited.
        evo_pose, init_smpl_pose, init_smpl_shape, final_mesh, smploutput = self.pose_mesh_coevo(
            pose3d_seq / 1000,
            img_feat,
            pose2d,
            is_train=is_train,
        )

        pose3d_mid = pose3d_seq[:, cfg.DATASET.seqlen // 2]
        final_theta = smploutput[-1]['theta']
        if final_theta.dim() == 3:
            final_theta = final_theta[:, -1]

        return {
            'joint_img': pose3d_mid,
            'smpl_mesh_cam': final_mesh,
            'smpl_pose': final_theta[:, 3:75],
            'smpl_shape': final_theta[:, 75:],
        }

    def forward(self, img, joints, is_train=True):
        """Dispatch by cfg.MODEL.name.

        Teacher:
            ``joints`` = GT 3D joints, shape (B,17,3), meters, root-relative.

        Student / ARTS:
            ``joints`` = 2D joints, shape (B,J,2/3) or (B,T,J,2/3).
        """
        if self.mode == self.TEACHER_NAME:
            return self._forward_teacher(img, joints, is_train=is_train)

        if self.mode in self.STUDENT_NAMES:
            return self._forward_student(img, joints, is_train=is_train)

        raise RuntimeError(f'Unsupported ARTS mode: {self.mode!r}')


def get_model(num_joint, embed_dim, depth):
    model = ARTS(num_joint, embed_dim, depth)
    return model