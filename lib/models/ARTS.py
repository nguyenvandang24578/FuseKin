import os
from functools import partial

import torch
import torch.nn as nn

from core.config import cfg
from models import Multimodel
from models.backbones.resnet import ResNetBackbone
from models.DSTformer import DSTformer
from models.teacher_student import Teacher, Student   # model end-to-end moi (khong con SPIN)

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
        # Nap trong so pretrain cho backbone ('spin' | 'imagenet' | '' = ngau nhien nhu cu).
        # Khi nap checkpoint da train, load_model_weights se ghi de backbone (va chuan hoa) bang ban trong checkpoint.
        bb_src = cfg.MODEL.get('backbone_pretrained', '')
        if bb_src:
            self.backbone.load_pretrained(
                bb_src,
                spin_checkpoint=cfg.MODEL.get('spin_checkpoint', 'data_final/base_data/spin_model_checkpoint.pth.tar'),
            )
        else:
            print('[CANH BAO] Backbone ResNet KHONG nap pretrain (cfg.MODEL.backbone_pretrained rong) -> dac trung anh ngau nhien.')

        # Lấy hpe_dim từ config để hỗ trợ MotionBERT-Lite (256) hoặc Full (512)
        mb_dim = cfg.MODEL.get('hpe_dim', 512)
        mb_mlp_ratio = cfg.MODEL.get('mlp_ratio', 4 if mb_dim == 256 else 2)

        # Override dim_feat và mlp_ratio theo config
        mb_config = MOTIONBERT_CONFIG.copy()
        mb_config['dim_feat'] = mb_dim
        mb_config['dim_rep'] = 512  # dim_rep luôn là 512 kể cả bản Lite
        mb_config['mlp_ratio'] = mb_mlp_ratio

        # Teacher chi can MotionBERT khi bat (A): tron GT voi joint lift (cfg.MODEL.teacher_lift_alpha_max > 0)
        self.teacher_lift = (self.mode == "teacher" and cfg.MODEL.get('teacher_lift_alpha_max', 0.0) > 0)
        if self.mode != "teacher" or self.teacher_lift:
            self.pose_lifter = DSTformer(
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                **mb_config,
            )
            self.load_pose_lifter_weights()
        else:
            self.pose_lifter = None

        if self.mode == "teacher":
            print("Đang gọi mô hình teacher")
            self.smpl_model = Teacher(num_joint=num_joint, embed_dim=embed_dim, depth=3,
                                      use_img_probe=bool(cfg.MODEL.get('img_probe', False)))
        elif self.mode == "student":
            print("Đang gọi mô hình student")
            self.smpl_model = Student(num_joint=num_joint, embed_dim=embed_dim, depth=3)
            self.privileged_joint_head = nn.Linear(embed_dim, 3)
        elif self.mode == "ARTS":
            self.pose_mesh_coevo = Multimodel.get_model(num_joint, embed_dim)
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
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if self.pose_lifter is not None:
            self.pose_lifter.eval()
        return self

    def get_image_features(self, image):
        feature_map, global_feature = self.backbone(image)
        global_feature = global_feature.reshape(image.shape[0], 1, -1)
        return feature_map, global_feature

    def lift_2d_to_3d(self, pose_2d, joints_mask=None):
        if pose_2d.dim() == 3:
            pose_2d = pose_2d.unsqueeze(1)

        first_frame = pose_2d[:, 0]
        xy = first_frame[..., :2]

        # CRITICAL: MotionBERT goc (MB_ft_h36m.yaml) KHONG tru root khoi input 2D, chi tru root
        # o 3D target/loss (rootrel: True). FuseKin truoc day tru ca input 2D -> mat tin hieu vi
        # tri tuyet doi trong khung hinh. Dieu khien boi cfg.MODEL.motionbert_2d_rootrel de so sanh.
        if cfg.MODEL.get('motionbert_2d_rootrel', True):
            xy = xy - xy[:, 0:1, :]

        # MotionBERT's 3rd input channel was finetuned on the real per-joint
        # validity mask (see main/finetune_motionbert.py), not a constant.
        if joints_mask is not None:
            confidence = joints_mask.to(dtype=xy.dtype, device=xy.device)
        elif first_frame.shape[-1] >= 3:
            confidence = first_frame[..., 2:3]
        else:
            confidence = torch.ones_like(xy[..., :1])
        pose_xyc = torch.cat([xy, confidence], dim=-1)

        # Truyền trực tiếp 1 frame tĩnh (F=1) vào thay vì lặp lại NUM_FRAMES lần
        single_frame = pose_xyc.unsqueeze(1)  # (B, 1, 17, 3)
        pose_3d = self.pose_lifter(single_frame)
        # MotionBERT pretrained weights natively output METERS.
        # Teacher expects METERS.
        return pose_3d[:, 0]  # (B, 17, 3) mét

    def forward_teacher(self, image, gt_pose_3d, is_train, pose_2d=None, joints_mask=None, lift_alpha=None):
        with torch.no_grad():
            feature_map, _ = self.get_image_features(image)

            # gt_pose_3d: (B, 17, 3), met, root-relative (da xu ly o Trainer)
            joints_in = gt_pose_3d
            lifted = None
            if self.teacher_lift and pose_2d is not None and lift_alpha is not None:
                # (A) Tron GT voi joint MotionBERT lift: input = (1-a)*GT + a*lift  (a=0: GT sach, a=1: nhu Student)
                lifted = self.lift_2d_to_3d(pose_2d, joints_mask=joints_mask)
                lifted = (lifted - lifted[:, 0:1, :]).to(gt_pose_3d.dtype)     # root-relative
                a = torch.as_tensor(lift_alpha, dtype=gt_pose_3d.dtype, device=gt_pose_3d.device)
                a = a.reshape(-1, 1, 1) if a.dim() > 0 else a.reshape(1, 1, 1)
                joints_in = (1.0 - a) * gt_pose_3d + a * lifted

        result = self.smpl_model(
            joints=joints_in,
            img_feats=feature_map,
            is_train=is_train,
            return_features=True,
        )
        result['teacher_input_joints'] = joints_in      # (B, 17, 3) input thuc su dua vao Teacher
        if lifted is not None:
            result['lifted_joints_3d'] = lifted         # (B, 17, 3) joint MotionBERT lift
        return result

    def forward_student(self, image, pose_2d, is_train, gt_pose_3d=None, alpha=1.0, joints_mask=None):
        with torch.no_grad():
            feature_map, _ = self.get_image_features(image)
            # MotionBERT finetune ra milimet, lift_2d_to_3d đã đổi về mét (root-relative bên dưới)
            pose_3d = self.lift_2d_to_3d(pose_2d, joints_mask=joints_mask)
            pose_3d = pose_3d - pose_3d[:, 0:1, :]            # root-relative

        result = self.smpl_model(
            joints=pose_3d,
            img_feats=feature_map,
            is_train=is_train,
            return_features=True,
        )
        result['privileged_3d'] = self.privileged_joint_head(result['feat_joint'])
        result['lifted_joints_3d'] = pose_3d   # joint nhiễu đưa vào student (để log / loss nếu cần)
        return result

    def forward_arts(self, image, pose_input, is_train, use_gt_3d=False,
                     gt_pose_6d=None, kp2d=None, kp_conf=None,
                     pose_valid_mask=None, gt_joints_3d=None, joints_mask=None):
        with torch.no_grad():
            ft_map, global_feature = self.get_image_features(image)

            if not use_gt_3d:
                # Output MotionBERT đã được đổi về mét trong lift_2d_to_3d
                pose_3d = self.lift_2d_to_3d(pose_input, joints_mask=joints_mask)
                pose_3d = pose_3d - pose_3d[:, 0:1, :]        # root-relative
            else:
                # Dùng trực tiếp GT 3D (đã là đơn vị Mét và root-relative từ Trainer)
                pose_3d = pose_input

        output = self.pose_mesh_coevo(
            pose_3d,
            ft_map,
            is_train=is_train,
            gt_pose_6d=gt_pose_6d,
            kp2d=kp2d,
            kp_conf=kp_conf,
            pose_valid_mask=pose_valid_mask
        )
        # MotionBERT outputs 3D joints in millimeters; convert to meters so
        # joint_img shares the unit of the GT joints used in the training loss.
        output["joint_img"] = pose_3d
        return output

    def forward(self, image, joints, is_train=True, use_gt_3d=False,
                gt_pose_6d=None, kp2d=None, kp_conf=None,
                pose_valid_mask=None, gt_joints_3d=None, alpha=1.0, joints_mask=None,
                pose_2d=None, lift_alpha=None):
        if self.mode == "teacher":
            return self.forward_teacher(image, joints, is_train, pose_2d=pose_2d,
                                        joints_mask=joints_mask, lift_alpha=lift_alpha)
        if self.mode == "student":
            return self.forward_student(image, joints, is_train, gt_pose_3d=gt_joints_3d, alpha=alpha, joints_mask=joints_mask)
        return self.forward_arts(
            image, joints, is_train, use_gt_3d,
            gt_pose_6d=gt_pose_6d, kp2d=kp2d, kp_conf=kp_conf,
            pose_valid_mask=pose_valid_mask, gt_joints_3d=gt_joints_3d,
            joints_mask=joints_mask,
        )


def get_model(num_joint, embed_dim, depth=None):
    return ARTS(num_joint, embed_dim)