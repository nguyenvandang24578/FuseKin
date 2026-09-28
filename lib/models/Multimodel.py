import os, sys
sys.path.append('./lib')
import numpy as np
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F


from core.config import cfg
import math

from models.hypergcn import HYPERGCv2
from models.teacher import Teacher
from models.Core_model import CrossAttentionBlock
from models.common import Vposer
from models.smpl_hyperdiff import SMPL_HyperDiff

from utils.transforms import rot6d_to_axis_angle
TEACHER_CHPT = cfg.MODEL.TEACHER
SMPL_MODEL_DIR = 'data_final/base_data'
SMPL_MEAN_PARAMS = 'data_final/base_data/smpl_mean_params.npz'
BASE_DATA_DIR = 'data_final/base_data'

class Pose2Mesh(nn.Module):
    def __init__(self, num_joint, embed_dim=512, smpl_head_hidden_dim: int = 256, smpl_head_depth: int = 3):
        super(Pose2Mesh, self).__init__()

        self.refiner = getattr(cfg.MODEL, 'REFINER', 'diffusion')

        self.vposer = Vposer()
        for param in self.vposer.parameters():
            param.requires_grad = False
        self.vposer.eval()

        from utils.smpl import SMPL as SMPLModel
        self.human_model = SMPLModel()
        self.human_model_layer = self.human_model.layer['neutral']
        self.joint_regressor = self.human_model.joint_regressor
        self.register_buffer(
            'joint_regressor_t',
            torch.from_numpy(self.joint_regressor).float()
        )

        # --- OLD pose path (kept for checkpoint compat) ---
        self.pose_embed  = nn.Linear(6, embed_dim)
        self.shape_embed  = nn.Linear(10, embed_dim)

        self.fuse_shape = CrossAttentionBlock(q_dim=512, k_dim=1024, v_dim=1024, kv_num = 1, num_heads=8, mlp_ratio=4., qkv_bias=True,
                                        drop=0., attn_drop=0., drop_path=0.2, has_mlp=True)
        # Cross-attention: each pose token attends to spatial img + joint tokens
        self.pose_context_attn = CrossAttentionBlock(
            q_dim=embed_dim, k_dim=embed_dim, v_dim=embed_dim,
            kv_num=273,  # 16*16 + 17; actual length is dynamic thanks to CrossAttention fix
            num_heads=8, mlp_ratio=4., qkv_bias=True,
            drop=0., attn_drop=0., drop_path=0.2, has_mlp=True
        )
        self.fusion = Teacher(num_joint, embed_dim, vert_anchors = 16, horz_anchors = 16, depth = 3)
        self.node_pe = nn.Embedding(24, embed_dim)
        self.num_hyper_layers = 3
        self.spatial_hypers = nn.ModuleList([
            HYPERGCv2(embed_dim, embed_dim, num_edges=5)
            for _ in range(self.num_hyper_layers)
        ])
        self.root_pose_head = MLP(embed_dim, smpl_head_hidden_dim, 6, 2)
        self.body_pose_head = MLP(embed_dim, smpl_head_hidden_dim, 32, smpl_head_depth)
        self.shape_head = MLP(embed_dim, smpl_head_hidden_dim, 10, smpl_head_depth)
        self.cam_head = MLP(1024, smpl_head_hidden_dim, 3, 2)
        self.shape_token = nn.Embedding(1, embed_dim)
        # deprecated, kept for backward-compat with old checkpoints
        self.gamma_proj = nn.Linear(1024, embed_dim)
        self.beta_proj  = nn.Linear(1024, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # --- NEW diffusion path ---
        if self.refiner == 'diffusion':
            self.diffusion = SMPL_HyperDiff()
            # Freeze old pose-path params so they don't cause DDP "unused param" errors.
            # They remain in state_dict for checkpoint compatibility.
            _old_pose_modules = [
                self.pose_embed, self.pose_context_attn, self.norm,
                self.node_pe, self.spatial_hypers, self.root_pose_head,
                self.body_pose_head, self.gamma_proj, self.beta_proj,
            ]
            for mod in _old_pose_modules:
                for p in mod.parameters():
                    p.requires_grad = False

    def forward(self, joints, img_feats, is_train=True, J_regressor=None,
                gt_pose_6d=None, kp2d=None, kp_conf=None, pose_valid_mask=None):
        """
        Args (new optional params — don't change positional order):
            gt_pose_6d     : (B,24,6) GT 6D rotations (train only)
            kp2d           : (B,17,2) 2D keypoints in [-1,1]
            kp_conf        : (B,17) binary/continuous confidence
            pose_valid_mask: (B,24) joint validity for diff_loss masking
        """
        batch_size = img_feats.shape[0]
        device = img_feats.device

        # ============================================================
        # 1. Teacher fusion  (always runs — provides cam/shape features)
        # ============================================================
        _, pred_pose_6d, pred_shape, pred_cam, feature = self.fusion(
            joints, img_feats, is_train=is_train,
            J_regressor=J_regressor, return_features=True
        )
        if isinstance(feature, dict):
            global_ft = feature['concat_feat']
        else:
            global_ft = feature

        # ============================================================
        # 2. Pose prediction
        # ============================================================
        diff_loss = torch.zeros(1, device=device).squeeze()

        if self.refiner == 'diffusion':
            # ----- diffusion path -----
            if is_train and gt_pose_6d is not None:
                pred_x0, diff_loss = self.diffusion(
                    gt_pose_6d, kp2d, kp_conf,
                    is_train=True, valid_mask=pose_valid_mask,
                )
                diff_loss = diff_loss.mean()
            else:
                pred_x0 = self.diffusion(
                    None, kp2d, kp_conf, is_train=False,
                )
            full_pose = rot6d_to_axis_angle(
                pred_x0.reshape(-1, 6)
            ).reshape(batch_size, -1)
        else:
            # ----- legacy HYPERGCv2 path -----
            pose_token = self.pose_embed(pred_pose_6d)
            shape_emb_ = self.shape_embed(pred_shape)

            img_out = feature['img_out']
            joint_out_ctx = feature['joint_out']
            context_tokens = torch.cat([img_out, joint_out_ctx], dim=1)
            pose_token_ctx = self.pose_context_attn(
                pose_token, context_tokens, context_tokens
            )
            idx = torch.arange(24, device=device)
            dang = self.norm(pose_token_ctx) + self.node_pe(idx)
            for hyper_layer in self.spatial_hypers:
                dang, _aux = hyper_layer(dang)
            pose_global = dang.mean(dim=1)
            root_feat = dang[:, 0, :]
            root_pose_6d = self.root_pose_head(root_feat)
            root_pose = rot6d_to_axis_angle(root_pose_6d)
            pose_latent = self.body_pose_head(pose_global)
            body_pose = self.vposer(pose_latent)
            full_pose = torch.cat([root_pose, body_pose], dim=1)
            pred_x0 = None

        # ============================================================
        # 3. Camera  (unchanged)
        # ============================================================
        cam_param = self.cam_head(global_ft)

        # ============================================================
        # 4. Shape  (unchanged)
        #    NOTE: shape_token uses pred_shape from SPIN (self.fusion).
        #    This is intentional — the SPIN shape init is still useful.
        # ============================================================
        shape_emb = self.shape_embed(pred_shape)
        shape_token = self.shape_token.weight.unsqueeze(0).expand(batch_size, 1, -1)
        shape_emb = shape_emb.unsqueeze(1)
        shape_token = shape_token + shape_emb
        global_ft_seq = global_ft.unsqueeze(1)
        shape_output = self.fuse_shape(shape_token, global_ft_seq, global_ft_seq)
        f_shape = self.shape_head(shape_output)
        shape_param = f_shape.reshape(batch_size, -1)

        # ============================================================
        # 5. SMPL forward  (unchanged)
        # ============================================================
        cam_trans = self.get_camera_trans(cam_param)
        joint_proj, joint_cam, mesh_cam, mesh_cam_render = self.get_coord(
            full_pose, shape_param, cam_trans
        )
        
        if is_train and getattr(cfg.LOSS, 'DETACH_POSE_FOR_PROJ', True):
            _, _, mesh_cam_proj, _ = self.get_coord(
                full_pose.detach(), shape_param, cam_trans
            )
        else:
            mesh_cam_proj = mesh_cam

        result = {
            'joint_proj': joint_proj,
            'joint_cam': joint_cam,
            'smpl_mesh_cam': mesh_cam,
            'smpl_mesh_cam_proj': mesh_cam_proj,
            'smpl_pose': full_pose,
            'smpl_shape': shape_param,
            'cam_param': cam_trans,
        }
        if self.refiner == 'diffusion':
            result['diff_loss'] = diff_loss
            result['pred_pose_6d_refined'] = pred_x0
        return result

    def get_camera_trans(self, cam_param):
        """Convert predicted camera parameters to camera translation.
        
        Args:
            cam_param: (B, 3) - [tx, ty, gamma]
        Returns:
            cam_trans: (B, 3) - [tx, ty, tz]
        """
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])
        k_value = math.sqrt(
            cfg.DATASET.focal[0] * cfg.DATASET.focal[1]
            * cfg.DATASET.camera_3d_size * cfg.DATASET.camera_3d_size
            / (cfg.input_img_shape[0] * cfg.input_img_shape[1])
        )
        t_z = k_value * gamma
        cam_trans = torch.cat([t_xy, t_z[:, None]], dim=1)
        return cam_trans

    def get_coord(self, smpl_pose, smpl_shape, smpl_trans):
        """Run SMPL forward pass to get mesh and joint coordinates.
        
        Args:
            smpl_pose: (B, 72) full SMPL pose in axis-angle
            smpl_shape: (B, 10) shape parameters
            smpl_trans: (B, 3) camera translation
        Returns:
            joint_proj: (B, 30, 2) projected 2D joints
            joint_cam: (B, 30, 3) root-relative 3D joints
            mesh_cam: (B, 6890, 3) root-relative mesh vertices
            mesh_cam_render: (B, 6890, 3) absolute mesh vertices
        """
        batch_size = smpl_pose.shape[0]
        mesh_cam, _ = self.human_model_layer(smpl_pose, smpl_shape, smpl_trans)

        joint_regressor = self.joint_regressor_t
        joint_cam = torch.bmm(
            joint_regressor[None, :, :].repeat(batch_size, 1, 1),
            mesh_cam
        )

        root_joint_idx = self.human_model.root_joint_idx

        x = joint_cam[:, :, 0] / (joint_cam[:, :, 2] + 1e-4) * cfg.DATASET.focal[0] + cfg.DATASET.princpt[0]
        y = joint_cam[:, :, 1] / (joint_cam[:, :, 2] + 1e-4) * cfg.DATASET.focal[1] + cfg.DATASET.princpt[1]
        x = x / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)

        mesh_cam_render = mesh_cam.clone()

        root_cam = joint_cam[:, root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam - root_cam

        return joint_proj, joint_cam, mesh_cam, mesh_cam_render

    def train(self, mode=True):
        super().train(mode)
        self.vposer.eval()

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                num_layers: int, sigmoid_output: bool = False) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x
def get_model(num_joint, embed_dim):
    model = Pose2Mesh(num_joint, embed_dim)
    return model