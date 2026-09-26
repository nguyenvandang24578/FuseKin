import os, sys
sys.path.append('./lib')
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F


from core.config import cfg
import math
from models.smpl_mps import SMPL_MEAN_PARAMS

from models.hypergcn import HYPERGCv2
from models.teacher import Teacher
from models.Core_model import CrossAttentionBlock
from models.common import Vposer
from models.spin import RegressorSpin

from utils.transforms import rot6d_to_axis_angle
TEACHER_CHPT = cfg.MODEL.TEACHER
SMPL_MODEL_DIR = 'data_final/base_data'
SMPL_MEAN_PARAMS = 'data_final/base_data/smpl_mean_params.npz'
BASE_DATA_DIR = 'data_final/base_data'

class Pose2Mesh(nn.Module):
    def __init__(self, num_joint, embed_dim=512, smpl_head_hidden_dim: int = 256, smpl_head_depth: int = 3):
        super(Pose2Mesh, self).__init__()

        self.vposer = Vposer()
        for param in self.vposer.parameters():
            param.requires_grad = False
        self.vposer.eval()

        # SMPL model layer for get_coord (mesh generation from params)
        from utils.smpl import SMPL as SMPLModel
        self.human_model = SMPLModel()
        self.human_model_layer = self.human_model.layer['neutral'].cuda()
        self.joint_regressor = self.human_model.joint_regressor
        self.regressorspin = RegressorSpin()
        pretrained_dict = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))['model']
        self.regressorspin.load_state_dict(pretrained_dict, strict=False)
# =========================================================
        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
#-------------------------------------------------------------------------------------
        self.pose_embed  = nn.Linear(6, embed_dim)
        self.shape_embed  = nn.Linear(10, embed_dim)

        self.fuse_shape = CrossAttentionBlock(q_dim=512, k_dim=1024, v_dim=1024, kv_num = 1, num_heads=8, mlp_ratio=4., qkv_bias=True,
                                        drop=0., attn_drop=0., drop_path=0.2, has_mlp=True)
#-------------------------------------------------------------------------------------
        self.fusion = Teacher(num_joint, embed_dim, vert_anchors = 16, horz_anchors = 16)
        pretrained_dict = torch.load(osp.join(TEACHER_CHPT, 'best.pth.tar'), weights_only=False)['model_state_dict']
        self.fusion.load_state_dict(pretrained_dict, strict=False)
        for param in self.fusion.parameters():
            param.requires_grad = False
        self.fusion.eval()
#-------------------------------------------------------------------------------------
        self.node_pe = nn.Embedding(24, embed_dim)
        self.num_hyper_layers = 3
        self.spatial_hypers = nn.ModuleList([
            HYPERGCv2(embed_dim, embed_dim, num_edges=5)
            for _ in range(self.num_hyper_layers)
        ])
#-------------------------------------------------------------------------------------
        # Heads theo pattern JOTR: root riêng, body qua vposer, shape, cam
        # self.root_pose_head = MLP(embed_dim, smpl_head_hidden_dim, 6, 2)     # root rotation 6D
        self.pose_head = MLP(embed_dim, smpl_head_hidden_dim, 6, smpl_head_depth)
        self.shape_head = MLP(embed_dim, smpl_head_hidden_dim, 10, smpl_head_depth)
        self.cam_head = MLP(1024, smpl_head_hidden_dim, 3, 2)           # camera params
        self.shape_token = nn.Embedding(1, embed_dim)
#-------------------------------------------------------------------------------------
        self.gamma_proj = nn.Linear(1024, embed_dim)
        self.beta_proj  = nn.Linear(1024, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, joints, img_feats,  is_train=True, J_regressor=None):
        batch_size = img_feats.shape[0]   # B

        mean_pose  = self.init_pose.view(1, 24, 6)              # (1, 24, 6)
        mean_shape  = self.init_shape.view(1, 10)              # (1, 24, 6)
        pose_emb   = self.pose_embed(mean_pose)                  # (1, 24, embed_dim)
        shape_emb = self.shape_embed(mean_shape) #(1, dim)
        pose_token = pose_emb.expand(
            batch_size, -1, -1
        )   
        shape_token = self.shape_token.weight.unsqueeze(0).expand(
            batch_size, 1, -1
        )
        shape_emb = shape_emb.unsqueeze(1)
        shape_token = shape_token + shape_emb

        output, feature = self.fusion(joints, img_feats, is_train=is_train, J_regressor=J_regressor, return_features=True) #(B, 1024)

        if isinstance(feature, dict):
            global_ft = feature['concat_feat']
        img_feats_trans = feature['img']
        gamma = self.gamma_proj(global_ft).unsqueeze(1) + 1.0 #(B, 1, 512)
        beta  = self.beta_proj(global_ft).unsqueeze(1)   # (B, 1, 512)

        out = gamma * pose_token + beta  # (B, 24, 512)
        idx = torch.arange(24, device=out.device)   # (24,)
        dang = self.norm(out) + self.node_pe(idx)   # (B, 24, 512) + (24, 512)
        
        # HYPERGCv2 v2 expects (B, T, 24, D) directly, no need to permute.
        dang_hyper = dang
        for hyper_layer in self.spatial_hypers:
            dang_hyper, aux = hyper_layer(dang_hyper)
        pose_token_op = dang_hyper + dang # (B, 24, D) + skip around HyperGCN
        f_pose  = self.pose_head(pose_token_op) # (B, T, 24, 6)   
        inv_pred2rot6d = f_pose.reshape(batch_size, -1)
        # # Pool tất cả joint tokens để lấy global pose feature
        # pose_global = pose_token_op.mean(dim=1)  # (B, 512)

        # # 1. Root pose: MLP -> 6D rotation -> axis-angle (B, 3)
        # root_pose_6d = self.root_pose_head(pose_global)            # (B, 6)
        # root_pose = rot6d_to_axis_angle(root_pose_6d)              # (B, 3)

        # 2. Body pose: MLP -> vposer latent (B, 32) -> decode -> axis-angle (B, 69)
        # pose_latent = self.body_pose_head(pose_global)             # (B, 32)
        # body_pose = self.vposer(pose_latent)                       # (B, 69) = 23 joints × 3

        # # 3. Ghép root + body -> full SMPL pose (B, 72)
        # full_pose = torch.cat([root_pose, body_pose], dim=1)       # (B, 72)

        # 4. Camera params
        cam_param = self.cam_head(global_ft)                     # (B, 3)

#---------------------------------------------------------------------------------------------------------------------------------------
        # 5. Shape: CrossAttention fusion
        global_ft_seq = global_ft.unsqueeze(1)                     # (B, 1, 1024)
        shape_output = self.fuse_shape(shape_token, global_ft_seq, global_ft_seq)  # (B, 1, 512)
        f_shape = self.shape_head(shape_output)                    # (B, 1, 10)
        shape_param = f_shape.reshape(batch_size, -1)              # (B, 10)

#---------------------------------------------------------------------------------------------------------------------------------------
        # 6. SMPL forward: get_coord
        cam_trans = self.get_camera_trans(cam_param)               # (B, 3)
        # joint_proj, joint_cam, mesh_cam, mesh_cam_render = self.get_coord(
        #     full_pose, shape_param, cam_trans
        # )
        output = self.regressorspin(img_feats_trans,
                                    init_pose=inv_pred2rot6d,
                                    init_shape=shape_param,
                                    init_cam=cam_param,
                                    is_train=is_train,
                                    J_regressor=J_regressor)[0]
        return {
            'joint_proj': output['kp_2d'],
            'joint_cam': output['kp_3d'],
            'smpl_mesh_cam': output['verts'],
            'smpl_pose': output['theta'][:, 3:75],  # full pose (B, 72)
            'smpl_shape': output['theta'][:, 75:85],
            'cam_param': output['theta'][:, :3]
        }

    def get_camera_trans(self, cam_param):
        """Convert predicted camera parameters to camera translation.
        
        Args:
            cam_param: (B, 3) - [tx, ty, gamma]
        Returns:
            cam_trans: (B, 3) - [tx, ty, tz]
        """
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])  # positive depth
        k_value = torch.FloatTensor([
            math.sqrt(
                cfg.DATASET.focal[0] * cfg.DATASET.focal[1]
                * cfg.DATASET.camera_3d_size * cfg.DATASET.camera_3d_size
                / (cfg.input_img_shape[0] * cfg.input_img_shape[1])
            )
        ]).cuda().view(-1)
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
        mesh_cam, _ = self.human_model_layer(smpl_pose, smpl_shape, smpl_trans)  # (B, 6890, 3)

        # Regress joints from mesh
        joint_regressor = torch.from_numpy(self.joint_regressor).float().cuda()
        joint_cam = torch.bmm(
            joint_regressor[None, :, :].repeat(batch_size, 1, 1),
            mesh_cam
        )  # (B, 30, 3)

        root_joint_idx = self.human_model.root_joint_idx

        # Project 3D to 2D
        x = joint_cam[:, :, 0] / (joint_cam[:, :, 2] + 1e-4) * cfg.DATASET.focal[0] + cfg.DATASET.princpt[0]
        y = joint_cam[:, :, 1] / (joint_cam[:, :, 2] + 1e-4) * cfg.DATASET.focal[1] + cfg.DATASET.princpt[1]
        x = x / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)

        mesh_cam_render = mesh_cam.clone()

        # Root-relative
        root_cam = joint_cam[:, root_joint_idx, None, :]
        joint_cam = joint_cam - root_cam
        mesh_cam = mesh_cam - root_cam

        return joint_proj, joint_cam, mesh_cam, mesh_cam_render

    def train(self, mode=True):
        super().train(mode)
        self.vposer.eval()
        self.fusion.eval()

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