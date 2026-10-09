"""
Teacher / Student end-to-end:

    img_feats + joints -> Fusion (cross-attn 2 chieu) -> HyperGCN -> heads -> VPoser -> SMPL -> mesh

- Teacher: joints = GT 3D (sach)      -> joint_encoder='gt'
- Student: joints = 3D joint nhieu    -> joint_encoder='noisy'
- Khong dung RegressorSpin. Query pose / shape khoi tao tu mean_pose / mean_shape.
- Nhanh diffusion (SMPL_HyperDiff) van con, chi chay khi refiner='diffusion'.
- Model KHONG tinh loss. Moi thu can cho loss / KD duoc tra ve trong dict output.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
from models.HGraph import create_layers
from models.Core_model import CrossAttentionBlock
from models.common import Vposer
from utils.transforms import rot6d_to_axis_angle

SMPL_MEAN_PARAMS = 'data_final/base_data/smpl_mean_params.npz'


# ============================================================
# Fusion blocks (chuyen tu student.py / teacher.py cu)
# ============================================================
class StudentJointExtractor(nn.Module):
    def __init__(self, in_dim=3, out_dim=512):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        # 1 layer Transformer de cac khop trao doi thong tin & tu sua nhieu
        self.denoise_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=8, dim_feedforward=1024, batch_first=True
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, noisy_joints, pos_emb):
        # noisy_joints: (B, J, 3)
        x = self.proj(noisy_joints) + pos_emb
        x = self.denoise_layer(x)
        return self.norm(x)


class LearnableCoefficient(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        return x * self.bias


class RGBJointCrossTransformerBlock(nn.Module):
    """
    rgb tokens:   (B, H*W, C)
    joint tokens: (B, J, C)
    Cross-attention 2 chieu, Pre-LN, residual dung LearnableCoefficient.
    """
    def __init__(self, d_model, h, block_exp=4, resid_pdrop=0.1):
        super().__init__()
        self.h = h
        self.d_k = d_model // h

        self.q_joint = nn.Linear(d_model, d_model)
        self.k_rgb = nn.Linear(d_model, d_model)
        self.v_rgb = nn.Linear(d_model, d_model)

        self.q_rgb = nn.Linear(d_model, d_model)
        self.k_joint = nn.Linear(d_model, d_model)
        self.v_joint = nn.Linear(d_model, d_model)

        self.out_joint = nn.Linear(d_model, d_model)
        self.out_rgb = nn.Linear(d_model, d_model)

        self.coef1 = LearnableCoefficient()
        self.coef2 = LearnableCoefficient()
        self.coef3 = LearnableCoefficient()
        self.coef4 = LearnableCoefficient()
        self.coef5 = LearnableCoefficient()
        self.coef6 = LearnableCoefficient()
        self.coef7 = LearnableCoefficient()
        self.coef8 = LearnableCoefficient()

        self.ln_joint1 = nn.LayerNorm(d_model)
        self.ln_rgb1 = nn.LayerNorm(d_model)
        self.ln_joint2 = nn.LayerNorm(d_model)
        self.ln_rgb2 = nn.LayerNorm(d_model)

        self.mlp_joint = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.GELU(),
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
        self.mlp_rgb = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.GELU(),
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def _split_heads(self, x, b):
        n = x.shape[1]
        return x.view(b, n, self.h, self.d_k).transpose(1, 2)  # (b, h, n, d_k)

    def forward(self, rgb_tok, joint_tok, return_intermediate=False):
        b = rgb_tok.shape[0]

        rgb_norm = self.ln_rgb1(rgb_tok)
        joint_norm = self.ln_joint1(joint_tok)

        # Joint attend to RGB
        q_j = self._split_heads(self.q_joint(joint_norm), b)
        k_r = self._split_heads(self.k_rgb(rgb_norm), b)
        v_r = self._split_heads(self.v_rgb(rgb_norm), b)
        attn_j = torch.softmax(q_j @ k_r.transpose(-2, -1) / self.d_k ** 0.5, dim=-1)
        joint_att = (attn_j @ v_r).transpose(1, 2).reshape(b, joint_tok.shape[1], -1)
        joint_att = self.out_joint(joint_att)

        # RGB attend to Joint
        q_r = self._split_heads(self.q_rgb(rgb_norm), b)
        k_j = self._split_heads(self.k_joint(joint_norm), b)
        v_j = self._split_heads(self.v_joint(joint_norm), b)
        attn_r = torch.softmax(q_r @ k_j.transpose(-2, -1) / self.d_k ** 0.5, dim=-1)
        rgb_att = (attn_r @ v_j).transpose(1, 2).reshape(b, rgb_tok.shape[1], -1)
        rgb_att = self.out_rgb(rgb_att)

        # Residual 1
        joint_out = self.coef1(joint_tok) + self.coef2(joint_att)
        rgb_out = self.coef3(rgb_tok) + self.coef4(rgb_att)

        # FFN + Residual 2
        joint_out = self.coef5(joint_out) + self.coef6(self.mlp_joint(self.ln_joint2(joint_out)))
        rgb_out = self.coef7(rgb_out) + self.coef8(self.mlp_rgb(self.ln_rgb2(rgb_out)))

        if return_intermediate:
            intermediate = [{
                'img': rgb_out,
                'joint': joint_out,
                'attn_joint_to_img': attn_j,
                'attn_img_to_joint': attn_r,
            }]
            return rgb_out, joint_out, intermediate
        return rgb_out, joint_out


class RGBJointCrossTransformer(nn.Module):
    def __init__(self, d_model, h, block_exp=4, resid_pdrop=0.1, depth=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            RGBJointCrossTransformerBlock(d_model, h, block_exp, resid_pdrop)
            for _ in range(depth)
        ])

    def forward(self, rgb_tok, joint_tok, return_intermediate=False):
        intermediate = []
        for block in self.blocks:
            out = block(rgb_tok, joint_tok, return_intermediate=return_intermediate)
            if return_intermediate:
                rgb_tok, joint_tok, block_features = out
                intermediate.extend(block_features)
            else:
                rgb_tok, joint_tok = out
        if return_intermediate:
            return rgb_tok, joint_tok, intermediate
        return rgb_tok, joint_tok


class RGBJointFusion(nn.Module):
    """img_feats (B,2048,H,W) + joints (B,J,3) -> token anh, token joint, feature toan cuc."""
    def __init__(self, num_joint=17, embed_dim=512, vert_anchors=8, horz_anchors=8,
                 in_channels=2048, depth=3, joint_encoder='gt'):
        super().__init__()
        assert joint_encoder in ('gt', 'noisy'), f"joint_encoder khong hop le: {joint_encoder}"
        self.joint_encoder = joint_encoder
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        self.img_proj = nn.Conv2d(in_channels, embed_dim, 1)

        if joint_encoder == 'noisy':
            self.projector_student = StudentJointExtractor(in_dim=3, out_dim=embed_dim)
        else:
            # print("TEACHER IN TEACHER_STUDENT.PY")
            self.joint_proj = nn.Linear(3, embed_dim)
            self.norm_joint_proj = nn.LayerNorm(embed_dim)

        self.pos_emb_img = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, embed_dim))
        self.pos_emb_joint = nn.Parameter(torch.zeros(1, num_joint, embed_dim))

        self.cfcer = RGBJointCrossTransformer(
            d_model=embed_dim, h=8, block_exp=4, resid_pdrop=0.1, depth=depth
        )
        self.norm_img_in = nn.LayerNorm(embed_dim)

    def forward(self, joints, img_feats, return_features=False):
        img_feats = self.img_proj(img_feats)               # (B, C, H, W)
        bs, c, h, w = img_feats.shape

        if self.joint_encoder == 'noisy':
            joints_tok = self.projector_student(joints, self.pos_emb_joint)
        else:
            joints_tok = self.norm_joint_proj(self.joint_proj(joints) + self.pos_emb_joint)

        img_tok = img_feats.flatten(2).transpose(1, 2)     # (B, H*W, C)

        if h * w != self.pos_emb_img.shape[1]:
            pos_emb = self.pos_emb_img.transpose(1, 2).reshape(1, c, self.vert_anchors, self.horz_anchors)
            pos_emb = F.interpolate(pos_emb, size=(h, w), mode='bilinear', align_corners=False)
            pos_emb = pos_emb.flatten(2).transpose(1, 2)
        else:
            pos_emb = self.pos_emb_img

        img_tok = self.norm_img_in(img_tok + pos_emb)

        out = self.cfcer(img_tok, joints_tok, return_intermediate=return_features)
        if return_features:
            img_out, joint_out, layers = out
        else:
            img_out, joint_out = out
            layers = []

        concat_feat = torch.cat([img_out.mean(dim=1), joint_out.mean(dim=1)], dim=1)  # (B, 2C)
        return {
            'img_out': img_out,          # (B, H*W, C)
            'joint_out': joint_out,      # (B, J, C)
            'joint_tok': joints_tok,     # (B, J, C) token joint sau encoder, truoc cross-attn
            'img_tok': img_tok,          # (B, H*W, C) token anh DAU VAO fusion (sau pos_emb + LN), chua nhin thay joint
            'concat_feat': concat_feat,  # (B, 2C)
            'layers': layers,
        }


# ============================================================
# Heads
# ============================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, sigmoid_output=False):
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


# ============================================================
# Model chung cho Teacher va Student
# ============================================================
class FusionPose2Mesh(nn.Module):
    def __init__(self, num_joint=17, embed_dim=512, joint_encoder='gt', refiner='hypergcn',
                 vert_anchors=8, horz_anchors=8, depth=3,
                 smpl_head_hidden_dim=256, smpl_head_depth=3, use_pose_pe=True, use_img_probe=False):
        super().__init__()
        assert refiner in ('hypergcn', 'diffusion'), f"refiner khong hop le: {refiner}"
        self.refiner = refiner
        self.num_joint = num_joint
        self.embed_dim = embed_dim
        self.use_pose_pe = use_pose_pe
        self._frozen = False

        # ---- VPoser (dong bang) ----
        self.vposer = Vposer()
        for p in self.vposer.parameters():
            p.requires_grad = False
        self.vposer.eval()

        # ---- SMPL ----
        from utils.smpl import SMPL as SMPLModel
        self.human_model = SMPLModel()
        self.human_model_layer = self.human_model.layer['neutral']
        self.joint_regressor = self.human_model.joint_regressor
        self.register_buffer('joint_regressor_t', torch.from_numpy(self.joint_regressor).float())

        # ---- mean pose / shape (thay cho output cua SPIN) ----
        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:].astype('float32')).view(1, 24, 6)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).view(1, 10)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)

        # ---- Fusion ----
        self.fusion = RGBJointFusion(
            num_joint=num_joint, embed_dim=embed_dim,
            vert_anchors=vert_anchors, horz_anchors=horz_anchors,
            depth=depth, joint_encoder=joint_encoder,
        )
        global_dim = embed_dim * 2

        # ---- Pose path (HyperGCN) ----
        self.pose_embed = nn.Linear(6, embed_dim)
        self.pose_pe = nn.Embedding(24, embed_dim) if use_pose_pe else None
        self.node_pe = nn.Embedding(num_joint, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.spatial_hypers = create_layers(
            dim=embed_dim, n_layers=3, mlp_ratio=4, act_layer=nn.GELU,
            attn_drop=0., drop_rate=0., drop_path_rate=0.,
            use_layer_scale=True, layer_scale_init_value=1e-5,
            use_adaptive_fusion=False, hierarchical=False, neighbour_num=4,
        )
        self.pose_context_attn = CrossAttentionBlock(
            q_dim=embed_dim, k_dim=embed_dim, v_dim=embed_dim, kv_num=num_joint,
            num_heads=8, mlp_ratio=4., qkv_bias=True,
            drop=0., attn_drop=0., drop_path=0.2, has_mlp=True,
        )
        self.root_pose_head = MLP(embed_dim, smpl_head_hidden_dim, 6, 2)
        self.body_pose_head = MLP(embed_dim, smpl_head_hidden_dim, 32, smpl_head_depth)

        # ---- Shape path (giu nguyen cach cu, init tu mean_shape) ----
        self.shape_embed = nn.Linear(10, embed_dim)
        self.shape_token = nn.Embedding(1, embed_dim)
        self.fuse_shape = CrossAttentionBlock(
            q_dim=embed_dim, k_dim=global_dim, v_dim=global_dim, kv_num=1,
            num_heads=8, mlp_ratio=4., qkv_bias=True,
            drop=0., attn_drop=0., drop_path=0.2, has_mlp=True,
        )
        self.shape_head = MLP(embed_dim, smpl_head_hidden_dim, 10, smpl_head_depth)

        # ---- Camera ----
        self.cam_head = MLP(global_dim, smpl_head_hidden_dim, 3, 2)

        # ---- Head tham do anh (tuy chon): doan joint 3D CHI tu token anh dau vao fusion ----
        # Muc dich: (1) ep img_proj / pos_emb_img / norm_img_in mang thong tin tu the,
        #           (2) do truc tiep xem dac trung anh co du tot khong (so voi MotionBERT).
        self.use_img_probe = use_img_probe
        if use_img_probe:
            self.probe_query = nn.Embedding(num_joint, embed_dim)
            self.probe_attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
            self.probe_norm = nn.LayerNorm(embed_dim)
            self.probe_head = MLP(embed_dim, smpl_head_hidden_dim, 3, 2)

        # ---- Diffusion (tuy chon) ----
        if refiner == 'diffusion':
            from models.smpl_hyperdiff import SMPL_HyperDiff
            self.diffusion = SMPL_HyperDiff()
            # Dong bang nhanh HyperGCN de tranh loi DDP "unused parameter"
            legacy = [self.pose_embed, self.pose_context_attn, self.norm, self.node_pe,
                      self.spatial_hypers, self.root_pose_head, self.body_pose_head]
            if self.pose_pe is not None:
                legacy.append(self.pose_pe)
            for mod in legacy:
                for p in mod.parameters():
                    p.requires_grad = False

    # --------------------------------------------------------
    def forward(self, joints, img_feats, is_train=True,
                gt_pose_6d=None, kp2d=None, kp_conf=None, pose_valid_mask=None,
                return_features=False):
        """
        Args:
            joints   : (B, J, 3) root-relative, don vi met (GT cho teacher, nhieu cho student)
            img_feats: (B, 2048, H, W) feature map backbone
            gt_pose_6d, kp2d, kp_conf, pose_valid_mask: chi dung khi refiner='diffusion'
            return_features: tra them feature trung gian cho KD
        """
        B = img_feats.shape[0]
        device = img_feats.device

        # 1. Fusion
        fus = self.fusion(joints, img_feats, return_features=return_features)
        joint_out = fus['joint_out']
        global_ft = fus['concat_feat']

        # 1b. Head tham do anh: chi dung token anh dau vao fusion (khong qua joint)
        img_probe_joints = None
        if self.use_img_probe:
            q = self.probe_query.weight.unsqueeze(0).expand(B, -1, -1)
            att, _ = self.probe_attn(q, fus['img_tok'], fus['img_tok'], need_weights=False)
            probe = self.probe_head(self.probe_norm(q + att))              # (B, J, 3)
            img_probe_joints = probe - probe[:, 0:1, :]                    # root-relative nhu GT

        # 2. Pose
        diff_loss = torch.zeros(1, device=device).squeeze()
        root_pose_6d = pose_latent = pose_feat = hyper_ctx = pred_x0 = None

        if self.refiner == 'diffusion':
            if kp2d is None:
                raise ValueError("refiner='diffusion' can kp2d (va kp_conf).")
            if is_train and gt_pose_6d is not None:
                pred_x0, diff_loss = self.diffusion(
                    gt_pose_6d, kp2d, kp_conf, is_train=True, valid_mask=pose_valid_mask,
                )
                diff_loss = diff_loss.mean()
            else:
                pred_x0 = self.diffusion(None, kp2d, kp_conf, is_train=False)
            full_pose = rot6d_to_axis_angle(pred_x0.reshape(-1, 6)).reshape(B, -1)
        else:
            # query: mean_pose (24, 6) -> (B, 24, dim)
            pose_token = self.pose_embed(self.init_pose.expand(B, -1, -1))
            if self.pose_pe is not None:
                pose_token = pose_token + self.pose_pe.weight.unsqueeze(0)

            idx = torch.arange(self.num_joint, device=device)
            ctx = self.norm(joint_out) + self.node_pe(idx)
            ctx = self.spatial_hypers(ctx.unsqueeze(1)).squeeze(1)       # (B, J, dim)
            hyper_ctx = ctx

            pose_feat = self.pose_context_attn(pose_token, ctx,ctx)     # (B, 24, dim)

            root_pose_6d = self.root_pose_head(pose_feat[:, 0, :])
            root_pose = rot6d_to_axis_angle(root_pose_6d)
            pose_latent = self.body_pose_head(pose_feat.mean(dim=1))
            body_pose = self.vposer(pose_latent)
            full_pose = torch.cat([root_pose, body_pose], dim=1)

        # 3. Camera
        cam_trans = self.get_camera_trans(self.cam_head(global_ft))

        # 4. Shape (init tu mean_shape)
        shape_emb = self.shape_embed(self.init_shape.expand(B, -1)).unsqueeze(1)   # (B, 1, dim)
        shape_token = self.shape_token.weight.unsqueeze(0).expand(B, 1, -1) + shape_emb
        global_seq = global_ft.unsqueeze(1)
        shape_out = self.fuse_shape(shape_token, global_seq, global_seq)
        shape_param = self.shape_head(shape_out).reshape(B, -1)

        # 5. SMPL
        joint_proj, joint_cam, mesh_cam, mesh_cam_render = self.get_coord(
            full_pose, shape_param, cam_trans
        )
        if is_train and getattr(cfg.LOSS, 'DETACH_POSE_FOR_PROJ', True):
            joint_proj_det, _, mesh_cam_proj, _ = self.get_coord(full_pose.detach(), shape_param, cam_trans)
        else:
            joint_proj_det, mesh_cam_proj = joint_proj, mesh_cam

        result = {
            # --- dung cho loss voi GT ---
            'joint_proj': joint_proj,                  # (B, K, 2) 2D, don vi output_hm_shape
            'joint_proj_det': joint_proj_det,          # nhu tren nhung pose bi detach neu DETACH_POSE_FOR_PROJ (dung cho loss chieu 2D)
            'joint_cam': joint_cam,                    # (B, K, 3) root-relative
            'smpl_mesh_cam': mesh_cam,                 # (B, 6890, 3) root-relative
            'smpl_mesh_cam_proj': mesh_cam_proj,
            'smpl_mesh_cam_render': mesh_cam_render,   # (B, 6890, 3) tuyet doi
            'smpl_pose': full_pose,
            'smpl_shape': shape_param,
            'cam_param': cam_trans,
            'root_pose_6d': root_pose_6d,              # (B, 6)  hoac None (diffusion)
            'pose_latent': pose_latent,                # (B, 32) hoac None (diffusion)
        }
        if self.refiner == 'diffusion':
            result['diff_loss'] = diff_loss
            result['pred_pose_6d_refined'] = pred_x0
        if self.use_img_probe:
            result['img_probe_joints'] = img_probe_joints          # (B, J, 3) chi tu anh

        # --- dung cho KD ---
        if return_features:
            result.update({
                'feat_joint': joint_out,               # (B, J, dim)
                'feat_img': fus['img_out'],            # (B, H*W, dim)
                'feat_joint_in': fus['joint_tok'],     # (B, J, dim)
                'feat_global': global_ft,              # (B, 2*dim)
                'feat_layers': fus['layers'],          # list dict: img, joint, attn_joint_to_img, attn_img_to_joint
                'feat_hyper': hyper_ctx,               # (B, J, dim) hoac None
                'feat_pose': pose_feat,                # (B, 24, dim) hoac None
            })
        return result

    # --------------------------------------------------------
    def get_camera_trans(self, cam_param):
        """cam_param (B,3)=[tx, ty, gamma] -> cam_trans (B,3)=[tx, ty, tz]."""
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])
        k_value = math.sqrt(
            cfg.DATASET.focal[0] * cfg.DATASET.focal[1]
            * cfg.DATASET.camera_3d_size * cfg.DATASET.camera_3d_size
            / (cfg.input_img_shape[0] * cfg.input_img_shape[1])
        )
        t_z = k_value * gamma
        return torch.cat([t_xy, t_z[:, None]], dim=1)

    def get_coord(self, smpl_pose, smpl_shape, smpl_trans):
        """Chay SMPL -> (joint_proj, joint_cam, mesh_cam root-relative, mesh_cam_render tuyet doi)."""
        batch_size = smpl_pose.shape[0]
        mesh_cam, _ = self.human_model_layer(smpl_pose, smpl_shape, smpl_trans)

        joint_cam = torch.bmm(
            self.joint_regressor_t[None, :, :].repeat(batch_size, 1, 1), mesh_cam
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
        if self._frozen:
            mode = False          # model da freeze thi luon o eval
        super().train(mode)
        self.vposer.eval()
        return self


def freeze_model(model):
    """Dong bang toan bo (dung cho teacher khi KD): khong grad, luon eval du goi .train()."""
    for p in model.parameters():
        p.requires_grad = False
    model._frozen = True
    model.eval()
    return model


# ============================================================
# Teacher / Student
# ============================================================
class Teacher(FusionPose2Mesh):
    """Anh + GT joint 3D."""
    def __init__(self, num_joint=17, embed_dim=512, depth=3, **kwargs):
        super().__init__(num_joint=num_joint, embed_dim=embed_dim,
                         joint_encoder='gt', depth=depth, **kwargs)


class Student(FusionPose2Mesh):
    """Anh + joint 3D nhieu."""
    def __init__(self, num_joint=17, embed_dim=512, depth=3, **kwargs):
        super().__init__(num_joint=num_joint, embed_dim=embed_dim,
                         joint_encoder='noisy', depth=depth, **kwargs)


def get_model(num_joint, embed_dim, mode='teacher', depth=3, **kwargs):
    if mode == 'teacher':
        return Teacher(num_joint, embed_dim, depth, **kwargs)
    if mode == 'student':
        return Student(num_joint, embed_dim, depth, **kwargs)
    raise ValueError(f"mode khong hop le: {mode}")