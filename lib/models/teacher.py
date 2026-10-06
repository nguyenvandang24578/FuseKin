import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
from models.backbones.mesh import Mesh
from models.spin import RegressorSpin
import numpy as np
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp, Attention
from functools import partial


BASE_DATA_DIR = cfg.DATASET.BASE_DATA_DIR
SMPL_MODEL_DIR = 'data_final/base_data'
SMPL_MEAN_PARAMS = 'data_final/base_data/smpl_mean_params.npz'
BASE_DATA_DIR = 'data_final/base_data'
class StudentJointExtractor(nn.Module):
    def __init__(self, in_dim=3, out_dim=512):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        # Thêm 1 layer Transformer hoặc GCN để các khớp trao đổi thông tin & tự sửa nhiễu
        self.denoise_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=8, dim_feedforward=1024, batch_first=True
        )
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, noisy_joints, pos_emb):
        # noisy_joints: (B, 17, 3)
        x = self.proj(noisy_joints) + pos_emb
        x = self.denoise_layer(x)  # Tự căn chỉnh và khử nhiễu giữa 17 khớp
        return self.norm(x)
class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out

class RGBJointCrossTransformerBlock(nn.Module):
    """
    rgb tokens:   (B, H*W, C)   -- Q, K, V riêng
    joint tokens: (B, 17, C)    -- Q, K, V riêng
    Chuẩn cross-attention 2 chiều, sử dụng LearnableCoefficient cho residual và Pre-LN.
    """
    def __init__(self, d_model, h, block_exp=4, resid_pdrop=0.1):
        super().__init__()
        self.h = h
        self.d_k = d_model // h

        self.q_joint = nn.Linear(d_model, d_model)
        self.k_rgb   = nn.Linear(d_model, d_model)
        self.v_rgb   = nn.Linear(d_model, d_model)

        self.q_rgb   = nn.Linear(d_model, d_model)
        self.k_joint = nn.Linear(d_model, d_model)
        self.v_joint = nn.Linear(d_model, d_model)

        self.out_joint = nn.Linear(d_model, d_model)
        self.out_rgb   = nn.Linear(d_model, d_model)

        self.coef1 = LearnableCoefficient()
        self.coef2 = LearnableCoefficient()
        self.coef3 = LearnableCoefficient()
        self.coef4 = LearnableCoefficient()
        self.coef5 = LearnableCoefficient()
        self.coef6 = LearnableCoefficient()
        self.coef7 = LearnableCoefficient()
        self.coef8 = LearnableCoefficient()

        self.ln_joint1 = nn.LayerNorm(d_model)
        self.ln_rgb1   = nn.LayerNorm(d_model)
        self.ln_joint2 = nn.LayerNorm(d_model)
        self.ln_rgb2   = nn.LayerNorm(d_model)

        self.mlp_joint = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.GELU(),
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop)
        )
        self.mlp_rgb = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.GELU(),
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop)
        )

    def _split_heads(self, x, b):
        n = x.shape[1]
        return x.view(b, n, self.h, self.d_k).transpose(1, 2)  # (b,h,n,d_k)

    def forward(self, rgb_tok, joint_tok, return_intermediate=False):
        b = rgb_tok.shape[0]
        intermediate = []

        rgb_norm = self.ln_rgb1(rgb_tok)
        joint_norm = self.ln_joint1(joint_tok)

        # Joint attend to RGB
        q_j = self._split_heads(self.q_joint(joint_norm), b)
        k_r = self._split_heads(self.k_rgb(rgb_norm), b)
        v_r = self._split_heads(self.v_rgb(rgb_norm), b)
        attn_j = torch.softmax(q_j @ k_r.transpose(-2, -1) / self.d_k**0.5, dim=-1)
        joint_att = (attn_j @ v_r).transpose(1, 2).reshape(b, joint_tok.shape[1], -1)
        joint_att = self.out_joint(joint_att)

        # RGB attend to Joint
        q_r = self._split_heads(self.q_rgb(rgb_norm), b)
        k_j = self._split_heads(self.k_joint(joint_norm), b)
        v_j = self._split_heads(self.v_joint(joint_norm), b)
        attn_r = torch.softmax(q_r @ k_j.transpose(-2, -1) / self.d_k**0.5, dim=-1)
        rgb_att = (attn_r @ v_j).transpose(1, 2).reshape(b, rgb_tok.shape[1], -1)
        rgb_att = self.out_rgb(rgb_att)

        # Residuals 1
        joint_out = self.coef1(joint_tok) + self.coef2(joint_att)
        rgb_out = self.coef3(rgb_tok) + self.coef4(rgb_att)

        # FFN + Residuals 2
        joint_out = self.coef5(joint_out) + self.coef6(self.mlp_joint(self.ln_joint2(joint_out)))
        rgb_out = self.coef7(rgb_out) + self.coef8(self.mlp_rgb(self.ln_rgb2(rgb_out)))

        if return_intermediate:
            intermediate.append({
                'img': rgb_out,
                'joint': joint_out,
                'attn_joint_to_img': attn_j,
                'attn_img_to_joint': attn_r,
            })

        if return_intermediate:
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
            block_output = block(
                rgb_tok, joint_tok, return_intermediate=return_intermediate
            )
            if return_intermediate:
                rgb_tok, joint_tok, block_features = block_output
                intermediate.extend(block_features)
            else:
                rgb_tok, joint_tok = block_output
        if return_intermediate:
            return rgb_tok, joint_tok, intermediate
        return rgb_tok, joint_tok


class Teacher(nn.Module):
    def __init__(self, num_joint, embed_dim=512, vert_anchors=16, horz_anchors=16, in_channels=2048, depth=1, norm_layer = None, name = "teacher"):
        super(Teacher, self).__init__()

        self.mesh = Mesh()
        self.regressorspin = RegressorSpin()
        pretrained_dict = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'), weights_only=False)['model']
        self.regressorspin.load_state_dict(pretrained_dict, strict=False)
        self.name = name
        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        #--------------------------------------------------------------
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        
        # Project image features từ ResNet (2048) -> embed_dim (512)
        self.img_proj = nn.Conv2d(in_channels, embed_dim, 1)
        # Project joints (B, 17, 3) -> (B, 17, 512)
        if self.name == "student":
            self.projector_student = StudentJointExtractor(in_dim=3, out_dim=512)
        else:
            self.joint_proj = nn.Linear(3, embed_dim)
            self.norm_joint_proj = nn.LayerNorm(embed_dim)
        # Positional Embeddings
        self.pos_emb_img = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, embed_dim))
        self.pos_emb_joint = nn.Parameter(torch.zeros(1, 17, embed_dim))

        # Khởi tạo RGBJointCrossTransformer với chiều sâu (depth)
        self.cfcer = RGBJointCrossTransformer(
            d_model=embed_dim, 
            h=8, 
            block_exp=4, 
            resid_pdrop=0.1,
            depth=depth
        )

        # Final LayerNorm sau cross-transformer (dùng cho cả teacher và student)
        self.norm_img = nn.LayerNorm(embed_dim)
        self.norm_joint = nn.LayerNorm(embed_dim)
        
        # LayerNorm đầu vào cho img_tok để cân bằng scale với joints_tok
        self.norm_img_out = nn.LayerNorm(embed_dim)
        self.norm_joint_out = nn.LayerNorm(embed_dim)        
        
        self.norm_img_in = nn.LayerNorm(embed_dim)
        self.norm_joint_in = nn.LayerNorm(embed_dim)
        # Output projection cho regressorspin (nhận concat 2 vector 512 -> 1024)
        self.out_proj = nn.Linear(embed_dim * 2, 2048)
    def forward(self, joints, img_feats, is_train=True, J_regressor=None, return_features=False):
        # Chiếu img_feats từ 2048 kênh xuống 512 kênh
        img_feats = self.img_proj(img_feats)
        bs, c, h, w = img_feats.shape
        mean_pose  = self.init_pose.expand(
            bs, -1
        )          # (bs, 144)
        mean_shape  = self.init_shape.expand(bs, 10)  # (bs, 10)
        # 1. Project joints (B, 17, 3) -> (B, 17, 512)
        joints_tok = self.joint_proj(joints) + self.pos_emb_joint
        
        # 2. Image tokens (B, C, H, W) -> (B, H*W, 512)
        img_tok = img_feats.view(bs, c, -1).permute(0, 2, 1)
        
        # Xử lý pos_emb_img nếu H*W không khớp với kích thước mặc định (vert_anchors * horz_anchors)
        if h * w != self.pos_emb_img.shape[1]:
            pos_emb = self.pos_emb_img.permute(0, 2, 1).view(1, c, self.vert_anchors, self.horz_anchors)
            pos_emb = F.interpolate(pos_emb, size=(h, w), mode='bilinear', align_corners=False)
            pos_emb = pos_emb.view(1, c, -1).permute(0, 2, 1)
        else:
            pos_emb = self.pos_emb_img
            
        img_tok = img_tok + pos_emb
        # print(f"\n[{self.name.upper() if getattr(self, 'name', None) else 'TEACHER'} - DEBUG SCALE] Feature before CFCER:")
        # print(f"  -> img_tok    : Mean = {img_tok.mean().item():.4f}, Std = {img_tok.std().item():.4f}, Norm = {torch.norm(img_tok, dim=-1).mean().item():.4f}")
        # print(f"  -> joints_tok : Mean = {joints_tok.mean().item():.4f}, Std = {joints_tok.std().item():.4f}, Norm = {torch.norm(joints_tok, dim=-1).mean().item():.4f}\n")
        img_norm = self.norm_img_in(img_tok)
        joint_norm = self.norm_joint_in(joints_tok)
        # print(f"\n[{self.name.upper() if getattr(self, 'name', None) else 'TEACHER'} - DEBUG SCALE] Feature after Norm:")
        # print(f"  -> img_tok    : Mean = {img_norm.mean().item():.4f}, Std = {img_norm.std().item():.4f}, Norm = {torch.norm(img_norm, dim=-1).mean().item():.4f}")
        # print(f"  -> joints_tok : Mean = {joint_norm.mean().item():.4f}, Std = {joint_norm.std().item():.4f}, Norm = {torch.norm(joint_norm, dim=-1).mean().item():.4f}\n")
        # 3. Cross Attention fusion
        img_out, joint_out = self.cfcer(img_norm, joint_norm)
        
        # 4. Global Pooling & Fusion
        # Lấy trung bình dọc theo chiều token
        img_out = self.norm_img_out(img_out)
        joint_out = self.norm_joint_out(joint_out)
        # print(f"\n[{self.name.upper() if getattr(self, 'name', None) else 'TEACHER'} - DEBUG SCALE] Feature after CFCER:")
        # print(f"  -> img_tok    : Mean = {img_out.mean().item():.4f}, Std = {img_out.std().item():.4f}, Norm = {torch.norm(img_out, dim=-1).mean().item():.4f}")
        # print(f"  -> joints_tok : Mean = {joint_out.mean().item():.4f}, Std = {joint_out.std().item():.4f}, Norm = {torch.norm(joint_out, dim=-1).mean().item():.4f}\n")
        img_global = img_out.mean(dim=1) # (B, 512)
        joint_global = joint_out.mean(dim=1) # (B, 512)
        
        concat_feat = torch.cat([img_global, joint_global], dim=1) # (B, 1024)
        
        # 5. Projection to (B, 2048) cho regressorspin
        img_feats_trans = self.out_proj(concat_feat) # (B, 2048)
        img_feats_trans = img_feats_trans.unsqueeze(1) # (B, 1, 2048)

        output = self.regressorspin(img_feats_trans, init_pose = mean_pose, init_shape = mean_shape, is_train=is_train, J_regressor=J_regressor)
        
        # Trích xuất kết quả từ SPIN để cung cấp khởi tạo cho Pose2Mesh
        spin_out = output[0]
        pose_6d = spin_out['pose_6d'].squeeze(1) # (B, 24, 6)
        shape = spin_out['shape'].squeeze(1)     # (B, 10)
        cam = spin_out['cam'].squeeze(1)         # (B, 3)
        
        if return_features:
            return spin_out, pose_6d, shape, cam, {
                'joint_out': joint_out,      # (B, 17, C) - per-joint tokens
                'img_out': img_out,          # (B, H*W, C) - unpooled image tokens
                'concat_feat': concat_feat,  # (B, 1024) - global pooled feature
            }
        return pose_6d, shape, cam
# ============================================================
# Factory
# ============================================================
def get_model(num_joint, embed_dim, vert_anchors=16, horz_anchors=16, depth=1):
    model = Teacher(num_joint, embed_dim, vert_anchors, horz_anchors, depth=depth)
    return model