import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
from models.backbones.mesh import Mesh
from models.spin import RegressorSpin

BASE_DATA_DIR = cfg.DATASET.BASE_DATA_DIR
class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out

class RGBJointCrossTransformer(nn.Module):
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

    def forward(self, rgb_tok, joint_tok):
        b = rgb_tok.shape[0]

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

        return rgb_out, joint_out


class Teacher(nn.Module):
    def __init__(self, num_joint, embed_dim=512, vert_anchors=16, horz_anchors=16, in_channels=2048):
        super(Teacher, self).__init__()

        self.mesh = Mesh()
        self.regressorspin = RegressorSpin()
        pretrained_dict = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))['model']
        self.regressorspin.load_state_dict(pretrained_dict, strict=False)
        #--------------------------------------------------------------
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        
        # Project image features từ ResNet (2048) -> embed_dim (512)
        self.img_proj = nn.Conv2d(in_channels, embed_dim, 1)

        # Project joints (B, 17, 3) -> (B, 17, 512)
        self.joint_proj = nn.Linear(3, embed_dim)

        # Positional Embeddings
        self.pos_emb_img = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, embed_dim))
        self.pos_emb_joint = nn.Parameter(torch.zeros(1, 17, embed_dim))

        # Khởi tạo RGBJointCrossTransformer mới
        self.cfcer = RGBJointCrossTransformer(
            d_model=embed_dim, 
            h=8, 
            block_exp=4, 
            resid_pdrop=0.1
        )
        
        # Output projection cho regressorspin (nhận concat 2 vector 512 -> 1024)
        self.out_proj = nn.Linear(embed_dim * 2, 2048)
    def forward(self, joints, img_feats, is_train=True, J_regressor=None):
        # Chiếu img_feats từ 2048 kênh xuống 512 kênh
        img_feats = self.img_proj(img_feats)
        bs, c, h, w = img_feats.shape
        
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
        
        # 3. Cross Attention fusion
        img_out, joint_out = self.cfcer(img_tok, joints_tok)
        
        # 4. Global Pooling & Fusion
        # Lấy trung bình dọc theo chiều token
        img_global = img_out.mean(dim=1) # (B, 512)
        joint_global = joint_out.mean(dim=1) # (B, 512)
        
        concat_feat = torch.cat([img_global, joint_global], dim=1) # (B, 1024)
        
        # 5. Projection to (B, 2048) cho regressorspin
        img_feats_trans = self.out_proj(concat_feat) # (B, 2048)
        img_feats_trans = img_feats_trans.unsqueeze(1) # (B, 1, 2048)

        output = self.regressorspin(img_feats_trans, is_train=is_train, J_regressor=J_regressor)
        
        return output
# ============================================================
# Factory
# ============================================================
def get_model(num_joint, embed_dim, vert_anchors=16, horz_anchors=16):
    model = Teacher(num_joint, embed_dim, vert_anchors, horz_anchors)
    return model