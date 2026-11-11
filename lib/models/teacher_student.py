"""
Student FuseKin (phien ban chinh thuc duy nhat tu 2026-11-11, truoc day goi la "v7 anchor"):

    anh (F8) + pose 3D lifter (+ PE 2D) -> Fusion (cross-attn 2 chieu) -> HyperGCN (khoa ma tran)
      -> ctx (17 khop) + token sieu canh (part_tok 10 doan chi, body_tok 5 chi / than)
      -> pose than = pose 3D lifter (KHONG sua); J29 = M x pose3d + delta 7 diem dau chi
      -> twist khop 1..23 (query SMPL + part_tok); beta = beta_xuong (Gauss-Newton, khong hoc) + dbeta
      -> camera -> IK HybrIK -> SMPL -> mesh

- Ban cu (Teacher, KD, v3 [B][C][D][E], v5, v6, nhanh goc 6D) da bo; code cu o _to_delete/bak_cleanup_v7_20261111/.
- Ten module / tham so GIU NGUYEN nhu ban v7 -> checkpoint v7 cu nap duoc (key per_joint_head / shape_head bi bo qua).
- Cac key ablation (cfg.MODEL.v7_*) van giu: v7_part_readout, v7_twist, v7_delta_limb, v7_dbeta, v7_delta17, v7_no_hgcn.
- Model KHONG tinh loss; moi thu can cho loss duoc tra ve trong dict output.
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
from models.HGraph import create_layers
from models.Core_model import CrossAttentionBlock

SMPL_MEAN_PARAMS = 'data_final/base_data/smpl_mean_params.npz'

# Module da bo: checkpoint cu con cac key nay -> bo qua khi nap (base.load_model_weights).
LEGACY_KEY_PARTS = ('vposer.', 'root_pose_head.', 'body_pose_head.', 'fuse_shape.', 'hint_proj.', 'hint_encoder.',
                    'hint_pred.', 'diffusion.', 'per_joint_head.', 'shape_head.')

# Cap xuong (cha, con) theo chi so H36M-17 cho moi khop SMPL-24: query cua khop j doc dac trung cua xuong do.
# H36M-17: 0 Pelvis, 1 R_Hip, 2 R_Knee, 3 R_Ankle, 4 L_Hip, 5 L_Knee, 6 L_Ankle, 7 Torso, 8 Neck, 9 Nose,
#          10 Head_top, 11 L_Shoulder, 12 L_Elbow, 13 L_Wrist, 14 R_Shoulder, 15 R_Elbow, 16 R_Wrist.
# Khop khong co xuong trong H36M (ban chan, co tay, ban tay) -> (chinh khop do, chinh khop do).
SMPL_BONE_PAIRS_H36M = (
    (4, 1),    # 0  Pelvis (root): L_Hip, R_Hip
    (4, 5),    # 1  L_Hip      -> L_Knee
    (1, 2),    # 2  R_Hip      -> R_Knee
    (0, 7),    # 3  Spine1     : Pelvis -> Torso
    (5, 6),    # 4  L_Knee     -> L_Ankle
    (2, 3),    # 5  R_Knee     -> R_Ankle
    (7, 8),    # 6  Spine2     : Torso -> Neck
    (6, 6),    # 7  L_Ankle
    (3, 3),    # 8  R_Ankle
    (7, 8),    # 9  Spine3     : Torso -> Neck
    (6, 6),    # 10 L_Foot
    (3, 3),    # 11 R_Foot
    (8, 10),   # 12 Neck       -> Head_top
    (8, 11),   # 13 L_Collar   : Neck -> L_Shoulder
    (8, 14),   # 14 R_Collar   : Neck -> R_Shoulder
    (9, 10),   # 15 Head       : Nose, Head_top
    (11, 12),  # 16 L_Shoulder -> L_Elbow
    (14, 15),  # 17 R_Shoulder -> R_Elbow
    (12, 13),  # 18 L_Elbow    -> L_Wrist
    (15, 16),  # 19 R_Elbow    -> R_Wrist
    (13, 13),  # 20 L_Wrist
    (16, 16),  # 21 R_Wrist
    (13, 13),  # 22 L_Hand (dau ra = 0)
    (16, 16),  # 23 R_Hand (dau ra = 0)
)

# Sieu canh (CUNG nhom khop voi hypergraph.py, dataset 'h36m') va anh xa giai ma theo bo phan
V7_PARTS = [[0, 7, 8], [8, 9, 10], [0, 1, 2], [2, 3], [0, 4, 5], [5, 6], [8, 14, 15], [8, 11, 12], [15, 16], [12, 13]]
V7_BODY = [[0, 1, 4, 7, 8, 9, 10, 11, 14], [14, 15, 16], [11, 12, 13], [1, 2, 3], [4, 5, 6]]   # than, tay P, tay T, chan P, chan T
# Twist khop SMPL 1..23 (xuong khop -> con) -> doan chi chua xuong do (chi so trong V7_PARTS)
V7_TWIST_PART = [4, 2, 0, 5, 3, 0, 5, 3, 0, 5, 3, 1, 7, 6, 1, 7, 6, 9, 8, 9, 8, 9, 8]
# 7 diem dau chi trong 29 diem HybrIK co delta: ban chan T/P, ban tay T/P, dinh dau, ngon chan cai T/P.
# (25, 26 giua ban tay bo: goc khop 22, 23 bi dat 0 nen khong anh huong mesh)
V7_LIMB_IDX = [10, 11, 22, 23, 24, 27, 28]
V7_LIMB_BODY = [4, 3, 2, 1, 0, 4, 3]          # chi / than chua diem do (chi so trong V7_BODY)
V7_LIMB_TOKEN = [10, 11, 22, 23, 15, 10, 11]  # token query SMPL dung (diem la -> token khop cha)


# ============================================================
# Fusion: token khop (pose 3D lifter) <-> token anh
# ============================================================
class StudentJointExtractor(nn.Module):
    """pose 3D (B, J, 3) -> token khop: Linear + PE + 1 lop Transformer (cac khop trao doi thong tin)."""
    def __init__(self, in_dim=3, out_dim=512):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.denoise_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=8, dim_feedforward=1024, batch_first=True
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, joints, pos_emb, pe2d=None):
        # pe2d (B, J, C) | None: PE vi tri 2D cua khop, cong TRUOC lop Transformer
        x = self.proj(joints) + pos_emb
        if pe2d is not None:
            x = x + pe2d
        x = self.denoise_layer(x)
        return self.norm(x)


class PosEnc2D(nn.Module):
    """PE vi tri 2D Fourier, DUNG CHUNG cho token khop (theo toa do 2D) va o anh (theo tam o) -> cung khong gian vi tri.
    (x, y) trong [-1, 1] -> [sin(2^k*pi*x), cos(2^k*pi*x), sin(2^k*pi*y), cos(2^k*pi*y)], k = 0..L-1 -> Linear -> C.
    Linear khoi tao 0 (khong bias) -> luc dau PE = 0."""
    def __init__(self, embed_dim=512, num_freq=6):
        super().__init__()
        self.register_buffer('freqs', (2.0 ** torch.arange(num_freq).float()) * math.pi, persistent=False)
        self.proj = nn.Linear(4 * num_freq, embed_dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, xy):
        ang = xy.unsqueeze(-1) * self.freqs.to(xy.dtype)              # (..., 2, L)
        feat = torch.cat([ang.sin(), ang.cos()], dim=-1)              # (..., 2, 2L)
        return self.proj(feat.flatten(-2))


class LearnableCoefficient(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        return x * self.bias


class RGBJointCrossTransformerBlock(nn.Module):
    """rgb tokens (B, H*W, C) <-> joint tokens (B, J, C): self-attn khop -> cross-attn 2 chieu -> FFN.
    Pre-LN, residual dung LearnableCoefficient."""
    def __init__(self, d_model, h, block_exp=4, resid_pdrop=0.1):
        super().__init__()
        self.h = h
        self.d_k = d_model // h

        # Self-attn giua cac token khop TRUOC cross-attn; out_proj khoi tao 0
        self.ln_joint_sa = nn.LayerNorm(d_model)
        self.joint_sa = nn.MultiheadAttention(d_model, h, dropout=0.0, batch_first=True)
        nn.init.zeros_(self.joint_sa.out_proj.weight)
        nn.init.zeros_(self.joint_sa.out_proj.bias)

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

        # 0. Self-attn giua cac khop: cac khop biet nhau truoc khi hoi anh
        jn = self.ln_joint_sa(joint_tok)
        sa, _ = self.joint_sa(jn, jn, jn, need_weights=False)
        joint_tok = joint_tok + sa

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
        self.h = h
        self.blocks = nn.ModuleList([
            RGBJointCrossTransformerBlock(d_model, h, block_exp, resid_pdrop) for _ in range(depth)
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
    """img_feats (B,2048,H,W) + joints (B,J,3) + 2D (PE) -> token anh, token khop, feature toan cuc."""
    def __init__(self, num_joint=17, embed_dim=512, vert_anchors=8, horz_anchors=8,
                 in_channels=2048, depth=3, pe2d_num_freq=6):
        super().__init__()
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        # PE vi tri 2D dung chung cho khop va o anh. emb_missing: khop mask = 0. Ca hai khoi tao 0.
        self.pe2d = PosEnc2D(embed_dim=embed_dim, num_freq=pe2d_num_freq)
        self.emb_missing = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.img_proj = nn.Conv2d(in_channels, embed_dim, 1)
        self.projector_student = StudentJointExtractor(in_dim=3, out_dim=embed_dim)

        self.pos_emb_img = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, embed_dim))
        self.pos_emb_joint = nn.Parameter(torch.zeros(1, num_joint, embed_dim))

        self.cfcer = RGBJointCrossTransformer(d_model=embed_dim, h=8, block_exp=4, resid_pdrop=0.1, depth=depth)
        self.norm_img_in = nn.LayerNorm(embed_dim)

    @staticmethod
    def _cell_centers(h, w, device, dtype):
        """Tam o luoi h x w trong [-1, 1], cung quy uoc chuan hoa 2D cua wrapper (x / W * 2 - 1).
        Thu tu giong img_feats.flatten(2): chi so = hang * w + cot -> (1, h*w, 2) dang (x, y)."""
        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h * 2 - 1
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w * 2 - 1
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([gx, gy], dim=-1).reshape(1, h * w, 2)

    def forward(self, joints, img_feats, joints_2d, joints_2d_mask=None, return_features=False):
        """joints (B, J, 3) met root-relative; joints_2d (B, J, 2) trong [-1, 1]; joints_2d_mask (B, J, 1) | (B, J)."""
        img_feats = self.img_proj(img_feats)               # (B, C, H, W)
        bs, c, h, w = img_feats.shape

        assert joints_2d is not None, 'Fusion can joints_2d (B, J, 2) cho PE vi tri 2D'
        if joints_2d_mask is None:
            m = torch.ones_like(joints_2d[..., :1])
        else:
            m = joints_2d_mask.to(joints_2d.dtype)
            if m.dim() == 2:
                m = m.unsqueeze(-1)
            m = (m > 0).to(joints_2d.dtype)
        pe_joint = m * self.pe2d(joints_2d[..., :2]) + (1.0 - m) * self.emb_missing   # (B, J, C)

        joints_tok = self.projector_student(joints, self.pos_emb_joint, pe2d=pe_joint)
        joints_tok_in = joints_tok                          # token khop sau encoder, truoc cross-attn

        img_tok = img_feats.flatten(2).transpose(1, 2)     # (B, H*W, C)
        if h * w != self.pos_emb_img.shape[1]:
            pos_emb = self.pos_emb_img.transpose(1, 2).reshape(1, c, self.vert_anchors, self.horz_anchors)
            pos_emb = F.interpolate(pos_emb, size=(h, w), mode='bilinear', align_corners=False)
            pos_emb = pos_emb.flatten(2).transpose(1, 2)
        else:
            pos_emb = self.pos_emb_img
        # Cung module PE voi khop -> vi tri o anh va vi tri khop nam cung khong gian
        pos_emb = pos_emb + self.pe2d(self._cell_centers(h, w, img_tok.device, img_tok.dtype))
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
            'joint_tok': joints_tok_in,  # (B, J, C)
            'img_tok': img_tok,          # (B, H*W, C) token anh dau vao fusion
            'concat_feat': concat_feat,  # (B, 2C)
            'layers': layers,
        }


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
# Student
# ============================================================
class Student(nn.Module):
    def __init__(self, num_joint=17, embed_dim=512, vert_anchors=8, horz_anchors=8, depth=3,
                 smpl_head_hidden_dim=256, pe2d_num_freq=6, hgcn_freeze_adj=False):
        super().__init__()
        assert num_joint == 17, 'Query neo xuong can 17 khop H36M (bang SMPL_BONE_PAIRS_H36M)'
        self.num_joint = num_joint
        self.embed_dim = embed_dim

        # ---- SMPL ----
        from utils.smpl import SMPL as SMPLModel
        self.human_model = SMPLModel()
        self.human_model_layer = self.human_model.layer['neutral']
        self.joint_regressor = self.human_model.joint_regressor
        self.register_buffer('joint_regressor_t', torch.from_numpy(self.joint_regressor).float())

        # ---- mean pose / shape (khoi tao query) ----
        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:].astype('float32')).view(1, 24, 6)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).view(1, 10)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)

        # ---- Fusion ----
        self.fusion = RGBJointFusion(num_joint=num_joint, embed_dim=embed_dim, vert_anchors=vert_anchors,
                                     horz_anchors=horz_anchors, depth=depth, pe2d_num_freq=pe2d_num_freq)
        global_dim = embed_dim * 2

        # ---- Pose path (HyperGCN, giong HyperDiff) ----
        self.pose_embed = nn.Linear(6, embed_dim)
        self.pose_pe = nn.Embedding(24, embed_dim)
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
        # Neo query vao xuong: query_j += Linear(concat(ctx[cha_j], ctx[con_j])), khoi tao 0
        self.register_buffer('bone_pairs', torch.tensor(SMPL_BONE_PAIRS_H36M, dtype=torch.long),
                             persistent=False)                                   # (24, 2)
        self.bone_proj = nn.Linear(2 * embed_dim, embed_dim)
        nn.init.zeros_(self.bone_proj.weight)
        nn.init.zeros_(self.bone_proj.bias)

        # ---- Shape token: cross-attn [ctx (17 khop), token anh] -> dau vao head dbeta ----
        self.shape_embed = nn.Linear(10, embed_dim)
        self.shape_token = nn.Embedding(1, embed_dim)
        self.fuse_shape_tok = CrossAttentionBlock(
            q_dim=embed_dim, k_dim=embed_dim, v_dim=embed_dim,
            kv_num=num_joint + vert_anchors * horz_anchors,
            num_heads=8, mlp_ratio=4., qkv_bias=True,
            drop=0., attn_drop=0., drop_path=0.2, has_mlp=True,
        )

        # ---- Camera ----
        self.cam_head = MLP(global_dim, smpl_head_hidden_dim, 3, 2)

        # ---- HyperGCN tren 24 token query SMPL (MODEL.smpl_hgcn, 2026-11-12): sau pose_context_attn, truoc head be mat ----
        self.smpl_hypers = None
        if cfg.MODEL.get('smpl_hgcn', False):
            from models.hypergraph_smpl import create_smpl_layers
            self.smpl_hypers = create_smpl_layers(embed_dim, n_layers=int(cfg.MODEL.smpl_hgcn_layers))
            print(f'===> HyperGCN SMPL-24 tren token query: {int(cfg.MODEL.smpl_hgcn_layers)} lop')

        # ---- Khoa ma tran sieu canh HyperGCN (MODEL.hgcn_freeze_adj): giu ma tran giai phau khoi tao ----
        if hgcn_freeze_adj:
            for tag, mod in (('H36M-17', self.spatial_hypers), ('SMPL-24', self.smpl_hypers)):
                if mod is None:
                    continue
                n_frz = 0
                for name, p in mod.named_parameters():
                    if name.endswith('p_adj') or name.endswith('b_adj'):
                        p.requires_grad_(False)
                        n_frz += 1
                print(f'===> HyperGCN {tag}: khoa {n_frz} ma tran sieu canh (p_adj / b_adj)')

        # ---- IK HybrIK + anh xa M (H36M-17 -> 29 diem HybrIK) ----
        from models.hybrik_smpl import HybrIKSMPL
        reg_path = os.path.join(cfg.DATASET.jotr_data_root, 'Human36M', 'J_regressor_h36m_correct.npy')
        self.hybrik = HybrIKSMPL(self.human_model_layer, np.load(reg_path))
        Mmap = np.load(cfg.MODEL.v6_m_path).astype(np.float32)
        assert Mmap.shape == (29, 17), f'v6_m_path phai la ma tran (29, 17), nhan {Mmap.shape}'
        self.register_buffer('v7_M', torch.from_numpy(Mmap))           # luu trong ckpt
        # Ma tran thuoc (khop x sieu canh) chuan hoa theo bac sieu canh: CUNG nhom khop voi hypergraph.py (h36m)
        hp = torch.zeros(num_joint, len(V7_PARTS))
        for e, js in enumerate(V7_PARTS):
            hp[js, e] = 1.0
        hb = torch.zeros(num_joint, len(V7_BODY))
        for e, js in enumerate(V7_BODY):
            hb[js, e] = 1.0
        self.register_buffer('v7_Hp', hp / hp.sum(0, keepdim=True), persistent=False)   # (17, 10)
        self.register_buffer('v7_Hb', hb / hb.sum(0, keepdim=True), persistent=False)   # (17, 5)

        # ---- Head be mat (moi head khoi tao 0 -> luc dau = "lifter + beta_xuong + IK") ----
        self.v7_readout = bool(cfg.MODEL.v7_part_readout)
        self.v7_no_hgcn = bool(cfg.MODEL.v7_no_hgcn)                    # CHI ablation
        fin2 = 2 * embed_dim if self.v7_readout else embed_dim
        if self.v7_readout:
            self.v7_part_norm = nn.LayerNorm(embed_dim)
            self.v7_body_norm = nn.LayerNorm(embed_dim)
        self.v7_tw = nn.Linear(fin2, 2) if cfg.MODEL.v7_twist else None          # twist (cos, sin) += ...
        self.v7_dlimb = nn.Linear(fin2, 3) if cfg.MODEL.v7_delta_limb else None   # delta 7 diem dau chi
        self.v7_db = nn.Linear(fin2, 10) if cfg.MODEL.v7_dbeta else None         # dbeta
        self.v7_d17 = nn.Linear(embed_dim, 3) if cfg.MODEL.v7_delta17 else None  # CHI ablation: sua pose than
        for m in (self.v7_tw, self.v7_dlimb, self.v7_db, self.v7_d17):
            if m is not None:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        print(f'===> Student: readout {self.v7_readout} | twist {cfg.MODEL.v7_twist} '
              f'dlimb {cfg.MODEL.v7_delta_limb} dbeta {cfg.MODEL.v7_dbeta} delta17 {cfg.MODEL.v7_delta17} '
              f'no_hgcn {self.v7_no_hgcn} | beta_xuong {cfg.MODEL.v7_bone_gn_iters} buoc, '
              f'lambda {cfg.MODEL.v7_bone_beta_reg}')

    # --------------------------------------------------------
    def _mesh_outputs(self, joints, ctx, pose_feat, shape_out, cam_raw, is_train):
        """Pose than = pose3d (khong sua) -> J29 = M x pose17 (+ delta 7 diem dau chi) ; beta = beta_xuong + dbeta ;
        twist tu part_tok -> IK -> goc SMPL -> SMPL. Tra (result, ctx3d | None)."""
        from models.hybrik_smpl import rotmat_to_aa_safe
        B = joints.shape[0]
        lift = joints - joints[:, 0:1]
        pose17, ctx3d = lift, None
        if self.v7_d17 is not None:                                                   # CHI ablation
            pose17 = lift + self.v7_d17(ctx)
            pose17 = pose17 - pose17[:, 0:1]
            ctx3d = pose17
        j29 = torch.einsum('jk,bkc->bjc', self.v7_M, pose17)                          # (B, 29, 3)
        part_tok = body_tok = None
        if self.v7_readout:                                                           # token sieu canh sau HyperGCN
            part_tok = self.v7_part_norm(torch.einsum('jp,bjc->bpc', self.v7_Hp, ctx))   # (B, 10, C)
            body_tok = self.v7_body_norm(torch.einsum('jp,bjc->bpc', self.v7_Hb, ctx))   # (B, 5, C)
        # delta 7 diem dau chi (sua loi anh xa M o ban tay / ban chan / dinh dau)
        if self.v7_dlimb is not None:
            q = pose_feat[:, V7_LIMB_TOKEN]                                           # (B, 7, C)
            inp = torch.cat([q, body_tok[:, V7_LIMB_BODY]], -1) if self.v7_readout else q
            dlimb = self.v7_dlimb(inp)                                                # (B, 7, 3)
            d29 = torch.zeros_like(j29)
            d29[:, V7_LIMB_IDX] = dlimb
            j29 = j29 + d29
        # beta = beta tu do dai xuong cua pose3d (hinh hoc, khong hoc) + dbeta (anh)
        with torch.no_grad():
            beta0 = self.hybrik.beta_from_bones(pose17.detach().float(), int(cfg.MODEL.v7_bone_gn_iters),
                                                float(cfg.MODEL.v7_bone_beta_reg))
        dbeta = torch.zeros_like(beta0)
        if self.v7_db is not None:
            s = shape_out[:, 0]                                                       # (B, C)
            inp = torch.cat([s, body_tok.mean(1)], -1) if self.v7_readout else s
            dbeta = self.v7_db(inp).float()
        beta = beta0 + dbeta
        # twist: khop SMPL 1..23 doc token doan chi chua xuong cua no
        phis = self.hybrik.default_phis(B, j29.device, torch.float32)
        if self.v7_tw is not None:
            q = pose_feat[:, 1:24]
            inp = torch.cat([q, part_tok[:, V7_TWIST_PART]], -1) if self.v7_readout else q
            phis = phis + self.v7_tw(inp).float()
        rot_mats = self.hybrik.ik(j29.float(), beta, phis)                            # (B, 24, 3, 3)
        full_pose = rotmat_to_aa_safe(rot_mats.reshape(-1, 3, 3)).reshape(B, 24, 3)
        full_pose = torch.cat([full_pose[:, :22], torch.zeros_like(full_pose[:, 22:])], dim=1).reshape(B, 72)
        result = self._smpl_outputs(full_pose, beta, cam_raw, is_train)
        result['root_pose_6d'] = rot_mats[:, 0, :, :2].reshape(B, 6)
        result['v7_j29'] = j29                                                        # loss J29 chi 7 diem dau chi
        result['v7_dbeta'] = dbeta                                                    # rang buoc |dbeta|^2
        result['v7_beta0'] = beta0
        return result, ctx3d

    def _smpl_outputs(self, full_pose, shape_param, cam_raw, is_train):
        """Goc axis-angle (B, 72) + beta (B, 10) + camera tho (B, 3) -> dict output SMPL."""
        cam_trans = self.get_camera_trans(cam_raw)
        joint_proj, joint_cam, mesh_cam, mesh_cam_render = self.get_coord(full_pose, shape_param, cam_trans)
        if is_train:
            # Loss chieu 2D: detach pose (giu shape, camera)
            _, _, _, mesh_render_det = self.get_coord(full_pose.detach(), shape_param, cam_trans)
        else:
            mesh_render_det = mesh_cam_render
        return {
            'joint_proj': joint_proj,                  # (B, K, 2) 2D SMPL-30, don vi output_hm_shape
            'joint_cam': joint_cam,                    # (B, K, 3) root-relative (pelvis SMPL)
            'smpl_mesh_cam': mesh_cam,                 # (B, 6890, 3) tru pelvis SMPL
            'smpl_mesh_cam_render': mesh_cam_render,   # (B, 6890, 3) tuyet doi
            'smpl_mesh_cam_render_det': mesh_render_det,  # nhu tren, pose detach luc train (loss chieu 2D)
            'smpl_pose': full_pose,
            'smpl_shape': shape_param,
            'cam_param': cam_trans,
        }

    # --------------------------------------------------------
    def forward(self, joints, img_feats, is_train=True, return_features=False,
                joints_2d=None, joints_2d_mask=None):
        """
        joints   : (B, J, 3) pose 3D lifter, root-relative, met
        img_feats: (B, 2048, H, W) feature map backbone (F8)
        joints_2d, joints_2d_mask: (B, J, 2) trong [-1, 1] + mask cho PE vi tri 2D
        return_features: tra them feature trung gian (feat_*)
        """
        B = img_feats.shape[0]
        device = img_feats.device

        # 1. Fusion
        fus = self.fusion(joints, img_feats, joints_2d, joints_2d_mask, return_features=return_features)
        joint_out = fus['joint_out']
        global_ft = fus['concat_feat']

        # 2. ctx (sau HyperGCN) -> query SMPL neo vao xuong -> cross-attn
        idx = torch.arange(self.num_joint, device=device)
        ctx = self.norm(joint_out) + self.node_pe(idx)
        if not self.v7_no_hgcn:                                      # v7_no_hgcn: bo qua HyperGCN (CHI ablation)
            ctx = self.spatial_hypers(ctx.unsqueeze(1)).squeeze(1)   # (B, J, dim)
        pose_token = self.pose_embed(self.init_pose.expand(B, -1, -1)) + self.pose_pe.weight.unsqueeze(0)
        bone = torch.cat([ctx[:, self.bone_pairs[:, 0]], ctx[:, self.bone_pairs[:, 1]]], dim=-1)  # (B, 24, 2C)
        pose_token = pose_token + self.bone_proj(bone)
        pose_feat = self.pose_context_attn(pose_token, ctx, ctx)     # (B, 24, dim)
        if self.smpl_hypers is not None:                             # HyperGCN SMPL-24: dong bo token trong cung chuoi chi
            pose_feat = self.smpl_hypers(pose_feat.unsqueeze(1)).squeeze(1)

        # 3. Camera (tham so tho [tx, ty, gamma])
        cam_raw = self.cam_head(global_ft)

        # 4. Shape token
        shape_emb = self.shape_embed(self.init_shape.expand(B, -1)).unsqueeze(1)   # (B, 1, dim)
        shape_token = self.shape_token.weight.unsqueeze(0).expand(B, 1, -1) + shape_emb
        kv = torch.cat([ctx, fus['img_out']], dim=1)                                # (B, J + H*W, dim)
        shape_out = self.fuse_shape_tok(shape_token, kv, kv)

        # 5. J29 -> IK -> SMPL
        result, ctx3d = self._mesh_outputs(joints, ctx, pose_feat, shape_out, cam_raw, is_train)
        if ctx3d is not None:
            result['ctx_3d'] = ctx3d                   # (B, J, 3) chi khi ablation v7_delta17

        if return_features:
            result.update({
                'feat_joint': joint_out,               # (B, J, dim)
                'feat_img': fus['img_out'],            # (B, H*W, dim)
                'feat_joint_in': fus['joint_tok'],     # (B, J, dim)
                'feat_img_in': fus['img_tok'],         # (B, H*W, dim)
                'feat_global': global_ft,              # (B, 2*dim)
                'feat_layers': fus['layers'],          # list dict: img, joint, attn_joint_to_img, attn_img_to_joint
                'feat_hyper': ctx,                     # (B, J, dim)
                'feat_pose': pose_feat,                # (B, 24, dim)
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
