"""
Teacher / Student end-to-end:

    img_feats + joints (+ PE 2D) -> Fusion (cross-attn 2 chieu) -> HyperGCN -> head goc xoay tung khop -> SMPL -> mesh

- Teacher: joints = GT 3D (sach)      -> joint_encoder='gt'
- Student: joints = 3D joint nhieu    -> joint_encoder='noisy'
- Kien truc v2 co dinh: PE 2D, self-attn khop, goc xoay tung khop, query neo xuong, shape doc token.
- FuseKin-v3 (cfg.MODEL.v3_*, mac dinh TAT = v2): [B] feature anh F16/F8 lay tai khop 2D cong vao token khop,
  [B'] bias attention theo hop distance giua cac khop, [C] head phan bo do sau tung khop (phan loai bin),
  [D] tinh chinh lap kieu ReFit (chieu khop -> lay feature + sai lech 2D -> GRU tung tham so -> delta),
  [E] log-sigma do bat dinh tung khop (loss khop 3D trong base.py).
- Model KHONG tinh loss. Moi thu can cho loss / KD duoc tra ve trong dict output.
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
from utils.transforms import rot6d_to_axis_angle

SMPL_MEAN_PARAMS = 'data_final/base_data/smpl_mean_params.npz'
H36M_PARENTS17 = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)


def hop_distance_h36m17():
    """Ma tran so buoc (hop) tren khung xuong H36M-17, (17, 17) long."""
    J = len(H36M_PARENTS17)
    d = np.full((J, J), 99, dtype=np.int64)
    np.fill_diagonal(d, 0)
    for j, p in enumerate(H36M_PARENTS17):
        if p >= 0:
            d[j, p] = d[p, j] = 1
    for k in range(J):
        d = np.minimum(d, d[:, k:k + 1] + d[k:k + 1, :])
    return torch.from_numpy(d)


def sample_joint_feats(fmaps, xy):
    """Lay mau bilinear cac feature map (list (B,C,h,w)) tai xy (B,J,2) trong [-1,1] -> (B, J, sum C)."""
    out = [F.grid_sample(f, xy[:, :, None, :].to(f.dtype), align_corners=False)[..., 0].transpose(1, 2) for f in fmaps]
    return torch.cat(out, -1)
# Module cua kien truc cu (VPoser, head pose/shape cu, goi y) da bo: checkpoint cu con cac key nay -> bo qua khi nap.
LEGACY_KEY_PARTS = ('vposer.', 'root_pose_head.', 'body_pose_head.', 'fuse_shape.', 'hint_proj.', 'hint_encoder.',
                    'hint_pred.', 'diffusion.')


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

    def forward(self, noisy_joints, pos_emb, pe2d=None):
        # noisy_joints: (B, J, 3)
        # pe2d (B, J, C) | None: PE vi tri 2D cua khop (PLAN_STUDENT_V2), cong TRUOC lop Transformer
        # de cac khop trao doi vi tri 2D voi nhau truoc khi hoi anh.
        x = self.proj(noisy_joints) + pos_emb
        if pe2d is not None:
            x = x + pe2d
        x = self.denoise_layer(x)
        return self.norm(x)


# Cap xuong (cha, con) theo chi so H36M-17 cho moi khop SMPL-24 (PLAN_PER_JOINT_ROT.md muc 3).
# H36M-17: 0 Pelvis, 1 R_Hip, 2 R_Knee, 3 R_Ankle, 4 L_Hip, 5 L_Knee, 6 L_Ankle, 7 Torso, 8 Neck, 9 Nose,
#          10 Head_top, 11 L_Shoulder, 12 L_Elbow, 13 L_Wrist, 14 R_Shoulder, 15 R_Elbow, 16 R_Wrist.
# Goc xoay cua khop SMPL j quyet dinh huong xuong tu j toi khop con -> query cua j lay dac trung cua dung xuong do.
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

# ---- Student v7: sieu canh (CUNG nhom khop voi hypergraph.py, dataset 'h36m') va anh xa giai ma theo bo phan ----
# H36M-17: 0 Pelvis, 1 R_Hip, 2 R_Knee, 3 R_Ankle, 4 L_Hip, 5 L_Knee, 6 L_Ankle, 7 Torso, 8 Neck, 9 Nose, 10 Head,
#          11 L_Shoulder, 12 L_Elbow, 13 L_Wrist, 14 R_Shoulder, 15 R_Elbow, 16 R_Wrist.
V7_PARTS = [[0, 7, 8], [8, 9, 10], [0, 1, 2], [2, 3], [0, 4, 5], [5, 6], [8, 14, 15], [8, 11, 12], [15, 16], [12, 13]]
V7_BODY = [[0, 1, 4, 7, 8, 9, 10, 11, 14], [14, 15, 16], [11, 12, 13], [1, 2, 3], [4, 5, 6]]   # than, tay P, tay T, chan P, chan T
# Twist khop SMPL 1..23 (xuong khop -> con) -> doan chi chua xuong do (chi so trong V7_PARTS)
V7_TWIST_PART = [4, 2, 0, 5, 3, 0, 5, 3, 0, 5, 3, 1, 7, 6, 1, 7, 6, 9, 8, 9, 8, 9, 8]
# 7 diem dau chi trong 29 diem HybrIK co delta: ban chan T/P, ban tay T/P, dinh dau, ngon chan cai T/P.
# (25, 26 giua ban tay bo: goc khop 22, 23 bi dat 0 nen khong anh huong mesh)
V7_LIMB_IDX = [10, 11, 22, 23, 24, 27, 28]
V7_LIMB_BODY = [4, 3, 2, 1, 0, 4, 3]          # chi / than chua diem do (chi so trong V7_BODY)
V7_LIMB_TOKEN = [10, 11, 22, 23, 15, 10, 11]  # token query SMPL dung (diem la -> token khop cha)


class PosEnc2D(nn.Module):
    """PE vi tri 2D Fourier, DUNG CHUNG cho token khop (theo toa do 2D) va o anh (theo tam o) -> cung khong gian vi tri.
    (x, y) trong [-1, 1] -> [sin(2^k*pi*x), cos(2^k*pi*x), sin(2^k*pi*y), cos(2^k*pi*y)], k = 0..L-1 -> Linear -> C.
    Linear khoi tao 0 (khong bias) -> luc dau PE = 0, mo hinh y het ban cu."""
    def __init__(self, embed_dim=512, num_freq=6):
        super().__init__()
        self.register_buffer('freqs', (2.0 ** torch.arange(num_freq).float()) * math.pi, persistent=False)
        self.proj = nn.Linear(4 * num_freq, embed_dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, xy):
        # xy: (..., 2) -> (..., C)
        ang = xy.unsqueeze(-1) * self.freqs.to(xy.dtype)              # (..., 2, L)
        feat = torch.cat([ang.sin(), ang.cos()], dim=-1)              # (..., 2, 2L)
        return self.proj(feat.flatten(-2))


class LearnableCoefficient(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        return x * self.bias


class JointImageSpreadGate(nn.Module):
    """Gate rieng tung khop: joint + thong tin anh + log1p(std*10), std don vi met."""
    def __init__(self, dim, hidden=64, zero_spread=False):
        super().__init__()
        if hidden < 1:
            raise ValueError('spread_gate_hidden phai >0')
        self.zero_spread = bool(zero_spread)
        self.override_value = None   # Chi inspector dung de dat gate=1; khong luu vao checkpoint.
        self.net = nn.Sequential(nn.Linear(2 * dim + 3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, joint_tok, image_update, joint_spread):
        if joint_spread is None or joint_spread.shape != (*joint_tok.shape[:2], 3):
            raise ValueError('Spread gate can joint_spread (B,J,3) met; khong duoc bo qua spread khi infer')
        if self.override_value is not None:
            return joint_tok.new_full((*joint_tok.shape[:2], 1), float(self.override_value))
        spread = torch.zeros_like(joint_spread) if self.zero_spread else joint_spread
        # Giu std=0 hop le tai pelvis; cung thang 10 nhu nhanh spread cu.
        spread = torch.log1p(spread.to(joint_tok.dtype).clamp_min(0) * 10.0)
        joint = F.layer_norm(joint_tok, joint_tok.shape[-1:])
        image = F.layer_norm(image_update, image_update.shape[-1:])
        # Lop cuoi khoi tao 0 => g=1 chinh xac; sau do g nam trong (0,2).
        return 2.0 * torch.sigmoid(self.net(torch.cat([joint, image, spread], dim=-1)))


class RGBJointCrossTransformerBlock(nn.Module):
    """
    rgb tokens:   (B, H*W, C)
    joint tokens: (B, J, C)
    Cross-attention 2 chieu, Pre-LN, residual dung LearnableCoefficient.
    """
    def __init__(self, d_model, h, block_exp=4, resid_pdrop=0.1, joint_self_attn=False,
                 spread_gate=False, spread_gate_zero=False, spread_gate_hidden=64):
        super().__init__()
        self.h = h
        self.d_k = d_model // h
        self.image_gate = JointImageSpreadGate(d_model, spread_gate_hidden, spread_gate_zero) if spread_gate else None

        # Self-attn giua cac token khop TRUOC cross-attn (PLAN_STUDENT_V2). out_proj khoi tao 0 -> luc dau = block cu.
        self.joint_sa = None
        if joint_self_attn:
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

    def forward(self, rgb_tok, joint_tok, return_intermediate=False, joint_spread=None, joint_attn_bias=None):
        b = rgb_tok.shape[0]

        # 0. Self-attn giua cac khop (tuy chon): cac khop biet nhau truoc khi hoi anh
        if self.joint_sa is not None:
            jn = self.ln_joint_sa(joint_tok)
            sa, _ = self.joint_sa(jn, jn, jn, need_weights=False, attn_mask=joint_attn_bias)
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
        gate = self.image_gate(joint_tok, joint_att, joint_spread) if self.image_gate is not None else None
        joint_update = joint_att if gate is None else gate * joint_att
        joint_out = self.coef1(joint_tok) + self.coef2(joint_update)
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
            if gate is not None:
                intermediate[0]['spread_gate'] = gate
            return rgb_out, joint_out, intermediate
        return rgb_out, joint_out


class RGBJointCrossTransformer(nn.Module):
    def __init__(self, d_model, h, block_exp=4, resid_pdrop=0.1, depth=3, joint_self_attn=False,
                 spread_gate=False, spread_gate_zero=False, spread_gate_hidden=64, hop_bias=False):
        super().__init__()
        self.h = h
        # [B'] Bias attention giua cac khop theo so hop tren khung xuong (BlockGCN), rieng tung head, khoi tao 0.
        self.hop_bias = None
        if hop_bias:
            assert joint_self_attn, 'v3_hop_bias can self-attn giua cac khop'
            hop = hop_distance_h36m17()
            self.register_buffer('hop_idx', hop.clamp(max=8), persistent=False)
            self.hop_bias = nn.Embedding(9, h)
            nn.init.zeros_(self.hop_bias.weight)
        self.blocks = nn.ModuleList([
            RGBJointCrossTransformerBlock(d_model, h, block_exp, resid_pdrop, joint_self_attn=joint_self_attn,
                                         spread_gate=spread_gate, spread_gate_zero=spread_gate_zero,
                                         spread_gate_hidden=spread_gate_hidden)
            for _ in range(depth)
        ])

    def forward(self, rgb_tok, joint_tok, return_intermediate=False, joint_spread=None):
        intermediate = []
        bias = None
        if self.hop_bias is not None and joint_tok.shape[1] == self.hop_idx.shape[0]:
            bias = self.hop_bias(self.hop_idx).permute(2, 0, 1)                     # (h, J, J)
            bias = bias.unsqueeze(0).expand(joint_tok.shape[0], -1, -1, -1).reshape(-1, *bias.shape[1:])
            bias = bias.to(joint_tok.dtype)
        for block in self.blocks:
            out = block(rgb_tok, joint_tok, return_intermediate=return_intermediate, joint_spread=joint_spread,
                        joint_attn_bias=bias)
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
                 in_channels=2048, depth=3, joint_encoder='gt',
                 joint_self_attn=False, use_pe2d=False, pe2d_num_freq=6,
                 spread_gate=False, spread_gate_zero=False, spread_gate_hidden=64, hop_bias=False):
        super().__init__()
        assert joint_encoder in ('gt', 'noisy'), f"joint_encoder khong hop le: {joint_encoder}"
        self.joint_encoder = joint_encoder
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        # PE vi tri 2D dung chung cho khop va o anh (PLAN_STUDENT_V2). emb_missing: khop mask = 0
        # (wrapper dat toa do = 0 = tam anh, phai thay bang embedding rieng). Ca hai khoi tao 0.
        self.use_pe2d = use_pe2d
        if use_pe2d:
            self.pe2d = PosEnc2D(embed_dim=embed_dim, num_freq=pe2d_num_freq)
            self.emb_missing = nn.Parameter(torch.zeros(1, 1, embed_dim))

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
            d_model=embed_dim, h=8, block_exp=4, resid_pdrop=0.1, depth=depth,
            joint_self_attn=joint_self_attn, spread_gate=spread_gate,
            spread_gate_zero=spread_gate_zero, spread_gate_hidden=spread_gate_hidden, hop_bias=hop_bias,
        )
        self.norm_img_in = nn.LayerNorm(embed_dim)

    @staticmethod
    def _cell_centers(h, w, device, dtype):
        """Tam o luoi h x w trong [-1, 1], cung quy uoc voi chuan hoa 2D cua wrapper (x / W * 2 - 1).
        Thu tu giong img_feats.flatten(2): chi so = hang * w + cot -> (1, h*w, 2) dang (x, y)."""
        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h * 2 - 1
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w * 2 - 1
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([gx, gy], dim=-1).reshape(1, h * w, 2)

    def forward(self, joints, img_feats, return_features=False,
                joints_2d=None, joints_2d_mask=None, joint_spread=None, joint_extra=None):
        """joints_2d (B, J, 2) trong [-1, 1] va joints_2d_mask (B, J, 1) | (B, J): chi dung khi use_pe2d."""
        img_feats = self.img_proj(img_feats)               # (B, C, H, W)
        bs, c, h, w = img_feats.shape

        pe_joint = None
        if self.use_pe2d:
            assert joints_2d is not None, 'joint_pe2d bat nhung khong co joints_2d (B, J, 2)'
            if joints_2d_mask is None:
                m = torch.ones_like(joints_2d[..., :1])
            else:
                m = joints_2d_mask.to(joints_2d.dtype)
                if m.dim() == 2:
                    m = m.unsqueeze(-1)
                m = (m > 0).to(joints_2d.dtype)
            pe_joint = m * self.pe2d(joints_2d[..., :2]) + (1.0 - m) * self.emb_missing   # (B, J, C)

        if self.joint_encoder == 'noisy':
            joints_tok = self.projector_student(joints, self.pos_emb_joint, pe2d=pe_joint)
        else:
            x = self.joint_proj(joints) + self.pos_emb_joint
            if pe_joint is not None:
                x = x + pe_joint
            joints_tok = self.norm_joint_proj(x)
        if joint_extra is not None:
            joints_tok = joints_tok + joint_extra           # [B]/[C] v3: feature anh tai khop + thong tin do sau
        joints_tok_in = joints_tok                          # token joint sau encoder (feat_joint_in cho KD)

        img_tok = img_feats.flatten(2).transpose(1, 2)     # (B, H*W, C)

        if h * w != self.pos_emb_img.shape[1]:
            pos_emb = self.pos_emb_img.transpose(1, 2).reshape(1, c, self.vert_anchors, self.horz_anchors)
            pos_emb = F.interpolate(pos_emb, size=(h, w), mode='bilinear', align_corners=False)
            pos_emb = pos_emb.flatten(2).transpose(1, 2)
        else:
            pos_emb = self.pos_emb_img
        if self.use_pe2d:
            # Cung module PE voi khop -> vi tri o anh va vi tri khop nam cung khong gian
            pos_emb = pos_emb + self.pe2d(self._cell_centers(h, w, img_tok.device, img_tok.dtype))

        img_tok = self.norm_img_in(img_tok + pos_emb)

        out = self.cfcer(img_tok, joints_tok, return_intermediate=return_features, joint_spread=joint_spread)
        if return_features:
            img_out, joint_out, layers = out
        else:
            img_out, joint_out = out
            layers = []

        concat_feat = torch.cat([img_out.mean(dim=1), joint_out.mean(dim=1)], dim=1)  # (B, 2C)
        return {
            'img_out': img_out,          # (B, H*W, C)
            'joint_out': joint_out,      # (B, J, C)
            'joint_tok': joints_tok_in,  # (B, J, C) token joint sau encoder, truoc cross-attn
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
# FuseKin-v3: head do sau tung khop [C] va tinh chinh lap [D]
# ============================================================
class JointDepthHead(nn.Module):
    """[C] (PandaPose): feature anh lay tai khop 2D -> phan bo do sau tuong doi pelvis (bins) cho tung khop.
    Tra logits (B,J,bins), do sau ky vong (B,J) met, entropy chuan hoa (B,J)."""
    def __init__(self, in_dim, num_joint=17, bins=64, depth_range=1.0, hidden=256):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.joint_emb = nn.Parameter(torch.zeros(1, num_joint, hidden))
        self.out = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, bins))
        self.register_buffer('centers', torch.linspace(-depth_range, depth_range, bins), persistent=False)

    def forward(self, joint_feats):
        logits = self.out(self.inp(joint_feats) + self.joint_emb)
        prob = torch.softmax(logits.float(), -1)
        exp_depth = (prob * self.centers).sum(-1)
        ent = -(prob * prob.clamp_min(1e-8).log()).sum(-1) / math.log(prob.shape[-1])
        return logits, exp_depth, ent


class StreamGRU(nn.Module):
    """S GRU doc lap chay song song (moi tham so 1 luong, ReFit), dau vao x dung chung (B, in)."""
    def __init__(self, streams, in_dim, hidden, out_dim):
        super().__init__()
        self.S, self.h = streams, hidden
        k = 1.0 / math.sqrt(hidden)
        self.w_ih = nn.Parameter(torch.empty(streams, in_dim, 3 * hidden).uniform_(-k, k))
        self.w_hh = nn.Parameter(torch.empty(streams, hidden, 3 * hidden).uniform_(-k, k))
        self.b_ih = nn.Parameter(torch.zeros(streams, 3 * hidden))
        self.b_hh = nn.Parameter(torch.zeros(streams, 3 * hidden))
        self.w1 = nn.Parameter(torch.empty(streams, hidden, hidden).uniform_(-k, k))
        self.b1 = nn.Parameter(torch.zeros(streams, hidden))
        self.w2 = nn.Parameter(torch.zeros(streams, hidden, out_dim))   # khoi tao 0 -> delta = 0 luc dau
        self.b2 = nn.Parameter(torch.zeros(streams, out_dim))

    def forward(self, x, h):
        """x (B, in), h (B, S, hidden) -> (delta (B, S, out), h moi)."""
        gi = torch.einsum('bi,sio->bso', x, self.w_ih) + self.b_ih
        gh = torch.einsum('bsh,sho->bso', h, self.w_hh) + self.b_hh
        ir, iz, inn = gi.chunk(3, -1)
        hr, hz, hn = gh.chunk(3, -1)
        r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        n = torch.tanh(inn + r * hn)
        h = (1 - z) * n + z * h
        y = F.gelu(torch.einsum('bsh,sho->bso', h, self.w1) + self.b1)
        return torch.einsum('bsh,sho->bso', y, self.w2) + self.b2, h


class RefitRefiner(nn.Module):
    """[D] Tinh chinh lap (ReFit): chieu H36M-17 cua mesh hien tai -> lay feature anh tai diem chieu (cua so 3x3
    tren F16) + sai lech 2D (Halpe - chieu) x conf + tham so hien tai + cam_bbox -> 26 GRU (24 goc xoay 6D, shape,
    camera) -> delta. Feedback dropout theo khop luc train."""
    def __init__(self, feat_in, num_joint=17, feat_dim=32, hidden=32, dropout=0.25):
        super().__init__()
        self.J, self.dropout = num_joint, dropout
        self.feat_proj = nn.Linear(feat_in, feat_dim)
        in_dim = num_joint * (feat_dim + 3) + 24 * 6 + 10 + 3 + 3
        self.norm = nn.LayerNorm(in_dim)
        self.gru = StreamGRU(26, in_dim, hidden, 10)

    def feedback(self, fmaps, xy_proj, xy_2d, conf_2d, rot6d, shape, cam_raw, cam_bbox, is_train):
        B = xy_proj.shape[0]
        step = 2.0 / fmaps[0].shape[-1]                                   # 1 o cua F16
        offs = torch.tensor([-1.0, 0.0, 1.0], device=xy_proj.device) * step
        grid = torch.stack(torch.meshgrid(offs, offs, indexing='xy'), -1).reshape(1, 1, 9, 2)
        pts = (xy_proj[:, :, None, :] + grid).reshape(B, -1, 2)                 # (B, J*9, 2)
        f = sample_joint_feats(fmaps, pts).reshape(B, self.J, 9, -1).mean(2)   # (B, J, C)
        f = self.feat_proj(f)
        res = torch.cat([(xy_2d - xy_proj) * conf_2d * 10.0, conf_2d], -1)       # (B, J, 3)
        fb = torch.cat([f, res], -1)
        if is_train and self.dropout > 0:
            keep = (torch.rand(B, self.J, 1, device=fb.device) >= self.dropout).to(fb.dtype)
            fb = fb * keep
        x = torch.cat([fb.flatten(1), rot6d.flatten(1), shape, cam_raw, cam_bbox], -1)
        return self.norm(x)


# ============================================================
# Model chung cho Teacher va Student
# ============================================================
class FusionPose2Mesh(nn.Module):
    def __init__(self, num_joint=17, embed_dim=512, joint_encoder='gt',
                 vert_anchors=8, horz_anchors=8, depth=3,
                 smpl_head_hidden_dim=256, smpl_head_depth=3, use_img_probe=False, pe2d_num_freq=6,
                 spread_gate=False, spread_gate_zero=False, spread_gate_hidden=64,
                 joint_feat=False, hop_bias=False, depth_head=False, depth_bins=64, depth_range=1.0,
                 refine_iters=0, refine_feat_dim=32, refine_hidden=32, refine_dropout=0.25, joint_uncert=False,
                 feat16_dim=1024, feat8_dim=2048, bone_query=False, v6_ik=False, hgcn_freeze_adj=False,
                 v7_anchor=False):
        super().__init__()
        assert num_joint == 17, "Query neo xuong can 17 khop H36M (bang SMPL_BONE_PAIRS_H36M)"
        self.num_joint = num_joint
        self.embed_dim = embed_dim
        self.joint_feat, self.depth_head_on, self.refine_iters = joint_feat, depth_head, int(refine_iters)
        self.need_f16 = joint_feat or depth_head or self.refine_iters > 0

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
            joint_self_attn=True, use_pe2d=True, pe2d_num_freq=pe2d_num_freq,
            spread_gate=spread_gate, spread_gate_zero=spread_gate_zero, spread_gate_hidden=spread_gate_hidden,
            hop_bias=hop_bias,
        )
        global_dim = embed_dim * 2

        # ---- Pose path (HyperGCN) ----
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

        # ---- Head goc xoay theo tung khop (a1) ----
        # Moi token trong 24 token pose -> delta 6D cua CHINH khop do, cong vao mean pose (goc local nhu SMPL).
        # Lop cuoi khoi tao 0 -> luc dau = mean pose.
        self.per_joint_head = MLP(embed_dim, smpl_head_hidden_dim, 6, 2)
        nn.init.zeros_(self.per_joint_head.layers[-1].weight)
        nn.init.zeros_(self.per_joint_head.layers[-1].bias)
        # ---- Neo query vao xuong (a2): query_j += Linear(concat(ctx[cha_j], ctx[con_j])), khoi tao 0 ----
        self.register_buffer('bone_pairs', torch.tensor(SMPL_BONE_PAIRS_H36M, dtype=torch.long),
                             persistent=False)                                   # (24, 2)
        self.bone_proj = nn.Linear(2 * embed_dim, embed_dim)
        nn.init.zeros_(self.bone_proj.weight)
        nn.init.zeros_(self.bone_proj.bias)
        # ---- v5 (2026-11-07) query nhan biet hinh hoc xuong: giai ma 3D tu ctx (SAU HyperGCN) -> vector xuong
        # cha -> con (huong don vi + do dai) -> MLP (lop cuoi khoi tao 0) cong vao query. ctx_3d co loss voi GT
        # (base.py, MODEL.v5_ctx_loss_w). Mac dinh TAT. ----
        self.bone_query = bool(bone_query)
        if self.bone_query:
            self.ctx_joint_head = nn.Linear(embed_dim, 3)
            self.bone_geo_proj = nn.Sequential(nn.Linear(4, 128), nn.GELU(), nn.Linear(128, embed_dim))
            nn.init.zeros_(self.bone_geo_proj[-1].weight)
            nn.init.zeros_(self.bone_geo_proj[-1].bias)

        # ---- Shape path (init tu mean_shape) ----
        self.shape_embed = nn.Linear(10, embed_dim)
        self.shape_token = nn.Embedding(1, embed_dim)
        self.shape_head = MLP(embed_dim, smpl_head_hidden_dim, 10, smpl_head_depth)
        # Shape doc token: shape token cross-attn [ctx (17 khop), token anh] -> doc duoc do dai xuong, be ngang co the.
        self.fuse_shape_tok = CrossAttentionBlock(
            q_dim=embed_dim, k_dim=embed_dim, v_dim=embed_dim,
            kv_num=num_joint + vert_anchors * horz_anchors,
            num_heads=8, mlp_ratio=4., qkv_bias=True,
            drop=0., attn_drop=0., drop_path=0.2, has_mlp=True,
        )

        # ---- Camera ----
        self.cam_head = MLP(global_dim, smpl_head_hidden_dim, 3, 2)

        # ---- Head tham do anh (tuy chon): doan joint 3D CHI tu token anh dau vao fusion ----
        # Muc dich: (1) ep img_proj / pos_emb_img / norm_img_in mang thong tin tu the,
        #           (2) do truc tiep xem dac trung anh co du tot khong.
        self.use_img_probe = use_img_probe
        if use_img_probe:
            self.probe_query = nn.Embedding(num_joint, embed_dim)
            self.probe_attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
            self.probe_norm = nn.LayerNorm(embed_dim)
            self.probe_head = MLP(embed_dim, smpl_head_hidden_dim, 3, 2)

        # ---- FuseKin-v3 ----
        fin = feat16_dim + feat8_dim
        if joint_feat:                                   # [B] feature anh tai khop -> token khop (khoi tao 0)
            self.jf_proj = nn.Linear(fin, embed_dim)
            nn.init.zeros_(self.jf_proj.weight)
            nn.init.zeros_(self.jf_proj.bias)
        if depth_head:                                   # [C] phan bo do sau -> token khop (khoi tao 0)
            self.depth_head = JointDepthHead(fin, num_joint, depth_bins, depth_range)
            self.depth_proj = nn.Linear(3, embed_dim)
            nn.init.zeros_(self.depth_proj.weight)
            nn.init.zeros_(self.depth_proj.bias)
        if self.refine_iters > 0:                        # [D] tinh chinh lap
            reg_path = os.path.join(cfg.DATASET.jotr_data_root, 'Human36M', 'J_regressor_h36m_correct.npy')
            self.register_buffer('j_h36m', torch.from_numpy(np.load(reg_path)).float(), persistent=False)
            self.refiner = RefitRefiner(fin, num_joint, refine_feat_dim, refine_hidden, refine_dropout)
        if joint_uncert:                                 # [E] log-sigma tung khop (loss trong base.py)
            self.joint_log_sigma = nn.Parameter(torch.zeros(num_joint))

        # ---- Khoa ma tran sieu canh HyperGCN (MODEL.hgcn_freeze_adj, §17): giu dung ma tran giai phau khoi tao ----
        if hgcn_freeze_adj:
            n_frz = 0
            for name, p in self.spatial_hypers.named_parameters():
                if name.endswith('p_adj') or name.endswith('b_adj'):
                    p.requires_grad_(False)
                    n_frz += 1
            print(f'===> HyperGCN: khoa {n_frz} ma tran sieu canh (p_adj / b_adj)')

        # ---- Student v6 (MODEL.v6_ik): J29 = M x (lift + delta17(ctx)) + delta29(query) -> IK HybrIK -> goc SMPL ----
        # Moi head khoi tao 0 (twist = 0) -> luc dau = baseline "lifter + M + IK" (experiment.md §19).
        self.v6_ik = bool(v6_ik)
        if self.v6_ik:
            assert self.refine_iters == 0, 'v6_ik khong dung chung voi [D] (v3_refine_iters phai = 0)'
            assert not self.bone_query, 'v6_ik khong dung chung voi v5_bone_query'
            from models.hybrik_smpl import HybrIKSMPL
            reg_path = os.path.join(cfg.DATASET.jotr_data_root, 'Human36M', 'J_regressor_h36m_correct.npy')
            self.hybrik = HybrIKSMPL(self.human_model_layer, np.load(reg_path))
            Mmap = np.load(cfg.MODEL.v6_m_path).astype(np.float32)
            assert Mmap.shape == (29, 17), f'v6_m_path phai la ma tran (29, 17), nhan {Mmap.shape}'
            self.register_buffer('v6_M', torch.from_numpy(Mmap))          # luu trong ckpt -> test dung dung M
            self.v6_d17 = self.v6_d29 = self.v6_dleaf = self.v6_tw = None
            if cfg.MODEL.v6_delta17:                     # hieu chinh pose lifter tu ctx (sau HyperGCN)
                self.v6_d17 = nn.Linear(embed_dim, 3)
            if cfg.MODEL.v6_delta29:                     # hieu chinh 24 khop SMPL + 5 diem dau chi tu token query
                self.v6_d29 = nn.Linear(embed_dim, 3)
                self.v6_dleaf = nn.Linear(embed_dim, 3)  # diem dau chi lay token cua khop cha
            if cfg.MODEL.v6_twist:                       # twist (cos, sin) cong vao (1, 0)
                self.v6_tw = nn.Linear(embed_dim, 2)
            for m in (self.v6_d17, self.v6_d29, self.v6_dleaf, self.v6_tw):
                if m is not None:
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
            print(f'===> Student v6 IK: M {cfg.MODEL.v6_m_path} | delta17 {cfg.MODEL.v6_delta17} '
                  f'delta29 {cfg.MODEL.v6_delta29} twist {cfg.MODEL.v6_twist}')

        # ---- Student v7 (MODEL.v7_anchor, 2026-11-09): mesh NEO theo pose lifter, Student chi doan phan BE MAT ----
        # Pose than = pose3d (khong sua); delta 7 diem dau chi tu body_tok; twist tu part_tok; beta = beta_xuong + dbeta.
        # Moi head khoi tao 0 -> luc dau = "lifter + beta tu xuong + IK" (experiment.md muc 21: ~ LIFT).
        self.v7 = bool(v7_anchor)
        if self.v7:
            assert not self.v6_ik, 'v7_anchor va v6_ik khong bat cung luc'
            assert self.refine_iters == 0 and not self.bone_query, 'v7_anchor can v3_refine_iters = 0, v5_bone_query = False'
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
            self.v7_readout = bool(cfg.MODEL.v7_part_readout)
            self.v7_no_hgcn = bool(cfg.MODEL.v7_no_hgcn)
            fin2 = 2 * embed_dim if self.v7_readout else embed_dim
            if self.v7_readout:
                self.v7_part_norm = nn.LayerNorm(embed_dim)
                self.v7_body_norm = nn.LayerNorm(embed_dim)
            self.v7_tw = nn.Linear(fin2, 2) if cfg.MODEL.v7_twist else None          # twist (cos, sin) += ...
            self.v7_dlimb = nn.Linear(fin2, 3) if cfg.MODEL.v7_delta_limb else None   # delta 7 diem dau chi
            self.v7_db = nn.Linear(fin2, 10) if cfg.MODEL.v7_dbeta else None         # dbeta
            self.v7_d17 = nn.Linear(embed_dim, 3) if cfg.MODEL.v7_delta17 else None  # CHI ablation
            for m in (self.v7_tw, self.v7_dlimb, self.v7_db, self.v7_d17):
                if m is not None:
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
            print(f'===> Student v7 anchor: readout {self.v7_readout} | twist {cfg.MODEL.v7_twist} '
                  f'dlimb {cfg.MODEL.v7_delta_limb} dbeta {cfg.MODEL.v7_dbeta} delta17 {cfg.MODEL.v7_delta17} '
                  f'no_hgcn {self.v7_no_hgcn}')

    # --------------------------------------------------------
    def _v6_ik_outputs(self, joints, ctx, pose_feat, shape_param, cam_raw, is_train):
        """v6: pose lifter (joints, root-relative) -> pose17 -> J29 -> IK (beta = shape_param, twist) -> goc SMPL
        -> cung dict output nhu _smpl_outputs. Tra (result, pose17, j29)."""
        from models.hybrik_smpl import LEAF_PARENTS, rotmat_to_aa_safe
        B = joints.shape[0]
        lift = joints - joints[:, 0:1]
        pose17 = lift
        if self.v6_d17 is not None:
            pose17 = lift + self.v6_d17(ctx)
            pose17 = pose17 - pose17[:, 0:1]
        j29 = torch.einsum('jk,bkc->bjc', self.v6_M, pose17)                              # (B, 29, 3)
        if self.v6_d29 is not None:
            d24 = self.v6_d29(pose_feat)                                                  # (B, 24, 3)
            dleaf = self.v6_dleaf(pose_feat[:, LEAF_PARENTS])                             # (B, 5, 3)
            j29 = j29 + torch.cat([d24, dleaf], dim=1)
        phis = self.hybrik.default_phis(B, j29.device, torch.float32)
        if self.v6_tw is not None:
            phis = phis + self.v6_tw(pose_feat[:, 1:24]).float()                          # khop 1..23
        rot_mats = self.hybrik.ik(j29.float(), shape_param.float(), phis)                 # (B, 24, 3, 3)
        full_pose = rotmat_to_aa_safe(rot_mats.reshape(-1, 3, 3)).reshape(B, 24, 3)
        full_pose = torch.cat([full_pose[:, :22], torch.zeros_like(full_pose[:, 22:])], dim=1).reshape(B, 72)
        result = self._smpl_outputs_aa(full_pose, shape_param, cam_raw, is_train)
        result['root_pose_6d'] = rot_mats[:, 0, :, :2].reshape(B, 6)
        result['v6_phis'] = phis
        return result, pose17, j29

    def _v7_outputs(self, joints, ctx, pose_feat, shape_out, cam_raw, is_train):
        """v7: pose than = pose3d (khong sua) -> J29 = M x pose17 (+ delta 7 diem dau chi) ; beta = beta_xuong + dbeta ;
        twist tu part_tok -> IK -> goc SMPL -> cung dict output nhu _smpl_outputs. Tra (result, ctx3d | None)."""
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
        dlimb = None
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
        result = self._smpl_outputs_aa(full_pose, beta, cam_raw, is_train)
        result['root_pose_6d'] = rot_mats[:, 0, :, :2].reshape(B, 6)
        result['v7_j29'] = j29                                                        # loss J29 chi 7 diem dau chi
        result['v7_dbeta'] = dbeta                                                    # rang buoc |dbeta|^2
        result['v7_beta0'] = beta0
        return result, ctx3d

    def _smpl_outputs_aa(self, full_pose, shape_param, cam_raw, is_train):
        """Nhu _smpl_outputs nhung nhan san goc axis-angle (B, 72) (dung cho v6). Cung key output."""
        cam_trans = self.get_camera_trans(cam_raw)
        joint_proj, joint_cam, mesh_cam, mesh_cam_render = self.get_coord(full_pose, shape_param, cam_trans)
        if is_train:
            _, _, _, mesh_render_det = self.get_coord(full_pose.detach(), shape_param, cam_trans)
        else:
            mesh_render_det = mesh_cam_render
        return {
            'joint_proj': joint_proj,
            'joint_cam': joint_cam,
            'smpl_mesh_cam': mesh_cam,
            'smpl_mesh_cam_render': mesh_cam_render,
            'smpl_mesh_cam_render_det': mesh_render_det,
            'smpl_pose': full_pose,
            'smpl_shape': shape_param,
            'cam_param': cam_trans,
        }

    def _smpl_outputs(self, rot6d, shape_param, cam_raw, is_train):
        """rot6d (B,24,6), shape (B,10), cam_raw (B,3) -> dict output SMPL (cung key voi ban v2)."""
        B = rot6d.shape[0]
        full_pose = rot6d_to_axis_angle(rot6d.reshape(-1, 6)).reshape(B, 24, 3)
        full_pose = torch.cat([full_pose[:, :22], torch.zeros_like(full_pose[:, 22:])], dim=1).reshape(B, 72)
        cam_trans = self.get_camera_trans(cam_raw)
        joint_proj, joint_cam, mesh_cam, mesh_cam_render = self.get_coord(full_pose, shape_param, cam_trans)
        if is_train:
            # Loss chieu 2D: detach pose (giu shape, camera) -> chi shape / camera hoc tu 2D
            _, _, _, mesh_render_det = self.get_coord(full_pose.detach(), shape_param, cam_trans)
        else:
            mesh_render_det = mesh_cam_render
        return {
            'joint_proj': joint_proj,                  # (B, K, 2) 2D SMPL-30, don vi output_hm_shape
            'joint_cam': joint_cam,                    # (B, K, 3) root-relative (pelvis SMPL)
            'smpl_mesh_cam': mesh_cam,                 # (B, 6890, 3) tru pelvis SMPL (eval tu tru goc lai)
            'smpl_mesh_cam_render': mesh_cam_render,   # (B, 6890, 3) tuyet doi (loss mesh tru pelvis H36M tu day)
            'smpl_mesh_cam_render_det': mesh_render_det,  # nhu tren, pose detach luc train (loss chieu 2D H36M-17)
            'smpl_pose': full_pose,
            'smpl_shape': shape_param,
            'cam_param': cam_trans,
            'root_pose_6d': rot6d[:, 0],               # (B, 6)
        }

    def _project_h36m(self, mesh_abs):
        """H36M-17 tu mesh tuyet doi -> toa do chuan hoa [-1,1] trong khung crop (cung quy uoc voi 2D dau vao)."""
        j = torch.matmul(self.j_h36m[None], mesh_abs)
        x = j[..., 0] / (j[..., 2] + 1e-4) * cfg.DATASET.focal[0] + cfg.DATASET.princpt[0]
        y = j[..., 1] / (j[..., 2] + 1e-4) * cfg.DATASET.focal[1] + cfg.DATASET.princpt[1]
        return torch.stack([x / cfg.input_img_shape[1] * 2 - 1, y / cfg.input_img_shape[0] * 2 - 1], -1)

    # --------------------------------------------------------
    def forward(self, joints, img_feats, is_train=True, return_features=False,
                joints_2d=None, joints_2d_mask=None, joint_spread=None, img_feats16=None, cam_bbox=None):
        """
        Args:
            joints   : (B, J, 3) root-relative, don vi met (GT cho teacher, nhieu cho student)
            img_feats: (B, 2048, H, W) feature map backbone (F8)
            return_features: tra them feature trung gian cho KD
            joints_2d, joints_2d_mask: (B, J, 2) trong [-1, 1] + mask cho PE vi tri 2D
            joint_spread: (B,J,3) met, std rieng cho gate, khong cong vao token.
            img_feats16: (B, 1024, 2H, 2W) feature layer3 (F16), can khi bat v3_joint_feat / depth_head / refine
            cam_bbox: (B, 3) thong tin bbox trong anh goc (CLIFF), dung trong vong tinh chinh [D]
        """
        B = img_feats.shape[0]
        device = img_feats.device

        # 0. FuseKin-v3: feature anh tai khop 2D -> [B] token khop, [C] phan bo do sau
        joint_extra, depth_out, fmaps = None, None, None
        if self.need_f16:
            assert img_feats16 is not None and joints_2d is not None, 'v3 can img_feats16 (F16) va joints_2d'
            fmaps = [img_feats16, img_feats]
            m2d = joints_2d_mask
            if m2d is not None:
                m2d = (m2d.reshape(B, -1, 1) > 0).to(img_feats.dtype)
            jf = sample_joint_feats(fmaps, joints_2d[..., :2])                    # (B, J, 3072)
            if m2d is not None:
                jf = jf * m2d                                                     # khop mask = 0 -> khong co feature
            if self.joint_feat:
                joint_extra = self.jf_proj(jf)
            if self.depth_head_on:
                logits, exp_d, ent = self.depth_head(jf)
                depth_out = (logits, exp_d)
                d_in = torch.stack([exp_d * 10.0, ent, (joints[..., 2] - exp_d) * 10.0], -1).to(jf.dtype)
                if m2d is not None:
                    d_in = d_in * m2d
                d_tok = self.depth_proj(d_in)
                joint_extra = d_tok if joint_extra is None else joint_extra + d_tok

        # 1. Fusion
        fus = self.fusion(joints, img_feats, return_features=return_features,
                          joints_2d=joints_2d, joints_2d_mask=joints_2d_mask, joint_spread=joint_spread,
                          joint_extra=joint_extra)
        joint_out = fus['joint_out']
        global_ft = fus['concat_feat']

        # 1b. Head tham do anh: chi dung token anh dau vao fusion (khong qua joint)
        img_probe_joints = None
        if self.use_img_probe:
            q = self.probe_query.weight.unsqueeze(0).expand(B, -1, -1)
            att, _ = self.probe_attn(q, fus['img_tok'], fus['img_tok'], need_weights=False)
            probe = self.probe_head(self.probe_norm(q + att))              # (B, J, 3)
            img_probe_joints = probe - probe[:, 0:1, :]                    # root-relative nhu GT

        # 2. Pose. ctx tinh TRUOC query (a2 can ctx de neo query vao xuong)
        idx = torch.arange(self.num_joint, device=device)
        ctx = self.norm(joint_out) + self.node_pe(idx)
        if not (self.v7 and self.v7_no_hgcn):                        # v7_no_hgcn: bo qua HyperGCN (CHI ablation)
            ctx = self.spatial_hypers(ctx.unsqueeze(1)).squeeze(1)   # (B, J, dim)

        # query: mean_pose (24, 6) -> (B, 24, dim)
        pose_token = self.pose_embed(self.init_pose.expand(B, -1, -1)) + self.pose_pe.weight.unsqueeze(0)
        # a2: query_j += Linear(concat(ctx[cha_j], ctx[con_j])) -> giu thu tu cha -> con (huong xuong)
        bone = torch.cat([ctx[:, self.bone_pairs[:, 0]], ctx[:, self.bone_pairs[:, 1]]], dim=-1)  # (B, 24, 2C)
        pose_token = pose_token + self.bone_proj(bone)
        ctx3d = None
        if self.bone_query:                                          # v5: hinh hoc xuong tuong minh tu ctx
            ctx3d = self.ctx_joint_head(ctx)
            ctx3d = ctx3d - ctx3d[:, 0:1]                            # (B, J, 3) met, root-relative
            vb = ctx3d[:, self.bone_pairs[:, 1]] - ctx3d[:, self.bone_pairs[:, 0]]   # (B, 24, 3) cha -> con
            lb = vb.norm(dim=-1, keepdim=True)
            geo = torch.cat([vb / lb.clamp_min(1e-4), lb * 10.0], dim=-1)          # huong + do dai (dm)
            pose_token = pose_token + self.bone_geo_proj(geo)
        pose_feat = self.pose_context_attn(pose_token, ctx, ctx)     # (B, 24, dim)

        # a1: moi token -> delta 6D cua khop do, cong mean pose (local)
        # v6: goc tinh bang IK o buoc 5 (per_joint_head khong dung)
        rot6d = None if (self.v6_ik or self.v7) else self.init_pose + self.per_joint_head(pose_feat)   # (B, 24, 6)

        # 3. Camera (tham so tho [tx, ty, gamma])
        cam_raw = self.cam_head(global_ft)

        # 4. Shape (init tu mean_shape)
        shape_emb = self.shape_embed(self.init_shape.expand(B, -1)).unsqueeze(1)   # (B, 1, dim)
        shape_token = self.shape_token.weight.unsqueeze(0).expand(B, 1, -1) + shape_emb
        kv = torch.cat([ctx, fus['img_out']], dim=1)                                # (B, J + H*W, dim)
        shape_out = self.fuse_shape_tok(shape_token, kv, kv)
        shape_param = self.shape_head(shape_out).reshape(B, -1)

        # 5. SMPL (uoc luong ban dau)
        if self.v6_ik:
            result, ctx3d, j29 = self._v6_ik_outputs(joints, ctx, pose_feat, shape_param, cam_raw, is_train)
            result['v6_j29'] = j29                     # (B, 29, 3) loss j29 (base.py); ctx3d = pose17 -> 'ctx_3d'
        elif self.v7:
            result, ctx3d = self._v7_outputs(joints, ctx, pose_feat, shape_out, cam_raw, is_train)
        else:
            result = self._smpl_outputs(rot6d, shape_param, cam_raw, is_train)

        # 6. [D] Tinh chinh lap: moi buoc nhin lai anh tai vi tri khop da chieu, cap nhat delta (ReFit).
        if self.refine_iters > 0:
            assert joints_2d is not None, 'v3_refine_iters can joints_2d (2D Halpe)'
            conf2d = joints_2d_mask.reshape(B, -1, 1).to(rot6d.dtype) if joints_2d_mask is not None \
                else torch.ones_like(joints_2d[..., :1])
            cb = cam_bbox.to(rot6d.dtype) if cam_bbox is not None else rot6d.new_zeros(B, 3)
            h = rot6d.new_zeros(B, 26, self.refiner.gru.h)
            history = []
            for _ in range(self.refine_iters):
                history.append(result)
                # Feedback dung uoc luong truoc DA DETACH (ReFit): grad buoc t khong chay nguoc qua phep chieu
                # (chia cho z, de no khi tz nho) vao uoc luong truoc; clamp de diem chieu lech xa khong lam no input.
                xy_proj = self._project_h36m(result['smpl_mesh_cam_render'].detach()).clamp(-2.0, 2.0)
                x = self.refiner.feedback(fmaps, xy_proj, joints_2d[..., :2], conf2d, rot6d.detach(),
                                          shape_param.detach(), cam_raw.detach(), cb, is_train and self.training)
                delta, h = self.refiner.gru(x, h)                                # (B, 26, 10)
                # Gradient chi qua delta, khong qua uoc luong truoc (ReFit) -> on dinh
                rot6d = rot6d.detach() + delta[:, :24, :6]
                shape_param = shape_param.detach() + delta[:, 24, :10]
                cam_raw = cam_raw.detach() + delta[:, 25, :3]
                result = self._smpl_outputs(rot6d, shape_param, cam_raw, is_train)
            result['refine_history'] = history          # uoc luong truoc buoc cuoi (Theta_0..Theta_{T-1}), cho loss moi buoc

        if ctx3d is not None:
            result['ctx_3d'] = ctx3d                               # (B, J, 3) v5: 3D giai ma tu ctx (sau HyperGCN)
        if self.use_img_probe:
            result['img_probe_joints'] = img_probe_joints          # (B, J, 3) chi tu anh
        if depth_out is not None:
            result['depth_logits'] = depth_out[0]                  # (B, J, bins) [C]
            result['depth_exp'] = depth_out[1]                     # (B, J) met

        # --- dung cho KD ---
        if return_features:
            result.update({
                'feat_joint': joint_out,               # (B, J, dim)
                'feat_img': fus['img_out'],            # (B, H*W, dim)
                'feat_joint_in': fus['joint_tok'],     # (B, J, dim)
                'feat_img_in': fus['img_tok'],         # (B, H*W, dim) token anh DAU VAO fusion
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
