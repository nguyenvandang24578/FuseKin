"""
lib/models/fusekin_v10.py -- FuseKin v10 (2026-11-18, nguoi dung duyet): mang END-TO-END 1 giai doan CHI NHAN ANH + 2D.
  KHONG dung ImgLifter, KHONG dung Refiner2D, KHONG dung IK.
  Ket hop:
    [1] Backbone ResNet-50 (SPIN) + tiem heatmap 2D o stride 4 -> F16 (1024x16x16) va F8 (2048x8x8).
    [2] Khoi Dac trung The tich 3D (8x8x8 = 512 voxel) + Neo do sau phi tuyen Z = r^k (k = exponent hoc duoc)
        + BasicBlock3D + 3D Transformer Encoder (refine_3d), theo JOTR / DPMesh.
    [3] Khoi tao 17 token khop tu 2D detector + F16 va noi suy vi tri 2.5D ban dau (u0, v0, d0) cho khop bi che.
    [4] 4 lop Decoder lap:
        - Self-Attn & Cross-Attn vao [64 token anh 2D, 17 token lay mau The tich 3D (8x8x8) + F16 (16x16)];
        - AdaptiveHyperGCN: Sieu do thi (16 xuong + 10 bo phan part + 5 chuoi chi body) co cong do tin cay dong
          theo du lieu tung khop (chan khop bi che phat nhieu, truyen tin khung xuong cuu khop khuat) + attention xuong;
        - Cap nhat toa do 2.5D (u, v, d) theo inverse_sigmoid (nhu JOTR) -> lay mau lai V3D (trilinear) va F16 (bilinear)
          de dua vao memory cua lop tiep theo.
    [5] Dau ra VPoser cong don theo tang (Cascade Residual nhu JOTR + Readout 5 chuoi chi tu AdaptiveHyperGCN):
        - root_pose_6d (6 chieu) + vposer_z (32 chieu -> VPoser dong bang -> 21 khop than + 2 ban tay = 0)
        - shape (10 chieu) + cam (3 chieu) -> SMPL mesh o MOI lop decoder (refine_history).
"""
import math
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
from models.ARTS import make_heatmaps
from models.backbones.resnet import ResNetBackbone
from models.fusekin_v8 import EncLayer, DecLayer, CascadeHead, V8Head, rot6d_to_rotmat, sine_pos_2d
from models.hybrik_smpl import rotmat_to_aa_safe
from models.img_lifter import H36M_BONES, h36m_adjacency
from models.teacher_student import V7_PARTS, V7_BODY

SMPL_MEAN_PARAMS = 'data_final/base_data/smpl_mean_params.npz'
N_SMPL_TOK = 3     # token pose, shape, camera
N_JOINT = 17       # 17 khop H36M
VPOSER_DIM = 32    # chieu ma an VPoser


def inverse_sigmoid(x, eps=1e-5):
    """Ham nguoc cua sigmoid tren (0, 1), giong JOTR/common/nets/transformer.py."""
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def sincos_1d(n_pos, dim, device, dtype):
    """PE sin/cos 1D (1, n_pos, dim) cho 512 voxel cua khoi the tich 3D (giong PositionalEncoding1D cua JOTR)."""
    pos = torch.arange(n_pos, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(n_pos, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0).to(dtype)


class VPoserDecoder(nn.Module):
    """Vo boc VPoser dong bang (nhu JOTR / DPMesh): ma an z (B, 32) -> goc xoay 23 khop than/tay (B, 23, 3) axis-angle."""

    def __init__(self, ckpt_dir=''):
        super().__init__()
        path = self._resolve_path(ckpt_dir)
        from human_body_prior.tools.model_loader import load_vposer
        self.vposer, _ = load_vposer(path, vp_model='snapshot')
        self.vposer.eval()
        for p in self.vposer.parameters():
            p.requires_grad = False
        print(f'===> [V10] Nap VPoser dong bang tu: {path}')

    @staticmethod
    def _resolve_path(ckpt_dir):
        cands = []
        if ckpt_dir:
            cands.append(ckpt_dir if osp.isabs(ckpt_dir) else osp.join(cfg.root_dir, ckpt_dir))
        for rel in (
            'data_final/base_data/human_model_files/smpl/VPOSER_CKPT',
            'data_final/human_model_files/smpl/VPOSER_CKPT',
            'JOTR/common/utils/human_model_files/smpl/VPOSER_CKPT',
            'DPMesh/common/utils/human_model_files/smpl/VPOSER_CKPT',
            'common/utils/human_model_files/smpl/VPOSER_CKPT',
        ):
            cands.append(osp.join(cfg.root_dir, rel))
        for p in cands:
            if osp.isdir(p):
                return p
        raise FileNotFoundError(
            f'Khong tim thay thu muc VPOSER_CKPT. Da thu: {cands[:3]}. '
            f'Kiem tra MODEL.v10_vposer_ckpt trong YAML.')

    def train(self, mode=True):
        super().train(mode)
        self.vposer.eval()
        return self

    def forward(self, z):
        """z (B, 32) -> body_aa (B, 23, 3): 21 khop do VPoser giai ma + 2 khop ban tay (L_Hand, R_Hand) = 0."""
        B = z.shape[0]
        zf = z.float()
        try:
            body = self.vposer.decode(zf, output_type='aa')
        except TypeError:
            out = self.vposer.decode(zf)
            body = out['pose_body'] if isinstance(out, dict) else out
        body = body.reshape(B, 21, 3)
        hands = torch.zeros(B, 2, 3, device=z.device, dtype=body.dtype)
        return torch.cat([body, hands], dim=1)


class BasicBlock3D(nn.Module):
    """Khoi Conv3d residual nhe tren luoi voxel 8x8x8 de lien ket cac mat phang do sau."""

    def __init__(self, dim):
        super().__init__()
        self.c1 = nn.Conv3d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(16, dim)
        self.c2 = nn.Conv3d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(16, dim)
        nn.init.zeros_(self.c2.weight)                         # luc dau = identity

    def forward(self, x):
        h = F.relu(self.n1(self.c1(x)), inplace=True)
        return x + self.n2(self.c2(h))


class AdaptiveHyperGCNLayer(nn.Module):
    """Sieu do thi Thich nghi theo Du lieu (Data-Adaptive Confidence-Guided HyperGCN) tren 17 token khop H36M:
      - Cong do tin cay dong tung khop g = sigmoid(MLP([jt, uvd, conf2d])) in (0, 1)^D:
        + Chan khop bi che / nhieu phat tin rac vao xuong va sieu canh (source gating);
        + Dieu chinh ma tran truyen tin theo tu the cua mau hien tai (data-dependent bone attention);
        + Truyen manh thong tin tu 10 sieu canh bo phan (part) va 5 chuoi chi (body) vao khop bi che (target gating).
      - Tra (jt_moi (B, 17, D), body_tok (B, 5, D), part_tok (B, 10, D)).
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Ma tran ke xuong H36M-17 (co them duong cheo) va ma tran thuoc 10 part / 5 body
        adj = h36m_adjacency(N_JOINT).float() + torch.eye(N_JOINT)
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer('bone_adj', adj, persistent=False)                         # (17, 17)
        self.register_buffer('bone_mask', h36m_adjacency(N_JOINT) | torch.eye(N_JOINT, dtype=torch.bool),
                             persistent=False)

        hp = torch.zeros(N_JOINT, len(V7_PARTS))
        for e, js in enumerate(V7_PARTS):
            hp[js, e] = 1.0
        hb = torch.zeros(N_JOINT, len(V7_BODY))
        for e, js in enumerate(V7_BODY):
            hb[js, e] = 1.0
        self.register_buffer('Hp_mask', hp, persistent=False)                           # (17, 10)
        self.register_buffer('Hb_mask', hb, persistent=False)                           # (17, 5)

        # Cong do tin cay dong theo du lieu: doc [jt (D), uvd (3), conf2d (1)]
        self.conf_gate = nn.Sequential(
            nn.Linear(dim + 4, dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(dim // 2, dim),
        )
        nn.init.zeros_(self.conf_gate[-1].weight)
        nn.init.constant_(self.conf_gate[-1].bias, 1.0)                                 # luc dau sigmoid(1) ~ 0.73

        # Nhanh 1: Do thi xuong ket hop attention phu thuoc du lieu
        self.norm_in = nn.LayerNorm(dim)
        self.q_bone = nn.Linear(dim, dim, bias=False)
        self.k_bone = nn.Linear(dim, dim, bias=False)
        self.v_bone = nn.Linear(dim, dim)
        self.alpha_dyn = nn.Parameter(torch.tensor(0.2))                                # pha giua ke tinh va dong

        # Nhanh 2: 10 sieu canh bo phan (part) co trong so tin cay
        self.part_in = nn.Linear(dim, dim)
        self.part_mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

        # Nhanh 3: 5 sieu canh chuoi chi (body: than, tay P, tay T, chan P, chan T) co trong so tin cay
        self.body_in = nn.Linear(dim, dim)
        self.body_mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

        # Tron 3 nhanh + FFN
        self.fuse_out = nn.Sequential(nn.LayerNorm(3 * dim), nn.Linear(3 * dim, dim))
        self.gamma1 = nn.Parameter(1e-3 * torch.ones(dim))
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.gamma2 = nn.Parameter(1e-3 * torch.ones(dim))

    def forward(self, jt, uvd, conf2d):
        """jt (B, 17, D), uvd (B, 17, 3) in [0, 1], conf2d (B, 17, 1) in [0, 1]."""
        B, J, D = jt.shape
        h = self.norm_in(jt)
        # Cong tin cay dong tung khop tren tung mau
        g = torch.sigmoid(self.conf_gate(torch.cat([h, uvd.to(h.dtype), conf2d.to(h.dtype)], dim=-1)))  # (B, 17, D)
        g_scalar = g.mean(dim=-1, keepdim=True)                                         # (B, 17, 1) trong so gop sieu canh
        hg = h * g                                                                      # loc nhieu truoc khi phat tin

        # ---- 1) Do thi xuong: A_static + alpha * Softmax(Q K^T) ----
        q = self.q_bone(hg).reshape(B, J, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_bone(hg).reshape(B, J, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_bone(hg).reshape(B, J, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        scores = scores.masked_fill(~self.bone_mask[None, None], float('-inf'))
        attn_dyn = torch.softmax(scores, dim=-1)
        a_stat = self.bone_adj[None, None].to(h.dtype)
        w_dyn = torch.sigmoid(self.alpha_dyn)
        attn = (1.0 - w_dyn) * a_stat + w_dyn * attn_dyn
        out_bone = (attn @ v).transpose(1, 2).reshape(B, J, D)

        # ---- 2) Sieu canh bo phan (10 part): gop co trong so tin cay g_scalar ----
        wp = self.Hp_mask[None].to(h.dtype) * (g_scalar + 1e-3)                         # (B, 17, 10)
        wp_norm = wp / wp.sum(dim=1, keepdim=True).clamp_min(1e-6)
        part_tok = torch.einsum('bje,bjd->bed', wp_norm, self.part_in(hg))              # (B, 10, D)
        part_tok = part_tok + self.part_mlp(part_tok)
        hp_back = self.Hp_mask.to(h.dtype) / self.Hp_mask.sum(dim=-1, keepdim=True).clamp_min(1.0).to(h.dtype)
        out_part = torch.einsum('je,bed->bjd', hp_back, part_tok)                       # (B, 17, D)

        # ---- 3) Sieu canh chuoi chi (5 body): gop co trong so tin cay g_scalar ----
        wb = self.Hb_mask[None].to(h.dtype) * (g_scalar + 1e-3)                         # (B, 17, 5)
        wb_norm = wb / wb.sum(dim=1, keepdim=True).clamp_min(1e-6)
        body_tok = torch.einsum('bje,bjd->bed', wb_norm, self.body_in(hg))              # (B, 5, D)
        body_tok = body_tok + self.body_mlp(body_tok)
        hb_back = self.Hb_mask.to(h.dtype) / self.Hb_mask.sum(dim=-1, keepdim=True).clamp_min(1.0).to(h.dtype)
        out_body = torch.einsum('je,bed->bjd', hb_back, body_tok)                       # (B, 17, D)

        # Khop co do tin cay thap (1 - g lon) nhan bu dap manh hon tu sieu canh part / body
        target_boost = 1.0 + 0.5 * (1.0 - g)
        merged = self.fuse_out(torch.cat([out_bone, out_part * target_boost, out_body * target_boost], dim=-1))
        jt = jt + self.gamma1 * merged
        jt = jt + self.gamma2 * self.ffn(self.norm_ffn(jt))
        return jt, body_tok, part_tok


class V10Head(nn.Module):
    """Khoi The tich 3D (8x8x8) + 4 lop Decoder Lay mau Dong (3D-Volume + F16) + AdaptiveHyperGCN + VPoser SMPL."""
    get_camera_trans = V8Head.get_camera_trans
    get_coord = V8Head.get_coord
    smpl_from_aa = V8Head.smpl_from_aa

    def __init__(self, dim=256, n_enc=3, n_dec=4, vol_depth=8, img_ch=2048, f16_ch=1024, vposer_ckpt=''):
        super().__init__()
        self.dim = int(dim)
        self.n_dec = int(n_dec)
        self.vol_d = int(vol_depth)

        # ---- 1) Encoder anh 2D (F8: 2048x8x8 -> 64 token) va F16 (1024x16x16) ----
        self.img_proj = nn.Conv2d(img_ch, dim, 1)
        self.img_enc = nn.ModuleList([EncLayer(dim) for _ in range(int(n_enc))])
        self.f16_proj = nn.Conv2d(f16_ch, dim, 1)

        # ---- 2) Khoi Dac trung The tich 3D (8x8x8) + Neo do sau phi tuyen (JOTR / DPMesh) ----
        self.conv2d_to_3d = nn.Conv2d(img_ch, dim * self.vol_d, 1)
        self.exponent = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))
        self.conv_3d_coord = nn.Sequential(
            nn.Conv3d(dim + 3, dim, 1),
            BasicBlock3D(dim),
        )
        self.refine_3d = nn.ModuleList([EncLayer(dim, ff=1024) for _ in range(int(n_enc))])

        # ---- 3) Khoi tao 17 query khop tu 2D detector + F16 + Noi suy vi tri 2.5D ban dau ----
        self.emb2d = nn.Sequential(nn.Linear(3, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim))
        self.miss2d_tok = nn.Parameter(torch.zeros(1, 1, dim))
        self.joint_id = nn.Embedding(N_JOINT, dim)
        self.init_norm = nn.LayerNorm(dim)
        self.init_layer = DecLayer(dim)
        # Du doan (du, dv) cho khop co detector, (u0, v0) cho khop bi che va do sau ban dau d0
        self.init_uvd_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 3))
        nn.init.normal_(self.init_uvd_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.init_uvd_head[-1].bias)

        # ---- 4) Decoder 4 lop + Lay mau Kep (3D-Volume + F16) + AdaptiveHyperGCN ----
        self.smpl_tgt = nn.Parameter(torch.zeros(1, N_SMPL_TOK, dim))
        self.smpl_qpos = nn.Embedding(N_SMPL_TOK, dim)
        self.joint_qpos = nn.Embedding(N_JOINT, dim)
        self.decoder = nn.ModuleList([DecLayer(dim) for _ in range(self.n_dec)])
        self.dec_norm = nn.LayerNorm(dim)
        self.ad_hgcn = nn.ModuleList([AdaptiveHyperGCNLayer(dim) for _ in range(self.n_dec)])

        # Cap nhat toa do 2.5D (u, v, d) in (0, 1)^3 theo inverse_sigmoid (nhu JOTR)
        self.uvd_regress = nn.ModuleList([nn.Linear(dim + 3, 3) for _ in range(self.n_dec)])
        for m in self.uvd_regress:
            nn.init.normal_(m.weight, std=1e-3)
            nn.init.zeros_(m.bias)
        self.fc_pos_3d = nn.Sequential(
            nn.Linear(3, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        # Tron dac trung lay mau The tich 3D (8x8x8) va F16 (16x16)
        self.sample_fuse = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim))
        self.sample_to_joint = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n_dec)])
        for m in self.sample_to_joint:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        # Hoi quy khop 3D phu (met, root-relative) theo tang tu 17 token khop (loss ctx_3d + v10_j17_layers)
        self.j3d_head = CascadeHead(dim, 3, self.n_dec)

        # ---- 5) Dau ra VPoser + Root 6D + Shape + Camera cong don theo tang ----
        self.vposer = VPoserDecoder(vposer_ckpt)
        # Gop 5 token chuoi chi (body_tok) va 5 khop khung than-hong (Pelvis, R_Hip, L_Hip, Torso, Neck)
        self.body_readout = nn.Sequential(nn.LayerNorm(5 * dim), nn.Linear(5 * dim, dim))
        self.root_readout = nn.Sequential(nn.LayerNorm(5 * dim), nn.Linear(5 * dim, dim))

        mean = np.load(SMPL_MEAN_PARAMS)
        init_root6d = torch.from_numpy(mean['pose'][:6].astype('float32')).reshape(1, 6)
        init_shape = torch.from_numpy(mean['shape'][:].astype('float32')).reshape(1, 10)
        self.root_pose_head = CascadeHead(dim, 6, self.n_dec, init=init_root6d)
        self.vposer_head = CascadeHead(dim, VPOSER_DIM, self.n_dec)
        self.shape_head = CascadeHead(dim, 10, self.n_dec, init=init_shape)
        self.cam_head = CascadeHead(dim, 3, self.n_dec)

        for p in (list(self.img_enc.parameters()) + list(self.refine_3d.parameters())
                  + list(self.decoder.parameters()) + list(self.init_layer.parameters())):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # ---- SMPL ----
        from utils.smpl import SMPL as SMPLModel
        self.human_model = SMPLModel()
        self.human_model_layer = self.human_model.layer['neutral']
        self.register_buffer('joint_regressor_t', torch.from_numpy(self.human_model.joint_regressor).float())

    def get_relative_depth_anchor(self, k, map_size=8, dtype=torch.float32):
        """Neo toa do 3D (Z = r^k, Y, X) kich thuoc (1, 3, D, H, W), giong JOTR/common/model.py:145."""
        r = torch.arange(map_size, dtype=torch.float32, device=k.device) / float(map_size)
        y_map = r.reshape(1, 1, 1, map_size, 1).expand(1, 1, map_size, map_size, map_size)
        x_map = r.reshape(1, 1, 1, 1, map_size).expand(1, 1, map_size, map_size, map_size)
        z_map = torch.pow(r, k.float()).reshape(1, 1, map_size, 1, 1).expand(1, 1, map_size, map_size, map_size)
        return torch.cat([z_map, y_map, x_map], dim=1).to(dtype)

    def sample_dual_3d_2d(self, dense_feat, f16_feat, uvd, exponent):
        """Lay mau dong tren Khoi The tich 3D (B, D, 8, 8, 8) tai (2u-1, 2v-1, 2*d^k-1) VA F16 (B, D, 16, 16) tai
        (2u-1, 2v-1). Tra (sample_feat (B, 17, D), sample_pos (B, 17, D))."""
        xy_grid = uvd[..., :2].float() * 2.0 - 1.0                                      # (B, 17, 2) in [-1, 1]
        z_grid = torch.pow(uvd[..., 2:3].float().clamp(1e-5, 1.0), exponent.float()) * 2.0 - 1.0
        grid_3d = torch.cat([xy_grid, z_grid], dim=-1)[:, None, None, :, :]             # (B, 1, 1, 17, 3)

        feat_3d = F.grid_sample(dense_feat.float(), grid_3d, mode='bilinear',
                                align_corners=True).squeeze(2).squeeze(2).transpose(1, 2)   # (B, 17, D)
        feat_2d = F.grid_sample(f16_feat.float(), xy_grid[:, :, None, :], mode='bilinear',
                                align_corners=False).squeeze(-1).transpose(1, 2)            # (B, 17, D)
        feat = self.sample_fuse(torch.cat([feat_3d, feat_2d], dim=-1).to(dense_feat.dtype))
        pos = self.fc_pos_3d(inverse_sigmoid(uvd.float()).to(dense_feat.dtype))
        return feat, pos

    def vposer_smpl_outputs(self, root_6d, vposer_z, shape, cam_raw, is_train):
        """root_6d (B, 6) + vposer_z (B, 32) + shape (B, 10) + cam_raw (B, 3) -> VPoser -> SMPL."""
        B = root_6d.shape[0]
        root_aa = rotmat_to_aa_safe(rot6d_to_rotmat(root_6d.float())).reshape(B, 1, 3)
        body_aa = self.vposer(vposer_z)                                                 # (B, 23, 3)
        aa = torch.cat([root_aa, body_aa], dim=1)                                       # (B, 24, 3)
        out = self.smpl_from_aa(aa, shape, cam_raw, is_train)
        out['v10_vposer_z'] = vposer_z
        return out

    def forward(self, fmap, f16, op_xy, op_mask, is_train=True):
        """fmap (B, 2048, 8, 8), f16 (B, 1024, 16, 16); op_xy (B, 17, 2) [-1, 1], op_mask (B, 17, 1) [0, 1]."""
        B, _, h, w = fmap.shape
        D = self.dim
        dtype = fmap.dtype

        # ---- [1] Bo nho anh 2D (64 token) va F16 (16x16) ----
        mem2d = self.img_proj(fmap).flatten(2).transpose(1, 2)                          # (B, 64, D)
        mpos2d = sine_pos_2d(h, w, D, mem2d.device, dtype)[None].expand(B, -1, -1)      # (B, 64, D)
        for layer in self.img_enc:
            mem2d = layer(mem2d, mpos2d)
        f16_feat = self.f16_proj(f16)                                                   # (B, D, 16, 16)

        # ---- [2] Khoi Dac trung The tich 3D (8x8x8 = 512 voxel) ----
        v3d = self.conv2d_to_3d(fmap).reshape(B, D, self.vol_d, h, w)
        exponent = self.exponent.clamp(1.0, 20.0)
        anchor = self.get_relative_depth_anchor(exponent, self.vol_d, dtype).expand(B, -1, -1, -1, -1)
        v3d = self.conv_3d_coord(torch.cat([v3d, anchor], dim=1))                       # (B, D, 8, 8, 8)
        v3d_tok = v3d.flatten(2).transpose(1, 2)                                        # (B, 512, D)
        pos3d = sincos_1d(v3d_tok.shape[1], D, v3d_tok.device, dtype)
        for layer in self.refine_3d:
            v3d_tok = layer(v3d_tok, pos3d)
        dense_feat = v3d_tok.transpose(1, 2).reshape(B, D, self.vol_d, h, w).contiguous()

        # ---- [3] Khoi tao 17 token khop tu 2D detector + Noi suy vi tri (u0, v0, d0) ----
        vis = (op_mask > 0).to(dtype)                                                   # (B, 17, 1)
        xy_in = (op_xy * vis).to(dtype)                                                 # (B, 17, 2) in [-1, 1]
        img_j0 = F.grid_sample(f16_feat, xy_in[:, :, None, :], mode='bilinear',
                               align_corners=False).squeeze(-1).transpose(1, 2)         # (B, 17, D)
        e2d = self.emb2d(torch.cat([xy_in, op_mask.to(dtype)], dim=-1))
        jt = self.init_norm(img_j0 * vis + e2d + (1.0 - vis) * self.miss2d_tok + self.joint_id.weight[None])
        j_qpos = self.joint_qpos.weight[None].expand(B, -1, -1)
        jt = self.init_layer(jt, j_qpos, mem2d, mpos2d)

        # Vi tri 2.5D ban dau uvd in (0, 1)^3: khop co detector neo quanh op_xy, khop bi che do mang tu noi suy
        raw_uvd0 = self.init_uvd_head(jt)
        det_uv = ((op_xy.float() + 1.0) * 0.5).clamp(1e-4, 1.0 - 1e-4)
        uv0_vis = torch.sigmoid(inverse_sigmoid(det_uv) + raw_uvd0[..., :2].float())
        uv0_occ = torch.sigmoid(raw_uvd0[..., :2].float())
        uv0 = vis.float() * uv0_vis + (1.0 - vis.float()) * uv0_occ
        d0 = torch.sigmoid(raw_uvd0[..., 2:3].float())
        uvd = torch.cat([uv0, d0], dim=-1)                                              # (B, 17, 3) in (0, 1)^3

        sample_feat, sample_pos = self.sample_dual_3d_2d(dense_feat, f16_feat, uvd, exponent)

        # ---- [4] 4 lop Decoder: Lay mau The tich 3D + AdaptiveHyperGCN + VPoser Cascade ----
        tgt = torch.cat([self.smpl_tgt.expand(B, -1, -1), jt], dim=1)                   # (B, 20, D)
        qpos = torch.cat([self.smpl_qpos.weight[None].expand(B, -1, -1), j_qpos], dim=1)

        root_6d = vposer_z = shape = cam = j3d = None
        per_layer = []
        rp_xys = [uvd[..., :2] * 2.0 - 1.0]                                             # theo doi vi tri 2D [-1, 1]
        j17_layers = []

        for i, layer in enumerate(self.decoder):
            mem = torch.cat([mem2d, sample_feat], dim=1)                                # (B, 64 + 17, D)
            mpos = torch.cat([mpos2d, sample_pos], dim=1)
            tgt = self.dec_norm(layer(tgt, qpos, mem, mpos))

            smpl_tok = tgt[:, :N_SMPL_TOK]                                              # (B, 3, D): pose, shape, cam
            jt = tgt[:, N_SMPL_TOK:]                                                    # (B, 17, D)

            # AdaptiveHyperGCN: loc tin theo do tin cay du lieu & truyen tin tren 16 xuong + 10 part + 5 body
            jt, body_tok, _ = self.ad_hgcn[i](jt, uvd.detach(), op_mask)

            # Cap nhat toa do 2.5D (u, v, d) theo inverse_sigmoid (JOTR) va lay mau lai 3D-Volume + F16
            inv_uvd = inverse_sigmoid(uvd.float())
            off_uvd = self.uvd_regress[i](torch.cat([jt.float(), inv_uvd], dim=-1))
            uvd = torch.sigmoid(inv_uvd + off_uvd)                                      # (B, 17, 3) in (0, 1)^3
            rp_xys.append(uvd[..., :2] * 2.0 - 1.0)

            sample_feat, sample_pos = self.sample_dual_3d_2d(dense_feat, f16_feat, uvd, exponent)
            jt = jt + self.sample_to_joint[i](sample_feat)
            tgt = torch.cat([smpl_tok, jt], dim=1)

            # Khop 3D phu (met, root-relative) cua 17 query khop
            j3d = self.j3d_head(i, jt, j3d)
            j3d_rel = j3d - j3d[:, 0:1]
            j17_layers.append(j3d_rel)

            # Readout 5 chuoi chi & khung Hong-Than cong vao token pose -> Cascade VPoser + Root 6D
            body_sum = self.body_readout(body_tok.flatten(1))                           # (B, D)
            root_sum = self.root_readout(jt[:, [0, 1, 4, 7, 8]].flatten(1))             # (B, D)
            pose_tok = smpl_tok[:, 0]
            root_6d = self.root_pose_head(i, pose_tok + root_sum, root_6d)              # (B, 6)
            vposer_z = self.vposer_head(i, pose_tok + body_sum, vposer_z)               # (B, 32)
            shape = self.shape_head(i, smpl_tok[:, 1], shape)                           # (B, 10)
            cam = self.cam_head(i, smpl_tok[:, 2], cam)                                 # (B, 3)
            per_layer.append((root_6d, vposer_z, shape, cam))

        result = self.vposer_smpl_outputs(*per_layer[-1], is_train=is_train)
        result['refine_history'] = [self.vposer_smpl_outputs(*it, is_train=is_train)
                                    for it in per_layer[:-1]] if is_train else []
        j3d_final = j17_layers[-1]
        result['ctx_3d'] = j3d_final
        result['lifted_joints_3d'] = j3d_final.detach()                                 # cot LIFT = khop 3D query
        result['privileged_3d'] = j3d_final
        result['feat_joint'] = jt
        result['feat_global'] = tgt[:, :N_SMPL_TOK].mean(1)
        if is_train:
            # Giam sat vi tri lay mau 2D [-1, 1] qua base._v9_reproj_loss (log rp2d: px_dau -> px_cuoi)
            result['v9_rp_xy'] = torch.stack(rp_xys[1:], dim=1)                         # (B, L, 17, 2)
            result['v9_rp_ref'] = rp_xys[0].detach()
            result['v10_j17_layers'] = j17_layers
        return result


class V10Net(nn.Module):
    """Vo boc FuseKin v10 (thay ARTS khi MODEL.v10_e2e): ResNet-50 + heatmap 2D + V10Head (VPoser + 3D Volume + AdaptiveHyperGCN)."""

    def __init__(self):
        super().__init__()
        self.mode = 'v10'
        self.backbone = ResNetBackbone(cfg.MODEL.resnet_type)
        self.backbone.enable_heatmap_fusion(N_JOINT)
        bb_src = cfg.MODEL.get('backbone_pretrained', '')
        if bb_src:
            self.backbone.load_pretrained(
                bb_src, spin_checkpoint=cfg.MODEL.get('spin_checkpoint',
                                                      'data_final/base_data/spin_model_checkpoint.pth.tar'))
        else:
            print('[CANH BAO] V10: backbone KHONG nap pretrain -> dac trung anh ngau nhien.')

        self.smpl_model = V10Head(
            dim=int(cfg.MODEL.v10_dim),
            n_enc=int(cfg.MODEL.v10_enc_layers),
            n_dec=int(cfg.MODEL.v10_dec_layers),
            vol_depth=int(cfg.MODEL.v10_vol_depth),
            vposer_ckpt=str(cfg.MODEL.get('v10_vposer_ckpt', '')),
        )
        self.joint_drop = float(cfg.MODEL.v10_joint_drop)
        print(f'===> FuseKin v10 (CHI ANH + 2D, KHONG Lifter/Refiner/IK): dim {cfg.MODEL.v10_dim} | '
              f'enc 2D+3D {cfg.MODEL.v10_enc_layers} lop | khoi 3D 8x8x{cfg.MODEL.v10_vol_depth} | '
              f'dec {cfg.MODEL.v10_dec_layers} lop (AdaptiveHyperGCN + VPoser) | che 2D {self.joint_drop}')

    def train(self, mode=True):
        super().train(mode)
        self.smpl_model.vposer.eval()
        return self

    def forward(self, image, joints, is_train=True, joints_mask=None, pose_2d=None, cam_bbox=None):
        """joints = 2D detector (B, 17, 2|3) [-1, 1]; joints_mask (B, 17, 1)."""
        B = image.shape[0]
        p2 = joints[:, 0] if joints.dim() == 4 else joints
        if joints_mask is None:
            m = (p2[..., 2:3] > 0).float() if p2.shape[-1] >= 3 else torch.ones_like(p2[..., :1])
        else:
            m = joints_mask.reshape(B, N_JOINT, 1).float()
        if self.training and self.joint_drop > 0:
            m = m * (torch.rand_like(m) >= self.joint_drop).float()
        heat = make_heatmaps(p2, m, size=image.shape[-1] // 4, sigma=float(cfg.MODEL.v3_heatmap_sigma))
        fmap, _, f16 = self.backbone(image, heatmap=heat, return_f16=True)              # (B,2048,8,8), (B,1024,16,16)
        return self.smpl_model(fmap, f16, p2[..., :2].float(), m, is_train=is_train)
