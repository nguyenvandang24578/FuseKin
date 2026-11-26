"""
lib/models/fusekin_v11.py -- FuseKin v11 / v11.2 (2026-11-25, nang cap thuan no-ron 2026-11-27, nguoi dung duyet):
  Reliability-Routed 3D-Kinematic Mesh Regressor voi Fusion 2 chieu (Anh <-> Khop), 100% mang no-ron (KHONG IK).
  To chuc thanh cac khoi nn.Module doc lap, de doc:
    1) CLIFFCamEmbed          : Ma hoa goc nhin lech tam cam_bbox = (dx, dy, s).
    2) F16Sampler             : Cham lay dac trung anh chi tiet tren ban do F16 (16x16) tai toa do 2D khop.
    3) ReliabilityRouter      : JRE cham diem tin cay R = (r3D, r2D) & khoi tao 17 token khop ban dau.
    4) BiCrossFusionBlock     : Fusion 2 chieu (Khop doc Anh [F8 + F16] & Anh F8 doc nguoc Khop 3D).
    5) KinematicRefineStep    : AdaptiveHyperGCN + Cong no-ron hoc duoc + Nan toa do xuong 3D + Doi diem F16.
    6) KinematicVPoserHead    : Dau ra VPoser goc cua V11 (giu tuong thich 100% checkpoint cu).
    7) PerJointNeuralSMPLHead : Dau ra phan cap VPoser tho + Phan du xoay 6D tung khop tren Sieu do thi SMPL-24.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
from models.ARTS import make_heatmaps
from models.backbones.resnet import ResNetBackbone
from models.fusekin_v8 import EncLayer, DecLayer, CascadeHead, V8Head, rot6d_to_rotmat, sine_pos_2d
from models.fusekin_v10 import (
    V10Head, VPoserDecoder, AdaptiveHyperGCNLayer, inverse_sigmoid,
    _corrupt_hyp_limbs, _wp_reproj_residual, SMPL_MEAN_PARAMS, N_SMPL_TOK, N_JOINT, VPOSER_DIM,
)
from models.hybrik_smpl import rotmat_to_aa_safe
from models.hypergraph_smpl import HGraphBlockSMPL24
from models.img_lifter import H36M_BONES
from models.teacher_student import SMPL_BONE_PAIRS_H36M

TORSO_HIP_IDX = (0, 1, 4, 7, 8)   # Pelvis, R_Hip, L_Hip, Torso, Neck
N_SMPL_ALL = 24
N_SMPL_BODY = 22                  # Pelvis (0) + 21 khop than/chi (1..21); 2 khop ban tay (22, 23) = 0


def aa_to_rotmat_safe(aa, eps=1e-6):
    """Chuyen axis-angle (N, 3) -> ma tran xoay (N, 3, 3) bang Rodrigues muot, grad huu han tai aa = 0."""
    theta = torch.sqrt((aa * aa).sum(dim=-1, keepdim=True) + eps * eps)         # (N, 1)
    k = aa / theta                                                              # (N, 3)
    kx, ky, kz = k[:, 0:1], k[:, 1:2], k[:, 2:3]
    z = torch.zeros_like(kx)
    K = torch.cat([z, -kz, ky, kz, z, -kx, -ky, kx, z], dim=-1).reshape(-1, 3, 3)
    sin_t = torch.sin(theta)[..., None]                                         # (N, 1, 1)
    cos_t = torch.cos(theta)[..., None]                                         # (N, 1, 1)
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype)[None]                  # (1, 3, 3)
    return eye + sin_t * K + (1.0 - cos_t) * torch.bmm(K, K)


class CLIFFCamEmbed(nn.Module):
    """Ma hoa thong tin vi tri bbox trong anh goc cam_bbox = (dx, dy, s) kieu CLIFF."""

    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU())
        self.to_joint = nn.Linear(dim, dim)
        self.to_root = nn.Linear(dim, dim)
        self.to_cam = nn.Linear(dim, dim)
        for m in (self.to_joint, self.to_root, self.to_cam):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, cam_bbox, dtype):
        feat = self.mlp(cam_bbox.to(dtype))                                         # (B, D)
        return self.to_joint(feat)[:, None], self.to_root(feat), self.to_cam(feat)  # (B,1,D), (B,D), (B,D)


class F16Sampler(nn.Module):
    """Cham lay mau song tuyen tren ban do F16 (B, D, 16, 16) tai 17 diem uv in (0, 1)^2."""

    def __init__(self, dim):
        super().__init__()
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim),
        )

    def forward(self, f16_feat, uv):
        xy_grid = uv.float() * 2.0 - 1.0                                            # (B, 17, 2) in [-1, 1]
        feat = F.grid_sample(f16_feat.float(), xy_grid[:, :, None, :],
                             mode='bilinear', align_corners=False)
        feat = feat.squeeze(-1).transpose(1, 2).to(f16_feat.dtype)                  # (B, 17, D)
        pos = self.pos_mlp(xy_grid.to(f16_feat.dtype))                              # (B, 17, D)
        return feat, pos


class CrossAttnBlock(nn.Module):
    """Khoi Cross-Attention 1 chieu (Query doc Key/Value) co Pre-Norm + LayerScale."""

    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, drop=0.1, layer_scale=1e-2):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )
        self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim))
        self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, q_tok, q_pos, kv_tok, kv_pos):
        qn = self.norm_q(q_tok)
        kvn = self.norm_kv(kv_tok)
        attn_out = self.attn(qn + q_pos, kvn + kv_pos, kvn, need_weights=False)[0]  # (B, Nq, D)
        q_tok = q_tok + self.gamma1 * attn_out
        q_tok = q_tok + self.gamma2 * self.mlp(self.norm_mlp(q_tok))
        return q_tok


class ReliabilityRouter(nn.Module):
    """Khoi 1: Cham diem do tin cay JRE R = (r3D, r2D) & khoi tao 17 token khop ban dau."""

    def __init__(self, dim, drop3d_sample=0.2, drop3d_joint=0.1, rr_corrupt_prob=0.3):
        super().__init__()
        self.drop3d_sample = float(drop3d_sample)
        self.drop3d_joint = float(drop3d_joint)
        self.rr_corrupt_prob = float(rr_corrupt_prob)

        self.lift_emb = nn.Sequential(nn.Linear(9, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim))
        self.lift_to_joint = nn.Linear(dim, dim)
        nn.init.zeros_(self.lift_to_joint.weight)
        nn.init.zeros_(self.lift_to_joint.bias)
        self.miss3d_tok = nn.Parameter(torch.zeros(1, 1, dim))

        self.jre_img_norm = nn.LayerNorm(dim)
        self.jre_head = nn.Sequential(
            nn.Linear(dim + 15, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 2),
        )

        self.emb2d = nn.Sequential(nn.Linear(4, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim))
        self.miss2d_tok = nn.Parameter(torch.zeros(1, 1, dim))
        self.joint_id = nn.Embedding(N_JOINT, dim)
        self.init_norm = nn.LayerNorm(dim)

    def _sample_keep3d(self, B, device, dtype, is_train, keep_s=None):
        if bool(cfg.MODEL.get('v11_test_drop3d', False)):
            return torch.zeros(B, N_JOINT, 1, device=device, dtype=dtype)
        if not is_train:
            if keep_s is not None:
                return keep_s.to(device=device, dtype=dtype).expand(B, N_JOINT, 1)
            return torch.ones(B, N_JOINT, 1, device=device, dtype=dtype)
        if keep_s is None:
            keep_s = (torch.rand(B, 1, 1, device=device) >= self.drop3d_sample).to(dtype)
        else:
            keep_s = keep_s.to(device=device, dtype=dtype)
        keep_j = (torch.rand(B, N_JOINT, 1, device=device) >= self.drop3d_joint).to(dtype)
        return keep_s * keep_j                                                      # (B, 17, 1)

    def forward(self, f16_feat, sampler, op_xy, op_mask, lift, hyp, ref_xy, ref_logb, cam_j,
                is_train=True, keep_s=None):
        B, _, _, _ = f16_feat.shape
        dtype = f16_feat.dtype

        # 1) Gia lap loi xoay chi tren hyp luc train & lay co keep3d
        if is_train and self.rr_corrupt_prob > 0.0:
            lift, hyp = _corrupt_hyp_limbs(hyp, self.rr_corrupt_prob)
        keep3d = self._sample_keep3d(B, f16_feat.device, dtype, is_train, keep_s=keep_s)   # (B, 17, 1)

        std3 = hyp.float().std(dim=1, unbiased=False) * 10.0                        # (B, 17, 3)
        feat_lift = torch.cat([lift.float(), std3, ref_xy.float(), ref_logb.float()[..., None]], dim=-1).to(dtype)
        e3d = self.lift_to_joint(self.lift_emb(feat_lift))                          # (B, 17, D)

        # 2) JRE doi chieu anh tai diem detector + lech 2D + phan du chieu 3D -> R = (r3D, r2D)
        vis = (op_mask > 0).to(dtype)                                               # (B, 17, 1)
        xy_det = (op_xy * vis).to(dtype)                                            # (B, 17, 2)
        img_at_det = F.grid_sample(f16_feat, xy_det[:, :, None, :],
                                   mode='bilinear', align_corners=False).squeeze(-1).transpose(1, 2)
        diff_2d = ((xy_det - ref_xy.to(dtype)) * vis) * keep3d                      # (B, 17, 2)
        reproj = _wp_reproj_residual(lift, ref_xy, ref_logb, keep3d).to(dtype) * 10.0 * keep3d

        jre_in = torch.cat([
            self.jre_img_norm(img_at_det.detach()),
            xy_det,
            op_mask.to(dtype),
            feat_lift * keep3d,
            diff_2d,
            reproj,
        ], dim=-1)                                                                  # (B, 17, D + 15)
        jre_pred = torch.sigmoid(self.jre_head(jre_in))                             # (B, 17, 2) = (r3D, r2D)
        r3d = jre_pred[..., 0:1].detach()
        r2d = jre_pred[..., 1:2].detach()

        # 3) Tach vai tro: w3d cho Lifter 3D; alpha_2d chon giua xy_det va ref_xy de lay mau F16
        w3d = keep3d.float() * r3d.float()                                          # (B, 17, 1)
        r_rel = torch.cat([w3d.to(dtype), r2d.to(dtype)], dim=-1)                   # (B, 17, 2)

        alpha_2d = vis.float() * r2d.float()                                        # (B, 17, 1)
        xy_fused = alpha_2d * xy_det.float() + (1.0 - alpha_2d) * ref_xy.float()
        xy_init = torch.where(keep3d > 0.5, xy_fused, xy_det.float()).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        uv0 = ((xy_init + 1.0) * 0.5).clamp(1e-4, 1.0 - 1e-4)                       # (B, 17, 2) in (0, 1)^2

        # 4) Gop 5 nguon thong tin thanh 17 token khop ban dau
        sample_feat, sample_pos = sampler(f16_feat, uv0)                            # (B, 17, D)
        has_2d = ((vis > 0) | (keep3d > 0)).to(dtype)
        e2d = self.emb2d(torch.cat([(uv0 * 2.0 - 1.0).to(dtype) * has_2d, vis, alpha_2d.to(dtype) * keep3d], dim=-1))
        lift_j = keep3d * (w3d.to(dtype) * e3d) + (1.0 - keep3d) * self.miss3d_tok

        joint_tok = self.init_norm(
            sample_feat * has_2d
            + e2d
            + (1.0 - has_2d) * self.miss2d_tok
            + self.joint_id.weight[None]
            + cam_j
            + lift_j
        )                                                                           # (B, 17, D)

        aux = {
            'jre_pred': jre_pred,
            'lift_used': lift,
            'keep3d': keep3d,
            'xy_det': xy_det,
            'vis': vis,
        }
        return joint_tok, uv0, sample_feat, sample_pos, w3d, r_rel, aux


class BiCrossFusionBlock(nn.Module):
    """Khoi 2: Fusion 2 chieu giua Khop [3 SMPL + 17 khop] va Anh [64 o F8 + 17 diem F16]."""

    def __init__(self, dim, num_heads=8, use_bi_attn=True):
        super().__init__()
        self.use_bi_attn = bool(use_bi_attn)
        self.joint_from_img = DecLayer(dim, nhead=num_heads)
        self.joint_norm = nn.LayerNorm(dim)
        if self.use_bi_attn:
            self.img_from_joint = CrossAttnBlock(dim, num_heads=num_heads, mlp_ratio=2.0, layer_scale=1e-2)

    def forward(self, smpl_tok, joint_tok, smpl_pos, joint_pos, mem2d, mpos2d, sample_feat, sample_pos):
        # Chieu 1 (Khop <- Anh): [3 SMPL | 17 khop] doc [64 o F8 | 17 diem F16]
        tgt = torch.cat([smpl_tok, joint_tok], dim=1)                               # (B, 20, D)
        qpos = torch.cat([smpl_pos, joint_pos], dim=1)                              # (B, 20, D)
        mem = torch.cat([mem2d, sample_feat], dim=1)                                # (B, 81, D)
        mpos = torch.cat([mpos2d, sample_pos], dim=1)                               # (B, 81, D)

        tgt = self.joint_norm(self.joint_from_img(tgt, qpos, mem, mpos))            # (B, 20, D)
        smpl_tok = tgt[:, :N_SMPL_TOK]                                              # (B, 3, D)
        joint_tok = tgt[:, N_SMPL_TOK:]                                             # (B, 17, D)

        # Chieu 2 (Anh <- Khop): 64 o F8 doc nguoc 17 token khop de loc nguoi muc tieu khoi nen / dam dong
        if self.use_bi_attn:
            mem2d = self.img_from_joint(mem2d, mpos2d, joint_tok, joint_pos)        # (B, 64, D)

        return smpl_tok, joint_tok, mem2d


class KinematicRefineStep(nn.Module):
    """Khoi 3 (1 vong): AdaptiveHyperGCN -> Nan toa do xuong 3D (met) -> Doi diem lay mau F16."""

    def __init__(self, dim, is_first=False, use_learn_gate=False):
        super().__init__()
        self.is_first = bool(is_first)
        self.use_learn_gate = bool(use_learn_gate) and self.is_first
        self.hyper_gcn = AdaptiveHyperGCNLayer(dim, use_rr_hgcn=True)

        if self.is_first:
            self.vis_3d_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 3))
            nn.init.normal_(self.vis_3d_head[-1].weight, std=1e-3)
            nn.init.zeros_(self.vis_3d_head[-1].bias)
            if self.use_learn_gate:
                # Cong no-ron hoc duoc g_3d in (0, 1): bias +2.0 -> luc dau ~0.88, giu lift o khop tot ngay tu ep1
                self.anchor_gate = nn.Sequential(
                    nn.Linear(dim + 7, dim // 2),
                    nn.LayerNorm(dim // 2),
                    nn.GELU(),
                    nn.Linear(dim // 2, 1),
                )
                nn.init.zeros_(self.anchor_gate[-1].weight)
                nn.init.constant_(self.anchor_gate[-1].bias, 2.0)

        self.delta_3d_head = nn.Sequential(nn.Linear(dim + 4, dim // 2), nn.GELU(), nn.Linear(dim // 2, 3))
        nn.init.normal_(self.delta_3d_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.delta_3d_head[-1].bias)

        self.uv_offset_head = nn.Linear(dim + 5, 2)
        nn.init.normal_(self.uv_offset_head.weight, std=1e-3)
        nn.init.zeros_(self.uv_offset_head.bias)

        self.sample_feedback = nn.Linear(dim, dim)
        self.geom_feedback = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.sample_feedback.weight)
        nn.init.zeros_(self.sample_feedback.bias)
        nn.init.zeros_(self.geom_feedback[-1].weight)
        nn.init.zeros_(self.geom_feedback[-1].bias)

    def forward(self, joint_tok, uv, j3d_prev, lift_masked, w3d, op_mask, r_rel, f16_feat, sampler, keep3d=None):
        dtype = joint_tok.dtype

        # 1) Truyen tin tren Sieu do thi xuong (16 xuong + 10 bo phan + 5 chuoi chi)
        hgcn_coord = torch.cat([uv.detach(), w3d.detach()], dim=-1)                 # (B, 17, 3)
        joint_tok, body_tok, _ = self.hyper_gcn(joint_tok, hgcn_coord, op_mask, r_rel=r_rel)

        # 2) Cap nhat toa do xuong 3D (met, root-relative tai Pelvis)
        j3d_vis0 = None
        w3d_eff = w3d
        if self.is_first:
            j3d_vis0 = self.vis_3d_head(joint_tok.float())                          # (B, 17, 3)
            j3d_vis0 = j3d_vis0 - j3d_vis0[:, 0:1]
            if self.use_learn_gate:
                assert keep3d is not None, 'v11_learn_gate can keep3d (B, 17, 1)'
                diff_lv = lift_masked - j3d_vis0.detach()                           # (B, 17, 3)
                dist_lv = torch.sqrt((diff_lv * diff_lv).sum(dim=-1, keepdim=True) + 1e-8)
                gate_in = torch.cat([
                    joint_tok.detach(),
                    r_rel.to(dtype),
                    op_mask.to(dtype),
                    diff_lv.to(dtype),
                    dist_lv.to(dtype),
                ], dim=-1)                                                          # (B, 17, D + 7)
                w3d_eff = keep3d.float() * torch.sigmoid(self.anchor_gate(gate_in)).float()
                delta = self.delta_3d_head(torch.cat([joint_tok.float(), lift_masked, w3d_eff], dim=-1))
                j3d = w3d_eff * lift_masked + (1.0 - w3d_eff) * j3d_vis0.detach() + delta
            else:
                delta = self.delta_3d_head(torch.cat([joint_tok.float(), lift_masked, w3d], dim=-1))
                j3d = w3d * lift_masked + (1.0 - w3d) * j3d_vis0 + delta
        else:
            delta = self.delta_3d_head(torch.cat([joint_tok.float(), j3d_prev, w3d], dim=-1))
            j3d = j3d_prev + delta
        j3d = j3d - j3d[:, 0:1]                                                     # (B, 17, 3)

        # 3) Dich diem lay mau uv tren F16 theo xuong 3D moi & bom nguoc vao joint_tok
        inv_uv = inverse_sigmoid(uv.float())
        uv = torch.sigmoid(inv_uv + self.uv_offset_head(torch.cat([joint_tok.float(), j3d.detach(), inv_uv], dim=-1)))
        sample_feat, sample_pos = sampler(f16_feat, uv)                             # (B, 17, D)

        joint_tok = (joint_tok
                     + self.sample_feedback(sample_feat)
                     + self.geom_feedback(j3d.detach().to(dtype)))                  # (B, 17, D)
        return joint_tok, body_tok, j3d, j3d_vis0, uv, sample_feat, sample_pos, w3d_eff


class KinematicVPoserHead(nn.Module):
    """Khoi 4A (V11 goc): Anh xa Khung xuong 3D (khop + 16 huong xuong + 16 do dai xuong) -> VPoser SMPL."""

    def __init__(self, dim, n_dec, vposer_ckpt=''):
        super().__init__()
        n_bones = len(H36M_BONES)
        bone_p = torch.tensor([p for p, _ in H36M_BONES], dtype=torch.long)
        bone_c = torch.tensor([c for _, c in H36M_BONES], dtype=torch.long)
        self.register_buffer('bone_p_idx', bone_p, persistent=False)
        self.register_buffer('bone_c_idx', bone_c, persistent=False)

        self.vposer = VPoserDecoder(vposer_ckpt)
        self.body_readout = nn.Sequential(nn.LayerNorm(5 * dim), nn.Linear(5 * dim, dim))
        self.root_readout = nn.Sequential(nn.LayerNorm(5 * dim), nn.Linear(5 * dim, dim))

        self.ctx_to_root = nn.Sequential(
            nn.Linear(N_JOINT * 3, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.ctx_to_vposer = nn.Sequential(
            nn.Linear((N_JOINT + n_bones) * 3, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.len_to_shape = nn.Sequential(
            nn.Linear(n_bones, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim),
        )
        for m in (self.ctx_to_root[-1], self.ctx_to_vposer[-1], self.len_to_shape[-1]):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

        mean = np.load(SMPL_MEAN_PARAMS)
        init_root6d = torch.from_numpy(mean['pose'][:6].astype('float32')).reshape(1, 6)
        init_shape = torch.from_numpy(mean['shape'][:].astype('float32')).reshape(1, 10)
        self.root_pose_head = CascadeHead(dim, 6, n_dec, init=init_root6d)
        self.vposer_head = CascadeHead(dim, VPOSER_DIM, n_dec)
        self.shape_head = CascadeHead(dim, 10, n_dec, init=init_shape)
        self.cam_head = CascadeHead(dim, 3, n_dec)

    @staticmethod
    def init_state():
        return (None, None, None, None)

    def step(self, layer_idx, smpl_tok, joint_tok, body_tok, j3d, cam_r, cam_c, prev_params,
             keep3d=None, is_train=True):
        dtype = joint_tok.dtype
        root_6d, vposer_z, shape, cam = prev_params

        # Rut 16 vector huong xuong don vi (giu grad) va 16 do dai xuong (detach cho shape)
        bone_vec = j3d[:, self.bone_c_idx] - j3d[:, self.bone_p_idx]                # (B, 16, 3)
        bone_len = torch.norm(bone_vec, dim=-1, keepdim=True).clamp_min(1e-6)       # (B, 16, 1)
        bone_dir = bone_vec / bone_len                                              # (B, 16, 3)

        ctx_r = self.ctx_to_root(j3d.to(dtype).flatten(1))                          # (B, D)
        ctx_v = self.ctx_to_vposer(torch.cat([j3d, bone_dir], dim=1).to(dtype).flatten(1))   # (B, 33, 3) -> 99
        ctx_s = self.len_to_shape(bone_len.squeeze(-1).detach().to(dtype))          # (B, D)

        body_sum = self.body_readout(body_tok.flatten(1))                           # (B, D)
        root_sum = self.root_readout(joint_tok[:, TORSO_HIP_IDX].flatten(1))        # (B, D)
        pose_tok, shape_tok, cam_tok = smpl_tok[:, 0], smpl_tok[:, 1], smpl_tok[:, 2]

        root_6d = self.root_pose_head(layer_idx, pose_tok + root_sum + ctx_r + cam_r, root_6d)
        vposer_z = self.vposer_head(layer_idx, pose_tok + body_sum + ctx_v, vposer_z)
        shape = self.shape_head(layer_idx, shape_tok + ctx_s, shape)
        cam = self.cam_head(layer_idx, cam_tok + cam_c, cam)
        return root_6d, vposer_z, shape, cam

    def decode(self, owner, params, is_train, cam_bbox=None):
        root_6d, vposer_z, shape, cam_raw = params
        B = root_6d.shape[0]
        root_aa = rotmat_to_aa_safe(rot6d_to_rotmat(root_6d.float())).reshape(B, 1, 3)
        body_aa = self.vposer(vposer_z)                                             # (B, 23, 3)
        aa = torch.cat([root_aa, body_aa], dim=1)                                   # (B, 24, 3)
        out = owner.smpl_from_aa_v10(aa, shape, cam_raw, is_train, cam_bbox)
        out['v10_vposer_z'] = vposer_z
        return out


class PerJointNeuralSMPLHead(nn.Module):
    """Khoi 4B (V11.2, 100% thuan no-ron, KHONG IK):
      - Buoc tho: VPoser + Root 6D co Visual-Pose Dropout (chong pose_tok at khung xuong j3d).
      - Buoc tinh: 24 token khop SMPL doc cap xuong cha-con (joint_tok + toa do/vector xuong 3D + R_tho),
        truyen tin tren Sieu do thi SMPL-24 (HGraphBlockSMPL24), du doan phan du xoay 6D dR_k tung khop:
        R_cuoi = R_tho @ dR_k.
    """

    def __init__(self, dim, n_dec, vposer_ckpt='', pose_tok_drop=0.3):
        super().__init__()
        self.pose_tok_drop = float(pose_tok_drop)

        n_bones = len(H36M_BONES)
        bone_p = torch.tensor([p for p, _ in H36M_BONES], dtype=torch.long)
        bone_c = torch.tensor([c for _, c in H36M_BONES], dtype=torch.long)
        self.register_buffer('bone_p_idx', bone_p, persistent=False)
        self.register_buffer('bone_c_idx', bone_c, persistent=False)

        smpl_p = torch.tensor([p for p, _ in SMPL_BONE_PAIRS_H36M], dtype=torch.long)
        smpl_c = torch.tensor([c for _, c in SMPL_BONE_PAIRS_H36M], dtype=torch.long)
        self.register_buffer('smpl_p_idx', smpl_p, persistent=False)               # (24,)
        self.register_buffer('smpl_c_idx', smpl_c, persistent=False)               # (24,)

        # 1) Nhanh VPoser + Root + Shape + Cam tho
        self.vposer = VPoserDecoder(vposer_ckpt)
        self.body_readout = nn.Sequential(nn.LayerNorm(5 * dim), nn.Linear(5 * dim, dim))
        self.root_readout = nn.Sequential(nn.LayerNorm(5 * dim), nn.Linear(5 * dim, dim))

        self.ctx_to_root = nn.Sequential(
            nn.Linear(N_JOINT * 3, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.ctx_to_vposer = nn.Sequential(
            nn.Linear((N_JOINT + n_bones) * 3, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.len_to_shape = nn.Sequential(
            nn.Linear(n_bones, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim),
        )
        for m in (self.ctx_to_root[-1], self.ctx_to_vposer[-1], self.len_to_shape[-1]):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

        mean = np.load(SMPL_MEAN_PARAMS)
        init_root6d = torch.from_numpy(mean['pose'][:6].astype('float32')).reshape(1, 6)
        init_shape = torch.from_numpy(mean['shape'][:].astype('float32')).reshape(1, 10)
        self.root_pose_head = CascadeHead(dim, 6, n_dec, init=init_root6d)
        self.vposer_head = CascadeHead(dim, VPOSER_DIM, n_dec)
        self.shape_head = CascadeHead(dim, 10, n_dec, init=init_shape)
        self.cam_head = CascadeHead(dim, 3, n_dec)

        # 2) Nhanh tinh chinh goc xoay 6D tung khop tren Sieu do thi SMPL-24
        self.smpl_joint_id = nn.Embedding(N_SMPL_ALL, dim)
        self.bone_pair_tok = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim))
        self.bone_kin_emb = nn.Sequential(
            nn.Linear(12, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.coarse_rot_emb = nn.Linear(9, dim)
        self.smpl_hgcn = nn.ModuleList([
            HGraphBlockSMPL24(dim, mlp_ratio=2.0, layer_scale_init_value=1e-3) for _ in range(n_dec)
        ])
        # Khoa ma tran sieu canh p_adj / b_adj de tranh troi tong cot ve 0 lam no BatchNorm
        for name, p in self.smpl_hgcn.named_parameters():
            if name.endswith('p_adj') or name.endswith('b_adj'):
                p.requires_grad_(False)

        init_ident6d = torch.eye(3, dtype=torch.float32)[:, :2].reshape(1, 6)       # [[1, 0, 0, 1, 0, 0]] = I_3
        self.res6d_head = CascadeHead(dim, 6, n_dec, init=init_ident6d)

    @staticmethod
    def init_state():
        return (None, None, None, None, None)

    def _coarse_rotmats(self, root_6d, vposer_z):
        B = root_6d.shape[0]
        R_root = rot6d_to_rotmat(root_6d.float()).reshape(B, 1, 3, 3)               # (B, 1, 3, 3)
        body_aa = self.vposer(vposer_z)                                             # (B, 23, 3)
        R_body = aa_to_rotmat_safe(body_aa.float().reshape(-1, 3)).reshape(B, 23, 3, 3)
        return torch.cat([R_root, R_body], dim=1)                                   # (B, 24, 3, 3)

    def step(self, layer_idx, smpl_tok, joint_tok, body_tok, j3d, cam_r, cam_c, prev_params,
             keep3d=None, is_train=True):
        B = j3d.shape[0]
        dtype = joint_tok.dtype
        root_6d, vposer_z, res_6d, shape, cam = prev_params

        # 1) Cap nhat VPoser + Root + Shape + Cam tho (co Visual-Pose Dropout tren mau co 3D)
        bone_vec = j3d[:, self.bone_c_idx] - j3d[:, self.bone_p_idx]                # (B, 16, 3)
        bone_len = torch.sqrt((bone_vec * bone_vec).sum(dim=-1, keepdim=True) + 1e-8)
        bone_dir = bone_vec / bone_len                                              # (B, 16, 3)

        ctx_r = self.ctx_to_root(j3d.to(dtype).flatten(1))                          # (B, D)
        ctx_v = self.ctx_to_vposer(torch.cat([j3d, bone_dir], dim=1).to(dtype).flatten(1))
        ctx_s = self.len_to_shape(bone_len.squeeze(-1).detach().to(dtype))          # (B, D)

        body_sum = self.body_readout(body_tok.flatten(1))                           # (B, D)
        root_sum = self.root_readout(joint_tok[:, TORSO_HIP_IDX].flatten(1))        # (B, D)
        pose_tok, shape_tok, cam_tok = smpl_tok[:, 0], smpl_tok[:, 1], smpl_tok[:, 2]

        vis_r = pose_tok + root_sum
        vis_v = pose_tok + body_sum
        if is_train and self.pose_tok_drop > 0.0 and keep3d is not None:
            has_3d = (keep3d.float().mean(dim=1) > 0.5).to(dtype)                   # (B, 1)
            drop_m = (torch.rand(B, 1, device=j3d.device) < self.pose_tok_drop).to(dtype) * has_3d
            vis_r = vis_r * (1.0 - drop_m)
            vis_v = vis_v * (1.0 - drop_m)

        root_6d = self.root_pose_head(layer_idx, vis_r + ctx_r + cam_r, root_6d)    # (B, 6)
        vposer_z = self.vposer_head(layer_idx, vis_v + ctx_v, vposer_z)             # (B, 32)
        shape = self.shape_head(layer_idx, shape_tok + ctx_s, shape)                # (B, 10)
        cam = self.cam_head(layer_idx, cam_tok + cam_c, cam)                        # (B, 3)

        # 2) Tao 24 token khop SMPL doc xuong cha-con 3D + R_tho -> HGraphBlockSMPL24 -> dR_k 6D
        R_coarse = self._coarse_rotmats(root_6d.detach(), vposer_z.detach())        # (B, 24, 3, 3)
        j_p = j3d[:, self.smpl_p_idx]                                               # (B, 24, 3)
        j_c = j3d[:, self.smpl_c_idx]                                               # (B, 24, 3)
        vec_pc = j_c - j_p                                                          # (B, 24, 3)
        len_pc = torch.sqrt((vec_pc * vec_pc).sum(dim=-1, keepdim=True) + 1e-8)     # (B, 24, 1)
        dir_pc = vec_pc / len_pc                                                    # (B, 24, 3)

        kin_in = torch.cat([j_p, j_c, vec_pc, dir_pc], dim=-1).to(dtype)            # (B, 24, 12)
        jt_pair = torch.cat([joint_tok[:, self.smpl_p_idx], joint_tok[:, self.smpl_c_idx]], dim=-1)
        q24 = (self.bone_pair_tok(jt_pair)
               + self.bone_kin_emb(kin_in)
               + self.coarse_rot_emb(R_coarse.reshape(B, N_SMPL_ALL, 9).to(dtype))
               + self.smpl_joint_id.weight[None])                                   # (B, 24, D)
        if torch.is_tensor(cam_r):
            q24 = torch.cat([q24[:, :1] + cam_r[:, None], q24[:, 1:]], dim=1)

        q24 = self.smpl_hgcn[layer_idx](q24.float().unsqueeze(1)).squeeze(1).to(dtype)
        res_6d = self.res6d_head(layer_idx, q24[:, :N_SMPL_BODY], res_6d)           # (B, 22, 6)
        return root_6d, vposer_z, res_6d, shape, cam

    def decode(self, owner, params, is_train, cam_bbox=None):
        root_6d, vposer_z, res_6d, shape, cam_raw = params
        B = root_6d.shape[0]
        R_coarse22 = self._coarse_rotmats(root_6d, vposer_z)[:, :N_SMPL_BODY]       # (B, 22, 3, 3)
        R_delta22 = rot6d_to_rotmat(res_6d.float().reshape(-1, 6)).reshape(B, N_SMPL_BODY, 3, 3)
        R_final22 = torch.matmul(R_coarse22, R_delta22)                             # (B, 22, 3, 3)

        aa22 = rotmat_to_aa_safe(R_final22.reshape(-1, 3, 3)).reshape(B, N_SMPL_BODY, 3)
        hands = torch.zeros(B, N_SMPL_ALL - N_SMPL_BODY, 3, device=aa22.device, dtype=aa22.dtype)
        aa = torch.cat([aa22, hands], dim=1)                                        # (B, 24, 3)

        out = owner.smpl_from_aa_v10(aa, shape, cam_raw, is_train, cam_bbox)
        out['v10_vposer_z'] = vposer_z
        if is_train:
            eye = torch.eye(3, device=aa22.device, dtype=torch.float32)[None, None]
            out['v11_res6d_reg'] = ((R_delta22 - eye) ** 2).mean()
        return out


class V11Head(nn.Module):
    """Mang chinh FuseKin v11: gom 4 khoi tuan tu ro rang, KHONG khoi Voxel 3D, KHONG IK."""
    get_camera_trans = V8Head.get_camera_trans
    get_coord = V8Head.get_coord
    smpl_from_aa = V8Head.smpl_from_aa
    smpl_from_aa_v10 = V10Head.smpl_from_aa_v10

    def __init__(self, dim=256, n_enc=3, n_dec=4, img_ch=2048, f16_ch=1024, vposer_ckpt='',
                 use_cam_bbox=True, proj_grad=True, use_bi_attn=True,
                 drop3d_sample=0.2, drop3d_joint=0.1, rr_corrupt_prob=0.3,
                 use_res6d_head=False, pose_tok_drop=0.0, use_learn_gate=False):
        super().__init__()
        self.dim = int(dim)
        self.n_dec = int(n_dec)
        self.use_cam_bbox = bool(use_cam_bbox)
        self.proj_grad = bool(proj_grad)
        self.use_bi_attn = bool(use_bi_attn)
        self.use_res6d_head = bool(use_res6d_head)
        self.use_learn_gate = bool(use_learn_gate)

        # 1. Ma hoa anh F8 (64 token) & F16 (16x16) + CLIFF cam_bbox
        self.img_proj = nn.Conv2d(img_ch, dim, 1)
        self.img_enc = nn.ModuleList([EncLayer(dim) for _ in range(int(n_enc))])
        self.f16_proj = nn.Conv2d(f16_ch, dim, 1)
        self.f16_sampler = F16Sampler(dim)
        self.cam_embed = CLIFFCamEmbed(dim) if self.use_cam_bbox else None

        # 2. Dinh tuyen do tin cay JRE & khoi tao 17 token khop
        self.router = ReliabilityRouter(dim, drop3d_sample=drop3d_sample,
                                        drop3d_joint=drop3d_joint, rr_corrupt_prob=rr_corrupt_prob)

        # 3. 4 vong Fusion 2 chieu (Anh <-> Khop) & Tinh chinh khung xuong 3D
        self.smpl_tgt = nn.Parameter(torch.zeros(1, N_SMPL_TOK, dim))
        self.smpl_qpos = nn.Embedding(N_SMPL_TOK, dim)
        self.joint_qpos = nn.Embedding(N_JOINT, dim)
        self.fusion_blocks = nn.ModuleList([
            BiCrossFusionBlock(dim, use_bi_attn=self.use_bi_attn) for _ in range(self.n_dec)
        ])
        self.refine_steps = nn.ModuleList([
            KinematicRefineStep(dim, is_first=(i == 0), use_learn_gate=self.use_learn_gate)
            for i in range(self.n_dec)
        ])

        # 4. Dau ra SMPL thuan no-ron tu khung xuong 3D
        if self.use_res6d_head:
            self.smpl_head = PerJointNeuralSMPLHead(
                dim, self.n_dec, vposer_ckpt=vposer_ckpt, pose_tok_drop=pose_tok_drop
            )
        else:
            self.smpl_head = KinematicVPoserHead(dim, self.n_dec, vposer_ckpt=vposer_ckpt)
        self.vposer = self.smpl_head.vposer

        for m in list(self.img_enc) + [b.joint_from_img for b in self.fusion_blocks]:
            for p in m.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        from utils.smpl import SMPL as SMPLModel
        self.human_model = SMPLModel()
        self.human_model_layer = self.human_model.layer['neutral']
        self.register_buffer('joint_regressor_t', torch.from_numpy(self.human_model.joint_regressor).float())

    def encode_image(self, fmap, f16):
        B, _, h, w = fmap.shape
        mem2d = self.img_proj(fmap).flatten(2).transpose(1, 2)                      # (B, 64, D)
        mpos2d = sine_pos_2d(h, w, self.dim, mem2d.device, fmap.dtype)[None].expand(B, -1, -1)
        for layer in self.img_enc:
            mem2d = layer(mem2d, mpos2d)
        f16_feat = self.f16_proj(f16)                                               # (B, D, 16, 16)
        return mem2d, mpos2d, f16_feat

    def decode_smpl(self, params, is_train, cam_bbox=None):
        return self.smpl_head.decode(self, params, is_train=is_train, cam_bbox=cam_bbox)

    def forward(self, fmap, f16, op_xy, op_mask, is_train=True, cam_bbox=None,
                lift=None, hyp=None, ref_xy=None, ref_logb=None, keep_s=None):
        assert lift is not None and hyp is not None and ref_xy is not None and ref_logb is not None, \
            'V11Head can lift, hyp, ref_xy, ref_logb tu Refiner2D + ImgLifter'
        B = fmap.shape[0]
        dtype = fmap.dtype
        lift_clean = lift

        # Step 1: Ma hoa anh (F8, F16) va goc nhin CLIFF cam_bbox
        mem2d, mpos2d, f16_feat = self.encode_image(fmap, f16)                      # (B,64,D), (B,D,16,16)
        if self.cam_embed is not None:
            assert cam_bbox is not None, 'v10_cam_bbox can cam_bbox (B, 3)'
            cam_j, cam_r, cam_c = self.cam_embed(cam_bbox, dtype)
        else:
            cam_j = cam_r = cam_c = 0.0

        # Step 2: Cham diem tin cay JRE & khoi tao 17 token khop ban dau
        joint_tok, uv, sample_feat, sample_pos, w3d, r_rel, aux = self.router(
            f16_feat, self.f16_sampler, op_xy, op_mask, lift, hyp, ref_xy, ref_logb, cam_j,
            is_train=is_train, keep_s=keep_s
        )

        # Step 3: 4 vong lap Fusion 2 chieu (Anh <-> Khop) + Tinh chinh Xuong 3D + Dau ra SMPL
        smpl_tok = self.smpl_tgt.expand(B, -1, -1)                                  # (B, 3, D)
        smpl_pos = self.smpl_qpos.weight[None].expand(B, -1, -1)                    # (B, 3, D)
        joint_pos = self.joint_qpos.weight[None].expand(B, -1, -1)                  # (B, 17, D)
        keep3d = aux['keep3d']
        lift_masked = aux['lift_used'].float() * keep3d.float()                     # (B, 17, 3)

        smpl_params = self.smpl_head.init_state()
        j3d = j3d_vis0 = None
        w3d_step = w3d
        w3d_log = w3d
        per_layer_params = []
        j17_layers = []
        rp_xys = [uv * 2.0 - 1.0]

        for i in range(self.n_dec):
            smpl_tok, joint_tok, mem2d = self.fusion_blocks[i](
                smpl_tok, joint_tok, smpl_pos, joint_pos, mem2d, mpos2d, sample_feat, sample_pos
            )
            joint_tok, body_tok, j3d, vis0_i, uv, sample_feat, sample_pos, w3d_eff = self.refine_steps[i](
                joint_tok, uv, j3d, lift_masked, w3d_step, op_mask, r_rel, f16_feat, self.f16_sampler, keep3d=keep3d
            )
            if i == 0:
                j3d_vis0 = vis0_i
                w3d_log = w3d_eff
                if self.use_learn_gate:
                    w3d_step = w3d_eff.detach()

            smpl_params = self.smpl_head.step(
                i, smpl_tok, joint_tok, body_tok, j3d, cam_r, cam_c, smpl_params,
                keep3d=keep3d, is_train=is_train
            )
            per_layer_params.append(smpl_params)
            j17_layers.append(j3d)
            rp_xys.append(uv * 2.0 - 1.0)

        # Step 4: Giai ma ra Mesh SMPL & dong goi output cho Trainer / Tester
        result = self.decode_smpl(per_layer_params[-1], is_train=is_train, cam_bbox=cam_bbox)
        result['refine_history'] = [
            self.decode_smpl(p, is_train=is_train, cam_bbox=cam_bbox) for p in per_layer_params[:-1]
        ] if is_train else []

        result['ctx_3d'] = j17_layers[-1]
        result['privileged_3d'] = j17_layers[-1]
        result['lifted_joints_3d'] = lift_clean.detach()
        result['feat_joint'] = joint_tok
        result['feat_global'] = smpl_tok.mean(dim=1)

        if is_train:
            result['v9_rp_xy'] = torch.stack(rp_xys[1:], dim=1)                     # (B, L, 17, 2)
            result['v9_rp_ref'] = rp_xys[0].detach()
            result['v10_j17_layers'] = [j3d_vis0] + j17_layers                      # giam sat GT 3D moi tang
            result['v10_jre_pred'] = aux['jre_pred']
            result['v10_lift_used'] = aux['lift_used'].detach()
            result['v10_keep3d'] = keep3d.detach()
            result['v10_op_xy'] = aux['xy_det'].detach()
            result['v10_op_vis'] = aux['vis'].detach()
            result['v10_w3d'] = w3d_log.detach()
        return result


class V11Net(nn.Module):
    """Vo boc FuseKin v11: Backbone ResNet-50 SPIN + Refiner2D/ImgLifter dong bang + V11Head."""

    def __init__(self):
        super().__init__()
        self.mode = 'v11'
        self.backbone = ResNetBackbone(cfg.MODEL.resnet_type)
        self.backbone.enable_heatmap_fusion(N_JOINT)
        bb_src = cfg.MODEL.get('backbone_pretrained', '')
        if bb_src:
            self.backbone.load_pretrained(
                bb_src, spin_checkpoint=cfg.MODEL.get('spin_checkpoint',
                                                      'data_final/base_data/spin_model_checkpoint.pth.tar'))
        else:
            print('[CANH BAO] V11: backbone KHONG nap pretrain -> dac trung anh ngau nhien.')

        from utils.img_lifter_tools import load_img_lifter, load_frozen_refiner
        assert cfg.MODEL.imglift_checkpoint, 'FuseKin v11 can MODEL.imglift_checkpoint'
        cpu = torch.device('cpu')
        self.img_lifter, ck = load_img_lifter(cfg.MODEL.imglift_checkpoint, cpu)
        self.refiner2d = load_frozen_refiner(
            cfg.MODEL.get('imglift_refiner_checkpoint', '') or ck['refiner_ckpt'], cpu)
        for p in self.img_lifter.parameters():
            p.requires_grad = False
        self.img_lifter.eval()

        self.drop3d_sample = float(cfg.MODEL.get('v11_drop3d_sample', 0.2))
        self.ref_heatmap = bool(cfg.MODEL.get('v11_ref_heatmap', False))
        self.smpl_model = V11Head(
            dim=int(cfg.MODEL.v11_dim),
            n_enc=int(cfg.MODEL.v11_enc_layers),
            n_dec=int(cfg.MODEL.v11_dec_layers),
            vposer_ckpt=str(cfg.MODEL.get('v10_vposer_ckpt', '')),
            use_cam_bbox=bool(cfg.MODEL.get('v10_cam_bbox', False)),
            proj_grad=bool(cfg.MODEL.get('v10_proj_grad', False)),
            use_bi_attn=bool(cfg.MODEL.get('v11_bi_attn', False)),
            drop3d_sample=self.drop3d_sample,
            drop3d_joint=float(cfg.MODEL.get('v11_drop3d_joint', 0.1)),
            rr_corrupt_prob=float(cfg.MODEL.get('v11_rr_corrupt_prob', 0.3)),
            use_res6d_head=bool(cfg.MODEL.get('v11_res6d_head', False)),
            pose_tok_drop=float(cfg.MODEL.get('v11_pose_tok_drop', 0.0)),
            use_learn_gate=bool(cfg.MODEL.get('v11_learn_gate', False)),
        )
        self.joint_drop = float(cfg.MODEL.v11_joint_drop)
        print(f'===> FuseKin v11 (Bi-CrossAttn={cfg.MODEL.get("v11_bi_attn", False)} | '
              f'res6d_head={cfg.MODEL.get("v11_res6d_head", False)} [drop_ptok={cfg.MODEL.get("v11_pose_tok_drop", 0.0)}] | '
              f'learn_gate={cfg.MODEL.get("v11_learn_gate", False)} | ref_hm={self.ref_heatmap} | '
              f'consist_w={cfg.MODEL.get("v11_consist_w", 0.0)}): '
              f'dim {cfg.MODEL.v11_dim} | enc {cfg.MODEL.v11_enc_layers} | dec {cfg.MODEL.v11_dec_layers} | '
              f'cam_bbox {cfg.MODEL.get("v10_cam_bbox", False)} | proj_grad {cfg.MODEL.get("v10_proj_grad", False)} | '
              f'corrupt_p {cfg.MODEL.get("v11_rr_corrupt_prob", 0.3)} | '
              f'drop3d ({self.drop3d_sample}, {cfg.MODEL.get("v11_drop3d_joint", 0.1)}) | '
              f'lifter {cfg.MODEL.imglift_checkpoint} (K={cfg.MODEL.imglift_K}, S={cfg.MODEL.imglift_S})')

    def train(self, mode=True):
        super().train(mode)
        self.smpl_model.vposer.eval()
        self.img_lifter.eval()
        self.refiner2d.eval()
        return self

    @torch.no_grad()
    def lift(self, image, p2, jm, cam_bbox):
        """Chay Refiner2D + ImgLifter dong bang -> lift (B,17,3), hyp (B,K,17,3), ref_xy (B,17,2), ref_logb (B,17)."""
        from models.img_lifter import ImgLifter, aggregate
        from utils.img_lifter_tools import refiner_inputs, lifter_det
        B = image.shape[0]
        jm = jm.reshape(B, N_JOINT).float()
        nc = int(getattr(self.img_lifter, 'num_ctx', 0))
        xy, log_b, f8, fj = refiner_inputs(self.refiner2d, image, p2[..., :2].float(), jm, N_JOINT + nc)
        cond7 = ImgLifter.make_cond7(xy, log_b, lifter_det(jm, nc), cam_bbox.float())
        gen = None
        if not self.training:
            gen = torch.Generator(device=image.device)
            gen.manual_seed(0)
        hyp = self.img_lifter.sample(cond7, fj, f8, K=int(cfg.MODEL.imglift_K), S=int(cfg.MODEL.imglift_S),
                                     eta=1.0, generator=gen)
        pose = aggregate(hyp, xy, log_b, 'mean')
        return pose - pose[:, 0:1], hyp - hyp[:, :, 0:1], xy[:, :N_JOINT], log_b[:, :N_JOINT]

    def forward(self, image, joints, is_train=True, joints_mask=None, pose_2d=None, cam_bbox=None):
        assert cam_bbox is not None, "FuseKin v11 can inputs['cam_bbox']"
        B = image.shape[0]
        p2 = joints[:, 0] if joints.dim() == 4 else joints
        if joints_mask is None:
            m = (p2[..., 2:3] > 0).float() if p2.shape[-1] >= 3 else torch.ones_like(p2[..., :1])
        else:
            m = joints_mask.reshape(B, N_JOINT, 1).float()

        lift, hyp, ref_xy, ref_logb = self.lift(image, p2, m, cam_bbox)
        if self.training and self.joint_drop > 0:
            m = m * (torch.rand_like(m) >= self.joint_drop).float()

        keep_s = None
        if self.ref_heatmap:
            if bool(cfg.MODEL.get('v11_test_drop3d', False)):
                keep_s = torch.zeros(B, 1, 1, device=image.device)
            elif is_train:
                keep_s = (torch.rand(B, 1, 1, device=image.device) >= self.drop3d_sample).float()
            else:
                keep_s = torch.ones(B, 1, 1, device=image.device)
            ref_conf = torch.exp(-ref_logb.float().clamp_min(0.0))[..., None].clamp(0.2, 1.0)
            if self.training and self.joint_drop > 0:
                ref_conf = ref_conf * (torch.rand_like(ref_conf) >= self.joint_drop).float()
            hm_xy = torch.where(keep_s > 0.5, ref_xy.float(), p2[..., :2].float())
            hm_mask = torch.where(keep_s > 0.5, ref_conf, m)
            heat = make_heatmaps(hm_xy, hm_mask, size=image.shape[-1] // 4, sigma=float(cfg.MODEL.v3_heatmap_sigma))
        else:
            heat = make_heatmaps(p2, m, size=image.shape[-1] // 4, sigma=float(cfg.MODEL.v3_heatmap_sigma))

        fmap, _, f16 = self.backbone(image, heatmap=heat, return_f16=True)
        return self.smpl_model(fmap, f16, p2[..., :2].float(), m, is_train=is_train,
                               cam_bbox=cam_bbox.float(),
                               lift=lift, hyp=hyp, ref_xy=ref_xy, ref_logb=ref_logb, keep_s=keep_s)
