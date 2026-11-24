"""
lib/models/fusekin_v11.py -- FuseKin v11 (2026-11-25, nguoi dung duyet):
  Reliability-Routed 3D-Kinematic Mesh Regressor (1 luong thang gon sach, BO khoi Voxel 3D 8x8x8 cua V10):
    [1] Nhanh 3D Prior dong bang (Refiner2D + ImgLifter v2, KxS) -> lift (17, 3), std3, ref_xy (17, 2), ref_logb;
        Nhanh Anh end-to-end (ResNet-50 SPIN + heatmap 2D) -> F16 (256x16x16) va F8 -> mem2d (64 token).
    [2] Joint Reliability Estimator (JRE) & Hop nhat Toa do (khong tu bop 3D):
        - JRE doc [img_j_det, xy_det, op_mask, feat_lift, xy_det - ref_xy, wp_reproj_res] -> R_i = (r3D, r2D) in (0,1)^2
          (luc train: p=0.3 xoay cung 1-2 chuoi chi tren hyp de day JRE bat loi Lifter sai huong chi).
        - Cong 3D: w3D_i = keep3d_i * r3D_i (giu tron tin hieu 3D tot o ca khop nhin ro, bo he so dim 1 - 0.7*r2D).
        - Vi tri lay mau 2D dau: p^(0)_i = (vis_i * r2D_i) * xy_det_i + (1 - vis_i * r2D_i) * ref_xy_i (khi keep3d=1).
    [3] 4 lop 3D-Anchored HyperGCN Decoder:
        - Self-Attn [3 token SMPL | 17 token khop jt] + Cross-Attn [64 mem2d | 17 mau F16 tai p^(l)] + AdaptiveHyperGCN.
        - Tinh chinh truc tiep khung xuong 3D J^(l) (met, root-relative) khong co xuong:
            J^(0) = w3D * lift + (1 - w3D) * j3d_vis0(jt) + delta_J0(jt)
            J^(l) = J^(l-1) + delta_Jl([jt, J^(l-1), w3D])   (l > 0)
          trong do j3d_vis0 cung duoc giam sat GT 3D de nhanh anh luon manh khi w3D thap / --drop3d.
        - Cap nhat vi tri lay mau 2D p^(l+1) tren F16 va phan hoi [mau F16 moi, J^(l)] vao jt cho lop ke tiep.
    [4] Kinematics-Grounded VPoser SMPL Cascade (tai moi lop decoder):
        - Tu J^(l) tinh 16 vector huong xuong don vi B_dir^(l) (co grad) va 16 do dai xuong B_len^(l) (detach):
            root_pose_head : pose_tok + root_readout(jt_than) + ctx_to_root(J^(l)) + cam_r (CLIFF)
            vposer_head    : pose_tok + body_readout(body_tok) + ctx_to_vposer([J^(l), B_dir^(l)])
            shape_head     : shape_tok + len_to_shape(B_len^(l))
            cam_head       : cam_tok + cam_c (CLIFF)
        - Giai ma qua VPoser dong bang -> SMPL (+ view_rot(-dx, -dy) cho loss chieu 2D khi v10_proj_grad=True).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
from models.ARTS import make_heatmaps
from models.backbones.resnet import ResNetBackbone
from models.fusekin_v8 import EncLayer, DecLayer, CascadeHead, V8Head, sine_pos_2d
from models.fusekin_v10 import (
    V10Head, VPoserDecoder, AdaptiveHyperGCNLayer, inverse_sigmoid,
    _corrupt_hyp_limbs, _wp_reproj_residual, SMPL_MEAN_PARAMS, N_SMPL_TOK, N_JOINT, VPOSER_DIM,
)
from models.img_lifter import H36M_BONES


class V11Head(nn.Module):
    """JRE Routing + 4 lop 3D-Anchored HyperGCN Decoder (chi F16 + F8, khong Voxel 3D) + Kinematics-Grounded VPoser."""
    get_camera_trans = V8Head.get_camera_trans
    get_coord = V8Head.get_coord
    smpl_from_aa = V8Head.smpl_from_aa
    smpl_from_aa_v10 = V10Head.smpl_from_aa_v10
    vposer_smpl_outputs = V10Head.vposer_smpl_outputs

    def __init__(self, dim=256, n_enc=3, n_dec=4, img_ch=2048, f16_ch=1024, vposer_ckpt='',
                 use_cam_bbox=True, proj_grad=True,
                 drop3d_sample=0.2, drop3d_joint=0.1, rr_corrupt_prob=0.3):
        super().__init__()
        self.dim = int(dim)
        self.n_dec = int(n_dec)
        self.use_cam_bbox = bool(use_cam_bbox)
        self.proj_grad = bool(proj_grad)
        self.drop3d_sample = float(drop3d_sample)
        self.drop3d_joint = float(drop3d_joint)
        self.rr_corrupt_prob = float(rr_corrupt_prob)

        # ---- [CLIFF] cam_bbox (dx, dy, s) -> token khop, root_pose_head, cam_head ----
        if self.use_cam_bbox:
            self.cam_emb = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU())
            self.cam_to_joint = nn.Linear(dim, dim)
            self.cam_to_root = nn.Linear(dim, dim)
            self.cam_to_cam = nn.Linear(dim, dim)
            for m in (self.cam_to_joint, self.cam_to_root, self.cam_to_cam):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        # ---- 1) Encoder anh 2D: F8 (2048x8x8 -> 64 token) va F16 (1024x16x16 -> Dx16x16) ----
        self.img_proj = nn.Conv2d(img_ch, dim, 1)
        self.img_enc = nn.ModuleList([EncLayer(dim) for _ in range(int(n_enc))])
        self.f16_proj = nn.Conv2d(f16_ch, dim, 1)

        # ---- 2) JRE (Module 1) + Nhung dau vao 2D / 3D Lifter ----
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
        self.fc_pos_2d = nn.Sequential(
            nn.Linear(2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # ---- 3) 4 lop 3D-Anchored HyperGCN Decoder ----
        self.smpl_tgt = nn.Parameter(torch.zeros(1, N_SMPL_TOK, dim))
        self.smpl_qpos = nn.Embedding(N_SMPL_TOK, dim)
        self.joint_qpos = nn.Embedding(N_JOINT, dim)
        self.decoder = nn.ModuleList([DecLayer(dim) for _ in range(self.n_dec)])
        self.dec_norm = nn.LayerNorm(dim)
        self.ad_hgcn = nn.ModuleList([AdaptiveHyperGCNLayer(dim, use_rr_hgcn=True) for _ in range(self.n_dec)])

        # Du doan 3D tu thi giac o lop 0 (dung cho to hop loi voi lift khi w3D thap / --drop3d) + phan du 3D moi lop
        self.j3d_vis_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 3))
        nn.init.normal_(self.j3d_vis_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.j3d_vis_head[-1].bias)

        self.j3d_res = nn.ModuleList([
            nn.Sequential(nn.Linear(dim + 4, dim // 2), nn.GELU(), nn.Linear(dim // 2, 3))
            for _ in range(self.n_dec)
        ])
        for m in self.j3d_res:
            nn.init.normal_(m[-1].weight, std=1e-3)
            nn.init.zeros_(m[-1].bias)

        # Cap nhat toa do lay mau 2D uv in (0, 1)^2 tren F16 va phan hoi F16 + hinh hoc J^(l) vao jt
        self.uv_regress = nn.ModuleList([nn.Linear(dim + 5, 2) for _ in range(self.n_dec)])
        for m in self.uv_regress:
            nn.init.normal_(m.weight, std=1e-3)
            nn.init.zeros_(m.bias)

        self.sample_to_joint = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n_dec)])
        self.geom_to_joint = nn.ModuleList([
            nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
            for _ in range(self.n_dec)
        ])
        for m in self.sample_to_joint:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.geom_to_joint:
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)

        # ---- 4) Kinematics-Grounded VPoser SMPL Cascade ----
        bone_p = torch.tensor([p for p, _ in H36M_BONES], dtype=torch.long)
        bone_c = torch.tensor([c for _, c in H36M_BONES], dtype=torch.long)
        self.register_buffer('bone_p_idx', bone_p, persistent=False)
        self.register_buffer('bone_c_idx', bone_c, persistent=False)

        self.vposer = VPoserDecoder(vposer_ckpt)
        self.body_readout = nn.Sequential(nn.LayerNorm(5 * dim), nn.Linear(5 * dim, dim))
        self.root_readout = nn.Sequential(nn.LayerNorm(5 * dim), nn.Linear(5 * dim, dim))

        n_bones = len(H36M_BONES)
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
        self.root_pose_head = CascadeHead(dim, 6, self.n_dec, init=init_root6d)
        self.vposer_head = CascadeHead(dim, VPOSER_DIM, self.n_dec)
        self.shape_head = CascadeHead(dim, 10, self.n_dec, init=init_shape)
        self.cam_head = CascadeHead(dim, 3, self.n_dec)

        for p in list(self.img_enc.parameters()) + list(self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        from utils.smpl import SMPL as SMPLModel
        self.human_model = SMPLModel()
        self.human_model_layer = self.human_model.layer['neutral']
        self.register_buffer('joint_regressor_t', torch.from_numpy(self.human_model.joint_regressor).float())

    def sample_f16(self, f16_feat, uv):
        """Lay mau song tuyen F16 (B, D, 16, 16) tai uv (B, 17, 2) in (0, 1)^2. Tra (feat, pos) cung shape (B, 17, D)."""
        xy_grid = uv.float() * 2.0 - 1.0                                                # (B, 17, 2) in [-1, 1]
        feat = F.grid_sample(f16_feat.float(), xy_grid[:, :, None, :], mode='bilinear',
                             align_corners=False).squeeze(-1).transpose(1, 2).to(f16_feat.dtype)
        pos = self.fc_pos_2d(xy_grid.to(f16_feat.dtype))
        return feat, pos

    def forward(self, fmap, f16, op_xy, op_mask, is_train=True, cam_bbox=None,
                lift=None, hyp=None, ref_xy=None, ref_logb=None):
        """fmap (B, 2048, 8, 8), f16 (B, 1024, 16, 16); op_xy (B, 17, 2) [-1, 1], op_mask (B, 17, 1) [0, 1];
        cam_bbox (B, 3); lift (B, 17, 3), hyp (B, K, 17, 3), ref_xy (B, 17, 2), ref_logb (B, 17)."""
        assert lift is not None and hyp is not None and ref_xy is not None and ref_logb is not None, \
            'V11Head can lift, hyp, ref_xy, ref_logb tu Refiner2D + ImgLifter'
        B, _, h, w = fmap.shape
        D = self.dim
        dtype = fmap.dtype

        cam_j = cam_r = cam_c = 0.0
        if self.use_cam_bbox:
            assert cam_bbox is not None, 'v10_cam_bbox can cam_bbox (B, 3)'
            ce = self.cam_emb(cam_bbox.to(dtype))
            cam_j = self.cam_to_joint(ce)[:, None]                                      # (B, 1, D)
            cam_r = self.cam_to_root(ce)                                                # (B, D)
            cam_c = self.cam_to_cam(ce)                                                 # (B, D)

        # ---- [0] Modality dropout & Lam hong chuoi chi tren hyp luc train ----
        lift_clean = lift
        if is_train and self.rr_corrupt_prob > 0.0:
            lift, hyp = _corrupt_hyp_limbs(hyp, self.rr_corrupt_prob)
        if bool(cfg.MODEL.get('v11_test_drop3d', False)):
            keep3d = torch.zeros(B, N_JOINT, 1, device=fmap.device, dtype=dtype)
        elif is_train:
            keep_s = (torch.rand(B, 1, 1, device=fmap.device) >= self.drop3d_sample).to(dtype)
            keep_j = (torch.rand(B, N_JOINT, 1, device=fmap.device) >= self.drop3d_joint).to(dtype)
            keep3d = keep_s * keep_j
        else:
            keep3d = torch.ones(B, N_JOINT, 1, device=fmap.device, dtype=dtype)

        std3 = hyp.float().std(dim=1, unbiased=False) * 10.0                            # (B, 17, 3)
        feat_l = torch.cat([lift.float(), std3, ref_xy.float(), ref_logb.float()[..., None]], dim=-1).to(dtype)
        e3d = self.lift_to_joint(self.lift_emb(feat_l))                                 # (B, 17, D)

        # ---- [1] Bo nho anh F8 (64 token) & F16 (16x16) ----
        mem2d = self.img_proj(fmap).flatten(2).transpose(1, 2)                          # (B, 64, D)
        mpos2d = sine_pos_2d(h, w, D, mem2d.device, dtype)[None].expand(B, -1, -1)      # (B, 64, D)
        for layer in self.img_enc:
            mem2d = layer(mem2d, mpos2d)
        f16_feat = self.f16_proj(f16)                                                   # (B, D, 16, 16)

        # ---- [2] JRE Routing & Hop nhat Toa do Lay mau 2D dau ----
        vis = (op_mask > 0).to(dtype)                                                   # (B, 17, 1)
        xy_in = (op_xy * vis).to(dtype)                                                 # (B, 17, 2)
        img_j_det = F.grid_sample(f16_feat, xy_in[:, :, None, :], mode='bilinear',
                                  align_corners=False).squeeze(-1).transpose(1, 2)      # (B, 17, D)
        diff_2d = ((xy_in - ref_xy.to(dtype)) * vis) * keep3d
        reproj = _wp_reproj_residual(lift, ref_xy, ref_logb, keep3d).to(dtype) * 10.0 * keep3d
        jre_in = torch.cat([
            self.jre_img_norm(img_j_det.detach()),
            xy_in,
            op_mask.to(dtype),
            feat_l * keep3d,
            diff_2d,
            reproj,
        ], dim=-1)
        jre_pred = torch.sigmoid(self.jre_head(jre_in))                                 # (B, 17, 2) = (r3D, r2D)
        r3d_det = jre_pred[..., 0:1].detach()
        r2d_det = jre_pred[..., 1:2].detach()

        # Tach rieng vai tro: w3D = keep3d * r3D (khong dim khop 2D ro); alpha_2d chon giua xy_det va ref_xy
        w3d = keep3d.float() * r3d_det.float()                                          # (B, 17, 1) in [0, 1]
        r_rel = torch.cat([w3d.to(dtype), r2d_det.to(dtype)], dim=-1)                   # (B, 17, 2) cho HyperGCN

        alpha_2d = vis.float() * r2d_det.float()                                        # (B, 17, 1)
        xy_fused = alpha_2d * xy_in.float() + (1.0 - alpha_2d) * ref_xy.float()
        xy_0 = torch.where(keep3d > 0.5, xy_fused, xy_in.float()).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        uv = ((xy_0 + 1.0) * 0.5).clamp(1e-4, 1.0 - 1e-4)                               # (B, 17, 2) in (0, 1)^2

        sample_feat, sample_pos = self.sample_f16(f16_feat, uv)
        has_2d = ((vis > 0) | (keep3d > 0)).to(dtype)
        e2d = self.emb2d(torch.cat([
            (uv * 2.0 - 1.0).to(dtype) * has_2d,
            vis,
            alpha_2d.to(dtype) * keep3d,
        ], dim=-1))
        lift_j = keep3d * (w3d.to(dtype) * e3d) + (1.0 - keep3d) * self.miss3d_tok
        jt = self.init_norm(sample_feat * has_2d + e2d + (1.0 - has_2d) * self.miss2d_tok
                            + self.joint_id.weight[None] + cam_j + lift_j)

        # ---- [3] 4 lop 3D-Anchored HyperGCN Decoder + Kinematics-Grounded VPoser Cascade ----
        j_qpos = self.joint_qpos.weight[None].expand(B, -1, -1)
        tgt = torch.cat([self.smpl_tgt.expand(B, -1, -1), jt], dim=1)                   # (B, 20, D)
        qpos = torch.cat([self.smpl_qpos.weight[None].expand(B, -1, -1), j_qpos], dim=1)

        root_6d = vposer_z = shape = cam = j3d = j3d_vis0 = None
        per_layer = []
        rp_xys = [uv * 2.0 - 1.0]
        j17_layers = []

        for i, layer in enumerate(self.decoder):
            mem = torch.cat([mem2d, sample_feat], dim=1)                                # (B, 64 + 17, D)
            mpos = torch.cat([mpos2d, sample_pos], dim=1)
            tgt = self.dec_norm(layer(tgt, qpos, mem, mpos))

            smpl_tok = tgt[:, :N_SMPL_TOK]
            jt = tgt[:, N_SMPL_TOK:]

            # AdaptiveHyperGCN nhan [uv (2), w3d (1)] o vi tri 3 chieu toa do + op_mask + r_rel
            hgcn_coord = torch.cat([uv.detach(), w3d.detach()], dim=-1)                 # (B, 17, 3)
            jt, body_tok, _ = self.ad_hgcn[i](jt, hgcn_coord, op_mask, r_rel=r_rel)

            # Cap nhat khung xuong 3D J^(i) (met, root-relative) khong co xuong
            if i == 0:
                j3d_vis0 = self.j3d_vis_head(jt.float())
                j3d_vis0 = j3d_vis0 - j3d_vis0[:, 0:1]
                lift_masked = lift.float() * keep3d.float()
                delta0 = self.j3d_res[0](torch.cat([jt.float(), lift_masked, w3d], dim=-1))
                j3d = w3d * lift_masked + (1.0 - w3d) * j3d_vis0 + delta0
            else:
                delta_i = self.j3d_res[i](torch.cat([jt.float(), j3d, w3d], dim=-1))
                j3d = j3d + delta_i
            j3d = j3d - j3d[:, 0:1]
            j17_layers.append(j3d)

            # Cap nhat vi tri lay mau 2D tren F16 & phan hoi F16 + hinh hoc J^(i) vao jt
            inv_uv = inverse_sigmoid(uv.float())
            off_uv = self.uv_regress[i](torch.cat([jt.float(), j3d.detach(), inv_uv], dim=-1))
            uv = torch.sigmoid(inv_uv + off_uv)
            rp_xys.append(uv * 2.0 - 1.0)

            sample_feat, sample_pos = self.sample_f16(f16_feat, uv)
            jt = jt + self.sample_to_joint[i](sample_feat) + self.geom_to_joint[i](j3d.detach().to(dtype))
            tgt = torch.cat([smpl_tok, jt], dim=1)

            # Kinematics-Grounded SMPL Heads: J^(i) + 16 huong xuong don vi (co grad) + 16 do dai xuong (detach)
            bone_vec = j3d[:, self.bone_c_idx] - j3d[:, self.bone_p_idx]                # (B, 16, 3)
            bone_len = torch.norm(bone_vec, dim=-1, keepdim=True).clamp_min(1e-6)       # (B, 16, 1)
            bone_dir = bone_vec / bone_len                                              # (B, 16, 3)

            ctx_r = self.ctx_to_root(j3d.to(dtype).flatten(1))
            ctx_v = self.ctx_to_vposer(torch.cat([j3d, bone_dir], dim=-1).to(dtype).flatten(1))
            ctx_s = self.len_to_shape(bone_len.squeeze(-1).detach().to(dtype))

            body_sum = self.body_readout(body_tok.flatten(1))
            root_sum = self.root_readout(jt[:, [0, 1, 4, 7, 8]].flatten(1))
            pose_tok = smpl_tok[:, 0]

            root_6d = self.root_pose_head(i, pose_tok + root_sum + ctx_r + cam_r, root_6d)
            vposer_z = self.vposer_head(i, pose_tok + body_sum + ctx_v, vposer_z)
            shape = self.shape_head(i, smpl_tok[:, 1] + ctx_s, shape)
            cam = self.cam_head(i, smpl_tok[:, 2] + cam_c, cam)
            per_layer.append((root_6d, vposer_z, shape, cam))

        result = self.vposer_smpl_outputs(*per_layer[-1], is_train=is_train, cam_bbox=cam_bbox)
        result['refine_history'] = [self.vposer_smpl_outputs(*it, is_train=is_train, cam_bbox=cam_bbox)
                                    for it in per_layer[:-1]] if is_train else []
        j3d_final = j17_layers[-1]
        result['ctx_3d'] = j3d_final
        result['lifted_joints_3d'] = lift_clean.detach()
        result['privileged_3d'] = j3d_final
        result['feat_joint'] = jt
        result['feat_global'] = tgt[:, :N_SMPL_TOK].mean(1)
        if is_train:
            result['v9_rp_xy'] = torch.stack(rp_xys[1:], dim=1)                         # (B, L, 17, 2)
            result['v9_rp_ref'] = rp_xys[0].detach()
            # Dua j3d_vis0 vao dau danh sach de base.py giam sat GT 3D tren 100% mau (chong shortcut learning)
            result['v10_j17_layers'] = [j3d_vis0] + j17_layers
            result['v10_jre_pred'] = jre_pred
            result['v10_lift_used'] = lift.detach()
            result['v10_keep3d'] = keep3d.detach()
            result['v10_op_xy'] = xy_in.detach()
            result['v10_op_vis'] = vis.detach()
            result['v10_w3d'] = w3d.detach()
        return result


class V11Net(nn.Module):
    """Vo boc FuseKin v11 (thay ARTS khi MODEL.v11_e2e): ResNet-50 SPIN + Refiner2D/ImgLifter dong bang + V11Head."""

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

        self.smpl_model = V11Head(
            dim=int(cfg.MODEL.v11_dim),
            n_enc=int(cfg.MODEL.v11_enc_layers),
            n_dec=int(cfg.MODEL.v11_dec_layers),
            vposer_ckpt=str(cfg.MODEL.get('v10_vposer_ckpt', '')),
            use_cam_bbox=bool(cfg.MODEL.get('v10_cam_bbox', False)),
            proj_grad=bool(cfg.MODEL.get('v10_proj_grad', False)),
            drop3d_sample=float(cfg.MODEL.get('v11_drop3d_sample', 0.2)),
            drop3d_joint=float(cfg.MODEL.get('v11_drop3d_joint', 0.1)),
            rr_corrupt_prob=float(cfg.MODEL.get('v11_rr_corrupt_prob', 0.3)),
        )
        self.joint_drop = float(cfg.MODEL.v11_joint_drop)
        print(f'===> FuseKin v11 (3D-Kinematic + JRE w3D=r3D + F16/F8, KHONG Voxel 3D, KHONG IK): '
              f'dim {cfg.MODEL.v11_dim} | enc {cfg.MODEL.v11_enc_layers} | dec {cfg.MODEL.v11_dec_layers} | '
              f'cam_bbox {cfg.MODEL.get("v10_cam_bbox", False)} | proj_grad {cfg.MODEL.get("v10_proj_grad", False)} | '
              f'corrupt_p {cfg.MODEL.get("v11_rr_corrupt_prob", 0.3)} | '
              f'drop3d ({cfg.MODEL.get("v11_drop3d_sample", 0.2)}, {cfg.MODEL.get("v11_drop3d_joint", 0.1)}) | '
              f'lifter {cfg.MODEL.imglift_checkpoint} (K={cfg.MODEL.imglift_K}, S={cfg.MODEL.imglift_S})')

    def train(self, mode=True):
        super().train(mode)
        self.smpl_model.vposer.eval()
        self.img_lifter.eval()
        self.refiner2d.eval()
        return self

    @torch.no_grad()
    def lift(self, image, p2, jm, cam_bbox):
        """Chay Refiner2D + ImgLifter dong bang. Tra lift (B, 17, 3), hyp (B, K, 17, 3) met root-relative,
        ref_xy (B, 17, 2) [-1, 1], ref_logb (B, 17)."""
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
        """joints = 2D detector (B, 17, 2|3) [-1, 1]; joints_mask (B, 17, 1); cam_bbox (B, 3)."""
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
        heat = make_heatmaps(p2, m, size=image.shape[-1] // 4, sigma=float(cfg.MODEL.v3_heatmap_sigma))
        fmap, _, f16 = self.backbone(image, heatmap=heat, return_f16=True)
        return self.smpl_model(fmap, f16, p2[..., :2].float(), m, is_train=is_train,
                               cam_bbox=cam_bbox.float(),
                               lift=lift, hyp=hyp, ref_xy=ref_xy, ref_logb=ref_logb)
