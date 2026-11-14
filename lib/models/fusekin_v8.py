"""
lib/models/fusekin_v8.py -- FuseKin v8, moc M0 (2026-11-13, nguoi dung duyet): mang END-TO-END, ANH LAM CHINH.

  anh + heatmap 2D detector (tiem o stride 4, nhu 3DCrowdNet / JOTR) -> ResNet-50 (train end-to-end, BN train)
  -> F (2048 x 8 x 8) -> Conv 1x1 (D kenh) + PE sin 2D -> Transformer encoder (v8_enc_layers lop)
  -> Transformer decoder (v8_dec_layers lop): 3 token SMPL (pose, shape, camera) + 17 query khop H36M
  -> sau MOI lop decoder: hoi quy phan du theo tang (pose 24 x 6D, beta 10, camera 3, khop 3D 17 x 3) -> SMPL.

M0 = moc tai tao JOTR KHONG co phan 3D cua ho (bang 4 JOTR, dong "flatting / none"): khong khoi dac trung 3D,
khong lay mau dac trung tai khop, khong contrastive. Cung KHONG co phan cua FuseKin (lifter, HyperGCN, IK):
cac phan nay them o M1+ (moi phan 1 key, mac dinh tat).

Giao dien giong ARTS -> Student_Trainer / Tester / test_student_3dpw_sets dung nguyen:
  forward(image, joints, is_train, joints_mask, pose_2d=None, cam_bbox=None) -> dict output nhu Student.
  - refine_history: output SMPL cua cac lop decoder truoc lop cuoi (base._hard_losses giam sat ca chung).
  - ctx_3d: khop 3D cua 17 query (lop cuoi), giam sat bang loss ctx (MODEL.v5_ctx_loss_w) trong Student_Trainer.
  - lifted_joints_3d = ctx_3d (detach): cot LIFT luc danh gia = do chinh xac cua query khop.
Thuoc tinh smpl_model BAT BUOC (base._joint_log_sigma doc net.smpl_model).

M2 (2026-11-14, nguoi dung duyet "lam M2 lun, bo qua M1"): them phan rieng cua FuseKin, moi phan 1 key (mac dinh TAT,
tat het = M0 nguyen ven):
  v8_lifter    : Refiner 2D + ImgLifter no3dpw (dong bang, eval) -> pose 3D (gop mean K gia thuyet). Pose nay cong vao
                 17 query khop va la diem xuat phat cua head khop 3D (lop 0: khop = lifter + phan du nho).
  v8_hyp_tokens: K x 17 gia thuyet -> token, noi vao bo nho decoder (query doc duoc do bat dinh cua lifter).
  v8_ik        : mesh = IK HybrIK tu khop 3D da sua cua tung lop (nhu Student v7): J29 = M x pose17 + delta 7 diem dau
                 chi, twist tu cap token xuong, beta = beta_xuong(pose17) + dbeta(token shape). Bo head 6D / shape.
  v8_hgcn      : HyperGCN HyperDiff (HGraph.create_layers, 1 khoi) tren 17 token khop sau MOI lop decoder.
  M1 = M2 tat v8_hgcn.
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
from models.backbones.resnet import ResNetBackbone
from models.ARTS import make_heatmaps
from models.hybrik_smpl import rotmat_to_aa_safe
from models.teacher_student import SMPL_BONE_PAIRS_H36M, V7_LIMB_IDX

# Token khop H36M cho 7 diem dau chi V7_LIMB_IDX [10 L_Foot, 11 R_Foot, 22 L_Hand, 23 R_Hand, 24 dinh dau,
# 27 ngon chan cai T, 28 ngon chan cai P] -> [L_Ankle, R_Ankle, L_Wrist, R_Wrist, Head_top, L_Ankle, R_Ankle]
V8_LIMB_TOKEN = [6, 3, 13, 16, 10, 6, 3]

SMPL_MEAN_PARAMS = 'data_final/base_data/smpl_mean_params.npz'
N_SMPL_TOK = 3     # pose, shape, camera
N_JOINT = 17       # query khop H36M-17


def rot6d_to_rotmat(x):
    """(N, 6) -> (N, 3, 3). Quy uoc SPIN / smpl_mean_params: x.view(-1, 3, 2), cot 0 = a1, cot 1 = a2."""
    x = x.reshape(-1, 3, 2)
    b1 = F.normalize(x[..., 0], dim=-1)
    a2 = x[..., 1]
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def sine_pos_2d(h, w, dim, device, dtype, temperature=10000.0):
    """PE sin 2D kieu DETR (normalize, scale 2 pi) -> (h * w, dim); thu tu o = hang * w + cot (giong flatten(2))."""
    npf = dim // 2
    y = (torch.arange(h, device=device, dtype=torch.float32) + 1) / h * 2 * math.pi
    x = (torch.arange(w, device=device, dtype=torch.float32) + 1) / w * 2 * math.pi
    gy, gx = torch.meshgrid(y, x, indexing='ij')                                   # (h, w)
    dim_t = torch.arange(npf, device=device, dtype=torch.float32)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / npf)
    px = gx[..., None] / dim_t
    py = gy[..., None] / dim_t
    px = torch.stack([px[..., 0::2].sin(), px[..., 1::2].cos()], dim=-1).flatten(-2)
    py = torch.stack([py[..., 0::2].sin(), py[..., 1::2].cos()], dim=-1).flatten(-2)
    return torch.cat([py, px], dim=-1).reshape(h * w, dim).to(dtype)


class EncLayer(nn.Module):
    """Lop encoder post-norm giong JOTR (normalize_before = False): self-attn -> FFN."""

    def __init__(self, d, nhead=8, ff=2048, drop=0.1):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        self.l1, self.l2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.d0, self.d1, self.d2 = nn.Dropout(drop), nn.Dropout(drop), nn.Dropout(drop)

    def forward(self, x, pos):
        q = k = x + pos
        x = self.n1(x + self.d1(self.sa(q, k, x, need_weights=False)[0]))
        x = self.n2(x + self.d2(self.l2(self.d0(F.relu(self.l1(x))))))
        return x


class DecLayer(nn.Module):
    """Lop decoder post-norm giong JOTR: self-attn giua query -> cross-attn toi token anh -> FFN."""

    def __init__(self, d, nhead=8, ff=2048, drop=0.1):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        self.ca = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        self.l1, self.l2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.d0, self.d1, self.d2, self.d3 = nn.Dropout(drop), nn.Dropout(drop), nn.Dropout(drop), nn.Dropout(drop)

    def forward(self, tgt, qpos, mem, mpos):
        q = k = tgt + qpos
        tgt = self.n1(tgt + self.d1(self.sa(q, k, tgt, need_weights=False)[0]))
        tgt = self.n2(tgt + self.d2(self.ca(tgt + qpos, mem + mpos, mem, need_weights=False)[0]))
        tgt = self.n3(tgt + self.d3(self.l2(self.d0(F.relu(self.l1(tgt))))))
        return tgt


class CascadeHead(nn.Module):
    """Hoi quy phan du theo tang (giong cascade_fc cua JOTR): lop 0 du doan tu token; lop l > 0 du doan phan du tu
    [token, du doan truoc]. init (1, k) | None: lop 0 = init + Linear(token) (vd pose trung binh)."""

    def __init__(self, d, k, n_layers, init=None):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(d, k)] + [nn.Linear(d + k, k) for _ in range(n_layers - 1)])
        for m in self.heads:                                                         # khoi tao nho: luc dau ~ init
            nn.init.normal_(m.weight, std=1e-3)
            nn.init.zeros_(m.bias)
        if init is not None:
            self.register_buffer('init', init.float().reshape(1, k))
        else:
            self.init = None

    def forward(self, i, h, prev):
        if i == 0:
            out = self.heads[0](h)
            return out + self.init if self.init is not None else out
        return prev + self.heads[i](torch.cat([h, prev], dim=-1))


class V8Head(nn.Module):
    """Encoder + decoder + cac head + SMPL (= smpl_model cua V8Net)."""

    def __init__(self, dim=256, n_enc=3, n_dec=3, img_ch=2048, use_lift=False, use_hyp=False, n_hyp=5,
                 use_ik=False, use_hgcn=False):
        super().__init__()
        self.n_dec = int(n_dec)
        self.use_lift, self.use_hyp, self.use_ik, self.use_hgcn = bool(use_lift), bool(use_hyp), bool(use_ik), \
            bool(use_hgcn)
        self.img_proj = nn.Conv2d(img_ch, dim, 1)
        self.encoder = nn.ModuleList([EncLayer(dim) for _ in range(int(n_enc))])
        self.decoder = nn.ModuleList([DecLayer(dim) for _ in range(self.n_dec)])
        self.dec_norm = nn.LayerNorm(dim)                       # JOTR: norm sau MOI lop decoder, dua vao lop tiep
        self.query_pos = nn.Embedding(N_SMPL_TOK + N_JOINT, dim)
        self.tgt = nn.Parameter(torch.zeros(1, N_SMPL_TOK + N_JOINT, dim))   # noi dung ban dau = 0 (giong JOTR)
        for p in list(self.encoder.parameters()) + list(self.decoder.parameters()):  # JOTR _reset_parameters
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if not self.use_ik:                                     # M0: hoi quy 6D + beta truc tiep (nhu JOTR)
            mean = np.load(SMPL_MEAN_PARAMS)
            init_pose = torch.from_numpy(mean['pose'][:].astype('float32')).reshape(1, 24 * 6)
            init_shape = torch.from_numpy(mean['shape'][:].astype('float32')).reshape(1, 10)
            self.pose_head = CascadeHead(dim, 24 * 6, self.n_dec, init_pose)
            self.shape_head = CascadeHead(dim, 10, self.n_dec, init_shape)
        self.cam_head = CascadeHead(dim, 3, self.n_dec)
        self.joint_head = CascadeHead(dim, 3, self.n_dec)      # khop 3D root-relative (met), moi query 1 khop

        # ---- M2 [v8_lifter]: pose lifter -> noi dung ban dau cua 17 query khop ----
        if self.use_lift:
            self.lift_embed = nn.Linear(3, dim)
        # ---- M2 [v8_hyp_tokens]: K x 17 gia thuyet -> token bo nho decoder ----
        if self.use_hyp:
            assert self.use_lift, 'v8_hyp_tokens can v8_lifter'
            self.n_hyp = int(n_hyp)
            self.hyp_embed = nn.Linear(3, dim)
            self.hyp_joint_id = nn.Embedding(N_JOINT, dim)
            self.hyp_id = nn.Embedding(self.n_hyp, dim)
            self.hyp_norm = nn.LayerNorm(dim)                   # dua ve cung thang do voi token anh (encoder post-norm)
        # ---- M2 [v8_hgcn]: 1 khoi HyperGCN (HyperDiff) sau moi lop decoder, layer scale 1e-5 -> luc dau ~ dong nhat ----
        if self.use_hgcn:
            from models.HGraph import create_layers
            self.hgcn = nn.ModuleList([
                create_layers(dim=dim, n_layers=1, mlp_ratio=4, act_layer=nn.GELU, attn_drop=0., drop_rate=0.,
                              drop_path_rate=0., use_layer_scale=True, layer_scale_init_value=1e-5,
                              use_adaptive_fusion=False, hierarchical=False, neighbour_num=4)
                for _ in range(self.n_dec)])

        from utils.smpl import SMPL as SMPLModel
        self.human_model = SMPLModel()
        self.human_model_layer = self.human_model.layer['neutral']
        self.register_buffer('joint_regressor_t', torch.from_numpy(self.human_model.joint_regressor).float())

        # ---- M2 [v8_ik]: IK HybrIK (nhu Student v7). Ten 'hybrik' BAT BUOC: Student_Trainer._v6_j29_loss doc no ----
        if self.use_ik:
            from models.hybrik_smpl import HybrIKSMPL
            reg_path = os.path.join(cfg.DATASET.jotr_data_root, 'Human36M', 'J_regressor_h36m_correct.npy')
            self.hybrik = HybrIKSMPL(self.human_model_layer, np.load(reg_path))
            Mmap = np.load(cfg.MODEL.v6_m_path).astype(np.float32)
            assert Mmap.shape == (29, 17), f'v6_m_path phai la ma tran (29, 17), nhan {Mmap.shape}'
            self.register_buffer('v8_M', torch.from_numpy(Mmap))
            self.register_buffer('bone_pairs', torch.tensor(SMPL_BONE_PAIRS_H36M, dtype=torch.long)[1:],
                                 persistent=False)                      # (23, 2): khop SMPL 1..23
            # Tat ca khoi tao 0 -> luc dau mesh = IK(M x khop, beta_xuong, twist 0)
            self.v8_dlimb_w = nn.Parameter(torch.zeros(len(V7_LIMB_IDX), dim, 3))   # moi diem dau chi 1 head rieng
            self.v8_dlimb_b = nn.Parameter(torch.zeros(len(V7_LIMB_IDX), 3))
            self.v8_tw = nn.Linear(2 * dim, 2)                    # twist (cos, sin) += f(token cha, token con)
            self.v8_tw_bias = nn.Parameter(torch.zeros(23, 2))   # bias rieng tung khop (cac khop chung cap token)
            self.v8_db = nn.Linear(dim, 10)                       # dbeta tu token shape
            for m in (self.v8_tw, self.v8_db):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    # ---------------- SMPL (chep tu Student, cung quy uoc camera / chieu) ----------------
    def get_camera_trans(self, cam_param):
        """cam_param (B, 3) = [tx, ty, gamma] -> cam_trans (B, 3) = [tx, ty, tz]."""
        t_xy = cam_param[:, :2]
        gamma = torch.sigmoid(cam_param[:, 2])
        k_value = math.sqrt(cfg.DATASET.focal[0] * cfg.DATASET.focal[1] * cfg.DATASET.camera_3d_size
                            * cfg.DATASET.camera_3d_size / (cfg.input_img_shape[0] * cfg.input_img_shape[1]))
        return torch.cat([t_xy, (k_value * gamma)[:, None]], dim=1)

    def get_coord(self, smpl_pose, smpl_shape, smpl_trans):
        B = smpl_pose.shape[0]
        mesh_cam, _ = self.human_model_layer(smpl_pose, smpl_shape, smpl_trans)
        joint_cam = torch.bmm(self.joint_regressor_t[None].repeat(B, 1, 1), mesh_cam)
        x = joint_cam[:, :, 0] / (joint_cam[:, :, 2] + 1e-4) * cfg.DATASET.focal[0] + cfg.DATASET.princpt[0]
        y = joint_cam[:, :, 1] / (joint_cam[:, :, 2] + 1e-4) * cfg.DATASET.focal[1] + cfg.DATASET.princpt[1]
        x = x / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
        y = y / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x, y), 2)
        mesh_cam_render = mesh_cam.clone()
        root_cam = joint_cam[:, self.human_model.root_joint_idx, None, :]
        return joint_proj, joint_cam - root_cam, mesh_cam - root_cam, mesh_cam_render

    def smpl_outputs(self, pose6d, shape, cam_raw, is_train):
        """pose6d (B, 144) + shape (B, 10) + camera tho (B, 3) -> dict output nhu Student._smpl_outputs."""
        B = pose6d.shape[0]
        aa = rotmat_to_aa_safe(rot6d_to_rotmat(pose6d.float().reshape(B * 24, 6))).reshape(B, 24, 3)
        return self.smpl_from_aa(aa, shape, cam_raw, is_train)

    def ik_outputs(self, joints, jt, s_tok, cam_raw, is_train):
        """M2 [v8_ik]: khop 3D da sua (B, 17, 3) + token khop jt (B, 17, D) + token shape (B, D) -> IK -> SMPL.
        Giong Student v7 _mesh_outputs (readout = token khop truc tiep, khong qua sieu canh)."""
        B = joints.shape[0]
        pose17 = (joints - joints[:, 0:1]).float()                                   # root-relative tai Pelvis H36M
        j29 = torch.einsum('jk,bkc->bjc', self.v8_M, pose17)                           # (B, 29, 3)
        q = jt[:, V8_LIMB_TOKEN].float()                                              # (B, 7, D)
        dlimb = torch.einsum('bpd,pdc->bpc', q, self.v8_dlimb_w) + self.v8_dlimb_b    # (B, 7, 3)
        d29 = torch.zeros_like(j29)
        d29[:, V7_LIMB_IDX] = dlimb
        j29 = j29 + d29
        with torch.no_grad():                                                         # beta tu do dai xuong (hinh hoc)
            beta0 = self.hybrik.beta_from_bones(pose17.detach(), int(cfg.MODEL.v7_bone_gn_iters),
                                                float(cfg.MODEL.v7_bone_beta_reg))
        dbeta = self.v8_db(s_tok).float()
        beta = beta0 + dbeta
        tw_in = torch.cat([jt[:, self.bone_pairs[:, 0]], jt[:, self.bone_pairs[:, 1]]], dim=-1)   # (B, 23, 2D)
        phis = self.hybrik.default_phis(B, j29.device, torch.float32) + self.v8_tw(tw_in).float() + self.v8_tw_bias
        rot_mats = self.hybrik.ik(j29, beta, phis)                                    # (B, 24, 3, 3)
        aa = rotmat_to_aa_safe(rot_mats.reshape(-1, 3, 3)).reshape(B, 24, 3)
        out = self.smpl_from_aa(aa, beta, cam_raw, is_train)
        out['root_pose_6d'] = rot_mats[:, 0, :, :2].reshape(B, 6)
        out['v7_j29'] = j29                                                           # loss J29 (7 diem dau chi)
        out['v7_dbeta'] = dbeta                                                       # rang buoc |dbeta|^2
        out['v7_beta0'] = beta0
        return out

    def smpl_from_aa(self, aa, shape, cam_raw, is_train):
        """Goc axis-angle (B, 24, 3) + beta (B, 10) + camera tho (B, 3) -> dict output SMPL."""
        B = aa.shape[0]
        aa = torch.cat([aa[:, :22], torch.zeros_like(aa[:, 22:])], dim=1).reshape(B, 72)   # ban tay = 0 (nhu Student)
        shape = shape.float()
        cam_trans = self.get_camera_trans(cam_raw.float())
        joint_proj, joint_cam, mesh_cam, mesh_render = self.get_coord(aa, shape, cam_trans)
        if is_train:                                    # loss chieu 2D: detach pose (giu shape, camera), nhu Student
            _, _, _, mesh_render_det = self.get_coord(aa.detach(), shape, cam_trans)
        else:
            mesh_render_det = mesh_render
        return {'joint_proj': joint_proj, 'joint_cam': joint_cam, 'smpl_mesh_cam': mesh_cam,
                'smpl_mesh_cam_render': mesh_render, 'smpl_mesh_cam_render_det': mesh_render_det,
                'smpl_pose': aa, 'smpl_shape': shape, 'cam_param': cam_trans}

    def forward(self, fmap, is_train=True, lift=None, hyp=None):
        """fmap (B, 2048, h, w). M2: lift (B, 17, 3) pose lifter root-relative (met), hyp (B, K, 17, 3) gia thuyet."""
        B, _, h, w = fmap.shape
        mem = self.img_proj(fmap).flatten(2).transpose(1, 2)                          # (B, h*w, D)
        mpos = sine_pos_2d(h, w, mem.shape[-1], mem.device, mem.dtype)[None]          # (1, h*w, D)
        for layer in self.encoder:
            mem = layer(mem, mpos)
        if self.use_hyp:                                                              # [v8_hyp_tokens]
            assert hyp is not None and hyp.shape[1] == self.n_hyp, \
                f'v8_hyp_tokens can hyp (B, {self.n_hyp}, 17, 3), nhan {None if hyp is None else tuple(hyp.shape)}'
            ht = self.hyp_embed(hyp.to(mem.dtype)) + self.hyp_joint_id.weight[None, None] + self.hyp_id.weight[None, :, None]
            ht = self.hyp_norm(ht.reshape(B, self.n_hyp * N_JOINT, -1))               # (B, K*17, D)
            mem = torch.cat([mem, ht], dim=1)
            mpos = torch.cat([mpos.expand(B, -1, -1), torch.zeros_like(ht)], dim=1)   # token gia thuyet: PE = 0
        qpos = self.query_pos.weight[None]                                           # (1, 20, D)
        tgt = self.tgt.expand(B, -1, -1)
        if self.use_lift:                                                             # [v8_lifter] query khop biet pose lifter
            assert lift is not None, 'v8_lifter can pose lifter'
            lift = lift.to(mem.dtype)
            tgt = torch.cat([tgt[:, :N_SMPL_TOK], tgt[:, N_SMPL_TOK:] + self.lift_embed(lift)], dim=1)
        pose = shape = cam = joints = None
        per_layer = []
        for i, layer in enumerate(self.decoder):
            tgt = self.dec_norm(layer(tgt, qpos, mem, mpos))
            if self.use_hgcn:                                                         # [v8_hgcn] tren 17 token khop
                jt = self.hgcn[i](tgt[:, N_SMPL_TOK:].unsqueeze(1)).squeeze(1)
                tgt = torch.cat([tgt[:, :N_SMPL_TOK], jt], dim=1)
            jt = tgt[:, N_SMPL_TOK:]                                                  # (B, 17, D)
            cam = self.cam_head(i, tgt[:, 2], cam)
            if i == 0 and self.use_lift:                                              # lop 0: lifter + phan du nho
                joints = lift + self.joint_head(0, jt, None)
            else:
                joints = self.joint_head(i, jt, joints)                               # (B, 17, 3)
            if self.use_ik:
                per_layer.append(('ik', joints, jt, tgt[:, 1], cam))
            else:
                pose = self.pose_head(i, tgt[:, 0], pose)
                shape = self.shape_head(i, tgt[:, 1], shape)
                per_layer.append(('6d', pose, shape, cam))

        def layer_out(item):
            if item[0] == 'ik':
                return self.ik_outputs(item[1], item[2], item[3], item[4], is_train)
            return self.smpl_outputs(item[1], item[2], item[3], is_train)

        result = layer_out(per_layer[-1])
        # Cac lop truoc lop cuoi -> refine_history (base._hard_losses giam sat voi trong so v3_refine_gamma)
        result['refine_history'] = [layer_out(it) for it in per_layer[:-1]] if is_train else []
        j3d = joints - joints[:, 0:1]                                                # root-relative tai Pelvis
        result['ctx_3d'] = j3d                                                        # loss ctx (v5_ctx_loss_w)
        # cot LIFT luc danh gia: M2 = pose lifter dau vao (so voi khop da sua o ctx_3d); M0 = query khop
        result['lifted_joints_3d'] = (lift - lift[:, 0:1]).detach() if self.use_lift else j3d.detach()
        result['privileged_3d'] = j3d                                                 # tat bang v7_no_privileged
        result['feat_joint'] = tgt[:, N_SMPL_TOK:]
        result['feat_global'] = tgt[:, :N_SMPL_TOK].mean(1)
        return result


class V8Net(nn.Module):
    """Vo boc FuseKin v8 (thay ARTS khi MODEL.v8_e2e): backbone + heatmap 2D + V8Head."""

    def __init__(self):
        super().__init__()
        self.mode = 'v8'
        self.backbone = ResNetBackbone(cfg.MODEL.resnet_type)                       # BN thuong (train), nhu JOTR
        self.backbone.enable_heatmap_fusion(N_JOINT)                                 # conv khoi tao 0
        bb_src = cfg.MODEL.get('backbone_pretrained', '')
        if bb_src:
            self.backbone.load_pretrained(
                bb_src, spin_checkpoint=cfg.MODEL.get('spin_checkpoint',
                                                      'data_final/base_data/spin_model_checkpoint.pth.tar'))
        else:
            print('[CANH BAO] V8: backbone KHONG nap pretrain -> dac trung anh ngau nhien.')
        # ---- M2 [v8_lifter]: Refiner 2D + ImgLifter dong bang (giong ARTS), luon eval ----
        self.use_lift = bool(cfg.MODEL.get('v8_lifter', False))
        self.use_hyp = bool(cfg.MODEL.get('v8_hyp_tokens', False))
        if self.use_lift:
            from utils.img_lifter_tools import load_img_lifter, load_frozen_refiner
            assert cfg.MODEL.imglift_checkpoint, 'v8_lifter can MODEL.imglift_checkpoint'
            cpu = torch.device('cpu')
            self.img_lifter, ck = load_img_lifter(cfg.MODEL.imglift_checkpoint, cpu)
            self.refiner2d = load_frozen_refiner(cfg.MODEL.get('imglift_refiner_checkpoint', '') or ck['refiner_ckpt'], cpu)
            for p in self.img_lifter.parameters():
                p.requires_grad = False
            self.img_lifter.eval()
        self.smpl_model = V8Head(dim=int(cfg.MODEL.v8_dim), n_enc=int(cfg.MODEL.v8_enc_layers),
                                 n_dec=int(cfg.MODEL.v8_dec_layers), use_lift=self.use_lift, use_hyp=self.use_hyp,
                                 n_hyp=int(cfg.MODEL.imglift_K), use_ik=bool(cfg.MODEL.get('v8_ik', False)),
                                 use_hgcn=bool(cfg.MODEL.get('v8_hgcn', False)))
        self.joint_drop = float(cfg.MODEL.v8_joint_drop)
        print(f'===> FuseKin v8: dim {cfg.MODEL.v8_dim} | encoder {cfg.MODEL.v8_enc_layers} lop | decoder '
              f'{cfg.MODEL.v8_dec_layers} lop | che khop 2D luc train {self.joint_drop} | lifter {self.use_lift} '
              f'(K={cfg.MODEL.imglift_K}, S={cfg.MODEL.imglift_S}) | token gia thuyet {self.use_hyp} | '
              f'IK {cfg.MODEL.get("v8_ik", False)} | HyperGCN {cfg.MODEL.get("v8_hgcn", False)}')
        if cfg.MODEL.get('v8_ik', False):
            assert cfg.MODEL.get('v7_no_pose_param_loss', False), \
                'v8_ik: goc do IK quyet dinh -> dat MODEL.v7_no_pose_param_loss True'

    def train(self, mode=True):
        super().train(mode)
        if self.use_lift:                                   # lifter / Refiner dong bang: luon eval
            self.img_lifter.eval()
            self.refiner2d.eval()
        return self

    @torch.no_grad()
    def lift(self, image, p2, jm, cam_bbox):
        """Chep ARTS.lift_imglift: anh + 2D detector -> Refiner day -> ImgLifter (K x S, gop mean).
        Tra pose (B, 17, 3) root-relative met, hyp (B, K, 17, 3)."""
        from models.img_lifter import ImgLifter, aggregate
        from utils.img_lifter_tools import refiner_inputs, lifter_det
        B = image.shape[0]
        jm = jm.reshape(B, N_JOINT).float()
        nc = int(getattr(self.img_lifter, 'num_ctx', 0))
        xy, log_b, f8, fj = refiner_inputs(self.refiner2d, image, p2[..., :2].float(), jm, N_JOINT + nc)
        cond7 = ImgLifter.make_cond7(xy, log_b, lifter_det(jm, nc), cam_bbox.float())
        gen = None
        if not self.training:                               # test: nhieu co dinh -> lap lai duoc (nhu ARTS)
            gen = torch.Generator(device=image.device)
            gen.manual_seed(0)
        hyp = self.img_lifter.sample(cond7, fj, f8, K=int(cfg.MODEL.imglift_K), S=int(cfg.MODEL.imglift_S),
                                     eta=1.0, generator=gen)
        pose = aggregate(hyp, xy, log_b, 'mean')
        return pose - pose[:, 0:1], hyp - hyp[:, :, 0:1]

    def forward(self, image, joints, is_train=True, joints_mask=None, pose_2d=None, cam_bbox=None):
        """joints = 2D detector (B, 17, 2|3) trong [-1, 1]; joints_mask (B, 17, 1). pose_2d: khong dung.
        cam_bbox (B, 3): bat buoc khi v8_lifter (dieu kien cua ImgLifter), M0 khong dung."""
        B = image.shape[0]
        p2 = joints[:, 0] if joints.dim() == 4 else joints
        if joints_mask is None:
            m = (p2[..., 2:3] > 0).float() if p2.shape[-1] >= 3 else torch.ones_like(p2[..., :1])
        else:
            m = joints_mask.reshape(B, N_JOINT, 1).float()
        lift = hyp = None
        if self.use_lift:                                   # lifter nhan mask detector DAY DU (truoc khi che ngau nhien)
            assert cam_bbox is not None, "v8_lifter can inputs['cam_bbox']"
            lift, hyp = self.lift(image, p2, m, cam_bbox)
        if self.training and self.joint_drop > 0:
            # Giong JOTR: che ngau nhien mot phan khop 2D dau vao luc train -> mang phai doc anh de bu
            m = m * (torch.rand_like(m) >= self.joint_drop).float()
        heat = make_heatmaps(p2, m, size=image.shape[-1] // 4, sigma=float(cfg.MODEL.v3_heatmap_sigma))
        fmap = self.backbone(image, heatmap=heat)[0]                                 # (B, 2048, 8, 8)
        return self.smpl_model(fmap, is_train=is_train, lift=lift, hyp=hyp if self.use_hyp else None)
