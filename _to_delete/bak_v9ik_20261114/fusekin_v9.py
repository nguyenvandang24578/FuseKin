"""
lib/models/fusekin_v9.py -- FuseKin v9 (2026-11-14, nguoi dung duyet): Student HOI QUY mesh tu anh + 2D + 3D.
Lifter chi la NGUON BANG CHUNG (khong neo, khong "sua lifter").

  anh + heatmap 2D (tiem stride 4) -> ResNet-50 SPIN (train) -> F32 (2048 x 8 x 8) + F16 (1024 x 16 x 16)
  Refiner 2D + ImgLifter no3dpw (DONG BANG) -> xy Refiner + log_b (17), pose 3D mean + K gia thuyet

  [1] 17 token bang chung (moi khop):
        anh  : F16 lay mau bilinear tai xy Refiner cua khop
        2D   : MLP([x, y OpenPose x mask, mask, x, y Refiner, log_b])
        3D   : MLP([pose lifter mean (3), do lech chuan K gia thuyet x 10 (3)])  (modality dropout luc train)
        vi tri (qpos): ID khop + PE sin tai xy Refiner (cung he toa do voi PE cua token anh)
  [2] khoi fuse (v9_fuse_layers lop): self-attn 17 khop -> cross-attn 64 token anh (F32 + PE, qua v9_img_enc lop
        encoder) -> FFN -> HyperGCN HyperDiff (1 khoi / lop)
  [3] query SMPL: 24 query khop SMPL (khoi tao = Linear(token cha, token con) theo SMPL_BONE_PAIRS_H36M) + token shape
        + token camera -> decoder (v9_smpl_dec lop) cross-attn vao [17 token khop da fuse, 64 token anh]
        -> moi lop hoi quy phan du theo tang: 6D tung khop (bat dau tu pose TRUNG BINH), beta, camera
  [4] SMPL -> mesh. Loss nhu M0 (base._smpl_hard_losses qua AWL, ca cac lop decoder) + khop 3D phu tu token khop
      (ctx_3d, v5_ctx_loss_w).

Giao dien giong ARTS / V8Net -> Student_Trainer / Tester / test_student_3dpw_sets dung nguyen.

v9 reliability (MODEL.v9_reliability, 2026-11-14, nguoi dung duyet; mac dinh TAT): head nho doan log loi THAT cua tung
nguon o tung khop [3D lifter (met), 2D Refiner ([-1, 1])] tu [anh tai khop, 2D, 3D]; giam sat bang loi that tren tap
train (base._v9_rel_loss, MODEL.v9_rel_w). Cong g = sigmoid(w * (-log loi du doan, detach) + b) nhan vao token 2D / 3D
-> nguon cang sai cang it duoc tin khi fuse.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.config import cfg
from models.backbones.resnet import ResNetBackbone
from models.ARTS import make_heatmaps
from models.hybrik_smpl import rotmat_to_aa_safe
from models.fusekin_v8 import EncLayer, DecLayer, CascadeHead, V8Head, rot6d_to_rotmat, SMPL_MEAN_PARAMS
from models.teacher_student import SMPL_BONE_PAIRS_H36M

N_JOINT = 17
N_SMPL = 24


# ---------------- PE sin 2D (cung cong thuc sine_pos_2d cua V8) ----------------
def sine_embed(gy, gx, dim, temperature=10000.0):
    """gy, gx (...) da nhan 2 pi -> (..., dim)."""
    npf = dim // 2
    dim_t = torch.arange(npf, device=gy.device, dtype=torch.float32)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / npf)
    px = gx[..., None].float() / dim_t
    py = gy[..., None].float() / dim_t
    px = torch.stack([px[..., 0::2].sin(), px[..., 1::2].cos()], dim=-1).flatten(-2)
    py = torch.stack([py[..., 0::2].sin(), py[..., 1::2].cos()], dim=-1).flatten(-2)
    return torch.cat([py, px], dim=-1)


def grid_pos(h, w, dim, device):
    """PE cho luoi h x w -> (h * w, dim); o (i, j) co toa do ((i + 1) / h, (j + 1) / w) x 2 pi (nhu V8)."""
    y = (torch.arange(h, device=device, dtype=torch.float32) + 1) / h * 2 * math.pi
    x = (torch.arange(w, device=device, dtype=torch.float32) + 1) / w * 2 * math.pi
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    return sine_embed(gy, gx, dim).reshape(h * w, dim)


def point_pos(xy, h, w, dim):
    """xy (B, J, 2) trong [-1, 1] -> PE (B, J, dim), CUNG he voi grid_pos (chi so o lien tuc, align_corners=False)."""
    u = (xy[..., 0].float() + 1) * 0.5 * w - 0.5
    v = (xy[..., 1].float() + 1) * 0.5 * h - 0.5
    return sine_embed((v + 1) / h * 2 * math.pi, (u + 1) / w * 2 * math.pi, dim)


def mlp(i, d):
    return nn.Sequential(nn.Linear(i, d), nn.ReLU(inplace=True), nn.Linear(d, d))


class V9Head(nn.Module):
    """Token bang chung + fuse + query SMPL + SMPL (= smpl_model cua V9Net)."""
    # Dung lai SMPL / camera / chieu cua V8 (cung quy uoc voi Student)
    get_camera_trans = V8Head.get_camera_trans
    get_coord = V8Head.get_coord
    smpl_from_aa = V8Head.smpl_from_aa

    def __init__(self, dim=256, n_img_enc=2, n_fuse=3, n_dec=3, use_hgcn=True, img_ch=2048, f16_ch=1024,
                 use_rel=False):
        super().__init__()
        self.n_dec = int(n_dec)
        # ---- [v9_reliability] head doan log loi [3D lifter, 2D Refiner] tung khop + cong ----
        self.use_rel = bool(use_rel)
        if self.use_rel:
            self.rel_head = nn.Sequential(nn.Linear(3 * dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, 2))
            nn.init.normal_(self.rel_head[-1].weight, std=1e-3)
            with torch.no_grad():                               # luc dau ~ loi dien hinh: 3D ~7 cm, 2D ~0.06 (~8 px)
                self.rel_head[-1].bias.copy_(torch.tensor([math.log(0.07), math.log(0.06)]))
            self.gate_w = nn.Parameter(torch.ones(2))
            self.gate_b = nn.Parameter(torch.zeros(2))           # luc dau cong ~ sigmoid(2.7) ~ 0.94
        # ---- token anh ----
        self.img_proj = nn.Conv2d(img_ch, dim, 1)
        self.img_enc = nn.ModuleList([EncLayer(dim) for _ in range(int(n_img_enc))])
        # ---- [1] token bang chung ----
        self.f16_proj = nn.Conv2d(f16_ch, dim, 1)
        self.emb2d = mlp(6, dim)
        self.emb3d = mlp(6, dim)
        self.miss3d = nn.Parameter(torch.zeros(1, 1, dim))     # thay token 3D khi bi bo (modality dropout)
        self.joint_id = nn.Embedding(N_JOINT, dim)
        self.tok_norm = nn.LayerNorm(dim)
        # ---- [2] fuse ----
        self.fuse = nn.ModuleList([DecLayer(dim) for _ in range(int(n_fuse))])
        self.hgcn = None
        if use_hgcn:
            from models.HGraph import create_layers
            self.hgcn = nn.ModuleList([
                create_layers(dim=dim, n_layers=1, mlp_ratio=4, act_layer=nn.GELU, attn_drop=0., drop_rate=0.,
                              drop_path_rate=0., use_layer_scale=True, layer_scale_init_value=1e-5,
                              use_adaptive_fusion=False, hierarchical=False, neighbour_num=4)
                for _ in range(int(n_fuse))])
        self.j3d_head = nn.Linear(dim, 3)                       # khop 3D phu (loss ctx_3d), KHONG dung lam mesh
        # ---- [3] query SMPL ----
        self.register_buffer('bone_pairs', torch.tensor(SMPL_BONE_PAIRS_H36M, dtype=torch.long), persistent=False)
        self.bone_proj = nn.Linear(2 * dim, dim)
        self.smpl_query = nn.Embedding(N_SMPL + 2, dim)        # noi dung hoc them (24 khop + shape + camera)
        self.smpl_qpos = nn.Embedding(N_SMPL + 2, dim)
        self.mem_type = nn.Embedding(2, dim)                    # 0 = token khop, 1 = token anh (trong bo nho decoder)
        self.dec = nn.ModuleList([DecLayer(dim) for _ in range(self.n_dec)])
        self.dec_norm = nn.LayerNorm(dim)
        for p in list(self.img_enc.parameters()) + list(self.fuse.parameters()) + list(self.dec.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        mean = np.load(SMPL_MEAN_PARAMS)
        self.register_buffer('init_pose', torch.from_numpy(mean['pose'][:].astype('float32')).reshape(1, N_SMPL, 6))
        init_shape = torch.from_numpy(mean['shape'][:].astype('float32')).reshape(1, 10)
        self.pose_heads = nn.ModuleList([nn.Linear(dim, 6)] + [nn.Linear(dim + 6, 6) for _ in range(self.n_dec - 1)])
        for m in self.pose_heads:                               # khoi tao nho: luc dau ~ pose trung binh
            nn.init.normal_(m.weight, std=1e-3)
            nn.init.zeros_(m.bias)
        self.shape_head = CascadeHead(dim, 10, self.n_dec, init_shape)
        self.cam_head = CascadeHead(dim, 3, self.n_dec)
        # ---- SMPL ----
        from utils.smpl import SMPL as SMPLModel
        self.human_model = SMPLModel()
        self.human_model_layer = self.human_model.layer['neutral']
        self.register_buffer('joint_regressor_t', torch.from_numpy(self.human_model.joint_regressor).float())

    def smpl_outputs(self, pose6d, shape, cam_raw, is_train):
        """pose6d (B, 24, 6) -> dict output SMPL (nhu V8 / Student)."""
        B = pose6d.shape[0]
        aa = rotmat_to_aa_safe(rot6d_to_rotmat(pose6d.float().reshape(B * N_SMPL, 6))).reshape(B, N_SMPL, 3)
        return self.smpl_from_aa(aa, shape, cam_raw, is_train)

    def forward(self, fmap, f16, op_xy, op_mask, ref_xy, ref_logb, lift, hyp, is_train=True):
        """fmap (B, 2048, h, w), f16 (B, 1024, 2h, 2w); op_xy (B, 17, 2) OpenPose [-1, 1], op_mask (B, 17, 1);
        ref_xy (B, 17, 2) Refiner [-1, 1], ref_logb (B, 17); lift (B, 17, 3) met root-relative; hyp (B, K, 17, 3)."""
        B, _, h, w = fmap.shape
        # ---- token anh ----
        mem = self.img_proj(fmap).flatten(2).transpose(1, 2)                          # (B, h*w, D)
        D = mem.shape[-1]
        mpos = grid_pos(h, w, D, mem.device).to(mem.dtype)[None]                      # (1, h*w, D)
        for layer in self.img_enc:
            mem = layer(mem, mpos)
        # ---- [1] token bang chung ----
        f = self.f16_proj(f16)
        img_j = F.grid_sample(f, ref_xy[:, :, None, :].to(f.dtype), mode='bilinear',
                              align_corners=False).squeeze(-1).transpose(1, 2)        # (B, 17, D)
        in2d = torch.cat([op_xy * op_mask, op_mask, ref_xy, ref_logb[..., None]], dim=-1).to(mem.dtype)
        e2d = self.emb2d(in2d)
        std = hyp.std(dim=1) if hyp.shape[1] > 1 else torch.zeros_like(lift)
        e3d = self.emb3d(torch.cat([lift, std * 10.0], dim=-1).to(mem.dtype))        # std x 10: dua ve bac ~ lift
        rel_pred = gate = None
        if self.use_rel:                                                              # [v9_reliability]
            rel_pred = self.rel_head(torch.cat([img_j, e2d, e3d], dim=-1))            # (B, 17, 2) log loi [3D, 2D]
            gate = torch.sigmoid(self.gate_w * (-rel_pred.detach()) + self.gate_b)    # cang sai -> cang nho
            e3d = e3d * gate[..., 0:1]
            e2d = e2d * gate[..., 1:2]
        if self.training:                                                             # modality dropout cho 3D
            keep = torch.ones(B, N_JOINT, 1, device=e3d.device, dtype=e3d.dtype)
            ps, pj = float(cfg.MODEL.v9_drop3d_sample), float(cfg.MODEL.v9_drop3d_joint)
            if ps > 0:
                keep = keep * (torch.rand(B, 1, 1, device=e3d.device) >= ps).to(e3d.dtype)
            if pj > 0:
                keep = keep * (torch.rand(B, N_JOINT, 1, device=e3d.device) >= pj).to(e3d.dtype)
            e3d = keep * e3d + (1 - keep) * self.miss3d
        jt = self.tok_norm(img_j + e2d + e3d)                                         # (B, 17, D)
        jpos = self.joint_id.weight[None] + point_pos(ref_xy, h, w, D).to(mem.dtype)  # (B, 17, D)
        # ---- [2] fuse: self-attn khop -> cross-attn anh -> FFN -> HyperGCN ----
        for i, layer in enumerate(self.fuse):
            jt = layer(jt, jpos, mem, mpos)
            if self.hgcn is not None:
                jt = self.hgcn[i](jt.unsqueeze(1)).squeeze(1)
        j3d = self.j3d_head(jt)
        j3d = j3d - j3d[:, 0:1]                                                       # root-relative tai Pelvis
        # ---- [3] query SMPL ----
        bp = self.bone_pairs
        bone = self.bone_proj(torch.cat([jt[:, bp[:, 0]], jt[:, bp[:, 1]]], dim=-1))  # (B, 24, D)
        q = torch.cat([bone, jt.new_zeros(B, 2, D)], dim=1) + self.smpl_query.weight[None]   # (B, 26, D)
        qpos = self.smpl_qpos.weight[None]
        dmem = torch.cat([jt, mem], dim=1)
        dpos = torch.cat([jpos + self.mem_type.weight[0], (mpos + self.mem_type.weight[1]).expand(B, -1, -1)], dim=1)
        pose = shape = cam = None
        per_layer = []
        for i, layer in enumerate(self.dec):
            q = self.dec_norm(layer(q, qpos, dmem, dpos))
            pt = q[:, :N_SMPL]
            if i == 0:
                pose = self.init_pose + self.pose_heads[0](pt)                       # (B, 24, 6)
            else:
                pose = pose + self.pose_heads[i](torch.cat([pt, pose], dim=-1))
            shape = self.shape_head(i, q[:, N_SMPL], shape)
            cam = self.cam_head(i, q[:, N_SMPL + 1], cam)
            per_layer.append((pose, shape, cam))
        result = self.smpl_outputs(*per_layer[-1], is_train)
        result['refine_history'] = [self.smpl_outputs(p, s, c, is_train) for (p, s, c) in per_layer[:-1]] \
            if is_train else []
        result['ctx_3d'] = j3d                                                        # loss ctx (v5_ctx_loss_w)
        result['lifted_joints_3d'] = (lift - lift[:, 0:1]).detach()                  # cot LIFT = pose lifter
        result['privileged_3d'] = j3d                                                 # tat bang v7_no_privileged
        result['feat_joint'] = jt
        result['feat_global'] = q[:, N_SMPL:].mean(1)
        if self.use_rel:                                                              # cho base._v9_rel_loss
            result['v9_rel_pred'] = rel_pred
            result['v9_rel_gate'] = gate
            result['v9_ref_xy'] = ref_xy
        return result


class V9Net(nn.Module):
    """Vo boc FuseKin v9 (thay ARTS khi MODEL.v9_e2e): backbone + heatmap 2D + Refiner / ImgLifter dong bang + V9Head."""

    def __init__(self):
        super().__init__()
        self.mode = 'v9'
        self.backbone = ResNetBackbone(cfg.MODEL.resnet_type)                       # BN thuong (train), nhu V8
        self.backbone.enable_heatmap_fusion(N_JOINT)
        bb_src = cfg.MODEL.get('backbone_pretrained', '')
        if bb_src:
            self.backbone.load_pretrained(
                bb_src, spin_checkpoint=cfg.MODEL.get('spin_checkpoint',
                                                      'data_final/base_data/spin_model_checkpoint.pth.tar'))
        else:
            print('[CANH BAO] V9: backbone KHONG nap pretrain -> dac trung anh ngau nhien.')
        # Refiner 2D + ImgLifter (dong bang, luon eval)
        from utils.img_lifter_tools import load_img_lifter, load_frozen_refiner
        assert cfg.MODEL.imglift_checkpoint, 'v9 can MODEL.imglift_checkpoint (lifter no3dpw)'
        cpu = torch.device('cpu')
        self.img_lifter, ck = load_img_lifter(cfg.MODEL.imglift_checkpoint, cpu)
        self.refiner2d = load_frozen_refiner(cfg.MODEL.get('imglift_refiner_checkpoint', '') or ck['refiner_ckpt'], cpu)
        for p in self.img_lifter.parameters():
            p.requires_grad = False
        self.img_lifter.eval()
        self.smpl_model = V9Head(dim=int(cfg.MODEL.v9_dim), n_img_enc=int(cfg.MODEL.v9_img_enc),
                                 n_fuse=int(cfg.MODEL.v9_fuse_layers), n_dec=int(cfg.MODEL.v9_smpl_dec),
                                 use_hgcn=bool(cfg.MODEL.v9_hgcn),
                                 use_rel=bool(cfg.MODEL.get('v9_reliability', False)))
        self.joint_drop = float(cfg.MODEL.v9_joint_drop)
        print(f'===> FuseKin v9: dim {cfg.MODEL.v9_dim} | encoder anh {cfg.MODEL.v9_img_enc} | fuse '
              f'{cfg.MODEL.v9_fuse_layers} lop (HyperGCN {cfg.MODEL.v9_hgcn}) | query SMPL {cfg.MODEL.v9_smpl_dec} lop | '
              f'lifter {cfg.MODEL.imglift_checkpoint} (K={cfg.MODEL.imglift_K}, S={cfg.MODEL.imglift_S}) | '
              f'bo 3D: mau {cfg.MODEL.v9_drop3d_sample}, khop {cfg.MODEL.v9_drop3d_joint} | che 2D {self.joint_drop} | '
              f'reliability {cfg.MODEL.get("v9_reliability", False)} (w {cfg.MODEL.get("v9_rel_w", 0)})')

    def train(self, mode=True):
        super().train(mode)
        self.img_lifter.eval()                                  # dong bang: luon eval
        self.refiner2d.eval()
        return self

    @torch.no_grad()
    def lift(self, image, p2, jm, cam_bbox):
        """Nhu ARTS.lift_imglift. Tra pose (B, 17, 3), hyp (B, K, 17, 3) root-relative met, xy Refiner (B, 17, 2),
        log_b Refiner (B, 17)."""
        from models.img_lifter import ImgLifter, aggregate
        from utils.img_lifter_tools import refiner_inputs, lifter_det
        B = image.shape[0]
        jm = jm.reshape(B, N_JOINT).float()
        nc = int(getattr(self.img_lifter, 'num_ctx', 0))
        xy, log_b, f8, fj = refiner_inputs(self.refiner2d, image, p2[..., :2].float(), jm, N_JOINT + nc)
        cond7 = ImgLifter.make_cond7(xy, log_b, lifter_det(jm, nc), cam_bbox.float())
        gen = None
        if not self.training:                                   # test: nhieu co dinh -> lap lai duoc
            gen = torch.Generator(device=image.device)
            gen.manual_seed(0)
        hyp = self.img_lifter.sample(cond7, fj, f8, K=int(cfg.MODEL.imglift_K), S=int(cfg.MODEL.imglift_S),
                                     eta=1.0, generator=gen)
        pose = aggregate(hyp, xy, log_b, 'mean')
        return pose - pose[:, 0:1], hyp - hyp[:, :, 0:1], xy[:, :N_JOINT], log_b[:, :N_JOINT]

    def forward(self, image, joints, is_train=True, joints_mask=None, pose_2d=None, cam_bbox=None):
        """joints = 2D detector (B, 17, 2|3) [-1, 1]; joints_mask (B, 17, 1); cam_bbox (B, 3) bat buoc."""
        assert cam_bbox is not None, "v9 can inputs['cam_bbox'] (dieu kien ImgLifter)"
        B = image.shape[0]
        p2 = joints[:, 0] if joints.dim() == 4 else joints
        if joints_mask is None:
            m = (p2[..., 2:3] > 0).float() if p2.shape[-1] >= 3 else torch.ones_like(p2[..., :1])
        else:
            m = joints_mask.reshape(B, N_JOINT, 1).float()
        lift, hyp, ref_xy, ref_logb = self.lift(image, p2, m, cam_bbox)              # lifter: mask DAY DU
        if self.training and self.joint_drop > 0:                                     # che 2D (heatmap + token 2D)
            m = m * (torch.rand_like(m) >= self.joint_drop).float()
        heat = make_heatmaps(p2, m, size=image.shape[-1] // 4, sigma=float(cfg.MODEL.v3_heatmap_sigma))
        fmap, _, f16 = self.backbone(image, heatmap=heat, return_f16=True)            # (B,2048,8,8), (B,1024,16,16)
        return self.smpl_model(fmap, f16, p2[..., :2].float(), m, ref_xy.float(), ref_logb.float(), lift.float(),
                               hyp.float(), is_train=is_train)
