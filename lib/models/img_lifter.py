"""
lib/models/img_lifter.py -- ImgLifter (2026-11-04): diffusion 2D -> 3D CO ANH, thay DiffLifter cu.

Dau vao (moi nguoi, moi anh) lay tu Refiner 2D (dong bang, lib/models/refiner2d.py):
  xy (B,17,2) 2D da sua, dinh nghia mesh, [-1,1] | log_b (B,17) do bat dinh (o luoi 64) | det (B,17) detector co khop
  cam_bbox (B,3) | f8 (B,2048,8,8) dac trung anh toan cuc | feat_j (B,17,C) dac trung anh tai tung khop
Dieu kien tung khop c_j = Linear([x, y, log_b, det, cam_bbox]) + Linear(feat_j).
  [1] cong vao token khop o dau vao; [2] adaLN-Zero (shift / scale / gate tu c_j + t) o MOI block;
  [3] cross-attention tu 17 token khop toi 64 token anh (f8) o MOI block.
Diffusion giong DiffLifter cu: cosine T=1000, du doan x0, scale pose x3, DDIM eta=1; loss MPJPE (L2) tren x0.
Suy luan: K gia thuyet x S buoc DDIM, gop 'mean' hoac 'jpma' (moi khop chon gia thuyet chieu 2D khop xy nhat;
khop co b > jpma_max_b giu trung binh). Output: (B,17,3) met, root-relative tai Pelvis, dinh nghia khop mesh.

Tuy chon (2026-11-05, y tuong Markov Optimization Layer cua DiffusionPose AAAI-26, mac dinh TAT):
  markov_layer: moi block them 1 lop truyen tin giua 17 token khop THEO XUONG (16 canh H36M), sau self-attn.
    msg_{j->i} = psi(h_i) * phi(h_i, h_j) (psi = h_i W_n, phi = [h_i; h_j] W_e, nhan tung phan tu), gop logsumexp
    tren cac khop ke j, chieu ve D (khoi tao 0 -> luc dau la identity), cong residual.
    markov_iters = 0: giu ban cu bo tich thong diep. > 0: cap nhat dong bo tren 32 canh co huong,
    msg_{j->i}^{r} = psi(h_i) * phi(h_i,h_j) * prod_{k ke j, k != i} msg_{k->j}^{r-1}, khoi tao msg^0 = 1.
    markov_msg_norm (2026-11-05): chuan hoa RMS moi vong (dau "ty le voi" cua cong thuc (1)) -> tich nhieu vong
    khong bung no / triet tieu (dac trung co dau, khong chan). Chi dung khi markov_iters > 0.
  markov_gate_b: phan tin cong vao nhan cong g = sigmoid(a * (log_b - c)) (a, c hoc duoc) -> khop mo ho (b lon,
    bi che / detector sot). markov_gate_positive: a = softplus(a_raw) + 1e-6 de cong luon tang theo log_b.
  loss: w_bone x do dai xuong + w_angle x (cos goc 2 xuong noi tiep: du doan - GT)^2, tren x0 du doan.
    bone_paper = False: L1 (met) nhu cu. True: -log(mean(exp(-sai_so^2 / (2 * var_GT + 1e-8)))),
    var_GT tren 16 xuong cua TUNG pose, unbiased=False; loss nay khong co don vi. Giu 13 cap goc nhu cu.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diff_lifter import cosine_beta_schedule, extract, Mlp, SinusoidalPositionEmbeddings

try:
    from timm.models.layers import DropPath, trunc_normal_
except ImportError:
    from timm.layers import DropPath, trunc_normal_

COND_DIM = 7     # [x, y, log_b, det, cam_bbox(3)]

# Xuong H36M-17 (cha, con), goc tai Pelvis. Thu tu khop: Pelvis, R_Hip, R_Knee, R_Ankle, L_Hip, L_Knee, L_Ankle,
# Torso, Neck, Nose, Head_top, L_Shoulder, L_Elbow, L_Wrist, R_Shoulder, R_Elbow, R_Wrist
H36M_BONES = ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
              (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16))
# Cap xuong noi tiep (xuong a ket thuc tai khop ma xuong b bat dau) -> rang buoc goc. 13 cap.
H36M_BONE_PAIRS = tuple((a, b) for a, (_, ca) in enumerate(H36M_BONES)
                        for b, (pb, _) in enumerate(H36M_BONES) if pb == ca)


def h36m_adjacency(num_joints=17):
    adj = torch.zeros(num_joints, num_joints, dtype=torch.bool)
    for p, c in H36M_BONES:
        adj[p, c] = adj[c, p] = True
    return adj


def bone_angle_losses(pred, gt, paper=False):
    """Tra bone, angle (B,). Bone L1 (met) hoac cong thuc (4) (khong don vi); angle la sai cos^2."""
    pi = torch.tensor([p for p, _ in H36M_BONES], device=pred.device)
    ci = torch.tensor([c for _, c in H36M_BONES], device=pred.device)
    vp, vg = pred[:, ci] - pred[:, pi], gt[:, ci] - gt[:, pi]               # (B,16,3)
    lp, lg = vp.norm(dim=-1), vg.norm(dim=-1)
    if paper:
        # Chon L_std rieng tung pose, tren 16 xuong; khong phu thuoc mau khac trong batch.
        var_gt = lg.var(dim=-1, unbiased=False, keepdim=True)
        score = -(lp - lg).square() / (2.0 * var_gt + 1e-8)
        # -log(mean(exp(score))) tinh on dinh, tranh exp bi hut ve 0.
        bone = math.log(len(H36M_BONES)) - torch.logsumexp(score, dim=-1)
    else:
        bone = (lp - lg).abs().mean(-1)
    a = torch.tensor([x for x, _ in H36M_BONE_PAIRS], device=pred.device)
    b = torch.tensor([y for _, y in H36M_BONE_PAIRS], device=pred.device)
    up, ug = vp / lp.clamp_min(1e-6)[..., None], vg / lg.clamp_min(1e-6)[..., None]
    cos_p, cos_g = (up[:, a] * up[:, b]).sum(-1), (ug[:, a] * ug[:, b]).sum(-1)
    angle = ((cos_p - cos_g) ** 2).mean(-1)
    return bone, angle


class MarkovLayer(nn.Module):
    """Truyen tin theo xuong giua 17 token khop (xem docstring dau file). h (N,17,D) da norm; log_b (N,17) / None."""

    def __init__(self, dim, num_joints=17, msg_dim=256, gate_b=False, iters=0, gate_positive=False,
                 msg_norm=False):
        super().__init__()
        if msg_norm and not int(iters):
            raise ValueError('markov_msg_norm can markov_iters > 0')
        self.msg_norm = bool(msg_norm)
        if num_joints != 17 or int(iters) < 0:
            raise ValueError('MarkovLayer can H36M-17 va markov_iters >= 0')
        if gate_positive and not gate_b:
            raise ValueError('markov_gate_positive can markov_gate_b')
        self.iters = int(iters)
        self.gate_positive = bool(gate_positive)
        self.register_buffer('adj', h36m_adjacency(num_joints), persistent=False)
        if self.iters:
            edges = [(p, c) for p, c in H36M_BONES] + [(c, p) for p, c in H36M_BONES]
            src, dst = zip(*edges)
            incoming = [[k for k, (u, v) in enumerate(edges) if v == j and u != i] for j, i in edges]
            node_incoming = [[k for k, (_, v) in enumerate(edges) if v == i] for i in range(num_joints)]
            # Dem canh thieu bang chi so 0 va mask; tich rong = 1, khong dung phep chia thong diep.
            for name, rows in (('incoming', incoming), ('node_incoming', node_incoming)):
                width = max(len(row) for row in rows)
                indices = torch.tensor([row + [0] * (width - len(row)) for row in rows], dtype=torch.long)
                mask = torch.tensor([[True] * len(row) + [False] * (width - len(row)) for row in rows])
                self.register_buffer(name, indices, persistent=False)
                self.register_buffer(name + '_mask', mask, persistent=False)
            self.register_buffer('edge_src', torch.tensor(src, dtype=torch.long), persistent=False)
            self.register_buffer('edge_dst', torch.tensor(dst, dtype=torch.long), persistent=False)
        self.psi = nn.Linear(dim, msg_dim)                                  # node potential: h_i W_n
        self.phi_i = nn.Linear(dim, msg_dim)                                # edge potential [h_i; h_j] W_e
        self.phi_j = nn.Linear(dim, msg_dim, bias=False)                    #   = h_i W_e1 + h_j W_e2
        self.out = nn.Linear(msg_dim, dim)
        nn.init.zeros_(self.out.weight)                                     # luc dau = identity
        nn.init.zeros_(self.out.bias)
        self.gate_b = bool(gate_b)
        if self.gate_b:
            init_a = math.log(math.expm1(1.0)) if self.gate_positive else 1.0
            self.gate_a = nn.Parameter(torch.tensor(init_a))
            self.gate_c = nn.Parameter(torch.tensor(0.0))

    def _norm(self, m):
        # Chuan hoa RMS tren chieu dac trung (giu dau), moi canh rieng; tat -> giu nguyen nhu ban v2
        if not self.msg_norm:
            return m
        return m / (m.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()

    def forward(self, h, log_b=None):
        psi = self.psi(h)                                                   # (N,J,M)
        if self.iters:
            # Canh e = j -> i; node potential van dung khop NHAN i theo cong thuc (1).
            phi = self.phi_i(h)[:, self.edge_dst] + self.phi_j(h)[:, self.edge_src]
            base = psi[:, self.edge_dst] * phi                              # (N,32,M)
            msg = self._norm(base)                                          # vong 1: cac thong diep truoc = 1
            for _ in range(1, self.iters):
                previous = msg[:, self.incoming]                            # (N,32,max_ke,M)
                previous = previous.masked_fill(~self.incoming_mask[None, :, :, None], 1.0)
                msg = self._norm(base * previous.prod(dim=2))               # chi k -> j voi k != i
            received = msg[:, self.node_incoming]                           # (N,17,max_bac,M)
            received = received.masked_fill(~self.node_incoming_mask[None, :, :, None], float('-inf'))
            agg = torch.logsumexp(received, dim=2)
        else:
            # Giu nguyen duong tinh cu cho YAML va checkpoint chua bat markov_iters.
            phi = self.phi_i(h)[:, :, None] + self.phi_j(h)[:, None, :]      # (N,J,J,M): [i, j]
            msg = psi[:, :, None] * phi
            msg = msg.masked_fill(~self.adj[None, :, :, None], float('-inf'))
            agg = torch.logsumexp(msg, dim=2)
        out = self.out(agg)
        if self.gate_b:
            assert log_b is not None, 'markov_gate_b can log_b'
            a = F.softplus(self.gate_a) + 1e-6 if self.gate_positive else self.gate_a
            out = out * torch.sigmoid(a * (log_b - self.gate_c))[..., None]
        return out


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ImgLifterBlock(nn.Module):
    """adaLN-Zero (theo tung khop) -> self-attn khop | cross-attn anh | FFN."""

    def __init__(self, dim, num_heads, mlp_ratio=2.0, drop_path=0.0, markov=False, markov_gate_b=False,
                 num_joints=17, markov_iters=0, markov_gate_positive=False, markov_msg_norm=False):
        super().__init__()
        self.num_joints = num_joints
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        if markov:                                                          # sau self-attn, chi 17 token khop
            self.norm_m = nn.LayerNorm(dim, eps=1e-6)
            self.markov = MarkovLayer(dim, num_joints, gate_b=markov_gate_b, iters=markov_iters,
                                      gate_positive=markov_gate_positive, msg_norm=markov_msg_norm)
        else:
            self.markov = None
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.ada[-1].weight)      # adaLN-Zero: luc dau moi block = identity
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, cond, img_tok, log_b=None):
        """x (N,P,D) token (17 khop + ngu canh); cond (N,P,D) dieu kien tung token (+ t); img_tok (N,64,D);
        log_b (N,17) chi can khi markov_gate_b."""
        s1, c1, g1, s2, c2, g2, s3, c3, g3 = self.ada(cond).chunk(9, dim=-1)
        h = modulate(self.norm1(x), s1, c1)
        x = x + self.drop_path(g1 * self.self_attn(h, h, h, need_weights=False)[0])
        if self.markov is not None:
            J = self.num_joints
            xj = x[:, :J] + self.drop_path(self.markov(self.norm_m(x[:, :J]), log_b))
            x = torch.cat([xj, x[:, J:]], dim=1)
        h = modulate(self.norm2(x), s2, c2)
        x = x + self.drop_path(g2 * self.cross_attn(h, img_tok, img_tok, need_weights=False)[0])
        h = modulate(self.norm3(x), s3, c3)
        x = x + self.drop_path(g3 * self.mlp(h))
        return x


class ImgLifterDenoiser(nn.Module):
    def __init__(self, num_joints=17, dim=512, depth=8, num_heads=8, mlp_ratio=2.0, drop_path=0.1,
                 img_ch=2048, feat_j_ch=256, img_tokens=64, num_ctx=0, markov=False, markov_gate_b=False,
                 markov_iters=0, markov_gate_positive=False, markov_msg_norm=False):
        """num_ctx > 0 (ImgLifter day, 2026-11-04): them num_ctx token NGU CANH 2D (diem day cua Refiner day, sau 17
        khop) vao chuoi self-attn: chi mang dieu kien [x, y, log_b, det=0, cam] + dac trung anh, KHONG co x_t va
        KHONG co dau ra 3D. num_ctx = 0 -> y het ban 17 khop (cung shape tham so)."""
        super().__init__()
        self.num_joints = num_joints
        self.num_ctx = int(num_ctx)
        self.x_embed = nn.Linear(3, dim)
        self.cond_embed = nn.Sequential(nn.Linear(COND_DIM, dim), nn.GELU(), nn.Linear(dim, dim))
        self.featj_embed = nn.Sequential(nn.LayerNorm(feat_j_ch), nn.Linear(feat_j_ch, dim))
        self.joint_pos = nn.Parameter(torch.zeros(1, num_joints + self.num_ctx, dim))   # vi tri khop (+ diem day)
        self.img_embed = nn.Sequential(nn.LayerNorm(img_ch), nn.Linear(img_ch, dim))
        self.img_pos = nn.Parameter(torch.zeros(1, img_tokens, dim))
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(dim), nn.Linear(dim, dim * 2), nn.GELU(),
                                      nn.Linear(dim * 2, dim))
        dpr = [v.item() for v in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([ImgLifterBlock(dim, num_heads, mlp_ratio, dpr[i], markov, markov_gate_b,
                                                    num_joints, markov_iters, markov_gate_positive,
                                                    markov_msg_norm)
                                     for i in range(depth)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.head = nn.Linear(dim, 3)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        trunc_normal_(self.joint_pos, std=0.02)
        trunc_normal_(self.img_pos, std=0.02)

    def encode_cond(self, cond7, feat_j, f8):
        """Tinh 1 lan cho moi anh (dung lai o moi buoc DDIM / moi gia thuyet).
        cond7 (B,P,7), feat_j (B,P,C) voi P = 17 (+ num_ctx), f8 (B,2048,8,8) -> c_j (B,P,D), img_tok (B,64,D)."""
        P = self.num_joints + self.num_ctx
        assert cond7.shape[1] == P and feat_j.shape[1] == P, f'can {P} diem dieu kien, nhan {cond7.shape[1]}'
        c_j = self.cond_embed(cond7) + self.featj_embed(feat_j)
        img = f8.flatten(2).transpose(1, 2)                                 # (B, 64, 2048)
        img_tok = self.img_embed(img) + self.img_pos
        return c_j, img_tok

    def forward(self, x_t, t, c_j, img_tok, log_b=None):
        """x_t (N,17,3) da chia scale; t (N,); c_j (N,P,D); img_tok (N,64,D); log_b (N,17) -> x0 (N,17,3) met."""
        J = self.num_joints
        x = self.x_embed(x_t) + c_j[:, :J] + self.joint_pos[:, :J]          # [1] dieu kien o dau vao
        if self.num_ctx:                                                    # token ngu canh: chi dieu kien 2D / anh
            x = torch.cat([x, c_j[:, J:] + self.joint_pos[:, J:]], dim=1)
        cond = c_j + self.time_mlp(t)[:, None, :]                           # [2] adaLN: dieu kien tung token + t
        for blk in self.blocks:
            x = blk(x, cond, img_tok, log_b)                                # [3] cross-attn anh trong blk
        x, cond = x[:, :J], cond[:, :J]                                     # chi 17 khop co dau ra 3D
        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        return self.head(modulate(self.final_norm(x), shift, scale))


class ImgLifter(nn.Module):
    def __init__(self, num_joints=17, dim=512, depth=8, num_heads=8, mlp_ratio=2.0, drop_path=0.1,
                 timesteps=1000, scale=3.0, feat_j_ch=256, num_ctx=0, markov=False, markov_gate_b=False,
                 markov_iters=0, markov_gate_positive=False, bone_paper=False, markov_msg_norm=False):
        super().__init__()
        if int(markov_iters) < 0 or (markov_iters and not markov):
            raise ValueError('markov_iters > 0 can markov_layer=True; khong dung so am')
        if markov_gate_positive and not (markov and markov_gate_b):
            raise ValueError('markov_gate_positive can markov_layer va markov_gate_b')
        self.num_joints = num_joints
        self.num_ctx = int(num_ctx)
        self.scale = float(scale)
        self.bone_paper = bool(bone_paper)
        self.denoiser = ImgLifterDenoiser(num_joints, dim, depth, num_heads, mlp_ratio, drop_path,
                                          feat_j_ch=feat_j_ch, num_ctx=num_ctx, markov=markov,
                                          markov_gate_b=markov_gate_b, markov_iters=markov_iters,
                                          markov_gate_positive=markov_gate_positive,
                                          markov_msg_norm=markov_msg_norm)
        betas = cosine_beta_schedule(timesteps)
        ac = torch.cumprod(1. - betas, dim=0)
        self.num_timesteps = int(betas.shape[0])
        self.register_buffer('alphas_cumprod', ac.to(torch.float32))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(ac).to(torch.float32))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - ac).to(torch.float32))

    @staticmethod
    def make_cond7(xy, log_b, det, cam_bbox):
        """(B,17,2), (B,17), (B,17), (B,3) -> (B,17,7)."""
        J = xy.shape[1]
        return torch.cat([xy, log_b.unsqueeze(-1), det.float().unsqueeze(-1),
                          cam_bbox[:, None, :].expand(-1, J, -1)], dim=-1)

    def _x0(self, x_t_scaled, t, c_j, img_tok, log_b=None):
        p = self.denoiser(x_t_scaled, t, c_j, img_tok, log_b)
        return p - p[:, 0:1, :]                                             # root-relative, met

    def loss(self, pose_3d, cond7, feat_j, f8, valid=None, w_bone=0.0, w_angle=0.0):
        """pose_3d (B,17,3) GT met root-relative. valid (B,) 0/1. Tra (loss tong, pred_x0).
        Tung thanh phan (MPJPE / bone / angle, chua nhan trong so) luu o self.last_terms de log."""
        B = pose_3d.shape[0]
        x0 = ((pose_3d - pose_3d[:, 0:1]) * self.scale).clamp(-1.1 * self.scale, 1.1 * self.scale)
        t = torch.randint(0, self.num_timesteps, (B,), device=pose_3d.device).long()
        noise = torch.randn_like(x0)
        x_t = extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0 + \
            extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        c_j, img_tok = self.denoiser.encode_cond(cond7, feat_j, f8)
        log_b = cond7[:, :self.num_joints, 2]
        pred = self._x0(x_t.clamp(-1.1 * self.scale, 1.1 * self.scale) / self.scale, t, c_j, img_tok, log_b)
        gt = pose_3d - pose_3d[:, 0:1]
        v = torch.ones(B, device=pose_3d.device) if valid is None else valid.float()
        vs = v.sum().clamp_min(1.0)
        err = torch.norm(pred - gt, dim=-1).mean(-1)                        # (B,)
        terms = {'mpjpe': (err * v).sum() / vs}
        total = terms['mpjpe']
        if w_bone > 0 or w_angle > 0:
            bone, angle = bone_angle_losses(pred, gt, paper=self.bone_paper)
            terms['bone'], terms['angle'] = (bone * v).sum() / vs, (angle * v).sum() / vs
            total = total + float(w_bone) * terms['bone'] + float(w_angle) * terms['angle']
        self.last_terms = {k: float(x.detach()) for k, x in terms.items()}
        self.last_terms['total'] = float(total.detach())
        return total, pred

    @torch.no_grad()
    def sample(self, cond7, feat_j, f8, K=10, S=4, eta=1.0, generator=None):
        """DDIM, K gia thuyet song song. Tra all_preds (B,K,17,3) met root-relative."""
        B, J = cond7.shape[0], self.num_joints                              # cond7 co the co 17 + num_ctx diem
        dev = cond7.device
        c_j, img_tok = self.denoiser.encode_cond(cond7, feat_j, f8)
        c_j = c_j.repeat_interleave(K, dim=0)                               # (B*K,P,D)
        img_tok = img_tok.repeat_interleave(K, dim=0)
        log_b = cond7[:, :J, 2].repeat_interleave(K, dim=0)                 # (B*K,17) cho cong Markov
        T = self.num_timesteps
        times = list(reversed(torch.linspace(-1, T - 1, steps=S + 1).int().tolist()))
        x = torch.randn(B * K, J, 3, device=dev, generator=generator)
        sc = self.scale
        for time, time_next in zip(times[:-1], times[1:]):
            t = torch.full((B * K,), time, device=dev, dtype=torch.long)
            x0 = (self._x0(x.clamp(-1.1 * sc, 1.1 * sc) / sc, t, c_j, img_tok, log_b) * sc).clamp(-1.1 * sc, 1.1 * sc)
            if time_next < 0:
                x = x0
                break
            a, an = self.alphas_cumprod[time], self.alphas_cumprod[time_next]
            sigma = eta * ((1 - a / an) * (1 - an) / (1 - a)).clamp(min=0).sqrt()
            c = (1 - an - sigma ** 2).clamp(min=0).sqrt()
            eps = (x - x0 * a.sqrt()) / (1 - a + 1e-8).sqrt()
            noise = torch.randn(x.shape, device=dev, dtype=x.dtype, generator=generator)
            x = x0 * an.sqrt() + c * eps + sigma * noise
        return (x / sc).reshape(B, K, J, 3)


def aggregate(all_preds, xy, log_b, mode='mean', jpma_max_b=3.0):
    """all_preds (B,K,17,3); xy (B,17,2) [-1,1] (2D Refiner); log_b (B,17).
    'mean': trung binh K. 'jpma': fit weak-perspective (s, t) tu XY cua pose trung binh sang xy (trong so 1 / b^2),
    moi khop co b <= jpma_max_b chon gia thuyet co loi chieu 2D nho nhat; con lai giu trung binh."""
    J0 = all_preds.shape[2]
    xy, log_b = xy[:, :J0], log_b[:, :J0]                                   # bo diem day (neu co), chi 17 khop
    mean = all_preds.mean(1)
    if mode == 'mean' or all_preds.shape[1] == 1:
        return mean - mean[:, 0:1]
    B, K, J, _ = all_preds.shape
    b = torch.exp(log_b)
    w = 1.0 / b.clamp_min(0.25) ** 2
    P = mean[..., :2]
    ws = w.sum(1, keepdim=True)[..., None]
    Pm = (w[..., None] * P).sum(1, keepdim=True) / ws
    pm = (w[..., None] * xy).sum(1, keepdim=True) / ws
    Pc, pc = P - Pm, xy - pm
    s = ((w * (Pc * pc).sum(-1)).sum(1) / (w * (Pc ** 2).sum(-1)).sum(1).clamp_min(1e-8))[:, None, None]
    t = pm - s * Pm
    proj = s[:, None] * all_preds[..., :2] + t[:, None]                    # (B,K,17,2)
    err = torch.norm(proj - xy[:, None], dim=-1)                            # (B,K,17)
    best = err.argmin(1)
    sel = torch.gather(all_preds, 1, best[:, None, :, None].expand(B, 1, J, 3)).squeeze(1)
    out = torch.where((b <= jpma_max_b)[..., None], sel, mean)
    return out - out[:, 0:1]
