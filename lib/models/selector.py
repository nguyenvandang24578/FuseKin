"""
Selector: chon gia thuyet DiffLifter THEO NHOM KHOP bang anh (khong can GT luc test).

Phuong an (moi mau): pose mac dinh S1/K1 + K gia thuyet DiffLifter -> N = K + 1 ung vien.
Nhom: 4 chi (goc, giua, ngon) + than-dau [Torso, Neck, Nose, Head_top]; Pelvis luon = 0.
  - Chi: N ung vien x 4 trang thai lat do sau (doan tren / doan duoi), neu use_flip; nguoc lai N.
  - Than-dau: N ung vien.
Mang: token anh (SPIN 2048x8x8) + token khop (2D Halpe + dac trung anh lay mau tai khop) -> encoder;
token phuong an (toa do 3D cua nhom + lech so voi trung binh cac ung vien + nhom + trang thai lat) ->
decoder (self-attn giua phuong an + cross-attn toi ngu canh) -> diem. Chon argmax tung nhom, ghep pose.
Loss: cross-entropy mem voi muc tieu softmax(-loi_nhom / tau) (loi so voi GT 3D, met).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

LIMBS = ((1, 2, 3), (4, 5, 6), (11, 12, 13), (14, 15, 16))      # (goc, giua, ngon) H36M-17
TORSO = (7, 8, 9, 10)                                            # Torso, Neck, Nose, Head_top
GROUP_NAMES = ('chan P', 'chan T', 'tay T', 'tay P', 'than-dau')


def limb_states(p, limb):
    """p (B, N, 17, 3) -> 4 trang thai lat do sau cua 1 chi: (B, N, 4, 3, 3) toa do (goc, giua, ngon).
    Trang thai: (khong lat), (lat doan duoi), (lat doan tren, keo theo doan duoi), (lat ca hai)."""
    a, b, c = limb
    out = []
    for up in (False, True):
        for lo in (False, True):
            q = p[:, :, [a, b, c]].clone()
            if up:
                nb = 2 * q[..., 0, 2] - q[..., 1, 2]
                q[..., 2, 2] = q[..., 2, 2] + (nb - q[..., 1, 2])
                q[..., 1, 2] = nb
            if lo:
                q[..., 2, 2] = 2 * q[..., 1, 2] - q[..., 2, 2]
            out.append(q)
    return torch.stack(out, 2)


def build_options(cands, use_flip=True):
    """cands (B, N, 17, 3) root-relative -> list 5 nhom, moi nhom (coords (B, O, J_g, 3), joint idx).
    Chi: O = N*4 (thu tu n*4 + trang thai) neu use_flip, nguoc lai N. Than-dau: O = N."""
    B, N = cands.shape[:2]
    groups = []
    for limb in LIMBS:
        if use_flip:
            st = limb_states(cands, limb)                       # (B, N, 4, 3, 3)
            groups.append((st.reshape(B, N * 4, 3, 3), list(limb)))
        else:
            groups.append((cands[:, :, list(limb)], list(limb)))
    groups.append((cands[:, :, list(TORSO)], list(TORSO)))
    return groups


def group_errors(groups, gt, valid):
    """Loi tung phuong an tung nhom (B, O) met + mask nhom hop le (B,). gt (B,17,3), valid (B,17)."""
    out = []
    for coords, idx in groups:
        v = valid[:, idx]                                        # (B, J_g)
        e = torch.norm(coords - gt[:, None, idx], dim=-1)        # (B, O, J_g)
        n = v.sum(-1).clamp_min(1.0)
        out.append(((e * v[:, None]).sum(-1) / n[:, None], v.sum(-1) > 0))
    return out


def assemble(groups, choice, like):
    """choice: list 5 tensor (B,) chi so phuong an -> pose (B, 17, 3), Pelvis = 0."""
    pose = torch.zeros_like(like)
    ar = torch.arange(like.shape[0], device=like.device)
    for (coords, idx), c in zip(groups, choice):
        pose[:, idx] = coords[ar, c]
    return pose


class SelectorNet(nn.Module):
    def __init__(self, dim=256, heads=8, enc_layers=2, dec_layers=2, img_dim=2048, num_joint=17, use_flip=True,
                 dropout=0.1):
        super().__init__()
        self.use_flip = use_flip
        self.img_proj = nn.Conv2d(img_dim, dim, 1)
        self.img_pos = nn.Parameter(torch.zeros(1, 64, dim))
        self.joint_in = nn.Linear(3, dim)                         # (x, y, conf)
        self.joint_feat = nn.Linear(img_dim, dim)                 # dac trung anh lay mau tai vi tri 2D khop
        self.joint_emb = nn.Embedding(num_joint, dim)
        enc = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers)
        # Phuong an: toa do nhom (toi da 4 khop x 3) + lech so voi trung binh ung vien (4 x 3), x10 (dm)
        self.opt_in = nn.Sequential(nn.Linear(24, dim), nn.GELU(), nn.Linear(dim, dim))
        self.group_emb = nn.Embedding(5, dim)
        self.flip_emb = nn.Embedding(4, dim)
        dec = nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def context(self, feat_map, pose_2d, joints_mask):
        """feat_map (B,2048,h,w), pose_2d (B,17,2|3) [-1,1], joints_mask (B,17,1) conf -> (B, h*w + 17, dim)."""
        B = feat_map.shape[0]
        img = self.img_proj(feat_map)                              # (B, dim, h, w)
        img_tok = img.flatten(2).transpose(1, 2)
        if img_tok.shape[1] == self.img_pos.shape[1]:
            img_tok = img_tok + self.img_pos
        xy = pose_2d[..., :2].float()
        conf = joints_mask.reshape(B, -1, 1).float()
        samp = F.grid_sample(feat_map, xy[:, :, None, :], align_corners=False)   # (B, 2048, 17, 1)
        samp = samp[..., 0].transpose(1, 2) * (conf > 0)
        jt = self.joint_in(torch.cat([xy, conf], -1)) + self.joint_feat(samp) + self.joint_emb.weight[None]
        return self.encoder(torch.cat([img_tok, jt], 1))

    def option_tokens(self, groups):
        toks = []
        for g, (coords, idx) in enumerate(groups):
            B, O, J, _ = coords.shape
            dev = coords - coords.mean(1, keepdim=True)
            x = torch.cat([coords, dev], -1) * 10.0                # (B, O, J, 6)
            x = F.pad(x.reshape(B, O, J * 6), (0, 24 - J * 6))
            t = self.opt_in(x) + self.group_emb.weight[g]
            if g < 4 and self.use_flip:
                state = torch.arange(O, device=coords.device) % 4
                t = t + self.flip_emb(state)[None]
            toks.append(t)
        return toks

    def forward(self, feat_map, pose_2d, joints_mask, groups):
        """Tra list 5 logits (B, O_g)."""
        ctx = self.context(feat_map, pose_2d, joints_mask)
        toks = self.option_tokens(groups)
        sizes = [t.shape[1] for t in toks]
        out = self.decoder(torch.cat(toks, 1), ctx)
        logits = self.head(out)[..., 0]
        return list(torch.split(logits, sizes, dim=1))


def soft_ce(logits, err, valid, tau):
    """Cross-entropy mem: muc tieu softmax(-err / tau). Tra loss trung binh tren mau hop le."""
    target = torch.softmax(-err / tau, dim=-1)
    ce = -(target * torch.log_softmax(logits, dim=-1)).sum(-1)
    v = valid.float()
    return (ce * v).sum() / v.sum().clamp_min(1.0)
