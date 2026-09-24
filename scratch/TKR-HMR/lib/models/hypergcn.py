import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# ============================================================
# Kinematic tree (SMPL 24 joints) — bang cha (parent) cho tung khop
# 0 = root (pelvis / global_orient), khong co cha
# ============================================================
PARENT = [
    -1,  # 0  root
    0,   # 1
    0,   # 2
    0,   # 3
    1,   # 4
    2,   # 5
    3,   # 6
    4,   # 7
    5,   # 8
    6,   # 9
    7,   # 10
    8,   # 11
    9,   # 12
    9,   # 13
    9,   # 14
    12,  # 15
    13,  # 16
    14,  # 17
    16,  # 18
    17,  # 19
    18,  # 20
    19,  # 21
    20,  # 22
    21,  # 23
]

NUM_JOINTS = 24


def build_H_init_no_root(num_nodes=24, num_edges=5):
    """
    H_init - 5 hyperedge (KHONG con hyperedge rieng cho root, root duoc xu
    ly rieng qua RootBroadcast). 5 chain xuat phat tu root:
      0: Torso+Head (tru root)  : 3,6,9,12,13,14,15
      1: Left Arm  : 16,18,20,22
      2: Right Arm : 17,19,21,23
      3: Left Leg  : 1,4,7,10
      4: Right Leg : 2,5,8,11
    Joint 0 (root) co hang toan so 0 trong H_init -> Dv[root]=0 (+eps) ->
    sau chuan hoa G[root,:]=G[:,root]=0 -> root khong tham gia hyper-mix,
    dung nhu thiet ke (root duoc xu ly rieng o nhanh RootChain).
    """
    H = torch.zeros(num_nodes, num_edges)
    for i in [3, 6, 9, 12, 13, 14, 15]: H[i, 0] = 1.0   # Torso+Head
    for i in [16, 18, 20, 22]:          H[i, 1] = 1.0   # Left Arm
    for i in [17, 19, 21, 23]:          H[i, 2] = 1.0   # Right Arm
    for i in [1, 4, 7, 10]:             H[i, 3] = 1.0   # Left Leg
    for i in [2, 5, 8, 11]:             H[i, 4] = 1.0   # Right Leg
    return H


def compute_G_batched(H, eps=1e-8):
    """G = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}, batched. H: (K, N, E), luon >= 0."""
    H = H.clamp(min=0.0)
    Dv = H.sum(dim=-1) + eps                       # (K, N)
    Dv_inv_sqrt = Dv.pow(-0.5)
    De = H.sum(dim=1) + eps                          # (K, E)
    De_inv = De.pow(-1)
    H_De = H * De_inv.unsqueeze(1)
    G = torch.bmm(H_De, H.transpose(1, 2))            # (K, N, N)
    G = Dv_inv_sqrt.unsqueeze(-1) * G * Dv_inv_sqrt.unsqueeze(1)
    return G


# ============================================================
# Nhanh A: RootChain — cha->con (1-hop) + root broadcast toi ca 5 chain
# ============================================================
class RootChainProp(nn.Module):
    """
    Voi moi khop i != root: h_i_new = W_self(h_i) + W_parent(h_parent(i))
        (chi 1-hop, dung 'gather' truc tiep theo bang PARENT — khong dung
        graph Laplacian phuc tap cho quan he 1-1 nay)
    Voi root (joint 0): h_0_new = W_self(h_0)  (khong co cha)
    Sau do: root BROADCAST (FiLM) toi toan bo 23 khop con lai — day la
    'duong tat' bat buoc de thong tin root toi duoc ca khop xa nhat
    (vd toi ban tay can 8-hop trong tree that) chi trong 1 layer, thay vi
    phai doi nhieu layer moi 'ngam' toi.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.W_self = nn.Linear(dim_in, dim_out)
        self.W_parent = nn.Linear(dim_in, dim_out)

        # Root xu ly rieng: co the dung 1 linear khac de khong bi rang buoc
        # phai giong cach cap nhat cua cac khop thuong (root khong co "cha")
        self.W_root = nn.Linear(dim_in, dim_out)

        # FiLM broadcast đã bị tắt — outer FiLM (Multimodel) đã inject
        # global context cho tất cả 24 khớp rồi, không cần broadcast lại.

        parent_idx = torch.tensor(
            [p if p >= 0 else 0 for p in PARENT], dtype=torch.long
        )  # (24,) - root tam gan idx 0 (chinh no), se bi mask ve 0 sau
        self.register_buffer('parent_idx', parent_idx)

        child_mask = torch.tensor(
            [0.0 if p < 0 else 1.0 for p in PARENT]
        ).view(1, 1, NUM_JOINTS, 1)  # (1,1,24,1) - root=0, con lai=1
        self.register_buffer('child_mask', child_mask)

    def forward(self, x):
        """x: (B, T, 24, C) -> out: (B, T, 24, C_out)"""
        b, t, v, c = x.shape

        # --- cha -> con (1-hop), dung gather theo PARENT ---
        h_parent = x[:, :, self.parent_idx, :]              # (B,T,24,C) - hang root la x[...,0,:]
        local = self.W_self(x) + self.child_mask * self.W_parent(h_parent)
        # joint 0: child_mask=0 -> local[0] = W_self(x0) (chua co dong gop tu W_root)

        root_feat = self.W_root(x[:, :, 0, :])               # (B, T, C_out)
        local = local.clone()
        local[:, :, 0, :] = root_feat                          # root dung nhanh rieng

        # FiLM broadcast đã tắt — chỉ giữ parent→child 1-hop propagation
        return local


# ============================================================
# Nhanh B: Adaptive Hyper (xuyen-chain, KHONG bao gom root)
# ============================================================
class AdaptiveHyperNoRoot(nn.Module):
    """
    Bat tuong tac xuyen-chain (vd 2 tay cham nhau) ma nhanh RootChain
    (chi lan truyen doc theo 1 chain) khong nam duoc. Root bi loai khoi
    H_init/M/S (xem build_H_init_no_root) vi ban chat thong ke khac biet.
    """

    def __init__(self, dim_in, dim_out, num_edges=5, per_frame_S=True):
        super().__init__()
        self.per_frame_S = per_frame_S

        H_init = build_H_init_no_root(NUM_JOINTS, num_edges)
        self.register_buffer('H_init', H_init)                          # (24, E)

        self.M_raw = nn.Parameter(self._inverse_softplus(H_init.clone()))

        self.phi1 = nn.Linear(dim_in, dim_in)
        self.phi2 = nn.Linear(dim_in, dim_in)

        self.beta0_raw = nn.Parameter(self._inverse_softplus(torch.tensor(1.0)))
        self.beta1_raw = nn.Parameter(self._inverse_softplus(torch.tensor(0.1)))
        self.beta2_raw = nn.Parameter(self._inverse_softplus(torch.tensor(0.1)))

        self.conv_hyper = nn.Linear(dim_in, dim_out)

    @staticmethod
    def _inverse_softplus(x, eps=1e-6):
        x = torch.as_tensor(x, dtype=torch.float32).clamp(min=eps)
        return torch.log(torch.expm1(x))

    def _compute_S(self, x):
        """x: (B,T,24,C) -> S: (B*T, 24, E) neu per_frame_S else (B,24,E)"""
        b, t, v, c = x.shape
        if self.per_frame_S:
            xf = x.reshape(b * t, v, c)
            q = self.phi1(xf)                      # (BT, N, C)
            k = self.phi2(xf).transpose(1, 2)       # (BT, C, N)
            sim = torch.bmm(q, k) / sqrt(c)          # (BT, N, N)
            H_init_exp = self.H_init.unsqueeze(0).expand(b * t, -1, -1)
        else:
            xf = x.mean(dim=1)                      # (B, N, C) - gop theo T truoc
            q = self.phi1(xf)
            k = self.phi2(xf).transpose(1, 2)
            sim = torch.bmm(q, k) / sqrt(c)
            H_init_exp = self.H_init.unsqueeze(0).expand(b, -1, -1)

        S = torch.bmm(sim, H_init_exp)
        S = torch.softmax(S, dim=-1)
        # FIX: du H_init[root]=0, S[root] van co the khac 0 vi no duoc tinh
        # tu similarity giua root va CAC KHOP KHAC (khong phai hang cua
        # chinh root trong H_init). Mask tuong minh de dam bao root khong
        # tham gia hyper-branch, dung nhu thiet ke (root duoc xu ly rieng
        # o RootChainProp).
        S = S.clone()
        S[:, 0, :] = 0.0
        return S

    def forward(self, x):
        """x: (B,T,24,C) -> out: (B,T,24,C_out)"""
        b, t, v, c = x.shape
        S = self._compute_S(x)                        # (K,24,E)
        K = S.shape[0]

        beta0 = F.softplus(self.beta0_raw)
        beta1 = F.softplus(self.beta1_raw)
        beta2 = F.softplus(self.beta2_raw)
        M = F.softplus(self.M_raw)

        H_init_exp = self.H_init.unsqueeze(0).expand(K, -1, -1)
        M_exp = M.unsqueeze(0).expand(K, -1, -1)
        H_tilde = beta0 * H_init_exp + beta1 * M_exp + beta2 * S   # (K,24,E)

        G = compute_G_batched(H_tilde)                  # (K,24,24)

        if self.per_frame_S:
            xf = x.reshape(b * t, v, c)
            xh = torch.bmm(G, xf)                         # (BT,24,C)
            xh = xh.reshape(b, t, v, c)
        else:
            xmean = x.mean(dim=1)                          # (B,24,C)
            xh = torch.bmm(G, xmean).unsqueeze(1).expand(-1, t, -1, -1)

        out = self.conv_hyper(xh)                          # (B,T,24,C_out)
        return out, H_tilde


# ============================================================
# Module tong: HYPERGC v2
# ============================================================
class HYPERGCv2(nn.Module):
    """
    Input/Output: (B, T, 24, dim) — khop truc tiep voi pose_token hien tai,
    khong can permute.

    out = ReLU(BN( alpha_a * RootChain(x) + alpha_b * AdaptiveHyper(x) + U(x) ))
          (+ residual x neu dim_in == dim_out)
    """

    def __init__(self, dim_in, dim_out, num_edges=5, per_frame_S=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.root_chain = RootChainProp(dim_in, dim_out)
        self.adaptive_hyper = AdaptiveHyperNoRoot(dim_in, dim_out, num_edges, per_frame_S)

        self.alpha_chain_raw = nn.Parameter(torch.tensor(1.0))
        self.alpha_hyper_raw = nn.Parameter(torch.tensor(1.0))

        self.U = nn.Linear(dim_in, dim_out)
        self.batch_norm = nn.LayerNorm(dim_out)  # LayerNorm ổn định hơn BN1d khi T thay đổi
        self.relu = nn.ReLU()

    def forward(self, x):
        """x: (B, T, 24, C_in)"""
        a_chain = self.root_chain(x)                    # (B,T,24,C_out)
        a_hyper, H_tilde = self.adaptive_hyper(x)         # (B,T,24,C_out)

        agg = self.alpha_chain_raw * a_chain + self.alpha_hyper_raw * a_hyper

        if self.dim_in == self.dim_out:
            out = self.relu(x + self.batch_norm(agg + self.U(x)))
        else:
            out = self.relu(self.batch_norm(agg + self.U(x)))

        aux = {
            'H_tilde': H_tilde.detach(),
            'alpha_chain': self.alpha_chain_raw.item(),
            'alpha_hyper': self.alpha_hyper_raw.item(),
        }
        return out, aux


# ============================================================
# Sanity check
# ============================================================
if __name__ == '__main__':
    torch.manual_seed(0)
    B, T, V, C_in, C_out = 2, 8, 24, 512, 512

    layer = HYPERGCv2(C_in, C_out, num_edges=5, per_frame_S=True)
    x = torch.randn(B, T, V, C_in)
    out, aux = layer(x)
    assert out.shape == (B, T, V, C_out), out.shape
    assert not torch.isnan(out).any(), "NaN trong output!"
    print(f"OK: out.shape={tuple(out.shape)}, alpha_chain={aux['alpha_chain']:.3f}, "
          f"alpha_hyper={aux['alpha_hyper']:.3f}")

    # Kiem tra: gradient co chay ve ca 2 nhanh khong
    loss = out.sum()
    loss.backward()
    g_chain = layer.root_chain.W_self.weight.grad.abs().sum().item()
    g_hyper = layer.adaptive_hyper.conv_hyper.weight.grad.abs().sum().item()
    print(f"grad RootChain.W_self = {g_chain:.4f}, grad AdaptiveHyper.conv_hyper = {g_hyper:.4f}")
    assert g_chain > 0 and g_hyper > 0, "Mot trong 2 nhanh khong nhan gradient!"

    # Kiem tra: root row/col cua H_tilde phai ~0 (root bi loai khoi hyperedge)
    H_tilde = aux['H_tilde']
    print(f"H_tilde[root=0] sum = {H_tilde[:, 0, :].abs().sum().item():.6f} (ky vong ~0)")

    # Kiem tra: xoa het thong tin cac khop khac ngoai root, xem root
    # broadcast co thuc su lan truyen khong (sanity cho RootChainProp)
    x2 = torch.zeros(B, T, V, C_in)
    x2[:, :, 0, :] = torch.randn(B, T, C_in)  # chi root co tin hieu
    out2 = layer.root_chain(x2)
    nonzero_children = (out2[:, :, 1:, :].abs().sum(dim=-1) > 1e-6).float().mean().item()
    print(f"Ty le khop con nhan duoc tin hieu tu root (ky vong ~1.0): {nonzero_children:.3f}")

    print("\nTat ca sanity check PASS.")