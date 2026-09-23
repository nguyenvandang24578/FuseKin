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
    H_init - 5 hyperedge (KHONG con hyperedge rieng cho root). 5 chain:
      0: Torso+Head : 3,6,9,12,13,14,15
      1: Left Arm   : 16,18,20,22
      2: Right Arm  : 17,19,21,23
      3: Left Leg   : 1,4,7,10
      4: Right Leg  : 2,5,8,11
    Joint 0 (root) co hang toan so 0 -> khong tham gia hyper-mix,
    duoc xu ly rieng o nhanh RootChainProp.
    """
    H = torch.zeros(num_nodes, num_edges)
    for i in [3, 6, 9, 12, 13, 14, 15]: H[i, 0] = 1.0   # Torso+Head
    for i in [16, 18, 20, 22]:          H[i, 1] = 1.0   # Left Arm
    for i in [17, 19, 21, 23]:          H[i, 2] = 1.0   # Right Arm
    for i in [1, 4, 7, 10]:             H[i, 3] = 1.0   # Left Leg
    for i in [2, 5, 8, 11]:             H[i, 4] = 1.0   # Right Leg
    return H


def compute_G_batched(H, eps=1e-8):
    """G = Dv^{-1/2} H De^{-1} H^T Dv^{-1/2}, batched. H: (B, N, E), luon >= 0."""
    H = H.clamp(min=0.0)
    Dv = H.sum(dim=-1) + eps                        # (B, N)
    Dv_inv_sqrt = Dv.pow(-0.5)
    De = H.sum(dim=1) + eps                          # (B, E)
    De_inv = De.pow(-1)
    H_De = H * De_inv.unsqueeze(1)
    G = torch.bmm(H_De, H.transpose(1, 2))           # (B, N, N)
    G = Dv_inv_sqrt.unsqueeze(-1) * G * Dv_inv_sqrt.unsqueeze(1)
    return G


# ============================================================
# Nhanh A: RootChainProp — cha->con (1-hop)
# ============================================================
class RootChainProp(nn.Module):
    """
    Voi moi khop i != root: h_i_new = W_self(h_i) + W_parent(h_parent(i))
        (chi 1-hop, dung 'gather' truc tiep theo bang PARENT)
    Voi root (joint 0)    : h_0_new = W_root(h_0)  (khong co cha)

    Input/Output: (B, 24, C)
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.W_self   = nn.Linear(dim_in, dim_out)
        self.W_parent = nn.Linear(dim_in, dim_out)
        self.W_root   = nn.Linear(dim_in, dim_out)

        parent_idx = torch.tensor(
            [p if p >= 0 else 0 for p in PARENT], dtype=torch.long
        )  # (24,) — root tam gan idx 0 (chinh no), se bi mask ve 0 sau
        self.register_buffer('parent_idx', parent_idx)

        child_mask = torch.tensor(
            [0.0 if p < 0 else 1.0 for p in PARENT]
        ).view(1, NUM_JOINTS, 1)  # (1, 24, 1) — root=0, con lai=1
        self.register_buffer('child_mask', child_mask)

    def forward(self, x):
        """x: (B, 24, C) -> (B, 24, C_out)"""
        h_parent = x[:, self.parent_idx, :]                   # (B, 24, C)
        local = self.W_self(x) + self.child_mask * self.W_parent(h_parent)

        root_feat = self.W_root(x[:, 0, :])                   # (B, C_out)
        local = local.clone()
        local[:, 0, :] = root_feat
        return local


# ============================================================
# Nhanh B: AdaptiveHyperNoRoot — xuyen-chain, KHONG bao gom root
# ============================================================
class AdaptiveHyperNoRoot(nn.Module):
    """
    Bat tuong tac xuyen-chain (vd 2 tay cham nhau) ma nhanh RootChain
    (chi lan truyen doc theo 1 chain) khong nam duoc.
    Root bi loai khoi H_init/M/S vi ban chat thong ke khac biet.

    H_tilde = beta0*H_init + beta1*M + beta2*S
      - H_init : cau truc co dinh (5 chain)
      - M       : ma tran hoc duoc, khoi tao bang H_init
      - S       : phu thuoc du lieu (similarity-based, per sample)

    Input/Output: (B, 24, C)
    """

    def __init__(self, dim_in, dim_out, num_edges=5):
        super().__init__()

        H_init = build_H_init_no_root(NUM_JOINTS, num_edges)
        self.register_buffer('H_init', H_init)                 # (24, E)

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
        """
        x: (B, 24, C) -> S: (B, 24, E)
        Tinh do tuong dong giua cac khop, chieu huong thong tin theo H_init.
        """
        b, v, c = x.shape
        q   = self.phi1(x)                                     # (B, 24, C)
        k   = self.phi2(x).transpose(1, 2)                     # (B, C, 24)
        sim = torch.bmm(q, k) / sqrt(c)                        # (B, 24, 24)

        H_init_exp = self.H_init.unsqueeze(0).expand(b, -1, -1)
        S = torch.softmax(torch.bmm(sim, H_init_exp), dim=-1)  # (B, 24, E)

        # Mask root: dam bao root khong tham gia hyper-branch
        S = S.clone()
        S[:, 0, :] = 0.0
        return S

    def forward(self, x):
        """x: (B, 24, C) -> out: (B, 24, C_out), H_tilde: (B, 24, E)"""
        b, v, c = x.shape
        S = self._compute_S(x)                                 # (B, 24, E)

        beta0 = F.softplus(self.beta0_raw)
        beta1 = F.softplus(self.beta1_raw)
        beta2 = F.softplus(self.beta2_raw)
        M     = F.softplus(self.M_raw)

        H_init_exp = self.H_init.unsqueeze(0).expand(b, -1, -1)
        M_exp      = M.unsqueeze(0).expand(b, -1, -1)
        H_tilde    = beta0 * H_init_exp + beta1 * M_exp + beta2 * S  # (B, 24, E)

        G   = compute_G_batched(H_tilde)                       # (B, 24, 24)
        xh  = torch.bmm(G, x)                                  # (B, 24, C)
        out = self.conv_hyper(xh)                              # (B, 24, C_out)
        return out, H_tilde


# ============================================================
# Module tong: HYPERGC v2
# ============================================================
class HYPERGCv2(nn.Module):
    """
    Input/Output: (B, 24, C)

    out = ReLU( LayerNorm( alpha_chain * RootChain(x)
                          + alpha_hyper * AdaptiveHyper(x)
                          + U(x) ) )
          (+ residual x neu dim_in == dim_out)
    """

    def __init__(self, dim_in, dim_out, num_edges=5):
        super().__init__()
        self.dim_in  = dim_in
        self.dim_out = dim_out

        self.root_chain     = RootChainProp(dim_in, dim_out)
        self.adaptive_hyper = AdaptiveHyperNoRoot(dim_in, dim_out, num_edges)

        self.alpha_chain_raw = nn.Parameter(torch.tensor(1.0))
        self.alpha_hyper_raw = nn.Parameter(torch.tensor(1.0))

        self.U          = nn.Linear(dim_in, dim_out)
        self.layer_norm = nn.LayerNorm(dim_out)
        self.relu       = nn.ReLU()

    def forward(self, x):
        """x: (B, 24, C_in) -> out: (B, 24, C_out), aux: dict"""
        a_chain              = self.root_chain(x)              # (B, 24, C_out)
        a_hyper, H_tilde     = self.adaptive_hyper(x)         # (B, 24, C_out)

        agg = self.alpha_chain_raw * a_chain + self.alpha_hyper_raw * a_hyper

        if self.dim_in == self.dim_out:
            out = self.relu(x + self.layer_norm(agg + self.U(x)))
        else:
            out = self.relu(self.layer_norm(agg + self.U(x)))

        aux = {
            'H_tilde'    : H_tilde.detach(),          # (B, 24, E)
            'alpha_chain': self.alpha_chain_raw.item(),
            'alpha_hyper': self.alpha_hyper_raw.item(),
        }
        return out, aux


# ============================================================
# Sanity check
# ============================================================
if __name__ == '__main__':
    torch.manual_seed(0)
    B, V, C_in, C_out = 2, 24, 512, 512

    layer = HYPERGCv2(C_in, C_out, num_edges=5)
    x = torch.randn(B, V, C_in)
    out, aux = layer(x)

    assert out.shape == (B, V, C_out), f"Shape sai: {out.shape}"
    assert not torch.isnan(out).any(), "NaN trong output!"
    print(f"OK: out.shape={tuple(out.shape)}, "
          f"alpha_chain={aux['alpha_chain']:.3f}, "
          f"alpha_hyper={aux['alpha_hyper']:.3f}")

    # Kiem tra gradient chay ve ca 2 nhanh
    loss = out.sum()
    loss.backward()
    g_chain = layer.root_chain.W_self.weight.grad.abs().sum().item()
    g_hyper = layer.adaptive_hyper.conv_hyper.weight.grad.abs().sum().item()
    print(f"grad RootChain.W_self={g_chain:.4f}, "
          f"grad AdaptiveHyper.conv_hyper={g_hyper:.4f}")
    assert g_chain > 0 and g_hyper > 0, "Mot nhanh khong nhan gradient!"

    # Kiem tra root row cua H_tilde ~0 (root bi loai khoi hyperedge)
    H_tilde = aux['H_tilde']                         # (B, 24, E)
    print(f"H_tilde[root=0] sum = {H_tilde[:, 0, :].abs().sum().item():.6f} "
          f"(ky vong ~0)")

    # Kiem tra chi 3 khop con truc tiep cua root (1,2,3) nhan tin hieu
    x2 = torch.zeros(B, V, C_in)
    x2[:, 0, :] = torch.randn(B, C_in)
    with torch.no_grad():
        out2 = layer.root_chain(x2)
    direct_children = [i for i, p in enumerate(PARENT) if p == 0]   # [1, 2, 3]
    signal = out2[:, direct_children, :].abs().sum(dim=-1).mean().item()
    print(f"Tin hieu trung binh tai 3 khop con truc tiep cua root "
          f"(joint 1,2,3): {signal:.4f} (ky vong > 0)")

    print("\nTat ca sanity check PASS.")