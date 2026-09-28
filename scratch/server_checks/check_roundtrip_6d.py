"""check_roundtrip_6d.py - Verify 6D rotation conversion.

Checks
======
0. Metric self-test: the geodesic metric must resolve a known 1e-5 rad rotation
   (otherwise thresholds below are meaningless).
1. aa -> 6d -> aa is lossless, measured by geodesic angle (float64, atan2 form),
   reported per angle group (near 0 / mid / near pi).
2. rot6d_to_rotmat (independent SPIN implementation) == rotmat(aa).
3. Layout [r00,r01,r10,r11,r20,r21] verified against HAND-COMPUTED rotations
   (90 deg about x, y, z), independent of any code path under test.
4. axis_angle_to_rotmat returns proper rotations (orthonormal, det=+1).

Thresholds are fixed here, before seeing results. Do not relax after the fact.
"""

import os
import sys
import argparse
import math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.smpl_hyperdiff import axis_angle_to_rot6d, axis_angle_to_rotmat
from utils.transforms import rot6d_to_axis_angle
from geometry import rot6d_to_rotmat

TH_ZERO = 1e-4
TH_MID = 1e-4
TH_PI = 1e-3
TH_ROTMAT = 1e-5
TH_LAYOUT = 1e-6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', type=str, default='')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--out_dir', type=str, default='logs/server_checks')
    p.add_argument('--n', type=int, default=3000)
    return p.parse_args()


def geodesic_angle(R1, R2):
    """Numerically stable geodesic angle between rotation batches (N,3,3), float64."""
    R1 = R1.double()
    R2 = R2.double()
    M = torch.bmm(R1.transpose(1, 2), R2)
    cos = (M.diagonal(dim1=1, dim2=2).sum(-1) - 1.0) / 2.0
    A = M - M.transpose(1, 2)
    vee = torch.stack([A[:, 2, 1], A[:, 0, 2], A[:, 1, 0]], dim=-1)
    sin = 0.5 * vee.norm(dim=-1)
    return torch.atan2(sin, cos)


def rodrigues_f64(aa):
    """Reference Rodrigues in float64, written independently of the repo."""
    aa = aa.double()
    theta = aa.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    k = aa / theta
    K = torch.zeros(aa.shape[0], 3, 3, dtype=torch.float64, device=aa.device)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]
    I = torch.eye(3, dtype=torch.float64, device=aa.device).expand_as(K)
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    return I + s * K + (1 - c) * torch.bmm(K, K)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = os.path.join(args.out_dir, 'check_roundtrip_6d.log')

    def log(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== check_roundtrip_6d ===\n")

    all_pass = True
    try:
        dev = args.device
        torch.manual_seed(0)

        # ---- Test 0: metric self-test -------------------------------------
        log("\n--- [Test 0] Metric self-test (must resolve 1e-5 rad) ---")
        axis = torch.tensor([[0.3, -0.5, 0.8]], dtype=torch.float64)
        axis = axis / axis.norm()
        aa_small = axis * 1e-5
        R_i = torch.eye(3, dtype=torch.float64)[None]
        R_s = rodrigues_f64(aa_small)
        est64 = geodesic_angle(R_i, R_s).item()
        est32 = geodesic_angle(R_i.float(), R_s.float()).item()  # mimics float32 inputs
        log(f"true=1.00e-05  est(f64 in)={est64:.3e}  est(f32 in)={est32:.3e}")
        if abs(est64 - 1e-5) > 1e-8 or abs(est32 - 1e-5) > 1e-6:
            log("FAIL: geodesic metric cannot resolve small angles; other results invalid")
            all_pass = False

        # ---- Test 4 (sanity of R): proper rotations -----------------------
        N = args.n
        d = torch.randn(N, 3)
        d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
        mag = torch.zeros(N, 1)
        n3 = N // 3
        mag[:n3] = torch.rand(n3, 1) * 0.1
        mag[n3:2 * n3] = math.pi - torch.rand(n3, 1) * 0.1
        mag[2 * n3:] = 0.1 + torch.rand(N - 2 * n3, 1) * (math.pi - 0.2)
        aa = (d * mag).to(dev)
        mag = mag.squeeze(-1)

        R_orig = axis_angle_to_rotmat(aa)
        R_ref = rodrigues_f64(aa.cpu()).to(R_orig.device)
        log("\n--- [Test 4] axis_angle_to_rotmat is a proper rotation ---")
        ortho = (torch.bmm(R_orig.double(), R_orig.double().transpose(1, 2))
                 - torch.eye(3, dtype=torch.float64, device=R_orig.device)).abs().max().item()
        det_err = (torch.det(R_orig.double()) - 1.0).abs().max().item()
        vs_ref = geodesic_angle(R_orig, R_ref).max().item()
        log(f"orthonormality err={ortho:.2e}  det err={det_err:.2e}  "
            f"geodesic vs independent Rodrigues (max)={vs_ref:.2e}")
        if ortho > 1e-5 or det_err > 1e-5 or vs_ref > 1e-5:
            log("FAIL: axis_angle_to_rotmat is not a proper/correct rotation")
            all_pass = False

        # ---- Test 1: roundtrip --------------------------------------------
        r6d = axis_angle_to_rot6d(aa)
        aa_back = rot6d_to_axis_angle(r6d)
        R_back = axis_angle_to_rotmat(aa_back)
        geo = geodesic_angle(R_orig, R_back)

        log("\n--- [Test 1] aa->6d->aa geodesic error by group (float64 atan2) ---")
        groups = [
            ("Near Zero (<0.1)", mag < 0.1, TH_ZERO),
            ("Mid", (mag >= 0.1) & (mag <= math.pi - 0.1), TH_MID),
            ("Near Pi (>pi-0.1)", mag > math.pi - 0.1, TH_PI),
        ]
        log(f"{'Group':>20} | {'Max(rad)':>10} | {'Mean(rad)':>10} | {'Thresh':>8} | Status")
        log("-" * 70)
        worst_rows = []
        for name, m, th in groups:
            m = m.to(geo.device)
            g = geo[m]
            mx, mn = g.max().item(), g.mean().item()
            ok = mx <= th
            all_pass &= ok
            log(f"{name:>20} | {mx:>10.2e} | {mn:>10.2e} | {th:>8.0e} | {'PASS' if ok else 'FAIL'}")
            if not ok:
                idx = torch.nonzero(m).squeeze(-1)
                top = idx[g.topk(min(3, len(g))).indices]
                for i in top.tolist():
                    worst_rows.append((name, aa[i].tolist(), aa_back[i].tolist(), geo[i].item()))
        for name, a, b, e in worst_rows:
            log(f"  worst[{name}] aa={['%.5f' % x for x in a]} back={['%.5f' % x for x in b]} geo={e:.2e}")

        # ---- Test 2: vs SPIN rot6d_to_rotmat ------------------------------
        R6 = rot6d_to_rotmat(r6d)
        if R6.shape != R_orig.shape:
            R6 = R6.reshape(N, 3, 3)
        err_mat = geodesic_angle(R6, R_orig).max().item()
        log("\n--- [Test 2] rot6d_to_rotmat(6d) vs rotmat(aa) (geodesic) ---")
        log(f"max={err_mat:.2e} (threshold {TH_ROTMAT:.0e})")
        if err_mat > TH_ROTMAT:
            log("FAIL: geometry.rot6d_to_rotmat disagrees with axis_angle_to_rot6d")
            all_pass = False

        # ---- Test 3: hand-computed layout ---------------------------------
        h = math.pi / 2
        cases = {
            "90deg about z": ([0, 0, h], [0, -1, 1, 0, 0, 0]),
            "90deg about x": ([h, 0, 0], [1, 0, 0, 0, 0, -1]),
            "90deg about y": ([0, h, 0], [0, 0, 0, 1, -1, 0]),
        }
        log("\n--- [Test 3] Layout [r00,r01,r10,r11,r20,r21] vs hand-computed ---")
        for name, (a, exp) in cases.items():
            out = axis_angle_to_rot6d(torch.tensor([a], dtype=torch.float32, device=dev))
            out = out.reshape(-1).double().cpu()
            err = (out - torch.tensor(exp, dtype=torch.float64)).abs().max().item()
            ok = err <= TH_LAYOUT
            all_pass &= ok
            log(f"{name}: got={['%.3f' % x for x in out.tolist()]} expected={exp} err={err:.1e} "
                f"{'PASS' if ok else 'FAIL'}")

        log(f"\nRan on device='{dev}'.")
        if all_pass:
            log("\nPASS - conversion, layout and metric all verified.")
        else:
            log("\nFAIL - see failing rows above.")
            sys.exit(1)

    except SystemExit:
        raise
    except Exception:
        import traceback
        log(f"FAIL - Exception occurred:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()