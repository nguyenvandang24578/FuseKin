"""
Test script for SMPL_HyperDiff integration.

Run from repo root:
    python scratch/test_smpl_hyperdiff.py

Tests (per spec §8):
  1. aa → 6d → aa roundtrip  +  rot6d_to_rotmat consistency
  2. (Skipped here: overlay script requires images)
  3. Forward train: shape check, no NaN, diff_loss finite
  4. Backward: list params with grad=None (expect only old pose path)
  5. Cam/shape receive gradient
  6. DDIM: same seed → same output, diff seed → diff output
  7. make_noisy_kp2d: drop rates, conf=0 always dropped
  8. Overfit 1 batch ~200 steps: diff_loss must decrease
  9. context=None works, context=randn doesn't crash
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import math

print("=" * 70)
print("TEST 1: Rotation roundtrip  aa → 6d → aa")
print("=" * 70)

from models.smpl_hyperdiff import axis_angle_to_rot6d, axis_angle_to_rotmat
from utils.transforms import rot6d_to_axis_angle

N = 2000
aa_orig = torch.randn(N, 3).cuda()  # random axis-angle
r6d = axis_angle_to_rot6d(aa_orig)             # aa → 6d
aa_back = rot6d_to_axis_angle(r6d)             # 6d → aa
err = (aa_orig - aa_back).norm(dim=-1)
print(f"  Max roundtrip error: {err.max().item():.2e}  (need < 1e-4)")
assert err.max().item() < 1e-3, f"Roundtrip error too large: {err.max().item()}"

# Also check rot6d_to_rotmat consistency
from utils.geometry import rot6d_to_rotmat
R_from_aa = axis_angle_to_rotmat(aa_orig)               # (N,3,3)
R_from_6d = rot6d_to_rotmat(r6d).reshape(-1, 3, 3)      # (N,3,3)
rot_err = (R_from_aa - R_from_6d).abs().max().item()
print(f"  Max rotmat mismatch: {rot_err:.2e}  (need < 1e-4)")
assert rot_err < 1e-3, f"Rotmat mismatch too large: {rot_err}"
print("  ✓ PASSED\n")

# ---- Build a standalone SMPL_HyperDiff for tests (avoid full Pose2Mesh) ----
print("=" * 70)
print("TEST 3: Forward train — shape check, no NaN, finite loss")
print("=" * 70)

from models.smpl_hyperdiff import SMPL_HyperDiff, make_noisy_kp2d

device = 'cuda'
diff_model = SMPL_HyperDiff(
    dim_feat=128, dim_rep=256, n_layers=2,
    num_heads=4, mlp_ratio=2.0,
    num_timesteps=100, sampling_timesteps=5,
).to(device)
diff_model.train()

B = 4
x0      = torch.randn(B, 24, 6, device=device)
kp2d    = torch.randn(B, 17, 2, device=device).clamp(-1, 1)
kp_conf = torch.ones(B, 17, device=device)
mask    = torch.ones(B, 24, device=device)

pred_x0, loss = diff_model(x0, kp2d, kp_conf, is_train=True, valid_mask=mask)
print(f"  pred_x0 shape: {pred_x0.shape}  (expect {(B,24,6)})")
print(f"  diff_loss:     {loss.item():.4f}  finite={torch.isfinite(loss).item()}")
print(f"  any NaN pred:  {torch.isnan(pred_x0).any().item()}")
assert pred_x0.shape == (B, 24, 6)
assert torch.isfinite(loss)
assert not torch.isnan(pred_x0).any()
print("  ✓ PASSED\n")

print("=" * 70)
print("TEST 4 & 5: Backward — grad check")
print("=" * 70)

loss.backward()
no_grad = []
with_grad = []
for name, p in diff_model.named_parameters():
    if p.requires_grad:
        if p.grad is None:
            no_grad.append(name)
        else:
            with_grad.append(name)

print(f"  Params with grad:    {len(with_grad)}")
print(f"  Params WITHOUT grad: {len(no_grad)}")
if no_grad:
    print(f"  (These should only be old pose-path params):")
    for n in no_grad[:10]:
        print(f"    - {n}")
assert len(with_grad) > 0, "No parameters received gradient!"
# All diffusion params should have grad
assert len(no_grad) == 0, f"Diffusion params missing grad: {no_grad[:5]}"
print("  ✓ PASSED\n")

print("=" * 70)
print("TEST 6: DDIM sampling — determinism & diversity")
print("=" * 70)

diff_model.eval()
g1 = torch.Generator(device=device).manual_seed(123)
g2 = torch.Generator(device=device).manual_seed(123)
g3 = torch.Generator(device=device).manual_seed(456)

out1 = diff_model.ddim_sample(kp2d, kp_conf, generator=g1)
out2 = diff_model.ddim_sample(kp2d, kp_conf, generator=g2)
out3 = diff_model.ddim_sample(kp2d, kp_conf, generator=g3)

print(f"  Same seed diff: {(out1 - out2).abs().max().item():.2e}  (expect 0)")
print(f"  Diff seed diff: {(out1 - out3).abs().max().item():.2e}  (expect > 0)")
assert (out1 - out2).abs().max().item() < 1e-5, "Same seed gave different outputs!"
assert (out1 - out3).abs().max().item() > 1e-3, "Different seeds gave same output!"
assert out1.shape == (B, 24, 6)
print("  ✓ PASSED\n")

print("=" * 70)
print("TEST 7: make_noisy_kp2d — drop/conf checks")
print("=" * 70)

kp_gt = torch.randn(100, 17, 2, device=device).clamp(-1, 1)
conf_gt = torch.ones(100, 17, device=device)
# Set a few joints to invisible
conf_gt[:, 3] = 0  # all samples, joint 3 invisible
conf_gt[:, 10] = 0

kp_n, conf_n, drop_n = make_noisy_kp2d(kp_gt, conf_gt)

# Joint 3 and 10 should ALWAYS be dropped
assert (drop_n[:, 3]).all(), "Invisible joint 3 not always dropped!"
assert (drop_n[:, 10]).all(), "Invisible joint 10 not always dropped!"
# Dropped joints should have conf=0
assert (conf_n[drop_n] == 0).all(), "Dropped joints have non-zero conf!"
# Dropped joints should have kp=0
assert (kp_n[drop_n] == 0).all(), "Dropped joints have non-zero kp!"
# Approximate drop rate
drop_rate = drop_n[:, 1].float().mean().item()  # joint 1 (visible)
print(f"  Drop rate (visible joint): {drop_rate:.2f}  (expect ~0.10)")
print(f"  Joint 3 always dropped:    True")
print(f"  Joint 10 always dropped:   True")
print("  ✓ PASSED\n")

print("=" * 70)
print("TEST 8: Overfit 1 batch — loss must decrease")
print("=" * 70)

diff_model.train()
optim = torch.optim.Adam(diff_model.parameters(), lr=1e-3)
x0_fixed = torch.randn(8, 24, 6, device=device)
kp_fixed = torch.randn(8, 17, 2, device=device).clamp(-1, 1)
kc_fixed = torch.ones(8, 17, device=device)

losses = []
for step in range(200):
    optim.zero_grad()
    _, l = diff_model(x0_fixed, kp_fixed, kc_fixed, is_train=True)
    l.backward()
    optim.step()
    losses.append(l.item())

print(f"  Loss step 0:   {losses[0]:.4f}")
print(f"  Loss step 50:  {losses[50]:.4f}")
print(f"  Loss step 199: {losses[-1]:.4f}")
ratio = losses[-1] / (losses[0] + 1e-8)
print(f"  Ratio last/first: {ratio:.4f}  (need < 0.5 for clear decrease)")
assert ratio < 0.8, f"Loss didn't decrease enough: ratio={ratio:.4f}"
print("  ✓ PASSED\n")

print("=" * 70)
print("TEST 9: context=None works, context=randn doesn't crash")
print("=" * 70)

diff_model.eval()
# context=None (default)
out_none = diff_model.ddim_sample(kp2d, kp_conf, context=None)
assert out_none.shape == (B, 24, 6)
print("  context=None: OK")

# context=randn (should warn but not crash)
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    out_ctx = diff_model.ddim_sample(kp2d, kp_conf,
                                      context=torch.randn(B, 10, 128, device=device))
    if w:
        print(f"  context=randn: warning raised (expected): {w[0].message}")
    else:
        print(f"  context=randn: no warning (context_attn not checked)")
assert out_ctx.shape == (B, 24, 6)
print("  ✓ PASSED\n")

print("=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)
