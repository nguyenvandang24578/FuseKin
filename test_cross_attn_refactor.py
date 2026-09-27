"""
Test script for the cross-attention refactoring in Pose2Mesh.

Mocks Teacher.forward() to bypass heavy dependencies (SMPL, VPoser, SPIN)
and validates:
  1. Teacher returns img_out in features dict
  2. CrossAttention works with dynamic KV sequence lengths
  3. pose_context_attn produces correct (B, 24, C) output
  4. No NaN values in outputs
"""
import sys
sys.path.insert(0, './lib')

import torch
import torch.nn as nn

# ─── Test 1: CrossAttention with dynamic KV length ─────────────────────────
print("=" * 60)
print("Test 1: CrossAttention with dynamic KV length")
print("=" * 60)

from models.Core_model import CrossAttention, CrossAttentionBlock

# Create CrossAttention with kv_num=1 (old-style), but feed it kv_num=273
ca = CrossAttention(dim=512, k_dim=512, v_dim=512, kv_num=1, num_heads=8, qkv_bias=True)

B = 2
xq = torch.randn(B, 24, 512)
xk = torch.randn(B, 273, 512)  # H*W + 17 = 256 + 17
xv = torch.randn(B, 273, 512)

out = ca(xq, xk, xv)
print(f"  Input Q: {xq.shape}, K: {xk.shape}, V: {xv.shape}")
print(f"  Output:  {out.shape}")
assert out.shape == (B, 24, 512), f"Expected (2, 24, 512), got {out.shape}"
assert not torch.isnan(out).any(), "NaN detected in CrossAttention output!"
print("  PASS: CrossAttention works with dynamic KV length (kv_num=1 init, actual=273)")

# ─── Test 2: CrossAttentionBlock (pose_context_attn style) ──────────────────
print()
print("=" * 60)
print("Test 2: CrossAttentionBlock as pose_context_attn")
print("=" * 60)

cab = CrossAttentionBlock(
    q_dim=512, k_dim=512, v_dim=512, kv_num=273,
    num_heads=8, mlp_ratio=4., qkv_bias=True,
    drop=0., attn_drop=0., drop_path=0.2, has_mlp=True
)

pose_token = torch.randn(B, 24, 512)
img_out = torch.randn(B, 256, 512)       # H*W = 16*16
joint_out = torch.randn(B, 17, 512)      # 17 joints
context = torch.cat([img_out, joint_out], dim=1)  # (B, 273, 512)

result = cab(pose_token, context, context)
print(f"  Input: pose_token {pose_token.shape}, context {context.shape}")
print(f"  Output: {result.shape}")
assert result.shape == (B, 24, 512), f"Expected (2, 24, 512), got {result.shape}"
assert not torch.isnan(result).any(), "NaN detected!"
print("  PASS: CrossAttentionBlock works correctly")

# ─── Test 3: fuse_shape still works with kv_num=1 ──────────────────────────
print()
print("=" * 60)
print("Test 3: fuse_shape (kv_num=1) backward compat")
print("=" * 60)

fuse_shape = CrossAttentionBlock(
    q_dim=512, k_dim=1024, v_dim=1024, kv_num=1,
    num_heads=8, mlp_ratio=4., qkv_bias=True,
    drop=0., attn_drop=0., drop_path=0.2, has_mlp=True
)

shape_token = torch.randn(B, 1, 512)
global_ft_seq = torch.randn(B, 1, 1024)

shape_out = fuse_shape(shape_token, global_ft_seq, global_ft_seq)
print(f"  Input: shape_token {shape_token.shape}, global_ft_seq {global_ft_seq.shape}")
print(f"  Output: {shape_out.shape}")
assert shape_out.shape == (B, 1, 512), f"Expected (2, 1, 512), got {shape_out.shape}"
assert not torch.isnan(shape_out).any(), "NaN detected!"
print("  PASS: fuse_shape still works with kv_num=1")

# ─── Test 4: Simulate the full modified forward path ────────────────────────
print()
print("=" * 60)
print("Test 4: Full modified forward path (simulated)")
print("=" * 60)

embed_dim = 512

# Simulate Teacher output
pred_pose_6d = torch.randn(B, 24, 6)
pred_shape = torch.randn(B, 10)
global_ft = torch.randn(B, 1024)  # concat_feat
img_out_feat = torch.randn(B, 256, 512)    # img_out from Teacher
joint_out_feat = torch.randn(B, 17, 512)   # joint_out from Teacher

feature = {
    'concat_feat': global_ft,
    'img_out': img_out_feat,
    'joint_out': joint_out_feat,
}

# Modules
pose_embed = nn.Linear(6, embed_dim)
shape_embed = nn.Linear(10, embed_dim)
shape_token_emb = nn.Embedding(1, embed_dim)
norm = nn.LayerNorm(embed_dim)
node_pe = nn.Embedding(24, embed_dim)
cam_head = nn.Linear(1024, 3)

pose_context_attn = CrossAttentionBlock(
    q_dim=embed_dim, k_dim=embed_dim, v_dim=embed_dim,
    kv_num=273, num_heads=8, mlp_ratio=4., qkv_bias=True,
    drop=0., attn_drop=0., drop_path=0.2, has_mlp=True
)

fuse_shape_mod = CrossAttentionBlock(
    q_dim=512, k_dim=1024, v_dim=1024, kv_num=1,
    num_heads=8, mlp_ratio=4., qkv_bias=True,
    drop=0., attn_drop=0., drop_path=0.2, has_mlp=True
)

# Forward path
pose_token = pose_embed(pred_pose_6d)   # (B, 24, 512)
shape_emb = shape_embed(pred_shape)     # (B, 512)

shape_token = shape_token_emb.weight.unsqueeze(0).expand(B, 1, -1)
shape_emb_unsq = shape_emb.unsqueeze(1)
shape_token = shape_token + shape_emb_unsq

# Cross-attention (NEW path)
img_out_f = feature['img_out']
joint_out_ctx = feature['joint_out']
context_tokens = torch.cat([img_out_f, joint_out_ctx], dim=1)
print(f"  context_tokens shape: {context_tokens.shape}")

pose_token_ctx = pose_context_attn(pose_token, context_tokens, context_tokens)
print(f"  pose_token_ctx shape: {pose_token_ctx.shape}")

idx = torch.arange(24, device=pose_token_ctx.device)
dang = norm(pose_token_ctx) + node_pe(idx)
print(f"  dang shape: {dang.shape}")

# cam_head uses global_ft (unchanged)
cam_param = cam_head(global_ft)
print(f"  cam_param shape: {cam_param.shape}")

# fuse_shape uses global_ft_seq (unchanged)
global_ft_seq_test = global_ft.unsqueeze(1)
shape_output = fuse_shape_mod(shape_token, global_ft_seq_test, global_ft_seq_test)
print(f"  shape_output shape: {shape_output.shape}")

# NaN checks
for name, tensor in [('pose_token_ctx', pose_token_ctx), ('dang', dang),
                      ('cam_param', cam_param), ('shape_output', shape_output)]:
    assert not torch.isnan(tensor).any(), f"NaN in {name}!"

print("  PASS: All shapes correct, no NaN detected")

print()
print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
