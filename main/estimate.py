import os
import argparse
import __init_path

from core.config import update_config, cfg

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Measure FPS of ARTS')
parser.add_argument('--seed',       type=int, default=123)
parser.add_argument('--cfg',        type=str, help='experiment configure file name')
parser.add_argument('--gpu',        type=str, default='0', help='assign GPU')

args = parser.parse_args()
if args.cfg:
    update_config(args.cfg)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
print("Work on GPU: ", os.environ['CUDA_VISIBLE_DEVICES'])

import torch
import time
torch.manual_seed(args.seed)

from core.base import Tester

# Load model đúng như file test
tester = Tester(args, load_dir=cfg.TEST.weight_path)
model = tester.model
model.eval()

# Dummy input
B, T, J = 1, cfg.DATASET.seqlen, 19
joint_img = torch.randn(B, T, J, 2).cuda()
img_feat  = torch.randn(B, T, 2048).cuda()

# ──────────────────────────────────────────────
# Đo Params
# ──────────────────────────────────────────────
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n{'='*45}")
print(f"  Total params     : {total_params/1e6:.2f} M")
print(f"  Trainable params : {trainable_params/1e6:.2f} M")

# ──────────────────────────────────────────────
# Đo GFLOPs
# ──────────────────────────────────────────────
try:
    from thop import profile, clever_format

    class _Wrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, ji, feat): return self.m(ji, feat)

    macs, _ = profile(_Wrapper(model), inputs=(joint_img, img_feat), verbose=False)
    flops_str = clever_format([macs * 2], '%.3f')[0]
    print(f"  GFLOPs (approx)  : {flops_str}  [B=1, seqlen={T}]")

except ImportError:
    try:
        from fvcore.nn import FlopCountAnalysis

        class _Wrapper(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, ji, feat): return self.m(ji, feat)

        flop_counter = FlopCountAnalysis(_Wrapper(model), (joint_img, img_feat))
        flop_counter.unsupported_ops_warnings(False)
        flops = flop_counter.total()
        print(f"  GFLOPs (approx)  : {flops/1e9:.3f} G  [B=1, seqlen={T}]")

    except ImportError:
        print("  GFLOPs           : install 'thop' hoặc 'fvcore'  (pip install thop)")

print(f"{'='*45}\n")

# ──────────────────────────────────────────────
# Warm-up
# ──────────────────────────────────────────────
print("Warming up...")
with torch.no_grad():
    for _ in range(50):
        _ = model(joint_img, img_feat)

# ──────────────────────────────────────────────
# Đo FPS
# ──────────────────────────────────────────────
print("Measuring FPS...")
torch.cuda.synchronize()
start = time.time()
N = 200

with torch.no_grad():
    for _ in range(N):
        _ = model(joint_img, img_feat)

torch.cuda.synchronize()
elapsed = time.time() - start

fps     = N / elapsed
latency = elapsed / N * 1000
print(f"\n{'='*45}")
print(f"  FPS     : {fps:.2f}")
print(f"  Latency : {latency:.2f} ms/frame")
print(f"{'='*45}")