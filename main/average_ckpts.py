"""
average_ckpts.py — Trung binh trong so nhieu checkpoint (model soup), KHONG can train
=====================================================================================

Dung cho cac Student cung kien truc va cung diem khoi tao (vd E1, E2, E3 deu khoi tao tu cung Teacher).
Trung binh tung tensor trong 'model_state_dict' (co the dat trong so rieng), tensor kieu so nguyen
(vd num_batches_tracked) lay tu checkpoint dau tien. Luu ra 1 checkpoint moi dung duoc ngay voi
main/test_ckpt.py.

Cach chay:
    python main/average_ckpts.py \
        --ckpts experiment/student_kd_e1/checkpoint/best.pth.tar experiment/student_kd_e3/checkpoint/best.pth.tar \
        --out experiment/soup_e1_e3.pth.tar [--weights 0.5 0.5]

    python main/test_ckpt.py --cfg ./config/train_student_kd_e1.yml --ckpt experiment/soup_e1_e3.pth.tar

Chi chay CPU, vai giay.
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
import argparse
import pickle

import torch

parser = argparse.ArgumentParser(description='Trung binh trong so nhieu checkpoint')
parser.add_argument('--ckpts', type=str, nargs='+', required=True)
parser.add_argument('--out', type=str, required=True)
parser.add_argument('--weights', type=float, nargs='*', default=None, help='trong so tung ckpt (mac dinh deu nhau)')
args = parser.parse_args()


class _NpUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


class _Shim:
    Unpickler = _NpUnpickler
    load = pickle.load
    Pickler = pickle.Pickler
    dump = pickle.dump


def load(path):
    ck = torch.load(path, map_location='cpu', pickle_module=_Shim, weights_only=False)
    for k in ('model_state_dict', 'state_dict', 'model'):
        if isinstance(ck, dict) and k in ck:
            return ck, ck[k], k
    return {'model_state_dict': ck}, ck, 'model_state_dict'


n = len(args.ckpts)
w = args.weights if args.weights else [1.0 / n] * n
if len(w) != n:
    sys.exit(f'[LOI] --weights co {len(w)} gia tri, can {n}')
s = sum(w)
w = [x / s for x in w]

loaded = [load(p) for p in args.ckpts]
base_ck, base_sd, key = loaded[0]
keys = list(base_sd.keys())
for p, (_, sd, _) in zip(args.ckpts[1:], loaded[1:]):
    miss = set(keys) ^ set(sd.keys())
    if miss:
        sys.exit(f'[LOI] {p} khac tap key voi ckpt dau ({len(miss)} key), vd: {sorted(miss)[:5]}')

avg = {}
n_int = 0
for k in keys:
    t0 = base_sd[k]
    if not torch.is_floating_point(t0):
        avg[k] = t0.clone()
        n_int += 1
        continue
    acc = torch.zeros_like(t0, dtype=torch.float32)
    for wi, (_, sd, _) in zip(w, loaded):
        if sd[k].shape != t0.shape:
            sys.exit(f'[LOI] lech shape o {k}: {tuple(sd[k].shape)} vs {tuple(t0.shape)}')
        acc += wi * sd[k].float()
    avg[k] = acc.to(t0.dtype)

out = {key: avg, 'epoch': base_ck.get('epoch', 0) if isinstance(base_ck, dict) else 0,
       'soup_of': list(args.ckpts), 'soup_weights': w}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
torch.save(out, args.out)
print(f'Da trung binh {n} checkpoint (trong so {[round(x, 3) for x in w]}), '
      f'{len(keys)} tensor ({n_int} tensor so nguyen lay tu ckpt dau) -> {args.out}')
