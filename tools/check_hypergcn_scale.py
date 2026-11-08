# -*- coding: utf-8 -*-
"""
Kiem tra nhanh HyperGCN va cac nhanh khoi tao 0 trong checkpoint Student (chi doc trong so, khong can data/GPU).

Muc dich:
  - layer_scale_1/2 cua 3 block HyperGCN (khoi tao 1e-5): con nho -> HyperGCN gan nhu dong nhat (khong duoc dung).
  - alpha (part/body/joint) cua mixer, thong ke BatchNorm (running_mean/var) -> xem BN co lech manh khong.
  - Cac lop khoi tao 0 cua query: bone_proj (v4), bone_geo_proj[-1] (v5) -> query co thuc su dung hinh hoc xuong khong.

Cach chay (co the dua nhieu ckpt de so sanh, vd v4 va v5):
  python tools/check_hypergcn_scale.py \
      experiment/student_v4_imglift_markov/checkpoint/last.pth.tar \
      experiment/student_v5_bonequery/checkpoint/last.pth.tar
"""
import os
import sys
import re
import torch

# Ten 17 khop H36M (thu tu dung trong hypergraph.py)
J17 = ['Pelvis', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'Torso', 'Neck', 'Nose', 'Head',
       'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']


def init_adj():
    """Lay p_adj / b_adj LUC KHOI TAO tu chinh class HyperGCN (repo) de so do troi. Loi import -> None."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
        from models.hypergraph import HyperGCN
        m = HyperGCN(8, 8, num_nodes=17, use_partscale=True, use_bodyscale=True, dataset='h36m')
        return m.p_adj.detach().float(), m.b_adj.detach().float()
    except Exception as e:  # khong co einops / loi khac -> bo qua phan so voi khoi tao
        print(f'(khong tao duoc p_adj/b_adj khoi tao: {e})')
        return None, None


INIT_P, INIT_B = None, None


def adj_report(name, A, A0):
    """In thong ke ma tran sieu canh A (17x17). norm() trong hypergraph.py chia moi cot cho (tong cot + 0.001)."""
    A = A.float()
    col = A.sum(0)                                   # tong cot, dung trong norm()
    factor = (col + 0.001).reciprocal()              # he so nhan cua norm() theo tung cot
    k = int(factor.abs().argmax())
    print(f'   {name}: tong cot min={col.min().item():+.4f} max={col.max().item():+.4f} | '
          f'|1/(tong+0.001)| max={factor.abs().max().item():.3e} tai cot {k} ({J17[k]}) | '
          f'so phan tu am={int((A < 0).sum())}/{A.numel()}')
    if A0 is not None:
        col0 = A0.sum(0)
        d = (A - A0).abs()
        print(f'   {"":5s} troi so voi khoi tao: mean|dA|={d.mean().item():.3e} max|dA|={d.max().item():.3e} | '
              f'tong cot khoi tao min={col0.min().item():+.4f} max={col0.max().item():+.4f}')
    # 3 cot co tong nho nhat (nguy co no cao nhat)
    idx = col.abs().argsort()[:3].tolist()
    print(f'   {"":5s} 3 cot |tong| nho nhat: ' + ', '.join(f'{J17[i]}={col[i].item():+.4f}' for i in idx))


def load_state(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = None
    for k in ('model_state_dict', 'state_dict', 'model', 'network'):
        if isinstance(ck, dict) and k in ck:
            sd = ck[k]
            break
    if sd is None:
        sd = ck
    # bo prefix 'module.' (DataParallel)
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    ep = ck.get('epoch', '?') if isinstance(ck, dict) else '?'
    return sd, ep


def stat(t):
    t = t.float()
    return f'mean|.|={t.abs().mean().item():.3e}  max|.|={t.abs().max().item():.3e}'


def report(path):
    sd, ep = load_state(path)
    print('=' * 100)
    print(f'{path}  | epoch {ep}')
    print('=' * 100)

    hyp = {k: v for k, v in sd.items() if 'spatial_hypers' in k}
    if not hyp:
        print('  (khong tim thay spatial_hypers trong checkpoint)')
    blocks = sorted({int(re.search(r'spatial_hypers\.(\d+)\.', k).group(1)) for k in hyp})
    for b in blocks:
        pre = [k for k in hyp if re.search(rf'spatial_hypers\.{b}\.', k)]
        print(f'-- HyperGCN block {b}')
        for name in ('layer_scale_1', 'layer_scale_2'):
            ks = [k for k in pre if k.endswith(name)]
            for k in ks:
                print(f'   {name:14s} {stat(hyp[k])}   (khoi tao 1e-5)')
        # alpha1 = part, alpha2 = body, alpha3 = joint (theo thu tu cong trong forward hypergraph.py)
        for a, lab in (('alpha1', 'part'), ('alpha2', 'body'), ('alpha3', 'joint')):
            ks = [k for k in pre if k.endswith('.' + a)]
            for k in ks:
                print(f'   {a} ({lab:5s})  = {hyp[k].item():+.4f}   (khoi tao 1)')
        for name in ('batch_norm.running_mean', 'batch_norm.running_var', 'batch_norm.weight'):
            ks = [k for k in pre if k.endswith(name)]
            for k in ks:
                v = hyp[k].float()
                print(f'   {name:24s} min={v.min().item():+.3e} max={v.max().item():+.3e} '
                      f'mean={v.mean().item():+.3e}')
        # running_var / running_mean theo tung khop (kenh BN = 17 khop)
        kv = [k for k in pre if k.endswith('batch_norm.running_var')]
        km = [k for k in pre if k.endswith('batch_norm.running_mean')]
        if kv and km:
            var, mean = hyp[kv[0]].float(), hyp[km[0]].float()
            print('   BN theo khop (var / mean):')
            row = []
            for j in range(var.numel()):
                flag = ' <-- NO' if var[j].item() > 1e3 else ''
                row.append(f'      {j:2d} {J17[j]:10s} var={var[j].item():.3e} mean={mean[j].item():+.3e}{flag}')
            print('\n'.join(row))
        # ma tran sieu canh part / body
        for suf, lab, A0 in (('mixer.p_adj', 'p_adj (part)', INIT_P), ('mixer.b_adj', 'b_adj (body)', INIT_B)):
            ks = [k for k in pre if k.endswith(suf)]
            for k in ks:
                adj_report(lab, hyp[k], A0)

    print('-- Nhanh khoi tao 0 cua query SMPL (lon hon 0 nhieu -> dang duoc dung)')
    for pat, lab in (('bone_proj.weight', 'bone_proj (v4, concat ctx cha/con)'),
                     ('bone_geo_proj.2.weight', 'bone_geo_proj cuoi (v5, huong + do dai xuong)'),
                     ('jf_proj.weight', '[B] jf_proj'),
                     ('depth_proj.weight', '[C] depth_proj')):
        ks = [k for k in sd if k.endswith(pat)]
        for k in ks:
            print(f'   {lab:45s} {stat(sd[k])}')
        if not ks:
            print(f'   {lab:45s} (khong co)')

    # tham chieu: do lon trong so mot lop binh thuong de so sanh
    ref = [k for k in sd if k.endswith('per_joint_head.layers.0.weight')]
    for k in ref:
        print(f'   {"tham chieu: per_joint_head lop 0":45s} {stat(sd[k])}')
    print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    INIT_P, INIT_B = init_adj()
    for p in sys.argv[1:]:
        report(p)
