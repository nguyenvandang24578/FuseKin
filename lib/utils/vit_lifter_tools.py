"""
lib/utils/vit_lifter_tools.py (2026-11-18) -- Tien ich cho main/train_vit_lifter.py va main/eval_vit_lifter.py.

eval_vit_lifter: tren loader 3DPW voi 2D ViTPose (DATASET.pw3d_2d_source = 'halpe26', pw3d_halpe_suffix):
  ViTPose -> ResNet-50 SPIN dong bang -> VitLifter (K gia thuyet, S buoc DDIM) -> gop 'mean'.
  GT 3D = H36M regressor x mesh GT, root-relative. Bao MPJPE17 / PA-MPJPE17 (mm), oracle min-K (chi do tiem nang).
Dung lai pa_errors / reproj_errors / format_eval cua img_lifter_tools (khong sua file do).
"""
import numpy as np
import torch

from core.config import cfg
from models.img_lifter import aggregate
from models.vit_lifter import VitLifter, FrozenSpinEncoder, vit_inputs, F16_CH
from utils.img_lifter_tools import pa_errors, reproj_errors, format_eval  # noqa: F401 (format_eval dung o script)


def vitlift_arch():
    L = cfg.IMGLIFT
    return {'dim': int(L.embed_dim), 'depth': int(L.depth), 'num_heads': int(L.num_heads),
            'mlp_ratio': float(L.mlp_ratio), 'drop_path': float(L.drop_path), 'timesteps': int(L.timesteps),
            'scale': float(L.scale), 'feat_j_ch': F16_CH, 'markov': bool(L.markov_layer),
            'markov_gate_b': bool(L.markov_gate_b), 'markov_iters': int(L.markov_iters),
            'markov_gate_positive': bool(L.markov_gate_positive), 'bone_paper': bool(L.bone_paper),
            'markov_msg_norm': bool(L.markov_msg_norm), 'glob_img': bool(L.vit_glob_img)}


def build_vit_lifter(arch=None):
    return VitLifter(num_joints=17, **dict(arch or vitlift_arch()))


def build_encoder(pretrained, dev):
    return FrozenSpinEncoder(pretrained).to(dev).eval()


def load_vit_lifter(path, dev):
    """Nap checkpoint train_vit_lifter: 'model_state' = trong so EMA, strict. Tra (lifter, encoder, ck)."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = build_vit_lifter(ck['vitlift_arch'])
    model.load_state_dict(ck['model_state'], strict=True)
    print(f'===> Nap VitLifter {path} (epoch {ck.get("epoch")}, encoder {ck.get("vit_enc_pretrained")})')
    enc = build_encoder(ck.get('vit_enc_pretrained', 'spin'), dev)
    return model.to(dev).eval(), enc, ck


@torch.no_grad()
def eval_vit_lifter(lifter, encoder, loader, dev, regressor, K=10, S=4, eta=1.0, logb_eps=0.01,
                    max_samples=0, seed=0):
    lifter.eval()
    reg = torch.as_tensor(regressor, dtype=torch.float32, device=dev)
    hm = torch.tensor([cfg.output_hm_shape[2], cfg.output_hm_shape[1]], dtype=torch.float32, device=dev)
    acc = {}
    cur = {'jm': None}

    def add(name, pred, gt17, g2=None, gin=None):
        e = (torch.norm(pred - gt17, dim=-1).mean(-1) * 1000).cpu().numpy()
        p = pa_errors(pred, gt17).cpu().numpy()
        a = acc.setdefault(name, {'mpjpe': [], 'pa': [], 'reproj': [], 'miss': [0.0, 0], 'vis': [0.0, 0]})
        a['mpjpe'].append(e)
        a['pa'].append(p)
        ej = torch.norm(pred - gt17, dim=-1) * 1000                         # (B,17) mm
        miss = cur['jm'] <= 0                                               # khop ViTPose conf <= nguong
        a['miss'][0] += float(ej[miss].sum()); a['miss'][1] += int(miss.sum())
        a['vis'][0] += float(ej[~miss].sum()); a['vis'][1] += int((~miss).sum())
        if g2 is not None:
            a['reproj'].append(reproj_errors(pred, g2, gin).cpu().numpy())

    seen = 0
    for bi, (inputs, targets, meta) in enumerate(loader):
        if max_samples and seen >= max_samples:
            break
        img = inputs['img'].to(dev).float()
        B = img.shape[0]
        joints = inputs['joints'].to(dev).float()
        jm = inputs['joints_mask'].to(dev).float().reshape(B, 17)
        cam = inputs['cam_bbox'].to(dev).float()
        gt17 = torch.einsum('jk,bkc->bjc', reg, targets['smpl_mesh_cam'].to(dev).float())
        gt17 = gt17 - gt17[:, 0:1]
        g2 = gin = None
        if 'orig_joint_img' in targets:                                     # can cfg.DATASET.pw3d_test_gt2d
            g2 = targets['orig_joint_img'].to(dev).float()[..., :2] / hm * 2 - 1
            gin = meta['orig_joint_trunc'].to(dev).float().reshape(B, 17) > 0
        cur['jm'] = jm
        cond7, fj, f8, xy, log_b = vit_inputs(encoder, img, joints, jm, cam, logb_eps)
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed) + bi)                                     # nhieu co dinh -> lap lai duoc
        hyp = lifter.sample(cond7, fj, f8, K=K, S=S, eta=eta, generator=gen)   # (B,K,17,3)
        add('mean', aggregate(hyp, xy, log_b, 'mean'), gt17, g2, gin)
        if K > 1:                                                           # oracle: CHI do tiem nang
            e_k = torch.norm(hyp - gt17[:, None], dim=-1).mean(-1)
            add('oracle_minK', hyp[torch.arange(B, device=dev), e_k.argmin(1)], gt17, g2, gin)
        if g2 is not None:                                                  # loi 2D ViTPose so GT 2D mesh (px crop 256)
            e2 = torch.norm(xy - g2, dim=-1) * 128.0
            m2 = (jm > 0) & gin
            a = acc.setdefault('_vitpose2d', [0.0, 0])
            a[0] += float(e2[m2].sum()); a[1] += int(m2.sum())
        seen += B

    def summ(v):
        d = {'mpjpe17': float(np.concatenate(v['mpjpe']).mean()), 'pa_mpjpe17': float(np.concatenate(v['pa']).mean())}
        if v['reproj']:
            d['reproj_px'] = float(np.concatenate(v['reproj']).mean())
        d['mpjpe_miss'] = v['miss'][0] / max(v['miss'][1], 1)
        d['mpjpe_vis'] = v['vis'][0] / max(v['vis'][1], 1)
        d['miss_ratio'] = v['miss'][1] / max(v['miss'][1] + v['vis'][1], 1)
        return d
    v2 = acc.pop('_vitpose2d', None)
    return {'n': seen, 'K': K, 'S': S, 'ablate_img': False, 'n_skip_invalid': 0,
            'vitpose_2d_px': (v2[0] / max(v2[1], 1)) if v2 else None,
            'res': {k: summ(v) for k, v in acc.items()}}


def format_vit_eval(name, r):
    s = format_eval(name, r)
    if r.get('vitpose_2d_px') is not None:
        s += f'\n  (loi 2D ViTPose so GT 2D mesh, khop conf > nguong: {r["vitpose_2d_px"]:.2f} px crop 256)'
    return s
