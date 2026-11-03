"""
lib/utils/refiner2d_tools.py -- Tien ich chung cho main/train_refiner2d.py va main/eval_refiner2d.py (Refiner 2D).

eval_refiner: tren loader 3DPW (dau vao OpenPose / HHRNet nhu benchmark, can cfg.DATASET.pw3d_test_gt2d = True):
  (a) chat luong 2D tren 12 khop tay chan so voi GT 2D (H36M regressor x mesh GT, chieu vao crop):
      loi / kich thuoc nguoi, % khop xa (> far_ratio x kich thuoc nguoi), % khop bo, % mau co >= 1 khop xa,
      so khop SUA duoc (op xa -> refine khong xa) / LAM HONG (op khong xa -> refine xa).
  (b) tuy chon lifter dong bang: MPJPE17 (mm, root-relative) voi 3 bien the 2D:
      op          = 2D detector nguyen ban (nhu hien tai)
      refine      = toa do Refiner, GIU mask detector
      refine_fill = nhu refine + dien khop tay chan detector bo sot neu Refiner du tin (b <= fill_max_b o luoi)
Khung toa do: [-1, 1] trong crop (ti le loi / kich thuoc nguoi khong doi giua khung crop va anh goc).
"""
import numpy as np
import torch

from core.config import cfg
from models.refiner2d import Refiner2D, far_counts, LIMB_IDX, LIMB_NAMES

VARIANTS = ('op', 'refine', 'refine_fill')


def refine2d_arch():
    R = cfg.REFINE2D
    return {'num_joints': 17, 'deconv_ch': int(R.deconv_ch), 'heat_sigma': float(R.heat_sigma),
            'bb_bn_eval': bool(R.bb_bn_eval)}


def build_refiner(pretrained, arch=None):
    a = dict(arch or refine2d_arch())
    return Refiner2D(num_joints=a['num_joints'], deconv_ch=a['deconv_ch'], heat_sigma=a['heat_sigma'],
                     pretrained=pretrained, bb_bn_eval=a['bb_bn_eval'])


def load_refiner(path, dev):
    """Nap checkpoint train_refiner2d (kien truc lay tu ckpt['refine2d_arch'], nap strict)."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = build_refiner(pretrained='', arch=ck['refine2d_arch'])
    model.load_state_dict(ck['model_state'], strict=True)
    print(f'===> Nap Refiner 2D {path} (epoch {ck.get("epoch")}, kien truc {ck["refine2d_arch"]})')
    return model.to(dev).eval(), ck


def _acc_new():
    L = len(LIMB_IDX)
    return {'e': np.zeros(L), 'det': np.zeros(L), 'far': np.zeros(L), 'miss': np.zeros(L), 'gt': np.zeros(L),
            'bad_samples': 0, 'n': 0}


def _acc_add(acc, c):
    for k in ('det', 'far', 'miss', 'gt'):
        acc[k] += c[k].sum(0).cpu().numpy()
    acc['e'] += (c['e'] * c['det']).sum(0).cpu().numpy()
    acc['bad_samples'] += int((c['far'].sum(1) > 0).sum())
    acc['n'] += int(c['e'].shape[0])


@torch.no_grad()
def eval_refiner(model, loader, dev, far_ratio=0.2, lifter=None, regressor=None, fill_max_b=3.0,
                 max_samples=0):
    """Tra dict ket qua (xem docstring module). regressor: (17, 6890) H36M regressor (can khi co lifter)."""
    from models.ARTS import diff_lift
    model.eval()
    hm = torch.tensor([cfg.output_hm_shape[2], cfg.output_hm_shape[1]], dtype=torch.float32, device=dev)
    limb = torch.zeros(17, dtype=torch.bool, device=dev)
    limb[list(LIMB_IDX)] = True
    acc = {v: _acc_new() for v in VARIANTS}
    fixed = np.zeros(len(LIMB_IDX))
    broke = np.zeros(len(LIMB_IDX))
    n_fill, n_missing = 0.0, 0.0
    lift = {v: [] for v in VARIANTS}
    logb_all, loge_all = [], []
    reg = torch.as_tensor(regressor, dtype=torch.float32, device=dev) if regressor is not None else None
    seen = 0
    for inputs, targets, meta in loader:
        if max_samples and seen >= max_samples:
            break
        img = inputs['img'].to(dev).float()
        B = img.shape[0]
        j2 = inputs['joints'].to(dev).float()[..., :2]                             # (B,17,2) [-1,1]
        jm = inputs['joints_mask'].to(dev).float().reshape(B, 17)
        det = jm > 0
        g2 = targets['orig_joint_img'].to(dev).float()[..., :2] / hm * 2 - 1      # GT 2D [-1,1]
        gin = meta['orig_joint_trunc'].to(dev).float().reshape(B, 17)
        if 'orig_joint_valid' in meta:
            gin = gin * meta['orig_joint_valid'].to(dev).float().reshape(B, 17)
        gin = gin > 0

        out = model(img, j2, jm.unsqueeze(-1))
        rx = out['xy'].float()
        b = torch.exp(out['log_b'].float())
        fill = det | (limb[None] & (b <= fill_max_b))
        n_fill += float((fill & ~det & limb[None]).sum())
        n_missing += float((~det & limb[None]).sum())
        cand = {'op': (j2, det), 'refine': (rx, det), 'refine_fill': (rx, fill)}
        cnt = {}
        for v, (xy, m) in cand.items():
            cnt[v] = far_counts(xy, m, g2, gin, far_ratio)
            _acc_add(acc[v], cnt[v])
        both = cnt['op']['det'] * cnt['refine']['det']
        fixed += (cnt['op']['far'] * (1 - cnt['refine']['far']) * both).sum(0).cpu().numpy()
        broke += ((1 - cnt['op']['far']) * cnt['refine']['far'] * both).sum(0).cpu().numpy()
        # tuong quan do bat dinh / loi that (khop co dau vao, GT trong khung, 12 khop tay chan)
        sel = (det & gin)[:, list(LIMB_IDX)]
        e_grid = torch.norm(rx - g2, dim=-1)[:, list(LIMB_IDX)] * hm[0] / 2
        logb_all.append(out['log_b'][:, list(LIMB_IDX)][sel].float().cpu().numpy())
        loge_all.append(torch.log(e_grid[sel].clamp_min(1e-3)).cpu().numpy())

        if lifter is not None:
            cam = inputs['cam_bbox'].to(dev).float()
            gt17 = torch.einsum('jk,bkc->bjc', reg, targets['smpl_mesh_cam'].to(dev).float())
            gt17 = gt17 - gt17[:, 0:1]
            for v, (xy, m) in cand.items():
                mm = m.float().unsqueeze(-1)
                p3 = diff_lift(lifter, xy * mm, mm, cam)
                lift[v].append((torch.norm(p3 - gt17, dim=-1).mean(-1) * 1000).cpu().numpy())
        seen += B

    def summ(a):
        d = lambda x, y: float(x.sum() / max(y.sum(), 1))
        return {'n': a['n'], 'err_pct': 100 * d(a['e'], a['det']), 'far_pct': 100 * d(a['far'], a['det']),
                'miss_pct': 100 * d(a['miss'], a['gt']), 'samples_far_pct': 100 * a['bad_samples'] / max(a['n'], 1),
                'per_joint_far_pct': {LIMB_NAMES[j]: 100 * float(a['far'][j] / max(a['det'][j], 1))
                                      for j in range(len(LIMB_IDX))}}
    res = {'n': seen, 'far_ratio': far_ratio, 'd2': {v: summ(acc[v]) for v in VARIANTS},
           'fixed_joints': int(fixed.sum()), 'broken_joints': int(broke.sum()),
           'fill_pct_of_missing': 100 * n_fill / max(n_missing, 1), 'fill_max_b': fill_max_b}
    lb, le = np.concatenate(logb_all), np.concatenate(loge_all)
    res['corr_logb_logerr'] = float(np.corrcoef(lb, le)[0, 1]) if lb.size > 2 else float('nan')
    if lifter is not None:
        res['lift_mpjpe17'] = {v: float(np.concatenate(lift[v]).mean()) for v in VARIANTS}
    return res


def format_result(name, r):
    lines = [f'[{name}] n={r["n"]} | khop xa = loi > {r["far_ratio"]} x kich thuoc nguoi (12 khop tay chan)']
    lines.append(f'  {"2D":<12} | {"loi %":>6} | {"xa %":>6} | {"bo %":>6} | {"mau co khop xa %":>16}')
    for v, s in r['d2'].items():
        lines.append(f'  {v:<12} | {s["err_pct"]:6.2f} | {s["far_pct"]:6.2f} | {s["miss_pct"]:6.2f} | '
                     f'{s["samples_far_pct"]:16.2f}')
    lines.append(f'  op -> refine: sua {r["fixed_joints"]} khop xa, lam hong {r["broken_joints"]} khop | '
                 f'dien {r["fill_pct_of_missing"]:.1f}% khop bo sot (b <= {r["fill_max_b"]}) | '
                 f'corr(log b, log loi) {r["corr_logb_logerr"]:.3f}')
    if 'lift_mpjpe17' in r:
        lines.append('  lifter MPJPE17 (mm): ' + ' | '.join(f'{v} {x:.2f}' for v, x in r['lift_mpjpe17'].items()))
    return '\n'.join(lines)
