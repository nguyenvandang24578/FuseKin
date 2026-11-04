"""
lib/utils/img_lifter_tools.py -- Tien ich chung cho main/train_img_lifter.py va main/eval_img_lifter.py (ImgLifter).

eval_img_lifter: tren loader 3DPW (dau vao OpenPose / HHRNet nhu benchmark):
  OpenPose -> Refiner 2D (dong bang) -> ImgLifter (K gia thuyet, S buoc DDIM) -> gop 'mean' / 'jpma'
  GT 3D = H36M regressor x mesh GT, root-relative (cung dinh nghia mesh voi muc tieu train).
  Bao MPJPE17 / PA-MPJPE17 (mm) cho mean, jpma, oracle min-K (chi do tiem nang) va (tuy chon) lifter cu voi OpenPose.
  ablate_img=True: dat dac trung anh cua lifter (f8, feat_j) = 0, giu 2D Refiner -> do lifter co dung anh khong.
"""
import numpy as np
import torch

from core.config import cfg
from models.img_lifter import ImgLifter, aggregate


def imglift_arch():
    L = cfg.IMGLIFT
    return {'dim': int(L.embed_dim), 'depth': int(L.depth), 'num_heads': int(L.num_heads),
            'mlp_ratio': float(L.mlp_ratio), 'drop_path': float(L.drop_path), 'timesteps': int(L.timesteps),
            'scale': float(L.scale), 'feat_j_ch': int(cfg.REFINE2D.deconv_ch)}


def build_img_lifter(arch=None):
    a = dict(arch or imglift_arch())
    return ImgLifter(num_joints=17, dim=a['dim'], depth=a['depth'], num_heads=a['num_heads'],
                     mlp_ratio=a['mlp_ratio'], drop_path=a['drop_path'], timesteps=a['timesteps'],
                     scale=a['scale'], feat_j_ch=a['feat_j_ch'])


def load_img_lifter(path, dev):
    """Nap checkpoint train_img_lifter: 'model_state' = trong so EMA (dung de danh gia), nap strict."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = build_img_lifter(ck['imglift_arch'])
    model.load_state_dict(ck['model_state'], strict=True)
    print(f'===> Nap ImgLifter {path} (epoch {ck.get("epoch")}, refiner {ck.get("refiner_ckpt")})')
    return model.to(dev).eval(), ck


def load_frozen_refiner(path, dev):
    from utils.refiner2d_tools import load_refiner
    model, _ = load_refiner(path, dev)
    for p in model.parameters():
        p.requires_grad = False
    return model.eval()


@torch.no_grad()
def refiner_inputs(refiner, img, j2, jm):
    """j2 (B,17,2) [-1,1], jm (B,17) -> xy (B,17,2), log_b (B,17), f8 (B,2048,8,8), feat_j (B,17,C)."""
    out = refiner(img, j2 * (jm > 0).unsqueeze(-1).float(), jm.unsqueeze(-1), return_feats=True)
    return out['xy'].float(), out['log_b'].float(), out['f8'].float(), out['feat_j'].float()


def pa_errors(pred, gt):
    """Procrustes (scale + xoay + dich) tung mau, giong compute_similarity_transform (SPIN). (B,J,3) met -> (B,) mm."""
    X = pred.double() - pred.double().mean(1, keepdim=True)
    Y = gt.double() - gt.double().mean(1, keepdim=True)
    var_x = (X ** 2).sum((1, 2)).clamp_min(1e-12)
    Km = X.transpose(1, 2) @ Y                                              # (B,3,3)
    U, _, Vh = torch.linalg.svd(Km)
    V = Vh.transpose(1, 2)
    Z = torch.eye(3, dtype=X.dtype, device=X.device).repeat(X.shape[0], 1, 1)
    Z[:, -1, -1] = torch.sign(torch.det(U @ V.transpose(1, 2)))
    R = V @ Z @ U.transpose(1, 2)
    s = torch.einsum('bii->b', R @ Km) / var_x
    aligned = s[:, None, None] * (X @ R.transpose(1, 2)) + gt.double().mean(1, keepdim=True)
    return (torch.norm(aligned - gt.double(), dim=-1).mean(-1) * 1000).float()


@torch.no_grad()
def eval_img_lifter(lifter, refiner, loader, dev, regressor, K=10, S=4, eta=1.0, jpma_max_b=3.0,
                    ablate_img=False, old_lifter=None, max_samples=0, seed=0):
    from models.ARTS import diff_lift
    lifter.eval()
    reg = torch.as_tensor(regressor, dtype=torch.float32, device=dev)
    acc = {}

    def add(name, pred, gt17):
        e = (torch.norm(pred - gt17, dim=-1).mean(-1) * 1000).cpu().numpy()
        p = pa_errors(pred, gt17).cpu().numpy()
        acc.setdefault(name, {'mpjpe': [], 'pa': []})
        acc[name]['mpjpe'].append(e)
        acc[name]['pa'].append(p)

    seen = 0
    for bi, (inputs, targets, meta) in enumerate(loader):
        if max_samples and seen >= max_samples:
            break
        img = inputs['img'].to(dev).float()
        B = img.shape[0]
        j2 = inputs['joints'].to(dev).float()[..., :2]
        jm = inputs['joints_mask'].to(dev).float().reshape(B, 17)
        cam = inputs['cam_bbox'].to(dev).float()
        gt17 = torch.einsum('jk,bkc->bjc', reg, targets['smpl_mesh_cam'].to(dev).float())
        gt17 = gt17 - gt17[:, 0:1]

        xy, log_b, f8, fj = refiner_inputs(refiner, img, j2, jm)
        if ablate_img:
            f8, fj = torch.zeros_like(f8), torch.zeros_like(fj)
        cond7 = ImgLifter.make_cond7(xy, log_b, jm > 0, cam)
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed) + bi)                                     # nhieu co dinh -> lap lai duoc
        hyp = lifter.sample(cond7, fj, f8, K=K, S=S, eta=eta, generator=gen)   # (B,K,17,3)
        add('mean', aggregate(hyp, xy, log_b, 'mean'), gt17)
        add('jpma', aggregate(hyp, xy, log_b, 'jpma', jpma_max_b), gt17)
        if K > 1:                                                           # oracle: gia thuyet gan GT nhat (CHI do tiem nang)
            e_k = torch.norm(hyp - gt17[:, None], dim=-1).mean(-1)          # (B,K)
            best = hyp[torch.arange(B, device=dev), e_k.argmin(1)]
            add('oracle_minK', best, gt17)
        if old_lifter is not None:
            add('old_lifter_op', diff_lift(old_lifter, j2, jm.unsqueeze(-1), cam), gt17)
        seen += B
    return {'n': seen, 'K': K, 'S': S, 'ablate_img': bool(ablate_img),
            'res': {k: {'mpjpe17': float(np.concatenate(v['mpjpe']).mean()),
                        'pa_mpjpe17': float(np.concatenate(v['pa']).mean())} for k, v in acc.items()}}


def format_eval(name, r):
    s = f'[{name}] n={r["n"]} K={r["K"]} S={r["S"]}' + (' | ABLATE anh (f8, feat_j = 0)' if r['ablate_img'] else '')
    return s + '\n' + '\n'.join(f'  {k:<14} MPJPE17 {v["mpjpe17"]:7.2f} | PA {v["pa_mpjpe17"]:6.2f}'
                                for k, v in r['res'].items())
