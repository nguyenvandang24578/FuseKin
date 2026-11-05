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
    # Chi 17 khop (Refiner day co them diem sau 17 khop; ImgLifter 17 diem chua dung diem day)
    return (out['xy'].float()[:, :17], out['log_b'].float()[:, :17], out['f8'].float(),
            out['feat_j'].float()[:, :17])


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


def reproj_errors(pred, g2, valid):
    """Do khop 2D cua pose 3D: fit weak-perspective (scale + dich, least squares tren khop valid) tu XY cua pred
    sang GT 2D g2 ([-1,1] crop), roi do loi tung khop. pred (B,17,3); g2 (B,17,2); valid (B,17) -> (B,) px crop 256.
    Chi do HINH DANG pose so voi 2D (s, t toi uu cho tung mau), khong do vi tri / do sau tuyet doi."""
    w = valid.float()
    ws = w.sum(1).clamp_min(1.0)
    P = pred[..., :2].float()
    Pm = (w[..., None] * P).sum(1, keepdim=True) / ws[:, None, None]
    gm = (w[..., None] * g2).sum(1, keepdim=True) / ws[:, None, None]
    Pc, gc = P - Pm, g2 - gm
    s = (w * (Pc * gc).sum(-1)).sum(1) / (w * (Pc ** 2).sum(-1)).sum(1).clamp_min(1e-8)
    proj = s[:, None, None] * Pc + gm
    e = torch.norm(proj - g2, dim=-1) * 128.0                               # [-1,1] -> px crop 256
    return (e * w).sum(1) / ws


@torch.no_grad()
def eval_img_lifter(lifter, refiner, loader, dev, regressor, K=10, S=4, eta=1.0, jpma_max_b=3.0,
                    ablate_img=False, old_lifter=None, max_samples=0, seed=0):
    from models.ARTS import diff_lift
    lifter.eval()
    reg = torch.as_tensor(regressor, dtype=torch.float32, device=dev)
    acc = {}
    hm = torch.tensor([cfg.output_hm_shape[2], cfg.output_hm_shape[1]], dtype=torch.float32, device=dev)

    def add(name, pred, gt17, g2=None, gin=None):
        e = (torch.norm(pred - gt17, dim=-1).mean(-1) * 1000).cpu().numpy()
        p = pa_errors(pred, gt17).cpu().numpy()
        acc.setdefault(name, {'mpjpe': [], 'pa': [], 'reproj': []})
        acc[name]['mpjpe'].append(e)
        acc[name]['pa'].append(p)
        if g2 is not None:                                                  # (chan doan 1) lech pha 3D - 2D
            acc[name]['reproj'].append(reproj_errors(pred, g2, gin).cpu().numpy())

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
        g2 = gin = None
        if 'orig_joint_img' in targets:                                     # can cfg.DATASET.pw3d_test_gt2d
            g2 = targets['orig_joint_img'].to(dev).float()[..., :2] / hm * 2 - 1
            gin = meta['orig_joint_trunc'].to(dev).float().reshape(B, 17) > 0
            add('gt3d_san', gt17, gt17, g2, gin)       # san: GT 3D (weak-perspective) vs GT 2D (phoi canh)

        xy, log_b, f8, fj = refiner_inputs(refiner, img, j2, jm)
        if ablate_img:
            f8, fj = torch.zeros_like(f8), torch.zeros_like(fj)
        cond7 = ImgLifter.make_cond7(xy, log_b, jm > 0, cam)
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed) + bi)                                     # nhieu co dinh -> lap lai duoc
        hyp = lifter.sample(cond7, fj, f8, K=K, S=S, eta=eta, generator=gen)   # (B,K,17,3)
        add('mean', aggregate(hyp, xy, log_b, 'mean'), gt17, g2, gin)
        add('jpma', aggregate(hyp, xy, log_b, 'jpma', jpma_max_b), gt17, g2, gin)
        if g2 is not None:                                                  # tung gia thuyet le (truoc khi gop)
            add('hyp_don_le', hyp[:, 0], gt17, g2, gin)
        if K > 1:                                                           # oracle: gia thuyet gan GT nhat (CHI do tiem nang)
            e_k = torch.norm(hyp - gt17[:, None], dim=-1).mean(-1)          # (B,K)
            best = hyp[torch.arange(B, device=dev), e_k.argmin(1)]
            add('oracle_minK', best, gt17, g2, gin)
        if old_lifter is not None:
            add('old_lifter_op', diff_lift(old_lifter, j2, jm.unsqueeze(-1), cam), gt17, g2, gin)
        seen += B

    def summ(v):
        d = {'mpjpe17': float(np.concatenate(v['mpjpe']).mean()), 'pa_mpjpe17': float(np.concatenate(v['pa']).mean())}
        if v['reproj']:
            d['reproj_px'] = float(np.concatenate(v['reproj']).mean())
        return d
    return {'n': seen, 'K': K, 'S': S, 'ablate_img': bool(ablate_img), 'res': {k: summ(v) for k, v in acc.items()}}


def format_eval(name, r):
    s = f'[{name}] n={r["n"]} K={r["K"]} S={r["S"]}' + (' | ABLATE anh (f8, feat_j = 0)' if r['ablate_img'] else '')
    lines = [s]
    for k, v in r['res'].items():
        rp = f' | lech 2D {v["reproj_px"]:5.2f} px' if 'reproj_px' in v else ''
        lines.append(f'  {k:<14} MPJPE17 {v["mpjpe17"]:7.2f} | PA {v["pa_mpjpe17"]:6.2f}{rp}')
    if any('reproj_px' in v for v in r['res'].values()):
        lines.append('  (lech 2D: fit scale + dich toi uu tu XY pose 3D sang GT 2D mesh, px crop 256; gt3d_san = muc san '
                     'do chieu phoi canh)')
    return '\n'.join(lines)
