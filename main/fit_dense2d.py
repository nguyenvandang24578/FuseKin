"""
main/fit_dense2d.py -- (2026-11-08) Fitting LUC TEST (khong train): tach pose / shape.
  Pose : ImgLifter (K, S nhu Student) -> M (29 x 17) -> IK HybrIK (beta 0, twist 0)  = khoi tao (baseline C, §19)
  Fit  : toi uu theta (goc SMPL), beta, t (camera) de
           (a) 17 khop H36M [+ 48 dinh mesh] chieu xuong khop voi 2D Refiner day, trong so 1 / b^2 (Huber, px crop 256)
           (b) 17 khop 3D cua mesh khong roi pose lifter (prior 3D, mm)
           (c) beta gan 0, theta gan khoi tao
       Bien the: 'j17' = chi 17 khop 2D | 'dense' = 17 khop + 48 dinh (do dong gop cua 2D day cho shape / MPVPE).
  Camera: phoi canh ao cua mo hinh (cfg.focal, cfg.princpt, crop cfg.input_img_shape) -> giong _project_h36m.
Do: MPJPE / PA 14 khop (H36M regressor x mesh, root Pelvis H36M) + MPVPE (tru Pelvis SMPL) nhu ik_upper_bound.py,
cung loi chieu 2D TB (px) so voi 2D Refiner va |beta| TB.

LUU Y giao thuc: CHON trong so (w2d, w3d, ...) tren 3dpw-val (lifter / Student khong dung tap nay), KHONG tren test.
Chay (thu nhanh, chon trong so tren val):
  python main/fit_dense2d.py --cfg config/train_img_lifter_dense48_3dpw_markov.yaml \
    --ckpt experiment/img_lifter_dense48_3dpw_markov/checkpoint/last.pth.tar \
    --sets 3dpw-val --max_samples 2000 --gpu 0
Chay chinh (sau khi chot trong so):
  ... --sets 3dpw,3dpw-oc,3dpw-pc,3dpw-crowd --ocpc_mode test
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse

parser = argparse.ArgumentParser(description='Fitting 2D day luc test tu baseline IK')
parser.add_argument('--cfg', type=str, required=True, help='yml da dung de train ImgLifter')
parser.add_argument('--ckpt', type=str, required=True, help='checkpoint ImgLifter')
parser.add_argument('--refiner_ckpt', type=str, default='', help='rong = refiner luu trong ckpt ImgLifter')
parser.add_argument('--M', type=str, default='data_final/h36m17_to_smpl29.npy')
parser.add_argument('--dense_json', type=str, default='config/dense48_fps.json', help="json {'vertex_ids': [...]}")
parser.add_argument('--sets', type=str, default='3dpw-val')
parser.add_argument('--ocpc_mode', type=str, default='test', choices=['all', 'test'])
parser.add_argument('--variants', type=str, default='j17,dense')
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--S', type=int, default=2)
parser.add_argument('--eta', type=float, default=1.0)
# toi uu
parser.add_argument('--iters_cam', type=int, default=30, help='giai doan 1: chi t + huong toan cuc')
parser.add_argument('--iters', type=int, default=60, help='giai doan 2: theta + beta + t')
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--w2d', type=float, default=1.0, help='Huber px (crop 256), trong so 1/b^2 chuan hoa TB = 1')
parser.add_argument('--huber_px', type=float, default=5.0)
parser.add_argument('--w3d', type=float, default=0.1, help='prior 3D: |khop mesh - pose lifter| (mm)')
parser.add_argument('--wbeta', type=float, default=1.0, help='TB beta^2')
parser.add_argument('--wpose', type=float, default=50.0, help='TB (theta - theta0)^2 (rad^2), khop than 1..21')
parser.add_argument('--b_min', type=float, default=0.25, help='kep duoi b truoc khi tinh 1/b^2 (nhu jpma)')
parser.add_argument('--no_view_rot', action='store_true',
                    help='tat xoay he camera that -> camera ao cua crop (tu cam_bbox) truoc khi chieu')
parser.add_argument('--max_samples', type=int, default=0, help='0 = toan bo tap')
parser.add_argument('--batch_size', type=int, default=0, help='0 = cfg.TEST.batch_size')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import copy
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from core.config import cfg, update_config
update_config(args.cfg)
assert cfg.DATASET.pw3d_2d_source == 'openpose', 'Benchmark: dau vao 3DPW la OpenPose / HHRNet'

import __init_path  # noqa: F401
from utils.jotr_dataset import get_test_dataset
from utils.img_lifter_tools import load_img_lifter, load_frozen_refiner, refiner_inputs, lifter_det, pa_errors
from models.img_lifter import ImgLifter, aggregate
from models.hybrik_smpl import HybrIKSMPL, rotmat_to_aa_safe

FOC = torch.tensor([float(cfg.focal[0]), float(cfg.focal[1])])
PP = torch.tensor([float(cfg.princpt[0]), float(cfg.princpt[1])])
IMG_WH = torch.tensor([float(cfg.input_img_shape[1]), float(cfg.input_img_shape[0])])


def view_rot_torch(cam_bbox):
    """Ban torch cua utils.cam_bbox.view_rot: cam_bbox (B,3) = (dx, dy, s) -> R (B,3,3) dua truc z camera AO
    (nhin thang vao tam bbox) thanh tia camera that d = (dx, dy, 1). pose_that = R @ pose_ao.
    Dung (1 - c) / s^2 = 1 / (1 + c) cho on dinh."""
    d = torch.cat([cam_bbox[:, :2], torch.ones_like(cam_bbox[:, :1])], 1)
    d = d / d.norm(dim=1, keepdim=True)
    B = d.shape[0]
    K = torch.zeros(B, 3, 3, device=d.device, dtype=d.dtype)
    v0, v1 = -d[:, 1], d[:, 0]                                              # v = z x d = (-dy, dx, 0)
    K[:, 0, 2], K[:, 1, 2] = v1, -v0
    K[:, 2, 0], K[:, 2, 1] = -v1, v0
    I = torch.eye(3, device=d.device, dtype=d.dtype)[None]
    return I + K + K @ K / (1.0 + d[:, 2])[:, None, None]


def project(P, t, R=None):
    """P (B,N,3) met root-relative (he camera THAT), t (B,3) -> toa do [-1,1] crop (B,N,2), giong _project_h36m.
    R (B,3,3) (view_rot): dua P ve he camera AO cua crop (P_ao = R^T P) truoc khi chieu."""
    if R is not None:
        P = P @ R                                                           # hang: p_ao^T = p^T R
    X = P + t[:, None]
    z = X[..., 2:3].clamp_min(1e-2)
    px = X[..., :2] / z * FOC.to(P) + PP.to(P)
    return px / IMG_WH.to(P) * 2 - 1


def init_trans(J, xy, w):
    """Nghiem weak-perspective co trong so: u = s * X + o (px, so voi tam anh). J (B,17,3), xy (B,17,2) [-1,1],
    w (B,17) -> t (B,3) sao cho chieu phoi canh ~ khop 2D."""
    B = J.shape[0]
    u = (xy + 1) / 2 * IMG_WH.to(J) - PP.to(J)                             # (B,17,2) px so voi tam
    A = torch.zeros(B, 34, 3, device=J.device, dtype=J.dtype)
    A[:, :17, 0], A[:, :17, 1] = J[..., 0], 1.0
    A[:, 17:, 0], A[:, 17:, 2] = J[..., 1], 1.0
    y = torch.cat([u[..., 0], u[..., 1]], 1)                                # (B,34)
    sw = torch.cat([w, w], 1).sqrt()
    sol = torch.linalg.lstsq(A * sw[..., None], (y * sw)[..., None]).solution[..., 0]   # (B,3) s, ox, oy
    s = sol[:, 0].clamp_min(1.0)                                            # px / met
    f = FOC.to(J).mean()
    return torch.stack([sol[:, 1] / s, sol[:, 2] / s, f / s], 1)


def huber(e, d):
    return torch.where(e < d, 0.5 * e * e / d, e - 0.5 * d)


class Fitter:
    def __init__(self, layer, reg17, dense_ids):
        self.layer, self.reg17, self.dense_ids = layer, reg17, dense_ids

    def mesh(self, pose, beta):
        """pose (B,24,3) axis-angle (khop 22, 23 = 0), beta (B,10) -> verts root-relative Pelvis H36M, J17."""
        p = torch.cat([pose[:, :22], torch.zeros_like(pose[:, 22:])], 1).reshape(pose.shape[0], 72)
        verts, _ = self.layer(p, beta)
        J17 = torch.einsum('jv,bvc->bjc', self.reg17, verts)
        root = J17[:, 0:1]
        return verts - root, J17 - root

    def fit(self, pose0, lift, xy, w, use_dense, R=None):
        """pose0 (B,24,3) khoi tao tu IK; lift (B,17,3); xy (B,65,2); w (B,65) trong so 2D; R (B,3,3) view_rot.
        Tra (pose, beta, t, px)."""
        B = pose0.shape[0]
        n2d = 65 if use_dense else 17
        wv = w[:, :n2d] / w[:, :n2d].mean(1, keepdim=True).clamp_min(1e-6)
        with torch.no_grad():
            _, J0 = self.mesh(pose0, torch.zeros(B, 10, device=pose0.device))
            t0 = init_trans(J0 @ R if R is not None else J0, xy[:, :17], w[:, :17])
        pose = pose0.clone().requires_grad_(True)
        beta = torch.zeros(B, 10, device=pose0.device, requires_grad=True)
        t = t0.clone().requires_grad_(True)

        def loss_fn(stage2):
            v, J = self.mesh(pose, beta)
            P = torch.cat([J, v[:, self.dense_ids]], 1)[:, :n2d]
            e = torch.norm(project(P, t, R) - xy[:, :n2d], dim=-1) * (IMG_WH.to(P).mean() / 2)   # px
            l2d = (huber(e, args.huber_px) * wv).mean()
            l3d = torch.sqrt(((J - lift) ** 2).sum(-1) + 1e-8).mean() * 1000            # mm
            lb = (beta ** 2).mean()
            lp = ((pose[:, 1:22] - pose0[:, 1:22]) ** 2).mean()
            L = args.w2d * l2d + args.w3d * l3d + (args.wbeta * lb + args.wpose * lp if stage2 else 0.0)
            return L, e

        # giai doan 1: chi t + huong toan cuc (khop 0)
        opt = torch.optim.Adam([t, pose], lr=args.lr)
        mask = torch.zeros_like(pose)
        mask[:, 0] = 1.0
        for _ in range(args.iters_cam):
            opt.zero_grad()
            L, _ = loss_fn(False)
            L.backward()
            pose.grad.mul_(mask)                                              # chi xoay toan cuc
            opt.step()
        # giai doan 2: theta (khop 0..21) + beta + t
        opt = torch.optim.Adam([t, pose, beta], lr=args.lr)
        for _ in range(args.iters):
            opt.zero_grad()
            L, e = loss_fn(True)
            L.backward()
            opt.step()
        with torch.no_grad():
            _, e = loss_fn(True)
        return pose.detach(), beta.detach(), t.detach(), e.detach()


def run_set(name, lifter, refiner, M, dense_ids, dev):
    cfg.DATASET.pw3d_ocpc_test_only = (args.ocpc_mode == 'test') and name in ('3dpw-oc', '3dpw-pc')
    label = name + (' (chi seq test)' if cfg.DATASET.pw3d_ocpc_test_only else '')
    ds = get_test_dataset(name, None)
    reg17 = torch.as_tensor(ds.h36m_joint_regressor, dtype=torch.float32, device=dev)
    jreg = torch.as_tensor(ds.joint_regressor, dtype=torch.float32, device=dev)
    eval_idx = torch.as_tensor(ds.h36m_eval_joint, device=dev)
    root = int(ds.h36m_root_joint_idx)
    layer = copy.deepcopy(ds.mesh_model.layer['neutral']).to(dev).eval()
    hyb = HybrIKSMPL(layer, ds.h36m_joint_regressor).to(dev).eval()
    fitter = Fitter(layer, reg17, dense_ids)
    if args.max_samples and args.max_samples < len(ds):
        ds = Subset(ds, list(range(args.max_samples)))
    bs = args.batch_size or cfg.TEST.batch_size
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=cfg.DATASET.workers)
    variants = ['init_IK'] + [v.strip() for v in args.variants.split(',') if v.strip()]
    acc = {v: {'mpjpe': [], 'pa': [], 'mpvpe': [], 'px': [], 'beta': []} for v in variants}

    def j14(j17):
        return (j17 - j17[:, root:root + 1])[:, eval_idx]

    def add(v, verts, gt14, gt_mesh_rel, px=None, beta=None):
        p14 = j14(torch.einsum('jv,bvc->bjc', reg17, verts))
        acc[v]['mpjpe'].append((torch.norm(p14 - gt14, dim=-1).mean(-1) * 1000).cpu().numpy())
        acc[v]['pa'].append(pa_errors(p14, gt14).cpu().numpy())
        mrel = verts - torch.einsum('v,bvc->bc', jreg[0], verts)[:, None]
        acc[v]['mpvpe'].append((torch.norm(mrel - gt_mesh_rel, dim=-1).mean(-1) * 1000).cpu().numpy())
        if px is not None:
            acc[v]['px'].append(px.mean(1).cpu().numpy())
        if beta is not None:
            acc[v]['beta'].append(beta.norm(dim=-1).cpu().numpy())

    for bi, (inputs, targets, meta) in enumerate(loader):
        img = inputs['img'].to(dev).float()
        B = img.shape[0]
        j2 = inputs['joints'].to(dev).float()[..., :2]
        jm = inputs['joints_mask'].to(dev).float().reshape(B, 17)
        cam = inputs['cam_bbox'].to(dev).float()
        gt_mesh = targets['smpl_mesh_cam'].to(dev).float()
        gt14 = j14(torch.einsum('jv,bvc->bjc', reg17, gt_mesh))
        gt_mesh_rel = gt_mesh - torch.einsum('v,bvc->bc', jreg[0], gt_mesh)[:, None]

        with torch.no_grad():
            nc = int(getattr(lifter, 'num_ctx', 0))
            assert nc == len(dense_ids), f'ImgLifter co {nc} diem day, json co {len(dense_ids)} dinh'
            xy, log_b, f8, fj = refiner_inputs(refiner, img, j2, jm, 17 + nc)        # (B,65,2), (B,65)
            cond7 = ImgLifter.make_cond7(xy, log_b, lifter_det(jm, nc), cam)
            gen = torch.Generator(device=dev)
            gen.manual_seed(int(args.seed) + bi)
            hyp = lifter.sample(cond7, fj, f8, K=args.K, S=args.S, eta=args.eta, generator=gen)
            lift = aggregate(hyp, xy, log_b, 'mean')
            lift = lift - lift[:, 0:1]
            # khoi tao: baseline C (M x lift -> IK, beta 0, twist 0)
            j29 = torch.einsum('jk,bkc->bjc', M, lift)
            beta0 = torch.zeros(B, 10, device=dev)
            rot = hyb.ik(j29, beta0, hyb.default_phis(B, dev, torch.float32))
            pose0 = rotmat_to_aa_safe(rot.reshape(-1, 3, 3)).reshape(B, 24, 3)
            v0, _ = fitter.mesh(pose0, beta0)
            w = 1.0 / torch.exp(log_b).clamp_min(args.b_min) ** 2                   # (B,65)
            R = None if args.no_view_rot else view_rot_torch(cam)
        add('init_IK', v0, gt14, gt_mesh_rel)
        for v in variants[1:]:
            with torch.enable_grad():
                pose, beta, t, e = fitter.fit(pose0, lift, xy, w, use_dense=(v == 'dense'), R=R)
            with torch.no_grad():
                vv, _ = fitter.mesh(pose, beta)
            add(v, vv, gt14, gt_mesh_rel, px=e, beta=beta)

    n = sum(len(x) for x in acc['init_IK']['mpjpe'])
    lines = [f'[{label}] n={n} | K={args.K} S={args.S} | iters {args.iters_cam}+{args.iters} lr {args.lr} | '
             f'w2d {args.w2d} huber {args.huber_px}px w3d {args.w3d} wbeta {args.wbeta} wpose {args.wpose} '
             f'view_rot {not args.no_view_rot}',
             f'  {"bien the":8s} | {"MPJPE":>7s} | {"PA":>6s} | {"MPVPE":>7s} | {"2D px":>6s} | {"|beta|":>6s}']
    for v in variants:
        a = acc[v]
        px = f'{np.concatenate(a["px"]).mean():6.2f}' if a['px'] else f'{"-":>6s}'
        bt = f'{np.concatenate(a["beta"]).mean():6.2f}' if a['beta'] else f'{"-":>6s}'
        lines.append(f'  {v:8s} | {np.concatenate(a["mpjpe"]).mean():7.2f} | {np.concatenate(a["pa"]).mean():6.2f} | '
                     f'{np.concatenate(a["mpvpe"]).mean():7.2f} | {px} | {bt}')
    return '\n'.join(lines)


def main():
    torch.manual_seed(args.seed)
    dev = torch.device('cuda')
    M = torch.as_tensor(np.load(args.M), dtype=torch.float32, device=dev)
    assert tuple(M.shape) == (29, 17), f'M phai (29, 17), nhan {tuple(M.shape)}'
    with open(args.dense_json, 'r', encoding='utf-8') as f:
        dense_ids = torch.as_tensor(json.load(f)['vertex_ids'], dtype=torch.long, device=dev)
    lifter, ck = load_img_lifter(args.ckpt, dev)
    refiner = load_frozen_refiner(args.refiner_ckpt or ck['refiner_ckpt'], dev)
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        print(run_set(name, lifter, refiner, M, dense_ids, dev), flush=True)
    print('Tham chieu baseline C (ik_upper_bound, beta 0, twist 0): 3DPW 73.18/49.23/90.59 | OC 82.02/51.60/96.85 | '
          'PC 64.19/67.97/103.61 | Crowd 77.56/54.08/101.08')


if __name__ == '__main__':
    main()
