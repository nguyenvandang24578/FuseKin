"""
main/ik_upper_bound.py -- (2026-11-08) Can tren cua duong "pose 3D -> IK giai tich (HybrIK) -> mesh", KHONG train.
Tra loi: neu Student chi lay pose lifter roi IK (khong hoc them gi), mesh dat bao nhieu so voi LIFT va Student v5?

Bien the (twist = 0 o moi bien the; beta = shape trung binh, TRU D):
  LIFT      : pose ImgLifter (K, S nhu Student), 14 khop, KHONG qua mesh (= cot LIFT cua test_student)
  A_gt29    : 29 diem lay tu mesh GT (J_regressor x mesh + 5 dinh)  -> IK  : san loi do beta TB + twist 0
  B_M_gt17  : M x H36M-17 GT                                         -> IK  : + loi anh xa 17 -> 29
  C_M_lift  : M x pose lifter                                        -> IK  : CAN TREN THUC TE (khong hoc them)
  D_M_lift_bb (2026-11-11): nhu C nhung beta = beta_xuong (HybrIKSMPL.beta_from_bones, Gauss-Newton, KHONG hoc,
              giong Student v7) -> tach loi ich cua Student: do beta_xuong hay do phan hoc (HyperGCN, twist, dbeta, delta)
Do: MPJPE / PA-MPJPE 14 khop (H36M regressor x mesh, root Pelvis H36M, h36m_eval_joint) + MPVPE (tru Pelvis SMPL),
giong dataset.evaluate / test_student_3dpw_sets.

Can truoc: tools/fit_h36m_to_smpl29.py -> M (29 x 17).
Chay:
  python main/ik_upper_bound.py --cfg config/train_img_lifter_dense48_3dpw_markov.yaml \
    --ckpt experiment/img_lifter_dense48_3dpw_markov/checkpoint/last.pth.tar \
    --M data_final/h36m17_to_smpl29.npy --sets 3dpw,3dpw-oc,3dpw-pc,3dpw-crowd --ocpc_mode test --gpu 0
  (thu nhanh: --max_samples 2000)
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse

parser = argparse.ArgumentParser(description='Can tren IK tu pose lifter')
parser.add_argument('--cfg', type=str, required=True, help='yml da dung de train ImgLifter')
parser.add_argument('--ckpt', type=str, required=True, help='checkpoint ImgLifter')
parser.add_argument('--refiner_ckpt', type=str, default='', help='rong = refiner luu trong ckpt ImgLifter')
parser.add_argument('--M', type=str, default='data_final/h36m17_to_smpl29.npy', help='anh xa 29 x 17')
parser.add_argument('--sets', type=str, default='3dpw,3dpw-oc,3dpw-pc,3dpw-crowd')
parser.add_argument('--ocpc_mode', type=str, default='test', choices=['all', 'test'])
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--S', type=int, default=2)
parser.add_argument('--eta', type=float, default=1.0)
parser.add_argument('--max_samples', type=int, default=0, help='0 = toan bo tap')
parser.add_argument('--batch_size', type=int, default=0, help='0 = cfg.TEST.batch_size')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--bone_gn_iters', type=int, default=3, help='D: so buoc Gauss-Newton (= v7_bone_gn_iters)')
parser.add_argument('--bone_beta_reg', type=float, default=1e-3, help='D: lambda (= v7_bone_beta_reg cua breg)')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from core.config import cfg, update_config
update_config(args.cfg)
assert cfg.DATASET.pw3d_2d_source == 'openpose', 'Benchmark: dau vao 3DPW la OpenPose / HHRNet'

import __init_path  # noqa: F401
from utils.jotr_dataset import get_test_dataset
from utils.img_lifter_tools import load_img_lifter, load_frozen_refiner, refiner_inputs, lifter_det, pa_errors
from models.img_lifter import ImgLifter, aggregate
from models.hybrik_smpl import HybrIKSMPL

VARIANTS = ['LIFT', 'A_gt29', 'B_M_gt17', 'C_M_lift', 'D_M_lift_bb']


@torch.no_grad()
def run_set(name, lifter, refiner, M, dev):
    cfg.DATASET.pw3d_ocpc_test_only = (args.ocpc_mode == 'test') and name in ('3dpw-oc', '3dpw-pc')
    label = name + (' (chi seq test)' if cfg.DATASET.pw3d_ocpc_test_only else '')
    ds = get_test_dataset(name, None)
    reg17 = torch.as_tensor(ds.h36m_joint_regressor, dtype=torch.float32, device=dev)        # (17, V)
    jreg = torch.as_tensor(ds.joint_regressor, dtype=torch.float32, device=dev)               # (30, V) SMPL
    eval_idx = torch.as_tensor(ds.h36m_eval_joint, device=dev)
    root = int(ds.h36m_root_joint_idx)
    # ban sao lop SMPL tren GPU (khong dung lop cua dataset)
    hyb = HybrIKSMPL(copy.deepcopy(ds.mesh_model.layer['neutral']), ds.h36m_joint_regressor).to(dev).eval()
    if args.max_samples and args.max_samples < len(ds):
        ds = Subset(ds, list(range(args.max_samples)))
    bs = args.batch_size or cfg.TEST.batch_size
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=cfg.DATASET.workers)

    acc = {v: {'mpjpe': [], 'pa': [], 'mpvpe': []} for v in VARIANTS}

    def j14_of(j17):
        return (j17 - j17[:, root:root + 1])[:, eval_idx]

    def add_joints(v, j17, gt14):
        p14 = j14_of(j17)
        acc[v]['mpjpe'].append((torch.norm(p14 - gt14, dim=-1).mean(-1) * 1000).cpu().numpy())
        acc[v]['pa'].append(pa_errors(p14, gt14).cpu().numpy())

    def add_mesh(v, verts, gt14, gt_mesh_rel):
        add_joints(v, torch.einsum('jv,bvc->bjc', reg17, verts), gt14)
        mesh_rel = verts - torch.einsum('v,bvc->bc', jreg[0], verts)[:, None]               # tru Pelvis SMPL
        acc[v]['mpvpe'].append((torch.norm(mesh_rel - gt_mesh_rel, dim=-1).mean(-1) * 1000).cpu().numpy())

    for bi, (inputs, targets, meta) in enumerate(loader):
        img = inputs['img'].to(dev).float()
        B = img.shape[0]
        j2 = inputs['joints'].to(dev).float()[..., :2]
        jm = inputs['joints_mask'].to(dev).float().reshape(B, 17)
        cam = inputs['cam_bbox'].to(dev).float()
        gt_mesh = targets['smpl_mesh_cam'].to(dev).float()                                   # (B, V, 3) met
        gt17 = torch.einsum('jv,bvc->bjc', reg17, gt_mesh)
        gt14 = j14_of(gt17)
        gt_mesh_rel = gt_mesh - torch.einsum('v,bvc->bc', jreg[0], gt_mesh)[:, None]
        gt17_rel = gt17 - gt17[:, root:root + 1]

        # pose lifter (giong Student: Refiner day -> ImgLifter K x S, gop mean, root-relative)
        nc = int(getattr(lifter, 'num_ctx', 0))
        xy, log_b, f8, fj = refiner_inputs(refiner, img, j2, jm, 17 + nc)
        cond7 = ImgLifter.make_cond7(xy, log_b, lifter_det(jm, nc), cam)
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(args.seed) + bi)
        hyp = lifter.sample(cond7, fj, f8, K=args.K, S=args.S, eta=args.eta, generator=gen)
        lift = aggregate(hyp, xy, log_b, 'mean')
        lift = lift - lift[:, 0:1]

        add_joints('LIFT', lift, gt14)
        # A: 29 diem GT tu mesh GT (cung goc Pelvis H36M)
        j29_gt = hyb.joints29_from_mesh(gt_mesh) - gt17[:, root:root + 1]
        add_mesh('A_gt29', hyb(j29_gt)['vertices'], gt14, gt_mesh_rel)
        # B: M x H36M-17 GT
        add_mesh('B_M_gt17', hyb(torch.einsum('jk,bkc->bjc', M, gt17_rel))['vertices'], gt14, gt_mesh_rel)
        # C: M x pose lifter
        j29_lift = torch.einsum('jk,bkc->bjc', M, lift)
        add_mesh('C_M_lift', hyb(j29_lift)['vertices'], gt14, gt_mesh_rel)
        # D: nhu C, beta = beta_xuong tu do dai 16 xuong cua pose lifter (khong hoc, giong Student v7)
        beta_bb = hyb.beta_from_bones(lift.float(), int(args.bone_gn_iters), float(args.bone_beta_reg))
        add_mesh('D_M_lift_bb', hyb(j29_lift, betas=beta_bb)['vertices'], gt14, gt_mesh_rel)

    n = sum(len(x) for x in acc['LIFT']['mpjpe'])
    lines = [f'[{label}] n={n} | K={args.K} S={args.S} | twist 0 | beta TB (A-C), beta_xuong (D: '
             f'{args.bone_gn_iters} buoc, lambda {args.bone_beta_reg:g})',
             f'  {"bien the":10s} | {"MPJPE":>7s} | {"PA":>6s} | {"MPVPE":>7s}']
    for v in VARIANTS:
        a = acc[v]
        mp = np.concatenate(a['mpjpe']).mean()
        pa = np.concatenate(a['pa']).mean()
        mv = f'{np.concatenate(a["mpvpe"]).mean():7.2f}' if a['mpvpe'] else f'{"-":>7s}'
        lines.append(f'  {v:10s} | {mp:7.2f} | {pa:6.2f} | {mv}')
    return '\n'.join(lines)


def main():
    dev = torch.device('cuda')
    M = torch.as_tensor(np.load(args.M), dtype=torch.float32, device=dev)
    assert tuple(M.shape) == (29, 17), f'M phai (29, 17), nhan {tuple(M.shape)}'
    lifter, ck = load_img_lifter(args.ckpt, dev)
    refiner = load_frozen_refiner(args.refiner_ckpt or ck['refiner_ckpt'], dev)
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        print(run_set(name, lifter, refiner, M, dev), flush=True)
    print('Tham chieu Student v5 (last, MPJPE / PA / MPVPE): 3DPW 73.15/48.82/89.77 | OC 81.74/51.67/96.62 | '
          'PC 68.58/68.60/108.55 | Crowd 80.39/56.50/103.87')
    print('Tham chieu Student v7 breg (last, chi seq test): 3DPW 73.32/48.16/90.11 | OC 80.54/50.60/95.45 | '
          'PC 61.56/64.96/100.09 | Crowd 72.30/52.53/95.18')


if __name__ == '__main__':
    main()
