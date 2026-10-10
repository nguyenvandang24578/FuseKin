"""
vposer_floor_test.py - Sai so "san" cua duong joint -> mesh (khong can train, khong can anh)
===============================================================================================

Cau hoi: Teacher nhan GT 3D sach van sai ~43-58 mm. Bao nhieu trong do la do CHINH CACH model
bieu dien tu the (VPoser 32 chieu, 2 khop tay = 0, SMPL neutral), chu khong phai do model hoc kem?

Cach lam: lay tham so SMPL GT (pose 72, shape 10, gender) cua 3DPW, roi so mesh GT voi cac phien ban
"tot nhat co the" ma kien truc hien tai bieu dien duoc:

  [0] GT          : SMPL theo GIOI TINH, pose/shape GT (dung nhu GT luc danh gia)
  [1] neutral     : SMPL NEUTRAL, pose/shape GT                    -> san do model luon dung neutral
  [2] + tay = 0   : nhu [1], 2 khop tay (SMPL 22, 23) = 0          -> + san do Vposer gan tay = 0
  [3] + VPoser    : nhu [2], 21 khop than di qua VPoser encode -> latent (mean) -> decode
                    (dung DUNG lop Vposer cua model)              -> san cua toan bo duong pose
  [4] + shape = 0 : nhu [3] nhung shape = 0 (dang trung binh)     -> tham khao: neu doan shape te nhat

Moi dong so voi [0] bang dung cach tinh cua dataset.evaluate (PW3D): H36M-17 regress tu mesh,
tru pelvis, 14 khop danh gia; PA-MPJPE; MPVPE (mesh tru khop 0 cua SMPL).
Ghi chu: [3] dung latent TOT NHAT VPoser tim duoc bang encoder (khong toi uu them), nen la uoc luong
lac quan nhe cua san; san that co the cao hon mot chut.

Cach chay (tu thu muc goc repo, can CUDA vi lop Vposer goi .cuda()):
    python main/vposer_floor_test.py --cfg ./config/train_student.yml [--num 2000] [--split 3dpw]
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')

import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser(description='San sai so cua duong joint -> mesh (VPoser / neutral / tay)')
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--split', type=str, default='3dpw', help="'3dpw' (test) | '3dpw-train'")
parser.add_argument('--num', type=int, default=2000, help='so mau lay ngau nhien (<=0: tat ca)')
parser.add_argument('--batch', type=int, default=256)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401
from data_final.PW3D.dataset import PW3D
from models.common import Vposer
from utils.transforms import rigid_align

if not torch.cuda.is_available():
    print('[LOI] Can CUDA (lop Vposer cua repo goi .cuda()).')
    sys.exit(1)
DEV = torch.device('cuda')


def aa_to_rotmat(aa):
    """aa (N,3) -> R (N,3,3), cong thuc Rodrigues."""
    angle = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis = aa / angle
    x, y, z = axis.unbind(-1)
    c, s = torch.cos(angle[:, 0]), torch.sin(angle[:, 0])
    C = 1.0 - c
    R = torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s,
                     y * x * C + z * s, c + y * y * C, y * z * C - x * s,
                     z * x * C - y * s, z * y * C + x * s, c + z * z * C], dim=-1)
    return R.view(-1, 3, 3)


class VposerRoundtrip:
    """Encode 21 khop than (axis-angle) -> latent (mean) -> decode bang DUNG Vposer.forward cua model."""

    def __init__(self):
        self.vp = Vposer()
        self.vp.vposer = self.vp.vposer.to(DEV).eval()
        inner = self.vp.vposer
        bn = getattr(inner, 'bodyprior_enc_bn1', None)
        self.n_in = bn.num_features if bn is not None else None
        print(f'[VPoser] dau vao encoder: {self.n_in} chieu '
              f'({"ma tran xoay 21x9" if self.n_in == 189 else "axis-angle 21x3" if self.n_in == 63 else "chua ro"})')

    @torch.no_grad()
    def __call__(self, body_aa):
        """body_aa (B,63) -> body pose sau VPoser (B,69) = 21 khop than + 2 khop tay = 0 (giong model)."""
        B = body_aa.shape[0]
        inner = self.vp.vposer
        if self.n_in == 63:
            enc_in = body_aa
        else:   # mac dinh VPoser v1: ma tran xoay
            enc_in = aa_to_rotmat(body_aa.reshape(-1, 3)).reshape(B, -1)
        q = inner.encode(enc_in)
        z = q.mean if hasattr(q, 'mean') and not torch.is_tensor(q) else q
        return self.vp(z)   # (B, 69): da gan tay = 0 nhu luc model chay


def smpl_mesh(layer, pose, shape):
    """pose (B,72), shape (B,10) tren CPU -> mesh (B,6890,3) met, trans = 0."""
    trans = torch.zeros(pose.shape[0], 3)
    verts, _ = layer(pose, shape, trans)
    return verts


def evaluate(mesh_pred, mesh_gt, h36m_reg, smpl_reg, eval_idx):
    """Giong PW3D.evaluate. Tra ve (mpjpe14, pa_mpjpe14, mpvpe, mpjpe17) moi mau, mm."""
    out = {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': [], 'mpjpe17': []}
    for p, g in zip(mesh_pred, mesh_gt):
        jg = h36m_reg @ g
        jg = jg - jg[0:1]
        jp = h36m_reg @ p
        jp = jp - jp[0:1]
        out['mpjpe17'].append(np.linalg.norm(jp - jg, axis=1).mean() * 1000)
        jg14, jp14 = jg[eval_idx], jp[eval_idx]
        out['mpjpe'].append(np.linalg.norm(jp14 - jg14, axis=1).mean() * 1000)
        out['pa_mpjpe'].append(np.linalg.norm(rigid_align(jp14, jg14) - jg14, axis=1).mean() * 1000)
        g0 = g - (smpl_reg @ g)[0:1]
        p0 = p - (smpl_reg @ p)[0:1]
        out['mpvpe'].append(np.linalg.norm(p0 - g0, axis=1).mean() * 1000)
    return out


def main():
    ds = PW3D(None, data_name=args.split)
    datalist = ds.datalist
    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(datalist)) if args.num <= 0 or args.num >= len(datalist) \
        else rng.choice(len(datalist), args.num, replace=False)
    print(f'Split {args.split}: {len(datalist)} mau, dung {len(idx)} mau')

    h36m_reg = np.asarray(ds.h36m_joint_regressor, dtype=np.float64)
    smpl_reg = np.asarray(ds.joint_regressor, dtype=np.float64)
    eval_idx = list(ds.h36m_eval_joint)
    layers = ds.smpl.layer
    vposer = VposerRoundtrip()

    names = ['[1] neutral', '[2] + tay = 0', '[3] + VPoser', '[4] + shape = 0']
    acc = {n: {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': [], 'mpjpe17': []} for n in names}
    latent_norm = []

    for st in range(0, len(idx), args.batch):
        chunk = [datalist[i]['smpl_param'] for i in idx[st:st + args.batch]]
        pose = torch.tensor(np.array([np.asarray(c['pose'], dtype=np.float32).reshape(72) for c in chunk]))
        shape = torch.tensor(np.array([np.asarray(c['shape'], dtype=np.float32).reshape(10) for c in chunk]))
        genders = [c['gender'] for c in chunk]

        with torch.no_grad():
            # [0] GT: SMPL theo gioi tinh (tung nhom gioi tinh)
            gt = torch.zeros(len(chunk), 6890, 3)
            for gname in set(genders):
                m = torch.tensor([g == gname for g in genders])
                gt[m] = smpl_mesh(layers[gname], pose[m], shape[m])

            neutral = layers['neutral']
            # [1] neutral
            m1 = smpl_mesh(neutral, pose, shape)
            # [2] tay = 0
            pose2 = pose.clone()
            pose2[:, 22 * 3:24 * 3] = 0
            m2 = smpl_mesh(neutral, pose2, shape)
            # [3] VPoser tren 21 khop than (SMPL 1..21), giu root GT
            body = pose[:, 3:66].to(DEV)
            body_vp = vposer(body).cpu()                       # (B, 69), tay = 0
            pose3 = torch.cat([pose[:, :3], body_vp], dim=1)
            m3 = smpl_mesh(neutral, pose3, shape)
            # [4] + shape = 0
            m4 = smpl_mesh(neutral, pose3, torch.zeros_like(shape))

        gt_np = gt.double().numpy()
        for name, m in zip(names, (m1, m2, m3, m4)):
            r = evaluate(m.double().numpy(), gt_np, h36m_reg, smpl_reg, eval_idx)
            for k in r:
                acc[name][k].extend(r[k])
        print(f'  ... {min(st + args.batch, len(idx))}/{len(idx)}')

    print('\n' + '=' * 78)
    print(f'SAN SAI SO so voi GT ({args.split}, {len(idx)} mau) - cung cach tinh voi dataset.evaluate')
    print('=' * 78)
    print(f"  {'':<18}{'MPJPE(14)':>11}{'PA-MPJPE':>10}{'MPVPE':>9}{'MPJPE(17)':>11}")
    for name in names:
        a = {k: float(np.mean(v)) for k, v in acc[name].items()}
        print(f"  {name:<18}{a['mpjpe']:11.2f}{a['pa_mpjpe']:10.2f}{a['mpvpe']:9.2f}{a['mpjpe17']:11.2f}")
    print('\n  Doc ket qua:')
    print('  - [3] la san cua DUONG POSE hien tai (neutral + tay = 0 + VPoser) khi moi thu khac hoan hao.')
    print('    So [3] voi Teacher nhan GT sach (~43-58 mm theo dataset.evaluate cang chinh xac hon).')
    print('  - [3] - [2] = phan VPoser lam mat; [2] - [1] = phan tay = 0; [1] = phan neutral thay gioi tinh.')
    print('  - [4] cho biet shape quan trong toi dau (moc tham khao, khong phai san).')


if __name__ == '__main__':
    main()
