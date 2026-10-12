"""
check_student_v2.py — Kiem tra nhanh kien truc Student v2 (PLAN_STUDENT_V2.md) TRUOC khi train. Khong can dataset.

Kiem tra:
  1. Build model tu yml, forward 1 batch gia (anh ngau nhien, 2D ngau nhien) -> shape dau ra, khong NaN.
  2. Luc khoi tao: head theo khop (lop cuoi = 0) -> goc xoay = mean pose; cac nhanh moi (khoi tao 0) chua lam doi ket qua.
  3. Backward: cac module moi deu nhan gradient khac 0 (khong bi dut duong).
  4. Noi day PE2D: dat trong so PE ngau nhien -> doi toa do 2D cua 1 khop thi dau ra phai doi;
     khop co mask = 0 thi doi toa do KHONG lam doi dau ra.
  5. So tham so: tong / trong so moi.

Chay:
    python main/check_student_v2.py --cfg ./config/train_student_v2.yml [--gpu 0]
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
import argparse

parser = argparse.ArgumentParser(description='Kiem tra nhanh Student v2')
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--skip_data', action='store_true',
                    help='bo qua buoc nap du lieu that tu DATASET.test_list (vd 3dpw-val, 3dpw)')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401
from models.ARTS import ARTS
from utils.transforms import rot6d_to_axis_angle

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)


def ok(flag, msg):
    print(('  [OK]   ' if flag else '  [LOI]  ') + msg)
    return flag


def make_batch(B):
    img = torch.rand(B, 3, 256, 256, device=DEVICE)
    xy = (torch.rand(B, 17, 2, device=DEVICE) * 1.6 - 0.8)
    joints = torch.cat([xy, torch.ones(B, 17, 1, device=DEVICE)], dim=-1)   # (B, 17, 3) giong wrapper
    mask = torch.ones(B, 17, 1, device=DEVICE)
    return img, joints, mask


def main():
    assert cfg.MODEL.name == 'student', 'Script nay cho Student (MODEL.name = student)'
    print('Co kien truc:', {k: cfg.MODEL.get(k) for k in ('joint_pe2d', 'pe2d_num_freq', 'fusion_joint_self_attn',
                                                          'pose_head_mode', 'pose_query_bone', 'shape_attn_mode')})
    model = ARTS(num_joint=17, embed_dim=cfg.MODEL.hpe_dim).to(DEVICE)
    sm = model.smpl_model
    all_ok = True

    # ---- 5. So tham so ----
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    new_names = ('pe2d', 'emb_missing', 'joint_sa', 'per_joint_head', 'bone_proj', 'fuse_shape_tok')
    n_new = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in new_names))
    print(f'Tham so train duoc: {n_train / 1e6:.2f} M | trong do module moi: {n_new / 1e6:.2f} M')

    # ---- 1. Forward ----
    B = args.batch_size
    img, joints, mask = make_batch(B)
    model.train()
    out = model(img, joints, is_train=True, joints_mask=mask)
    all_ok &= ok(tuple(out['smpl_pose'].shape) == (B, 72), f"smpl_pose {tuple(out['smpl_pose'].shape)}")
    all_ok &= ok(tuple(out['smpl_shape'].shape) == (B, 10), f"smpl_shape {tuple(out['smpl_shape'].shape)}")
    all_ok &= ok(tuple(out['smpl_mesh_cam'].shape) == (B, 6890, 3), f"smpl_mesh_cam {tuple(out['smpl_mesh_cam'].shape)}")
    finite = all(torch.isfinite(out[k]).all().item() for k in ('smpl_pose', 'smpl_shape', 'smpl_mesh_cam'))
    all_ok &= ok(finite, 'khong NaN / Inf')

    # ---- 2. Khoi tao: goc xoay = mean pose ----
    if cfg.MODEL.pose_head_mode == 'per_joint':
        mean_aa = rot6d_to_axis_angle(sm.init_pose.reshape(-1, 6)).reshape(1, 24, 3)[:, :22]
        pred_aa = out['smpl_pose'].reshape(B, 24, 3)[:, :22]
        err = (pred_aa - mean_aa).abs().max().item()
        all_ok &= ok(err < 1e-4, f'luc khoi tao goc xoay = mean pose (lech max {err:.2e})')
        hands = out['smpl_pose'].reshape(B, 24, 3)[:, 22:].abs().max().item()
        all_ok &= ok(hands == 0.0, 'khop ban tay (22, 23) = 0')

    # ---- 3. Backward: module moi nhan gradient ----
    # Lop cuoi head theo khop = 0 chan gradient ve query (bone_proj, pose_context_attn) o buoc dau tien
    # (binh thuong: sau buoc toi uu dau tien lop nay khac 0). De kiem tra duong noi, dat no ngau nhien nho.
    if cfg.MODEL.pose_head_mode == 'per_joint':
        with torch.no_grad():
            torch.nn.init.normal_(sm.per_joint_head.layers[-1].weight, std=0.01)
        out = model(img, joints, is_train=True, joints_mask=mask)
    model.zero_grad()
    loss = out['smpl_mesh_cam'].pow(2).mean() + out['smpl_shape'].pow(2).mean() + out['feat_joint'].pow(2).mean()
    loss.backward()
    checks = []
    if cfg.MODEL.pose_head_mode == 'per_joint':
        checks.append(('per_joint_head (lop cuoi)', sm.per_joint_head.layers[-1].weight))
    if cfg.MODEL.pose_query_bone:
        checks.append(('bone_proj', sm.bone_proj.weight))
    if cfg.MODEL.joint_pe2d:
        checks.append(('fusion.pe2d.proj', sm.fusion.pe2d.proj.weight))
    if cfg.MODEL.fusion_joint_self_attn:
        checks.append(('joint_sa.out_proj (block 0)', sm.fusion.cfcer.blocks[0].joint_sa.out_proj.weight))
    if cfg.MODEL.shape_attn_mode == 'tokens':
        checks.append(('fuse_shape_tok.attn.wq', sm.fuse_shape_tok.attn.wq.weight))
    for name, p in checks:
        g = p.grad
        all_ok &= ok(g is not None and g.abs().sum().item() > 0,
                     f'gradient toi {name}: ' + ('None' if g is None else f'{g.abs().sum().item():.3e}'))

    # ---- 4. Noi day PE2D ----
    if cfg.MODEL.joint_pe2d:
        model.eval()
        with torch.no_grad():
            torch.nn.init.normal_(sm.fusion.pe2d.proj.weight, std=0.5)     # chi de kiem tra, model khong luu
            if cfg.MODEL.pose_head_mode == 'per_joint':
                torch.nn.init.normal_(sm.per_joint_head.layers[-1].weight, std=0.05)
            base = model(img, joints, is_train=False, joints_mask=mask)['smpl_mesh_cam']
            # Doi toa do 2D cua khop 13 (L_Wrist). Doi o anh 2D KHONG doi joint 3D vi MotionBERT cung nhan 2D ->
            # so sanh voi truong hop mask = 0 de tach rieng anh huong qua PE.
            j2 = joints.clone()
            j2[:, 13, :2] = -j2[:, 13, :2]
            moved = model(img, j2, is_train=False, joints_mask=mask)['smpl_mesh_cam']
            all_ok &= ok((moved - base).abs().max().item() > 0, 'doi 2D 1 khop -> dau ra doi (PE2D + MotionBERT co noi)')
            # Tach rieng PE: giu nguyen 3D (lift tu joints goc), chi doi 2D dua vao PE
            feat, _ = model.get_image_features(img)
            p3 = model.lift_2d_to_3d(joints, joints_mask=mask)
            p3 = p3 - p3[:, 0:1]
            o_a = sm(joints=p3, img_feats=feat, is_train=False, joints_2d=joints[..., :2], joints_2d_mask=mask)
            o_b = sm(joints=p3, img_feats=feat, is_train=False, joints_2d=j2[..., :2], joints_2d_mask=mask)
            all_ok &= ok((o_b['smpl_mesh_cam'] - o_a['smpl_mesh_cam']).abs().max().item() > 0,
                         'CHI doi 2D vao PE (3D giu nguyen) -> dau ra doi')
            m0 = mask.clone()
            m0[:, 13] = 0
            o_c = sm(joints=p3, img_feats=feat, is_train=False, joints_2d=joints[..., :2], joints_2d_mask=m0)
            o_d = sm(joints=p3, img_feats=feat, is_train=False, joints_2d=j2[..., :2], joints_2d_mask=m0)
            all_ok &= ok((o_d['smpl_mesh_cam'] - o_c['smpl_mesh_cam']).abs().max().item() < 1e-6,
                         'khop mask = 0: doi toa do 2D KHONG lam doi dau ra')

    print('\n' + ('TAT CA DEU OK' if all_ok else 'CO LOI - xem cac dong [LOI] o tren'))


if __name__ == '__main__':
    main()
