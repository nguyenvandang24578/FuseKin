"""
check_student_v2.py — Kiem tra nhanh kien truc v2 (PLAN_STUDENT_V2.md) TRUOC khi train.
Ho tro: Student (MODEL.name = student) va Teacher T-hint (MODEL.name = teacher, teacher_hint = True).

Kiem tra:
  1. Build model tu yml, forward 1 batch gia (anh ngau nhien, 2D ngau nhien, GT 3D gia cho T-hint) -> shape, khong NaN.
  2. Luc khoi tao: head theo khop (lop cuoi = 0) -> goc xoay = mean pose; cac nhanh moi (khoi tao 0) chua lam doi ket qua.
  3. Backward: cac module moi (va nhanh goi y cua T-hint) deu nhan gradient khac 0 (khong bi dut duong).
  4. Noi day PE2D: dat trong so PE ngau nhien -> doi toa do 2D cua 1 khop thi dau ra phai doi;
     khop co mask = 0 thi doi toa do KHONG lam doi dau ra. T-hint: bat / tat goi y -> dau ra phai khac.
  5. So tham so: tong / trong so moi.
  6. Du lieu that: nap tung dataset trong DATASET.test_list (vd 3dpw-val, 3dpw), forward 1 batch (--skip_data de bo qua).
     T-hint: forward ca CO goi y va KHONG goi y (giong Teacher_Tester).

Chay:
    python main/check_student_v2.py --cfg ./config/train_student_v2.yml [--gpu 0]
    python main/check_student_v2.py --cfg ./config/train_teacher_hint_k16_v2.yml   # Teacher T-hint v2
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
# Gioi han luong BLAS / OpenMP: dat TRUOC khi import numpy / torch de server chay duoc nhieu worker
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # cho Apple Accelerate/vecLib
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import argparse

parser = argparse.ArgumentParser(description='Kiem tra nhanh kien truc v2 (Student / Teacher T-hint)')
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
IS_THINT = cfg.MODEL.name == 'teacher' and bool(cfg.MODEL.get('teacher_hint', False))


def ok(flag, msg):
    print(('  [OK]   ' if flag else '  [LOI]  ') + msg)
    return flag


def make_batch(B):
    img = torch.rand(B, 3, 256, 256, device=DEVICE)
    xy = (torch.rand(B, 17, 2, device=DEVICE) * 1.6 - 0.8)
    joints = torch.cat([xy, torch.ones(B, 17, 1, device=DEVICE)], dim=-1)   # (B, 17, 3) giong wrapper
    mask = torch.ones(B, 17, 1, device=DEVICE)
    gt3d = torch.randn(B, 17, 3, device=DEVICE) * 0.3                         # GT 3D gia (met) cho T-hint
    gt3d = gt3d - gt3d[:, 0:1]
    return img, joints, mask, gt3d


def fwd(model, img, joints, mask, is_train, gt3d=None, hint_on=None, raw=None):
    """Goi model dung nhu Trainer / Tester: Student nhan 2D; T-hint nhan GT 3D + 2D (+ hint_on)."""
    if IS_THINT:
        if hint_on is not None:
            hint_on = torch.full((img.shape[0],), float(hint_on), device=img.device)
        return model(img, gt3d, is_train=is_train, pose_2d=joints, joints_mask=mask, hint_on=hint_on)
    # lifter PMCE can joints_raw; chua truyen thi dung tam joints (chi de smoke-test shape/finite)
    if getattr(model, 'use_pmce', False) and raw is None:
        raw = joints
    return model(img, joints, is_train=is_train, joints_mask=mask, pose_2d_raw=raw)


def main():
    assert cfg.MODEL.name == 'student' or IS_THINT, \
        'Script nay cho Student (MODEL.name = student) hoac Teacher T-hint (MODEL.name = teacher, teacher_hint = True)'
    print('Mo hinh:', 'Teacher T-hint' if IS_THINT else 'Student')
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
    img, joints, mask, gt3d = make_batch(B)
    model.train()
    out = fwd(model, img, joints, mask, True, gt3d)
    all_ok &= ok(tuple(out['smpl_pose'].shape) == (B, 72), f"smpl_pose {tuple(out['smpl_pose'].shape)}")
    all_ok &= ok(tuple(out['smpl_shape'].shape) == (B, 10), f"smpl_shape {tuple(out['smpl_shape'].shape)}")
    all_ok &= ok(tuple(out['smpl_mesh_cam'].shape) == (B, 6890, 3), f"smpl_mesh_cam {tuple(out['smpl_mesh_cam'].shape)}")
    finite = all(torch.isfinite(out[k]).all().item() for k in ('smpl_pose', 'smpl_shape', 'smpl_mesh_cam'))
    all_ok &= ok(finite, 'khong NaN / Inf')
    if IS_THINT:
        all_ok &= ok('hint_z' in out and tuple(out['hint_z'].shape) == (B, int(cfg.MODEL.hint_dim)),
                     f"hint_z {tuple(out['hint_z'].shape) if 'hint_z' in out else None}")

    # ---- 2. Khoi tao: goc xoay = mean pose ----
    if cfg.MODEL.pose_head_mode == 'per_joint':
        mean_aa = rot6d_to_axis_angle(sm.init_pose.reshape(-1, 6)).reshape(1, 24, 3)[:, :22]
        pred_aa = out['smpl_pose'].reshape(B, 24, 3)[:, :22]
        err = (pred_aa - mean_aa).abs().max().item()
        all_ok &= ok(err < 1e-4, f'luc khoi tao goc xoay = mean pose (lech max {err:.2e})')
        hands = out['smpl_pose'].reshape(B, 24, 3)[:, 22:].abs().max().item()
        all_ok &= ok(hands == 0.0, 'khop ban tay (22, 23) = 0')

    # ---- 3. Backward: module moi nhan gradient ----
    # Lop khoi tao 0 (lop cuoi head theo khop, hint_proj) chan gradient ve phia truoc o buoc dau tien
    # (binh thuong: sau buoc toi uu dau tien cac lop nay khac 0). De kiem tra duong noi, dat chung ngau nhien nho.
    with torch.no_grad():
        if cfg.MODEL.pose_head_mode == 'per_joint':
            torch.nn.init.normal_(sm.per_joint_head.layers[-1].weight, std=0.01)
        if IS_THINT:
            torch.nn.init.normal_(sm.hint_proj.weight, std=0.01)
    out = fwd(model, img, joints, mask, True, gt3d, hint_on=1.0)
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
    if IS_THINT:
        checks.append(('hint_proj', sm.hint_proj.weight))
        checks.append(('hint_encoder (lop cuoi)', sm.hint_encoder.net[-1].weight))
    for name, p in checks:
        g = p.grad
        all_ok &= ok(g is not None and g.abs().sum().item() > 0,
                     f'gradient toi {name}: ' + ('None' if g is None else f'{g.abs().sum().item():.3e}'))

    # ---- 4. Noi day PE2D (+ goi y cho T-hint) ----
    model.eval()
    with torch.no_grad():
        if cfg.MODEL.pose_head_mode == 'per_joint':
            torch.nn.init.normal_(sm.per_joint_head.layers[-1].weight, std=0.05)
        if IS_THINT:
            torch.nn.init.normal_(sm.hint_proj.weight, std=0.05)
            o_on = fwd(model, img, joints, mask, False, gt3d, hint_on=1.0)['smpl_mesh_cam']
            o_off = fwd(model, img, joints, mask, False, gt3d, hint_on=0.0)['smpl_mesh_cam']
            all_ok &= ok((o_on - o_off).abs().max().item() > 0, 'T-hint: CO goi y va KHONG goi y cho dau ra khac nhau')
        if cfg.MODEL.joint_pe2d and not getattr(model, 'use_pmce', False):
            torch.nn.init.normal_(sm.fusion.pe2d.proj.weight, std=0.5)     # chi de kiem tra, model khong luu
            base = fwd(model, img, joints, mask, False, gt3d, hint_on=0.0)['smpl_mesh_cam']
            # Doi toa do 2D cua khop 13 (L_Wrist). Doi 2D cung doi joint 3D (MotionBERT nhan 2D) ->
            # ben duoi tach rieng anh huong qua PE bang cach giu nguyen 3D.
            j2 = joints.clone()
            j2[:, 13, :2] = -j2[:, 13, :2]
            moved = fwd(model, img, j2, mask, False, gt3d, hint_on=0.0)['smpl_mesh_cam']
            all_ok &= ok((moved - base).abs().max().item() > 0, 'doi 2D 1 khop -> dau ra doi (PE2D + MotionBERT co noi)')
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

    # ---- 6. Du lieu that: nap tung dataset trong test_list + forward 1 batch ----
    if not args.skip_data:
        from torch.utils.data import DataLoader
        from utils.jotr_dataset import get_test_dataset
        sel = cfg.TEST.get('select_dataset', '')
        all_ok &= ok(not sel or sel in cfg.DATASET.test_list,
                     f"TEST.select_dataset='{sel}' nam trong test_list {list(cfg.DATASET.test_list)}")
        # Model sach (trong so khoi tao lai) de forward du lieu that
        model = ARTS(num_joint=17, embed_dim=cfg.MODEL.hpe_dim).to(DEVICE).eval()
        for name in cfg.DATASET.test_list:
            try:
                ds = get_test_dataset(name, None)
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
                inputs, targets, meta = next(iter(loader))
                img = inputs['img'].to(DEVICE).float()
                j2 = inputs['joints'].to(DEVICE).float()
                jm = inputs['joints_mask'].to(DEVICE).float()
                jraw = inputs['joints_raw'].to(DEVICE).float() if 'joints_raw' in inputs else None
                gt = None
                if IS_THINT:
                    # GT 3D giong jotr_evaluation: H36M-17 regress tu GT mesh, root-relative, met
                    mesh = targets['smpl_mesh_cam'].to(DEVICE).float()
                    reg = torch.as_tensor(ds.h36m_joint_regressor, dtype=torch.float32, device=DEVICE)
                    gt = torch.einsum('jk,bkc->bjc', reg, mesh)
                    gt = gt - gt[:, 0:1]
                with torch.no_grad():
                    outs = [fwd(model, img, j2, jm, False, gt, hint_on=h, raw=jraw) for h in ((1.0, 0.0) if IS_THINT else (None,))]
                fin = all(torch.isfinite(o['smpl_mesh_cam']).all().item() for o in outs)
                rng = (j2[..., :2].min().item(), j2[..., :2].max().item())
                tag = ' (CO + KHONG goi y)' if IS_THINT else ''
                all_ok &= ok(fin, f"{name}: {len(ds)} mau | joints {tuple(j2.shape)}, mask {tuple(jm.shape)} "
                                  f"(ti le hop le {jm.mean().item():.2f}) | 2D trong [{rng[0]:.2f}, {rng[1]:.2f}] | forward OK{tag}")
            except Exception as e:   # bao loi ro rang thay vi dung ca script
                all_ok &= ok(False, f'{name}: nap / forward loi -> {type(e).__name__}: {e}')

    print('\n' + ('TAT CA DEU OK' if all_ok else 'CO LOI - xem cac dong [LOI] o tren'))


if __name__ == '__main__':
    main()
