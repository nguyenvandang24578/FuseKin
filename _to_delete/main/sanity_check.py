"""
sanity_check.py — Kiem tra "suc khoe ky thuat" cua model Teacher/Student (FuseKin)
====================================================================================

Muc dich: tra loi cau hoi "model co dung ky thuat khong, gradient co on khong"
MA KHONG phu thuoc vao dataset that, learning-rate schedule, hay so epoch da train.
Day la bo kiem tra tieu chuan trong deep learning debugging (tham khao: Karpathy's
"A Recipe for Training Neural Networks" - buoc "overfit one batch").

Script nay dung du lieu GIA (synthetic, sinh ngau nhien dung shape) de co lap hoan
toan nguyen nhan: neu model khong overfit duoc 1 batch gia (vua sinh, target chac
chan nam trong khong gian ma model co the bieu dien duoc), thi loi nam o kien truc /
gradient / cau hinh freeze — KHONG lien quan gi toi du lieu that, KD, hay AWL.

Script thuc hien 5 buoc kiem tra, theo thu tu tu re -> dat:

  1. Parameter audit       : liet ke tham so nao dong bang / co the train, canh bao
                              neu khac voi ky vong (backbone, pose_lifter, vposer phai
                              dong bang; fusion/hypergcn/heads phai train duoc).
  2. Forward sanity         : chay 1 forward pass, kiem tra NaN/Inf trong moi output.
  3. Gradient audit         : chay backward voi 1 loss "tong hop" tren moi output
                              chinh, roi kiem tra TUNG tham so co the train xem:
                                - grad = None       -> THAM SO CHET (khong noi vao graph)
                                - grad toan so 0     -> NGHI NGO (khong nhan tin hieu)
                                - grad co NaN/Inf     -> GAY (vo cuc / explode)
                                - grad binh thuong    -> OK
  4. Input-sensitivity test : so sanh output khi dua 2 bo khop 3D/2D RAT KHAC NHAU
                              vao (eval mode, khong dropout). Neu output gan nhu
                              khong doi -> model dang "lo" (collapse ve 1 pose trung
                              binh bat ke input).
  5. Single-batch overfit   : test chuan nhat. Sinh 1 target (pose/shape/mesh) CHAC
                              CHAN nam trong khong gian model co the dat toi (vi target
                              duoc tao ra tu chinh VPoser + SMPL cua model), roi train
                              lien tuc tren DUY NHAT 1 batch trong vai tram buoc voi LR
                              cao. Neu model khong the ep loss giam manh (>90%) tren
                              chinh 1 batch no da thay, thi chac chan co van de ve
                              gradient/kien truc/freeze — khong phai do thieu du lieu,
                              thieu epoch, hay KD chua du tot.

Cach chay:
    cd <repo_root>
    python main/sanity_check.py --cfg <duong_dan_toi_config_yaml_cua_ban>

Vi du:
    python main/sanity_check.py --cfg assets/configs/student.yml
    python main/sanity_check.py --cfg assets/configs/teacher.yml --steps 500 --lr 3e-3

Luu ý: script doi hoi CUDA vi mot so module trong repo (Vposer, SMPL layer...) co
goi .cuda() cung trong code, khong chay duoc tren CPU (giong het yeu cau khi train
binh thuong).
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='FuseKin sanity check (gradient / kien truc)')
parser.add_argument('--cfg', type=str, required=True, help='duong dan file config yaml (giong khi train)')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--steps', type=int, default=300, help='so buoc optimize cho overfit test')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate dung RIENG cho overfit test (cao hon luc train that)')
parser.add_argument('--print_every', type=int, default=25)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401  (dam bao sys.path giong cac script khac trong main/)
import models
from utils.transforms import rot6d_to_axis_angle

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not torch.cuda.is_available():
    print("\n[LOI] Khong tim thay CUDA. Mot so module trong repo (Vposer, SMPL layer) "
          "goi .cuda() cung trong code nen BAT BUOC phai chay tren GPU, giong het luc "
          "train binh thuong. Hay chay script nay tren may/server co GPU.\n")
    sys.exit(1)

DEVICE = torch.device('cuda')


def hr(title=''):
    print('\n' + '=' * 78)
    if title:
        print(title)
        print('=' * 78)


# --------------------------------------------------------------------------
# Xay model giong het prepare_network() nhung KHONG dong dataset that
# --------------------------------------------------------------------------
def build_model():
    model = models.ARTS.get_model(
        num_joint=17, embed_dim=cfg.MODEL.hpe_dim, depth=cfg.MODEL.hpe_dep
    )
    return model.to(DEVICE)


# --------------------------------------------------------------------------
# Sinh batch gia (synthetic) dung shape voi pipeline that
# --------------------------------------------------------------------------
def make_fake_batch(batch_size):
    img = torch.rand(batch_size, 3, cfg.input_img_shape[0], cfg.input_img_shape[1], device=DEVICE)
    # 2D input: dataset thuc te normalize ve [-1, 1] truoc khi dua vao model (xem
    # Human36M17Dataset trong utils/jotr_dataset.py) -> mo phong dung pham vi do.
    joints2d = (torch.rand(batch_size, 17, 2, device=DEVICE) * 2 - 1)
    joints_mask = (torch.rand(batch_size, 17, 1, device=DEVICE) > 0.1).float()
    # GT 3D: don vi met, root-relative, bien do nho (~20-40cm quanh root) giong du lieu that.
    gt_pose3d = (torch.rand(batch_size, 17, 3, device=DEVICE) - 0.5) * 0.4
    gt_pose3d[:, 0, :] = 0.0  # root luon = 0 (root-relative)
    return img, joints2d, joints_mask, gt_pose3d


def run_model(model, batch, is_train=True):
    img, joints2d, joints_mask, gt_pose3d = batch
    if cfg.MODEL.name == 'teacher':
        return model(img, gt_pose3d, is_train=is_train)
    elif cfg.MODEL.name == 'student':
        return model(img, joints2d, is_train=is_train, joints_mask=joints_mask)
    else:
        # ARTS / mode khac: best-effort, giong duong goi cua student
        return model(img, joints2d, is_train=is_train, joints_mask=joints_mask)


# --------------------------------------------------------------------------
# 1) Parameter audit
# --------------------------------------------------------------------------
def audit_parameters(model):
    hr("[1/5] PARAMETER AUDIT — tham so nao dong bang / co the train")
    frozen, trainable = [], []
    for name, p in model.named_parameters():
        (trainable if p.requires_grad else frozen).append(name)

    total = len(frozen) + len(trainable)
    print(f"Tong so tensor tham so : {total}")
    print(f"  - Dong bang (requires_grad=False) : {len(frozen)}")
    print(f"  - Co the train (requires_grad=True): {len(trainable)}")

    expect_frozen_prefixes = ('backbone.', 'pose_lifter.', 'smpl_model.vposer.')
    unexpected_frozen = [n for n in frozen if not n.startswith(expect_frozen_prefixes)]
    unexpected_trainable = [n for n in trainable if n.startswith(expect_frozen_prefixes)]

    if unexpected_frozen:
        print(f"\n[CANH BAO] {len(unexpected_frozen)} tensor BI DONG BANG ngoai du kien "
              f"(ky vong chi backbone/pose_lifter/vposer dong bang):")
        for n in unexpected_frozen[:15]:
            print('    -', n)
        if len(unexpected_frozen) > 15:
            print(f'    ... va {len(unexpected_frozen) - 15} tensor khac')
    else:
        print("\n[OK] Khong co tham so nao bi dong bang ngoai du kien.")

    if unexpected_trainable:
        print(f"\n[CANH BAO] {len(unexpected_trainable)} tensor thuoc backbone/pose_lifter/"
              f"vposer nhung lai co requires_grad=True (dang ra phai dong bang):")
        for n in unexpected_trainable[:15]:
            print('    -', n)
    else:
        print("[OK] backbone / pose_lifter / vposer deu dong bang dung nhu thiet ke.")

    return trainable


# --------------------------------------------------------------------------
# 2) Forward sanity (NaN/Inf)
# --------------------------------------------------------------------------
def check_forward_nan(out):
    bad = []
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any().item() or torch.isinf(v).any().item():
                bad.append(k)
    return bad


# --------------------------------------------------------------------------
# 3) Gradient audit
# --------------------------------------------------------------------------
PROBE_OUTPUT_KEYS = [
    'smpl_pose', 'smpl_shape', 'cam_param', 'smpl_mesh_cam',
    'joint_proj', 'joint_proj_det', 'joint_cam',
    'feat_joint', 'feat_img', 'feat_joint_in', 'feat_global',
    'privileged_3d',
]


def generic_probe_loss(out):
    terms = []
    for key in PROBE_OUTPUT_KEYS:
        v = out.get(key, None)
        if isinstance(v, torch.Tensor):
            terms.append(v.float().pow(2).mean())
    assert terms, "Khong tim thay output nao de tinh probe loss — kiem tra lai forward()."
    return sum(terms)


def audit_gradients(model):
    dead, zero, bad, ok = [], [], [], 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            dead.append(name)
        elif torch.isnan(p.grad).any().item() or torch.isinf(p.grad).any().item():
            bad.append(name)
        elif p.grad.abs().sum().item() == 0.0:
            zero.append(name)
        else:
            ok += 1
    return dead, zero, bad, ok


# --------------------------------------------------------------------------
# 4) Input-sensitivity test
# --------------------------------------------------------------------------
def input_sensitivity_test(model, batch_size):
    model.eval()
    with torch.no_grad():
        img = torch.rand(batch_size, 3, cfg.input_img_shape[0], cfg.input_img_shape[1], device=DEVICE)
        mask = torch.ones(batch_size, 17, 1, device=DEVICE)

        jointsA2d = (torch.rand(batch_size, 17, 2, device=DEVICE) * 2 - 1)
        jointsB2d = (torch.rand(batch_size, 17, 2, device=DEVICE) * 2 - 1)
        gtA3d = (torch.rand(batch_size, 17, 3, device=DEVICE) - 0.5) * 0.4; gtA3d[:, 0, :] = 0.0
        gtB3d = (torch.rand(batch_size, 17, 3, device=DEVICE) - 0.5) * 0.4; gtB3d[:, 0, :] = 0.0

        outA = run_model(model, (img, jointsA2d, mask, gtA3d), is_train=False)
        outB = run_model(model, (img, jointsB2d, mask, gtB3d), is_train=False)

        pose_diff = (outA['smpl_pose'] - outB['smpl_pose']).abs().mean().item()
        mesh_diff_mm = (outA['smpl_mesh_cam'] - outB['smpl_mesh_cam']).abs().mean().item() * 1000.0
    model.train()
    return pose_diff, mesh_diff_mm


# --------------------------------------------------------------------------
# 5) Single-batch overfit test — bai test "vang" cho gradient dung
# --------------------------------------------------------------------------
def make_reachable_target(core_model, batch_size):
    """Sinh target CHAC CHAN nam trong khong gian model co the dat toi, bang cach
    dung chinh VPoser + SMPL layer cua model de tao ra no (khong phai target random
    tuy tien ma VPoser khong the decode dung)."""
    with torch.no_grad():
        target_latent = torch.randn(batch_size, 32, device=DEVICE) * 0.7
        target_body_pose = core_model.vposer(target_latent)                     # (B, 69)

        target_root_6d = torch.randn(batch_size, 6, device=DEVICE) * 0.5
        target_root_pose = rot6d_to_axis_angle(target_root_6d)                  # (B, 3)

        target_full_pose = torch.cat([target_root_pose, target_body_pose], dim=1)  # (B, 72)
        target_shape = torch.randn(batch_size, 10, device=DEVICE) * 0.5
        target_cam = torch.zeros(batch_size, 3, device=DEVICE)
        target_cam[:, 2] = 0.5  # tz hop ly (don vi theo get_camera_trans cua model)

        _, _, target_mesh, _ = core_model.get_coord(target_full_pose, target_shape, target_cam)
    return target_full_pose, target_shape, target_cam, target_mesh


def overfit_one_batch(model, batch, steps, lr, print_every):
    core_model = getattr(model, 'smpl_model', None)
    if core_model is None:
        print("[BO QUA] cfg.MODEL.name khong phai 'teacher'/'student' (khong co "
              "model.smpl_model) — overfit test chi ho tro 2 mode nay.")
        return None

    B = batch[0].shape[0]
    target_full_pose, target_shape, target_cam, target_mesh = make_reachable_target(core_model, B)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    model.train()
    loss_history = []
    for step in range(steps):
        optimizer.zero_grad()
        out = run_model(model, batch, is_train=True)
        loss = (
            F.l1_loss(out['smpl_mesh_cam'], target_mesh)
            + F.l1_loss(out['smpl_pose'], target_full_pose)
            + F.l1_loss(out['smpl_shape'], target_shape)
            + F.l1_loss(out['cam_param'], target_cam)
        )
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if step % print_every == 0 or step == steps - 1:
            mesh_err_mm = (out['smpl_mesh_cam'] - target_mesh).abs().mean().item() * 1000.0
            print(f"  step {step:4d}/{steps} | loss={loss.item():.5f} | mesh_L1={mesh_err_mm:.2f} mm")

    return loss_history


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    print(f"cfg.MODEL.name = '{cfg.MODEL.name}'  |  embed_dim = {cfg.MODEL.hpe_dim}  |  "
          f"input_img_shape = {cfg.input_img_shape}")

    model = build_model()
    trainable_names = audit_parameters(model)

    # ---- 2) forward sanity ----
    hr("[2/5] FORWARD SANITY — kiem tra NaN/Inf trong output")
    batch = make_fake_batch(args.batch_size)
    model.train()
    out = run_model(model, batch, is_train=True)
    bad_keys = check_forward_nan(out)
    if bad_keys:
        print(f"[FAIL] Cac output sau co NaN/Inf: {bad_keys}")
    else:
        print("[OK] Khong co NaN/Inf trong bat ky output nao sau 1 forward pass.")

    # ---- 3) gradient audit ----
    hr("[3/5] GRADIENT AUDIT — backward voi probe loss tren moi output chinh")
    model.zero_grad(set_to_none=True)
    probe_loss = generic_probe_loss(out)
    probe_loss.backward()
    dead, zero, bad, ok = audit_gradients(model)
    print(f"Tham so co the train: {len(trainable_names)}")
    print(f"  - OK (grad binh thuong)      : {ok}")
    print(f"  - CHET (grad = None)         : {len(dead)}")
    print(f"  - NGHI NGO (grad toan so 0)  : {len(zero)}")
    print(f"  - GAY (grad co NaN/Inf)      : {len(bad)}")
    if dead:
        print("\n  Vi du tham so CHET (khong noi vao graph tinh loss nay):")
        for n in dead[:15]:
            print('    -', n)
    if zero:
        print("\n  Vi du tham so NGHI NGO (grad = 0 het, co the do input gia qua don gian):")
        for n in zero[:15]:
            print('    -', n)
    if bad:
        print("\n  Tham so GAY (NaN/Inf trong grad — can xu ly ngay):")
        for n in bad[:15]:
            print('    -', n)

    # ---- 4) input sensitivity ----
    hr("[4/5] INPUT-SENSITIVITY TEST — doi input khop 3D/2D rat khac nhau, output co doi khong")
    pose_diff, mesh_diff_mm = input_sensitivity_test(model, args.batch_size)
    print(f"  |pose_A - pose_B| (axis-angle, rad) trung binh: {pose_diff:.6f}")
    print(f"  |mesh_A - mesh_B| trung binh                 : {mesh_diff_mm:.4f} mm")
    if mesh_diff_mm < 1.0:
        print("  [CANH BAO] Mesh gan nhu KHONG DOI du input khop khac nhau hoan toan "
              "-> nghi ngo model dang collapse ve 1 pose trung binh, bat ke input.")
    else:
        print("  [OK] Output thay doi ro ret theo input -> model co phan ung voi du lieu dau vao.")

    # ---- 5) overfit 1 batch ----
    hr("[5/5] SINGLE-BATCH OVERFIT TEST — bai test 'vang' cho gradient dung")
    print(f"Chay {args.steps} buoc Adam (lr={args.lr}) tren DUY NHAT 1 batch ({args.batch_size} sample).")
    print("Target duoc sinh tu chinh VPoser + SMPL cua model nen CHAC CHAN nam trong")
    print("khong gian model co the dat toi — neu loss khong giam manh, do la BUG that.\n")
    loss_history = overfit_one_batch(model, batch, args.steps, args.lr, args.print_every)

    # ---- tong ket ----
    hr("TONG KET")
    issues = []
    if bad_keys:
        issues.append(f"- {len(bad_keys)} output co NaN/Inf ngay o forward dau tien.")
    if dead:
        issues.append(f"- {len(dead)} tham so KHONG BAO GIO nhan gradient (chet/disconnect khoi graph).")
    if bad:
        issues.append(f"- {len(bad)} tham so co gradient NaN/Inf (numerically unstable).")
    if mesh_diff_mm < 1.0:
        issues.append("- Output gan nhu khong doi theo input (nghi ngo collapse / bottleneck).")

    if loss_history is not None:
        init_loss, final_loss = loss_history[0], loss_history[-1]
        drop_pct = 100.0 * (1 - final_loss / max(init_loss, 1e-8))
        print(f"Overfit test: loss {init_loss:.5f} -> {final_loss:.5f}  (giam {drop_pct:.1f}%)")
        if drop_pct < 50:
            issues.append(
                f"- Overfit 1 batch CHUA DAT (chi giam {drop_pct:.1f}%). Day la dau hieu RO RANG "
                f"nhat cho thay co van de ve gradient/kien truc/freeze — KHONG phai do thieu du "
                f"lieu, thieu epoch, hay KD/AWL chua toi uu."
            )
        elif drop_pct < 90:
            issues.append(
                f"- Overfit 1 batch co hoc duoc nhung CHAM (giam {drop_pct:.1f}%). Co the do "
                f"bottleneck (VD: layer_scale init qua nho, hoac pose query/VPoser han che bieu "
                f"dien), hoac can nhieu buoc/lr cao hon de hoi tu het."
            )
        else:
            print("[OK] Model overfit 1 batch rat tot -> kien truc va gradient VE CO BAN la dung "
                  "ky thuat. Neu MPJPE that van phang, nguyen nhan nam o PHIA DU LIEU/CHIEN LUOC "
                  "HOC (vi du: GT 3D nhieu/it da dang, AWL can chinh trong so, learning rate/epoch, "
                  "chat luong input tu MotionBERT) chu khong phai o ban than kien truc model.")

    if issues:
        print("\nCac van de phat hien duoc:")
        for it in issues:
            print(it)
    else:
        print("\nKhong phat hien van de ky thuat nao ve gradient/kien truc trong cac test tren.")


if __name__ == '__main__':
    main()
