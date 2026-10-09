"""
content_ablation_test.py — Anh co mang tin hieu dung duoc khong? (FuseKin, eval-only)
===========================================================================================

Muc dich: tra loi CHINH XAC cau hoi "anh co dang dong gop gi vao du doan cuoi cung khong,
hay fusion/cross-attention da bo qua NOI DUNG anh hoan toan" — tren CHECKPOINT DA TRAIN
san, khong can retrain (chay trong vai phut).

Cach lam: thay feature_map that (dau ra backbone) bang torch.randn_like(feature_map) NGAY
TRUOC khi dua vao RGBJointFusion — dung het cau truc (shape, pos_emb_img, LayerNorm) nhung
xoa SACH noi dung anh. Giong het code ban da viet de retrain Teacher, chi khac la chay o
EVAL MODE tren checkpoint co san, khong train lai.

Khac voi shuffle test (da chay o compare_teacher_student.py — tron ANH THAT giua cac mau,
GT giu nguyen): shuffle chi tra loi "model co phan biet duoc anh CUA MAU NAO khong". Con
ablation nay tra loi cau hoi manh hon: "model co doc duoc NOI DUNG anh (bat ky anh nao) hay
khong, hay no dang coi moi token anh nhu nhau bat ke pixel ben trong la gi".

Dieu kien quyet dinh:
  - Neu MPJPE(blind) ~ MPJPE(that) (gan nhu khong doi) -> nhanh anh dang "CHET VE NOI DUNG":
    kien truc/gradient khong doc duoc pixel, chi co the dang dua vao pos_emb_img (vi tri)
    hoac khong dung gi tu token anh ca. KD/retrain downstream se khong giai quyet duoc,
    phai sua o CHINH nhanh fusion (vd: kiem tra gradient toi img_proj/pos_emb_img, hoac
    tang trong so loss body_joint_proj — nguon tin hieu DUY NHAT bat buoc phai dung anh).
  - Neu MPJPE(blind) te hon RO RET -> anh MANG tin hieu dung duoc that, kien truc DOC duoc
    no. Luc do cau hoi chuyen sang: tai sao shuffle test truoc lai cho thay model khong
    nhay voi anh? (Co the vi GT-joint da du de thoa man loss, nen model KHONG CAN anh de
    giam loss, du van "doc" duoc no khi bi ep phai dung — van de o INCENTIVE cua loss, khong
    phai o kien truc.)

Cach chay:
    cd <repo_root>
    python main/content_ablation_test.py \
        --cfg <config_cua_student> \
        --student_ckpt <checkpoint_student>.pth.tar \
        [--teacher_ckpt <checkpoint_teacher>.pth.tar]   # mac dinh: cfg.MODEL.TEACHER
        [--max_batches 30] [--batch_size 16]

Can CUDA.
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
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Content-ablation test (anh = randn) cho Teacher/Student (FuseKin)')
parser.add_argument('--cfg', type=str, required=True, help='config yaml (dung config cua Student)')
parser.add_argument('--student_ckpt', type=str, required=True)
parser.add_argument('--teacher_ckpt', type=str, default='', help='mac dinh: cfg.MODEL.TEACHER')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_batches', type=int, default=30)
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--blind_thresh', type=float, default=3.0,
                    help='(mm) nguong de ket luan "anh khong dong gop noi dung gi"')
args = parser.parse_args()

from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401
from models.ARTS import ARTS
from core.base import load_model_weights
from utils.jotr_dataset import get_test_dataset

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not torch.cuda.is_available():
    print("\n[LOI] Can CUDA de chay script nay.\n")
    sys.exit(1)

DEVICE = torch.device('cuda')


def hr(title=''):
    print('\n' + '=' * 78)
    if title:
        print(title)
        print('=' * 78)


def build_and_load(mode, ckpt_path):
    old_mode = cfg.MODEL.name
    cfg.MODEL.name = mode
    model = ARTS(num_joint=17, embed_dim=cfg.MODEL.hpe_dim)
    cfg.MODEL.name = old_mode
    model = model.to(DEVICE)
    model = load_model_weights(model, ckpt_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def h36m_from_mesh(mesh, regressor):
    j = torch.einsum('jk,bkc->bjc', regressor, mesh)
    return j - j[:, 0:1, :]


def mpjpe_mm(j_pred, j_gt):
    return (torch.norm(j_pred - j_gt, dim=-1).mean(-1) * 1000.0)  # (B,)


def paired(d):
    d = np.asarray(d, dtype=np.float64)
    if d.size < 2:
        return float(d.mean()), float('nan')
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(d.size))


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    return f"mean={x.mean():.2f}  median={np.median(x):.2f}  std={x.std():.2f}  n={x.size}"


def main():
    teacher_ckpt = args.teacher_ckpt or cfg.MODEL.get('TEACHER', '')
    if not teacher_ckpt:
        print("[LOI] Thieu teacher checkpoint: dung --teacher_ckpt hoac dat cfg.MODEL.TEACHER")
        sys.exit(1)

    print(f"Student ckpt : {args.student_ckpt}")
    print(f"Teacher ckpt : {teacher_ckpt}")

    hr("Dang build va nap checkpoint...")
    student = build_and_load('student', args.student_ckpt)
    teacher = build_and_load('teacher', teacher_ckpt)

    hr("Dang chuan bi tap val (3DPW test split)...")
    test_name = cfg.DATASET.test_list[0]
    dataset = get_test_dataset(test_name, None)
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=gen,
                         num_workers=args.workers, pin_memory=False)
    print(f"Dataset: {test_name} | so sample: {len(dataset)} | se duyet toi da {args.max_batches} batch")

    h36m_reg = torch.as_tensor(dataset.h36m_joint_regressor, dtype=torch.float32, device=DEVICE)

    s_real, s_blind, t_real, t_blind = [], [], [], []
    s_mean, t_mean = [], []
    n_batches = 0
    with torch.no_grad():
        for inputs, targets, meta in loader:
            if n_batches >= args.max_batches:
                break
            n_batches += 1

            img = inputs['img'].to(DEVICE).float()
            joints2d = inputs['joints'].to(DEVICE).float()
            joints_mask = inputs['joints_mask'].to(DEVICE).float()
            gt_mesh = targets['smpl_mesh_cam'].to(DEVICE).float()
            gt_h36m = h36m_from_mesh(gt_mesh, h36m_reg)

            # ---------------- Teacher ----------------
            feat_t_real, _ = teacher.get_image_features(img)
            feat_t_blind = torch.randn_like(feat_t_real)  # xoa sach noi dung anh, giu nguyen shape
            # mean-feature: 1 token duy nhat (TB kenh ca batch), ap cho MOI mau -> van "trong
            # phan phoi" (khong soc OOD nhu randn) nhung xoa sach thong tin RIENG tung mau.
            feat_t_mean = feat_t_real.mean(dim=0, keepdim=True).expand_as(feat_t_real).contiguous()

            out_t_real = teacher.smpl_model(joints=gt_h36m, img_feats=feat_t_real,
                                             is_train=False, return_features=True)
            out_t_blind = teacher.smpl_model(joints=gt_h36m, img_feats=feat_t_blind,
                                              is_train=False, return_features=True)
            out_t_mean = teacher.smpl_model(joints=gt_h36m, img_feats=feat_t_mean,
                                             is_train=False, return_features=True)
            t_real.append(mpjpe_mm(h36m_from_mesh(out_t_real['smpl_mesh_cam'], h36m_reg), gt_h36m).cpu())
            t_blind.append(mpjpe_mm(h36m_from_mesh(out_t_blind['smpl_mesh_cam'], h36m_reg), gt_h36m).cpu())
            t_mean.append(mpjpe_mm(h36m_from_mesh(out_t_mean['smpl_mesh_cam'], h36m_reg), gt_h36m).cpu())

            # ---------------- Student ----------------
            feat_s_real, _ = student.get_image_features(img)
            feat_s_blind = torch.randn_like(feat_s_real)
            feat_s_mean = feat_s_real.mean(dim=0, keepdim=True).expand_as(feat_s_real).contiguous()
            pose_3d = student.lift_2d_to_3d(joints2d, joints_mask=joints_mask)
            pose_3d = pose_3d - pose_3d[:, 0:1, :]

            out_s_real = student.smpl_model(joints=pose_3d, img_feats=feat_s_real,
                                             is_train=False, return_features=True)
            out_s_blind = student.smpl_model(joints=pose_3d, img_feats=feat_s_blind,
                                              is_train=False, return_features=True)
            out_s_mean = student.smpl_model(joints=pose_3d, img_feats=feat_s_mean,
                                             is_train=False, return_features=True)
            s_real.append(mpjpe_mm(h36m_from_mesh(out_s_real['smpl_mesh_cam'], h36m_reg), gt_h36m).cpu())
            s_blind.append(mpjpe_mm(h36m_from_mesh(out_s_blind['smpl_mesh_cam'], h36m_reg), gt_h36m).cpu())
            s_mean.append(mpjpe_mm(h36m_from_mesh(out_s_mean['smpl_mesh_cam'], h36m_reg), gt_h36m).cpu())

            if n_batches % 5 == 0:
                print(f"  ... da xu ly {n_batches}/{args.max_batches} batch")

    s_real = torch.cat(s_real).numpy(); s_blind = torch.cat(s_blind).numpy(); s_mean = torch.cat(s_mean).numpy()
    t_real = torch.cat(t_real).numpy(); t_blind = torch.cat(t_blind).numpy(); t_mean = torch.cat(t_mean).numpy()

    d_s = s_blind - s_real   # duong = blind te hon = anh co ich
    d_t = t_blind - t_real
    dm_s = s_mean - s_real   # chenh lech do mean-feature gay ra (so voi anh that)
    dm_t = t_mean - t_real
    ds_mean, ds_ci = paired(d_s)
    dt_mean, dt_ci = paired(d_t)
    dms_mean, dms_ci = paired(dm_s)
    dmt_mean, dmt_ci = paired(dm_t)

    hr("KET QUA — MPJPE (mm): anh that vs anh = nhieu trang (randn) vs anh = trung binh batch (mean)")
    print(f"  Student  | that : {summarize(s_real)}")
    print(f"           | blind (randn): {summarize(s_blind)}")
    print(f"           | mean  (TB batch): {summarize(s_mean)}")
    print(f"           | blind - that: {ds_mean:+.2f} +/- {ds_ci:.2f} mm (95%, n={len(d_s)}); "
          f"{(d_s > 0).mean():.1%} mau blind te hon")
    print(f"           | mean  - that: {dms_mean:+.2f} +/- {dms_ci:.2f} mm (95%, n={len(dm_s)}); "
          f"{(dm_s > 0).mean():.1%} mau mean te hon")
    print(f"  Teacher  | that : {summarize(t_real)}")
    print(f"           | blind (randn): {summarize(t_blind)}")
    print(f"           | mean  (TB batch): {summarize(t_mean)}")
    print(f"           | blind - that: {dt_mean:+.2f} +/- {dt_ci:.2f} mm (95%, n={len(d_t)}); "
          f"{(d_t > 0).mean():.1%} mau blind te hon")
    print(f"           | mean  - that: {dmt_mean:+.2f} +/- {dmt_ci:.2f} mm (95%, n={len(dm_t)}); "
          f"{(dm_t > 0).mean():.1%} mau mean te hon")

    hr("KET LUAN")
    def verdict(name, mean, ci, mean_feat_mean, mean_feat_ci, thresh):
        lo = mean - ci if not np.isnan(ci) else mean
        if abs(mean) < thresh and (np.isnan(ci) or lo < thresh):
            print(f"  [{name}] Xoa sach noi dung anh (randn) HAU NHU KHONG doi MPJPE "
                  f"({mean:+.2f}mm, trong nguong {thresh}mm).")
            print(f"    -> Nhanh anh cua {name} dang CHET VE NOI DUNG: kien truc khong doc duoc pixel,")
            print(f"       co the chi dang 'thay' token anh nhu nhung vi tri co dinh (qua pos_emb_img),")
            print(f"       khong phai nhu nguon thong tin thi giac. KD/retrain downstream se KHONG")
            print(f"       sua duoc dieu nay — phai kiem tra truc tiep nhanh fusion/img_proj/gradient toi")
            print(f"       anh, hoac tang trong so loss body_joint_proj (nguon duy nhat bat buoc dung anh).")
        else:
            print(f"  [{name}] Xoa noi dung anh (randn) lam MPJPE DOI RO RET ({mean:+.2f} +/- {ci:.2f}mm).")
            # ---- phan xu: do la OOD-shock hay thuc su can thong tin rieng tung mau? ----
            lo_m = mean_feat_mean - mean_feat_ci if not np.isnan(mean_feat_ci) else mean_feat_mean
            if abs(mean_feat_mean) < thresh and (np.isnan(mean_feat_ci) or lo_m < thresh):
                print(f"    -> NHUNG mean-feature (van trong phan phoi, xoa thong tin RIENG tung mau)")
                print(f"       chi doi {mean_feat_mean:+.2f}mm — GAN NHU KHONG DOI.")
                print(f"       => Ket luan: {name} KHONG thuc su can noi dung anh cua TUNG MAU. Cai lam")
                print(f"       MPJPE xau o randn la SOC OUT-OF-DISTRIBUTION (backbone pretrain tren anh")
                print(f"       that, nhieu trang pha vo thong ke LayerNorm/attention), KHONG PHAI bang")
                print(f"       chung cho viec {name} doc va dung noi dung anh mot cach co y nghia.")
            else:
                print(f"    -> VA mean-feature cung lam MPJPE xau di ro ret ({mean_feat_mean:+.2f} +/- "
                      f"{mean_feat_ci:.2f}mm).")
                print(f"       => Ket luan: {name} THUC SU can thong tin RIENG cua TUNG MAU anh, khong")
                print(f"       chi can 'co anh hop le'. Day la bang chung that cho viec kien truc doc")
                print(f"       duoc va dung noi dung anh — neu truoc do shuffle-test cho thay khong")
                print(f"       nhay voi DOI anh giua cac mau, can xem lai do nhay cua shuffle-test do.")

    verdict('Student', ds_mean, ds_ci, dms_mean, dms_ci, args.blind_thresh)
    verdict('Teacher', dt_mean, dt_ci, dmt_mean, dmt_ci, args.blind_thresh)


if __name__ == '__main__':
    main()
