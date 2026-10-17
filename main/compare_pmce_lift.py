"""So sanh pose lift cua PMCE (GraphormerNet, checkpoint train tren Human36M 17 khop) voi MotionBERT hien tai.

Ho tro 2 split test, dung chung quy uoc (17 khop H36M, root-relative, mm) nhu main/compare_motionbert_gt2d.py:
  --split 3dpw      : 2D dau vao la 2D THUC TE (raw_joint_coord_img, cung nguon voi inputs['joints']/'halpe').
                       So sanh truc tiep voi 2 dong da co cua MotionBERT: halpe=72.50mm / gt2d=63.17mm
                       (xem 01_PROJECT_CONTEXT.md). PMCE la CHEO-DOMAIN (chi train Human3.6M, khong thay 3DPW).
  --split Human36M  : giao thuc GT 2D SACH (oracle) co san cua tap test H36M (S9, S11, cam 4) -- xem
                       data_final/Human36M/dataset.py nhanh "if self.data_split != 'train'". KHONG cheo-domain
                       (PMCE train chinh tren H36M) nhung 2D la oracle, khong phai detector thuc -> KHONG so
                       sanh truc tiep so voi dong '3dpw' (2 giao thuc 2D khac nhau).

Khac MotionBERT:
  - GraphormerNet nhan THEM dac trung anh (SPIN ResNet-50, global-average-pool -> 2048-d) dua vao moi khop,
    khong chi 2D don thuan.
  - PMCE train voi seqlen=16 frame lien tiep. Hien FuseKin chua co frame lien tiep thuc te -> nhan ban 1 anh
    thanh 16 ban giong nhau (dung dung hanh vi fallback goc cua PMCE khi start_index==end_index,
    xem PMCE/Human36M/dataset.py get_single_item/get_sequence).
  - 2D dau vao phai chuan hoa theo quy uoc cua PMCE: normalize_screen_coordinates(X, w, h) = X/w*2 - [1, h/w],
    dung W,H cua ANH GOC (truoc crop) -- khac han 2D chuan hoa [-1,1] theo khung crop ma FuseKin dung noi bo.
    Script nay dung meta['raw_joint_coord_img'] (2D pixel ANH GOC) + meta['img_shape'] (= (h, w) ANH GOC),
    2 truong moi them vao data_final/PW3D/dataset.py + lib/utils/jotr_dataset.py, chi co khi
    dataset.keep_vis_info = True (bat trong script nay, KHONG doi hanh vi mac dinh cua dataset).
  - PMCE goc dung crop 224x224, FuseKin dung crop 256x256. Vi dung global-average-pool tren toan bo feature map
    (khong phai AvgPool2d kernel 7x7 co dinh cua PMCE) nen chenh lech kich thuoc crop nay duoc coi la xap xi
    chap nhan duoc (da thong nhat voi nguoi dung, xem phan "huong moi" trong 01_PROJECT_CONTEXT.md).
  - Chi so sanh dong "2D thuc te" (tuong duong dong 'halpe' cua compare_motionbert_gt2d.py), dung
    raw_joint_coord_img + inputs['joints_mask'] lam 2D/conf dau vao. Dong 'gt2d' oracle cho PMCE la buoc
    tiep theo, chua lam trong script nay.
  - Checkpoint PMCE goc (Human3.6M) thuong xuat pose3d theo milimet, trong khi GT cua FuseKin (orig_joint_cam /
    tu smpl_mesh_cam) theo met -> script tu phat hien ti le qua do dai xuong dui (khop 1-0) VA IN RO ra man hinh
    de nguoi dung tu kiem tra, khong am tham gia dinh (giong cach lam cua main/debug_pose_estimation.py).

Chay (tu thu muc goc repo):
  python main/compare_pmce_lift.py --cfg ./config/finetune_motionbert_halpe26.yaml \
      --pmce_ckpt /path/to/pmce_human36m_posenet.pth.tar \
      --split 3dpw --max_samples 2000

Checkpoint PMCE: dang format {'epoch','model_state_dict','optim_state_dict','scheduler_state_dict','train_log','test_log'}
(xem PMCE/base.py Trainer/Tester). Script tu bo prefix 'module.' neu co va nap voi strict=False, in ro
missing/unexpected keys de nguoi dung kiem tra dung loai checkpoint (17 khop human36, KHONG phai 19 khop smplcoco).
"""
import os, sys, json
sys.path.append('./lib')
sys.path.append('./')
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from core.config import cfg, update_config

JOINTS = ['Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose',
          'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist']

SEQLEN = 16  # co dinh theo checkpoint PMCE (num_frames hardcode trong lib/models/PoseEstimation.py GraphormerNet)


def normalize_screen_coordinates_np(xy, h, w):
    """Quy uoc PMCE (PMCE/Human36M/dataset.py): X/w*2 - [1, h/w]. xy: (..., 2) pixel toa do ANH GOC."""
    offset = np.array([1.0, h / w], dtype=np.float32)
    return xy / w * 2.0 - offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='config/finetune_motionbert_halpe26.yaml',
                    help='Dung chung cfg voi compare_motionbert_gt2d.py de cung giao thuc du lieu (lift_only, pw3d_2d_source, ...)')
    ap.add_argument('--pmce_ckpt', required=True,
                    help="Checkpoint GraphormerNet cua PMCE (da train tren Human36M, 17 khop, hpe_dim=256, hpe_dep=3, seqlen=16)")
    ap.add_argument('--spin_ckpt', default='data_final/base_data/spin_model_checkpoint.pth.tar',
                    help='Checkpoint SPIN ResNet-50 dung de trich dac trung anh (giong backbone chinh cua FuseKin)')
    ap.add_argument('--split', default='3dpw', help="'3dpw' (2D thuc te) hoac 'Human36M' (2D GT sach, oracle)")
    ap.add_argument('--max_samples', type=int, default=2000, help='<=0: toan bo; lay mau trai deu')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gpu', default='0')
    args, _ = ap.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    update_config(args.cfg)
    cfg.DATASET.workers = args.workers
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda')

    # QUAN TRONG: khac voi compare_motionbert_gt2d.py (chi can 2D, khong doc anh), GraphormerNet can ANH THUC
    # de trich dac trung SPIN. cfg.DATASET.lift_only=True (thuong duoc bat trong cfg finetune MotionBERT) se
    # lam PW3D bo qua viec doc anh (img = torch.zeros(1), xem data_final/PW3D/dataset.py) -> ep ve False o day,
    # doc lap voi gia tri trong file --cfg, de dam bao luon co anh thuc.
    if cfg.DATASET.lift_only:
        print(f'[cfg] lift_only=True trong {args.cfg} -> ep ve False de PW3D doc anh thuc cho SPIN backbone.')
    cfg.DATASET.lift_only = False

    # ---------------- Models ----------------
    from models.backbones.resnet import ResNetBackbone
    from models.PoseEstimation import get_model as get_pmce_model

    print('==> Nap SPIN ResNet-50 (trich dac trung anh, global-average-pool)...')
    backbone = ResNetBackbone(resnet_type=50)
    backbone.load_pretrained(source='spin', spin_checkpoint=args.spin_ckpt)
    backbone = backbone.to(device).eval()

    print('==> Khoi tao GraphormerNet (PMCE) va nap checkpoint...')
    pmce = get_pmce_model(num_joint=len(JOINTS), embed_dim=cfg.MODEL.hpe_dim, depth=cfg.MODEL.hpe_dep, pretrained=False)
    pmce = pmce.to(device)
    ckpt = torch.load(args.pmce_ckpt, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
    missing, unexpected = pmce.load_state_dict(state_dict, strict=False)
    print(f'  [ckpt] epoch={ckpt.get("epoch", "?")} | missing={len(missing)} {missing[:5]} | '
          f'unexpected={len(unexpected)} {unexpected[:5]}')
    if missing or unexpected:
        print('  CANH BAO: nap khong khop hoan toan -- kiem tra lai checkpoint co dung la ban 17 khop '
              '(human36), hpe_dim=256, hpe_dep=3 hay khong (ban 19 khop smplcoco se lech shape spatial_pos_embed).')
    pmce = pmce.eval()

    # ---------------- Data ----------------
    from utils.jotr_dataset import get_test_dataset
    dataset = get_test_dataset(args.split, None)
    dataset.keep_vis_info = True   # bat de lay raw_joint_coord_img + img_shape qua meta (xem lib/utils/jotr_dataset.py)
    n_total = len(dataset)
    if args.max_samples <= 0 or args.max_samples >= n_total:
        idxs = np.arange(n_total)
    else:
        idxs = np.linspace(0, n_total - 1, args.max_samples).astype(int)
    loader = DataLoader(Subset(dataset, idxs.tolist()), batch_size=args.batch_size, shuffle=False,
                        num_workers=cfg.DATASET.workers, pin_memory=True)
    reg = torch.as_tensor(dataset.dataset.h36m_joint_regressor, dtype=torch.float32, device=device)
    print(f'[data] {args.split}: dung {len(idxs)}/{n_total} mau, seqlen={SEQLEN} (nhan ban 1 anh)')

    if dataset[0][2].get('raw_joint_coord_img') is None:
        raise SystemExit("meta khong co 'raw_joint_coord_img'. Kiem tra data_final/PW3D/dataset.py + "
                         "lib/utils/jotr_dataset.py da duoc cap nhat (VIS_META_KEYS) va dataset.keep_vis_info=True.")

    err_pj, pa_err, mask_sum = [], [], np.zeros(len(JOINTS))
    bonelen_ratio_log = []
    n_seen = 0

    # import batch_pa_errors tu finetune_motionbert.py de dung chung cong thuc PA-MPJPE voi compare_motionbert_gt2d.py
    from finetune_motionbert import batch_pa_errors

    with torch.no_grad():
        for bi, (inputs, targets, meta) in enumerate(tqdm(loader, desc='PMCE GraphormerNet')):
            B = inputs['img'].shape[0]

            # ---- 3D GT (giong compare_motionbert_gt2d.py) ----
            if 'orig_joint_cam' in targets:
                gt3d = targets['orig_joint_cam'].to(device).float()
            else:
                gt3d = torch.matmul(reg[None], targets['smpl_mesh_cam'].to(device).float())
            gt3d = gt3d - gt3d[:, 0:1, :]   # root-relative, met

            # ---- Dac trung anh: SPIN global-average-pool, nhan ban qua T=16 frame ----
            img = inputs['img'].to(device).float()
            _, feat_token = backbone(img)                       # feat_token: (B, 1, 2048)
            global_feat = feat_token.view(B, -1)                 # (B, 2048)
            img_feat = global_feat.unsqueeze(1).repeat(1, SEQLEN, 1)   # (B, T, 2048)

            # ---- 2D dau vao: raw_joint_coord_img (pixel ANH GOC) -> normalize_screen_coordinates(h, w) ----
            raw_xy = meta['raw_joint_coord_img'][..., :2].numpy().astype(np.float32)   # (B, 17, 2) pixel anh goc
            # data['img_shape'] = (img_height, img_width) la tuple 2 so (xem data_final/PW3D/dataset.py) ->
            # DataLoader default_collate tra ve TUPLE 2 tensor (B,), KHONG phai 1 tensor (B,2).
            img_h = meta['img_shape'][0].numpy().astype(np.float32)                    # (B,)
            img_w = meta['img_shape'][1].numpy().astype(np.float32)                    # (B,)
            conf = inputs['joints_mask'].numpy().astype(np.float32)                    # (B, 17, 1) -- cung nguon voi raw_xy

            xy_norm = np.empty_like(raw_xy)
            for b in range(B):
                xy_norm[b] = normalize_screen_coordinates_np(raw_xy[b], img_h[b], img_w[b])
            xy_norm = xy_norm - xy_norm[:, 0:1, :]            # root-relative, giong PMCE/Human36M/dataset.py
            xy_norm = xy_norm * (conf[..., 0:1] > 0)           # khop khong hop le -> 0 (giong jotr_dataset.py)

            x = torch.from_numpy(xy_norm).to(device).float().unsqueeze(1).repeat(1, SEQLEN, 1, 1)   # (B, T, 17, 2)

            pred = pmce(x, img_feat)                           # (B, 17, 3)
            pred_rel = pred - pred[:, 0:1, :]

            # ---- Tu phat hien ti le mm/met qua do dai xuong dui (khop 1-0), giong debug_pose_estimation.py ----
            gt_bone = torch.norm(gt3d[:, 1] - gt3d[:, 0], dim=-1)          # (B,) met
            pred_bone = torch.norm(pred_rel[:, 1] - pred_rel[:, 0], dim=-1)
            ratio = (pred_bone / gt_bone.clamp_min(1e-6)).mean().item()
            bonelen_ratio_log.append(ratio)
            if ratio > 10:
                pred_rel = pred_rel / 1000.0   # pred dang la mm -> doi ve met cho khop voi GT

            err_pj.append((torch.norm(pred_rel - gt3d, dim=-1) * 1000).cpu().numpy())   # mm
            pa_err.append((batch_pa_errors(pred_rel, gt3d) * 1000).cpu().numpy())        # mm
            mask_sum += (conf[..., 0] > 0).sum(0)
            n_seen += B

    err_pj = np.concatenate(err_pj)      # (N, 17) mm
    pa_err = np.concatenate(pa_err)      # (N,) mm
    mpjpe = err_pj.mean(1)               # (N,)
    avg_ratio = float(np.mean(bonelen_ratio_log))

    # ---------------- In ket qua ----------------
    is_3dpw = args.split.startswith('3dpw')
    row_label = 'pmce (raw2d)' if is_3dpw else 'pmce (gt2d-oracle)'
    print('\n' + '=' * 78)
    print(f'PMCE GraphormerNet lift | {args.split} | n={n_seen} | 17 khop H36M, root-relative (mm)')
    print(f'(ti le do dai xuong dui pred/gt trung binh = {avg_ratio:.2f} -> '
          f'{"da chia pred cho 1000 (mm->m)" if avg_ratio > 10 else "KHONG doi ti le (coi pred da la met)"})')
    print('=' * 78)
    print(f'{"dau vao 2D":<20}{"MPJPE":>10}{"PA-MPJPE":>11}{"khop co mat":>14}')
    print(f'{row_label:<20}{mpjpe.mean():>10.2f}{pa_err.mean():>11.2f}{mask_sum.mean() / n_seen * 100:>13.1f}%')
    if is_3dpw:
        print('\n  So sanh voi MotionBERT hien tai (xem 01_PROJECT_CONTEXT.md, cung tap test 3dpw, cung giao thuc 2D thuc te):')
        print('    MotionBERT halpe (2D detector thuc): 72.50 mm')
        print('    MotionBERT gt2d  (2D GT oracle)     : 63.17 mm')
        print('  LUU Y: PMCE checkpoint train CHEO-DOMAIN (chi Human3.6M, KHONG thay 3DPW luc train) -> ket qua nay')
        print('  phan anh kha nang GENERALIZE, khong phai so sanh cong bang 100% ve du lieu train.')
    else:
        print('\n  LUU Y: day la giao thuc GT 2D SACH (oracle) co san cua tap test H36M -- KHONG CUNG giao thuc 2D')
        print('  voi dong "pmce (raw2d)" cua --split 3dpw (2D thuc te). KHONG so sanh truc tiep 2 split nay voi nhau.')
        print('  PMCE KHONG cheo-domain o day (checkpoint train chinh tren H36M) -> ket qua nay la "san nha" cua PMCE.')

    print(f'\n{"khop":<12}{"pmce (mm)":>12}')
    for j, jn in enumerate(JOINTS):
        print(f'{jn:<12}{err_pj[:, j].mean():>12.1f}')

    out = {'pmce_ckpt': args.pmce_ckpt, 'split': args.split, 'n': int(n_seen), 'cfg': args.cfg,
           'seqlen': SEQLEN, 'bonelen_ratio': avg_ratio,
           'mpjpe_mm': float(mpjpe.mean()), 'pa_mpjpe_mm': float(pa_err.mean()),
           'per_joint_mm': err_pj.mean(0).tolist()}
    os.makedirs(cfg.res_dir, exist_ok=True)
    path = os.path.join(cfg.res_dir, 'compare_pmce_lift.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nDa luu: {path}')


if __name__ == '__main__':
    main()
