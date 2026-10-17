"""So sanh pose lift cua PMCE (GraphormerNet, checkpoint train tren Human36M 17 khop) voi MotionBERT,
tren CUNG mau, CUNG 2D dau vao, CUNG giao thuc (17 khop H36M, root-relative, mm).

Split ho tro:
  --split 3dpw | 3dpw-oc | 3dpw-pc : 2D dau vao = 2D detector cua dataset (DATASET.pw3d_2d_source, vd Halpe-26).
                                     PMCE la CHEO-DOMAIN (chi train Human3.6M).
  --split Human36M                 : S9, S11, camera 4. 2D dau vao theo DATASET.h36m_2d_source:
                                     'halpe26' = 2D detector that (can Human36M_test_halpe26.npz),
                                     'gt_posefix' = GT 2D sach (oracle).
  --gt2d                           : them dong GT 2D sach (oracle) cho ca 2 model (3DPW: tu bat pw3d_test_gt2d).

Cach dua 2D vao tung model (chung 1 nguon 2D, chi khac cach chuan hoa):
  * MotionBERT: inputs['joints'] (chuan hoa [-1, 1] theo khung crop, wrapper) + joints_mask -- giong
    finetune_motionbert.py / compare_motionbert_gt2d.py.
  * PMCE      : 2D pixel ANH GOC -> normalize_screen_coordinates(X, w, h) = X/w*2 - [1, h/w], KHONG tru root
    (dung nhu PMCE/Human36M/dataset.py get_single_item; ban cu tru root la SAI -> --pmce_2d_rootrel de tai hien),
    lap 16 frame (fallback goc cua PMCE khi chi co 1 anh). 2D detector lay tu meta['raw_joint_coord_img'];
    GT 2D lay bang nghich dao phep affine crop (img2bb_trans) cua targets['orig_joint_img'].
  * PMCE them dac trung anh SPIN ResNet-50 (global-average-pool, 2048-d); crop 256 (PMCE goc 224) -> xap xi.
  * Checkpoint PMCE goc thuong xuat mm, GT FuseKin la met -> tu phat hien qua do dai xuong dui (khop 1-0), IN RA.

Chay (tu thu muc goc repo):
  python main/compare_pmce_lift.py --cfg ./config/finetune_motionbert_halpe26.yaml \
      --pmce_ckpt /path/to/pmce_human36m_posenet.pth.tar \
      --mb_ckpt experiment/finetune_motionbert_halpe26/best_epoch.bin \
      --split 3dpw --max_samples 0 --gt2d
  # H36M voi 2D Halpe: them 'h36m_2d_source: halpe26' vao muc DATASET cua yml, roi --split Human36M
Khong truyen --mb_ckpt -> chi chay PMCE (nhu ban cu).

Checkpoint PMCE: {'epoch','model_state_dict',...} (PMCE/base.py). Tu bo prefix 'module.', nap strict=False va in
missing/unexpected keys (ban 17 khop human36, KHONG phai 19 khop smplcoco).
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


def hm_to_raw_px(xy_hm, img2bb):
    """Toa do khung output_hm_shape (B, J, 2) -> pixel ANH GOC, dung nghich dao affine crop img2bb (B, 2, 3)."""
    xy = xy_hm.copy()
    xy[..., 0] = xy[..., 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    xy[..., 1] = xy[..., 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    out = np.empty_like(xy)
    for b in range(xy.shape[0]):
        A = np.concatenate([img2bb[b], np.array([[0, 0, 1]], dtype=img2bb.dtype)], axis=0)
        Ainv = np.linalg.inv(A)[:2]
        xy1 = np.concatenate([xy[b], np.ones_like(xy[b][:, :1])], axis=1)
        out[b] = xy1 @ Ainv.T
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='config/finetune_motionbert_halpe26.yaml',
                    help='Dung chung cfg voi compare_motionbert_gt2d.py (MOTIONBERT.*, pw3d/h36m_2d_source, ...)')
    ap.add_argument('--pmce_ckpt', required=True,
                    help="Checkpoint GraphormerNet cua PMCE (Human36M, 17 khop, hpe_dim=256, hpe_dep=3, seqlen=16)")
    ap.add_argument('--mb_ckpt', default='',
                    help="(Tuy chon) checkpoint MotionBERT (khoa 'model_pos') -> chay tren CUNG mau, CUNG 2D voi PMCE")
    ap.add_argument('--gt2d', action='store_true', help='Them dong GT 2D sach (oracle) cho ca 2 model')
    ap.add_argument('--pmce_2d_rootrel', action='store_true',
                    help='(Chi de tai hien ban cu) tru root 2D truoc khi dua vao PMCE. PMCE goc KHONG tru root.')
    ap.add_argument('--spin_ckpt', default='data_final/base_data/spin_model_checkpoint.pth.tar',
                    help='Checkpoint SPIN ResNet-50 dung de trich dac trung anh (giong backbone chinh cua FuseKin)')
    ap.add_argument('--split', default='3dpw', help="'3dpw' | '3dpw-oc' | '3dpw-pc' | 'Human36M'")
    ap.add_argument('--max_samples', type=int, default=2000, help='<=0: toan bo; lay mau trai deu')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gpu', default='0')
    args, _ = ap.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    update_config(args.cfg)
    cfg.DATASET.workers = args.workers
    if args.gt2d:
        cfg.DATASET.pw3d_test_gt2d = True   # 3DPW test tra them targets['orig_joint_img'] (H36M test luon co)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda')

    # GraphormerNet can ANH THUC (dac trung SPIN) -> ep lift_only = False bat ke --cfg.
    if cfg.DATASET.lift_only:
        print(f'[cfg] lift_only=True trong {args.cfg} -> ep ve False de dataset doc anh thuc cho SPIN backbone.')
    cfg.DATASET.lift_only = False

    is_h36m = args.split == 'Human36M'
    if is_h36m:
        src = cfg.DATASET.get('h36m_2d_source', 'gt_posefix')
        det_label = 'halpe' if src == 'halpe26' else 'gt2d-input'
    else:
        src = cfg.DATASET.pw3d_2d_source
        det_label = 'halpe' if src == 'halpe26' else 'openpose'
    print(f'[cfg] split={args.split} | nguon 2D dau vao = {src} (dong "{det_label}") | gt2d oracle = {args.gt2d}')
    if is_h36m and src != 'halpe26':
        print('  LUU Y: h36m_2d_source != halpe26 -> dau vao test H36M la GT 2D sach (oracle), khong phai detector.')

    # ---------------- Models ----------------
    from models.backbones.resnet import ResNetBackbone
    from models.PoseEstimation import get_model as get_pmce_model
    # dung chung ham voi finetune_motionbert.py -> cung giao thuc dau vao / PA-MPJPE voi compare_motionbert_gt2d.py
    from finetune_motionbert import build_model, make_mb_input, normalize_hm_xy, batch_pa_errors
    from utils.h36m_adapter import h36m_joint_indices

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

    mb = None
    if args.mb_ckpt:
        print('==> Nap MotionBERT...')
        mb = build_model(cfg.MOTIONBERT)
        mck = torch.load(args.mb_ckpt, map_location=lambda s, l: s)
        mb.load_state_dict(mck['model_pos'], strict=True)
        mb.eval()
        rootrel = cfg.MODEL.motionbert_2d_rootrel
        drop_idx = h36m_joint_indices(list(cfg.MODEL.motionbert_drop_joints))
        meta_ck = mck.get('fusekin_meta', {})
        print(f"  [ckpt] {args.mb_ckpt} | epoch {mck.get('epoch', '?')} | train_list: {meta_ck.get('train_list', '?')} "
              f"| rootrel_2d={rootrel} | drop={list(cfg.MODEL.motionbert_drop_joints)}")

    # ---------------- Data ----------------
    from utils.jotr_dataset import get_test_dataset
    dataset = get_test_dataset(args.split, None)
    dataset.keep_vis_info = True
    n_total = len(dataset)
    if args.max_samples <= 0 or args.max_samples >= n_total:
        idxs = np.arange(n_total)
    else:
        idxs = np.linspace(0, n_total - 1, args.max_samples).astype(int)
    loader = DataLoader(Subset(dataset, idxs.tolist()), batch_size=args.batch_size, shuffle=False,
                        num_workers=cfg.DATASET.workers, pin_memory=True)
    reg = torch.as_tensor(dataset.dataset.h36m_joint_regressor, dtype=torch.float32, device=device)
    print(f'[data] {args.split}: dung {len(idxs)}/{n_total} mau, seqlen PMCE={SEQLEN} (nhan ban 1 anh)')

    s0 = dataset[0]
    if s0[2].get('raw_joint_coord_img') is None:
        raise SystemExit("meta khong co 'raw_joint_coord_img' (kiem tra dataset + VIS_META_KEYS).")
    if args.gt2d and 'orig_joint_img' not in s0[1]:
        raise SystemExit(f"--gt2d nhung targets khong co 'orig_joint_img' (keys: {list(s0[1].keys())}).")

    variants = [det_label] + (['gt2d'] if args.gt2d else [])
    models = ['pmce'] + (['motionbert'] if mb is not None else [])
    res = {(m, v): {'err': [], 'pa': [], 'mask': np.zeros(len(JOINTS))} for m in models for v in variants}
    ratio_log = []
    n_seen = 0

    def run_pmce(raw_xy, mask, img_h, img_w, img_feat):
        """raw_xy (B, 17, 2) pixel anh goc, mask (B, 17, 1) numpy -> pred root-relative (B, 17, 3) met."""
        B = raw_xy.shape[0]
        xy_norm = np.empty_like(raw_xy)
        for b in range(B):
            xy_norm[b] = normalize_screen_coordinates_np(raw_xy[b], img_h[b], img_w[b])
        if args.pmce_2d_rootrel:
            xy_norm = xy_norm - xy_norm[:, 0:1, :]      # ban cu (sai so voi PMCE goc)
        xy_norm = xy_norm * (mask[..., 0:1] > 0)         # khop khong hop le -> 0
        x = torch.from_numpy(xy_norm).to(device).float().unsqueeze(1).repeat(1, SEQLEN, 1, 1)   # (B, T, 17, 2)
        pred = pmce(x, img_feat)
        return pred - pred[:, 0:1, :]

    with torch.no_grad():
        for bi, (inputs, targets, meta) in enumerate(tqdm(loader, desc='PMCE vs MotionBERT')):
            B = inputs['img'].shape[0]

            # ---- 3D GT ----
            if 'orig_joint_cam' in targets:
                gt3d = targets['orig_joint_cam'].to(device).float()
            else:
                gt3d = torch.matmul(reg[None], targets['smpl_mesh_cam'].to(device).float())
            gt3d = gt3d - gt3d[:, 0:1, :]   # root-relative, met

            # ---- Dac trung anh cho PMCE ----
            img = inputs['img'].to(device).float()
            _, feat_token = backbone(img)
            img_feat = feat_token.view(B, -1).unsqueeze(1).repeat(1, SEQLEN, 1)   # (B, T, 2048)
            img_h = meta['img_shape'][0].numpy().astype(np.float32)
            img_w = meta['img_shape'][1].numpy().astype(np.float32)

            # ---- 2D cho tung bien the: (xy MotionBERT [-1,1] crop, conf tensor), (raw px PMCE, mask numpy) ----
            conf_det = inputs['joints_mask'].float()
            v2d = {det_label: (inputs['joints'][..., :2].float(), conf_det,
                               meta['raw_joint_coord_img'][..., :2].numpy().astype(np.float32),
                               conf_det.numpy())}
            if args.gt2d:
                gt_hm = targets['orig_joint_img'][..., :2].float()
                gt_mask = (meta['orig_joint_valid'] * meta['orig_joint_trunc']).float()
                gt_xy = normalize_hm_xy(gt_hm) * (gt_mask > 0)
                gt_raw = hm_to_raw_px(gt_hm.numpy(), meta['img2bb_trans'].numpy().astype(np.float64))
                v2d['gt2d'] = (gt_xy, gt_mask, gt_raw, gt_mask.numpy())

            for v, (xy_mb, conf_mb, raw_xy, mask_np) in v2d.items():
                # PMCE
                pred = run_pmce(raw_xy, mask_np, img_h, img_w, img_feat)
                gt_bone = torch.norm(gt3d[:, 1] - gt3d[:, 0], dim=-1)
                ratio = (torch.norm(pred[:, 1] - pred[:, 0], dim=-1) / gt_bone.clamp_min(1e-6)).mean().item()
                ratio_log.append(ratio)
                if ratio > 10:
                    pred = pred / 1000.0   # pred dang la mm -> met
                r = res[('pmce', v)]
                r['err'].append((torch.norm(pred - gt3d, dim=-1) * 1000).cpu().numpy())
                r['pa'].append((batch_pa_errors(pred, gt3d) * 1000).cpu().numpy())
                r['mask'] += (mask_np[..., 0] > 0).sum(0)
                # MotionBERT
                if mb is not None:
                    pm = mb(make_mb_input(xy_mb.to(device), conf_mb.to(device), rootrel, drop_idx))[:, 0]
                    pm = pm - pm[:, 0:1, :]
                    r = res[('motionbert', v)]
                    r['err'].append((torch.norm(pm - gt3d, dim=-1) * 1000).cpu().numpy())
                    r['pa'].append((batch_pa_errors(pm, gt3d) * 1000).cpu().numpy())
                    r['mask'] += (conf_mb[..., 0] > 0).float().sum(0).numpy()
            n_seen += B

    for r in res.values():
        r['err'] = np.concatenate(r['err'])     # (N, 17) mm
        r['pa'] = np.concatenate(r['pa'])       # (N,) mm
    avg_ratio = float(np.mean(ratio_log))

    # ---------------- In ket qua ----------------
    print('\n' + '=' * 78)
    print(f'Lift 2D -> 3D | {args.split} | n={n_seen} | 17 khop H36M, root-relative (mm) | cung mau, cung 2D')
    print(f'(PMCE: ti le xuong dui pred/gt TB = {avg_ratio:.2f} -> '
          f'{"da chia pred cho 1000 (mm->m)" if avg_ratio > 10 else "giu nguyen (pred da la met)"})')
    print('=' * 78)
    print(f'{"model":<12}{"dau vao 2D":<14}{"MPJPE":>10}{"PA-MPJPE":>11}{"khop co mat":>14}')
    for m in models:
        for v in variants:
            r = res[(m, v)]
            print(f'{m:<12}{v:<14}{r["err"].mean(1).mean():>10.2f}{r["pa"].mean():>11.2f}'
                  f'{r["mask"].mean() / n_seen * 100:>13.1f}%')

    if mb is not None:
        print()
        for v in variants:
            d = res[('motionbert', v)]['err'].mean(1) - res[('pmce', v)]['err'].mean(1)
            se = d.std(ddof=1) / np.sqrt(len(d))
            print(f'[{v}] motionbert - pmce (MPJPE): {d.mean():+.2f} +/- {1.96 * se:.2f} mm | '
                  f'motionbert tot hon o {(d < 0).mean() * 100:.1f}% mau')
        print('  (khoang tin cay coi cac mau doc lap -> lac quan voi frame lien tiep cua 3DPW / H36M)')

    print('\nLUU Y mien du lieu:')
    print('  * PMCE: checkpoint chi train Human3.6M -> 3DPW la CHEO-DOMAIN, H36M la "san nha".')
    if mb is not None:
        print('  * MotionBERT: xem train_list cua checkpoint o tren (neu co 3dpw-train -> 3DPW test cung mien, '
              'khong cong bang voi PMCE).')
    if is_h36m and src != 'halpe26':
        print('  * H36M voi gt2d-input: 2D oracle, khong so truc tiep voi so cua --split 3dpw (2D detector).')

    hdr = f'\n{"khop":<12}' + ''.join(f'{(m[:4] + "/" + v)[:13]:>14}' for m in models for v in variants)
    print(hdr)
    for j, jn in enumerate(JOINTS):
        print(f'{jn:<12}' + ''.join(f'{res[(m, v)]["err"][:, j].mean():>14.1f}' for m in models for v in variants))

    out = {'pmce_ckpt': args.pmce_ckpt, 'mb_ckpt': args.mb_ckpt, 'split': args.split, 'n': int(n_seen),
           'cfg': args.cfg, '2d_source': src, 'seqlen': SEQLEN, 'pmce_bonelen_ratio': avg_ratio,
           'results': {f'{m}/{v}': {'mpjpe_mm': float(res[(m, v)]['err'].mean()),
                                    'pa_mpjpe_mm': float(res[(m, v)]['pa'].mean()),
                                    'per_joint_mm': res[(m, v)]['err'].mean(0).tolist()}
                       for m in models for v in variants}}
    os.makedirs(cfg.res_dir, exist_ok=True)
    path = os.path.join(cfg.res_dir, f'compare_pmce_lift_{args.split}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nDa luu: {path}')


if __name__ == '__main__':
    main()
