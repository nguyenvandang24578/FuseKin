"""
main/diag_2d_crowd.py -- Chan doan: sai so PC / Crowd co phai do 2D dau vao (Halpe) khong?

D1 (chat luong 2D Halpe, so voi GT 2D = chieu H36M-17 tu mesh GT vao cung khung crop 256 x 256):
   - loi trung binh / trung vi (px) tren khop Halpe phat hien (mask > 0) VA GT nam trong khung crop
   - % khop xa (> --far_px), % khop xa MA tu tin cao (conf >= --conf_hi) -> nghi "bat nham nguoi"
   - % khop GT nam trong khung nhung Halpe bo (mask 0); % mau co >= 1 khop xa tu tin
D2 (oracle 2D): chay Student voi 3 loai 2D; MOI nhanh (DiffLifter, heatmap [A], feature tai khop [B]/[C], PE,
   tinh chinh [D]) deu nhan cung 2D do:
   - halpe  : nhu test that (phai trung so test)
   - dropfar: chi BO (mask 0) cac khop xa, khong biet toa do dung -> loi ich neu chi PHAT HIEN duoc khop sai
   - fixfar : chi thay cac khop xa (> far_px) bang GT (conf 1), giu nguyen khop con lai
              -> tach phan loi do khop sai tho (nham nguoi) voi phan do nhieu nho
   - gt     : toan bo GT 2D (conf 1; khop GT ngoai khung -> mask 0). LUU Y (ket qua 2026-10-28): GT 2D chieu tu
              mesh (dinh nghia khop H36M) LECH dinh nghia khop Halpe ma DiffLifter da hoc -> lifter / mesh TE HON
              ca Halpe tren 3DPW / OC -> KHONG phai tran tren hop le; dropfar / fixfar moi la oracle dung.
   Moi loai: lifter MPJPE17 (so voi GT H36M-17 tu mesh) + mesh dataset.evaluate (MPJPE14 / PA / MPVPE, mm),
   va mesh MPJPE14 tach theo mau CO / KHONG co khop xa tu tin (xac dinh tren Halpe).
Gioi han: crop va cam_bbox van tinh tu 2D Halpe (nhu test that) -> neu Halpe bat nham nguoi, crop co the lech.

Chi danh gia, khong train. OC / PC mac dinh TOAN BO (giao thuc JOTR); --ocpc_mode test = chi seq test.
In them so mau tung tap (n) de doi chieu dinh nghia tap voi JOTR / DPMesh.
Chay:
  python main/diag_2d_crowd.py --cfg ./config/train_student_v3_scratch_jotr.yml \
      --ckpt ./experiment/student_v3_scratch_jotr/checkpoint/best.pth.tar --gpu 0
  (chay thu nhanh: them --max_batches 20)
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import json

parser = argparse.ArgumentParser(description='Chan doan 2D Halpe tren 3DPW / OC / PC / Crowd')
parser.add_argument('--cfg', type=str, required=True, help='yml DA DUNG de train checkpoint (dung kien truc v3)')
parser.add_argument('--ckpt', type=str, required=True, help='checkpoint Student (best.pth.tar | last.pth.tar)')
parser.add_argument('--sets', type=str, default='3dpw,3dpw-oc,3dpw-pc,3dpw-crowd')
parser.add_argument('--ocpc_mode', type=str, default='all', choices=['all', 'test'])
parser.add_argument('--far_px', type=float, default=40.0, help='nguong khop "xa" (px trong crop 256)')
parser.add_argument('--conf_hi', type=float, default=0.5, help='nguong conf "tu tin" cua Halpe')
parser.add_argument('--batch_size', type=int, default=0, help='0 = cfg.TEST.batch_size')
parser.add_argument('--max_batches', type=int, default=0, help='0 = toan bo tap')
parser.add_argument('--variants', type=str, default='halpe,dropfar,fixfar,gt',
                    help="bien the 2D cho D2 (luon co 'halpe'); vd chi 'halpe' de chay nhanh D1 + D3 + hinh")
parser.add_argument('--miss_thr', type=int, default=8, help='D3: mau "mat nhieu 2D" = so khop Halpe mask 0 >= nguong')
parser.add_argument('--iou_thr', type=float, default=0.5, help='D3: mau "crop lech" = IoU(crop that, crop ly tuong) < nguong')
parser.add_argument('--vis_k', type=int, default=20, help='so mau te nhat moi tap luu hinh toan anh (0 = khong ve)')
parser.add_argument('--lifter_ckpt', type=str, default='',
                    help='thay DiffLifter trong Student bang checkpoint finetune_difflifter (.bin) -> do lift17 cua lifter '
                         'moi (mesh van qua Student cu, chi tham khao)')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import cfg, update_config
update_config(args.cfg)
cfg.DATASET.pw3d_test_gt2d = True          # dataset tra them GT 2D (targets['orig_joint_img'], meta orig_joint_trunc)

import __init_path  # noqa: F401
from utils.ckpt_tools import build_and_load
from utils.jotr_dataset import get_test_dataset

J = 17
H36M_NAMES = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose',
              'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
VARIANTS = tuple(['halpe'] + [v.strip() for v in args.variants.split(',') if v.strip() and v.strip() != 'halpe'])
assert set(VARIANTS) <= {'halpe', 'dropfar', 'fixfar', 'gt'}, VARIANTS
H36M_PAIRS = ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
              (12, 13), (8, 14), (14, 15), (15, 16))
PX = cfg.input_img_shape[1] / 2.0          # [-1, 1] -> pixel trong crop


@torch.no_grad()
def run_set(model, ds, loader, dev):
    reg = torch.as_tensor(ds.h36m_joint_regressor, dtype=torch.float32, device=dev)
    hm = torch.tensor([cfg.output_hm_shape[2], cfg.output_hm_shape[1]], dtype=torch.float32, device=dev)
    acc = {k: np.zeros(J) for k in ('err', 'det', 'far', 'hi', 'far_hi', 'gin', 'miss')}
    err_list, sample_bad = [], []
    mesh = {v: {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': []} for v in VARIANTS + ('halpe_init',)}
    lift = {v: [] for v in VARIANTS}
    d3 = {k: [] for k in ('iou', 'off', 'scale', 'nmiss')}
    keep = {'path': [], 'bb2img': [], 'j2': [], 'g2': []}      # de ve hinh mau te nhat
    seen = 0
    for bi, (inputs, targets, meta) in enumerate(tqdm(loader, leave=False)):
        if args.max_batches and bi >= args.max_batches:
            break
        img = inputs['img'].to(dev).float()
        B = img.shape[0]
        j2 = inputs['joints'].to(dev).float()                                     # (B,17,3) [-1,1] + cot 3
        jm = inputs['joints_mask'].to(dev).float().reshape(B, J, 1)               # conf (0 = bo)
        cam = inputs['cam_bbox'].to(dev).float()
        gt_mesh = targets['smpl_mesh_cam'].to(dev).float()
        g2 = targets['orig_joint_img'].to(dev).float()[..., :2] / hm * 2 - 1     # GT 2D (B,17,2) [-1,1]
        gin = meta['orig_joint_trunc'].to(dev).float().reshape(B, J, 1)          # 1 = khop GT trong khung crop

        # ---------- D1: chat luong 2D Halpe ----------
        det = (jm > 0).float()
        valid = det * gin
        e = torch.norm(j2[..., :2] - g2, dim=-1, keepdim=True) * PX              # px
        far = (e > args.far_px).float() * valid
        hi = (jm >= args.conf_hi).float() * valid
        for k, t in (('err', e * valid), ('det', valid), ('far', far), ('hi', hi), ('far_hi', far * hi),
                     ('gin', gin), ('miss', (1 - det) * gin)):
            acc[k] += t.sum(0).reshape(-1).cpu().numpy()
        err_list.append(e[valid > 0].cpu().numpy())
        sample_bad.append(((far * hi).sum((1, 2)) > 0).cpu().numpy())

        # ---------- D3: crop co om dung nguoi khong ----------
        # Crop "ly tuong" = cung quy tac test (bbox om khop x1.2, vuong, x1.1) nhung tinh tu GT 2D (ca khop bi che).
        # Toa do trong khung crop that ([-1, 1]^2) -> IoU, lech tam (don vi nua crop), ti le kich thuoc.
        gx, gy = g2[..., 0], g2[..., 1]
        cx, cy = (gx.amax(1) + gx.amin(1)) / 2, (gy.amax(1) + gy.amin(1)) / 2
        side = torch.maximum(gx.amax(1) - gx.amin(1), gy.amax(1) - gy.amin(1)) * 1.2 * 1.1
        ix = (torch.clamp(cx + side / 2, max=1.0) - torch.clamp(cx - side / 2, min=-1.0)).clamp_min(0)
        iy = (torch.clamp(cy + side / 2, max=1.0) - torch.clamp(cy - side / 2, min=-1.0)).clamp_min(0)
        inter = ix * iy
        d3['iou'].append((inter / (side ** 2 + 4.0 - inter)).cpu().numpy())
        d3['off'].append(torch.sqrt(cx ** 2 + cy ** 2).cpu().numpy())
        d3['scale'].append((side / 2).cpu().numpy())
        d3['nmiss'].append((jm <= 0).float().sum((1, 2)).cpu().numpy())
        if args.vis_k > 0:
            keep['path'].extend(list(meta['img_path']))
            keep['bb2img'].append(meta['bb2img_trans'].float().numpy())
            keep['j2'].append(torch.cat([j2[..., :2], jm], -1).cpu().numpy())
            keep['g2'].append(g2.cpu().numpy())

        # ---------- D2: Student voi 3 loai 2D ----------
        gt17 = torch.einsum('jk,bkc->bjc', reg, gt_mesh)
        gt17 = gt17 - gt17[:, 0:1]
        gt_in = torch.cat([g2 * gin, gin], -1)
        fix_conf = torch.where(far > 0, torch.ones_like(jm), jm)
        fix_in = torch.cat([j2[..., :2] * (1 - far) + g2 * far, fix_conf], -1)
        tgt_np = gt_mesh.cpu().numpy()
        drop_conf = jm * (1 - far)                                                # khop xa -> mask 0 (coi nhu mat)
        drop_in = torch.cat([j2[..., :2] * (drop_conf > 0), drop_conf], -1)
        cand = {'halpe': (j2, jm), 'dropfar': (drop_in, drop_conf), 'fixfar': (fix_in, fix_conf), 'gt': (gt_in, gin)}
        for v in VARIANTS:
            jj, mm = cand[v]
            out = model(img, jj, is_train=False, joints_mask=mm, cam_bbox=cam)
            lift[v].append((torch.norm(out['lifted_joints_3d'] - gt17, dim=-1).mean(-1) * 1000).cpu().numpy())
            todo = [(v, out['smpl_mesh_cam'])]
            if v == 'halpe' and out.get('refine_history'):
                todo.append(('halpe_init', out['refine_history'][0]['smpl_mesh_cam']))   # truoc tinh chinh [D]
            for name_, mesh_t in todo:
                pm = mesh_t.float().cpu().numpy()
                res = ds.evaluate([{'smpl_mesh_cam': pm[k], 'smpl_mesh_cam_target': tgt_np[k]} for k in range(B)], seen)
                for k_ in mesh[name_]:
                    mesh[name_][k_].extend(res[k_])
        seen += B

    bad = np.concatenate(sample_bad)
    errs = np.concatenate(err_list) if err_list else np.zeros(0)
    d = lambda a, b: float(a.sum() / max(b.sum(), 1))
    r = {'n_eval': int(seen),
         'd1': {'err_px_mean': d(acc['err'], acc['det']), 'err_px_median': float(np.median(errs)) if errs.size else 0.0,
                'far_pct': 100 * d(acc['far'], acc['det']), 'far_conf_hi_pct': 100 * d(acc['far_hi'], acc['det']),
                'miss_pct': 100 * d(acc['miss'], acc['gin']), 'samples_with_far_conf_hi_pct': 100 * float(bad.mean()),
                'per_joint_far_pct': {H36M_NAMES[j]: 100 * float(acc['far'][j] / max(acc['det'][j], 1)) for j in range(J)}},
         'd2': {}}
    for v in VARIANTS + ('halpe_init',):
        if not mesh[v]['mpjpe']:
            continue
        mp = np.asarray(mesh[v]['mpjpe'])
        lf = np.concatenate(lift['halpe' if v == 'halpe_init' else v])
        r['d2'][v] = {'lift_mpjpe17': float(lf.mean()),
                      'lift_bad_samples': float(lf[bad].mean()) if bad.any() else float('nan'),
                      'lift_clean_samples': float(lf[~bad].mean()) if (~bad).any() else float('nan'),
                      **{k_: float(np.mean(x)) for k_, x in mesh[v].items()},
                      'mpjpe_bad_samples': float(mp[bad].mean()) if bad.any() else float('nan'),
                      'mpjpe_clean_samples': float(mp[~bad].mean()) if (~bad).any() else float('nan')}
    # ---------- D3 ----------
    iou, off = np.concatenate(d3['iou']), np.concatenate(d3['off'])
    scale, nmiss = np.concatenate(d3['scale']), np.concatenate(d3['nmiss'])
    mp_h = np.asarray(mesh['halpe']['mpjpe'])
    g_miss, g_crop = nmiss >= args.miss_thr, iou < args.iou_thr
    rest = ~(bad | g_miss | g_crop)
    mean_rest = float(mp_h[rest].mean()) if rest.any() else float('nan')
    N = max(len(mp_h), 1)

    def grp(m):
        if not m.any():
            return {'n': 0, 'pct': 0.0, 'mpjpe': float('nan'), 'gop_mm': 0.0}
        # gop_mm: MPJPE trung binh ca tap giam bao nhieu neu nhom nay chi sai bang muc nhom "con lai"
        return {'n': int(m.sum()), 'pct': 100 * float(m.mean()), 'mpjpe': float(mp_h[m].mean()),
                'gop_mm': float((mp_h[m].sum() - m.sum() * mean_rest) / N)}
    r['d3'] = {'iou_mean': float(iou.mean()), 'iou_mean_bad': float(iou[bad].mean()) if bad.any() else float('nan'),
               'iou_mean_clean': float(iou[~bad].mean()) if (~bad).any() else float('nan'),
               'off_mean': float(off.mean()), 'scale_mean': float(scale.mean()),
               'groups': {f'mat >= {args.miss_thr} khop 2D': grp(g_miss), f'crop lech (IoU < {args.iou_thr})': grp(g_crop),
                          '2D nham nguoi (khop xa tu tin)': grp(bad), 'nham nguoi & crop lech': grp(bad & g_crop),
                          'con lai': grp(rest)}}
    r['_vis'] = (keep, mp_h, iou, nmiss, bad)
    return r


def vis_worst(label, r):
    """Luu hinh TOAN ANH goc cua vis_k mau te nhat (MPJPE14 halpe): khung crop (xanh duong), 2D Halpe conf > 0
    (xanh la), GT 2D (do), noi Halpe -> GT (vang) khi lech > far_px."""
    import cv2
    keep, mp_h, iou, nmiss, bad = r.pop('_vis')
    if args.vis_k <= 0 or not keep['path']:
        return
    bb2img, j2, g2 = (np.concatenate(keep[k]) for k in ('bb2img', 'j2', 'g2'))
    W, H = cfg.input_img_shape[1], cfg.input_img_shape[0]
    out_dir = os.path.join(cfg.output_dir, 'vis_worst', label.replace(' ', '_'))
    os.makedirs(out_dir, exist_ok=True)

    def to_img(xy, T):
        px = (xy + 1) / 2 * np.array([W, H], dtype=np.float32)
        return np.concatenate([px, np.ones_like(px[:, :1])], 1) @ T.T

    for rank, i in enumerate(np.argsort(-mp_h)[:args.vis_k]):
        im = cv2.imread(keep['path'][i])
        if im is None:
            continue
        T = bb2img[i]
        corners = to_img(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32), T).astype(np.int32)
        cv2.polylines(im, [corners], True, (255, 128, 0), 3)
        gp, hp = to_img(g2[i], T), to_img(j2[i][:, :2], T)
        hv = j2[i][:, 2] > 0
        for a, b in H36M_PAIRS:
            cv2.line(im, tuple(gp[a].astype(int)), tuple(gp[b].astype(int)), (0, 0, 255), 2)
            if hv[a] and hv[b]:
                cv2.line(im, tuple(hp[a].astype(int)), tuple(hp[b].astype(int)), (0, 200, 0), 2)
        for j in range(J):
            cv2.circle(im, tuple(gp[j].astype(int)), 4, (0, 0, 255), -1)
            if hv[j]:
                cv2.circle(im, tuple(hp[j].astype(int)), 4, (0, 200, 0), -1)
                if np.linalg.norm((j2[i][j, :2] - g2[i][j]) * PX) > args.far_px:
                    cv2.line(im, tuple(hp[j].astype(int)), tuple(gp[j].astype(int)), (0, 255, 255), 1)
        txt = (f'#{i} MPJPE {mp_h[i]:.0f} | IoU crop {iou[i]:.2f} | mat 2D {int(nmiss[i])}/17 | '
               f'nham nguoi {"CO" if bad[i] else "khong"}')
        cv2.putText(im, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(im, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        f = 1280.0 / max(im.shape[:2])
        if f < 1:
            im = cv2.resize(im, None, fx=f, fy=f)
        cv2.imwrite(os.path.join(out_dir, f'{rank:02d}_idx{i}_mpjpe{mp_h[i]:.0f}.jpg'), im)
    print(f'   hinh {min(args.vis_k, len(mp_h))} mau te nhat: {out_dir}  (xanh duong = crop, la = Halpe, do = GT, vang = lech xa)')


def show(label, n, r):
    a = r['d1']
    print(f'\n===== [{label}] n tap = {n} | n danh gia = {r["n_eval"]} =====')
    print(f'D1 2D Halpe: loi TB {a["err_px_mean"]:.1f} px (trung vi {a["err_px_median"]:.1f}) | khop xa (>{args.far_px:.0f}px) '
          f'{a["far_pct"]:.1f}% | xa & conf>={args.conf_hi} {a["far_conf_hi_pct"]:.1f}% | Halpe bo khop GT trong khung '
          f'{a["miss_pct"]:.1f}% | mau co >=1 khop xa tu tin {a["samples_with_far_conf_hi_pct"]:.1f}%')
    top = sorted(a['per_joint_far_pct'].items(), key=lambda x: -x[1])[:5]
    print('   khop xa nhieu nhat: ' + ', '.join(f'{k} {v:.1f}%' for k, v in top))
    print(f'D2 {"2D":<10} | {"lift17":>7} | {"MPJPE":>7} | {"PA":>6} | {"MPVPE":>7} | {"MPJPE xau/sach":>15} | {"lift17 xau/sach":>15}')
    for v, m in r['d2'].items():
        print(f'   {v:<10} | {m["lift_mpjpe17"]:>7.2f} | {m["mpjpe"]:>7.2f} | {m["pa_mpjpe"]:>6.2f} | {m["mpvpe"]:>7.2f} | '
              f'{m["mpjpe_bad_samples"]:>7.2f}/{m["mpjpe_clean_samples"]:<7.2f} | '
              f'{m["lift_bad_samples"]:>7.2f}/{m["lift_clean_samples"]:<7.2f}')
    print('   (halpe_init = mesh TRUOC tinh chinh [D], cung 2D halpe)')
    c = r['d3']
    print(f'D3 crop: IoU TB {c["iou_mean"]:.2f} (mau nham nguoi {c["iou_mean_bad"]:.2f} / sach {c["iou_mean_clean"]:.2f}) | '
          f'lech tam TB {c["off_mean"]:.2f} nua-crop | kich thuoc nguoi / crop {c["scale_mean"]:.2f}')
    print(f'   {"nhom (halpe)":<34} | {"n":>6} | {"%":>5} | {"MPJPE":>7} | {"gop vao TB (mm)":>15}')
    for k, g in c['groups'].items():
        print(f'   {k:<34} | {g["n"]:>6} | {g["pct"]:>5.1f} | {g["mpjpe"]:>7.2f} | {g["gop_mm"]:>15.2f}')
    print('   (gop vao TB = MPJPE ca tap giam bao nhieu neu nhom nay chi sai bang nhom "con lai"; cac nhom co the giao nhau)')


def main():
    dev = torch.device('cuda')
    model = build_and_load('student', args.ckpt, dev)[0]
    if args.lifter_ckpt:
        from models.ARTS import build_diff_lifter
        lifter = build_diff_lifter(args.lifter_ckpt, cfg.MODEL.get('difflift_steps', 1), cfg.MODEL.get('difflift_k', 1))
        lifter = lifter.to(dev).eval()
        for p_ in lifter.parameters():
            p_.requires_grad = False
        model.diff_lifter = lifter
        print(f'===> Thay DiffLifter bang {args.lifter_ckpt} (lift17 = lifter moi; mesh qua Student cu -> chi tham khao)')
    bs = args.batch_size or cfg.TEST.batch_size
    summary = {}
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        cfg.DATASET.pw3d_ocpc_test_only = (args.ocpc_mode == 'test') and name in ('3dpw-oc', '3dpw-pc')
        label = name + (' (chi seq test)' if cfg.DATASET.pw3d_ocpc_test_only else '')
        ds = get_test_dataset(name, None)
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=cfg.DATASET.workers)
        r = run_set(model, ds, loader, dev)
        r['n_set'] = len(ds)
        show(label, len(ds), r)
        vis_worst(label, r)
        summary[label] = r
    os.makedirs(cfg.output_dir, exist_ok=True)
    out = os.path.join(cfg.output_dir, 'diag_2d_crowd.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'ckpt': args.ckpt, 'cfg': args.cfg, 'far_px': args.far_px, 'conf_hi': args.conf_hi,
                   'results': summary}, f, indent=2, ensure_ascii=False)
    print('\n' + '=' * 100)
    print(f'TOM TAT ({args.ckpt}) | mesh MPJPE14: halpe / dropfar / fixfar / gt | lift17 cung thu tu | 2D xa tu tin %')
    for label, r in summary.items():
        m = r['d2']
        print(f'  {label:<26} n={r["n_set"]:>6} | ' + ' / '.join(f'{m[v]["mpjpe"]:6.2f}' for v in VARIANTS) + ' | '
              + ' / '.join(f'{m[v]["lift_mpjpe17"]:6.2f}' for v in VARIANTS) + f' | {r["d1"]["far_conf_hi_pct"]:5.1f}%'
              + f' (xa moi conf {r["d1"]["far_pct"]:4.1f}%)')
    print(f'Da luu: {out}')


if __name__ == '__main__':
    main()
