"""
main/compare_2d_sources.py -- So sanh nguon 2D tren CUNG mau 3DPW, voi GT 2D:
  - halpe    : Halpe-26 (RTMPose, top-down) -- FuseKin dang dung
  - openpose : OpenPose co san trong annotation (JOTR / 3DCrowdNet dung cho 3DPW / OC / PC)
  - hhrnet   : Higher-HRNet bottom-up (JOTR / 3DCrowdNet dung cho 3DPW-Crowd; chi co o tap crowd)
GT 2D = H36M-17 hoi quy tu mesh SMPL GT, chieu len ANH GOC. So tren 12 khop chi (vai, khuyu, co tay, hong, goi, co chan)
co cung ten o ca 3 dinh dang. Loi chuan hoa theo kich thuoc nguoi (canh lon nhat bbox GT 2D, px).
Moi nguon: loi TB / trung vi, % khop xa (> --far x kich thuoc nguoi) trong khop phat hien, % khop bo (conf <= nguong rieng
cua nguon: Halpe DATASET.halpe26_conf_thr, OpenPose / HHRNet PW3D.conf_thr), % mau co >= 1 khop xa;
va trong cac mau Halpe XAU: % mau nguon khac TOT (khong khop xa, phat hien >= 8 / 12 khop) -> "cuu duoc".
Ve --vis_k mau Halpe te nhat: cac o Halpe | OpenPose | HHRNet (do = GT, xanh = nguon, vang = khop lech xa).
Khong can GPU / model. OC / PC luon TOAN BO tap (giao thuc JOTR).
Chay:
  python main/compare_2d_sources.py --cfg ./config/train_student_v3_scratch_jotr.yml --sets 3dpw-pc,3dpw-crowd
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import json

parser = argparse.ArgumentParser(description='So sanh 2D Halpe / OpenPose / Higher-HRNet tren 3DPW')
parser.add_argument('--cfg', type=str, required=True, help='yml bat ky co duong dan du lieu (vd yml Student v3)')
parser.add_argument('--sets', type=str, default='3dpw-pc,3dpw-crowd')
parser.add_argument('--far', type=float, default=0.2, help='nguong khop xa, ti le kich thuoc nguoi (~40 px crop 256)')
parser.add_argument('--vis_k', type=int, default=30)
parser.add_argument('--vis_by', type=str, default='halpe', choices=['halpe', 'union'],
                    help="halpe: ve mau Halpe CU xau (cu); union: ve TAT CA mau Halpe cu HOAC Halpe moi xau, chia thu muc "
                         "sua_duoc / lam_hong / ca_hai_xau (can --halpe_new_suffix)")
parser.add_argument('--mask_dir', type=str, default='',
                    help='thu muc sam_masks_<split>.npz -> to mask SAM len o halpe_new (mac dinh <jotr_data_root>/PW3D/sam_masks)')
parser.add_argument('--max_samples', type=int, default=0, help='0 = toan bo tap')
parser.add_argument('--dump_bad', type=str, default='',
                    help='ghi ann_id cac mau Halpe CU xau (theo GT) ra <dump_bad>_<split>.txt, xau nhat truoc '
                         '(dung cho tools/sam_masks_3dpw.py / vis_sam_rtmpose_3dpw.py --ann_ids_file)')
parser.add_argument('--halpe_base_suffix', type=str, default='',
                    help="nguon 'halpe' (base) doc 3DPW_<tag>_halpe26<suffix>.json ('' = Halpe cu; vd _sam)")
parser.add_argument('--halpe_new_suffix', type=str, default='',
                    help="them nguon 'halpe_new' doc 3DPW_<tag>_halpe26<suffix>.json (vd _opbox) -> dem sua / lam hong theo GT")
args = parser.parse_args()

import numpy as np
from tqdm import tqdm
from core.config import cfg, update_config
update_config(args.cfg)
cfg.DATASET.pw3d_2d_source = 'halpe26'
cfg.DATASET.pw3d_halpe_suffix = args.halpe_base_suffix     # nguon 'halpe' (base)
cfg.DATASET.pw3d_ocpc_test_only = False

import __init_path  # noqa: F401
from utils.jotr_dataset import get_test_dataset
from utils.transforms import cam2pixel
from utils.h36m_adapter import halpe26_to_human36m

LIMB = ('L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
        'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle')
PAIRS = ((0, 2), (2, 4), (1, 3), (3, 5), (6, 8), (8, 10), (7, 9), (9, 11), (0, 1), (6, 7), (0, 6), (1, 7))
SOURCES = ('halpe', 'halpe_new', 'openpose', 'hhrnet')


def label(s):
    """Ten hien thi: halpe -> halpe<base_suffix> (cu = halpe_cu), halpe_new -> halpe<new_suffix>."""
    if s == 'halpe':
        return 'halpe' + args.halpe_base_suffix if args.halpe_base_suffix else 'halpe_cu'
    if s == 'halpe_new':
        return 'halpe' + args.halpe_new_suffix if args.halpe_new_suffix else 'halpe_new'
    return s


def collect(name):
    ds = get_test_dataset(name, None).dataset          # PW3D
    h_idx = [ds.h36m_joints_name.index(n) for n in LIMB]
    o_idx = [ds.openpose_joints_name.index(n) for n in LIMB]
    c_idx = [ds.coco_joints_name.index(n) for n in LIMB]
    thr = {'halpe': float(cfg.DATASET.halpe26_conf_thr), 'halpe_new': float(cfg.DATASET.halpe26_conf_thr),
           'openpose': float(ds.conf_thr), 'hhrnet': float(ds.conf_thr)}
    new = None
    if args.halpe_new_suffix:
        tag = {'3dpw-oc': 'oc', '3dpw-pc': 'pc'}.get(name, ds.data_split)
        p = os.path.join(cfg.DATASET.jotr_data_root, 'PW3D', f'3DPW_{tag}_halpe26{args.halpe_new_suffix}.json')
        if os.path.isfile(p):
            with open(p) as f:
                new = json.load(f)
            print(f'halpe_new: {p} ({len(new)} nguoi)')
        else:
            print(f'[CANH BAO] khong thay {p} -> bo qua halpe_new')
    hh = None
    if name == '3dpw-crowd':
        p = os.path.join(cfg.DATASET.jotr_data_root, 'PW3D', '3DPW_validation_crowd_hhrnet_result.json')
        if os.path.isfile(p):
            with open(p) as f:
                hh = json.load(f)
        else:
            print(f'[CANH BAO] khong thay {p} -> bo qua hhrnet')
    items = ds.datalist[:args.max_samples] if args.max_samples else ds.datalist
    rec = []
    for it in tqdm(items, desc=name):
        mesh, _ = ds.get_smpl_coord(it['smpl_param'])
        gt = cam2pixel(np.dot(ds.h36m_joint_regressor, mesh), it['cam_param']['focal'], it['cam_param']['princpt'])[:, :2]
        size = float(max(np.ptp(gt[:, 0]), np.ptp(gt[:, 1]), 1.0))
        gt12 = gt[h_idx]
        src = {'halpe': np.asarray(it['halpe_h36m'])[h_idx] if it['halpe_h36m'] is not None else None,
               'openpose': np.asarray(it['openpose'])[o_idx]}
        if new is not None and str(it['ann_id']) in new:
            src['halpe_new'] = halpe26_to_human36m(new[str(it['ann_id'])])[h_idx]
        if hh is not None:
            r = hh.get(str(it['ann_id']))
            src['hhrnet'] = np.asarray(r['coco_joints'], dtype=np.float32)[c_idx] if r is not None else None
        out = {'img_path': it['img_path'], 'gt12': gt12, 'size': size, 'ann_id': int(it['ann_id']), 'gt17': gt,
               'gt_bbox': np.asarray(it['tight_bbox'], dtype=np.float32)}   # ann['bbox'] xywh (bbox trich Halpe cu)
        for s, k in src.items():
            if k is None:
                continue
            det = k[:, 2] > thr[s]
            e = np.linalg.norm(k[:, :2] - gt12, axis=1) / size
            out[s] = {'xy': k[:, :2], 'det': det, 'e': e, 'far': det & (e > args.far)}
        rec.append(out)
    return rec


def summarize(rec):
    res = {}
    for s in SOURCES:
        rs = [r for r in rec if s in r]
        if not rs:
            continue
        det = np.stack([r[s]['det'] for r in rs]); e = np.stack([r[s]['e'] for r in rs])
        far = np.stack([r[s]['far'] for r in rs])
        ed = e[det]
        res[s] = {'n': len(rs), 'err_mean_pct': 100 * float(ed.mean()) if ed.size else float('nan'),
                  'err_median_pct': 100 * float(np.median(ed)) if ed.size else float('nan'),
                  'far_pct': 100 * float(far.sum() / max(det.sum(), 1)), 'miss_pct': 100 * float(1 - det.mean()),
                  'samples_far_pct': 100 * float(far.any(1).mean()),
                  'per_joint_far_pct': {LIMB[j]: 100 * float(far[:, j].sum() / max(det[:, j].sum(), 1)) for j in range(len(LIMB))}}
    both = [r for r in rec if 'halpe' in r and 'halpe_new' in r]
    if both:
        ob = np.array([r['halpe']['far'].any() for r in both]); nb = np.array([r['halpe_new']['far'].any() for r in both])
        res['new_vs_old'] = {'n': len(both), 'old_bad': int(ob.sum()), 'new_bad': int(nb.sum()),
                             'fixed': int((ob & ~nb).sum()), 'broke': int((~ob & nb).sum())}
    bad = [r for r in rec if 'halpe' in r and r['halpe']['far'].any()]
    for s in ('halpe_new', 'openpose', 'hhrnet'):
        both = [r for r in bad if s in r]
        if both:
            ok = [(not r[s]['far'].any()) and r[s]['det'].sum() >= 8 for r in both]
            res.setdefault('rescue', {})[s] = {'halpe_bad': len(both), 'pct_ok': 100 * float(np.mean(ok))}
    return res


def show(name, res):
    print(f'\n===== [{name}] khop xa = loi > {args.far:.2f} x kich thuoc nguoi | 12 khop chi =====')
    print(f'   {"nguon":<13} | {"n":>6} | {"loi TB %":>8} | {"trung vi %":>10} | {"xa %":>6} | {"bo %":>6} | {"mau co khop xa %":>16}')
    for s in SOURCES:
        if s in res:
            m = res[s]
            print(f'   {label(s):<13} | {m["n"]:>6} | {m["err_mean_pct"]:>8.1f} | {m["err_median_pct"]:>10.1f} | {m["far_pct"]:>6.2f} | '
                  f'{m["miss_pct"]:>6.1f} | {m["samples_far_pct"]:>16.1f}')
            top = sorted(m['per_joint_far_pct'].items(), key=lambda x: -x[1])[:4]
            print('             khop xa nhieu: ' + ', '.join(f'{k} {v:.1f}%' for k, v in top))
    if 'new_vs_old' in res:
        m = res['new_vs_old']
        print(f'   THEO GT: mau co khop xa  {label("halpe")} {m["old_bad"]} -> {label("halpe_new")} {m["new_bad"]} (n = {m["n"]}) | '
              f'sua duoc {m["fixed"]} | LAM HONG {m["broke"]}')
    for s, m in res.get('rescue', {}).items():
        print(f'   Mau {label("halpe")} XAU ({m["halpe_bad"]}): {label(s)} TOT (khong khop xa, >= 8/12 khop) o {m["pct_ok"]:.1f}%')


def vis(name, rec):
    import cv2
    def is_bad(r, s):
        return s in r and bool(r[s]['far'].any())

    def nfar(r, s):
        return int(r[s]['far'].sum()) if s in r else 0
    out_dir = os.path.join(cfg.output_dir, 'vis', name)
    # mask SAM cua dung nguoi (de to len o halpe_new); doc 1 lan vao RAM
    msk = None
    split = {'3dpw': 'test', '3dpw-oc': 'oc', '3dpw-pc': 'pc', '3dpw-crowd': 'validation'}.get(name, name)
    mpath = os.path.join(args.mask_dir or os.path.join(cfg.DATASET.jotr_data_root, 'PW3D', 'sam_masks'),
                         f'sam_masks_{split}.npz')
    # to mask len o nao dang la ban SAM (suffix bat dau '_sam')
    mask_src = 'halpe' if args.halpe_base_suffix.startswith('_sam') else (
        'halpe_new' if args.halpe_new_suffix.startswith('_sam') else None)
    if mask_src and os.path.isfile(mpath):
        md = dict(np.load(mpath))
        msk = (md, {int(a): k for k, a in enumerate(md['ann_ids'])})
    elif mask_src:
        print(f'[CANH BAO] khong thay {mpath} -> khong to mask')

    def get_mask(aid):
        if msk is None or aid not in msk[1]:
            return None
        md, k = msk[0], msk[1][aid]
        rr = md['runs'][md['off'][k]:md['off'][k + 1]]
        return np.repeat(np.arange(len(rr)) % 2, rr).astype(bool).reshape(int(md['H'][k]), int(md['W'][k]))
    if args.vis_by == 'union':
        # mau xau o Halpe cu HOAC Halpe moi; nhom theo ket qua cua Halpe moi so voi Halpe cu
        bad = [r for r in rec if is_bad(r, 'halpe') or is_bad(r, 'halpe_new')]
        bad.sort(key=lambda r: -max(nfar(r, 'halpe'), nfar(r, 'halpe_new')))

        def group(r):
            if is_bad(r, 'halpe') and is_bad(r, 'halpe_new'):
                return 'ca_hai_xau'
            return 'sua_duoc' if is_bad(r, 'halpe') else 'lam_hong'
        for g in ('sua_duoc', 'lam_hong', 'ca_hai_xau'):
            os.makedirs(os.path.join(out_dir, g), exist_ok=True)
    else:
        bad = [r for r in rec if is_bad(r, 'halpe')]
        bad.sort(key=lambda r: (-int(r['halpe']['far'].sum()), -float(r['halpe']['e'][r['halpe']['det']].max())))
        os.makedirs(out_dir, exist_ok=True)

        def group(r):
            return ''
    for rank, r in enumerate(bad[:args.vis_k]):
        im = cv2.imread(r['img_path'])
        if im is None:
            continue
        g = r['gt12']
        bx, by, bw, bh = r['gt_bbox']
        c, half = g.mean(0), max(0.9 * r['size'], 0.75 * max(bw, bh))
        x0, y0 = int(max(c[0] - half, 0)), int(max(c[1] - half, 0))
        x1, y1 = int(min(c[0] + half, im.shape[1])), int(min(c[1] + half, im.shape[0]))
        panels = []
        for s in SOURCES:
            if s not in r:
                continue
            p = im.copy()
            if s == mask_src:
                m = get_mask(r['ann_id'])
                if m is not None and m.shape == p.shape[:2]:
                    p[m] = (0.5 * p[m] + 0.5 * np.array([255, 128, 0])).astype(np.uint8)   # mask SAM (xanh duong)
            bx, by, bw, bh = r['gt_bbox']
            cv2.rectangle(p, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (255, 0, 255), 2)   # bbox annotation
            # vung RTMPose thuc su nhin (bbox annotation noi ~1.25 quanh tam, giong tien xu ly top-down)
            cx, cy, ex = bx + bw / 2, by + bh / 2, 1.25
            cv2.rectangle(p, (int(cx - bw * ex / 2), int(cy - bh * ex / 2)), (int(cx + bw * ex / 2), int(cy + bh * ex / 2)),
                          (200, 0, 200), 1)
            for a, b in PAIRS:
                cv2.line(p, tuple(g[a].astype(int)), tuple(g[b].astype(int)), (0, 0, 255), 2)
            d, xy = r[s]['det'], r[s]['xy']
            for a, b in PAIRS:
                if d[a] and d[b]:
                    cv2.line(p, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), (0, 200, 0), 2)
            for j in range(len(LIMB)):
                if d[j]:
                    cv2.circle(p, tuple(xy[j].astype(int)), 4, (0, 200, 0), -1)
                    if r[s]['far'][j]:
                        cv2.line(p, tuple(xy[j].astype(int)), tuple(g[j].astype(int)), (0, 255, 255), 2)
            p = p[y0:y1, x0:x1]
            if p.size == 0:
                continue
            f = 384.0 / p.shape[0]
            p = cv2.resize(p, None, fx=f, fy=f)
            txt = f'{label(s)}: xa {int(r[s]["far"].sum())} | bo {int((~d).sum())}'
            cv2.putText(p, txt, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(p, txt, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            panels.append(p)
        if panels:
            h = min(q.shape[0] for q in panels)
            cv2.imwrite(os.path.join(out_dir, group(r), f'{rank:04d}_{os.path.basename(os.path.dirname(r["img_path"]))}_'
                                              f'{os.path.basename(r["img_path"])}'),
                        cv2.hconcat([q[:h] for q in panels]))
    if args.vis_by == 'union':
        cnt = {g: sum(group(r) == g for r in bad[:args.vis_k]) for g in ('sua_duoc', 'lam_hong', 'ca_hai_xau')}
        print(f'   union: {cnt}')
    print(f'   hinh {min(args.vis_k, len(bad))} mau ({args.vis_by}): {out_dir}  (do = GT, xanh = nguon 2D, vang = lech xa, tim = bbox annotation, tim manh = vung RTMPose nhin ~x1.25)')


def main():
    summary = {}
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        rec = collect(name)
        res = summarize(rec)
        show(name, res)
        if args.vis_k > 0:
            vis(name, rec)
        summary[name] = res
        if args.dump_bad:
            bad = [r for r in rec if 'halpe' in r and r['halpe']['far'].any()]
            bad.sort(key=lambda r: (-int(r['halpe']['far'].sum()), -float(r['halpe']['e'][r['halpe']['det']].max())))
            split = {'3dpw': 'test', '3dpw-oc': 'oc', '3dpw-pc': 'pc', '3dpw-crowd': 'validation'}.get(name, name)
            path = f'{args.dump_bad}_{split}.txt'
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            with open(path, 'w') as f:
                f.write('\n'.join(str(r['ann_id']) for r in bad) + '\n')
            gpath = f'{args.dump_bad}_{split}_gt2d.json'
            with open(gpath, 'w') as f:
                json.dump({str(r['ann_id']): np.round(r['gt17'], 1).tolist() for r in bad}, f)
            print(f'   {len(bad)} ann_id Halpe xau -> {path} (dung --split {split} --ann_ids_file {path})')
            print(f'   GT 2D H36M-17 (pixel anh goc) cua cac mau do -> {gpath} (--gt2d_json, CHI DE THU)')
    os.makedirs(cfg.output_dir, exist_ok=True)
    out = os.path.join(cfg.output_dir, 'compare_2d_sources.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'far': args.far, 'results': summary}, f, indent=2, ensure_ascii=False)
    print(f'\nDa luu: {out}')


if __name__ == '__main__':
    main()
