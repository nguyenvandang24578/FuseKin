"""Chan doan chat luong 2D Halpe-26 (da doi H36M-17) so voi GT 2D tren tap TRAIN cua lifter.

Chi doc datalist (pixel anh goc, KHONG augmentation), khong doc anh, khong chay model.
Ho tro: Human36M (GT 2D = chieu mocap), MuCo (GT 2D keypoints_img, doi ten MPI-INF-3DHP -> H36M-17).
Moi mau:
  - loi pixel tung khop, chuan hoa theo max(w, h) cua tight bbox GT;
  - so khop "sai" (loi chuan hoa > --bad_thr), co nghi bat nham nguoi (>= --bad_min_joints khop sai);
  - kiem tra hoan doi trai/phai: loi khi doi L/R nho hon nhieu loi goc.
Tuy chon --lifter_csv: samples.csv cua main/eval_difflifter_3dpw.py (cung YAML, --max_samples 0)
  -> ghep theo (dataset, index) de xem loi lifter co tang theo loi 2D khong.
Chi so khop co GT nam trong anh; Pelvis/Torso cua Halpe la khop noi suy (trung diem), van tinh rieng.
--target_samples N (mac dinh 5000, 0 = tat): kiem tra target 3D tren N mau ngau nhien co NeuralAnnot:
  (1) chieu lai GT 3D native (f, c cua dataset) so voi GT 2D -> 3D va anh co nhat quan khong;
  (2) chieu lai H36M-17 tu SMPL NeuralAnnot so voi GT 2D;
  (3) chenh lech root-relative giua target native (lifter dang hoc) va H36M regressor x SMPL NeuralAnnot
      (quy uoc COCO/CrowdPose/Student va GT 3DPW). H36M la moc: mocap va regressor von gan nhau.

Vi du:
  python main/diag_halpe2d.py --cfg config/eval_difflifter_r25_cam_norot.yaml --sets MuCo,Human36M \
      --lifter_csv experiment/_eval/lifter_r25_cam_train_norot_full/samples.csv \
      --out_dir experiment/_eval/halpe2d_train_diag
"""
import os
import sys
sys.path.append('./lib')
sys.path.append('./')

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from torchvision import transforms

from core.config import update_config, cfg
from utils.jotr_dataset import TRAIN_DATASETS
from utils.h36m_adapter import HUMAN36M_JOINTS

SUPPORTED = ('Human36M', 'MuCo')
# Cap trai/phai trong H36M-17
LR_PAIRS = ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16))


def parse_args():
    p = argparse.ArgumentParser(description='Do loi 2D Halpe so voi GT 2D tren tap train, khong chay model')
    p.add_argument('--cfg', required=True, help='YAML lifter (vd config/eval_difflifter_r25_cam_norot.yaml)')
    p.add_argument('--sets', default='MuCo,Human36M', help='cac bo cach nhau dau phay: ' + ','.join(SUPPORTED))
    p.add_argument('--max_samples', type=int, default=0, help='0 = toan bo; >0 lay ngau nhien moi bo')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--bad_thr', type=float, default=0.15, help='nguong loi chuan hoa theo max(w,h) bbox de coi khop sai')
    p.add_argument('--bad_min_joints', type=int, default=3, help='so khop sai toi thieu de nghi bat nham nguoi')
    p.add_argument('--swap_ratio', type=float, default=0.6, help='loi sau doi L/R < ratio * loi goc -> nghi hoan doi')
    p.add_argument('--lifter_csv', default='', help='samples.csv cua eval_difflifter_3dpw.py de ghep loi lifter')
    p.add_argument('--target_samples', type=int, default=5000, help='so mau kiem tra target 3D; 0 = tat')
    p.add_argument('--out_dir', required=True, help='thu muc MOI de luu ket qua')
    return p.parse_args()


def gt2d_h36m17(ds, name, data):
    """Tra ve GT 2D (17, 2) pixel anh goc theo thu tu HUMAN36M_JOINTS."""
    if name == 'Human36M':
        names = ds.h36m_joints_name
        src = np.asarray(data['joint_img'], dtype=np.float64)[:, :2]
    else:
        names = ds.muco_joints_name_h36m_alias   # Thorax->Neck, Spine->Torso, Head->Nose
        src = np.asarray(data['joint_img'], dtype=np.float64)[:, :2]
    out = np.zeros((17, 2), dtype=np.float64)
    for j, jn in enumerate(HUMAN36M_JOINTS):
        if jn not in names:
            raise ValueError(f'{name}: thieu khop {jn} khi doi sang H36M-17')
        out[j] = src[names.index(jn)]
    return out


def ranks(x):
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x))
    return r


def native3d_h36m17(ds, name, data):
    """GT 3D native (17, 3) mm, toa do camera, thu tu HUMAN36M_JOINTS (chua tru root)."""
    if name == 'Human36M':
        names, src = ds.h36m_joints_name, np.asarray(data['joint_cam'], dtype=np.float64)
    else:
        names, src = ds.muco_joints_name_h36m_alias, np.asarray(data['joint_cam'], dtype=np.float64)
    return np.stack([src[names.index(jn)] for jn in HUMAN36M_JOINTS])


def project(cam, cam_param):
    f = np.asarray(cam_param['focal'], dtype=np.float64).reshape(-1)
    c = np.asarray(cam_param['princpt'], dtype=np.float64).reshape(-1)
    if f.size == 1:
        f = np.repeat(f, 2)
    z = np.where(np.abs(cam[:, 2]) < 1e-6, 1e-6, cam[:, 2])
    return np.stack([cam[:, 0] / z * f[0] + c[0], cam[:, 1] / z * f[1] + c[1]], 1)


def target_check(name, ds, args, lifter_err, log, writer):
    datalist = ds.datalist
    cand = [i for i in range(len(datalist)) if datalist[i].get('smpl_param') is not None]
    n_no_smpl = len(datalist) - len(cand)
    if not cand:
        log(f'  [target3d] {name}: khong co mau nao co NeuralAnnot -> bo qua')
        return None
    rng = np.random.default_rng(args.seed + 1)
    pick = np.sort(rng.choice(len(cand), min(args.target_samples, len(cand)), replace=False))
    idxs = [cand[k] for k in pick]
    rep_nat, rep_smpl, rr_diff, root_diff, sizes, used = [], [], [], [], [], []
    n_fail = 0
    for i in idxs:
        data = datalist[i]
        try:
            gt2d = gt2d_h36m17(ds, name, data)
            nat = native3d_h36m17(ds, name, data)
            mesh = ds.get_smpl_coord(data['smpl_param'], data['cam_param'], False, data['img_shape'])[0]
            smpl17 = np.asarray(ds.h36m_joint_regressor, dtype=np.float64) @ np.asarray(mesh, dtype=np.float64)
        except Exception as e:  # mau loi tham so -> dem, khong dung lai
            n_fail += 1
            if n_fail <= 3:
                log(f'  [target3d] loi mau {i}: {type(e).__name__}: {e}')
            continue
        h, w = data['img_shape']
        ins = (gt2d[:, 0] >= 0) & (gt2d[:, 0] < w) & (gt2d[:, 1] >= 0) & (gt2d[:, 1] < h)
        tb = np.asarray(data['tight_bbox'], dtype=np.float64)
        size = max(tb[2], tb[3], 1.0)
        e_nat = np.linalg.norm(project(nat, data['cam_param']) - gt2d, axis=1)
        e_smpl = np.linalg.norm(project(smpl17, data['cam_param']) - gt2d, axis=1)
        e_nat[~ins] = np.nan
        e_smpl[~ins] = np.nan
        rep_nat.append(e_nat)
        rep_smpl.append(e_smpl)
        rr_diff.append(np.linalg.norm((nat - nat[0]) - (smpl17 - smpl17[0]), axis=1))
        root_diff.append(float(np.linalg.norm(nat[0] - smpl17[0])))
        sizes.append(size)
        used.append(i)
    if not used:
        log(f'  [target3d] {name}: tat ca mau loi -> bo qua')
        return None
    rep_nat, rep_smpl = np.asarray(rep_nat), np.asarray(rep_smpl)
    rr_diff, root_diff, sizes = np.asarray(rr_diff), np.asarray(root_diff), np.asarray(sizes)
    samp_rr = rr_diff[:, 1:].mean(1)
    res = {'n': len(used), 'n_no_smpl': n_no_smpl, 'n_fail': n_fail,
           'reproj_native_px_mean': float(np.nanmean(rep_nat)),
           'reproj_native_norm_mean': float(np.nanmean(rep_nat / sizes[:, None])),
           'reproj_smpl_px_mean': float(np.nanmean(rep_smpl)),
           'reproj_smpl_norm_mean': float(np.nanmean(rep_smpl / sizes[:, None])),
           'root_diff_mm_median': float(np.median(root_diff)),
           'rr_diff_mm_mean': float(samp_rr.mean()), 'rr_diff_mm_median': float(np.median(samp_rr)),
           'rr_diff_mm_p90': float(np.quantile(samp_rr, 0.9)),
           'pct_rr_diff_gt50mm': float((samp_rr > 50).mean()),
           'per_joint': {jn: {'rr_diff_mm': float(rr_diff[:, j].mean()),
                              'reproj_native_px': float(np.nanmean(rep_nat[:, j])) if np.isfinite(rep_nat[:, j]).any() else None,
                              'reproj_smpl_px': float(np.nanmean(rep_smpl[:, j])) if np.isfinite(rep_smpl[:, j]).any() else None}
                         for j, jn in enumerate(HUMAN36M_JOINTS)}}
    log(f'\n  [target3d] {name}: n={len(used)} (khong NeuralAnnot={n_no_smpl}, loi={n_fail})')
    log(f'    chieu lai GT3D native -> GT2D: {res["reproj_native_px_mean"]:.1f} px ({res["reproj_native_norm_mean"]:.4f} x bbox)')
    log(f'    chieu lai SMPL H36M17 -> GT2D: {res["reproj_smpl_px_mean"]:.1f} px ({res["reproj_smpl_norm_mean"]:.4f} x bbox)')
    log(f'    native vs SMPL, root-relative (bo pelvis): mean {res["rr_diff_mm_mean"]:.1f} | median {res["rr_diff_mm_median"]:.1f}'
        f' | p90 {res["rr_diff_mm_p90"]:.1f} mm | >50mm: {100 * res["pct_rr_diff_gt50mm"]:.1f}% | lech root median {res["root_diff_mm_median"]:.0f} mm')
    log('    Tung khop: ten | native-vs-SMPL mm | chieu lai native px | chieu lai SMPL px')
    for jn, v in res['per_joint'].items():
        rn = f'{v["reproj_native_px"]:.1f}' if v['reproj_native_px'] is not None else '-'
        rs = f'{v["reproj_smpl_px"]:.1f}' if v['reproj_smpl_px'] is not None else '-'
        log(f'      {jn:<10} | {v["rr_diff_mm"]:7.1f} | {rn:>7} | {rs:>7}')
    if lifter_err is not None:
        lif = np.array([lifter_err.get((name, i), np.nan) for i in used])
        ok = np.isfinite(lif)
        if ok.sum() > 10:
            sp = float(np.corrcoef(ranks(samp_rr[ok]), ranks(lif[ok]))[0, 1])
            res['spearman_rrdiff_vs_lifter'] = sp
            hi = samp_rr[ok] > 50
            res['lifter_mpjpe_rrdiff_gt50'] = float(lif[ok][hi].mean()) if hi.any() else None
            res['lifter_mpjpe_rrdiff_le50'] = float(lif[ok][~hi].mean()) if (~hi).any() else None
            log(f'    Spearman(native-vs-SMPL, MPJPE17 lifter) = {sp:.3f} | MPJPE lifter khi lech>50mm = '
                f'{res["lifter_mpjpe_rrdiff_gt50"]} / <=50mm = {res["lifter_mpjpe_rrdiff_le50"]}')
    for r, i in enumerate(used):
        writer.writerow({'dataset': name, 'index': i, 'rr_diff_mm': f'{samp_rr[r]:.2f}',
                         'root_diff_mm': f'{root_diff[r]:.1f}',
                         'reproj_native_px': f'{np.nanmean(rep_nat[r]):.2f}' if np.isfinite(rep_nat[r]).any() else '',
                         'reproj_smpl_px': f'{np.nanmean(rep_smpl[r]):.2f}' if np.isfinite(rep_smpl[r]).any() else ''})
    return res


def analyze(name, args, lifter_err, log, writer, twriter=None):
    ds = TRAIN_DATASETS[name](transforms.ToTensor(), 'train')
    datalist = ds.datalist
    total = len(datalist)
    if total == 0:
        raise ValueError(f'{name}: datalist rong')
    if args.max_samples > 0:
        idx_all = np.sort(np.random.default_rng(args.seed).choice(total, min(args.max_samples, total), replace=False))
    else:
        idx_all = np.arange(total)
    thr = cfg.DATASET.halpe26_conf_thr

    err_px, err_n, score, inside, rows_idx = [], [], [], [], []
    n_no_halpe, n_swap = 0, 0
    swap_flags = []
    for i in idx_all:
        data = datalist[i]
        if data.get('halpe_h36m') is None:
            n_no_halpe += 1
            continue
        k = np.asarray(data['halpe_h36m'], dtype=np.float64)
        gt = gt2d_h36m17(ds, name, data)
        h, w = data['img_shape']
        tb = np.asarray(data['tight_bbox'], dtype=np.float64)
        size = max(tb[2], tb[3], 1.0)
        ins = (gt[:, 0] >= 0) & (gt[:, 0] < w) & (gt[:, 1] >= 0) & (gt[:, 1] < h)
        e = np.linalg.norm(k[:, :2] - gt, axis=1)
        # Thu doi trai/phai cua Halpe
        ks = k.copy()
        for a, b in LR_PAIRS:
            ks[[a, b]] = ks[[b, a]]
        es = np.linalg.norm(ks[:, :2] - gt, axis=1)
        m = ins.astype(np.float64)
        base = (e * m).sum() / max(m.sum(), 1)
        swp = (es * m).sum() / max(m.sum(), 1)
        is_swap = bool(base > 0 and swp < args.swap_ratio * base and base / size > 0.05)
        n_swap += int(is_swap)
        swap_flags.append(is_swap)
        err_px.append(e)
        err_n.append(e / size)
        score.append(k[:, 2])
        inside.append(ins)
        rows_idx.append(int(i))

    err_px = np.asarray(err_px)
    err_n = np.asarray(err_n)
    score = np.asarray(score)
    inside = np.asarray(inside)
    swap_flags = np.asarray(swap_flags)
    n = len(rows_idx)
    if n == 0:
        raise ValueError(f'{name}: khong co mau nao co Halpe')
    msk = inside.astype(np.float64)
    bad_joint = (err_n > args.bad_thr) & inside
    n_bad = bad_joint.sum(1)
    samp_mean = (err_n * msk).sum(1) / np.maximum(msk.sum(1), 1)
    suspect = n_bad >= args.bad_min_joints

    def masked_mean(a, mm):
        return float((a * mm).sum() / max(mm.sum(), 1))

    per_joint = {}
    for j, jn in enumerate(HUMAN36M_JOINTS):
        mj = inside[:, j]
        ej = err_n[mj, j]
        hi = mj & (score[:, j] > thr)
        lo = mj & (score[:, j] <= thr)
        per_joint[jn] = {
            'mean_px': float(err_px[mj, j].mean()) if mj.any() else None,
            'median_norm': float(np.median(ej)) if mj.any() else None,
            'pck05': float((ej <= 0.05).mean()) if mj.any() else None,
            'pck10': float((ej <= 0.10).mean()) if mj.any() else None,
            'pct_bad': float((ej > args.bad_thr).mean()) if mj.any() else None,
            'pct_low_conf': float((score[mj, j] <= thr).mean()) if mj.any() else None,
            'mean_norm_conf_hi': float(err_n[hi, j].mean()) if hi.any() else None,
            'mean_norm_conf_lo': float(err_n[lo, j].mean()) if lo.any() else None,
            'pct_gt_inside': float(mj.mean()),
        }
    res = {'dataset': name, 'datalist_n': total, 'n': n, 'n_no_halpe': n_no_halpe,
           'mean_norm_err': masked_mean(err_n, msk), 'mean_px_err': masked_mean(err_px, msk),
           'pct_suspect_wrong_person': float(suspect.mean()), 'pct_lr_swap': float(swap_flags.mean()),
           'bad_thr': args.bad_thr, 'bad_min_joints': args.bad_min_joints, 'per_joint': per_joint}

    log(f'\n[{name}] n={n}/{total} | khong Halpe={n_no_halpe}')
    log(f'  loi 2D trung binh = {res["mean_px_err"]:.1f} px | {res["mean_norm_err"]:.4f} x max(w,h) bbox')
    log(f'  nghi bat nham nguoi (>= {args.bad_min_joints} khop loi > {args.bad_thr}): {100 * suspect.mean():.2f}%'
        f' | nghi hoan doi L/R: {100 * swap_flags.mean():.2f}%')
    log('  Tung khop: ten | px | median_norm | PCK@0.1 | %sai | %conf<=thr | norm conf>thr / conf<=thr')
    for jn, v in per_joint.items():
        if v['mean_px'] is None:
            continue
        hi_s = f'{v["mean_norm_conf_hi"]:.3f}' if v['mean_norm_conf_hi'] is not None else '-'
        lo_s = f'{v["mean_norm_conf_lo"]:.3f}' if v['mean_norm_conf_lo'] is not None else '-'
        log(f'    {jn:<10} | {v["mean_px"]:7.1f} | {v["median_norm"]:.3f} | {100 * v["pck10"]:5.1f} | '
            f'{100 * v["pct_bad"]:5.1f} | {100 * v["pct_low_conf"]:5.1f} | {hi_s} / {lo_s}')

    # Ghep loi lifter (cung index datalist)
    lif = None
    if lifter_err is not None:
        lif = np.array([lifter_err.get((name, i), np.nan) for i in rows_idx], dtype=np.float64)
        ok = np.isfinite(lif)
        res['lifter_joined'] = int(ok.sum())
        if ok.sum() < 0.9 * n:
            log(f'  CHU Y: chi ghep duoc {int(ok.sum())}/{n} mau voi lifter_csv; kiem tra cung YAML va --max_samples 0')
        if ok.sum() > 10:
            x, y = samp_mean[ok], lif[ok]
            sp = float(np.corrcoef(ranks(x), ranks(y))[0, 1])
            res['spearman_2derr_vs_lifter'] = sp
            res['lifter_mpjpe_suspect'] = float(y[suspect[ok]].mean()) if suspect[ok].any() else None
            res['lifter_mpjpe_clean'] = float(y[~suspect[ok]].mean()) if (~suspect[ok]).any() else None
            res['lifter_mpjpe_swap'] = float(y[swap_flags[ok]].mean()) if swap_flags[ok].any() else None
            qs = np.quantile(x, [0.2, 0.4, 0.6, 0.8])
            bins = np.digitize(x, qs)
            res['lifter_by_2d_quintile'] = []
            log(f'  Ghep lifter: n={int(ok.sum())} | Spearman(loi 2D, MPJPE17 lifter) = {sp:.3f}')
            log(f'  MPJPE17 lifter: mau sach = {res["lifter_mpjpe_clean"]} | nghi nham nguoi = '
                f'{res["lifter_mpjpe_suspect"]} | nghi doi L/R = {res["lifter_mpjpe_swap"]} (mm)')
            log('  Theo nhom loi 2D (Q1 tot -> Q5 te): loi2D_norm trung binh | MPJPE17 lifter | so mau')
            for b in range(5):
                sel = bins == b
                if sel.any():
                    item = {'q': b + 1, 'n': int(sel.sum()), 'mean_2d_norm': float(x[sel].mean()),
                            'lifter_mpjpe17_mm': float(y[sel].mean())}
                    res['lifter_by_2d_quintile'].append(item)
                    log(f'    Q{b + 1}: {item["mean_2d_norm"]:.4f} | {item["lifter_mpjpe17_mm"]:.2f} | {item["n"]}')
            # Neu bo mau nghi nham nguoi thi loi lifter con bao nhieu
            res['lifter_mpjpe_all'] = float(y.mean())
            log(f'  MPJPE17 lifter toan bo = {y.mean():.2f} | bo mau nghi nham nguoi = '
                f'{y[~suspect[ok]].mean():.2f} mm (chi la phep loc chan doan, khong phai ket qua train lai)')

    for r, i in enumerate(rows_idx):
        writer.writerow({'dataset': name, 'index': i, 'img_path': datalist[i].get('img_path', ''),
                         'mean_norm_err': f'{samp_mean[r]:.5f}', 'n_bad_joints': int(n_bad[r]),
                         'suspect_wrong_person': bool(suspect[r]), 'lr_swap': bool(swap_flags[r]),
                         'n_gt_inside': int(inside[r].sum()),
                         'lifter_mpjpe17_mm': (f'{lif[r]:.3f}' if lif is not None and np.isfinite(lif[r]) else ''),
                         **{f'norm_{jn}': f'{err_n[r, j]:.4f}' for j, jn in enumerate(HUMAN36M_JOINTS)}})
    if args.target_samples > 0 and twriter is not None:
        res['target3d'] = target_check(name, ds, args, lifter_err, log, twriter)
    return res


def load_lifter_csv(path):
    out = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('split') != 'train' or not row.get('mpjpe17_mm'):
                continue
            out[(row['dataset'], int(row['index']))] = float(row['mpjpe17_mm'])
    if not out:
        raise ValueError(f'{path}: khong co dong split=train co mpjpe17_mm')
    return out


def main():
    args = parse_args()
    update_config(args.cfg)
    names = [s.strip() for s in args.sets.split(',') if s.strip()]
    for nm in names:
        if nm not in SUPPORTED:
            raise ValueError(f'{nm} chua ho tro; chi {SUPPORTED}')
    out_dir = Path(args.out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f'Thu muc ket qua khong rong: {out_dir}; chon thu muc moi')
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)
        (out_dir / 'log.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    log(f'cfg={args.cfg} | sets={names} | lift_only={cfg.DATASET.lift_only} | conf_thr={cfg.DATASET.halpe26_conf_thr}')
    log('Loi 2D o pixel anh goc, khong augmentation; chuan hoa theo max(w,h) tight bbox GT; chi khop GT nam trong anh.')
    lifter_err = load_lifter_csv(args.lifter_csv) if args.lifter_csv else None
    fields = ['dataset', 'index', 'img_path', 'mean_norm_err', 'n_bad_joints', 'suspect_wrong_person', 'lr_swap',
              'n_gt_inside', 'lifter_mpjpe17_mm'] + [f'norm_{jn}' for jn in HUMAN36M_JOINTS]
    summary = {'args': vars(args), 'results': []}
    tfields = ['dataset', 'index', 'rr_diff_mm', 'root_diff_mm', 'reproj_native_px', 'reproj_smpl_px']
    with (out_dir / 'samples_2d.csv').open('w', encoding='utf-8', newline='') as f, \
            (out_dir / 'samples_target3d.csv').open('w', encoding='utf-8', newline='') as tf:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        twriter = csv.DictWriter(tf, fieldnames=tfields)
        twriter.writeheader()
        for nm in names:
            log(f'==================== {nm} ====================')
            summary['results'].append(analyze(nm, args, lifter_err, log, writer, twriter))
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    log(f'\nKet qua: {out_dir}')


if __name__ == '__main__':
    main()
