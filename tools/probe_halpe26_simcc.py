"""
Trich thu SimCC Halpe-26 tren mot tap nho 3DPW bang model RTMPose hien co.
Chay rieng trong moi truong pose2d, khong import torch hay config.py.

Dau ra rieng: <out_dir>/3DPW_<split>_halpe26_simcc_probe.npz, *_summary.json, vis/*.png.
simcc_x/y la DAU RA THO float32 cua ONNX, chua mac dinh coi la xac suat.
Moi mau giu center/scale va crop_to_image de dua toa do crop ve pixel anh goc.
Khong doi file Halpe cu, dataset hay mo hinh DiffLifter.

Nguon API da doi chieu:
https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/pose_estimation/rtmpose.py

Vi du:
  python tools/probe_halpe26_simcc.py --cfg config/probe_halpe26_simcc.yaml --inspect_only
  python tools/probe_halpe26_simcc.py --cfg config/probe_halpe26_simcc.yaml --split validation
  python tools/probe_halpe26_simcc.py --cfg config/probe_halpe26_simcc_temperatures.yaml --out_dir output/probe_halpe26_simcc_temperatures
"""
import argparse
import inspect
import json
import os
import os.path as osp
from importlib import metadata

import cv2
import numpy as np
from tqdm import tqdm

from extract_halpe26_3dpw import build_model, draw, HALPE26_SKELETON


# Giu dong bo voi nhom HALPE_SIMCC trong lib/core/config.py.
DEFAULTS = dict(enabled=False, max_people=32, vis_num=8, seed=42,
                split_ratio=2.0, normalization='softmax', temperature=1.0,
                from_npz='', temperatures=[])
VIS_JOINTS = [(9, 'L_Wrist'), (10, 'R_Wrist'), (15, 'L_Ankle'), (16, 'R_Ankle')]
COLORS = [(40, 70, 230), (220, 100, 30), (70, 170, 50), (190, 60, 170)]


def parse_args():
    p = argparse.ArgumentParser(description='Probe dau ra SimCC cua RTMPose Halpe-26')
    p.add_argument('--cfg', default='config/probe_halpe26_simcc.yaml')
    p.add_argument('--data_root', default=osp.join(os.environ.get('JOTR_DATA_ROOT', 'data_final'), 'PW3D'))
    p.add_argument('--split', default='validation', choices=['train', 'validation', 'test', 'oc', 'pc'])
    p.add_argument('--out_dir', default='output/probe_halpe26_simcc')
    p.add_argument('--mode', default='performance', choices=['performance', 'balanced', 'lightweight'])
    p.add_argument('--onnx', default='')
    p.add_argument('--input_size', type=int, nargs=2, default=None, metavar=('W', 'H'))
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--inspect_only', action='store_true', help='Chi xem API va input/output ONNX, khong doc dataset')
    p.add_argument('--max_people', type=int, default=None)
    p.add_argument('--vis_num', type=int, default=None)
    p.add_argument('--temperature', type=float, default=None)
    p.add_argument('--normalization', choices=['softmax', 'positive_mass'], default=None)
    p.add_argument('--from_npz', default=None, help='Doc raw da luu, khong chay detector')
    p.add_argument('--temperatures', type=float, nargs='+', default=None, help='So sanh T khi dung --from_npz')
    p.add_argument('--overwrite', action='store_true', help='Cho phep ghi de dau ra probe cung split')
    return p.parse_args()


def load_settings(args):
    # Chi doc YAML nay, tranh import core.config co tac dung phu va can easydict.
    import yaml
    with open(args.cfg, encoding='utf-8') as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict) or set(doc) != {'HALPE_SIMCC'}:
        raise ValueError('YAML probe chi duoc chua nhom HALPE_SIMCC')
    section = doc['HALPE_SIMCC']
    if not isinstance(section, dict):
        raise ValueError('HALPE_SIMCC phai la mapping')
    unknown = set(section) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Key HALPE_SIMCC khong ho tro: {sorted(unknown)}')
    settings = dict(DEFAULTS)
    settings.update(section)
    for name in ('max_people', 'vis_num', 'temperature', 'normalization', 'from_npz', 'temperatures'):
        if getattr(args, name) is not None:
            settings[name] = getattr(args, name)
    if not isinstance(settings['enabled'], bool) or not settings['enabled']:
        raise ValueError('Can HALPE_SIMCC.enabled: True trong YAML probe')
    for name in ('max_people', 'vis_num', 'seed'):
        if isinstance(settings[name], bool) or not isinstance(settings[name], int):
            raise ValueError(f'{name} phai la so nguyen')
    if settings['max_people'] <= 0 or settings['vis_num'] < 0 or not 0 <= settings['seed'] < 2 ** 32:
        raise ValueError('Can max_people > 0, vis_num >= 0 va 0 <= seed < 2**32')
    for name in ('split_ratio', 'temperature'):
        settings[name] = float(settings[name])
        if not np.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(f'{name} phai huu han va > 0')
    if settings['normalization'] not in ('softmax', 'positive_mass'):
        raise ValueError('normalization chi nhan softmax hoac positive_mass')
    if not isinstance(settings['from_npz'], str):
        raise ValueError('from_npz phai la duong dan chuoi, hoac rong de trich moi')
    values = settings['temperatures']
    if not isinstance(values, (list, tuple)):
        raise ValueError('temperatures phai la danh sach')
    if any(isinstance(v, bool) for v in values):
        raise ValueError('temperature khong duoc la boolean')
    values = [float(v) for v in values]
    if any(not np.isfinite(v) or v <= 0 for v in values) or len(set(values)) != len(values):
        raise ValueError('temperatures phai huu han, > 0 va khong trung nhau')
    settings['temperatures'] = values
    if values and (not settings['from_npz'] or settings['normalization'] != 'softmax'):
        raise ValueError('Quet temperatures can from_npz va normalization=softmax')
    return settings


def model_report(model, settings):
    for name in ('preprocess', 'inference', 'postprocess'):
        if not callable(getattr(model, name, None)):
            raise RuntimeError(f'Ban rtmlib nay khong co RTMPose.{name}; gui lai log API de doi chieu')
    signature = inspect.signature(model.postprocess)
    if 'simcc_split_ratio' not in signature.parameters:
        raise RuntimeError(f'API postprocess khac ban da doi chieu: {signature}')
    session = getattr(model, 'session', None)
    if session is None or not hasattr(session, 'get_outputs'):
        raise RuntimeError('Can backend onnxruntime co session.get_outputs()')
    size = np.asarray(model.model_input_size, dtype=np.float32)
    if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any():
        raise RuntimeError(f'model_input_size khong hop le: {size}')
    info = dict(model_class=f'{type(model).__module__}.{type(model).__name__}',
                model_source=str(getattr(model, 'onnx_model', 'unknown')),
                input_size_wh=size.tolist(), providers=session.get_providers(),
                methods={name: str(inspect.signature(getattr(model, name)))
                         for name in ('preprocess', 'inference', 'postprocess')})
    info['versions'] = {}
    for name in ('rtmlib', 'onnxruntime-gpu', 'onnxruntime'):
        try:
            info['versions'][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    for kind, nodes in [('inputs', session.get_inputs()), ('outputs', session.get_outputs())]:
        info[kind] = [dict(name=n.name, shape=list(n.shape), dtype=n.type) for n in nodes]
    print(json.dumps(info, indent=2))
    if len(info['outputs']) != 2:
        raise RuntimeError('Model ONNX khong co dung 2 output SimCC. Khong tu suy phan bo tu keypoint da decode.')
    for node, dim in zip(session.get_outputs(), size):
        shape = node.shape
        if len(shape) != 3:
            raise RuntimeError(f'Output {node.name} can dang (B,26,L), nhan {shape}')
        if isinstance(shape[1], int) and shape[1] != 26:
            raise RuntimeError(f'Output {node.name} khong phai Halpe-26: {shape}')
        if isinstance(shape[2], int) and not np.isclose(shape[2], dim * settings['split_ratio']):
            raise RuntimeError(f'Output {node.name}: {shape}; khong khop input_size/split_ratio. '
                               'Kiem tra thu tu output X/Y va YAML truoc khi trich.')
    return info


def select_people(args, settings):
    path = osp.join(args.data_root, f'3DPW_latest_{args.split}.json')
    if not osp.isfile(path):
        path = osp.join(args.data_root, f'3DPW_{args.split}.json')
    with open(path, encoding='utf-8') as f:
        db = json.load(f)
    images = {im['id']: im for im in db['images']}
    anns = sorted(db['annotations'], key=lambda a: int(a['id']))
    if len({int(a['id']) for a in anns}) != len(anns):
        raise ValueError(f'Trung annotation id trong {path}')
    if not anns:
        raise ValueError(f'Khong co annotation trong {path}')
    rng = np.random.RandomState(settings['seed'])
    count = min(settings['max_people'], len(anns))
    chosen = rng.choice(len(anns), size=count, replace=False)
    # Lay mau tren toan split de tranh chi lay mot chuoi frame lien tiep.
    people = [(anns[int(i)], images[anns[int(i)]['image_id']]) for i in chosen]
    return path, people


def normalize_simcc(raw, settings):
    # Quy uoc de ve/lay mau thu, CHUA la xac suat da hieu chuan.
    if settings['normalization'] == 'softmax':
        z = raw.astype(np.float64) / settings['temperature']
        weights = np.exp(z - z.max(axis=-1, keepdims=True))
    else:
        if (raw < 0).any():
            raise ValueError('positive_mass yeu cau raw >= 0; dung softmax neu output la logits')
        weights = raw.astype(np.float64)
    total = weights.sum(axis=-1, keepdims=True)
    if not np.isfinite(total).all() or (total <= 0).any():
        raise ValueError('Khong chuan hoa duoc SimCC: tong rong/khong huu han')
    return (weights / total).astype(np.float32)


def distribution_stats(prob, pixel_per_bin):
    bins = np.arange(prob.shape[-1], dtype=np.float64)
    mean = (prob * bins).sum(axis=-1)
    var = (prob * (bins[None] - mean[:, None]) ** 2).sum(axis=-1)
    sigma_px = np.sqrt(np.maximum(var, 0)) * pixel_per_bin
    entropy = -(prob * np.log(np.maximum(prob, 1e-30))).sum(axis=-1) / np.log(prob.shape[-1])
    return sigma_px.astype(np.float32), entropy.astype(np.float32)


def extract_person(model, image, bbox, settings):
    crop, center, scale = model.preprocess(image, bbox)
    center = np.asarray(center, dtype=np.float32).reshape(2)
    scale = np.asarray(scale, dtype=np.float32).reshape(2)
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('preprocess tra center/scale khong hop le')
    outputs = model.inference(crop)
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
        raise RuntimeError('inference khong tra 2 mang SimCC X/Y')
    size = np.asarray(model.model_input_size, dtype=np.float32)
    ratio = settings['split_ratio']
    raw = []
    for axis, output in enumerate(outputs):
        arr = np.asarray(output, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[:2] != (1, 26) or arr.shape[2] < 2:
            raise RuntimeError(f'SimCC axis {axis} phai la (1,26,L), nhan {arr.shape}')
        if not np.isclose(arr.shape[-1], size[axis] * ratio):
            raise RuntimeError(f'SimCC axis {axis} co L={arr.shape[-1]}, input={size[axis]}, ratio={ratio}')
        if not np.isfinite(arr).all():
            raise ValueError(f'SimCC axis {axis} co NaN/Inf')
        raw.append(arr[0].copy())
    keypoints, scores = model.postprocess(outputs, center, scale, simcc_split_ratio=ratio)
    keypoints, scores = np.asarray(keypoints), np.asarray(scores)
    if keypoints.shape != (1, 26, 2) or scores.shape != (1, 26):
        raise RuntimeError(f'postprocess tra shape khac du kien: {keypoints.shape}, {scores.shape}')
    keypoints, scores = keypoints[0].astype(np.float32), scores[0].astype(np.float32)
    if not np.isfinite(keypoints).all() or not np.isfinite(scores).all():
        raise ValueError('Keypoint/score co NaN/Inf')
    factor = scale / size
    offset = center - scale / 2
    affine = np.array([[factor[0], 0, offset[0]], [0, factor[1], offset[1]]], dtype=np.float32)
    # Doi argmax ve anh goc roi doi chieu decoder cua chinh rtmlib dang cai.
    argmax = np.stack([r.argmax(axis=-1) for r in raw], axis=-1).astype(np.float32)
    reconstructed = argmax / ratio * factor + offset
    valid = scores > 0
    decode_error = float(np.abs(reconstructed[valid] - keypoints[valid]).max()) if valid.any() else 0.0
    if decode_error > 0.05:
        raise RuntimeError(f'Anh xa SimCC -> pixel lech decoder {decode_error:.4f}px; '
                           'dung lai de kiem tra API, khong luu du lieu sai he toa do')
    probs = [normalize_simcc(r, settings) for r in raw]
    stats = [distribution_stats(p, factor[a] / ratio) for a, p in enumerate(probs)]
    return dict(simcc_x=raw[0], simcc_y=raw[1], kpts=np.concatenate([keypoints, scores[:, None]], axis=-1),
                center=center, scale=scale, crop_to_image=affine,
                sigma_image_px=np.stack([s[0] for s in stats], axis=-1),
                entropy_norm=np.stack([s[1] for s in stats], axis=-1),
                decode_error_px=np.float32(decode_error)), probs


def curve_panel(prob, title, color, width=350, height=130):
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    left, right, top, bottom = 12, width - 12, 40, height - 22
    cv2.putText(panel, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (25, 25, 25), 1)
    cv2.putText(panel, f'peak={prob.max():.4f}  bins=0..{len(prob)-1}', (8, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1)
    xs = np.linspace(left, right, len(prob))
    ys = bottom - prob / max(float(prob.max()), 1e-12) * (bottom - top)
    points = np.stack([xs, ys], axis=-1).round().astype(np.int32)
    cv2.line(panel, (left, bottom), (right, bottom), (150, 150, 150), 1)
    cv2.polylines(panel, [points], False, color, 1, cv2.LINE_AA)
    peak_x = int(xs[prob.argmax()])
    cv2.line(panel, (peak_x, top), (peak_x, bottom), (180, 180, 180), 1)
    return panel


def save_visualization(path, image, record, probs, settings, seed):
    size = np.asarray(record['input_size_wh'], dtype=np.float32)
    width = 320
    height = max(1, int(round(width * size[1] / size[0])))
    scale, center = record['scale'], record['center']
    factors = np.array([width, height], dtype=np.float32) / scale
    offset = -(center - scale / 2) * factors
    matrix = np.array([[factors[0], 0, offset[0]], [0, factors[1], offset[1]]], dtype=np.float32)
    crop = cv2.warpAffine(image, matrix, (width, height))
    kpts = record['kpts'].copy()
    kpts[:, :2] = kpts[:, :2] * factors + offset
    rng = np.random.RandomState(seed)
    # Tich hai phan bo 1D chi dung minh hoa; chua bieu dien tuong quan X/Y hay giua cac khop.
    for (joint, _), color in zip(VIS_JOINTS, COLORS):
        px, py = probs[0][joint], probs[1][joint]
        sample_x = rng.choice(len(px), size=50, p=px.astype(np.float64) / px.sum(dtype=np.float64))
        sample_y = rng.choice(len(py), size=50, p=py.astype(np.float64) / py.sum(dtype=np.float64))
        xy = np.stack([sample_x, sample_y], axis=-1) / settings['split_ratio'] / size * [width, height]
        for point in xy:
            cv2.circle(crop, tuple(np.rint(point).astype(int)), 1, color, -1)
    draw(crop, kpts[:, :2], kpts[:, 2], HALPE26_SKELETON, 0.3)
    canvas = np.full((max(height, 4 * 130) + 65, width + 20 + 700, 3), 245, dtype=np.uint8)
    canvas[55:55 + height, :width] = crop
    title = f"ann={record['ann_id']}  {settings['normalization']} T={settings['temperature']:g} (UNCALIBRATED)"
    cv2.putText(canvas, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1)
    cv2.putText(canvas, 'Dots: independent X/Y samples. Curves: relative height, not shared Y scale.',
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1)
    for row, ((joint, name), color) in enumerate(zip(VIS_JOINTS, COLORS)):
        for axis, label in enumerate(('X', 'Y')):
            sigma = record['sigma_image_px'][joint, axis]
            entropy = record['entropy_norm'][joint, axis]
            title = f'{name} {label}: std={sigma:.1f}px H={entropy:.2f}'
            panel = curve_panel(probs[axis][joint], title, color)
            y, x = 55 + row * 130, width + 20 + axis * 350
            canvas[y:y + 130, x:x + 350] = panel
    if not cv2.imwrite(path, canvas):
        raise IOError(f'Khong ghi duoc anh: {path}')


def replot_saved(args, settings):
    # Nhanh CPU: doc raw, khong build_model va khong doc annotation/GT.
    required = ('simcc_x', 'simcc_y', 'kpts', 'center', 'scale', 'input_size_wh',
                'ann_id', 'file_name')
    with np.load(settings['from_npz'], allow_pickle=False) as archive:
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f'NPZ thieu key: {sorted(missing)}')
        arrays = {key: archive[key].copy() for key in required}
        if 'split_ratio' not in archive.files:
            raise ValueError('NPZ thieu split_ratio de doi bin ve pixel')
        ratio = float(np.asarray(archive['split_ratio']).item())
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError('split_ratio trong NPZ khong hop le')
    raw_x, raw_y = arrays['simcc_x'], arrays['simcc_y']
    if raw_x.ndim != 3 or raw_y.ndim != 3 or raw_x.shape[:2] != raw_y.shape[:2]:
        raise ValueError('SimCC trong NPZ can dang (N,26,L), cung N va so khop')
    count, joints = raw_x.shape[:2]
    if count == 0 or joints != 26 or raw_x.shape[-1] < 2 or raw_y.shape[-1] < 2:
        raise ValueError('NPZ rong hoac khong phai Halpe-26 SimCC')
    shapes = dict(kpts=(count, 26, 3), center=(count, 2), scale=(count, 2),
                  input_size_wh=(count, 2), ann_id=(count,), file_name=(count,))
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f'{key}: can {shape}, nhan {arrays[key].shape}')
    for key in ('simcc_x', 'simcc_y', 'kpts', 'center', 'scale', 'input_size_wh'):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f'{key} co NaN/Inf')
    if (arrays['scale'] <= 0).any() or (arrays['input_size_wh'] <= 0).any():
        raise ValueError('scale/input_size phai > 0')
    if not np.allclose(arrays['input_size_wh'] * ratio, [raw_x.shape[-1], raw_y.shape[-1]]):
        raise ValueError('Shape raw khong khop input_size/split_ratio trong NPZ')
    count = min(count, settings['max_people'])
    # Giu cung nguoi va seed cho cac T, chi thay doi phep chuan hoa.
    arrays = {key: value[:count] for key, value in arrays.items()}
    temperatures = settings['temperatures'] or [settings['temperature']]
    summary_path = osp.join(args.out_dir, 'temperature_sweep_summary.json')
    folders = [osp.join(args.out_dir, f'T{value:g}'.replace('.', 'p')) for value in temperatures]
    if not args.overwrite:
        for path in [summary_path] + folders:
            if osp.exists(path):
                raise FileExistsError(f'Dau ra da co: {path}. Doi --out_dir hoac dung --overwrite.')
    os.makedirs(args.out_dir, exist_ok=True)
    report = dict(source_npz=osp.abspath(settings['from_npz']), settings=settings,
                  used_people=count, split_ratio_from_npz=ratio,
                  note='Chi so sanh temperature tren cung raw. Chua hieu chuan bang GT.', results=[])
    print('[REPLOT] Doc SimCC da luu; khong nap RTMPose/ONNX, khong can GPU.')
    print('T | median std X/Y (px) | median H X/Y | median peak X/Y')
    for temperature, folder in zip(temperatures, folders):
        local = dict(settings, temperature=temperature, split_ratio=ratio)
        if settings['vis_num']:
            os.makedirs(folder, exist_ok=True)
        sigma_all, entropy_all, peak_all, image_missing = [], [], [], []
        written = 0
        for index in range(count):
            record = {key: values[index] for key, values in arrays.items()}
            probs = [normalize_simcc(record[key], local) for key in ('simcc_x', 'simcc_y')]
            factors = record['scale'] / record['input_size_wh'] / ratio
            stats = [distribution_stats(prob, factors[a]) for a, prob in enumerate(probs)]
            sigma = np.stack([s[0] for s in stats], axis=-1)
            entropy = np.stack([s[1] for s in stats], axis=-1)
            sigma_all.append(sigma)
            entropy_all.append(entropy)
            peak_all.append(np.stack([p.max(axis=-1) for p in probs], axis=-1))
            if index < settings['vis_num']:
                image_path = osp.join(args.data_root, str(record['file_name']))
                image = cv2.imread(image_path)
                if image is None:
                    image_missing.append(image_path)
                    continue
                record.update(sigma_image_px=sigma, entropy_norm=entropy)
                name = f'ann{int(record["ann_id"])}.png'
                save_visualization(osp.join(folder, name), image, record, probs, local,
                                   (settings['seed'] + index) % (2 ** 32))
                written += 1
        sigma_all, entropy_all = np.stack(sigma_all), np.stack(entropy_all)
        result = dict(temperature=temperature,
                      sigma_image_px_median_xy=np.median(sigma_all, axis=(0, 1)).tolist(),
                      entropy_norm_median_xy=np.median(entropy_all, axis=(0, 1)).tolist(),
                      peak_probability_median_xy=np.median(np.stack(peak_all), axis=(0, 1)).tolist(),
                      images_written=written, missing_images=image_missing, vis_dir=folder)
        # Score detector khong phai nhan visibility; chi tach nhom de tham khao.
        result['by_detector_score'] = {}
        for label, mask in [('score_le_0p3', arrays['kpts'][..., 2] <= 0.3),
                            ('score_gt_0p3', arrays['kpts'][..., 2] > 0.3)]:
            result['by_detector_score'][label] = dict(n_joints=int(mask.sum()))
            if mask.any():
                result['by_detector_score'][label].update(
                    sigma_image_px_median_xy=np.median(sigma_all[mask], axis=0).tolist(),
                    entropy_norm_median_xy=np.median(entropy_all[mask], axis=0).tolist())
        report['results'].append(result)
        print(f'{temperature:g} | {result["sigma_image_px_median_xy"]} | '
              f'{result["entropy_norm_median_xy"]} | {result["peak_probability_median_xy"]}')
        if image_missing:
            print(f'[CANH BAO] Thieu {len(image_missing)} anh de ve; kiem tra --data_root.')
    with open(summary_path, 'w' if args.overwrite else 'x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f'[OK] {summary_path}')
    print('[LUU Y] Giam T lam phan bo sac hon; khong tu dong chung minh uncertainty dung hon.')


def main():
    args = parse_args()
    settings = load_settings(args)
    if settings['from_npz']:
        if args.inspect_only:
            raise ValueError('inspect_only kiem tra ONNX; khong dung chung voi from_npz')
        replot_saved(args, settings)
        return
    stem = f'3DPW_{args.split}_halpe26_simcc_probe'
    npz_path = osp.join(args.out_dir, stem + '.npz')
    summary_path = osp.join(args.out_dir, stem + '_summary.json')
    if not args.inspect_only and not args.overwrite:
        for path in (npz_path, summary_path):
            if osp.exists(path):
                raise FileExistsError(f'Dau ra da co: {path}. Doi --out_dir hoac dung --overwrite.')
    model = build_model(args)
    info = model_report(model, settings)
    if args.inspect_only:
        print('[OK] Metadata ONNX/API phu hop. Chua chay inference; can trich thu de kiem tra output thuc.')
        return
    ann_path, people = select_people(args, settings)
    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = osp.join(args.out_dir, 'vis')
    if settings['vis_num']:
        os.makedirs(vis_dir, exist_ok=True)
    records, missing = [], []
    for index, (ann, im) in enumerate(tqdm(people, desc=f'SimCC {args.split}')):
        relative = osp.join('imageFiles', im['sequence'], im['file_name'])
        image_path = osp.join(args.data_root, relative)
        image = cv2.imread(image_path)
        if image is None:
            missing.append(dict(ann_id=int(ann['id']), image_path=image_path))
            continue
        x, y, w, h = [float(v) for v in ann['bbox']]
        bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
        if not np.isfinite(bbox).all() or w <= 0 or h <= 0:
            raise ValueError(f'BBox khong hop le, ann={ann["id"]}: {bbox}')
        try:
            record, probs = extract_person(model, image, bbox.tolist(), settings)
        except Exception as exc:
            raise RuntimeError(f'Loi khi trich ann={ann["id"]}, anh={image_path}: {exc}') from exc
        record.update(ann_id=int(ann['id']), image_id=int(im['id']), file_name=relative,
                      bbox_xyxy=bbox, image_hw=np.array(image.shape[:2], dtype=np.int32),
                      input_size_wh=np.asarray(model.model_input_size, dtype=np.float32))
        if records and any(record[k].shape != records[0][k].shape for k in ('simcc_x', 'simcc_y')):
            raise RuntimeError('Shape SimCC thay doi giua cac mau')
        records.append(record)
        if len(records) <= settings['vis_num']:
            save_visualization(osp.join(vis_dir, f'{args.split}_ann{ann["id"]}.png'),
                               image, record, probs, settings, (settings['seed'] + index) % (2 ** 32))
    if not records:
        raise RuntimeError(f'Khong doc duoc anh nao; kiem tra --data_root. Vi du: {missing[:3]}')
    arrays = {key: np.stack([r[key] for r in records]) for key in records[0]}
    arrays.update(format_version=np.int32(1), joint_set=np.asarray('Halpe-26'),
                  split_ratio=np.float32(settings['split_ratio']),
                  normalization=np.asarray(settings['normalization']), temperature=np.float32(settings['temperature']))
    mode = 'wb' if args.overwrite else 'xb'
    with open(npz_path, mode) as f:
        np.savez_compressed(f, **arrays)
    summary = dict(settings=settings, model=info, annotation_file=ann_path,
                   selected_people=len(people), saved_people=len(records), missing_images=missing,
                   array_shapes={key: list(value.shape) for key, value in arrays.items()},
                   decode_error_max_px=float(arrays['decode_error_px'].max()),
                   notes=['simcc_x/y: raw float32, not assumed calibrated probabilities',
                          'kpts: Halpe-26 [x,y,score] in original image pixels; score from rtmlib decoder',
                          'crop_to_image: affine mapping detector INPUT crop pixels to original image pixels',
                          'SimCC bin -> crop pixel: divide bin by split_ratio before crop_to_image',
                          'sigma_image_px/entropy_norm: diagnostic normalization only, not calibrated uncertainty',
                          'X/Y normalized separately; samples assume independent axes; no H36M conversion yet'])
    summary['raw_stats'] = {}
    for key in ('simcc_x', 'simcc_y'):
        values = arrays[key]
        summary['raw_stats'][key] = dict(min=float(values.min()), max=float(values.max()),
                                         negative_fraction=float((values < 0).mean()),
                                         median_axis_sum=float(np.median(values.sum(axis=-1))))
    summary['sigma_image_px_median_xy'] = np.median(arrays['sigma_image_px'], axis=(0, 1)).tolist()
    summary['entropy_norm_median_xy'] = np.median(arrays['entropy_norm'], axis=(0, 1)).tolist()
    with open(summary_path, 'w' if args.overwrite else 'x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'[OK] {len(records)} nguoi -> {npz_path}')
    print(f'[OK] Anh xa SimCC -> decoder: sai khac lon nhat {summary["decode_error_max_px"]:.6f} px')
    print(f'[INFO] Raw stats: {json.dumps(summary["raw_stats"])}')
    print(f'[INFO] Median std pixel X/Y: {summary["sigma_image_px_median_xy"]}')
    print('[LUU Y] Phan bo de ve chua hieu chuan; raw da luu de doi normalization/temperature sau nay.')
    print(f'[OK] Summary: {summary_path} | vis: {vis_dir}')


if __name__ == '__main__':
    main()
