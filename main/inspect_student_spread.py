"""Chan doan gate Student, Teacher GT hoac so scratch/KD, chi suy luan.

Gate: dung YAML train_student_kd_e1_v2_diff_spread_gate.yml va checkpoint cua run.
real dung gate da hoc; off dat gate=1; shuffle doi std giua mau trong cung batch.
Giu pose S1, feature anh va PE2D cua real khi replay, kiem tra output trung wrapper.
--mode gate la mac dinh; --mode spread la alias cu, cung chi ho tro gate.
--mode teacher doc GT 3D, co the doi chieu CSV da luu bang --spread_csv.
--mode teacher_relation ve cosine(LN(token)) truoc/sau fusion, trung binh va bien thien theo mau.
--mode attention so attention joint->image that cua Teacher/Student, overlay crop va thong ke do lech.
--mode compare so scratch/KD (DiffLifter thuan hoac gate) voi Teacher GT.
Metric mesh dung dataset.evaluate (14 khop); chan doan lifter dung H36M-17 bo pelvis.
Nhanh cong embedding spread da duoc go bo; khong nap checkpoint nhanh cu.
"""
import argparse
import csv
import copy
from contextlib import contextmanager
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')


def parse_args():
    p = argparse.ArgumentParser(description='Chan doan gate/Teacher GT hoac so scratch-KD, khong train')
    p.add_argument('--mode', choices=['gate', 'spread', 'teacher', 'teacher_relation', 'attention', 'compare'], default='gate')
    p.add_argument('--cfg', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--gpu', default='0', help='mot GPU, giong cau hinh eval batch=32')
    p.add_argument('--set', default='3dpw', choices=['3dpw', '3dpw-val', '3dpw-oc', '3dpw-pc', '3dpw-crowd'])
    p.add_argument('--batch_size', type=int, default=0, help='0: dung TEST.batch_size trong YAML')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--max_batches', type=int, default=0, help='0: toan tap; >0: chi chan doan tap con')
    p.add_argument('--seed', type=int, default=0, help='chon donor shuffle va mau ngau nhien; lifter giu seed eval=0')
    p.add_argument('--num_vis', type=int, default=4,
                   help='teacher_relation: tong so hinh mau; mode khac: so mau moi nhom; 0: bo hinh mau')
    p.add_argument('--student_cfg', default='', help='teacher_relation/attention: YAML Student KD hien tai')
    p.add_argument('--student_ckpt', default='', help='teacher_relation/attention: checkpoint Student KD hien tai')
    p.add_argument('--relation_cfg', default='', help='teacher_relation: YAML Student KD quan he, tuy chon')
    p.add_argument('--relation_ckpt', default='', help='teacher_relation: checkpoint KD quan he da train, tuy chon')
    p.add_argument('--attn_block', type=int, default=-1, help='attention: index block tu 0; -1 la block cuoi')
    p.add_argument('--attn_joints', nargs='+', default=['L_Wrist', 'R_Wrist', 'L_Elbow', 'R_Elbow', 'L_Ankle', 'R_Ankle'],
                   help='attention: ten khop H36M can ve; all de ve 17 khop, thong ke luon du 17')
    p.add_argument('--relation_samples', '--max_samples', dest='relation_samples', type=int, default=512,
                   help='teacher_relation/attention: toi da N mau ngau nhien khong hoan lai; 0=toan tap')
    p.add_argument('--out_dir', default='', help='mac dinh tao thu muc moi canh checkpoint')
    p.add_argument('--sample_ids', type=int, nargs='+', default=None,
                   help='teacher/teacher_relation: chi eval cac sample nay; compare: them hinh, van eval ca tap')
    p.add_argument('--spread_csv', default='',
                   help='chi cho mode teacher: samples.csv cua spread, xac minh dung anh/ann_id va so metric')
    p.add_argument('--baseline_cfg', default='', help='compare: YAML scratch da train')
    p.add_argument('--baseline_ckpt', default='', help='compare: best.pth.tar cua scratch')
    p.add_argument('--teacher_cfg', default='', help='compare: YAML Teacher GT da train')
    p.add_argument('--teacher_ckpt', default='', help='compare: best.pth.tar cua Teacher GT')
    p.add_argument('--gain_threshold', type=float, default=1.0,
                   help='compare: nguong mo ta tot/xau hon, mm; khong phai kiem dinh thong ke')
    p.add_argument('--expected_epoch', type=int, default=None,
                   help='compare: dung neu epoch checkpoint KD khac gia tri nay')
    args = p.parse_args()
    if ',' in args.gpu or args.batch_size < 0 or args.workers < 0 or args.max_batches < 0 or args.num_vis < 0 or args.relation_samples < 0:
        p.error('Dung mot GPU; cac so luong phai >= 0')
    if args.spread_csv and args.mode != 'teacher':
        p.error('--spread_csv chi dung cho --mode teacher')
    if args.sample_ids is not None and args.mode not in ('teacher', 'teacher_relation', 'attention', 'compare'):
        p.error('--sample_ids chi dung cho --mode teacher/teacher_relation/attention/compare')
    relation_paths = (args.student_cfg, args.student_ckpt, args.relation_cfg, args.relation_ckpt)
    if any(relation_paths) and args.mode not in ('teacher_relation', 'attention'):
        p.error('Cac co student_cfg/ckpt va relation_cfg/ckpt chi dung cho --mode teacher_relation/attention')
    if bool(args.student_cfg) != bool(args.student_ckpt) or bool(args.relation_cfg) != bool(args.relation_ckpt):
        p.error('Moi Student can DU cap --*_cfg va --*_ckpt')
    if args.mode == 'attention' and (not args.student_cfg or args.relation_cfg):
        p.error('attention can Teacher + Student hien tai; dat student_cfg/ckpt va bo relation_cfg/ckpt')
    if args.relation_cfg and not args.student_cfg:
        p.error('So truoc/sau KD quan he can Student KD hien tai qua --student_cfg/--student_ckpt')
    compare_paths = (args.baseline_cfg, args.baseline_ckpt, args.teacher_cfg, args.teacher_ckpt)
    if args.mode == 'compare' and not all(compare_paths):
        p.error('compare can baseline_cfg/baseline_ckpt/teacher_cfg/teacher_ckpt; cfg/ckpt la KD')
    if args.mode != 'compare' and (any(compare_paths) or args.expected_epoch is not None):
        p.error('Cac co baseline/teacher/expected_epoch chi dung cho --mode compare')
    if not 0 < args.gain_threshold < float('inf'):
        p.error('--gain_threshold phai huu han va >0')
    return args


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank_average(values):
    # Gan hang trung binh cho gia tri trung nhau, khong can scipy.
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(x, y):
    x, y = np.asarray(x), np.asarray(y)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float('nan')
    rx, ry = rank_average(x[valid]), rank_average(y[valid])
    if rx.std() == 0 or ry.std() == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def value_at(meta, key, i, default=''):
    if key not in meta:
        return default
    value = meta[key][i]
    if torch.is_tensor(value):
        value = value.item() if value.numel() == 1 else value.tolist()
    return value


class VisualPool:
    # Chi giu toi da 3*num_vis mau, khong luu mesh cua ca dataset trong RAM.
    def __init__(self, count, seed, groups=None):
        self.count = count
        self.rng = np.random.RandomState(seed)
        self.groups = {name: [] for name in (groups or ('improved', 'worsened', 'random'))}
        self.payloads = {}

    def consider(self, sample_id, gain, make_payload, scores=None):
        if not self.count:
            return
        candidates = {'random': float(self.rng.random_sample())}
        if scores is not None:
            candidates.update(scores)
        elif gain > 0:
            candidates['improved'] = float(gain)
        elif gain < 0:
            candidates['worsened'] = float(-gain)
        selected = False
        for group, score in candidates.items():
            items = self.groups[group]
            if len(items) < self.count or score > items[-1][0]:
                items.append((score, sample_id))
                items.sort(reverse=True)
                del items[self.count:]
                selected = True
        if selected:
            self.payloads[sample_id] = make_payload()
        keep = {idx for items in self.groups.values() for _, idx in items}
        self.payloads = {idx: payload for idx, payload in self.payloads.items() if idx in keep}


def load_model(args, device):
    from core.base import _PickleShim, validate_spread_gate_checkpoint
    from models.ARTS import ARTS

    ckpt = torch.load(args.ckpt, map_location='cpu', pickle_module=_PickleShim, weights_only=False)
    if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt:
        raise ValueError('Can checkpoint Teacher/Student co model_state_dict')
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in ckpt['model_state_dict'].items()}
    if args.mode in ('gate', 'spread'):
        if not cfg.MODEL.get('difflift_spread_gate', False):
            raise ValueError('Mode gate can YAML bat difflift_spread_gate; nhanh cong spread cu da bo')
        if not any('.image_gate.net.' in key for key in state):
            raise ValueError('YAML gate nhung checkpoint khong co gate da train')
    if args.mode in ('teacher', 'teacher_relation'):
        # Alias checkpoint cu da duoc ho tro trong core.base; van nap strict sau khi doi ten.
        old_prefix = 'teacher_model.'
        state = {('smpl_model.' + k[len(old_prefix):] if k.startswith(old_prefix) else k): v
                 for k, v in state.items()}
        if 'smpl_model.fusion.joint_proj.weight' not in state:
            raise ValueError('Can checkpoint Teacher GT thuan, khong phai Student hoac T-hint')
    model = ARTS(num_joint=17, embed_dim=cfg.MODEL.hpe_dim).to(device)
    # Chi ho tro checkpoint khop chinh xac YAML, khong loc key sai shape/thieu backbone.
    validate_spread_gate_checkpoint(model, ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, ckpt.get('epoch')


def capture_real(model, image, p2, mask, cam):
    # Compare van ho tro Student DiffLifter thuan, khong tao std gia.
    if not model.diff_spread_gate:
        result = model.forward_student(image, p2, is_train=False, joints_mask=mask, cam_bbox=cam)
        return result, {}
    cache = {}

    def backbone_hook(module, inputs, output):
        cache['features'] = output[0]

    def diffusion_hook(module, inputs, output):
        # Lan goi cuoi cua wrapper la nhom gia thuyet de tinh std cho gate.
        pred = output[2]
        cache['hyp'] = pred - pred[:, :, 0:1]

    hooks = [model.backbone.register_forward_hook(backbone_hook),
             model.diff_lifter.register_forward_hook(diffusion_hook)]
    try:
        result = model.forward_student(image, p2, is_train=False, joints_mask=mask, cam_bbox=cam)
    finally:
        for hook in hooks:
            hook.remove()
    cache['std'] = result['joint_spread_3d']
    cache['gates'] = torch.stack([layer['spread_gate'] for layer in result['feat_layers']])
    if not all(k in cache for k in ('features', 'hyp', 'std', 'gates')):
        raise RuntimeError('Khong bat duoc day du dau vao gate tu forward_student')
    if cache['hyp'].shape[1] != int(cfg.MODEL.difflift_spread_k):
        raise RuntimeError('Nhom gia thuyet bat duoc khong khop K std trong YAML')
    if not torch.allclose(cache['hyp'].std(dim=1), cache['std'], atol=1e-6, rtol=1e-5):
        raise RuntimeError('std bat duoc khong khop gia thuyet; dung de tranh phan tich sai')
    return result, cache


@contextmanager
def override_image_gates(model, value):
    # off cua mode gate la g=1; khoi phuc ngay ca khi replay gap loi.
    gates = [block.image_gate for block in model.smpl_model.fusion.cfcer.blocks if block.image_gate is not None]
    old = [gate.override_value for gate in gates]
    try:
        for gate in gates:
            gate.override_value = value
        yield
    finally:
        for gate, previous in zip(gates, old):
            gate.override_value = previous


def evaluate_mesh(ds, prediction, target, offset):
    # Dung evaluator that, khong thay bang MPJPE 17 khop tu viet.
    outputs = [{'smpl_mesh_cam': prediction[i].copy(), 'smpl_mesh_cam_target': target[i].copy()}
               for i in range(len(prediction))]
    return ds.evaluate(outputs, offset)


def aggregate(rows, indices, condition):
    return {metric: float(np.mean([rows[i][f'{condition}_{metric}'] for i in indices]))
            for metric in METRICS}


def summarize(rows, joints, out_dir, log):
    all_ids = np.arange(len(rows))
    summary = {}
    log('\nMesh: MPJPE 14 khop / PA-MPJPE / MPVPE, mm (thap hon tot hon)')
    for condition in CONDITIONS:
        ids = [i for i in all_ids if np.isfinite(rows[i][f'{condition}_mpjpe'])]
        if not ids:
            log(f'{condition:8s}: khong co mau hop le (batch cuoi chi co 1 mau)')
            continue
        metric = aggregate(rows, ids, condition)
        summary[condition] = {'n': len(ids), **metric}
        log(f'{condition:8s} n={len(ids):6d} | ' + ' | '.join(f'{k}={v:.3f}' for k, v in metric.items()))
    log('\nChenh lech tren CUNG mau (real - bien the): am = spread that tot hon')
    for condition in ('off', 'shuffle'):
        ids = [i for i in all_ids if np.isfinite(rows[i][f'{condition}_mpjpe'])]
        if not ids:
            continue
        for metric in METRICS:
            delta = np.array([rows[i][f'real_{metric}'] - rows[i][f'{condition}_{metric}'] for i in ids])
            log(f'real - {condition}, {metric}: mean={delta.mean():+.3f}, median={np.median(delta):+.3f} mm; '
                f'tot hon >0.1mm={(delta < -0.1).mean()*100:.1f}%, '
                f'xau hon >0.1mm={(delta > 0.1).mean()*100:.1f}%, con lai={(np.abs(delta) <= 0.1).mean()*100:.1f}%')

    # Khong chon nguong theo loi; chia bon nhom bang phan vi spread mau.
    spreads = np.array([r['spread_mm'] for r in rows])
    edges = np.quantile(spreads, [0.25, 0.5, 0.75])
    labels = np.searchsorted(edges, spreads, side='right') + 1
    grouped = []
    log('\nBon nhom spread mau (norm std, trung binh 16 khop bo pelvis):')
    for q in range(1, 5):
        ids = np.flatnonzero(labels == q)
        if not len(ids):
            log(f'Q{q}: rong do spread trung nhau')
            continue
        record = {'quartile': q, 'n': len(ids), 'spread_min_mm': float(spreads[ids].min()),
                  'spread_max_mm': float(spreads[ids].max())}
        for condition in CONDITIONS:
            valid = [i for i in ids if np.isfinite(rows[i][f'{condition}_mpjpe'])]
            record[f'{condition}_n'] = len(valid)
            for metric in METRICS:
                record[f'{condition}_{metric}'] = (aggregate(rows, valid, condition)[metric]
                                                    if valid else float('nan'))
        record['gain_off_minus_real_mpjpe'] = record['off_mpjpe'] - record['real_mpjpe']
        grouped.append(record)
        log(f'Q{q} n={len(ids)} spread={spreads[ids].mean():.2f} mm | '
            f'real={record["real_mpjpe"]:.3f}, off={record["off_mpjpe"]:.3f}, '
            f'gain(off-real)={record["gain_off_minus_real_mpjpe"]:+.3f} mm')

    joint_spread = np.array([r['spread_mm'] for r in joints])
    joint_error = np.array([r['s1_error_mm'] for r in joints])
    corr = spearman(joint_spread, joint_error)
    log(f'\nSpearman(spread, loi pose S=1) tren cac khop KHONG pelvis: {corr:.4f}')
    by_joint = []
    for j in range(1, 17):
        part = [r for r in joints if r['joint'] == j]
        value = spearman([r['spread_mm'] for r in part], [r['s1_error_mm'] for r in part])
        by_joint.append({'joint': j, 'name': JOINT_NAMES[j], 'n': len(part), 'spearman': value,
                         'spread_mm': float(np.mean([r['spread_mm'] for r in part])),
                         's1_error_mm': float(np.mean([r['s1_error_mm'] for r in part]))})
    log('Spearman tung khop: ' + ', '.join(f'{r["name"]}={r["spearman"]:.3f}' for r in by_joint))
    log('Tuong quan gop co the bi anh huong boi loai khop; can doc ca tuong quan tung khop.')
    write_csv(out_dir / 'quartiles.csv', grouped)
    write_csv(out_dir / 'joint_summary.csv', by_joint)
    return {'conditions': summary, 'spread_quartile_edges_mm': edges.tolist(),
            'spearman_spread_s1_error_no_pelvis': corr}, grouped


def plot_overview(rows, joints, grouped, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].hexbin([r['spread_mm'] for r in joints], [r['s1_error_mm'] for r in joints],
                   gridsize=40, bins='log', mincnt=1, cmap='viridis')
    axes[0].set(xlabel='Spread (mm)', ylabel='S=1 joint error (mm)', title='H36M-17, pelvis excluded')
    gains = np.array([r['gain_off_minus_real_mpjpe'] for r in rows])
    axes[1].hist(gains, bins=60, color='steelblue')
    axes[1].axvline(0, color='black', lw=1)
    axes[1].set(xlabel='MPJPE off - real (mm); positive = improvement', ylabel='Samples', title='Same checkpoint')
    axes[2].bar([f'Q{r["quartile"]}' for r in grouped],
                [r['gain_off_minus_real_mpjpe'] for r in grouped], color='steelblue')
    axes[2].axhline(0, color='black', lw=1)
    axes[2].set(xlabel='Sample spread quartile', ylabel='MPJPE off - real (mm)', title='Higher spread: larger benefit?')
    fig.tight_layout()
    fig.savefig(out_dir / 'overview.png', dpi=160)
    plt.close(fig)


def draw_skeleton(ax, pose, horizontal, color, alpha=1.0, lw=1.5):
    for joint, parent in BONES:
        ax.plot(pose[[joint, parent], horizontal], pose[[joint, parent], 1],
                color=color, alpha=alpha, lw=lw)


def set_view(ax, points, horizontal):
    xy = np.concatenate([p[:, [horizontal, 1]] for p in points], axis=0)
    center = (xy.max(0) + xy.min(0)) / 2
    radius = max(float(np.ptp(xy, axis=0).max()) * 0.57, 100.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] + radius, center[1] - radius)
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)' if horizontal == 0 else 'z (mm)')
    ax.set_ylabel('y (mm)')
    ax.tick_params(labelsize=7)


def draw_mesh(ax, vertices, faces, horizontal, color):
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgb
    depth = 2 if horizontal == 0 else 0
    triangles = vertices[faces]
    order = np.argsort(triangles[:, :, depth].mean(1))[::-1]
    polygons = triangles[order][:, :, [horizontal, 1]]
    # To bong theo phap tuyen de thay duoc be mat, khong chi thay duong bao.
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    light = np.zeros(3)
    light[horizontal], light[1], light[depth] = 0.3, -0.5, -1.0
    light /= np.linalg.norm(light)
    brightness = 0.35 + 0.65 * np.abs(normals @ light)
    colors = np.asarray(to_rgb(color))[None] * brightness[order, None]
    ax.add_collection(PolyCollection(polygons, facecolors=colors, edgecolors='none'))


def save_visuals(pool, ds, out_dir, log):
    vis_dir = out_dir / 'vis'
    vis_dir.mkdir()
    faces = np.asarray(ds.smpl.face, dtype=np.int64)
    mesh_regressor = np.asarray(ds.joint_regressor)
    for sample_id, payload in sorted(pool.payloads.items()):
        groups = [name for name, items in pool.groups.items() if any(idx == sample_id for _, idx in items)]
        fig, axes = plt.subplots(3, 3, figsize=(13, 12))
        image, xy, mask = payload['image'], payload['xy'], payload['mask']
        ax = axes[0, 0]
        ax.imshow(np.clip(image.transpose(1, 2, 0), 0, 1))
        pixels = (xy[:, :2] + 1) / 2 * np.array([image.shape[2], image.shape[1]])
        valid = mask.reshape(-1) > 0
        for joint, parent in BONES:
            if valid[joint] and valid[parent]:
                ax.plot(pixels[[joint, parent], 0], pixels[[joint, parent], 1], color='lime', lw=1)
        points = ax.scatter(pixels[valid, 0], pixels[valid, 1], c=payload['std_norm'][valid],
                            cmap='magma', vmin=0, s=30, edgecolors='white', linewidths=0.4)
        for j in np.flatnonzero(valid):
            ax.text(pixels[j, 0], pixels[j, 1], str(j), color='cyan', fontsize=6)
        fig.colorbar(points, ax=ax, label='3D spread (mm)', fraction=0.04)
        ax.set_title(f'Crop + detector 2D; missing={int((~valid).sum())}')
        ax.axis('off')
        pose, gt, hyp = payload['pose'] * 1000, payload['gt_pose'] * 1000, payload['hyp'] * 1000
        for column, horizontal in ((1, 0), (2, 2)):
            ax = axes[0, column]
            for candidate in hyp:
                draw_skeleton(ax, candidate, horizontal, 'gray', alpha=0.1, lw=0.6)
            draw_skeleton(ax, pose, horizontal, 'tab:orange', lw=2)
            draw_skeleton(ax, gt, horizontal, 'black', lw=2)
            set_view(ax, [pose, gt, hyp.reshape(-1, 3)], horizontal)
            ax.set_title('S=1 orange / GT black / K gray: ' + ('front' if horizontal == 0 else 'side'))
        # Mesh can theo SMPL pelvis, dung quy uoc MPVPE cua evaluator; khong PA align.
        meshes = {}
        for name in ('real', 'off', 'gt'):
            vertices = payload['meshes'][name]
            meshes[name] = (vertices - (mesh_regressor @ vertices)[0:1]) * 1000
        for row, horizontal in ((1, 0), (2, 2)):
            for column, (name, color) in enumerate((('real', 'steelblue'), ('off', 'darkorange'), ('gt', 'lightgray'))):
                ax = axes[row, column]
                draw_mesh(ax, meshes[name], faces, horizontal, color)
                set_view(ax, list(meshes.values()), horizontal)
                title = name.upper() + (' front' if horizontal == 0 else ' side')
                if name != 'gt':
                    title += f' | J14={payload["row"][name + "_mpjpe"]:.2f}, V={payload["row"][name + "_mpvpe"]:.2f}'
                ax.set_title(title, fontsize=9)
        fig.suptitle(f'#{sample_id} {", ".join(groups)} | gain off-real={payload["row"]["gain_off_minus_real_mpjpe"]:+.2f} mm\n'
                     f'{payload["row"]["img_path"]}\nOrthographic root-relative views; NOT a camera overlay', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(vis_dir / f'{sample_id:06d}_{"_".join(groups)}.png', dpi=140)
        plt.close(fig)
    log(f'Da luu {len(pool.payloads)} hinh (mau trung nhom chi luu mot lan) vao {vis_dir}')


def read_spread_reference(path):
    if not path:
        return {}
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'sample', 'img_path', 'ann_id', 'real_mpjpe', 'real_pa_mpjpe', 'real_mpvpe'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('spread_csv phai la samples.csv do inspector spread tao')
        rows = {}
        for row in reader:
            idx = int(row['sample'])
            if idx in rows:
                raise ValueError(f'spread_csv co sample trung: {idx}')
            rows[idx] = row
    return rows


def verify_spread_reference(reference, sample_id, img_path, ann_id):
    if sample_id not in reference:
        return None
    row = reference[sample_id]
    # Cho phep doi root dataset tren server, nhung sequence/frame va ann_id phai khop.
    image_key = lambda value: '/'.join(str(value).replace('\\', '/').split('/')[-2:])
    if not row['img_path'] or image_key(row['img_path']) != image_key(img_path):
        raise ValueError(f'Sample {sample_id}: anh khong khop spread_csv; khong duoc so hai mau khac nhau')
    if str(row['ann_id']).strip() != str(ann_id).strip():
        raise ValueError(f'Sample {sample_id}: ann_id khong khop spread_csv')
    return row


def save_teacher_visuals(pool, ds, out_dir, log, selected_only=False):
    vis_dir = out_dir / 'vis'
    vis_dir.mkdir()
    faces = np.asarray(ds.smpl.face, dtype=np.int64)
    mesh_regressor = np.asarray(ds.joint_regressor)
    for sample_id, payload in sorted(pool.payloads.items()):
        row = payload['row']
        groups = [name for name, items in pool.groups.items() if any(idx == sample_id for _, idx in items)]
        if selected_only:
            groups = ['selected subset']
        fig, axes = plt.subplots(3, 3, figsize=(13, 12))
        image = payload['image']
        axes[0, 0].imshow(np.clip(image.transpose(1, 2, 0), 0, 1))
        pixels = (payload['xy'][:, :2] + 1) / 2 * np.array([image.shape[2], image.shape[1]])
        valid = payload['mask'].reshape(-1) > 0
        for j, p in BONES:
            if valid[j] and valid[p]:
                axes[0, 0].plot(pixels[[j, p], 0], pixels[[j, p], 1], color='lime', lw=1)
        axes[0, 0].scatter(pixels[valid, 0], pixels[valid, 1], c='cyan', s=15)
        axes[0, 0].set_title(f'Crop + detector PE2D; missing={row["missing_2d"]}')
        axes[0, 0].axis('off')
        gt_pose, pred_pose = payload['gt_pose'] * 1000, payload['pred_pose'] * 1000
        for column, horizontal in ((1, 0), (2, 2)):
            ax = axes[0, column]
            draw_skeleton(ax, gt_pose, horizontal, 'black', lw=2)
            draw_skeleton(ax, pred_pose, horizontal, 'tab:blue', lw=1.5)
            set_view(ax, [gt_pose, pred_pose], horizontal)
            ax.set_title('GT input black / mesh joints blue: ' + ('front' if horizontal == 0 else 'side'))
        meshes = {}
        for name in ('teacher', 'gt'):
            vertices = payload['meshes'][name]
            meshes[name] = (vertices - (mesh_regressor @ vertices)[0:1]) * 1000
        for line, horizontal in ((1, 0), (2, 2)):
            for column, name in enumerate(('teacher', 'gt')):
                ax = axes[line, column]
                draw_mesh(ax, meshes[name], faces, horizontal, 'steelblue' if name == 'teacher' else 'lightgray')
                set_view(ax, list(meshes.values()), horizontal)
                ax.set_title(name.upper() + (' front' if horizontal == 0 else ' side'))
        joint_error = np.linalg.norm(pred_pose - gt_pose, axis=-1)
        axes[1, 2].barh(np.arange(1, 17), joint_error[1:], color='steelblue')
        axes[1, 2].set_yticks(np.arange(1, 17))
        axes[1, 2].set_yticklabels(JOINT_NAMES[1:], fontsize=7)
        axes[1, 2].set_xlabel('Mesh joint error (mm); H36M-17, no pelvis')
        axes[1, 2].invert_yaxis()
        axes[2, 2].axis('off')
        text = (f'Teacher GT input\nMPJPE(14): {row["teacher_mpjpe"]:.3f} mm\n'
                f'PA-MPJPE: {row["teacher_pa_mpjpe"]:.3f} mm\nMPVPE: {row["teacher_mpvpe"]:.3f} mm\n'
                f'GT input error: {row["teacher_input_error_mm"]:.6g} mm')
        if row['reference_matched']:
            text += (f'\n\nStudent spread (saved CSV)\nMPJPE(14): {row["student_real_mpjpe"]:.3f} mm\n'
                     f'MPVPE: {row["student_real_mpvpe"]:.3f} mm\nDifferent checkpoint/input, not a pure input ablation')
        axes[2, 2].text(0, 0.95, text, va='top', fontsize=10)
        fig.suptitle(f'Teacher GT | #{sample_id} | {", ".join(groups)}\n{row["img_path"]}\n'
                     'Orthographic root-relative views; NOT a camera overlay', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(vis_dir / f'{sample_id:06d}_teacher_gt.png', dpi=140)
        plt.close(fig)
    log(f'Da luu {len(pool.payloads)} hinh Teacher GT vao {vis_dir}')


def inspect_teacher(args):
    from torch.utils.data import DataLoader, Subset
    from utils.jotr_dataset import get_test_dataset
    from tqdm import tqdm
    global plt

    if cfg.MODEL.name != 'teacher' or cfg.MODEL.get('teacher_hint', False):
        raise ValueError('Can YAML Teacher GT thuan (name=teacher, teacher_hint=False), khong dung YAML KD Student')
    if cfg.MODEL.get('teacher_lift_alpha_max', 0.0) != 0:
        raise ValueError('Teacher dang cau hinh tron lifter; hay dung dung YAML Teacher GT thuan da train')
    if not torch.cuda.is_available():
        raise RuntimeError('Chay tren server GPU co CUDA')
    if not Path(args.ckpt).is_file():
        raise FileNotFoundError(args.ckpt)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    batch_size = args.batch_size or int(cfg.TEST.batch_size)
    if batch_size < 1:
        raise ValueError('batch_size phai >=1')
    cfg.DATASET.lift_only = False
    cfg.render, cfg.vis = False, False
    if args.num_vis:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent / f'inspect_teacher_gt_{args.set}_{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f'Thu muc ket qua khong rong: {out_dir}')
    lines = []

    def log(message):
        print(message, flush=True)
        lines.append(message)
        (out_dir / 'inspect_log.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    model, epoch = load_model(args, device)
    ds = get_test_dataset(args.set, None)
    selected = sorted(set(args.sample_ids)) if args.sample_ids is not None else list(range(len(ds)))
    if not selected or selected[0] < 0 or selected[-1] >= len(ds):
        raise ValueError(f'sample_ids phai nam trong [0, {len(ds)-1}] va khong duoc rong')
    reference = read_spread_reference(args.spread_csv)
    if args.sample_ids is not None and args.spread_csv and any(idx not in reference for idx in selected):
        raise ValueError('Co sample duoc chon khong nam trong spread_csv')
    loader = DataLoader(Subset(ds, selected), batch_size=batch_size, shuffle=False, num_workers=args.workers)
    reg = torch.as_tensor(ds.h36m_joint_regressor, device=device, dtype=torch.float32)
    pool = VisualPool(args.num_vis, args.seed, groups=('worst', 'best', 'random'))
    rows, joints = [], []
    seen = 0
    log(f'mode=teacher | cfg={args.cfg}\nckpt={args.ckpt} | epoch={epoch} | set={args.set} | batch_size={batch_size}')
    log(f'Train={list(cfg.DATASET.train_list)} | chon best={cfg.TEST.get("select_dataset", "")}')
    log('Input: GT mesh -> H36M-17 -> tru pelvis -> met; khong lifter, khong corruption luc eval.')
    log(f'PE2D: {"detector Halpe va mask nhu YAML train" if model.joint_pe2d else "tat theo YAML"}; khong thay bang GT 2D.')
    log('GT input la oracle de chan doan, khong phai ket qua trien khai. Teacher dung checkpoint rieng cua no.')
    log(f'sample_ids={args.sample_ids if args.sample_ids is not None else "toan tap"} | spread_csv={args.spread_csv or "khong co"}')
    with torch.no_grad():
        for bi, (inputs, targets, meta) in enumerate(tqdm(loader, desc=f'Teacher GT {args.set}')):
            if args.max_batches and bi >= args.max_batches:
                break
            image = inputs['img'].to(device).float()
            p2 = inputs['joints'].to(device).float()
            mask = inputs['joints_mask'].to(device).float()
            target = targets['smpl_mesh_cam'].float()
            batch = len(image)
            ids = selected[seen:seen + batch]
            matched = [verify_spread_reference(reference, idx, value_at(meta, 'img_path', i),
                                               value_at(meta, 'aid', i)) for i, idx in enumerate(ids)]
            gt = torch.matmul(reg[None], target.to(device))
            gt = gt - gt[:, 0:1]
            result = model(image, gt, is_train=False, pose_2d=p2, joints_mask=mask)
            input_error = torch.norm(result['teacher_input_joints'] - gt, dim=-1).mean(-1).cpu().numpy() * 1000
            if not np.isfinite(input_error).all() or input_error.max() > 1e-3:
                raise RuntimeError('Teacher khong nhan dung GT 3D; dung chan doan')
            prediction = result['smpl_mesh_cam'].cpu().numpy()
            target_np = target.numpy()
            if not np.isfinite(prediction).all():
                raise RuntimeError('Teacher xuat NaN/Inf')
            # Subset khong lien tiep: evaluate tung mau voi dung index dataset goc.
            metrics = {name: [] for name in METRICS}
            if ids == list(range(ids[0], ids[0] + batch)):
                metrics = evaluate_mesh(ds, prediction, target_np, ids[0])
            else:
                for i, idx in enumerate(ids):
                    one = evaluate_mesh(ds, prediction[i:i+1], target_np[i:i+1], idx)
                    for metric in METRICS:
                        metrics[metric].extend(one[metric])
            if any(not np.isfinite(metrics[m]).all() for m in METRICS):
                raise RuntimeError('Metric Teacher co NaN/Inf')
            pred_pose = torch.matmul(reg[None], result['smpl_mesh_cam'])
            pred_pose = pred_pose - pred_pose[:, 0:1]
            error = torch.norm(pred_pose - gt, dim=-1).cpu().numpy() * 1000
            gt_np, pred_np, mask_np = gt.cpu().numpy(), pred_pose.cpu().numpy(), mask.cpu().numpy().reshape(batch, 17)
            for i, sample_id in enumerate(ids):
                row = {'sample': sample_id, 'img_path': value_at(meta, 'img_path', i),
                       'ann_id': value_at(meta, 'aid', i), 'missing_2d': int((mask_np[i] <= 0).sum()),
                       'teacher_input_error_mm': float(input_error[i]), 'reference_matched': matched[i] is not None}
                for metric in METRICS:
                    row[f'teacher_{metric}'] = float(metrics[metric][i])
                    for name in ('real', 'off', 'shuffle'):
                        key = f'{name}_{metric}'
                        row[f'student_{key}'] = float(matched[i][key]) if matched[i] is not None and key in matched[i] else float('nan')
                rows.append(row)
                if args.sample_ids is not None:
                    extra = f' | Student real J14={row["student_real_mpjpe"]:.3f}' if row['reference_matched'] else ''
                    log(f'#{sample_id} ann_id={row["ann_id"]} | Teacher J14={row["teacher_mpjpe"]:.3f} '
                        f'PA={row["teacher_pa_mpjpe"]:.3f} V={row["teacher_mpvpe"]:.3f} mm | missing2D={row["missing_2d"]}'
                        f'{extra}\n  {row["img_path"]}')
                for j in range(1, 17):
                    joints.append({'sample': sample_id, 'joint': j, 'name': JOINT_NAMES[j],
                                   'teacher_mesh_joint_error_mm': float(error[i, j]), 'valid_2d': int(mask_np[i, j] > 0)})

                def make_payload():
                    return {'row': row.copy(), 'image': image[i].cpu().numpy().copy(),
                            'xy': p2[i].cpu().numpy().copy(), 'mask': mask_np[i].copy(),
                            'gt_pose': gt_np[i].copy(), 'pred_pose': pred_np[i].copy(),
                            'meshes': {'teacher': prediction[i].copy(), 'gt': target_np[i].copy()}}

                pool.consider(sample_id, 0.0, make_payload,
                              scores={'worst': row['teacher_mpjpe'], 'best': -row['teacher_mpjpe']})
            seen += batch
    if not rows:
        raise RuntimeError('Khong co mau de chan doan')
    write_csv(out_dir / 'samples.csv', rows)
    write_csv(out_dir / 'joints.csv', joints)
    summary = {'mode': 'teacher_gt', 'args': vars(args), 'epoch': epoch, 'n': len(rows),
               'dataset_n': len(ds), 'subset': len(rows) < len(ds),
               'metrics': {metric: float(np.mean([r[f'teacher_{metric}'] for r in rows])) for metric in METRICS},
               'max_gt_input_error_mm': float(max(r['teacher_input_error_mm'] for r in rows)),
               'reference_matched_n': sum(r['reference_matched'] for r in rows),
               'train_list': list(cfg.DATASET.train_list), 'select_dataset': cfg.TEST.get('select_dataset', ''),
               'pe2d_source': 'detector' if model.joint_pe2d else 'disabled'}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    log(f'\nTeacher GT | n={len(rows)}/{len(ds)} | {"TAP CON" if summary["subset"] else "TOAN TAP"}')
    log('Mesh MPJPE 14 khop / PA-MPJPE / MPVPE, mm: ' + ' | '.join(f'{m}={v:.3f}' for m, v in summary['metrics'].items()))
    log(f'GT input max error={summary["max_gt_input_error_mm"]:.6g} mm; CSV matched={summary["reference_matched_n"]}')
    if args.num_vis:
        save_teacher_visuals(pool, ds, out_dir, log, selected_only=args.sample_ids is not None)
    log('Joints.csv/pose tren hinh dung H36M-17 bo pelvis; metric mesh tong ket dung evaluator 14 khop.')
    log('Teacher co GT ma van sai -> xem fusion/head/GT17 sang SMPL va anh/PE2D; khong ket luan rieng do anh khi chua ablation.')
    log('Teacher dung GT tot hon Student khong tu chung minh KD hieu qua; can log Student KD de ket luan.')
    log(f'Ket qua: {out_dir}')


def read_compare_config(path, defaults):
    import yaml
    from core.config import _update_dict

    snapshot = copy.deepcopy(defaults)
    with open(path, encoding='utf-8-sig') as f:
        values = yaml.safe_load(f)
    if not isinstance(values, dict):
        raise ValueError(f'YAML khong hop le: {path}')
    # Moi YAML bat dau tu default, khong ke thua key cua model vua nap.
    _update_dict(snapshot, values)
    snapshot.DATASET.lift_only = False
    snapshot.render = False
    snapshot.vis = False
    return snapshot


@contextmanager
def compare_config(snapshot):
    # ARTS va dataset doc cfg toan cuc ca khi forward; phai khoi phuc sau moi luot.
    previous = dict(cfg)
    cfg.clear()
    cfg.update(copy.deepcopy(snapshot))
    try:
        yield
    finally:
        cfg.clear()
        cfg.update(previous)


def summarize_compare(rows, args, out_dir, log):
    n = len(rows)
    errors = np.array([row['scratch_mpjpe'] for row in rows])
    order = np.argsort(errors, kind='mergesort')
    all_ids = np.arange(n)
    groups = {'all': all_ids}
    for q, ids in enumerate(np.array_split(order, 4), 1):
        groups[f'Q{q}_scratch_error'] = ids
        for i in ids:
            rows[i]['scratch_error_quartile'] = q
    hard = order[-max(1, int(np.ceil(n * 0.1))):]
    hard_set = set(hard.tolist())
    groups['hardest_10pct_scratch'] = hard
    missing = np.array([row['missing_2d'] for row in rows])
    for name, condition in (('missing_0', missing == 0), ('missing_1_4', (missing >= 1) & (missing <= 4)),
                            ('missing_5_8', (missing >= 5) & (missing <= 8)), ('missing_9_17', missing >= 9)):
        groups[name] = all_ids[condition]
    opportunity = np.array([row['scratch_mpjpe'] - row['teacher_mpjpe'] > args.gain_threshold for row in rows])
    groups['teacher_better_than_scratch'] = all_ids[opportunity]
    groups['hard10_teacher_better'] = hard[opportunity[hard]]
    groups['teacher_not_better_than_scratch'] = all_ids[~opportunity]
    for i, row in enumerate(rows):
        row['hardest_10pct_scratch'] = int(i in hard_set)
        row['teacher_better_than_scratch'] = int(opportunity[i])
    records = []
    for name, ids in groups.items():
        if not len(ids):
            continue
        record = {'group': name, 'n': len(ids)}
        for metric in METRICS:
            for model in ('scratch', 'kd', 'teacher'):
                record[f'{model}_{metric}'] = float(np.mean([rows[i][f'{model}_{metric}'] for i in ids]))
            gains = np.array([rows[i][f'gain_{metric}'] for i in ids])
            record[f'gain_{metric}_mean'] = float(gains.mean())
            record[f'gain_{metric}_median'] = float(np.median(gains))
            record[f'gain_{metric}_p10'] = float(np.percentile(gains, 10))
            record[f'gain_{metric}_p90'] = float(np.percentile(gains, 90))
            record[f'improved_{metric}_pct'] = float(100 * np.mean(gains > args.gain_threshold))
            record[f'worsened_{metric}_pct'] = float(100 * np.mean(gains < -args.gain_threshold))
            record[f'within_threshold_{metric}_pct'] = float(100 * np.mean(np.abs(gains) <= args.gain_threshold))
        records.append(record)
        log(f'{name}: n={len(ids)} | MPJPE scratch={record["scratch_mpjpe"]:.3f} '
            f'KD={record["kd_mpjpe"]:.3f} TeacherGT={record["teacher_mpjpe"]:.3f} | '
            f'gain={record["gain_mpjpe_mean"]:+.3f}, median={record["gain_mpjpe_median"]:+.3f} mm | '
            f'tot/xau >{args.gain_threshold:g}mm={record["improved_mpjpe_pct"]:.1f}%/'
            f'{record["worsened_mpjpe_pct"]:.1f}%')
    log('\nCac mau da eval: MPJPE14 / PA-MPJPE / MPVPE (mm)')
    for model in ('scratch', 'kd', 'teacher'):
        values = [records[0][f'{model}_{metric}'] for metric in METRICS]
        log(f'{model:8s}: ' + ' / '.join(f'{value:.3f}' for value in values))
    write_csv(out_dir / 'groups.csv', records)
    return records, hard


def select_compare_visuals(rows, hard, args, log):
    if not args.num_vis:
        return {}
    selected = {}

    def add(idx, label):
        selected.setdefault(int(idx), []).append(label)

    fixed = args.sample_ids if args.sample_ids is not None else ([2749, 8785, 17500] if args.set == '3dpw' else [])
    for idx in sorted(set(fixed)):
        if 0 <= idx < len(rows):
            add(idx, 'fixed')
        else:
            log(f'Khong ve fixed sample {idx}: ngoai pham vi da eval ({len(rows)} mau)')
    gains = np.array([row['gain_mpjpe'] for row in rows])
    for idx in np.argsort(-gains, kind='mergesort')[:args.num_vis]:
        if gains[idx] > args.gain_threshold:
            add(idx, 'improved')
    for idx in np.argsort(gains, kind='mergesort')[:args.num_vis]:
        if gains[idx] < -args.gain_threshold:
            add(idx, 'worsened')
    rng = np.random.RandomState(args.seed)
    for idx in rng.choice(hard, size=min(args.num_vis, len(hard)), replace=False):
        add(idx, 'hard_random')
    return selected


def save_compare_visual(payload, ds, out_dir):
    row, meshes, labels = payload['row'], payload['meshes'], payload['groups']
    faces = np.asarray(ds.smpl.face, dtype=np.int64)
    reg = np.asarray(ds.joint_regressor)
    h36m_reg = np.asarray(ds.h36m_joint_regressor)
    colors = {'gt': 'lightgray', 'scratch': 'darkorange', 'kd': 'steelblue', 'teacher': 'seagreen'}
    names = ('gt', 'scratch', 'kd', 'teacher')
    centered, poses = {}, {}
    for name in names:
        vertices = meshes[name]
        centered[name] = (vertices - (reg @ vertices)[0:1]) * 1000
        joints = h36m_reg @ vertices
        poses[name] = (joints - joints[0:1]) * 1000
    fig, axes = plt.subplots(3, 4, figsize=(17, 12))
    image, xy = payload['image'], payload['xy']
    valid = payload['mask'].reshape(-1) > 0
    axes[0, 0].imshow(np.clip(image.transpose(1, 2, 0), 0, 1))
    pixels = (xy[:, :2] + 1) / 2 * np.array([image.shape[2], image.shape[1]])
    for joint, parent in BONES:
        if valid[joint] and valid[parent]:
            axes[0, 0].plot(pixels[[joint, parent], 0], pixels[[joint, parent], 1], color='lime', lw=1)
    axes[0, 0].scatter(pixels[valid, 0], pixels[valid, 1], color='cyan', s=20)
    axes[0, 0].set_title(f'Crop + detector; missing={row["missing_2d"]}/17')
    axes[0, 0].axis('off')
    for col, horizontal in ((1, 0), (2, 2)):
        for name in names:
            draw_skeleton(axes[0, col], poses[name], horizontal, 'black' if name == 'gt' else colors[name])
        set_view(axes[0, col], list(poses.values()), horizontal)
        axes[0, col].set_title('H36M17: GT black / scratch orange / KD blue / T green', fontsize=8)
    axes[0, 3].axis('off')
    text = ['MPJPE14 / PA / MPVPE (mm)']
    for name in names[1:]:
        text.append(f'{name}: ' + ' / '.join(f'{row[name + "_" + metric]:.2f}' for metric in METRICS))
    text.extend([f'Gain scratch-KD: {row["gain_mpjpe"]:+.2f} mm',
                 f'Scratch quartile: Q{row["scratch_error_quartile"]}',
                 f'Hardest 10%: {bool(row["hardest_10pct_scratch"])}', 'Teacher uses GT 3D (oracle)'])
    axes[0, 3].text(0, 1, '\n'.join(text), va='top', fontsize=9)
    for r, horizontal in ((1, 0), (2, 2)):
        for c, name in enumerate(names):
            draw_mesh(axes[r, c], centered[name], faces, horizontal, colors[name])
            set_view(axes[r, c], list(centered.values()), horizontal)
            axes[r, c].set_title(name.upper() + (' front' if horizontal == 0 else ' side'))
    fig.suptitle(f'Sample #{row["sample"]} | {", ".join(labels)} | ann_id={row["ann_id"]}\n'
                 f'{row["img_path"]}\nOrthographic root-relative; not a camera overlay', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / 'vis' / f'{row["sample"]:06d}_{"_".join(labels)}.png', dpi=140)
    plt.close(fig)


def plot_compare_overview(rows, groups, out_dir):
    gains = np.array([row['gain_mpjpe'] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(gains, bins=80)
    axes[0].axvline(0, color='black', lw=1)
    axes[0].set_xlabel('MPJPE gain scratch - KD (mm)')
    axes[0].set_ylabel('Samples')
    quartiles = [record for record in groups if record['group'].startswith('Q')]
    x = np.arange(len(quartiles))
    for model, shift, color in (('scratch', -0.25, 'darkorange'), ('kd', 0, 'steelblue'), ('teacher', 0.25, 'seagreen')):
        axes[1].bar(x + shift, [record[f'{model}_mpjpe'] for record in quartiles], width=0.25, label=model, color=color)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([record['group'].split('_')[0] for record in quartiles])
    axes[1].set_xlabel('Groups fixed by scratch error')
    axes[1].set_ylabel('MPJPE14 (mm)')
    axes[1].legend()
    axes[2].hexbin([row['scratch_mpjpe'] for row in rows], gains, gridsize=45, mincnt=1, bins='log')
    axes[2].axhline(0, color='black', lw=1)
    axes[2].set_xlabel('Scratch MPJPE14 (mm)')
    axes[2].set_ylabel('MPJPE gain scratch - KD (mm)')
    fig.tight_layout()
    fig.savefig(out_dir / 'overview.png', dpi=150)
    plt.close(fig)


def inspect_compare(args, defaults):
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from utils.jotr_dataset import get_test_dataset
    from tqdm import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError('Chay tren server GPU co CUDA')
    paths = {'scratch': (args.baseline_cfg, args.baseline_ckpt), 'kd': (args.cfg, args.ckpt),
             'teacher': (args.teacher_cfg, args.teacher_ckpt)}
    for config_path, checkpoint_path in paths.values():
        for path in (config_path, checkpoint_path):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    configs = {name: read_compare_config(config_path, defaults) for name, (config_path, _) in paths.items()}
    for name in ('scratch', 'kd'):
        model_cfg = configs[name].MODEL
        if (model_cfg.name != 'student' or model_cfg.pose_lifter != 'diffusion'
                or model_cfg.difflift_steps != 1 or model_cfg.difflift_k != 1
                or model_cfg.difflift_select or model_cfg.student_hint
                or not model_cfg.student_3d_rootrel or model_cfg.REFINER != 'hypergcn'):
            raise ValueError(f'{name}: can Student DiffLifter S1/K1, root-relative, hypergcn, khong selector/hint')
        if model_cfg.difflift_spread_gate and model_cfg.difflift_spread_k < 2:
            raise ValueError(f'{name}: gate can K>=2 de tinh std')
        if configs[name].DATASET.seqlen != 1 or configs[name].DATASET.use_gt_input:
            raise ValueError(f'{name}: can image-based seqlen=1, detector input')
    if any(configs[name].DATASET.input_joint_set != 'human36' or configs[name].DATASET.target_joint_set != 'human36'
           for name in configs):
        raise ValueError('Can joint set H36M-17 cho ca ba YAML')
    if configs['scratch'].MODEL.student_use_kd or not configs['kd'].MODEL.student_use_kd:
        raise ValueError('baseline_cfg phai la scratch khong KD; cfg phai la Student KD')
    teacher_cfg = configs['teacher'].MODEL
    if teacher_cfg.name != 'teacher' or teacher_cfg.teacher_hint or teacher_cfg.teacher_lift_alpha_max != 0:
        raise ValueError('teacher_cfg phai la Teacher GT thuan, khong hint/tron lifter')
    if teacher_cfg.REFINER != 'hypergcn':
        raise ValueError('Chi ho tro Teacher GT hypergcn')
    # Cung mot loader cho hai Student; tu choi neu YAML doi quy uoc dau vao.
    ignored = {'train_list', 'test_list', 'workers'}
    for key, value in configs['scratch'].DATASET.items():
        if key not in ignored and value != configs['kd'].DATASET[key]:
            raise ValueError(f'YAML Student khac DATASET.{key}; khong duoc so nhu cung dau vao')
    for key in ('input_img_shape', 'output_hm_shape', 'focal', 'princpt', 'bbox_3d_size', 'camera_3d_size'):
        if key in defaults and any(configs[name][key] != configs['scratch'][key] for name in ('kd', 'teacher')):
            raise ValueError(f'YAML khac {key}; can cung crop/camera/evaluator')
    for key in ('pw3d_2d_source', 'halpe26_conf_thr', 'input_joint_set', 'target_joint_set'):
        if configs['teacher'].DATASET[key] != configs['scratch'].DATASET[key]:
            raise ValueError(f'Teacher khac DATASET.{key}; can cung detector/PE2D')
    compare_std = all(configs[name].MODEL.difflift_spread_gate for name in ('scratch', 'kd'))
    shared_keys = ['difflift_steps', 'difflift_k']
    if compare_std:
        shared_keys += ['difflift_spread_steps', 'difflift_spread_k']
    for key in shared_keys:
        if configs['scratch'].MODEL[key] != configs['kd'].MODEL[key]:
            raise ValueError(f'Hai Student khac MODEL.{key}')
    batch_size = args.batch_size or int(configs['kd'].TEST.batch_size)
    if batch_size < 1:
        raise ValueError('batch_size phai >0')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent / f'compare_scratch_kd_{args.set}_{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f'Thu muc khong rong: {out_dir}; chon thu muc moi')
    lines = []

    def log(message):
        print(message, flush=True)
        lines.append(message)
        (out_dir / 'compare_log.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    models, epochs = {}, {}
    for name, (config_path, checkpoint_path) in paths.items():
        mode = 'teacher' if name == 'teacher' else 'compare'
        with compare_config(configs[name]):
            models[name], epochs[name] = load_model(SimpleNamespace(mode=mode, ckpt=checkpoint_path), device)
        (out_dir / f'{name}_config.yml').write_text(Path(config_path).read_text(encoding='utf-8-sig'), encoding='utf-8')
        log(f'{name}: cfg={config_path} | ckpt={checkpoint_path} | epoch={epochs[name]} | '
            f'train={list(configs[name].DATASET.train_list)} | select={configs[name].TEST.select_dataset}')
    if args.expected_epoch is not None and epochs['kd'] != args.expected_epoch:
        raise ValueError(f'KD checkpoint epoch={epochs["kd"]}, khong phai {args.expected_epoch}; kiem tra best.pth.tar')
    log(f'Set={args.set} | batch_size={batch_size} | mot GPU | thu tu dataset co dinh | lifter seed=0')
    log(f'Gain = scratch - KD, duong la KD tot hon; nguong mo ta {args.gain_threshold:g} mm.')
    log('Teacher GT la oracle. Day la so hai run, con khac khoi tao/hard loss/lich LR; khong tach rieng tac dung loss KD.')
    log('Khong chon checkpoint moi. Ket qua test da dung chon best chi la phan tich phat trien.')
    log('Q1-Q4 va top10% xep hang theo MPJPE scratch; neu bang loi, giu thu tu sample. Khong phai nhom spread.')
    rows = []
    input_parity = {'pose_max_abs_m': 0.0, 'std_max_abs_m': 0.0 if compare_std else None,
                    'std_compared': compare_std, 'teacher_gt_max_error_mm': 0.0}
    std_source = next((name for name in ('scratch', 'kd') if configs[name].MODEL.difflift_spread_gate), None)
    log(f'Kiem tra std giua hai Student: {compare_std}; nguon spread_mm trong CSV: {std_source}')
    # Giu cfg scratch khi dataset/worker chay; moi forward duoc boc cfg rieng.
    with compare_config(configs['scratch']):
        ds = get_test_dataset(args.set, None)
        if not len(ds):
            raise ValueError('Dataset rong')
        if args.sample_ids is not None and any(i < 0 or i >= len(ds) for i in args.sample_ids):
            raise ValueError('--sample_ids chua chi so ngoai dataset')
        reg = torch.as_tensor(ds.h36m_joint_regressor, device=device, dtype=torch.float32)

        def run_batch(inputs, targets, offset):
            image = inputs['img'].to(device).float()
            p2 = inputs['joints'].to(device).float()
            mask = inputs['joints_mask'].to(device).float()
            cam = inputs['cam_bbox'].to(device).float()
            target = targets['smpl_mesh_cam'].float()
            gt = torch.matmul(reg[None], target.to(device))
            gt = gt - gt[:, 0:1]
            predictions, metrics, student_inputs = {}, {}, {}
            with torch.no_grad():
                for name in ('scratch', 'kd', 'teacher'):
                    with compare_config(configs[name]):
                        if name == 'teacher':
                            result = models[name](image, gt, is_train=False, pose_2d=p2, joints_mask=mask)
                            gt_error = float(torch.norm(result['teacher_input_joints'] - gt, dim=-1).max()) * 1000
                            input_parity['teacher_gt_max_error_mm'] = max(input_parity['teacher_gt_max_error_mm'], gt_error)
                            if not np.isfinite(gt_error) or gt_error > 0.001:
                                raise RuntimeError('Teacher khong nhan dung GT 3D')
                        else:
                            result, cache = capture_real(models[name], image, p2, mask, cam)
                            student_inputs[name] = (result['lifted_joints_3d'], cache.get('std'))
                            del cache
                        predictions[name] = result['smpl_mesh_cam'].cpu().numpy()
                        del result
                    metrics[name] = evaluate_mesh(ds, predictions[name], target.numpy(), offset)
                    if (not np.isfinite(predictions[name]).all()
                            or any(not np.isfinite(metrics[name][metric]).all() for metric in METRICS)):
                        raise RuntimeError(f'NaN/Inf: {name}, offset={offset}')
                parity_keys = [(0, 'pose_max_abs_m')]
                if compare_std:
                    parity_keys.append((1, 'std_max_abs_m'))
                for j, key in parity_keys:
                    delta = float((student_inputs['scratch'][j] - student_inputs['kd'][j]).abs().max())
                    input_parity[key] = max(input_parity[key], delta)
                    if not np.isfinite(delta) or delta > 1e-6:
                        raise RuntimeError(f'Hai Student khac dau vao {key}={delta}; kiem tra lifter/YAML/batch')
                pose = student_inputs['scratch'][0]
                lift_error = torch.norm(pose - gt, dim=-1)[:, 1:].mean(-1).cpu().numpy() * 1000
                spread = None
                if std_source is not None:
                    std = student_inputs[std_source][1]
                    spread = std[:, 1:].norm(dim=-1).mean(-1).cpu().numpy() * 1000
            return predictions, metrics, target.numpy(), lift_error, spread

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=args.workers,
                            generator=torch.Generator().manual_seed(args.seed))
        offset = 0
        for bi, (inputs, targets, meta) in enumerate(tqdm(loader, desc=f'compare {args.set}')):
            if args.max_batches and bi >= args.max_batches:
                break
            predictions, metrics, target, lift_error, spread = run_batch(inputs, targets, offset)
            batch = len(target)
            mask = inputs['joints_mask'].numpy().reshape(batch, 17)
            for i in range(batch):
                row = {'sample': offset + i, 'img_path': value_at(meta, 'img_path', i),
                       'ann_id': value_at(meta, 'aid', i), 'dataset_idx': value_at(meta, 'idx', i),
                       'missing_2d': int(np.sum(mask[i] <= 0)), 's1_mpjpe16_mm': float(lift_error[i])}
                if spread is not None:
                    row.update({'spread_mm': float(spread[i]), 'spread_source': std_source})
                if not row['img_path'] or str(row['ann_id']).strip() == '':
                    raise RuntimeError('Dataset thieu img_path/ann_id de xac minh mau')
                for name in ('scratch', 'kd', 'teacher'):
                    for metric in METRICS:
                        row[f'{name}_{metric}'] = float(metrics[name][metric][i])
                for metric in METRICS:
                    row[f'gain_{metric}'] = row[f'scratch_{metric}'] - row[f'kd_{metric}']
                rows.append(row)
            offset += batch
            del predictions, metrics, target
        if not rows:
            raise ValueError('Khong co mau duoc eval')
        log(f'\nDa eval {len(rows)}/{len(ds)} mau; '
            + ('TOAN TAP' if len(rows) == len(ds) else 'TAP CON, khong coi la ket qua toan tap'))
        log(f'Kiem tra dau vao Student (std chi so khi ca hai dung gate): {input_parity}')
        groups, hard = summarize_compare(rows, args, out_dir, log)
        selected = select_compare_visuals(rows, hard, args, log)
        for row in rows:
            row['visual_groups'] = '|'.join(selected.get(row['sample'], []))
        write_csv(out_dir / 'samples.csv', rows)
        summary = {'args': vars(args), 'epochs': epochs, 'n': len(rows), 'dataset_n': len(ds),
                   'full_dataset': len(rows) == len(ds), 'input_parity': input_parity,
                   'gain_definition': 'scratch - kd, mm; positive = KD better',
                   'hard_group_definition': 'rank scratch MPJPE; top ceil(0.1*N); stable sample order for ties',
                   'groups': groups, 'visual_samples': selected,
                   'configs': {name: json.loads(json.dumps(config, default=str)) for name, config in configs.items()}}
        summary['visuals_complete'] = not bool(args.num_vis)
        summary['visual_replay_max_mm'] = 0.0
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        if args.num_vis:
            global plt
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plot_compare_overview(rows, groups, out_dir)
            (out_dir / 'vis').mkdir()
            # Nap LAI nguyen batch goc cho mau can ve; khong eval mau le lam doi nhieu DDIM.
            batch_ids = sorted({idx // batch_size for idx in selected})
            batches = [list(range(bi * batch_size, min((bi + 1) * batch_size, len(ds)))) for bi in batch_ids]
            vis_loader = DataLoader(ds, batch_sampler=batches, num_workers=args.workers,
                                    generator=torch.Generator().manual_seed(args.seed))
            for bi, (inputs, targets, meta) in zip(batch_ids, tqdm(vis_loader, desc='compare visuals')):
                batch_offset = bi * batch_size
                predictions, metrics, target, _, _ = run_batch(inputs, targets, batch_offset)
                for i in range(len(target)):
                    idx = batch_offset + i
                    if idx not in selected:
                        continue
                    verify_spread_reference({idx: rows[idx]}, idx, value_at(meta, 'img_path', i), value_at(meta, 'aid', i))
                    for name in ('scratch', 'kd', 'teacher'):
                        for metric in METRICS:
                            replay_delta = abs(float(metrics[name][metric][i]) - rows[idx][f'{name}_{metric}'])
                            summary['visual_replay_max_mm'] = max(summary['visual_replay_max_mm'], replay_delta)
                            if replay_delta > 0.01:
                                raise RuntimeError(f'Visual replay khac luot eval: sample={idx}, {name}/{metric}')
                    payload = {'row': rows[idx], 'groups': selected[idx], 'image': inputs['img'][i].numpy(),
                               'xy': inputs['joints'][i].numpy(), 'mask': inputs['joints_mask'][i].numpy(),
                               'meshes': {name: predictions[name][i] for name in predictions}}
                    payload['meshes']['gt'] = target[i]
                    save_compare_visual(payload, ds, out_dir)
            log(f'Da ve {len(selected)} mau; replay metric lech toi da cho phep 0.01mm; xem vis/ va overview.png')
    summary['visuals_complete'] = True
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    log('Cac nhom kho duoc chon bang loi scratch tren tap dang phan tich; khong dung de tinh chinh tren test.')
    log('Gain trung binh cua mot run khong chung minh y nghia thong ke; cac frame cung video co tuong quan.')
    log(f'Ket qua: {out_dir}')


def inspect_relation_compare(args, defaults):
    """So Teacher, KD hien tai va KD quan he (neu co) tren cung mau/crop."""
    from types import SimpleNamespace
    from torch.utils.data import DataLoader, Subset
    from utils.jotr_dataset import get_test_dataset
    from utils.joint_relation import joint_relation_matrix
    from tqdm import tqdm
    import matplotlib
    matplotlib.use('Agg')
    global plt
    import matplotlib.pyplot as plt

    attention_mode = args.mode == 'attention'
    if attention_mode:
        from utils.attention_diagnostics import extract_joint_image_attention, save_attention_diagnostics
        bad_joints = set(args.attn_joints) - set(JOINT_NAMES) - {'all'}
        if bad_joints or ('all' in args.attn_joints and len(args.attn_joints) != 1):
            raise ValueError(f'attn_joints khong hop le: {args.attn_joints}')
    if not torch.cuda.is_available():
        raise RuntimeError('Chay tren server GPU co CUDA')
    paths = {'teacher': (args.cfg, args.ckpt), 'kd_current': (args.student_cfg, args.student_ckpt)}
    if args.relation_cfg:
        paths['kd_relation'] = (args.relation_cfg, args.relation_ckpt)
    for pair in paths.values():
        for path in pair:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    configs = {name: read_compare_config(pair[0], defaults) for name, pair in paths.items()}
    teacher_cfg = configs['teacher'].MODEL
    if teacher_cfg.name != 'teacher' or teacher_cfg.teacher_hint or teacher_cfg.teacher_lift_alpha_max != 0:
        raise ValueError('--cfg phai la Teacher GT thuan, khong hint/tron lifter')
    student_names = [name for name in paths if name != 'teacher']
    for name, snapshot in configs.items():
        if snapshot.DATASET.seqlen != 1 or snapshot.DATASET.input_joint_set != 'human36' or snapshot.DATASET.target_joint_set != 'human36':
            raise ValueError(f'{name}: can seqlen1 va joint H36M17')
        for key in ('input_img_shape', 'output_hm_shape', 'focal', 'princpt', 'bbox_3d_size', 'camera_3d_size'):
            if key in defaults and snapshot[key] != configs['kd_current'][key]:
                raise ValueError(f'{name}: khac {key}; can cung crop/camera')
        for key in ('pw3d_2d_source', 'halpe26_conf_thr', 'use_gt_input'):
            if snapshot.DATASET[key] != configs['kd_current'].DATASET[key]:
                raise ValueError(f'{name}: khac DATASET.{key}; can cung detector/PE2D')
        if name == 'teacher':
            continue
        mc = snapshot.MODEL
        if (mc.name != 'student' or mc.pose_lifter != 'diffusion' or mc.difflift_steps != 1 or mc.difflift_k != 1
                or mc.difflift_select or mc.student_hint or not mc.student_3d_rootrel
                or not mc.student_use_kd or mc.kd_mode != 'feat' or mc.kd_lambda <= 0):
            raise ValueError(f'{name}: can Student KD feat DiffLifter S1K1 root-relative, khong selector/hint')
        if bool(mc.kd_joint_relation) != (name == 'kd_relation'):
            raise ValueError(f'{name}: co kd_joint_relation khong khop vai tro checkpoint/YAML')
    # So khac cau hinh train de tranh quy moi chenh lech cho mot loss.
    train_differences = {}
    if 'kd_relation' in paths:
        for section in ('DATASET', 'MODEL', 'TRAIN'):
            for key, old in configs['kd_current'][section].items():
                if section == 'MODEL' and key in ('kd_joint_relation', 'kd_joint_relation_log_grad'):
                    continue
                new = configs['kd_relation'][section][key]
                if old != new:
                    train_differences[f'{section}.{key}'] = {'current': old, 'relation': new}
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.RandomState(args.seed)
    device = torch.device('cuda')
    batch_size = args.batch_size or int(configs['kd_current'].TEST.batch_size)
    if batch_size < 1:
        raise ValueError('batch_size phai >=1')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_prefix = 'compare_attention' if attention_mode else 'compare_relations'
    out_dir = (Path(args.out_dir) if args.out_dir else
               Path(args.student_ckpt).parent / f'{output_prefix}_{args.set}_{stamp}')
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f'Thu muc ket qua khong rong: {out_dir}')
    if not attention_mode:
        (out_dir / 'vis').mkdir()
    lines = []

    def log(message):
        print(message, flush=True)
        lines.append(message)
        (out_dir / 'inspect_log.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    models, epochs = {}, {}
    for name, (config_path, checkpoint_path) in paths.items():
        with compare_config(configs[name]):
            mode = 'teacher' if name == 'teacher' else 'compare'
            models[name], epochs[name] = load_model(SimpleNamespace(mode=mode, ckpt=checkpoint_path), device)
        (out_dir / f'{name}_config.yml').write_text(Path(config_path).read_text(encoding='utf-8-sig'), encoding='utf-8')
        log(f'{name}: cfg={config_path} | ckpt={checkpoint_path} | epoch={epochs[name]} | '
            f'train={list(configs[name].DATASET.train_list)} | select={configs[name].TEST.select_dataset}')
    if train_differences:
        log('HAI STUDENT KHAC CAU HINH NGOAI LOSS: ' + ', '.join(train_differences) + '; khong quy gain rieng cho KD quan he.')
    compare_std = len(student_names) == 2 and all(configs[name].MODEL.difflift_spread_gate for name in student_names)
    parity = {'pose_max_abs_m': None if len(student_names) < 2 else 0.0,
              'std_compared': compare_std, 'std_max_abs_m': 0.0 if compare_std else None,
              'teacher_gt_max_error_mm': 0.0}
    log('Cung anh/crop/detector. Teacher nhan GT3D; Student nhan lifter cua chinh checkpoint, khong dung GT dau vao.')
    log('Before/after FUSION la vi tri feature; kd_current/kd_relation la HAI RUN, khong phai hai luot cua mot model.')
    names = list(paths)
    offdiag = ~np.eye(17, dtype=bool)
    raw = {name: {} for name in names}
    rows, payloads, gt_poses = [], {}, []
    if attention_mode:
        log(f'ATTENTION joint->image that, block={args.attn_block} (-1=cuoi), mean heads; khong cosine joint-joint.')
    # Giu cau hinh du lieu Student khi iterator/worker chay; chi doi cfg trong moi forward.
    with compare_config(configs['kd_current']):
        ds = get_test_dataset(args.set, None)
        if not len(ds):
            raise ValueError('Dataset rong')
        if args.sample_ids is not None:
            selected = sorted(set(args.sample_ids))
            if not selected or selected[0] < 0 or selected[-1] >= len(ds):
                raise ValueError(f'sample_ids phai nam trong [0, {len(ds)-1}]')
        else:
            selected = sorted(rng.choice(len(ds), size=min(args.relation_samples or len(ds), len(ds)), replace=False).tolist())
        if args.max_batches:
            selected = selected[:args.max_batches * batch_size]
        visual_ids = set(rng.choice(selected, size=min(args.num_vis, len(selected)), replace=False).tolist())
        loader = DataLoader(Subset(ds, selected), batch_size=batch_size, shuffle=False, num_workers=args.workers)
        reg = torch.as_tensor(ds.h36m_joint_regressor, device=device, dtype=torch.float32)
        log(f'Set={args.set} n={len(selected)}/{len(ds)} seed={args.seed} batch={batch_size}; vis={len(visual_ids)}')
        seen = 0
        with torch.no_grad():
            for inputs, targets, meta in tqdm(loader, desc=f'{output_prefix} {args.set}'):
                image = inputs['img'].to(device).float()
                p2 = inputs['joints'].to(device).float()
                mask = inputs['joints_mask'].to(device).float()
                cam = inputs['cam_bbox'].to(device).float()
                gt = torch.matmul(reg[None], targets['smpl_mesh_cam'].to(device).float())
                gt = gt - gt[:, 0:1]
                if not torch.isfinite(gt).all() or (gt.abs().sum(dim=(1, 2)) == 0).any():
                    raise RuntimeError('GT rong hoac NaN/Inf')
                outputs = {}
                for name in names:
                    with compare_config(configs[name]):
                        pose_input = gt if name == 'teacher' else p2
                        extra = {} if name == 'teacher' else {'cam_bbox': cam}
                        outputs[name] = models[name](image, pose_input, is_train=False,
                                                     pose_2d=p2, joints_mask=mask, **extra)
                gt_error = torch.norm(outputs['teacher']['teacher_input_joints'] - gt, dim=-1).max().item() * 1000
                if not np.isfinite(gt_error) or gt_error > 1e-3:
                    raise RuntimeError('Teacher khong nhan dung GT 3D')
                parity['teacher_gt_max_error_mm'] = max(parity['teacher_gt_max_error_mm'], gt_error)
                if len(student_names) == 2:
                    a, b = [outputs[name]['lifted_joints_3d'] for name in student_names]
                    gap = (a - b).abs().max().item()
                    if not np.isfinite(gap) or gap > 1e-6:
                        raise RuntimeError(f'Hai Student khac pose lifter: max_abs={gap} m; kiem tra checkpoint/input')
                    parity['pose_max_abs_m'] = max(parity['pose_max_abs_m'], gap)
                    if compare_std:
                        gap = (outputs['kd_current']['joint_spread_3d'] - outputs['kd_relation']['joint_spread_3d']).abs().max().item()
                        if not np.isfinite(gap) or gap > 1e-6:
                            raise RuntimeError(f'Hai Student khac std: max_abs={gap} m')
                        parity['std_max_abs_m'] = max(parity['std_max_abs_m'], gap)
                batch_data = {}
                for name in names:
                    out = outputs[name]
                    pred = torch.matmul(reg[None], out['smpl_mesh_cam'])
                    pred = pred - pred[:, 0:1]
                    if attention_mode:
                        values = extract_joint_image_attention(out, args.attn_block)
                        values['detector_mask'] = mask.cpu().numpy().reshape(len(image), 17)
                        values['input_pose'] = (out['teacher_input_joints'] if name == 'teacher'
                                                else out['lifted_joints_3d']).cpu().numpy()
                        values['pose'] = pred.cpu().numpy()
                    else:
                        values = {'before': joint_relation_matrix(out['feat_joint_in']).cpu().numpy(),
                                  'after': joint_relation_matrix(out['feat_joint']).cpu().numpy(), 'pose': pred.cpu().numpy()}
                    if any(not np.isfinite(value).all() for value in values.values()):
                        raise RuntimeError(f'{name}: feature/pose NaN/Inf')
                    if not attention_mode and (values['after'].shape[1:] != (17, 17) or values['before'].shape[1:] != (17, 17)):
                        raise RuntimeError('Can token H36M17')
                    batch_data[name] = values
                    for key, value in values.items():
                        raw[name].setdefault(key, []).append(value)
                gt_np = gt.cpu().numpy()
                gt_poses.append(gt_np)
                for i in range(len(image)):
                    idx = selected[seen + i]
                    row = {'sample': idx, 'img_path': value_at(meta, 'img_path', i),
                           'ann_id': value_at(meta, 'aid', i), 'missing_2d': int((mask[i] <= 0).sum().item())}
                    for name in names:
                        row[f'{name}_mpjpe17_mm'] = float(np.linalg.norm(batch_data[name]['pose'][i] - gt_np[i], axis=-1).mean() * 1000)
                        if not attention_mode:
                            for stage in ('before', 'after'):
                                delta = batch_data[name][stage][i] - batch_data['teacher'][stage][i]
                                row[f'{name}_{stage}_mse_vs_teacher'] = float((delta[offdiag] ** 2).mean())
                    for name in student_names:
                        row[f'{name}_input_mpjpe17_mm'] = float(torch.norm(outputs[name]['lifted_joints_3d'][i] - gt[i], dim=-1).mean().item() * 1000)
                    if 'kd_relation' in paths:
                        row['relation_mse_gain_current_minus_new'] = row['kd_current_after_mse_vs_teacher'] - row['kd_relation_after_mse_vs_teacher']
                        row['mpjpe17_gain_current_minus_new_mm'] = row['kd_current_mpjpe17_mm'] - row['kd_relation_mpjpe17_mm']
                    rows.append(row)
                    if idx in visual_ids:
                        payloads[idx] = {'row': row, 'image': image[i].cpu().numpy().copy(), 'p2': p2[i].cpu().numpy().copy(),
                                         'mask': mask[i].cpu().numpy().reshape(17).copy(), 'gt': gt_np[i].copy(),
                                         'models': {name: {key: value[i].copy() for key, value in values.items()}
                                                    for name, values in batch_data.items()}}
                seen += len(image)
    if seen != len(selected):
        raise RuntimeError('Khong du mau trong danh sach da chon')
    for name in names:
        raw[name] = {key: np.concatenate(value) for key, value in raw[name].items()}
    if attention_mode:
        save_attention_diagnostics(args, out_dir, raw, rows, payloads, np.concatenate(gt_poses),
                                   selected, JOINT_NAMES, epochs, parity, len(ds), log)
        return
    write_csv(out_dir / 'samples.csv', rows)
    archive = {'sample_ids': np.asarray(selected), 'joint_names': np.asarray(JOINT_NAMES), 'gt_pose_m': np.concatenate(gt_poses)}
    for name in names:
        for key, value in raw[name].items():
            archive[f'{name}_pose_m' if key == 'pose' else f'{name}_{key}'] = value
    np.savez_compressed(out_dir / 'relations.npz', **archive)
    pairs = []
    for i in range(17):
        for j in range(i + 1, 17):
            row = {'joint_i': JOINT_NAMES[i], 'joint_j': JOINT_NAMES[j]}
            for name in names:
                for stage in ('before', 'after'):
                    values = raw[name][stage][:, i, j]
                    row[f'{name}_{stage}_mean'] = float(values.mean())
                    row[f'{name}_{stage}_std_across_samples'] = float(values.std())
            pairs.append(row)
    write_csv(out_dir / 'pairs.csv', pairs)
    labels = {'teacher': 'Teacher GT', 'kd_current': 'Student current KD', 'kd_relation': 'Student relation KD'}
    colors = {'teacher': 'seagreen', 'kd_current': 'darkorange', 'kd_relation': 'steelblue'}
    fig, axes = plt.subplots(4, len(names), figsize=(6 * len(names), 20), squeeze=False)
    std_max = max(1e-6, max(float(raw[name][stage].std(0)[offdiag].max()) for name in names for stage in ('before', 'after')))
    for col, name in enumerate(names):
        for r, stage in enumerate(('before', 'after')):
            plot_joint_relation(axes[r, col], raw[name][stage].mean(0), f'{labels[name]} | mean {stage.upper()} FUSION')
            plot_joint_relation(axes[r + 2, col], raw[name][stage].std(0), f'{labels[name]} | STD across samples {stage}', 0.0, std_max, 'viridis')
    fig.suptitle(f'Same {seen} samples | {args.set} | columns = separate checkpoints; rows = feature stages\n'
                 'Feature cosine, NOT attention/3D distance. Gray diagonal excluded.', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / 'overview.png', dpi=150)
    plt.close(fig)
    delta_max = max(1e-6, max(float(np.abs(raw[name]['after'] - raw['teacher']['after'])[:, offdiag].max()) for name in student_names))
    for idx, payload in sorted(payloads.items()):
        fig, axes = plt.subplots(4, 3, figsize=(18, 19))
        for ax in axes.flat:
            ax.axis('off')
        crop = np.clip(payload['image'].transpose(1, 2, 0), 0, 1)
        axes[0, 0].imshow(crop)
        pixels = (payload['p2'][:, :2] + 1) * 0.5 * np.array([crop.shape[1], crop.shape[0]])
        valid = payload['mask'] > 0
        axes[0, 0].scatter(pixels[valid, 0], pixels[valid, 1], c='cyan', s=12)
        axes[0, 0].set_title('Common crop + detector PE2D')
        for col, horizontal in ((1, 0), (2, 2)):
            ax = axes[0, col]
            ax.axis('on')
            draw_skeleton(ax, payload['gt'] * 1000, horizontal, 'black', lw=2)
            for name in names:
                draw_skeleton(ax, payload['models'][name]['pose'] * 1000, horizontal, colors[name])
            set_view(ax, [payload['gt'] * 1000] + [payload['models'][name]['pose'] * 1000 for name in names], horizontal)
            legend = 'GT black / ' + ' / '.join(f'{name}: {colors[name]}' for name in names)
            ax.set_title(legend + ': ' + ('front' if col == 1 else 'side'), fontsize=8)
        for col, name in enumerate(names):
            for r, stage in ((1, 'before'), (2, 'after')):
                axes[r, col].axis('on')
                plot_joint_relation(axes[r, col], payload['models'][name][stage], f'{labels[name]} | {stage.upper()} FUSION')
            if name != 'teacher':
                axes[3, col].axis('on')
                plot_joint_relation(axes[3, col], payload['models'][name]['after'] - payload['models']['teacher']['after'],
                                    f'{labels[name]} AFTER - Teacher AFTER', -delta_max, delta_max)
        if 'kd_relation' not in paths:
            axes[1, 2].text(0.5, 0.5, 'Relation KD checkpoint not provided', ha='center', va='center',
                            transform=axes[1, 2].transAxes, fontsize=10)
        row = payload['row']
        text = ['MPJPE17 mm / raw relation MSE vs T (after fusion)']
        for name in names:
            text.append(f'{labels[name]}: {row[name + "_mpjpe17_mm"]:.2f} / {row[name + "_after_mse_vs_teacher"]:.6f}')
        axes[3, 0].text(0, 0.95, '\n'.join(text), va='top', fontsize=9, transform=axes[3, 0].transAxes)
        fig.suptitle(f'#{idx} | ann={row["ann_id"]} | {row["img_path"]}\n'
                     'Columns = checkpoint comparison, NOT sequential inference. Root-relative pose views, not overlays.', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_dir / 'vis' / f'{idx:06d}_compare_relations.png', dpi=140)
        plt.close(fig)
    summary = {'mode': 'relation_compare', 'args': vars(args), 'epochs': epochs, 'n': seen,
               'dataset_n': len(ds), 'subset': seen < len(ds), 'selected_indices': selected,
               'metric_protocol': 'MPJPE17 root-relative mm; raw cosine(LN(token)) MSE off-diagonal, all mesh GT joints',
               'input_parity': parity, 'training_config_differences': train_differences, 'models': {}}
    for name in names:
        summary['models'][name] = {key: float(np.mean([row[key] for row in rows]))
                                   for key in rows[0] if key.startswith(name + '_') and isinstance(rows[0][key], (int, float))}
        log(f'{labels[name]}: MPJPE17={summary["models"][name][name + "_mpjpe17_mm"]:.3f} mm | '
            f'AFTER relation MSE vs Teacher={summary["models"][name][name + "_after_mse_vs_teacher"]:.6f}')
    if 'kd_relation' in paths:
        summary['mean_relation_mse_gain_current_minus_new'] = float(np.mean([r['relation_mse_gain_current_minus_new'] for r in rows]))
        summary['mean_mpjpe17_gain_current_minus_new_mm'] = float(np.mean([r['mpjpe17_gain_current_minus_new_mm'] for r in rows]))
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    log(f'Input parity: {parity}')
    log('Relation MSE la raw cosine gap (khong projector/khong full loss KD E1); nho hon chua bao dam mesh tot hon.')
    log('Before/after fusion khac voi truoc/sau train KD quan he. Teacher GT la oracle; khong phai metric mesh14.')
    log('Cung pose dau vao chua du chung minh cung Teacher init/hard loss/du lieu/seed train; can doi chieu run.')
    log(f'Ket qua: {out_dir} | overview.png, vis/, relations.npz, samples.csv, pairs.csv, summary.json')

def plot_joint_relation(ax, matrix, title, vmin=-1.0, vmax=1.0, cmap='coolwarm'):
    # Duong cheo khong vao loss; dung cung mau/thang cho moi model va mau.
    offdiag = ~np.eye(len(JOINT_NAMES), dtype=bool)
    colors = copy.copy(plt.get_cmap(cmap))
    colors.set_bad('lightgray')
    im = ax.imshow(np.ma.array(matrix, mask=~offdiag), cmap=colors,
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xticks(np.arange(len(JOINT_NAMES)))
    ax.set_yticks(np.arange(len(JOINT_NAMES)))
    ax.set_xticklabels(JOINT_NAMES, rotation=90, fontsize=6)
    ax.set_yticklabels(JOINT_NAMES, fontsize=6)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def inspect_teacher_relation(args):
    """Ve dung quan he feature Teacher dung lam muc tieu KD, khong train."""
    from torch.utils.data import DataLoader, Subset
    from utils.jotr_dataset import get_test_dataset
    from utils.joint_relation import joint_relation_matrix
    from tqdm import tqdm
    import matplotlib
    matplotlib.use('Agg')
    global plt
    import matplotlib.pyplot as plt

    if cfg.MODEL.name != 'teacher' or cfg.MODEL.get('teacher_hint', False):
        raise ValueError('teacher_relation can YAML Teacher GT thuan, name=teacher va teacher_hint=False')
    if cfg.DATASET.seqlen != 1 or cfg.MODEL.get('teacher_lift_alpha_max', 0.0) != 0:
        raise ValueError('Can Teacher GT seqlen1, khong tron lifter')
    if not torch.cuda.is_available():
        raise RuntimeError('Chay tren server GPU co CUDA')
    if not Path(args.ckpt).is_file():
        raise FileNotFoundError(args.ckpt)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device('cuda')
    batch_size = args.batch_size or int(cfg.TEST.batch_size)
    if batch_size < 1:
        raise ValueError('batch_size phai >=1')
    cfg.DATASET.lift_only = False
    cfg.render, cfg.vis = False, False
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out_dir = (Path(args.out_dir) if args.out_dir else
               Path(args.ckpt).parent / f'teacher_relation_{args.set}_{stamp}')
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f'Thu muc ket qua khong rong: {out_dir}')
    vis_dir = out_dir / 'vis'
    vis_dir.mkdir()
    lines = []

    def log(message):
        print(message, flush=True)
        lines.append(message)
        (out_dir / 'inspect_log.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    model, epoch = load_model(args, device)
    ds = get_test_dataset(args.set, None)
    if not len(ds):
        raise ValueError('Dataset rong')
    if args.sample_ids is not None:
        selected = sorted(set(args.sample_ids))
        if not selected or selected[0] < 0 or selected[-1] >= len(ds):
            raise ValueError(f'sample_ids phai nam trong [0, {len(ds)-1}]')
    else:
        count = min(args.relation_samples or len(ds), len(ds))
        selected = sorted(rng.choice(len(ds), size=count, replace=False).tolist())
    if args.max_batches:
        selected = selected[:args.max_batches * batch_size]
    visual_ids = set(rng.choice(selected, size=min(args.num_vis, len(selected)), replace=False).tolist())
    loader = DataLoader(Subset(ds, selected), batch_size=batch_size, shuffle=False, num_workers=args.workers)
    reg = torch.as_tensor(ds.h36m_joint_regressor, device=device, dtype=torch.float32)
    rows, before, after, gt_poses, pred_poses, payloads = [], [], [], [], [], {}
    seen = 0
    offdiag = ~np.eye(17, dtype=bool)
    log(f'mode=teacher_relation | cfg={args.cfg}\nckpt={args.ckpt} | epoch={epoch} | set={args.set}')
    log(f'n={len(selected)}/{len(ds)} | random_seed={args.seed} | batch_size={batch_size} | vis={len(visual_ids)}')
    log('Teacher eval/no_grad, GT mesh -> H36M17 root-relative met, PE2D detector nhu YAML; khong lifter.')
    log('R=cosine(LN(token)), FP32, dung CHUNG helper voi loss KD. Before=feat_joint_in; after=feat_joint.')
    log('Tat ca 17 khop GT mesh hop le; mask detector chi de ve PE2D, KHONG loai cap quan he.')
    with torch.no_grad():
        for inputs, targets, meta in tqdm(loader, desc=f'Teacher relation {args.set}'):
            image = inputs['img'].to(device).float()
            p2 = inputs['joints'].to(device).float()
            detector_mask = inputs['joints_mask'].to(device).float()
            gt_mesh = targets['smpl_mesh_cam'].to(device).float()
            gt = torch.matmul(reg[None], gt_mesh)
            gt = gt - gt[:, 0:1]
            if not torch.isfinite(gt).all() or (gt.abs().sum(dim=(1, 2)) == 0).any():
                raise RuntimeError('GT 3D rong hoac NaN/Inf; khong tao ma tran gia')
            output = model(image, gt, is_train=False, pose_2d=p2, joints_mask=detector_mask)
            input_error = torch.norm(output['teacher_input_joints'] - gt, dim=-1).amax(dim=-1).cpu().numpy() * 1000
            if not np.isfinite(input_error).all() or input_error.max() > 1e-3:
                raise RuntimeError('Teacher khong nhan dung GT 3D sach')
            r_in = joint_relation_matrix(output['feat_joint_in']).cpu().numpy()
            r_out = joint_relation_matrix(output['feat_joint']).cpu().numpy()
            pred = torch.matmul(reg[None], output['smpl_mesh_cam'])
            pred = pred - pred[:, 0:1]
            gt_np, pred_np = gt.cpu().numpy(), pred.cpu().numpy()
            if not all(np.isfinite(x).all() for x in (r_in, r_out, pred_np)):
                raise RuntimeError('Teacher feature/pose co NaN/Inf')
            if r_in.shape[1:] != (17, 17) or r_out.shape[1:] != (17, 17):
                raise RuntimeError('Can dung token H36M17')
            for i in range(len(image)):
                sample_id = selected[seen + i]
                row = {'sample': sample_id, 'img_path': value_at(meta, 'img_path', i),
                       'ann_id': value_at(meta, 'aid', i),
                       'missing_2d': int((detector_mask[i] <= 0).sum().item()),
                       'teacher_input_max_error_mm': float(input_error[i]),
                       'teacher_mpjpe17_mm': float(np.linalg.norm(pred_np[i] - gt_np[i], axis=-1).mean() * 1000),
                       'before_pair_mean': float(r_in[i][offdiag].mean()),
                       'after_pair_mean': float(r_out[i][offdiag].mean())}
                rows.append(row)
                if sample_id in visual_ids:
                    payloads[sample_id] = {'row': row, 'image': image[i].cpu().numpy().copy(),
                                           'p2': p2[i].cpu().numpy().copy(),
                                           'mask': detector_mask[i].cpu().numpy().reshape(17).copy(),
                                           'gt': gt_np[i].copy(), 'pred': pred_np[i].copy(),
                                           'before': r_in[i].copy(), 'after': r_out[i].copy()}
            before.append(r_in)
            after.append(r_out)
            gt_poses.append(gt_np)
            pred_poses.append(pred_np)
            seen += len(image)
    if seen != len(selected):
        raise RuntimeError('So mau forward khong khop danh sach chi so')
    before, after = np.concatenate(before), np.concatenate(after)
    mean_in, mean_out = before.mean(0), after.mean(0)
    std_in, std_out = before.std(0), after.std(0)
    for i, row in enumerate(rows):
        row['before_template_rms'] = float(np.sqrt(((before[i] - mean_in)[offdiag] ** 2).mean()))
        row['after_template_rms'] = float(np.sqrt(((after[i] - mean_out)[offdiag] ** 2).mean()))
    write_csv(out_dir / 'samples.csv', rows)
    pair_rows = []
    for i in range(17):
        for j in range(i + 1, 17):
            pair_rows.append({'joint_i': JOINT_NAMES[i], 'joint_j': JOINT_NAMES[j],
                              'before_mean': float(mean_in[i, j]), 'before_std_across_samples': float(std_in[i, j]),
                              'after_mean': float(mean_out[i, j]), 'after_std_across_samples': float(std_out[i, j])})
    write_csv(out_dir / 'pairs.csv', pair_rows)
    np.savez_compressed(out_dir / 'relations.npz', sample_ids=np.asarray(selected),
                        joint_names=np.asarray(JOINT_NAMES), before=before, after=after,
                        gt_pose_m=np.concatenate(gt_poses), teacher_pose_m=np.concatenate(pred_poses))

    heatmap = plot_joint_relation

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    std_max = max(float(std_in[offdiag].max()), float(std_out[offdiag].max()), 1e-6)
    heatmap(axes[0, 0], mean_in, 'Mean relation BEFORE fusion')
    heatmap(axes[0, 1], mean_out, 'Mean relation AFTER fusion (KD target)')
    heatmap(axes[1, 0], std_in, 'STD across samples BEFORE fusion', 0.0, std_max, 'viridis')
    heatmap(axes[1, 1], std_out, 'STD across samples AFTER fusion', 0.0, std_max, 'viridis')
    fig.suptitle(f'Teacher GT | {args.set} | n={seen} | gray diagonal excluded\n'
                 'Feature similarity, NOT attention or 3D distance', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / 'overview.png', dpi=160)
    plt.close(fig)
    delta_max = max(float(np.abs(after - mean_out)[..., offdiag].max()), 1e-6)
    for sample_id, payload in sorted(payloads.items()):
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        crop = np.clip(payload['image'].transpose(1, 2, 0), 0, 1)
        axes[0, 0].imshow(crop)
        pixels = (payload['p2'][:, :2] + 1) * 0.5 * np.array([crop.shape[1], crop.shape[0]])
        valid = payload['mask'] > 0
        for joint, parent in BONES:
            if valid[joint] and valid[parent]:
                axes[0, 0].plot(pixels[[joint, parent], 0], pixels[[joint, parent], 1], color='lime', lw=1)
        axes[0, 0].scatter(pixels[valid, 0], pixels[valid, 1], c='cyan', s=12)
        for j in np.flatnonzero(valid):
            axes[0, 0].text(*pixels[j], str(j), fontsize=7, color='yellow')
        axes[0, 0].set_title('Crop + detector PE2D (numbers = joint indices)', fontsize=9)
        axes[0, 0].axis('off')
        for col, horizontal in ((1, 0), (2, 2)):
            gt_mm, pred_mm = payload['gt'] * 1000, payload['pred'] * 1000
            draw_skeleton(axes[0, col], gt_mm, horizontal, 'black', lw=2)
            draw_skeleton(axes[0, col], pred_mm, horizontal, 'tab:blue')
            set_view(axes[0, col], [gt_mm, pred_mm], horizontal)
            axes[0, col].set_title('GT black / Teacher mesh joints blue: ' + ('front' if col == 1 else 'side'))
        heatmap(axes[1, 0], payload['before'], 'Relation BEFORE fusion')
        heatmap(axes[1, 1], payload['after'], 'Relation AFTER fusion (KD target)')
        heatmap(axes[1, 2], payload['after'] - mean_out, 'AFTER minus dataset-subset mean', -delta_max, delta_max)
        row = payload['row']
        fig.suptitle(f'#{sample_id} | ann={row["ann_id"]} | Teacher MPJPE17={row["teacher_mpjpe17_mm"]:.2f} mm | '
                     f'relation delta RMS={row["after_template_rms"]:.5f}\n{row["img_path"]}\n'
                     'Pose views are root-relative, not camera overlays; relation is feature cosine.', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(vis_dir / f'{sample_id:06d}_teacher_relation.png', dpi=150)
        plt.close(fig)
    summary = {'args': vars(args), 'epoch': epoch, 'n': seen, 'dataset_n': len(ds),
               'selected_indices': selected, 'joint_names': JOINT_NAMES,
               'relation': 'cosine(LayerNorm(token)), FP32, same helper as KD; diagonal excluded in stats',
               'gt_mask': 'all 17 mesh GT joints; detector mask only used by PE2D',
               'teacher_mpjpe17_mm': float(np.mean([r['teacher_mpjpe17_mm'] for r in rows])),
               'max_teacher_input_error_mm': float(max(r['teacher_input_max_error_mm'] for r in rows)),
               'before_std_across_samples_mean': float(std_in[offdiag].mean()),
               'after_std_across_samples_mean': float(std_out[offdiag].mean()),
               'before_constant_template_mse': float((std_in[offdiag] ** 2).mean()),
               'after_constant_template_mse': float((std_out[offdiag] ** 2).mean()),
               'train_list': list(cfg.DATASET.train_list), 'select_dataset': cfg.TEST.get('select_dataset', '')}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (out_dir / 'config.yml').write_text(Path(args.cfg).read_text(encoding='utf-8'), encoding='utf-8')
    log(f'After: mean STD across samples={summary["after_std_across_samples_mean"]:.6f}; '
        f'constant-template MSE={summary["after_constant_template_mse"]:.6f}')
    if seen < 2:
        log('CHI 1 MAU: std=0 theo dinh nghia, khong du ket luan quan he gan nhu co dinh.')
    log('STD across samples do cung mot cap joint thay doi giua cac mau; khong phai STD giua cac cap trong mot mau.')
    log('Before da co joint encoding/PE2D; khong coi la feature pose 3D thuan hay attention map.')
    log('Quan he thay doi KHONG tu chung minh co ich cho KD/den tu anh; GT pose va dinh danh token cung anh huong.')
    log('Metric tren hinh la MPJPE17 tu mesh, KHONG phai MPJPE14 dataset.evaluate. Khong train/chon checkpoint.')
    log(f'Da luu overview.png, {len(payloads)} hinh trong vis/, relations.npz, pairs.csv, samples.csv, summary.json: {out_dir}')

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # Import sau khi chon GPU; khong chay bat ky phan train nao.
    global np, torch, cfg, plt, JOINT_NAMES, BONES, METRICS, CONDITIONS
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import __init_path  # noqa: F401
    from core.config import cfg, update_config
    from utils.jotr_dataset import get_test_dataset
    from utils.h36m_adapter import HUMAN36M_JOINTS
    from models.diff_lifter import H36M_PARENTS
    from tqdm import tqdm

    defaults = copy.deepcopy(cfg) if args.mode in ('compare', 'teacher_relation', 'attention') else None
    update_config(args.cfg)
    JOINT_NAMES = list(HUMAN36M_JOINTS)
    BONES = [(j, p) for j, p in enumerate(H36M_PARENTS) if p >= 0]
    METRICS = ('mpjpe', 'pa_mpjpe', 'mpvpe')
    if args.mode == 'compare':
        inspect_compare(args, defaults)
        return
    if args.mode == 'attention':
        inspect_relation_compare(args, defaults)
        return
    if args.mode == 'teacher_relation':
        if args.student_cfg:
            inspect_relation_compare(args, defaults)
        else:
            inspect_teacher_relation(args)
        return
    if args.mode == 'teacher':
        inspect_teacher(args)
        return
    if (cfg.MODEL.name != 'student' or cfg.MODEL.pose_lifter != 'diffusion'
            or not cfg.MODEL.difflift_spread_gate):
        raise ValueError('Can YAML Student DiffLifter da bat difflift_spread_gate')
    if cfg.MODEL.get('difflift_select', False) or cfg.MODEL.get('student_hint', False):
        raise ValueError('Mode gate khong ho tro selector/hint')
    if cfg.MODEL.difflift_steps != 1 or cfg.MODEL.difflift_k != 1 or cfg.MODEL.difflift_spread_k < 2:
        raise ValueError('Can pose chinh S=1/K=1 va spread K>=2, giu dung thiet ke run')
    if not cfg.MODEL.get('student_3d_rootrel', True):
        raise ValueError('Chan doan nay can pose 3D root-relative')
    if not torch.cuda.is_available():
        raise RuntimeError('Chay script tren server GPU co CUDA')
    if not Path(args.ckpt).is_file():
        raise FileNotFoundError(args.ckpt)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    batch_size = args.batch_size or int(cfg.TEST.batch_size)
    if batch_size < 2:
        raise ValueError('batch_size phai >=2 de hoan doi spread')
    cfg.DATASET.lift_only = False
    cfg.render = False
    cfg.vis = False
    if args.num_vis:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    JOINT_NAMES = list(HUMAN36M_JOINTS)
    BONES = [(j, p) for j, p in enumerate(H36M_PARENTS) if p >= 0]
    CONDITIONS = ('real', 'off', 'shuffle')
    METRICS = ('mpjpe', 'pa_mpjpe', 'mpvpe')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent / f'inspect_gate_{args.set}_{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise FileExistsError(f'Thu muc ket qua khong rong: {out_dir}; hay chon thu muc moi')
    lines = []

    def log(message):
        print(message, flush=True)
        lines.append(message)
        (out_dir / 'inspect_log.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    model, epoch = load_model(args, device)
    ds = get_test_dataset(args.set, None)
    if not len(ds):
        raise ValueError('Dataset rong')
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=args.workers)
    reg = torch.as_tensor(ds.h36m_joint_regressor, device=device, dtype=torch.float32)
    log(f'cfg={args.cfg}\nckpt={args.ckpt} | epoch={epoch} | set={args.set} | batch_size={batch_size}')
    log(f'Pose S=1/K=1; spread S={cfg.MODEL.difflift_spread_steps}/K={cfg.MODEL.difflift_spread_k}; '
        f'lifter seed=0 nhu wrapper, shuffle/vis seed={args.seed}')
    log(f'Train={list(cfg.DATASET.train_list)} | chon best={cfg.TEST.get("select_dataset", "")} | '
        f'chi chan doan {"tap con" if args.max_batches else "toan tap"}, khong train/chon checkpoint moi')
    log('Giu batch_size va mot GPU nhu phep do can doi chieu: wrapper reset seed moi batch, '
        'doi cach chia batch co the doi nhieu DDIM cua tung mau.')
    log(f'Spread gate: zero_std={cfg.MODEL.difflift_spread_gate_zero}; off dat moi gate=1, '
        'shuffle doi std that truoc gate.')
    log('real-off chi do tac dung gate; real-shuffle moi kiem tra std dung mau. Ban zero_std khong dung std that.')
    log('Shuffle trong batch co the gap cac frame gan nhau; tac dong nho chua chung minh model bo qua spread.')
    rows, joint_rows = [], []
    pool = VisualPool(args.num_vis, args.seed)
    rng = np.random.RandomState(args.seed)
    offset = 0
    parity_max = None
    with torch.no_grad():
        for bi, (inputs, targets, meta) in enumerate(tqdm(loader, desc=f'gate {args.set}')):
            if args.max_batches and bi >= args.max_batches:
                break
            image = inputs['img'].to(device).float()
            p2 = inputs['joints'].to(device).float()
            mask = inputs['joints_mask'].to(device).float()
            cam = inputs['cam_bbox'].to(device).float()
            target = targets['smpl_mesh_cam'].float()
            result, cache = capture_real(model, image, p2, mask, cam)
            pose = result['lifted_joints_3d']
            pe = model._pe2d_kwargs(p2, mask)

            def replay(neutral_gate=False, spread_override=None):
                std_in = cache['std'] if spread_override is None else spread_override
                with override_image_gates(model, 1.0 if neutral_gate else None):
                    return model.smpl_model(joints=pose, img_feats=cache['features'], is_train=False,
                                            return_features=False, joint_spread=std_in, **pe)['smpl_mesh_cam']

            # Tu kiem tra tren GPU khi nguoi dung chay: duong replay phai trung wrapper.
            if bi == 0:
                parity_max = float((replay() - result['smpl_mesh_cam']).abs().max())
                log(f'Kiem tra replay real vs forward_student: max |mesh delta|={parity_max:.3g} met')
                if parity_max > 1e-6 or not np.isfinite(parity_max):
                    raise RuntimeError('Replay khong trung wrapper; dung chan doan')
            predictions = {'real': result['smpl_mesh_cam'].cpu().numpy(), 'off': replay(neutral_gate=True).cpu().numpy()}
            batch = len(image)
            donor = np.roll(np.arange(batch), int(rng.randint(1, batch))) if batch > 1 else None
            if donor is not None:
                donor_tensor = torch.as_tensor(donor, device=device, dtype=torch.long)
                shuffled_std = cache['std'].index_select(0, donor_tensor)
                predictions['shuffle'] = replay(spread_override=shuffled_std).cpu().numpy()
            target_np = target.numpy()
            metrics = {name: evaluate_mesh(ds, prediction, target_np, offset)
                       for name, prediction in predictions.items()}
            for name, prediction in predictions.items():
                if not np.isfinite(prediction).all() or any(not np.isfinite(metrics[name][m]).all() for m in METRICS):
                    raise RuntimeError(f'Gap NaN/Inf o {name}, batch {bi}')
            gt = torch.matmul(reg[None], target.to(device))
            gt = gt - gt[:, 0:1]
            error = torch.norm(pose - gt, dim=-1).cpu().numpy() * 1000
            std = cache['std'].cpu().numpy() * 1000
            std_norm = np.linalg.norm(std, axis=-1)
            pose_np, gt_np = pose.cpu().numpy(), gt.cpu().numpy()
            mask_np = mask.cpu().numpy().reshape(batch, 17)
            for i in range(batch):
                sample_id = offset + i
                row = {'sample': sample_id, 'img_path': value_at(meta, 'img_path', i),
                       'ann_id': value_at(meta, 'aid', i), 'dataset_idx': value_at(meta, 'idx', i),
                       'shuffle_donor_sample': offset + int(donor[i]) if donor is not None else -1,
                       'spread_mm': float(std_norm[i, 1:].mean()), 's1_mpjpe16_mm': float(error[i, 1:].mean())}
                gate_values = cache['gates'][:, i].detach()
                row.update({'gate_mean': float(gate_values.mean()), 'gate_std': float(gate_values.std(unbiased=False)),
                            'gate_min': float(gate_values.min()), 'gate_max': float(gate_values.max())})
                for name in CONDITIONS:
                    for metric in METRICS:
                        row[f'{name}_{metric}'] = float(metrics[name][metric][i]) if name in metrics else float('nan')
                row['gain_off_minus_real_mpjpe'] = row['off_mpjpe'] - row['real_mpjpe']
                row['gain_off_minus_real_mpvpe'] = row['off_mpvpe'] - row['real_mpvpe']
                rows.append(row)
                for j in range(1, 17):
                    joint_rows.append({'sample': sample_id, 'joint': j, 'name': JOINT_NAMES[j],
                                       'valid_2d': int(mask_np[i, j] > 0), 'confidence': float(mask_np[i, j]),
                                       'std_x_mm': float(std[i, j, 0]), 'std_y_mm': float(std[i, j, 1]),
                                       'std_z_mm': float(std[i, j, 2]), 'spread_mm': float(std_norm[i, j]),
                                       's1_error_mm': float(error[i, j])})
                    joint_rows[-1]['gate_mean'] = float(cache['gates'][:, i, j, 0].mean())

                def make_payload():
                    return {'row': row.copy(), 'image': image[i].cpu().numpy().copy(),
                            'xy': p2[i].cpu().numpy().copy(), 'mask': mask[i].cpu().numpy().copy(),
                            'pose': pose_np[i].copy(), 'gt_pose': gt_np[i].copy(),
                            'hyp': cache['hyp'][i].cpu().numpy().copy(), 'std_norm': std_norm[i].copy(),
                            'meshes': {'real': predictions['real'][i].copy(), 'off': predictions['off'][i].copy(),
                                       'gt': target_np[i].copy()}}

                pool.consider(sample_id, row['gain_off_minus_real_mpjpe'], make_payload)
            offset += batch
            del result, cache
    if not rows:
        raise RuntimeError('Khong co mau de chan doan')
    write_csv(out_dir / 'samples.csv', rows)
    write_csv(out_dir / 'joints.csv', joint_rows)
    summary, grouped = summarize(rows, joint_rows, out_dir, log)
    summary['spread_encoding'] = 'gate'
    summary['gate_zero_std'] = bool(cfg.MODEL.get('difflift_spread_gate_zero', False))
    summary.update({'args': vars(args), 'epoch': epoch, 'n': offset, 'dataset_n': len(ds),
                    'subset': offset < len(ds), 'replay_max_mesh_delta_m': parity_max,
                    'pose_steps': 1, 'pose_k': 1, 'spread_steps': int(cfg.MODEL.difflift_spread_steps),
                    'spread_k': int(cfg.MODEL.difflift_spread_k), 'lifter_seed': 0,
                    'train_list': list(cfg.DATASET.train_list), 'select_dataset': cfg.TEST.get('select_dataset', '')})

    def json_safe(value):
        if isinstance(value, dict):
            return {k: json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_safe(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    (out_dir / 'summary.json').write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding='utf-8')
    if args.num_vis:
        plot_overview(rows, joint_rows, grouped, out_dir)
        save_visuals(pool, ds, out_dir, log)
    log('\nCach doc: real tot hon shuffle -> thong tin std dung mau co ich; real tot hon off -> gate da hoc co loi so voi gate=1.')
    log('Bo nhanh la can thiep ngoai phan phoi train, khong thay the doi chung retrain baseline. Khong suy y nghia thong ke tu mot run.')
    log('Mau tot/xau chon theo MPJPE off-real tren CUNG checkpoint; khong phai so voi checkpoint baseline khac.')
    log(f'Ket qua: {out_dir}')


if __name__ == '__main__':
    main()
