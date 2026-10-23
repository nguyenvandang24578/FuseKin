"""Chan doan spread cua Student tren cung checkpoint, chi suy luan.

Chay tu goc repo, voi YAML da train checkpoint spread:
    python main/inspect_student_spread.py --cfg config/train_student_v2_diff_spread.yml \
        --ckpt experiment/student_v2_diff_spread/checkpoint/best.pth.tar --gpu 0

Khong sua model/YAML, khong train. real di qua forward_student; off va shuffle
dung lai pose S=1, feature anh va PE2D cua real. off bo CA embedding (ke ca bias).
Mesh dung dataset.evaluate (14 khop); chan doan lifter dung H36M-17, bo pelvis.
"""
import argparse
import csv
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
    p = argparse.ArgumentParser(description='Kiem tra tac dung spread len mesh, khong train')
    p.add_argument('--cfg', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--gpu', default='0', help='mot GPU, giong cau hinh eval batch=32')
    p.add_argument('--set', default='3dpw', choices=['3dpw', '3dpw-val', '3dpw-oc', '3dpw-pc'])
    p.add_argument('--batch_size', type=int, default=0, help='0: dung TEST.batch_size trong YAML')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--max_batches', type=int, default=0, help='0: toan tap; >0: chi chan doan tap con')
    p.add_argument('--seed', type=int, default=0, help='chon donor shuffle va mau ngau nhien; lifter giu seed eval=0')
    p.add_argument('--num_vis', type=int, default=4, help='so mau moi nhom: improved, worsened, random; 0: bo hinh')
    p.add_argument('--out_dir', default='', help='mac dinh tao thu muc moi canh checkpoint')
    args = p.parse_args()
    if ',' in args.gpu or args.batch_size < 0 or args.workers < 0 or args.max_batches < 0 or args.num_vis < 0:
        p.error('Dung mot GPU; cac so luong phai >= 0')
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
    def __init__(self, count, seed):
        self.count = count
        self.rng = np.random.RandomState(seed)
        self.groups = {'improved': [], 'worsened': [], 'random': []}
        self.payloads = {}

    def consider(self, sample_id, gain, make_payload):
        if not self.count:
            return
        candidates = {'random': float(self.rng.random_sample())}
        if gain > 0:
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
    from core.base import _PickleShim
    from models.ARTS import ARTS

    ckpt = torch.load(args.ckpt, map_location='cpu', pickle_module=_PickleShim, weights_only=False)
    if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt:
        raise ValueError('Can checkpoint Student co model_state_dict')
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in ckpt['model_state_dict'].items()}
    if 'spread_proj.weight' not in state or 'spread_proj.bias' not in state:
        raise ValueError('Checkpoint khong co spread_proj da train; khong duoc dung baseline de gia lap spread')
    model = ARTS(num_joint=17, embed_dim=cfg.MODEL.hpe_dim).to(device)
    # Chi ho tro checkpoint khop chinh xac YAML, khong loc key sai shape/thieu backbone.
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, ckpt.get('epoch')


def capture_real(model, image, p2, mask, cam):
    cache = {}

    def backbone_hook(module, inputs, output):
        cache['features'] = output[0]

    def diffusion_hook(module, inputs, output):
        # Lan goi cuoi cua wrapper la nhom gia thuyet de tinh spread.
        pred = output[2]
        cache['hyp'] = pred - pred[:, :, 0:1]

    def spread_hook(module, inputs, output):
        cache['std'] = inputs[0] / 10.0
        cache['embedding'] = output

    hooks = [model.backbone.register_forward_hook(backbone_hook),
             model.diff_lifter.register_forward_hook(diffusion_hook),
             model.spread_proj.register_forward_hook(spread_hook)]
    try:
        result = model.forward_student(image, p2, is_train=False, joints_mask=mask, cam_bbox=cam)
    finally:
        for hook in hooks:
            hook.remove()
    if not all(k in cache for k in ('features', 'hyp', 'std', 'embedding')):
        raise RuntimeError('Khong bat duoc day du dau vao spread tu forward_student')
    if cache['hyp'].shape[1] != int(cfg.MODEL.difflift_spread_k):
        raise RuntimeError('Nhom gia thuyet bat duoc khong khop K spread trong YAML')
    if not torch.allclose(cache['hyp'].std(dim=1), cache['std'], atol=1e-6, rtol=1e-5):
        raise RuntimeError('std bat duoc khong khop gia thuyet; dung de tranh phan tich sai')
    return result, cache


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

    update_config(args.cfg)
    if cfg.MODEL.name != 'student' or cfg.MODEL.pose_lifter != 'diffusion' or not cfg.MODEL.difflift_spread:
        raise ValueError('Can YAML Student diffusion da bat difflift_spread')
    if cfg.MODEL.get('difflift_select', False) or cfg.MODEL.get('student_hint', False):
        raise ValueError('Script chi danh gia spread rieng, khong selector/hint')
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
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent / f'inspect_spread_{args.set}_{stamp}'
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
    log('off: bo ca embedding/bias. shuffle: doi ca tensor (17,3) giua cac mau, khong doi thu tu khop.')
    log('Shuffle trong batch co the gap cac frame gan nhau; tac dong nho chua chung minh model bo qua spread.')
    log(f'spread_proj ||W||={model.spread_proj.weight.norm().item():.6g}, '
        f'||bias||={model.spread_proj.bias.norm().item():.6g}; norm khac 0 khong tu chung minh hieu qua.')
    rows, joint_rows = [], []
    pool = VisualPool(args.num_vis, args.seed)
    rng = np.random.RandomState(args.seed)
    offset = 0
    parity_max = None
    with torch.no_grad():
        for bi, (inputs, targets, meta) in enumerate(tqdm(loader, desc=f'spread {args.set}')):
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

            def replay(embedding):
                return model.smpl_model(joints=pose, img_feats=cache['features'], is_train=False,
                                        return_features=False, extra_tok=embedding, **pe)['smpl_mesh_cam']

            # Tu kiem tra tren GPU khi nguoi dung chay: duong replay phai trung wrapper.
            if bi == 0:
                parity_max = float((replay(cache['embedding']) - result['smpl_mesh_cam']).abs().max())
                log(f'Kiem tra replay real vs forward_student: max |mesh delta|={parity_max:.3g} met')
                if parity_max > 1e-6 or not np.isfinite(parity_max):
                    raise RuntimeError('Replay khong trung wrapper; dung chan doan')
            predictions = {'real': result['smpl_mesh_cam'].cpu().numpy(), 'off': replay(None).cpu().numpy()}
            batch = len(image)
            donor = np.roll(np.arange(batch), int(rng.randint(1, batch))) if batch > 1 else None
            if donor is not None:
                donor_tensor = torch.as_tensor(donor, device=device, dtype=torch.long)
                swapped = model.spread_proj(cache['std'].index_select(0, donor_tensor) * 10.0)
                predictions['shuffle'] = replay(swapped).cpu().numpy()
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
            embeddings = cache['embedding'].norm(dim=-1).mean(-1).cpu().numpy()
            pose_np, gt_np = pose.cpu().numpy(), gt.cpu().numpy()
            mask_np = mask.cpu().numpy().reshape(batch, 17)
            for i in range(batch):
                sample_id = offset + i
                row = {'sample': sample_id, 'img_path': value_at(meta, 'img_path', i),
                       'ann_id': value_at(meta, 'aid', i), 'dataset_idx': value_at(meta, 'idx', i),
                       'shuffle_donor_sample': offset + int(donor[i]) if donor is not None else -1,
                       'spread_mm': float(std_norm[i, 1:].mean()), 's1_mpjpe16_mm': float(error[i, 1:].mean()),
                       'spread_embedding_norm': float(embeddings[i])}
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
    log('\nCach doc: real tot hon shuffle -> dung thong tin spread dung mau; real tot hon off -> phu thuoc nhanh spread.')
    log('Bo nhanh la can thiep ngoai phan phoi train, khong thay the doi chung retrain baseline. Khong suy y nghia thong ke tu mot run.')
    log('Mau tot/xau chon theo MPJPE off-real tren CUNG checkpoint; khong phai so voi checkpoint baseline khac.')
    log(f'Ket qua: {out_dir}')


if __name__ == '__main__':
    main()
