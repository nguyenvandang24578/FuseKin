"""Kiem chung quy uoc root mesh / joint projection, khong train hay sua checkpoint.

Chay tu repo root voi dung YAML cua run. Mac dinh: mesh SMPL tong hop + 64 mau
ngau nhien MOI bo train. --ckpt them forward checkpoint va gradient cuc bo theo
pose/shape/camera; --synthetic_only chi can SMPL + regressor, khong can anh/model.
Loss khac 0 voi mesh hoan hao xac nhan phan du quy uoc, KHONG chung minh tac gia
khong co chu dich hay thay loss se cai thien mesh sau khi train lai.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cfg', required=True)
    p.add_argument('--ckpt', default='', help='checkpoint mesh best.pth.tar, tuy chon')
    p.add_argument('--sets', nargs='+', default=['config'], help='config hoac ten bo TRAIN')
    p.add_argument('--samples', type=int, default=64, help='mau ngau nhien moi bo train')
    p.add_argument('--synthetic_samples', type=int, default=16)
    p.add_argument('--synthetic_only', action='store_true')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--gpu', default='0', help='mot GPU')
    p.add_argument('--tol_mm', type=float, default=0.01, help='dung sai so hoc, khong la nguong y nghia thuc te')
    p.add_argument('--tol_px', type=float, default=0.01)
    p.add_argument('--out_dir', required=True, help='thu muc moi, khong ghi de')
    a = p.parse_args()
    if min(a.samples, a.synthetic_samples, a.batch_size) < 1 or min(a.tol_mm, a.tol_px) <= 0:
        p.error('So mau/batch/dung sai phai > 0')
    if a.synthetic_only and a.ckpt:
        p.error('--synthetic_only khong di kem --ckpt')
    if ',' in a.gpu:
        p.error('Chan doan nay dung mot GPU')
    return a


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def regress(reg, mesh):
    return torch.matmul(reg[None], mesh)


def center(mesh, reg):
    return mesh - regress(reg, mesh)[:, :1]


def project(joints, epsilon=1e-4):
    # Dung dung phep chieu/camera/crop cua FusionPose2Mesh.get_coord.
    focal = joints.new_tensor(cfg.DATASET.focal)
    principal = joints.new_tensor(cfg.DATASET.princpt)
    scale = joints.new_tensor([cfg.output_hm_shape[2] / cfg.input_img_shape[1],
                              cfg.output_hm_shape[1] / cfg.input_img_shape[0]])
    return (joints[..., :2] / (joints[..., 2:3] + epsilon) * focal + principal) * scale


def px_distance(a, b):
    scale = a.new_tensor([cfg.input_img_shape[1] / cfg.output_hm_shape[2],
                          cfg.input_img_shape[0] / cfg.output_hm_shape[1]])
    return ((a - b) * scale).norm(dim=-1)


def finite(name, tensor):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f'{name}: co NaN/Inf, dung chan doan')


def reduce_samples(value, mask):
    # Giong trainer: nhan mask roi mean tat ca toa do, khong chia so khop hop le.
    return (value * mask).flatten(1).mean(1)


def geometry_forward(ctx, pose, shape, camera):
    from models.teacher_student import FusionPose2Mesh
    return FusionPose2Mesh.get_coord(ctx, pose, shape, camera)


def geometry_rows(mesh, pred_mesh, current_proj, rh, rs, selected, stage, ids, depths):
    """Cung mesh lam prediction va GT; tach so hoc FP32/FP64, epsilon va quy uoc."""
    hmesh = center(mesh, rh)
    smesh = center(mesh, rs)
    hs = regress(rh, mesh)
    ss = regress(rs, mesh)[:, selected]
    hp, sp = project(hs), project(ss)
    if min(hs[..., 2].min().item(), ss[..., 2].min().item()) <= 0:
        raise RuntimeError('Camera kiem soat dat khop sau camera; can xem lai don vi mesh')
    parity_m = (pred_mesh - smesh).abs().flatten(1).max(1).values
    parity_px = px_distance(current_proj, sp).max(1).values
    # Khong chuan hoa tong hang regressor: do chinh xac trong so ma code dang dung.
    md, rd, sd = mesh.double(), rh.double(), rs.double()
    hd, sjd = regress(rd, md), regress(sd, md)[:, selected]
    mesh64 = (center(md, sd) - center(md, rd)).abs().flatten(1).mean(1) * 1000
    proj64 = px_distance(project(sjd), project(hd))
    # Doi chung tong hang: chuan hoa CHI de phan tich, khong thay trong so model.
    hsum, ssum = rd.sum(-1, keepdim=True), sd.sum(-1, keepdim=True)
    if min(float(hsum.abs().min()), float(ssum.abs().min())) < 1e-12:
        raise ValueError('Regressor co hang tong gan 0, khong the doi chung chuan hoa')
    hn, sn = regress(rd / hsum, md), regress(sd / ssum, md)[:, selected]
    root_norm = (sn[:, 0] - hn[:, 0]).norm(dim=-1) * 1000
    proj_norm = px_distance(project(sn), project(hn)).mean(1)
    delta3 = (ss - hs).norm(dim=-1) * 1000
    gap_px = px_distance(sp, hp)
    eps_px = px_distance(hp, project(hs, epsilon=0.0))
    root_gap = (regress(rs, mesh)[:, 0] - hs[:, 0]).norm(dim=-1) * 1000
    rows, joints = [], []
    for b, index in enumerate(ids):
        rows.append({'stage': stage, 'index': int(index), 'virtual_depth_m': float(depths[b]),
                     'root_gap_mm': float(root_gap[b]),
                     'perfect_mesh_current_l1_mm': float((pred_mesh[b] - hmesh[b]).abs().mean() * 1000),
                     'perfect_mesh_fp64_l1_mm': float(mesh64[b]),
                     'root_gap_row_normalized_mm': float(root_norm[b]),
                     'same_root_self_control_l1_mm': float((hmesh[b] - hmesh[b].detach()).abs().mean() * 1000),
                     'joint_definition_gap_mm': float(delta3[b].mean()),
                     'perfect_proj_current_mean_px': float(gap_px[b].mean()),
                     'perfect_proj_fp64_mean_px': float(proj64[b].mean()),
                     'projection_gap_row_normalized_mean_px': float(proj_norm[b]),
                     'same_regressor_self_control_px': float(px_distance(hp[b], hp[b].detach()).mean()),
                     'projection_epsilon_only_mean_px': float(eps_px[b].mean()),
                     'get_coord_mesh_parity_max_mm': float(parity_m[b] * 1000),
                     'get_coord_proj_parity_max_px': float(parity_px[b])})
        for j, name in enumerate(JOINT_NAMES):
            joints.append({'stage': stage, 'index': int(index), 'joint': name,
                           'definition_gap_mm': float(delta3[b, j]), 'definition_gap_px': float(gap_px[b, j]),
                           'epsilon_only_px': float(eps_px[b, j])})
    return rows, joints


def loss_graph(ctx, pose, shape, camera, target_mesh, target2d, fit_mask, trunc, rh, selected):
    """Dau vao da detach khoi network; gradient chi theo pose/shape/camera, khong optimizer."""
    pose, shape, camera = [v.detach().clone().requires_grad_(True) for v in (pose, shape, camera)]
    _, _, mesh, _ = geometry_forward(ctx, pose, shape, camera)
    proj_pose = pose.detach() if cfg.LOSS.DETACH_POSE_FOR_PROJ else pose
    proj, _, _, absolute = geometry_forward(ctx, proj_pose, shape, camera)
    hp = project(regress(rh, absolute))
    losses = {
        'mesh_current_m': reduce_samples((mesh - target_mesh).abs(), fit_mask),
        'mesh_same_h36m_root_m': reduce_samples((center(mesh, rh) - center(target_mesh, rh)).abs(), fit_mask),
        'proj_current_hm': reduce_samples((proj[:, selected] - target2d).abs(), trunc),
        'proj_same_h36m_regressor_hm': reduce_samples((hp - target2d).abs(), trunc),
    }
    # Doi chieu loss hien tai voi chinh lop loss trainer dung, tranh mirror sai mask/reduction.
    from core.loss import CoordLoss, JOTRCoordLoss
    production_mesh = CoordLoss(has_valid=True)(mesh, target_mesh, fit_mask)
    production_proj = JOTRCoordLoss()(proj[:, selected], target2d, trunc).mean()
    for key, production in (('mesh_current_m', production_mesh), ('proj_current_hm', production_proj)):
        if not torch.allclose(losses[key].mean(), production, atol=1e-7, rtol=1e-6):
            raise RuntimeError(f'{key}: cong thuc chan doan khong khop lop loss trainer')
    return losses, (pose, shape, camera), mesh, proj[:, selected], hp


def gradient_rows(losses, leaves, stage):
    rows, saved = [], {}
    for term, values in losses.items():
        grads = torch.autograd.grad(values.mean(), leaves, retain_graph=True, allow_unused=True)
        saved[term] = grads
        for name, grad in zip(('pose_axis_angle', 'shape_beta', 'camera_translation_m'), grads):
            if grad is not None:
                finite(f'gradient {term}/{name}', grad)
            rows.append({'stage': stage, 'term': term, 'parameter': name,
                         'connected': grad is not None,
                         'grad_l2': float(grad.norm()) if grad is not None else 0.0,
                         'loss_raw_mean': float(values.detach().mean())})
    for old, new in (('mesh_current_m', 'mesh_same_h36m_root_m'),
                     ('proj_current_hm', 'proj_same_h36m_regressor_hm')):
        for i, name in enumerate(('pose_axis_angle', 'shape_beta', 'camera_translation_m')):
            a, b = saved[old][i], saved[new][i]
            cosine = None
            if a is not None and b is not None and min(float(a.norm()), float(b.norm())) > 0:
                cosine = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
            rows.append({'stage': stage, 'term': f'{old} vs {new}', 'parameter': name,
                         'gradient_cosine': cosine})
    return rows


def strict_model(path, device):
    from core.base import _PickleShim, validate_spread_gate_checkpoint
    from models.ARTS import ARTS
    ck = torch.load(path, map_location='cpu', pickle_module=_PickleShim, weights_only=False)
    if not isinstance(ck, dict) or 'model_state_dict' not in ck:
        raise ValueError('Can checkpoint mesh co model_state_dict')
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in ck['model_state_dict'].items()}
    state = {('smpl_model.' + k[len('teacher_model.'):] if k.startswith('teacher_model.') else k): v
             for k, v in state.items()}
    model = ARTS(num_joint=17, embed_dim=cfg.MODEL.hpe_dim).to(device)
    validate_spread_gate_checkpoint(model, ck)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, ck.get('epoch')


def checkpoint_forward(model, inputs, targets, device):
    image = inputs['img'].to(device).float()
    p2 = inputs['joints'].to(device).float()
    mask = inputs['joints_mask'].to(device).float()
    if cfg.MODEL.name == 'teacher':
        # Cung nguon GT nhu Teacher_Trainer, khong corruption hay tron lifter.
        gt = targets['orig_joint_cam'].to(device).float()
        gt = gt - gt[:, :1]
        return model(image, gt, is_train=False, pose_2d=p2, joints_mask=mask)
    raw, cam = inputs.get('joints_raw'), inputs.get('cam_bbox')
    return model(image, p2, is_train=False, joints_mask=mask,
                 pose_2d_raw=raw.to(device).float() if raw is not None else None,
                 cam_bbox=cam.to(device).float() if cam is not None else None)


def numeric_summary(rows):
    result = {}
    for stage in sorted({r['stage'] for r in rows}):
        group = [r for r in rows if r['stage'] == stage]
        stats = {}
        for key in dict.fromkeys(k for r in group for k in r):
            if key in ('stage', 'index', 'valid_fit', 'img_path'):
                continue
            vals = [r[key] for r in group if isinstance(r.get(key), (float, int))]
            if vals:
                a = np.asarray(vals, dtype=np.float64)
                if not np.isfinite(a).all():
                    raise RuntimeError(f'{stage}/{key}: so lieu khong huu han')
                stats[key] = {'mean': float(a.mean()), 'median': float(np.median(a)),
                              'max': float(a.max()), 'n': len(vals)}
        result[stage] = {'n': len(group), 'statistics': stats}
        if any('valid_fit' in r for r in group):
            result[stage]['valid_fit_n'] = sum(bool(r.get('valid_fit', False)) for r in group)
    return result


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    out = Path(args.out_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit(f'Thu muc ket qua da co noi dung: {out}')
    global torch, np, cfg, JOINT_NAMES
    import numpy as np
    import torch
    import yaml
    from core.config import cfg, update_config
    from utils.h36m_adapter import HUMAN36M_JOINTS
    from utils.smpl import SMPL

    update_config(args.cfg)
    JOINT_NAMES = HUMAN36M_JOINTS
    if not torch.cuda.is_available():
        raise SystemExit('Can CUDA va moi truong du lieu/SMPL cua repo')
    if not args.synthetic_only and (cfg.DATASET.lift_only or cfg.DATASET.seqlen != 1):
        raise ValueError('Can YAML mesh seqlen=1, lift_only=False; khong tu doi quy uoc loader')
    if args.ckpt and (cfg.MODEL.name not in ('teacher', 'student') or cfg.MODEL.REFINER != 'hypergcn'
                      or cfg.MODEL.teacher_hint or cfg.MODEL.student_hint
                      or cfg.MODEL.teacher_lift_alpha_max > 0):
        raise ValueError('Checkpoint chi ho tro Teacher GT thuan/Student khong hint, refiner hypergcn')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(json.loads(json.dumps(cfg)), allow_unicode=True), encoding='utf-8')
    log_lines = []

    def log(message):
        print(message, flush=True)
        log_lines.append(message)
        (out / 'log.txt').write_text('\n'.join(log_lines) + '\n', encoding='utf-8')

    human = SMPL()
    layer = human.layer['neutral'].to(device).eval()
    for p in layer.parameters():
        p.requires_grad_(False)
    rs = torch.as_tensor(human.joint_regressor, dtype=torch.float32, device=device)
    reg_path = Path(cfg.DATASET.jotr_data_root) / 'Human36M' / 'J_regressor_h36m_correct.npy'
    rh = torch.as_tensor(np.load(reg_path), dtype=torch.float32, device=device)
    selected = [human.joints_name.index(name) for name in JOINT_NAMES]
    ctx = SimpleNamespace(human_model_layer=layer, joint_regressor_t=rs, human_model=human)
    if rh.shape != (17, 6890) or rs.shape != (30, 6890) or human.root_joint_idx != 0:
        raise ValueError('Regressor/skeleton khac quy uoc da doc cua FuseKin')
    finite('H36M regressor', rh)
    finite('SMPL regressor', rs)
    reg_rows = [{'joint': name, 'h36m_row_sum': float(rh[j].double().sum()),
                 'smpl_row_sum': float(rs[selected[j]].double().sum()),
                 'row_l1_difference': float((rh[j].double() - rs[selected[j]].double()).abs().sum())}
                for j, name in enumerate(JOINT_NAMES)]
    write_csv(out / 'regressors.csv', reg_rows)
    summary = {'args': vars(args), 'cfg_sha256': file_hash(args.cfg), 'regressor_sha256': file_hash(reg_path),
               'status': 'running', 'checkpoint': None, 'selected_indices': {},
               'notes': ['Loss L1 mesh don vi m; L1 projection don vi heatmap. Khong phai MPJPE14/MPVPE.',
                         'Gradient la dao ham cuc bo theo SMPL pose/shape/camera, chua nhan AWL/KD/hard_w.',
                         'Camera mau that o geometry_* la camera ao kiem soat, khong phai camera GT dataset.',
                         'row_normalized chi la doi chung tong hang, khong sua regressor that.',
                         'Mean loss checkpoint giu ca mau invalid voi mask0 nhu trainer; doc valid_fit_n.',
                         'Khac quy uoc khong xac dinh y dinh tac gia hay loi ich sau retrain.']}
    summary['source_sha256'] = {name: file_hash(ROOT / name) for name in (
        'main/check_loss_conventions.py', 'lib/core/base.py', 'lib/core/config.py',
        'lib/models/teacher_student.py', 'lib/models/ARTS.py', 'lib/utils/fusekin_sample.py',
        'lib/utils/jotr_dataset.py', 'data_final/PW3D/dataset.py')}
    rows, joint_rows, grads = [], [], []
    # Mesh tong hop: co T-pose va nhieu pose/shape; camera duong sau, khong dung network.
    n = args.synthetic_samples
    pose = torch.randn(n, 72, device=device) * 0.2
    shape = torch.randn(n, 10, device=device) * 0.5
    pose[0], shape[0] = 0, 0
    camera = torch.zeros(n, 3, device=device)
    camera[:, 2] = 3.0 + torch.arange(n, device=device) % 4
    with torch.no_grad():
        projection, _, mesh, absolute = geometry_forward(ctx, pose, shape, camera)
        finite('synthetic mesh', absolute)
        a, b = geometry_rows(absolute, mesh, projection[:, selected], rh, rs, selected,
                             'synthetic', range(n), camera[:, 2])
        rows.extend(a)
        joint_rows.extend(b)
        target_mesh = center(absolute, rh).detach()
        target2d = project(regress(rh, absolute)).detach()
    losses, leaves, replay_mesh, replay_proj, replay_hp = loss_graph(ctx, pose, shape, camera, target_mesh, target2d,
                                         torch.ones(n, 1, 1, device=device),
                                         torch.ones(n, 17, 1, device=device), rh, selected)
    grads.extend(gradient_rows(losses, leaves, 'synthetic_perfect'))
    summary['synthetic_loss_graph_raw'] = {key: float(value.detach().mean()) for key, value in losses.items()}
    del losses, leaves, replay_mesh, replay_proj, replay_hp
    log(f'[synthetic] {n} mesh; dung get_coord that, doi chung FP64/epsilon/regressor')
    model = None
    if args.ckpt:
        model, epoch = strict_model(args.ckpt, device)
        summary['checkpoint'] = {'path': args.ckpt, 'sha256': file_hash(args.ckpt), 'epoch': epoch,
                                 'strict': True}
        log(f'[checkpoint] strict load epoch={epoch}: {args.ckpt}')
        if getattr(model, 'use_diff', False):
            path = cfg.MODEL.difflift_checkpoint
            summary['lifter_constructor'] = {'path': path, 'sha256': file_hash(path),
                'scale': model.diff_lifter.scale, 'timesteps': model.diff_lifter.num_timesteps,
                'sampling_steps': model.diff_lifter.sampling_timesteps,
                'k': model.diff_lifter.num_proposals}
    if not args.synthetic_only:
        from torch.utils.data import DataLoader, Subset
        from utils.jotr_dataset import get_train_dataset
        names = list(cfg.DATASET.train_list) if args.sets == ['config'] else args.sets
        if not names or 'config' in names:
            raise ValueError('--sets can config hoac danh sach ten bo train')
        for ds_id, name in enumerate(names):
            # Seed rieng moi bo; workers=0 de tai lap augmentation va thu tu mau.
            random.seed(args.seed + ds_id)
            np.random.seed(args.seed + ds_id)
            torch.manual_seed(args.seed + ds_id)
            ds = get_train_dataset(name, None)
            if len(ds) == 0:
                raise ValueError(f'{name}: dataset rong')
            rds = torch.as_tensor(ds.h36m_joint_regressor, device=device, dtype=torch.float32)
            if not torch.equal(rds, rh):
                raise ValueError(f'{name}: H36M regressor khac regressor kiem soat')
            if not np.array_equal(ds.mesh_model.joint_regressor, human.joint_regressor):
                raise ValueError(f'{name}: SMPL regressor khac model')
            indices = np.random.default_rng(args.seed + ds_id).choice(len(ds), min(args.samples, len(ds)), replace=False).tolist()
            summary['selected_indices'][name] = indices
            loader = DataLoader(Subset(ds, indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
            offset = 0
            for bi, (inputs, targets, meta) in enumerate(loader):
                for key in ('smpl_mesh_cam', 'orig_joint_cam', 'orig_joint_img'):
                    if key not in targets:
                        raise ValueError(f'{name}: thieu {key}; dung YAML mesh unified/3dpw-train')
                gt = targets['smpl_mesh_cam'].to(device).float()
                finite(f'{name}/GT mesh', gt)
                batch = gt.shape[0]
                ids = indices[offset:offset + batch]
                fit = meta['is_valid_fit'].to(device).float().reshape(batch, 1, 1)
                # Cung mesh GT, tao prediction hoan hao theo quy uoc root SMPL.
                stored_s = regress(rs, gt)[:, 0].norm(dim=-1) * 1000
                stored_h = regress(rh, gt)[:, 0].norm(dim=-1) * 1000
                gt_j = targets['orig_joint_cam'].to(device).float()
                h_j = regress(rh, gt)
                joint_parity = ((h_j - h_j[:, :1]) - (gt_j - gt_j[:, :1])).norm(dim=-1).mean(1) * 1000
                for k, index in enumerate(ids):
                    rows.append({'stage': f'train_target/{name}', 'index': index, 'valid_fit': bool(fit[k].item()),
                                 'img_path': str(meta.get('img_path', [''] * batch)[k]),
                                 'stored_root_smpl_mm': float(stored_s[k]), 'stored_root_h36m_mm': float(stored_h[k]),
                                 'stored_target_joint_parity_mm': float(joint_parity[k]),
                                 'perfect_prediction_vs_stored_target_l1_mm': float((center(gt, rs)[k] - gt[k]).abs().mean() * 1000)})
                # Chieu dinh nghia khop tren mesh that nhung voi camera AO 3m, khong so voi GT2D dataset.
                valid_ids = [k for k in range(batch) if fit[k].item() > 0]
                if valid_ids:
                    virtual = center(gt[valid_ids], rh).clone()
                    virtual[..., 2] += 3.0
                    ga, gb = geometry_rows(virtual, center(virtual, rs), project(regress(rs, virtual)[:, selected]),
                                           rh, rs, selected, f'geometry_virtual_camera/{name}',
                                           [ids[k] for k in valid_ids],
                                           torch.full((len(valid_ids),), 3.0, device=device))
                    rows.extend(ga)
                    joint_rows.extend(gb)
                if model is not None:
                    with torch.no_grad():
                        result = checkpoint_forward(model, inputs, targets, device)
                    trunc = meta['orig_joint_trunc'].to(device).float()
                    target2d = targets['orig_joint_img'][..., :2].to(device).float()
                    losses, leaves, replay, rp, hp = loss_graph(model.smpl_model, result['smpl_pose'],
                        result['smpl_shape'], result['cam_param'], gt, target2d, fit, trunc, rh, selected)
                    parity_mm = float((replay.detach() - result['smpl_mesh_cam']).abs().max() * 1000)
                    parity_px = float(px_distance(rp.detach(), result['joint_proj_det'][:, selected]).max())
                    if parity_mm > args.tol_mm or parity_px > args.tol_px:
                        raise RuntimeError(f'{name}: replay get_coord khong khop wrapper: {parity_mm}mm / {parity_px}px')
                    if bi == 0:
                        grads.extend(gradient_rows(losses, leaves, f'checkpoint_first_batch/{name}'))
                    for k, index in enumerate(ids):
                        row = {'stage': f'checkpoint/{name}', 'index': index, 'valid_fit': bool(fit[k].item()),
                               'valid_2d_joints': int((trunc[k] > 0).sum()),
                               'replay_mesh_max_mm': parity_mm, 'replay_proj_max_px': parity_px}
                        row.update({term: float(value[k].detach()) for term, value in losses.items()})
                        rows.append(row)
                    del losses, leaves, result, replay, rp, hp
                offset += batch
            log(f'[train] {name}: {offset}/{len(ds)} mau, augmentation theo YAML, khong optimizer')
            del ds, loader
    write_csv(out / 'samples.csv', rows)
    write_csv(out / 'joints.csv', joint_rows)
    write_csv(out / 'gradients.csv', grads)
    summary['stages'] = numeric_summary(rows)
    synth = [r for r in rows if r['stage'] == 'synthetic']
    parity_ok = all(r['get_coord_mesh_parity_max_mm'] <= args.tol_mm and
                    r['get_coord_proj_parity_max_px'] <= args.tol_px for r in synth)
    summary['synthetic_checks'] = {
        'get_coord_parity_ok': parity_ok,
        'same_root_self_control_ok': all(r['same_root_self_control_l1_mm'] <= args.tol_mm for r in synth),
        'same_regressor_self_control_ok': all(r['same_regressor_self_control_px'] <= args.tol_px for r in synth),
        'root_residual_above_numeric_tolerance_fp64': any(r['perfect_mesh_fp64_l1_mm'] > args.tol_mm for r in synth),
        'joint_projection_gap_above_numeric_tolerance_fp64': any(r['perfect_proj_fp64_mean_px'] > args.tol_px for r in synth),
    }
    if not parity_ok:
        raise RuntimeError('Synthetic replay khong khop get_coord; khong duoc dien giai phan du')
    summary['status'] = 'completed'
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    log('[checks] ' + json.dumps(summary['synthetic_checks']))
    for stage, values in summary['stages'].items():
        stats = values['statistics']
        keys = ('perfect_mesh_fp64_l1_mm', 'perfect_proj_fp64_mean_px', 'stored_root_smpl_mm',
                'mesh_current_m', 'mesh_same_h36m_root_m', 'proj_current_hm', 'proj_same_h36m_regressor_hm')
        log(f'[{stage}] n={values["n"]}: ' + ' | '.join(f'{k}={stats[k]["mean"]:.6g}' for k in keys if k in stats))
    log('Chi xac nhan phan du quy uoc, khong suy ra y dinh tac gia hay loi ich retrain. '
        'Gradient o gan loss=0 co the nhay do L1/lam tron; doc cung dung sai va FP64. '
        'Checkpoint loss khong phai metric danh gia; can train doi chung tung loss neu muon ket luan hieu qua.')
    log(f'Da luu: {out}')


if __name__ == '__main__':
    main()
