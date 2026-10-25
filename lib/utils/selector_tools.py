"""Dung chung cho main/train_selector.py va main/eval_selector.py: dung backbone / lifter, sinh ung vien, danh gia."""
import numpy as np
import torch

from core.config import cfg
from models.ARTS import build_diff_lifter, diff_lift
from models.backbones.resnet import ResNetBackbone
from models.selector import build_options, group_errors, assemble, GROUP_NAMES


def build_frozen_parts(device):
    """Backbone SPIN + DiffLifter (dong bang). SELECTOR.lifter_from_ckpt (neu co) ghi de trong so lifter bang
    'diff_lifter.*' trong checkpoint Student -> dung DUNG lifter ma Student / oracle da dung."""
    backbone = ResNetBackbone(cfg.MODEL.resnet_type)
    backbone.load_pretrained(cfg.MODEL.backbone_pretrained, spin_checkpoint=cfg.MODEL.spin_checkpoint)
    lifter = build_diff_lifter(cfg.MODEL.difflift_checkpoint, 1, 1)
    src = cfg.SELECTOR.get('lifter_from_ckpt', '')
    if src:
        ck = torch.load(src, map_location='cpu', weights_only=False)
        sd = ck.get('model_state_dict', ck)
        sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
        sub = {k[len('diff_lifter.'):]: v for k, v in sd.items() if k.startswith('diff_lifter.')}
        assert sub, f'{src} khong co trong so diff_lifter.*'
        lifter.load_state_dict(sub, strict=True)
        print(f'===> Lifter: nap {len(sub)} tensor diff_lifter.* tu {src} (epoch {ck.get("epoch")})')
    for m in (backbone, lifter):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False
    return backbone.to(device), lifter.to(device)


@torch.no_grad()
def make_candidates(lifter, j2, jm, cam, random_noise):
    """Pose mac dinh S1/K1 (nhieu co dinh, = L0 cua Student) + K gia thuyet S buoc -> (B, K+1, 17, 3) root-rel."""
    base = diff_lift(lifter, j2, jm, cam).float()
    hyps = diff_lift(lifter, j2, jm, cam, all_hyp=True, random_noise=random_noise,
                     steps=cfg.SELECTOR.steps, k=cfg.SELECTOR.K).float()
    return torch.cat([base[:, None], hyps], 1)


def gt_and_valid(targets, meta, device):
    """GT 3D H36M-17 root-relative (met) + mask khop hop le (B, 17) giong loss khop cua Teacher / Student."""
    gt = targets['orig_joint_cam'].to(device).float()
    gt = gt - gt[:, 0:1]
    v = meta['fit_joint_trunc'].to(device).float()
    v = v.squeeze(-1) if v.dim() == 3 else v
    return gt, v * meta['is_valid_fit'].to(device).float()[:, None]


@torch.no_grad()
def evaluate(selector, backbone, lifter, loader, ds, device, teacher=None, max_batches=0, seed=0):
    """Danh gia tren 1 tap test 3DPW: L0 (mac dinh) / Selector / Oracle (cung tap phuong an).
    Pose: MPJPE17 root-relative (mm). Neu co teacher: mesh qua Teacher -> dataset.evaluate (MPJPE14 / PA / MPVPE).
    Tra dict ket qua."""
    selector.eval()
    reg = torch.as_tensor(ds.h36m_joint_regressor, dtype=torch.float32, device=device)
    rows = ('L0 mac dinh', 'Selector', 'Oracle')
    pose_err = {r: [] for r in rows}
    mesh = {r: {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': []} for r in rows}
    hit, nvalid, keep = np.zeros(5), np.zeros(5), np.zeros(5)
    gsel, gbase, gora = np.zeros(5), np.zeros(5), np.zeros(5)
    seen = 0
    for bi, (inputs, targets, meta) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        img = inputs['img'].to(device).float()
        j2 = inputs['joints'].to(device).float()
        jm = inputs['joints_mask'].to(device).float()
        cam = inputs['cam_bbox'].to(device).float()
        gt_mesh = targets['smpl_mesh_cam'].to(device).float()
        gt17 = torch.einsum('jk,bkc->bjc', reg, gt_mesh)
        gt17 = gt17 - gt17[:, 0:1]
        valid = torch.ones(gt17.shape[:2], device=device)
        torch.manual_seed(seed * 100003 + bi)
        cands = make_candidates(lifter, j2, jm, cam, random_noise=True)
        groups = build_options(cands, cfg.SELECTOR.use_flip)
        errs = group_errors(groups, gt17, valid)
        logits = selector(backbone(img)[0], j2, jm, groups)
        choice_sel = [lg.argmax(-1) for lg in logits]
        choice_ora = [e.argmin(-1) for e, _ in errs]
        choice_l0 = [torch.zeros_like(c) for c in choice_sel]   # phuong an 0 = pose mac dinh, khong lat
        poses = {'L0 mac dinh': assemble(groups, choice_l0, gt17), 'Selector': assemble(groups, choice_sel, gt17),
                 'Oracle': assemble(groups, choice_ora, gt17)}
        for g, ((e, _), cs, co) in enumerate(zip(errs, choice_sel, choice_ora)):
            ar = torch.arange(e.shape[0], device=device)
            hit[g] += float((cs == co).sum())
            keep[g] += float((cs == 0).sum())
            nvalid[g] += e.shape[0]
            gsel[g] += float(e[ar, cs].sum()) * 1000
            gbase[g] += float(e[:, 0].sum()) * 1000
            gora[g] += float(e[ar, co].sum()) * 1000
        for r in rows:
            pose_err[r].append((torch.norm(poses[r] - gt17, dim=-1).mean(-1) * 1000).cpu())
        if teacher is not None:
            feat_t = teacher.get_image_features(img)[0]
            gt_np = gt_mesh.cpu().numpy()
            for r in rows:
                out = teacher.smpl_model(joints=poses[r], img_feats=feat_t, is_train=False, return_features=False,
                                         **teacher._pe2d_kwargs(j2, jm))
                pm = out['smpl_mesh_cam'].float().cpu().numpy()
                res = ds.evaluate([{'smpl_mesh_cam': pm[k], 'smpl_mesh_cam_target': gt_np[k].copy()}
                                   for k in range(pm.shape[0])], min(seen, len(ds) - pm.shape[0]))
                for k_ in mesh[r]:
                    mesh[r][k_].extend(res[k_])
        seen += img.shape[0]
    out = {'n': seen, 'pose_mpjpe17': {r: float(torch.cat(pose_err[r]).mean()) for r in rows},
           'group': {GROUP_NAMES[g]: {'acc_top1': hit[g] / max(nvalid[g], 1), 'keep_L0': keep[g] / max(nvalid[g], 1),
                                      'L0_mm': gbase[g] / max(nvalid[g], 1),
                                      'sel_mm': gsel[g] / max(nvalid[g], 1), 'oracle_mm': gora[g] / max(nvalid[g], 1)}
                     for g in range(5)}}
    p = out['pose_mpjpe17']
    out['recovered_pose'] = (p['L0 mac dinh'] - p['Selector']) / max(p['L0 mac dinh'] - p['Oracle'], 1e-6)
    if teacher is not None:
        out['teacher_mesh'] = {r: {k_: float(np.mean(v)) for k_, v in mesh[r].items()} for r in rows}
        m = out['teacher_mesh']
        out['recovered_mesh'] = (m['L0 mac dinh']['mpjpe'] - m['Selector']['mpjpe']) / \
            max(m['L0 mac dinh']['mpjpe'] - m['Oracle']['mpjpe'], 1e-6)
    return out


def format_eval(name, r):
    lines = [f'[{name}] n={r["n"]} | pose MPJPE17 (mm): ' + ' | '.join(f'{k} {v:.2f}' for k, v in r['pose_mpjpe17'].items())
             + f' | lay lai {100 * r["recovered_pose"]:.1f}% khoang L0 -> Oracle']
    if 'teacher_mesh' in r:
        lines.append(f'[{name}] mesh qua Teacher MPJPE/PA/MPVPE (14 khop): ' + ' | '.join(
            f'{k} {v["mpjpe"]:.2f}/{v["pa_mpjpe"]:.2f}/{v["mpvpe"]:.2f}' for k, v in r['teacher_mesh'].items())
            + f' | lay lai {100 * r["recovered_mesh"]:.1f}%')
    lines.append(f'[{name}] tung nhom (mm, L0 / Selector / Oracle, top1, giu L0): ' + ' | '.join(
        f'{g} {v["L0_mm"]:.0f}/{v["sel_mm"]:.0f}/{v["oracle_mm"]:.0f} ({100 * v["acc_top1"]:.0f}%, {100 * v["keep_L0"]:.0f}%)'
        for g, v in r['group'].items()))
    return '\n'.join(lines)
