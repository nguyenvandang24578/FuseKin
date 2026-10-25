"""
analyze_teacher.py -- Mo xe Teacher GT (va so voi Student) de tim kien thuc nao co the truyen. Chi eval, khong train.

Cac phep do (MPJPE17 H36M root-rel tu mesh / PA-MPJPE / MPVPE / goc xoay toan than, mm & do):
 A. San cua Teacher voi pose GT: loi den tu dau (xoay toan than, khop nao, shape).
 B. Do nhay cua Teacher voi pose dau vao bi lam hong co kiem soat:
      nhieu Gauss tung khop (--noise_mm), lat do sau 1 chi ngau nhien, phan xa do sau toan than;
    moi muc chay anh that va anh TB batch -> "anh sua duoc bao nhieu" khi pose sai (Teacher co biet sua khong?).
 C. (can --student_ckpt) Student voi pose lifter, Student voi pose GT (Student khai thac pose tot den dau?).
 D. (can --student_ckpt) Ghep kich hoat (activation patching) Teacher -> Student:
      thay token khop sau bo ma hoa / token khop sau fusion / token anh sau fusion / ca hai bang cua Teacher
    -> % khoang cach Student-Teacher duoc lap = noi kien thuc Teacher thuc su nam.
 E. Attention khop->anh tung block: entropy va do lech tam attention so voi GT 2D khop (chi tap 3DPW).

Vi du:
  python main/analyze_teacher.py --cfg config/train_student_kd_e1_v2.yml \
      --student_ckpt experiment/<run>/checkpoint/best.pth.tar --set 3dpw --max_batches 300 \
      --out_dir experiment/_eval/analyze_teacher_3dpw
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Phan tich Teacher GT va so voi Student (eval-only)')
parser.add_argument('--cfg', type=str, required=True, help='YAML KD (chua MODEL.TEACHER, lifter cua Student)')
parser.add_argument('--teacher_ckpt', type=str, default='', help='mac dinh cfg.MODEL.TEACHER')
parser.add_argument('--student_ckpt', type=str, default='', help='bo trong -> chi phan tich Teacher (A, B, E)')
parser.add_argument('--set', type=str, default='', help='tap test; rong = cfg.DATASET.test_list[0]')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--max_batches', type=int, default=300)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--noise_mm', type=str, default='10,20,40,80', help='muc nhieu Gauss tung khop (mm)')
parser.add_argument('--out_dir', type=str, required=True, help='thu muc MOI luu log + summary.json')
args = parser.parse_args()

from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401
from models.ARTS import ARTS
from models.teacher_student import RGBJointFusion
from core.base import load_model_weights, _PickleShim
from utils.jotr_dataset import get_test_dataset

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if not torch.cuda.is_available():
    print("\n[LOI] Can CUDA de chay script nay.\n")
    sys.exit(1)
DEVICE = torch.device('cuda')

def hr(title=''):
    print('\n' + '=' * 78)
    if title:
        print(title)
        print('=' * 78)


def inspect_ckpt(mode, ckpt_path):
    """Doc checkpoint -> (kind, info). kind: 'teacher' | 'student' | 'khong ro'.
    info: has_probe, hint ('teacher' | 'student' | None), hint_dim."""
    if not os.path.isfile(ckpt_path):
        print(f"[LOI] Khong thay checkpoint {mode}: {ckpt_path}")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location='cpu', pickle_module=_PickleShim, weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in ckpt:
                state = ckpt[k]
                break
    keys = list(state.keys())
    has_hint_enc = any('hint_encoder.' in k for k in keys)
    has_hint_pred = any('hint_pred.' in k for k in keys)
    hint_dim = 0
    for k in keys:
        if k.endswith('hint_proj.weight'):
            hint_dim = int(state[k].shape[1])
    if has_hint_enc:
        kind = 'teacher'          # T-hint: kien truc Student + HintEncoder
    elif any('fusion.projector_student.' in k for k in keys):
        kind = 'student'
    elif any('fusion.joint_proj.' in k for k in keys):
        kind = 'teacher'
    else:
        kind = 'khong ro'
    # Kien truc v2 (PLAN_STUDENT_V2.md): nhan dien tu ten key -> dung lai dung kien truc, khong phu thuoc yml
    pe2d_key = next((k for k in keys if k.endswith('fusion.pe2d.proj.weight')), None)
    arch = {
        'joint_pe2d': pe2d_key is not None,
        'pe2d_num_freq': int(state[pe2d_key].shape[1]) // 4 if pe2d_key else 6,
        'fusion_joint_self_attn': any('.joint_sa.' in k for k in keys),
        'pose_head_mode': 'per_joint' if any('per_joint_head.' in k for k in keys) else 'vposer',
        'pose_query_bone': any('bone_proj.' in k for k in keys),
        'shape_attn_mode': 'tokens' if any('fuse_shape_tok.' in k for k in keys) else 'global',
    }
    info = {'has_probe': any('probe_' in k for k in keys),
            'hint': 'teacher' if has_hint_enc else ('student' if has_hint_pred else None),
            'hint_dim': hint_dim, 'arch': arch}
    tag = f" (T-hint k={hint_dim})" if has_hint_enc else (f" (S-hint k={hint_dim})" if has_hint_pred else "")
    v2 = {k: v for k, v in arch.items() if v not in (False, 'vposer', 'global') and k != 'pe2d_num_freq'}
    if v2:
        tag += f" (v2: {v2})"
    print(f"  [{mode}] {ckpt_path} -> loai: {kind}{tag}" + (" (co head tham do anh)" if info['has_probe'] else ""))
    if kind != 'khong ro' and kind != mode:
        print(f"\n[LOI] Checkpoint {kind.upper()} dat vao vi tri {mode.upper()}: {ckpt_path}")
        sys.exit(1)
    return info


def build_and_load(mode, ckpt_path, need_lift=False):
    info = inspect_ckpt(mode, ckpt_path)
    keys = ('name', 'teacher_lift_alpha_max', 'img_probe', 'teacher_hint', 'student_hint', 'hint_dim') \
        + tuple(info['arch'].keys())
    saved = {k: cfg.MODEL[k] for k in keys if k in cfg.MODEL}
    cfg.MODEL.name = mode
    for k, v in info['arch'].items():   # kien truc v2 lay tu checkpoint (Teacher cu va Student v2 khac nhau)
        cfg.MODEL[k] = v
    if mode == 'teacher':
        cfg.MODEL.teacher_hint = info['hint'] == 'teacher'
        cfg.MODEL.teacher_lift_alpha_max = 1.0 if (need_lift and not cfg.MODEL.teacher_hint) else 0.0
        cfg.MODEL.img_probe = info['has_probe']
    else:
        cfg.MODEL.student_hint = info['hint'] == 'student'
    if info['hint_dim'] > 0:
        cfg.MODEL.hint_dim = info['hint_dim']
    model = ARTS(num_joint=17, embed_dim=cfg.MODEL.hpe_dim)
    for k in keys:   # tra cfg ve nhu cu (key khong co tu dau thi xoa)
        if k in saved:
            cfg.MODEL[k] = saved[k]
        elif k in cfg.MODEL:
            del cfg.MODEL[k]
    model = model.to(DEVICE)
    model = load_model_weights(model, ckpt_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, info



def h36m_from_mesh(mesh, regressor):
    return torch.einsum('jk,bkc->bjc', regressor, mesh)


def errors_mm(mesh_pred, mesh_gt, reg):
    """MPJPE (17 khop H36M, root-rel) va MPVPE (mesh can theo Pelvis H36M), mm, theo tung mau."""
    jp, jg = h36m_from_mesh(mesh_pred, reg), h36m_from_mesh(mesh_gt, reg)
    rp, rg = jp[:, 0:1], jg[:, 0:1]
    mpjpe = torch.norm((jp - rp) - (jg - rg), dim=-1).mean(-1) * 1000.0
    mpvpe = torch.norm((mesh_pred - rp) - (mesh_gt - rg), dim=-1).mean(-1) * 1000.0
    return mpjpe.cpu(), mpvpe.cpu()



# ============================================================
LR_PAIRS = ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16))
LIMBS = ((1, 2, 3), (4, 5, 6), (11, 12, 13), (14, 15, 16))


def procrustes(pred, gt, with_scale=True):
    """pred, gt (B, J, 3) -> (pred da can, goc xoay (do)). Can giua, xoay (+ ty le) pred ve gt."""
    mp, mg = pred.mean(1, keepdim=True), gt.mean(1, keepdim=True)
    p, g = pred - mp, gt - mg
    h = p.transpose(1, 2) @ g                                   # (B, 3, 3)
    u, s, vt = torch.linalg.svd(h)
    d = torch.sign(torch.det(vt.transpose(1, 2) @ u.transpose(1, 2)))
    dm = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    r = vt.transpose(1, 2) @ dm @ u.transpose(1, 2)             # (B, 3, 3): p @ r^T ~ g
    scale = (s * torch.diagonal(dm, dim1=1, dim2=2)).sum(-1) / (p ** 2).sum((1, 2)).clamp_min(1e-12) \
        if with_scale else torch.ones_like(d)
    aligned = scale[:, None, None] * p @ r.transpose(1, 2) + mg
    cosang = ((torch.diagonal(r, dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(-1, 1)
    return aligned, torch.rad2deg(torch.acos(cosang))


class Stats:
    def __init__(self):
        self.d = {}

    def add(self, name, **vals):
        e = self.d.setdefault(name, {})
        for k, v in vals.items():
            e.setdefault(k, []).append(v.detach().float().cpu().reshape(-1))

    def summary(self):
        out = {}
        for name, e in self.d.items():
            out[name] = {k: float(torch.cat(v).mean()) for k, v in e.items()}
            if 'rot_deg' in e:
                r = torch.cat(e['rot_deg'])
                out[name]['pct_rot_gt30'] = float((r > 30).float().mean() * 100)
                out[name]['pct_rot_gt90'] = float((r > 90).float().mean() * 100)
            out[name]['n'] = int(torch.cat(next(iter(e.values()))).numel())
        return out


def mesh_metrics(out, gt_mesh, reg, gt17):
    """MPJPE17 / PA / MPVPE (mm) va goc xoay toan than (do) theo tung mau."""
    mesh = out['smpl_mesh_cam']
    jp = h36m_from_mesh(mesh, reg)
    jg = h36m_from_mesh(gt_mesh, reg)
    rp, rg = jp[:, 0:1], jg[:, 0:1]
    jp, jg_r = jp - rp, jg - rg
    mpjpe = torch.norm(jp - jg_r, dim=-1).mean(-1) * 1000
    mpvpe = torch.norm((mesh - rp) - (gt_mesh - rg), dim=-1).mean(-1) * 1000
    al, ang = procrustes(jp, jg_r, with_scale=True)
    pa = torch.norm(al - jg_r, dim=-1).mean(-1) * 1000
    return dict(mpjpe=mpjpe, pa=pa, mpvpe=mpvpe, rot_deg=ang)


def in_err(pose, gt17):
    return torch.norm(pose - gt17, dim=-1).mean(-1) * 1000


def corrupt_noise(gt17, sigma_m, gen):
    n = torch.randn(gt17.shape, device=gt17.device, generator=gen) * sigma_m
    n[:, 0] = 0
    return gt17 + n


def corrupt_limb_flip(gt17, gen):
    """Moi mau: chon 1 chi ngau nhien, phan xa do sau ca chi quanh khop goc (2D ~ khong doi)."""
    q = gt17.clone()
    pick = torch.randint(0, len(LIMBS), (q.shape[0],), device=q.device, generator=gen)
    for li, (a, b, c) in enumerate(LIMBS):
        m = (pick == li)
        if m.any():
            za = q[m, a, 2:3]
            q[m, b, 2] = (2 * za - q[m, b, 2:3]).squeeze(-1)
            q[m, c, 2] = (2 * za - q[m, c, 2:3]).squeeze(-1)
    return q


def depth_reflect(p):
    q = p.clone()
    q[..., 2] = -q[..., 2]
    return q


def attn_stats(layers, gt2d_n, valid):
    """layers: list block -> 'attn_joint_to_img' (B, h, J, HW). Tra ve list (entropy chuan hoa, do lech tam / canh crop)."""
    res = []
    for blk in layers:
        a = blk['attn_joint_to_img'].mean(1)                     # (B, J, HW)
        hw = a.shape[-1]
        s = int(round(hw ** 0.5))
        cen = RGBJointFusion._cell_centers(s, s, a.device, a.dtype)[0]   # (HW, 2)
        loc = a @ cen                                            # (B, J, 2) [-1, 1]
        ent = -(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(-1) / np.log(hw)   # (B, J)
        if gt2d_n is not None:
            dist = torch.norm(loc - gt2d_n, dim=-1) / 2.0
            dist = (dist * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        else:
            dist = torch.full(ent.shape[:1], float('nan'), device=a.device)
        res.append((ent.mean(-1), dist))
    return res


class patched:
    """Tam thoi thay forward cua 1 module bang ham tra ve gia tri co dinh."""
    def __init__(self, module, fn):
        self.m, self.fn = module, fn

    def __enter__(self):
        self.m.forward = self.fn

    def __exit__(self, *a):
        del self.m.forward


def main():
    out_dir = Path(args.out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        print(f"[LOI] Thu muc ket qua khong rong: {out_dir}")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg=''):
        print(msg)
        lines.append(msg)
        (out_dir / 'log.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    teacher_ckpt = args.teacher_ckpt or cfg.MODEL.get('TEACHER', '')
    if not teacher_ckpt:
        print('[LOI] Can --teacher_ckpt hoac cfg.MODEL.TEACHER')
        sys.exit(1)
    hr('Nap checkpoint')
    teacher, _ = build_and_load('teacher', teacher_ckpt)
    student = build_and_load('student', args.student_ckpt)[0] if args.student_ckpt else None

    test_name = args.set or cfg.DATASET.test_list[0]
    use_gt2d = test_name.startswith('3dpw')
    if use_gt2d:
        cfg.DATASET.pw3d_test_gt2d = True
    ds = get_test_dataset(test_name, None)
    gen_l = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, generator=gen_l,
                        num_workers=args.workers, pin_memory=False)
    reg = torch.as_tensor(ds.h36m_joint_regressor, dtype=torch.float32, device=DEVICE)
    sigmas = [float(x) for x in args.noise_mm.split(',') if x.strip()]
    log(f'cfg={args.cfg} | teacher={teacher_ckpt} | student={args.student_ckpt or "-"} | set={test_name} '
        f'| n<= {args.max_batches * args.batch_size} | noise_mm={sigmas}')

    st = Stats()
    att = {}
    with torch.no_grad():
        for bi, (inputs, targets, meta) in enumerate(loader):
            if bi >= args.max_batches:
                break
            gen = torch.Generator(device=DEVICE).manual_seed(args.seed * 100003 + bi)
            img = inputs['img'].to(DEVICE).float()
            j2 = inputs['joints'].to(DEVICE).float()
            jm = inputs['joints_mask'].to(DEVICE).float()
            gt_mesh = targets['smpl_mesh_cam'].to(DEVICE).float()
            gt17 = h36m_from_mesh(gt_mesh, reg)
            gt17 = gt17 - gt17[:, 0:1]
            gt2d_n, valid2d = None, None
            if use_gt2d and 'orig_joint_img' in targets:
                g = targets['orig_joint_img'][..., :2].to(DEVICE).float()
                gt2d_n = torch.stack([g[..., 0] / cfg.output_hm_shape[2] * 2 - 1,
                                      g[..., 1] / cfg.output_hm_shape[1] * 2 - 1], -1)
                valid2d = ((gt2d_n.abs() <= 1).all(-1)).float()

            def run(net, joints_in, feats, mean=False):
                f = feats.mean(0, keepdim=True).expand_as(feats).contiguous() if mean else feats
                return net.smpl_model(joints=joints_in, img_feats=f, is_train=False, return_features=True,
                                      **net._pe2d_kwargs(j2, jm))

            # ---------------- Teacher ----------------
            feat_t, _ = teacher.get_image_features(img)
            t_gt = run(teacher, gt17, feat_t)
            st.add('A. Teacher | pose GT | anh that', in_err=in_err(gt17, gt17), **mesh_metrics(t_gt, gt_mesh, reg, gt17))
            st.add('A. Teacher | pose GT | anh TB', in_err=in_err(gt17, gt17),
                   **mesh_metrics(run(teacher, gt17, feat_t, True), gt_mesh, reg, gt17))
            # loi tung khop cua Teacher (san)
            jt = h36m_from_mesh(t_gt['smpl_mesh_cam'], reg)
            jt = jt - jt[:, 0:1]
            st.add('A. Teacher | loi tung khop', **{f'j{j:02d}': torch.norm(jt[:, j] - gt17[:, j], dim=-1) * 1000
                                                    for j in range(17)})
            for blk_i, (ent, dist) in enumerate(attn_stats(t_gt['feat_layers'], gt2d_n, valid2d)):
                st.add(f'E. attn Teacher(GT) block{blk_i}', entropy=ent, tam_lech_crop=dist)

            corrupts = [(f'nhieu Gauss {s:.0f}mm', corrupt_noise(gt17, s / 1000.0, gen)) for s in sigmas]
            corrupts += [('lat do sau 1 chi', corrupt_limb_flip(gt17, gen)),
                         ('phan xa do sau toan than', depth_reflect(gt17))]
            for cname, pin in corrupts:
                e = in_err(pin, gt17)
                st.add(f'B. Teacher | {cname} | anh that', in_err=e, **mesh_metrics(run(teacher, pin, feat_t), gt_mesh, reg, gt17))
                st.add(f'B. Teacher | {cname} | anh TB', in_err=e,
                       **mesh_metrics(run(teacher, pin, feat_t, True), gt_mesh, reg, gt17))

            # ---------------- Student ----------------
            if student is not None:
                feat_s, _ = student.get_image_features(img)
                if getattr(student, 'use_diff', False):
                    cam_bbox = inputs.get('cam_bbox', torch.zeros(img.shape[0], 3)).to(DEVICE).float()
                    lifted = student.lift_diffusion(j2, jm, cam_bbox)
                elif getattr(student, 'use_pmce', False):
                    jr = inputs.get('joints_raw', j2).to(DEVICE).float()
                    lifted = student.lift_pmce(jr, jm, student.get_image_features(img)[1])
                else:
                    lifted = student.lift_2d_to_3d(j2, joints_mask=jm)
                lifted = (lifted - lifted[:, 0:1]).float()
                e_l = in_err(lifted, gt17)
                s_l = run(student, lifted, feat_s)
                st.add('C. Student | pose lifter | anh that', in_err=e_l, **mesh_metrics(s_l, gt_mesh, reg, gt17))
                st.add('C. Student | pose lifter | anh TB', in_err=e_l,
                       **mesh_metrics(run(student, lifted, feat_s, True), gt_mesh, reg, gt17))
                s_g = run(student, gt17, feat_s)
                st.add('C. Student | pose GT | anh that', in_err=in_err(gt17, gt17), **mesh_metrics(s_g, gt_mesh, reg, gt17))
                st.add('C. Teacher | pose lifter | anh that', in_err=e_l,
                       **mesh_metrics(run(teacher, lifted, feat_t), gt_mesh, reg, gt17))
                for blk_i, (ent, dist) in enumerate(attn_stats(s_l['feat_layers'], gt2d_n, valid2d)):
                    st.add(f'E. attn Student(lifter) block{blk_i}', entropy=ent, tam_lech_crop=dist)
                for blk_i, (ent, dist) in enumerate(attn_stats(s_g['feat_layers'], gt2d_n, valid2d)):
                    st.add(f'E. attn Student(GT) block{blk_i}', entropy=ent, tam_lech_crop=dist)

                # ---- D. activation patching Teacher -> Student (pose lifter) ----
                fus = student.smpl_model.fusion
                tok_T = t_gt['feat_joint_in']
                with patched(fus.projector_student, lambda *a, **k: tok_T):
                    st.add('D. Student + token khop SAU BO MA HOA cua Teacher', in_err=e_l,
                           **mesh_metrics(run(student, lifted, feat_s), gt_mesh, reg, gt17))

                def fake_fus(joint_out, img_out):
                    d = {'img_out': img_out, 'joint_out': joint_out, 'joint_tok': s_l['feat_joint_in'],
                         'img_tok': s_l['feat_img_in'],
                         'concat_feat': torch.cat([img_out.mean(1), joint_out.mean(1)], 1), 'layers': []}
                    return lambda *a, **k: d
                for pname, jo, io in (('token khop SAU FUSION', t_gt['feat_joint'], s_l['feat_img']),
                                      ('token anh SAU FUSION', s_l['feat_joint'], t_gt['feat_img']),
                                      ('ca hai SAU FUSION', t_gt['feat_joint'], t_gt['feat_img'])):
                    with patched(fus, fake_fus(jo, io)):
                        st.add(f'D. Student + {pname} cua Teacher', in_err=e_l,
                               **mesh_metrics(run(student, lifted, feat_s), gt_mesh, reg, gt17))
            if (bi + 1) % 20 == 0:
                print(f'  ... {bi + 1} batch')

    S = st.summary()
    hr('KET QUA (mm; rot = goc xoay toan than sai, do)')
    log(f"{'dieu kien':62s} {'n':>6s} {'pose vao':>8s} {'MPJPE':>7s} {'PA':>7s} {'MPVPE':>7s} {'rot':>6s} {'%>30':>5s} {'%>90':>5s}")
    for name, v in S.items():
        if 'mpjpe' not in v:
            continue
        log(f"{name:62s} {v['n']:6d} {v.get('in_err', float('nan')):8.1f} {v['mpjpe']:7.2f} {v['pa']:7.2f} "
            f"{v['mpvpe']:7.2f} {v['rot_deg']:6.1f} {v['pct_rot_gt30']:5.1f} {v['pct_rot_gt90']:5.1f}")
    jn = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top',
          'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
    pj = S.get('A. Teacher | loi tung khop', {})
    if pj:
        log('\nA. Teacher pose GT - loi tung khop (mm): ' + ', '.join(f'{jn[j]}={pj[f"j{j:02d}"]:.1f}' for j in range(17)))
    log('\nB. Anh giup Teacher bao nhieu khi pose sai (anh TB - anh that, MPJPE mm):')
    base = S.get('A. Teacher | pose GT | anh TB', {}).get('mpjpe', float('nan')) - \
        S.get('A. Teacher | pose GT | anh that', {}).get('mpjpe', float('nan'))
    log(f'   pose GT: {base:+.2f}')
    for name in S:
        if name.startswith('B. Teacher') and name.endswith('anh that'):
            mname = name.replace('anh that', 'anh TB')
            if mname in S:
                r, m = S[name], S[mname]
                log(f"   {name[13:-11]:28s} pose vao {r['in_err']:6.1f} -> mesh {r['mpjpe']:6.2f} | anh giup {m['mpjpe'] - r['mpjpe']:+6.2f}")
    if student is not None:
        s0 = S['C. Student | pose lifter | anh that']['mpjpe']
        t0 = S['A. Teacher | pose GT | anh that']['mpjpe']
        log(f'\nD. Ghep kich hoat: khoang cach Student(lifter) {s0:.2f} - Teacher(GT) {t0:.2f} = {s0 - t0:.2f} mm')
        for name in S:
            if name.startswith('D.'):
                v = S[name]['mpjpe']
                log(f'   {name[3:]:50s} {v:7.2f} mm | lap {100 * (s0 - v) / max(s0 - t0, 1e-6):6.1f}% khoang cach')
    log('\nE. Attention khop->anh (entropy chuan hoa 0..1; tam lech = khoang cach tam attention - GT 2D khop, / canh crop):')
    for name, v in S.items():
        if name.startswith('E.'):
            log(f"   {name[3:]:40s} entropy {v['entropy']:.3f} | tam lech {v['tam_lech_crop']:.3f}")
    (out_dir / 'summary.json').write_text(json.dumps(S, indent=2, ensure_ascii=False), encoding='utf-8')
    log(f'\nDa luu: {out_dir}')


if __name__ == '__main__':
    main()
