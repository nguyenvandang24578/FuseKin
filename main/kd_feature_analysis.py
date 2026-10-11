"""
kd_feature_analysis.py — Dac trung Student co THAT SU giong Teacher (nho KD) khong? (eval-only)
===============================================================================================

So sanh dac trung SAU FUSION (diem KD: feat_joint (B,J,C), feat_global (B,2C)) cua tung Student voi
Teacher tren CUNG mau val 3DPW. Teacher nhan GT 3D, Student nhan joint MotionBERT (dung nhu luc train).
Nen chay cho ca Student KD (E1) lan Student KHONG KD (init_only, scratch) de co doi chung.

4 phep do:
  [1] Cosine theo mau (sau LayerNorm, KHONG qua projector): cung mau vs mau KHAC (xao tron).
      KD that -> cung mau cao, mau khac thap (giong RIENG tung mau, khong chi giong "trung binh").
  [2] Truy hoi (retrieval): voi moi Student, Teacher gan nhat trong N mau co dung la chinh mau do?
      Top-1 / Top-5 (ngau nhien = 1/N). Do muc "Student biet minh la mau nao theo cach cua Teacher".
  [3] CKA tuyen tinh (Kornblith 2019): giong nhau o muc PHAN PHOI / hinh hoc bieu dien (0..1).
  [4] DOI DECODER (phep thu manh nhat): dua dac trung fusion cua Student vao DECODER DONG BANG cua
      Teacher (HyperGCN -> head -> VPoser -> SMPL), do MPJPE. Neu dac trung Student nam trong "khong
      gian" cua Teacher thi decoder Teacher van doc duoc -> MPJPE gan voi decoder cua chinh Student.
      Student khong KD -> decoder Teacher doc ra rac.
  (+ neu checkpoint co 'projector_state_dict': cosine sau projector KD, dung khoang cach da toi uu)

Cach chay (tu thu muc goc repo, can CUDA):
    python main/kd_feature_analysis.py --cfg ./config/train_student_kd_e1.yml \
        --teacher_ckpt <teacher_gt>/checkpoint/best.pth.tar \
        --student_ckpts experiment/student_kd_e1/checkpoint/best.pth.tar \
                        experiment/student_init_only/checkpoint/best.pth.tar \
                        <scratch>/checkpoint/best.pth.tar \
        --names e1 init_only scratch [--max_batches 60] [--batch_size 32] [--gpu 0]

Cac so MPJPE o day la H36M-17 root-relative (thuoc do chan doan), CHI so sanh TUONG DOI giua cac dong.
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Do muc dong nhat dac trung Student-Teacher sau fusion')
parser.add_argument('--cfg', type=str, required=True, help='config Student (dung MotionBERT + tap test)')
parser.add_argument('--teacher_ckpt', type=str, required=True)
parser.add_argument('--student_ckpts', type=str, nargs='+', required=True)
parser.add_argument('--names', type=str, nargs='*', default=None)
parser.add_argument('--max_batches', type=int, default=60)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

from core.config import cfg, update_config
update_config(args.cfg)
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import __init_path  # noqa: F401
from models.ARTS import ARTS
from core.base import load_model_weights, _PickleShim
from utils.jotr_dataset import get_test_dataset
from utils.transforms import rot6d_to_axis_angle

if not torch.cuda.is_available():
    print("[LOI] Can CUDA.")
    sys.exit(1)
DEV = torch.device('cuda')
torch.manual_seed(args.seed)
np.random.seed(args.seed)

names = args.names if args.names else [f's{i}' for i in range(len(args.student_ckpts))]
if len(names) != len(args.student_ckpts):
    sys.exit(f"[LOI] --names ({len(names)}) khac so --student_ckpts ({len(args.student_ckpts)})")


def hr(t=''):
    print('\n' + '=' * 78)
    if t:
        print(t)
        print('=' * 78)


def read_state(path):
    ck = torch.load(path, map_location='cpu', pickle_module=_PickleShim, weights_only=False)
    return ck if isinstance(ck, dict) else {}


def build(mode, path):
    """Dung ARTS theo mode ('teacher'/'student'), dung kien truc (probe) theo checkpoint, nap trong so."""
    if not os.path.isfile(path):
        sys.exit(f"[LOI] Khong thay checkpoint: {path}")
    ck = read_state(path)
    sd = ck.get('model_state_dict', ck.get('state_dict', ck))
    keys = list(sd.keys()) if isinstance(sd, dict) else []
    is_student = any('fusion.projector_student.' in k for k in keys)
    is_teacher = any('fusion.joint_proj.' in k for k in keys)
    if (mode == 'student' and is_teacher and not is_student) or (mode == 'teacher' and is_student):
        sys.exit(f"[LOI] {path} khong phai checkpoint {mode.upper()}.")
    old = (cfg.MODEL.name, cfg.MODEL.get('teacher_lift_alpha_max', 0.0), cfg.MODEL.get('img_probe', False))
    cfg.MODEL.name = mode
    if mode == 'teacher':
        cfg.MODEL.teacher_lift_alpha_max = 0.0
        cfg.MODEL.img_probe = any('probe_' in k for k in keys)
    m = ARTS(num_joint=17, embed_dim=cfg.MODEL.hpe_dim).to(DEV)
    cfg.MODEL.name, cfg.MODEL.teacher_lift_alpha_max, cfg.MODEL.img_probe = old
    m = load_model_weights(m, path)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m, ck.get('projector_state_dict')


def build_projector(psd):
    """Tai tao projector KD (LN khong affine -> Linear) tu 'projector_state_dict' (neu co)."""
    if not psd:
        return None
    mods = {}
    for key in ('joint', 'global'):
        w = psd.get(f'{key}.1.weight')
        if w is None:
            return None
        C = w.shape[0]
        lin = torch.nn.Linear(C, C)
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(psd[f'{key}.1.bias'])
        mods[key] = torch.nn.Sequential(torch.nn.LayerNorm(C, elementwise_affine=False), lin).to(DEV).eval()
    return mods


@torch.no_grad()
def decode(m, joint_out, global_ft):
    """Phan SAU fusion cua FusionPose2Mesh (nhanh hypergcn), giong het forward -> mesh_cam (root-rel)."""
    B = joint_out.shape[0]
    pose_token = m.pose_embed(m.init_pose.expand(B, -1, -1))
    if m.pose_pe is not None:
        pose_token = pose_token + m.pose_pe.weight.unsqueeze(0)
    idx = torch.arange(m.num_joint, device=joint_out.device)
    ctx = m.norm(joint_out) + m.node_pe(idx)
    ctx = m.spatial_hypers(ctx.unsqueeze(1)).squeeze(1)
    pose_feat = m.pose_context_attn(pose_token, ctx, ctx)
    root_pose = rot6d_to_axis_angle(m.root_pose_head(pose_feat[:, 0, :]))
    body_pose = m.vposer(m.body_pose_head(pose_feat.mean(dim=1)))
    full_pose = torch.cat([root_pose, body_pose], dim=1)
    cam_trans = m.get_camera_trans(m.cam_head(global_ft))
    shape_emb = m.shape_embed(m.init_shape.expand(B, -1)).unsqueeze(1)
    shape_token = m.shape_token.weight.unsqueeze(0).expand(B, 1, -1) + shape_emb
    g = global_ft.unsqueeze(1)
    shape_param = m.shape_head(m.fuse_shape(shape_token, g, g)).reshape(B, -1)
    _, _, mesh_cam, _ = m.get_coord(full_pose, shape_param, cam_trans)
    return mesh_cam


def h36m(mesh, reg):
    j = torch.einsum('jk,bkc->bjc', reg, mesh)
    return j - j[:, 0:1]


def mpjpe(a, b):
    return (torch.norm(a - b, dim=-1).mean(-1) * 1000).cpu()


def ln(x):
    return F.layer_norm(x, x.shape[-1:])


def linear_cka(X, Y):
    """X, Y: (N, D) numpy. CKA tuyen tinh dang Gram (N x N)."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    K, L = X @ X.T, Y @ Y.T
    return float((K * L).sum() / (np.linalg.norm(K) * np.linalg.norm(L) + 1e-12))


def retrieval(S, T, ks=(1, 5)):
    """S, T: (N, D) torch (da chuan hoa). Ty le Teacher gan nhat (cosine) dung la chinh mau do."""
    sim = S @ T.t()
    order = sim.argsort(dim=1, descending=True)
    gt = torch.arange(S.shape[0])[:, None]
    return {k: float((order[:, :k] == gt).any(1).float().mean()) for k in ks}


def main():
    hr("Dung va nap model")
    teacher, _ = build('teacher', args.teacher_ckpt)
    students = []
    for n, p in zip(names, args.student_ckpts):
        s, psd = build('student', p)
        students.append((n, s, build_projector(psd)))
        print(f"  [{n}] projector KD: {'co' if students[-1][2] is not None else 'khong'}")

    dataset = get_test_dataset(cfg.DATASET.test_list[0], None)
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=gen,
                        num_workers=args.workers, pin_memory=True)
    reg = torch.as_tensor(dataset.h36m_joint_regressor, dtype=torch.float32, device=DEV)

    # bo nho: dac trung (CPU) + loi
    T_j, T_g = [], []
    S = {n: {'j': [], 'g': [], 'pj': [], 'pg': [], 'own': [], 'swap': []} for n, _, _ in students}
    t_err, sanity = [], []

    hr(f"Chay {args.max_batches} batch x {args.batch_size} mau tren {cfg.DATASET.test_list[0]}")
    with torch.no_grad():
        for bi, (inputs, targets, meta) in enumerate(loader):
            if bi >= args.max_batches:
                break
            img = inputs['img'].to(DEV).float()
            j2d = inputs['joints'].to(DEV).float()
            jm = inputs['joints_mask'].to(DEV).float()
            gt = h36m(targets['smpl_mesh_cam'].to(DEV).float(), reg)

            ft, _ = teacher.get_image_features(img)
            to = teacher.smpl_model(joints=gt, img_feats=ft, is_train=False, return_features=True)
            t_err.append(mpjpe(h36m(to['smpl_mesh_cam'], reg), gt))
            if bi == 0:   # kiem tra ham decode() tai tao dung forward
                rec = decode(teacher.smpl_model, to['feat_joint'], to['feat_global'])
                sanity.append(float((rec - to['smpl_mesh_cam']).abs().max()))
            T_j.append(ln(to['feat_joint']).cpu())
            T_g.append(ln(to['feat_global']).cpu())

            for n, s, proj in students:
                fs, _ = s.get_image_features(img)
                lift = s.lift_2d_to_3d(j2d, joints_mask=jm)
                lift = lift - lift[:, 0:1]
                so = s.smpl_model(joints=lift, img_feats=fs, is_train=False, return_features=True)
                S[n]['own'].append(mpjpe(h36m(so['smpl_mesh_cam'], reg), gt))
                S[n]['swap'].append(mpjpe(h36m(decode(teacher.smpl_model, so['feat_joint'], so['feat_global']), reg), gt))
                S[n]['j'].append(ln(so['feat_joint']).cpu())
                S[n]['g'].append(ln(so['feat_global']).cpu())
                if proj is not None:
                    S[n]['pj'].append(proj['joint'](so['feat_joint']).cpu())
                    S[n]['pg'].append(proj['global'](so['feat_global']).cpu())
            if (bi + 1) % 10 == 0:
                print(f"  ... {bi + 1}/{args.max_batches}")

    Tj, Tg = torch.cat(T_j), torch.cat(T_g)                       # (N,J,C), (N,2C)
    N = Tg.shape[0]
    t_err = torch.cat(t_err).mean().item()
    perm = torch.randperm(N)
    while N > 1 and (perm == torch.arange(N)).any():              # dam bao moi mau ghep voi mau KHAC
        perm = torch.randperm(N)
    print(f"\n  Kiem tra decode(): sai khac max so voi forward Teacher = {sanity[0]:.2e} (phai ~0)")

    rows = []
    for n, _, _ in students:
        Sj, Sg = torch.cat(S[n]['j']), torch.cat(S[n]['g'])
        cj_same = F.cosine_similarity(Sj, Tj, dim=-1).mean().item()
        cj_diff = F.cosine_similarity(Sj, Tj[perm], dim=-1).mean().item()
        cg_same = F.cosine_similarity(Sg, Tg, dim=-1).mean().item()
        cg_diff = F.cosine_similarity(Sg, Tg[perm], dim=-1).mean().item()
        r_g = retrieval(F.normalize(Sg, dim=-1), F.normalize(Tg, dim=-1))
        r_j = retrieval(F.normalize(Sj.flatten(1), dim=-1), F.normalize(Tj.flatten(1), dim=-1))
        cka_j = linear_cka(Sj.flatten(1).double().numpy(), Tj.flatten(1).double().numpy())
        cka_g = linear_cka(Sg.double().numpy(), Tg.double().numpy())
        pj = pg = None
        if S[n]['pj']:
            pj = F.cosine_similarity(torch.cat(S[n]['pj']), Tj, dim=-1).mean().item()
            pg = F.cosine_similarity(torch.cat(S[n]['pg']), Tg, dim=-1).mean().item()
        rows.append(dict(n=n, cj_same=cj_same, cj_diff=cj_diff, cg_same=cg_same, cg_diff=cg_diff,
                         rg1=r_g[1], rg5=r_g[5], rj1=r_j[1], rj5=r_j[5], cka_j=cka_j, cka_g=cka_g,
                         pj=pj, pg=pg,
                         own=torch.cat(S[n]['own']).mean().item(), swap=torch.cat(S[n]['swap']).mean().item()))

    hr(f"[1] COSINE Student-Teacher (sau LayerNorm), N = {N} mau")
    print(f"  {'ban':<12}{'joint cung':>11}{'joint khac':>11}{'chenh':>8}   {'global cung':>12}{'global khac':>12}{'chenh':>8}")
    for r in rows:
        print(f"  {r['n']:<12}{r['cj_same']:11.3f}{r['cj_diff']:11.3f}{r['cj_same'] - r['cj_diff']:+8.3f}   "
              f"{r['cg_same']:12.3f}{r['cg_diff']:12.3f}{r['cg_same'] - r['cg_diff']:+8.3f}")
    if any(r['pj'] is not None for r in rows):
        print("  Sau projector KD (khoang cach da toi uu luc train):")
        for r in rows:
            if r['pj'] is not None:
                print(f"  {r['n']:<12} joint {r['pj']:.3f} | global {r['pg']:.3f}")

    hr(f"[2] TRUY HOI: Teacher gan nhat co dung la CHINH mau do? (ngau nhien: top1 = {1 / N:.4f}, top5 = {5 / N:.4f})")
    print(f"  {'ban':<12}{'global top1':>12}{'global top5':>12}{'joint top1':>12}{'joint top5':>12}")
    for r in rows:
        print(f"  {r['n']:<12}{r['rg1']:12.3f}{r['rg5']:12.3f}{r['rj1']:12.3f}{r['rj5']:12.3f}")

    hr("[3] CKA tuyen tinh voi Teacher (0 = khong lien quan, 1 = cung hinh hoc bieu dien)")
    print(f"  {'ban':<12}{'CKA joint':>11}{'CKA global':>12}")
    for r in rows:
        print(f"  {r['n']:<12}{r['cka_j']:11.3f}{r['cka_g']:12.3f}")

    hr("[4] DOI DECODER: dac trung fusion Student -> decoder DONG BANG cua Teacher (MPJPE H36M-17, mm)")
    print(f"  Teacher (GT 3D) voi decoder cua chinh no: {t_err:.2f}")
    print(f"  {'ban':<12}{'decoder Student':>16}{'decoder Teacher':>16}{'chenh':>9}")
    for r in rows:
        print(f"  {r['n']:<12}{r['own']:16.2f}{r['swap']:16.2f}{r['swap'] - r['own']:+9.2f}")

    hr("CACH DOC")
    print("  KD THAT SU co tac dung neu ban KD (vd e1) so voi doi chung (init_only/scratch):")
    print("   - [1] cosine CUNG mau cao hon ro VA chenh (cung - khac) lon hon -> giong rieng tung mau.")
    print("   - [2] truy hoi top1/top5 cao hon nhieu muc ngau nhien va cao hon doi chung.")
    print("   - [3] CKA cao hon doi chung.")
    print("   - [4] decoder Teacher doc duoc dac trung Student: 'chenh' nho (vai mm). Doi chung -> chenh lon.")
    print("  Neu e1 ~ doi chung o ca 4 phep -> cai thien MPJPE KHONG den tu viec giong Teacher.")


if __name__ == '__main__':
    main()
