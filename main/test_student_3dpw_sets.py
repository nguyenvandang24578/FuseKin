"""
main/test_student_3dpw_sets.py
Danh gia 1 checkpoint Student / Teacher FuseKin (best.pth.tar cua main/train.py) tren nhieu tap 3DPW:
3dpw (test), 3dpw-oc, 3dpw-pc. Dung DUNG ham danh gia luc train (utils.jotr_evaluation.evaluate_3dpw_subset):
MPJPE / PA-MPJPE (14 khop) va MPVPE, mm.

RO RI OC / PC: 3dpw-oc va 3dpw-pc gom ca sequence cua 3dpw-train / val. Model FuseKin train tren 3dpw-train
-> so tren TOAN BO OC / PC bi ro ri (da thay sequence luc train). --ocpc_mode:
  test : chi sequence thuoc 3dpw test (DATASET.pw3d_ocpc_test_only = True) -> so cong bang
  all  : toan bo OC / PC (giong giao thuc JOTR, nhung JOTR KHONG train 3dpw -> khong so truc tiep)
  both : (mac dinh) in ca hai

Chan doan GOPY.MD (2026-11-17, mac dinh tat, khong doi ket qua khi khong bat):
  --view_rot fwd|inv : D1 - xoay mesh / ctx / token Student quanh goc bang view_rot(cam_bbox dx, dy) (fwd) hoac
                       nghich dao (inv) truoc khi do -> kiem tra Student (nhat la --drop3d) co du doan trong he camera
                       'nhin thang tam bbox' thay vi camera that hay khong. LIFT khong xoay.
  --blend a[,b,...]  : D2 - tron khop 3D Student voi lifter TRUOC IK: j3d = w*lift + (1-w)*ctx (MODEL.v9_test_blend);
                       nhieu gia tri -> quet. --blend_mode global | opmask (khop OpenPose sot -> w = 1).
  --oracle_lc        : D2 - them tang 'oracle_lc' = moi khop chon lift hoac ctx GAN GT hon (CHI do tran, khong bao cao).
Chay:
  python main/test_student_3dpw_sets.py --cfg ./config/train_student_kd_e1_v2_diff.yml --gpu 0 \
      --ckpt ./experiment/student_kd_e1_v2_diff/checkpoint/best.pth.tar
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True, help='yml dung de train checkpoint (cung kien truc, cung lifter)')
parser.add_argument('--ckpt', type=str, required=True, help='best.pth.tar')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--sets', type=str, default='3dpw,3dpw-oc,3dpw-pc')
parser.add_argument('--ocpc_mode', type=str, default='both', choices=['test', 'all', 'both'])
parser.add_argument('--batch_size', type=int, default=0, help='0 = cfg.TEST.batch_size')
parser.add_argument('--drop3d', action='store_true',
                    help='V9/V10/V11 chan doan: bo HET token 3D lifter (miss3d) luc test, chi con anh + 2D')
parser.add_argument('--lifter_ckpt', type=str, default='',
                    help='Ghi de checkpoint ImgLifter luc test (giu nguyen trong so Student, khong bi state_dict Student de lai)')
parser.add_argument('--view_rot', type=str, default='none', choices=['none', 'fwd', 'inv'],
                    help='D1: xoay dau ra Student bang view_rot(cam_bbox) (fwd) / nghich dao (inv)')
parser.add_argument('--blend', type=str, default='', help="D2: vd '0,0.5,1' (rong = tat)")
parser.add_argument('--blend_mode', type=str, default='global', choices=['global', 'opmask'])
parser.add_argument('--oracle_lc', action='store_true', help='D2: tang oracle chon lift / ctx tung khop theo GT')
parser.add_argument('--align_rot', type=str, default='none',
                    help="Can xoay cung SO(3) (Kabsch, 0%% GT) mesh theo khop du doan: 'none' | 'ctx' | 'torso_ctx' | 'lift' "
                         "(co the truyen nhieu gia tri cach dau phay, vd 'none,ctx,torso_ctx,lift')")
args = parser.parse_args()
VALID_ALIGN_ROT = ('none', 'ctx', 'torso_ctx', 'lift')
ALIGN_MODES = [x.strip() for x in args.align_rot.split(',') if x.strip()] or ['none']
for _m in ALIGN_MODES:
    if _m not in VALID_ALIGN_ROT:
        raise ValueError(f"--align_rot khong hop le: '{_m}' (chon trong {VALID_ALIGN_ROT})")
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from torch.utils.data import DataLoader
from core.config import cfg, update_config
update_config(args.cfg)
if args.lifter_ckpt:
    cfg.MODEL.imglift_checkpoint = args.lifter_ckpt
if args.drop3d:
    cfg.MODEL.v9_test_drop3d = True
    cfg.MODEL.v10_test_drop3d = True
    cfg.MODEL.v11_test_drop3d = True

import __init_path  # noqa: F401
from models.ARTS import get_model
from utils.jotr_dataset import get_test_dataset
from utils.jotr_evaluation import evaluate_3dpw_subset
from funcs_utils import check_data_pararell


def view_rot_torch(dx, dy):
    """Ban torch (theo lo) cua utils.cam_bbox.view_rot: (B,), (B,) -> (B, 3, 3). Xoay nho nhat dua truc z camera ao
    (nhin thang tam bbox) thanh tia d = (dx, dy, 1) toi nguoi: pose_camera_that = R @ pose_camera_ao."""
    d = torch.stack([dx, dy, torch.ones_like(dx)], -1).double()
    d = d / d.norm(dim=-1, keepdim=True)
    v = torch.stack([-d[:, 1], d[:, 0], torch.zeros_like(d[:, 0])], -1)               # z x d
    s, c = v.norm(dim=-1), d[:, 2]
    K = torch.zeros(dx.shape[0], 3, 3, dtype=torch.float64, device=dx.device)
    K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
    K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
    coef = ((1 - c) / (s * s).clamp_min(1e-16))[:, None, None]
    R = torch.eye(3, dtype=torch.float64, device=dx.device)[None] + K + K @ K * coef
    R[s < 1e-8] = torch.eye(3, dtype=torch.float64, device=dx.device)
    return R.float()


TORSO_ALIGN_IDX = (0, 1, 4, 7, 8, 11, 14)   # Pelvis, R_Hip, L_Hip, Torso, Neck, L_Shoulder, R_Shoulder


def kabsch_rot_so3(src, dst):
    """Tim ma tran xoay thuan tuy R in SO(3) (det = +1, khong co gian, xoay quanh goc Pelvis da tru san)
    cuc tieu hoa ||src @ R^T - dst||_F^2. src, dst: (B, K, 3) -> R: (B, 3, 3) float32."""
    X = src.double()
    Y = dst.double()
    H = X.transpose(1, 2) @ Y                                                         # (B, 3, 3)
    U, _, Vh = torch.linalg.svd(H)
    V = Vh.transpose(1, 2)
    d = torch.sign(torch.det(V @ U.transpose(1, 2)))
    d = torch.where(d == 0, torch.ones_like(d), d)
    Z = torch.eye(3, dtype=torch.float64, device=src.device).unsqueeze(0).repeat(X.shape[0], 1, 1)
    Z[:, 2, 2] = d
    R = V @ Z @ U.transpose(1, 2)                                                     # (B, 3, 3) in SO(3)
    return R.float()


def make_post_fn(view_rot_mode, oracle_lc, align_rot_mode='none'):
    """post_fn cho evaluate_3dpw_subset (chan doan D1 / D2 / Kabsch SO(3)). None neu khong bat gi -> danh gia nhu cu."""
    if view_rot_mode == 'none' and not oracle_lc and align_rot_mode == 'none':
        return None

    def post(outputs, inputs, target_mesh, dataset):
        if view_rot_mode != 'none':
            cb = inputs['cam_bbox'].float()                                       # (B, 3) = [dx, dy, size]
            R = view_rot_torch(cb[:, 0], cb[:, 1])
            if view_rot_mode == 'inv':
                R = R.transpose(1, 2)

            def rot(x):                                                           # xoay quanh goc toa do
                return torch.einsum('bij,bnj->bni', R.to(x.dtype), x)
            for k in ('smpl_mesh_cam', 'ctx_3d', 'privileged_3d'):
                if outputs.get(k) is not None:
                    outputs[k] = rot(outputs[k])
            for h in outputs.get('refine_history') or []:
                h['smpl_mesh_cam'] = rot(h['smpl_mesh_cam'])
        if align_rot_mode != 'none' and outputs.get('smpl_mesh_cam') is not None:
            mesh = outputs['smpl_mesh_cam']
            reg = torch.as_tensor(dataset.h36m_joint_regressor, device=mesh.device, dtype=torch.float32)
            mesh_j17 = torch.einsum('jk,bkc->bjc', reg, mesh.float())
            X = mesh_j17 - mesh_j17[:, 0:1]
            ref_key = 'lifted_joints_3d' if align_rot_mode == 'lift' else 'ctx_3d'
            assert outputs.get(ref_key) is not None, f'--align_rot {align_rot_mode} can outputs[{ref_key!r}]'
            T = outputs[ref_key].float()
            Y = T - T[:, 0:1]
            if align_rot_mode == 'torso_ctx':
                R_so3 = kabsch_rot_so3(X[:, TORSO_ALIGN_IDX], Y[:, TORSO_ALIGN_IDX])
            else:
                R_so3 = kabsch_rot_so3(X, Y)
            outputs['smpl_mesh_cam'] = torch.einsum('bij,bvj->bvi', R_so3.to(mesh.dtype), mesh)
        if oracle_lc and outputs.get('ctx_3d') is not None and 'lifted_joints_3d' in outputs:
            reg = torch.as_tensor(dataset.h36m_joint_regressor, device=target_mesh.device, dtype=torch.float32)
            gt = torch.einsum('jk,bkc->bjc', reg, target_mesh.float())
            gt = gt - gt[:, 0:1]
            L = outputs['lifted_joints_3d'].float()
            L = L - L[:, 0:1]
            C = outputs['ctx_3d'].float()
            C = C - C[:, 0:1]
            pick_l = (torch.norm(L - gt, dim=-1) < torch.norm(C - gt, dim=-1))[..., None]
            outputs.setdefault('extra_stages', {})['oracle_lc'] = torch.where(pick_l, L, C)
        return outputs
    return post


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_joint=17, embed_dim=cfg.MODEL.hpe_dim).to(device)   # v8_e2e -> V8Net, khong thi ARTS nhu cu
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    st = check_data_pararell(ck['model_state_dict'])
    if args.lifter_ckpt:
        st = {k: v for k, v in st.items() if not k.startswith(('img_lifter.', 'refiner2d.'))}
        print(f'===> [--lifter_ckpt] giu ImgLifter + Refiner2D tu: {args.lifter_ckpt} (bo qua lifter trong ckpt Student)')
    missing, unexpected = model.load_state_dict(st, strict=False)
    if args.lifter_ckpt:
        missing = [k for k in missing if not k.startswith(('img_lifter.', 'refiner2d.'))]
    print(f'===> nap {args.ckpt} (epoch {ck.get("epoch")}) | thieu {len(missing)} | thua {len(unexpected)} key')
    if missing:
        print(f'     thieu (vai key dau): {missing[:5]}')
    model.eval()
    bs = args.batch_size or cfg.TEST.batch_size

    runs = []   # (ten hien thi, ten tap, ocpc_test_only)
    for name in [s.strip() for s in args.sets.split(',') if s.strip()]:
        if name in ('3dpw-oc', '3dpw-pc'):
            if args.ocpc_mode in ('test', 'both'):
                runs.append((f'{name} (chi seq test)', name, True))
            if args.ocpc_mode in ('all', 'both'):
                runs.append((f'{name} (toan bo, ro ri)', name, False))
        else:
            runs.append((name, name, False))

    blends = [float(x) for x in args.blend.split(',') if x.strip()] or [None]
    if blends != [None]:
        cfg.MODEL.v9_test_blend_mode = args.blend_mode
        if not bool(cfg.MODEL.get('v9_ik', False)):
            print('[canh bao] --blend chi doi mesh khi MODEL.v9_ik True (khop 3D di vao IK)')
    multi_align = len(ALIGN_MODES) > 1 or ALIGN_MODES[0] != 'none'
    runs = [
        (lb + (f' | w={a:g}' if a is not None else '') + (f' | rot={rm}' if multi_align else ''), nm, to, a, rm)
        for rm in ALIGN_MODES for a in blends for lb, nm, to in runs
    ]

    results = []
    for label, name, test_only, blend_a, align_rm in runs:
        cfg.DATASET.pw3d_ocpc_test_only = test_only      # PW3D doc co nay luc tao dataset
        cfg.MODEL.v9_test_blend = -1.0 if blend_a is None else blend_a
        post_fn = make_post_fn(args.view_rot, args.oracle_lc, align_rot_mode=align_rm)
        try:
            ds = get_test_dataset(name, None)
            loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=cfg.DATASET.workers)
            with torch.no_grad():
                m = evaluate_3dpw_subset(model, ds, loader, device=device, with_lifter=True, post_fn=post_fn)
            results.append((label, len(ds), m))
            print(f'[{label}] n={len(ds)} | MPJPE {m["mpjpe"]:.2f} | PA {m["pa_mpjpe"]:.2f} | MPVPE {m["mpvpe"]:.2f}')
            stages = [k[:-len('_mpjpe')] for k in m if k.endswith('_mpjpe') and not k.endswith('_pa_mpjpe')
                      and k not in ('mpjpe', 'pa_mpjpe')]
            if stages:   # chan doan tung tang Student, cung thuoc do 14 khop
                print('    tang: ' + ' | '.join(f'{s} {m[s + "_mpjpe"]:.2f} / PA {m[s + "_pa_mpjpe"]:.2f}'
                                               for s in stages) + f' | mesh_cuoi {m["mpjpe"]:.2f} / PA {m["pa_mpjpe"]:.2f}')
        except Exception as e:
            results.append((label, 0, None))
            print(f'[{label}] LOI: {type(e).__name__}: {e}')

    print('\n' + '=' * 78)
    print(f'{args.ckpt} | epoch {ck.get("epoch")} | {cfg.MODEL.name}'
          + (f' | lifter_ckpt={args.lifter_ckpt}' if args.lifter_ckpt else '')
          + (' | TAT 3D (drop3d: token 3D = miss3d, chi anh + 2D)' if args.drop3d else '')
          + (f' | D1 view_rot {args.view_rot}' if args.view_rot != 'none' else '')
          + (f' | align_rot={args.align_rot}' if multi_align else '')
          + (f' | D2 blend {args.blend} ({args.blend_mode}): j3d = w*lift + (1-w)*ctx truoc IK' if args.blend else '')
          + (' | D2 oracle_lc (chi do tran)' if args.oracle_lc else ''))
    print(f'{"tap":<28} | {"n":>6} | {"MPJPE":>7} | {"PA-MPJPE":>8} | {"MPVPE":>7} | {"LIFT":>7} | {"LIFT-PA":>7}'
          '   (mm, 14 khop)')
    print('-' * 98)
    for label, n, m in results:
        if m is None:
            print(f'{label:<36} | LOI (xem phia tren)')
        else:
            lf = (f' | {m["lift_mpjpe"]:>7.2f} | {m["lift_pa_mpjpe"]:>7.2f}' if 'lift_mpjpe' in m else '')
            print(f'{label:<36} | {n:>6} | {m["mpjpe"]:>7.2f} | {m["pa_mpjpe"]:>8.2f} | {m["mpvpe"]:>7.2f}{lf}')
    print('=' * 98)
    print('LIFT / LIFT-PA: pose 3D dau vao Student (lifter), CUNG thuoc do 14 khop voi mesh (chan doan)')
    print('Chan doan tung tang (MPJPE / PA, 14 khop): lift = dau vao | token = privileged_3d giai ma tu token pose '
          'sau fusion | mesh_it0 = mesh tu query SMPL truoc [D] | mesh_it1.. = sau tung vong [D] | mesh_cuoi')
    for label, n, m in results:
        if m is not None and 'lift_mpjpe' in m:
            stages = [k[:-len('_mpjpe')] for k in m if k.endswith('_mpjpe') and not k.endswith('_pa_mpjpe')
                      and k not in ('mpjpe', 'pa_mpjpe')]
            print(f'{label:<36} | ' + ' -> '.join(f'{s} {m[s + "_mpjpe"]:.2f}' for s in stages)
                  + f' -> mesh_cuoi {m["mpjpe"]:.2f}')


if __name__ == '__main__':
    main()
