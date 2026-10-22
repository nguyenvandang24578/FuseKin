"""
main/oracle_student_diff.py
Do TRAN cua viec "dung anh chon gia thuyet diffusion" trong FuseKin (CHI eval, khong train).

Lay Student da train voi lifter DiffLifter (pose_lifter = 'diffusion'), giu nguyen moi thu, CHI thay pose 3D
dua vao fusion bang cac phuong an sau roi do MPJPE / PA-MPJPE / MPVPE bang dataset.evaluate (giong luc test):
  1. S=1 (nhu luc train)       : pose lifter dung luc train Student (moc)
  2. trung binh K (S, K)        : trung binh K gia thuyet diffusion
  3. oracle ca tu the            : chon 1 gia thuyet co loi nho nhat so voi GT (can tren cua cach A)
  4. oracle tung khop            : moi khop chon gia thuyet tot nhat rieng (can tren cua cach A')
  5. GT 3D                       : lifter hoan hao (can tren cua MOI cai tien lifter)
Kem theo MPJPE cua chinh pose dua vao (17 khop, root-relative) de doi chieu.

Luu y: Student duoc train voi dau vao S=1 (tron, giong trung binh). Gia thuyet le (S=5) "nhieu" hon -> Student
chua quen -> so oracle co the THAP hon tran that su neu Student duoc train cung voi gia thuyet.

Chay:
  python main/oracle_student_diff.py --cfg ./config/train_student_v2_diff.yml --gpu 0 \
      --ckpt ./experiment/student_v2_diff/checkpoint/best.pth.tar
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, required=True)
parser.add_argument('--ckpt', type=str, required=True, help='best.pth.tar cua Student (pose_lifter = diffusion)')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--set', type=str, default='3dpw')
parser.add_argument('--S', type=int, default=5, help='so buoc DDIM khi sinh gia thuyet')
parser.add_argument('--K', type=int, default=20, help='so gia thuyet')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_batches', type=int, default=0, help='>0: chi chay N batch dau (thu nhanh)')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from core.config import cfg, update_config
update_config(args.cfg)

import __init_path  # noqa: F401
from models.ARTS import ARTS
from utils.jotr_dataset import get_test_dataset
from funcs_utils import check_data_pararell

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VARIANTS = ['S=1 (nhu luc train)', f'trung binh K={args.K} (S={args.S})', 'oracle ca tu the',
            'oracle tung khop', 'GT 3D']


def lifter_cond(pose_2d, joints_mask, cam_bbox):
    """Dieu kien DiffLifter (B, 17, 6), giong het ARTS.lift_diffusion."""
    xy = pose_2d[..., :2].float()
    conf = joints_mask.float().reshape(xy.shape[0], xy.shape[1], 1)
    xy = xy * (conf > 0)
    cam = cam_bbox.float()[:, None, :].expand(-1, xy.shape[1], -1)
    return torch.cat([xy, conf, cam], dim=-1)


@torch.no_grad()
def main():
    assert cfg.MODEL.name == 'student' and cfg.MODEL.get('pose_lifter', '') == 'diffusion', \
        'can yml Student voi MODEL.pose_lifter = diffusion'
    model = ARTS(num_joint=17, embed_dim=cfg.MODEL.hpe_dim).to(DEVICE)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(check_data_pararell(ck['model_state_dict']), strict=False)
    print(f'===> nap {args.ckpt} (epoch {ck.get("epoch")}) | thieu {len(missing)} | thua {len(unexpected)}')
    model.eval()
    dl = model.diff_lifter

    ds = get_test_dataset(args.set, None)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=cfg.DATASET.workers)
    reg = torch.as_tensor(ds.h36m_joint_regressor, dtype=torch.float32, device=DEVICE)   # (17, 6890)
    acc = {v: {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': []} for v in VARIANTS}
    lift_err = {v: [] for v in VARIANTS}
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(0)
    sample_index = 0

    for bi, (inputs, targets, meta) in enumerate(tqdm(loader, desc=f'oracle {args.set}')):
        if args.max_batches and bi >= args.max_batches:
            break
        img = inputs['img'].to(DEVICE).float()
        pose_2d = inputs['joints'].to(DEVICE).float()
        mask = inputs['joints_mask'].to(DEVICE).float()
        cam = inputs['cam_bbox'].to(DEVICE).float()
        mesh_gt = targets['smpl_mesh_cam'].to(DEVICE).float()
        gt = torch.matmul(reg[None], mesh_gt)
        gt = gt - gt[:, 0:1]                                                     # (B, 17, 3) met

        # Pose 3D cho tung phuong an (root-relative, met)
        dl.sampling_timesteps, dl.num_proposals = 1, 1
        p_s1 = model.lift_diffusion(pose_2d, mask, cam)
        dl.sampling_timesteps, dl.num_proposals = args.S, args.K
        _, _, hyp = dl(lifter_cond(pose_2d, mask, cam), generator=gen)           # (B, K, 17, 3)
        hyp = hyp - hyp[:, :, 0:1]
        err = torch.norm(hyp - gt[:, None], dim=-1)                              # (B, K, 17)
        best_pose = hyp[torch.arange(hyp.shape[0]), err.mean(-1).argmin(1)]     # (B, 17, 3)
        best_joint = torch.gather(hyp, 1, err.argmin(1)[:, None, :, None].expand(-1, 1, -1, 3))[:, 0]
        poses = dict(zip(VARIANTS, [p_s1, hyp.mean(1), best_pose, best_joint, gt]))

        feature_map, _ = model.get_image_features(img)
        pe = model._pe2d_kwargs(pose_2d, mask)
        target_mesh = mesh_gt.cpu().numpy()
        for name, p3 in poses.items():
            p3 = p3 - p3[:, 0:1]
            lift_err[name].append(torch.norm(p3 - gt, dim=-1).mean(-1).cpu())
            out = model.smpl_model(joints=p3, img_feats=feature_map, is_train=False, return_features=True, **pe)
            pred_mesh = out['smpl_mesh_cam'].cpu().numpy()
            res = ds.evaluate([{'smpl_mesh_cam': pred_mesh[i], 'smpl_mesh_cam_target': target_mesh[i]}
                               for i in range(pred_mesh.shape[0])], sample_index)
            for k, v in res.items():
                if k in acc[name]:
                    acc[name][k].extend(v)
        sample_index += img.shape[0]

    print('\n' + '=' * 92)
    print(f'{args.set} | n = {sample_index} | ckpt epoch {ck.get("epoch")} | gia thuyet S={args.S} K={args.K}')
    print(f'{"pose dua vao fusion":<26} | {"lifter MPJPE (17)":>17} || {"FuseKin MPJPE(14)":>17} | {"PA-MPJPE":>8} | {"MPVPE":>7}')
    print('-' * 92)
    for name in VARIANTS:
        le = torch.cat(lift_err[name]).mean().item() * 1000
        m = {k: (sum(v) / len(v) if v else float('nan')) for k, v in acc[name].items()}
        print(f'{name:<26} | {le:>17.2f} || {m["mpjpe"]:>17.2f} | {m["pa_mpjpe"]:>8.2f} | {m["mpvpe"]:>7.2f}')
    print('=' * 92)


if __name__ == '__main__':
    main()
