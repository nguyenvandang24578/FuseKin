"""
tools/extract_vitpose_train.py (2026-11-17, nguoi dung yeu cau) -- Trich 2D ViTPose cho Human36M / MuCo / MSCOCO /
CrowdPose, DUNG LAI nguyen cac tool Halpe (extract_halpe26_{h36m,muco,coco}.py): cung cach chon nguoi / loc frame /
bbox GT annotation / chia shard / gop / ve hinh, CHI thay model RTMPose bang ViTPose (ViTPoseModel cua
extract_vitpose_3dpw.py: COCO-17 -> dinh dang Halpe-26, dinh dau + ban chan score 0) va them hau to file ra.

Chay trong env `arts` (repo ViTPose goc). Bbox = bbox GT annotation (cac bo nay KHONG co OpenPose).
Dau ra  : giong tool Halpe nhung ten co hau to: <Name>_<split>_halpe26_vitpose.npz (khong ghi de file Halpe cu)
          H36M: Human36M_{train,test}_halpe26_vitpose.npz (khoa file_name) | MuCo / MSCOCO / CrowdPose:
          <Name>_train_halpe26_vitpose.npz (khoa ann_id); kpts (N, 26, 3) [x, y, score] pixel anh goc.
Luu y: dataset FuseKin hien doc <Name>_train_halpe26.npz (khong hau to) -> chua tu dung file ViTPose.

Cach dung: tham so rieng cua script nay (--tool, --pose_config, --pose_ckpt, --suffix), PHAN CON LAI chuyen nguyen
cho tool Halpe tuong ung (vd --data_root, --splits, --num_shards, --shard, --merge, --max_items, --vis_num ...).
  CFG=ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py
  CKPT=pose_detector/vitpose-h-multi-coco.pth
  # thu nhanh
  python tools/extract_vitpose_train.py --tool muco --pose_config $CFG --pose_ckpt $CKPT --max_items 200 --vis_num 8
  # 4 GPU, 1 bo
  for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python tools/extract_vitpose_train.py --tool h36m --pose_config $CFG \
      --pose_ckpt $CKPT --splits train --num_shards 4 --shard $i > log_vit_h36m_$i.txt 2>&1 & done; wait
  python tools/extract_vitpose_train.py --tool h36m --pose_config $CFG --pose_ckpt $CKPT --splits train --num_shards 4 --merge
  (--tool coco can them --dataset mscoco | crowdpose cua tool Halpe)
"""
import argparse
import importlib
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description='Trich 2D ViTPose cho Human36M / MuCo / MSCOCO / CrowdPose', add_help=True)
    p.add_argument('--tool', required=True, choices=['h36m', 'muco', 'coco'],
                   help='tool Halpe dung lai: extract_halpe26_<tool>.py (coco = MSCOCO va CrowdPose)')
    p.add_argument('--pose_config', required=True)
    p.add_argument('--pose_ckpt', required=True)
    p.add_argument('--vit_device', default='cuda:0', help='GPU cho ViTPose (dung CUDA_VISIBLE_DEVICES de chon card)')
    p.add_argument('--suffix', default='_vitpose', help="hau to them sau '_halpe26' trong ten file ra")
    args, rest = p.parse_known_args()

    import extract_vitpose_3dpw as V                       # ViTPoseModel (COCO-17 -> Halpe-26)
    mod = importlib.import_module(f'extract_halpe26_{args.tool}')

    # Thay model: tool Halpe goi build_model(args) roi model(img, bboxes=[xyxy, ...]) -> (k (N,26,2), s (N,26))
    mod.build_model = lambda _a: V.ViTPoseModel(args.pose_config, args.pose_ckpt, args.vit_device)
    # Them hau to vao moi ten file (ca shard, merge, debug) -> khong ghi de file Halpe cu
    orig_out = mod.out_name

    def out_name(*a, **k):
        path = orig_out(*a, **k)
        assert '_halpe26' in path, path
        return path.replace('_halpe26', '_halpe26' + args.suffix, 1)
    mod.out_name = out_name

    sys.argv = [mod.__file__] + rest
    print(f'[extract_vitpose_train] tool {mod.__file__} | ViTPose {args.pose_ckpt} | hau to {args.suffix} | '
          f'tham so tool: {rest}')
    mod.main()


if __name__ == '__main__':
    main()
