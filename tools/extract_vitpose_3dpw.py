"""
tools/extract_vitpose_3dpw.py (2026-11-17, nguoi dung duyet) -- Trich 2D keypoint ViTPose cho 3DPW, ghi CUNG dinh
dang Halpe-26 de dung lai nguyen pipeline Halpe cua FuseKin (khong sua dataset).

Chay trong env `arts` da cai repo ViTPose goc (ViTAE-Transformer/ViTPose, fork mmpose 0.x, backbone 'ViT'),
KHONG can code FuseKin. API giong ARTS/main/run_demo.py: init_pose_model + inference_top_down_pose_model.
Dung lai tu tools/extract_halpe26_3dpw.py: chon bbox (gt / openpose / openpose_best / combo), vong lap split,
thong ke nghi nham nguoi so voi OpenPose, ve hinh kiem tra (--vis_num, --vis_compare).

ViTPose COCO-17 (0 Nose .. 16 RAnkle) -> Halpe-26:
  0..16 = COCO-17 (cung thu tu Halpe)
  17 Head (dinh dau): KHONG co -> toa do = Nose, score 0 (giong OpenPose: H36M Head_top bi che / bo)
  18 Neck = trung diem 2 vai, score = min 2 vai
  19 Hip  = trung diem 2 hong, score = min 2 hong
  20..25 ban chan: KHONG co -> toa do = co chan tuong ung, score 0
Dau ra : <out_dir>/3DPW_{split}_halpe26{--out_suffix}.json = { "<ann_id>": [[x, y, score] x 26], ... } (pixel anh goc)
         mac dinh suffix '_vitpose' (bbox gt) | '_vitpose_opbox' (bbox openpose*) -> KHONG ghi de file Halpe cu.
Dung trong FuseKin: DATASET.pw3d_2d_source 'halpe26' + DATASET.pw3d_halpe_suffix '_vitpose' (hoac '_vitpose_opbox').

Vi du (tu thu muc goc FuseKin):
  conda activate arts
  # thu it anh, khong ghi json, ve hinh
  python tools/extract_vitpose_3dpw.py --pose_config <ViTPose>/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py \
      --pose_ckpt pose_detector/vitpose-h-multi-coco.pth --data_root data_final/PW3D --splits test \
      --max_images 200 --dry_run --vis_num 5 --vis_dir debug_vitpose
  # trich that: test / oc / pc / validation (Crowd doc validation)
  python tools/extract_vitpose_3dpw.py --pose_config ... --pose_ckpt ... --data_root data_final/PW3D \
      --splits test oc pc validation --bbox_source gt
  python tools/extract_vitpose_3dpw.py --pose_config ... --pose_ckpt ... --data_root data_final/PW3D \
      --splits test oc pc validation --bbox_source openpose_best
"""
import argparse
import json
import os
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
import extract_halpe26_3dpw as H   # noqa: E402  (dung lai ham chon bbox / vong lap / ve hinh)


def parse_args():
    p = argparse.ArgumentParser(description='Trich 2D ViTPose cho 3DPW (dinh dang Halpe-26)')
    p.add_argument('--pose_config', required=True, help='config ViTPose (repo ViTPose goc)')
    p.add_argument('--pose_ckpt', required=True, help='checkpoint ViTPose, vd vitpose-h-multi-coco.pth')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--data_root', default=os.environ.get('JOTR_DATA_ROOT', 'data_final') + '/PW3D',
                   help='Thu muc PW3D (co 3DPW_latest_{split}.json va imageFiles/)')
    p.add_argument('--splits', nargs='+', default=['test', 'oc', 'pc', 'validation'])
    p.add_argument('--out_dir', default='', help='Mac dinh = data_root')
    p.add_argument('--vis_num', type=int, default=0, help='So anh ve kiem tra moi split (0 = tat)')
    p.add_argument('--vis_dir', default='debug_vitpose')
    p.add_argument('--bbox_source', default='gt', choices=['gt', 'openpose', 'openpose_best', 'combo', 'gt2d'],
                   help="gt2d: bbox om 17 khop 2D GT H36M (file tu tools/dump_pw3d_gt2d.py) x --gt2d_expand")
    p.add_argument('--gt2d_expand', type=float, default=1.25, help='he so noi bbox tu 2D GT (giong op_expand)')
    p.add_argument('--gt2d_dir', default='', help='thu muc chua 3DPW_{split}_gt2d_h36m.json (mac dinh = data_root)')
    p.add_argument('--op_expand', type=float, default=1.25, help='he so noi bbox tu khop OpenPose')
    p.add_argument('--op_min_joints', type=int, default=6, help='it hon so khop OpenPose nay -> dung bbox GT')
    p.add_argument('--out_suffix', default=None,
                   help="hau to file ra; mac dinh '_vitpose' (gt) | '_vitpose_opbox' (openpose*)")
    p.add_argument('--seqs', nargs='*', default=[], help='chi xu ly cac sequence nay (thu nghiem)')
    p.add_argument('--max_images', type=int, default=0, help='dung sau N anh (0 = het)')
    p.add_argument('--dry_run', action='store_true', help='khong ghi json')
    p.add_argument('--vis_compare', type=int, default=0,
                   help='ve K nguoi ma Halpe CU lech OpenPose: [Halpe cu] | [ViTPose] | [OpenPose]')
    a = p.parse_args()
    if a.out_suffix is None:
        a.out_suffix = {'gt': '_vitpose', 'gt2d': '_vitpose_gt2dbox'}.get(a.bbox_source, '_vitpose_opbox')
    return a


def gt2d_box(j, expand):
    """j (17, 3) [x, y, in_img] 2D GT -> bbox xyxy om CA 17 khop (ke ca bi che / ngoai anh), noi expand quanh tam."""
    j = np.asarray(j, dtype=np.float32)
    x0, y0 = j[:, 0].min(), j[:, 1].min()
    x1, y1 = j[:, 0].max(), j[:, 1].max()
    cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * expand, (y1 - y0) * expand
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


_infer_orig = H.infer


def infer(model, img, anns, args):
    """Them che do 'gt2d' cho H.infer; cac che do khac goi ham goc. Nguoi khong co 2D GT -> bbox GT annotation."""
    if args.bbox_source != 'gt2d':
        return _infer_orig(model, img, anns, args)
    boxes = []
    for a in anns:
        j = args.gt2d.get(str(a['id']))
        if j is None:
            args.gt2d_miss += 1
            boxes.append(H.gt_box(a))
        else:
            boxes.append(gt2d_box(j, args.gt2d_expand))
    k, s = model(img, bboxes=boxes)
    return np.asarray(k), np.asarray(s), boxes


H.infer = infer   # run_split goi H.infer -> dung ban co 'gt2d'


def coco17_to_halpe26(k17):
    """k17 (17, 3) [x, y, score] COCO -> (k (26, 2), s (26,)) theo thu tu Halpe-26 (xem docstring dau file)."""
    k17 = np.asarray(k17, dtype=np.float32)
    k = np.zeros((26, 2), dtype=np.float32)
    s = np.zeros((26,), dtype=np.float32)
    k[:17], s[:17] = k17[:, :2], k17[:, 2]
    k[17], s[17] = k17[0, :2], 0.0                                   # dinh dau: khong co
    k[18], s[18] = (k17[5, :2] + k17[6, :2]) * 0.5, min(k17[5, 2], k17[6, 2])     # Neck
    k[19], s[19] = (k17[11, :2] + k17[12, :2]) * 0.5, min(k17[11, 2], k17[12, 2])  # Hip
    k[20:23] = k17[15, :2]                                           # ban chan trai: dat tai co chan, score 0
    k[23:26] = k17[16, :2]                                           # ban chan phai
    return k, s


class ViTPoseModel:
    """Boc ViTPose de goi giong RTMPose cua rtmlib: model(img_bgr, bboxes=[xyxy, ...]) -> (k (N,26,2), s (N,26))."""
    def __init__(self, config, ckpt, device):
        from mmpose.apis import init_pose_model
        from mmpose.datasets import DatasetInfo
        self.model = init_pose_model(config, ckpt, device=device)
        self.dataset = self.model.cfg.data['test']['type']
        info = self.model.cfg.data['test'].get('dataset_info', None)
        self.dataset_info = DatasetInfo(info) if info is not None else None
        print(f'[ViTPose] {config} | {ckpt} | dataset {self.dataset} | device {device}')

    def __call__(self, img, bboxes):
        from mmpose.apis import inference_top_down_pose_model
        persons = [{'bbox': np.asarray(b, dtype=np.float32)} for b in bboxes]
        # img: mang BGR (cv2.imread); mmpose 0.x LoadImageFromFile chuyen BGR -> RGB cho ndarray
        res, _ = inference_top_down_pose_model(self.model, img, persons, bbox_thr=None, format='xyxy',
                                               dataset=self.dataset, dataset_info=self.dataset_info,
                                               return_heatmap=False, outputs=None)
        assert len(res) == len(bboxes), f'ViTPose tra {len(res)} nguoi cho {len(bboxes)} bbox'
        ks, ss = zip(*[coco17_to_halpe26(r['keypoints']) for r in res])
        return np.stack(ks), np.stack(ss)


def main():
    args = parse_args()
    out_dir = args.out_dir or args.data_root
    os.makedirs(out_dir, exist_ok=True)
    model = ViTPoseModel(args.pose_config, args.pose_ckpt, args.device)
    for split in args.splits:
        if args.bbox_source == 'gt2d':
            p = osp.join(args.gt2d_dir or args.data_root, f'3DPW_{split}_gt2d_h36m.json')
            assert osp.isfile(p), f'Khong thay {p}. Chay truoc: python tools/dump_pw3d_gt2d.py --tags {split}'
            with open(p) as f:
                args.gt2d = json.load(f)
            args.gt2d_miss = 0
            print(f'[{split}] 2D GT: {p} ({len(args.gt2d)} nguoi)')
        H.run_split(model, args, split, out_dir)
        if args.bbox_source == 'gt2d':
            print(f'[{split}] nguoi khong co 2D GT (dung bbox annotation): {args.gt2d_miss}')


if __name__ == '__main__':
    main()
