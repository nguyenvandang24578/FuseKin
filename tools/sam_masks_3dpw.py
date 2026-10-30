"""
tools/sam_masks_3dpw.py -- BUOC 1 (moi truong co torch, vd FuseKin): tinh mask SAM cho cac nguoi duoc chon, luu
<mask_dir>/<ann_id>.npz (m_<prompt> = mask bool nen packbits, s_<prompt> = diem SAM, shape = (H, W)).
BUOC 2 (moi truong pose2d): tools/vis_sam_rtmpose_3dpw.py ... --mask_dir <mask_dir>  (CUNG tham so chon mau).
Tach 2 buoc vi torch trong pose2d bi segfault (xung dot CUDA voi onnxruntime-gpu).
Cai SAM trong moi truong FuseKin KHONG doi torch:
  pip install --no-deps segment-anything                  (SAM1, can file sam_vit_h_4b8939.pth)
  pip install --no-deps sam2 && pip install hydra-core iopath   (SAM2)
Vi du:
  python tools/sam_masks_3dpw.py --data_root data_final/PW3D --split validation \\
      --seqs courtyard_hug_00 courtyard_dancing_00 --only_multi --stride 50 --max_people 20 --mask_dir debug_sam_masks
"""
import os
import os.path as osp
import sys

import cv2
import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from vis_sam_rtmpose_3dpw import parse_args, select, Sam, op_points, gt_box   # noqa: E402


def main():
    args = parse_args()
    out = args.mask_dir or 'debug_sam_masks'
    os.makedirs(out, exist_ok=True)
    imgs, by_img, people = select(args)
    print(f'{len(people)} nguoi | prompts {args.prompts} | SAM {args.sam} -> {out}')
    sam = Sam(args)
    cur = None
    for k, a in people:
        im = imgs[k]
        if k != cur:
            img = cv2.imread(osp.join(args.data_root, 'imageFiles', im['sequence'], im['file_name']))
            cur = k
            if img is None:
                continue
            sam.set_image(img)
        box = gt_box(a)
        pos = op_points(a)
        neg = [pt for b in by_img[k] if b is not a for pt in op_points(b)]
        save = {'shape': np.asarray(img.shape[:2])}
        for pr in args.prompts:
            m, sc = sam.mask(box, pos if pr != 'box' else None, neg if pr == 'box_op_neg' else None)
            save[f'm_{pr}'] = np.packbits(m.reshape(-1))
            save[f's_{pr}'] = np.float32(sc)
        np.savez_compressed(osp.join(out, f'{a["id"]}.npz'), **save)
    print(f'Xong: {len(people)} mask -> {out}')


if __name__ == '__main__':
    main()
