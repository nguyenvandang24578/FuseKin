"""
tools/vis_sam_rtmpose_3dpw.py -- THU NHANH: SAM tach mask DUNG nguoi -> che phan ngoai mask -> RTMPose Halpe-26.
Chi ve hinh de xem bang mat (khong ghi json). Chay trong moi truong `pose2d` (rtmlib + SAM), tu thu muc goc FuseKin.

Moi nguoi (annotation 3DPW) ve 1 hinh, cac o:
  [Halpe cu (file 3DPW_<split>_halpe26.json, bbox GT)] | [mask SAM + Halpe moi] cho tung --prompts | [OpenPose]
  khung tim = bbox GT; mau mask: xanh duong trong suot.
--prompts (co the nhieu, moi cai 1 o):
  box        : chi bbox GT
  box_op     : bbox GT + diem duong = khop OpenPose cua nguoi do
  box_op_neg : nhu tren + diem AM = khop OpenPose cua NGUOI KHAC trong anh (de SAM tach 2 nguoi)
SAM: --sam sam2 (pip install sam2; tai 'facebook/sam2.1-hiera-large' tu HuggingFace)
     --sam sam1 --sam_ckpt sam_vit_h_4b8939.pth (pip install segment-anything)
Vi du:
  python tools/vis_sam_rtmpose_3dpw.py --data_root data_final/PW3D --split validation \\
      --seqs courtyard_hug_00 courtyard_dancing_00 --only_multi --max_people 20 --stride 50 --flip --out_dir debug_sam
  python tools/vis_sam_rtmpose_3dpw.py --data_root data_final/PW3D --split pc --only_multi --max_people 20 --stride 20 --flip
"""
import argparse
import json
import os
import os.path as osp
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch  # noqa: F401  nap torch (CUDA / cuDNN cua torch) TRUOC onnxruntime -> tranh xung dot thu vien (free(): invalid pointer)

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from extract_halpe26_3dpw import (build_model, draw, HALPE26_SKELETON, OPENPOSE18_SKELETON, OPENPOSE_THR,  # noqa: E402
                                  HALPE_THR, op_joints, gt_box)

HALPE_FLIP = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (20, 21), (22, 23), (24, 25))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='data_final/PW3D')
    p.add_argument('--split', default='validation', help='test | validation | train | oc | pc')
    p.add_argument('--seqs', nargs='*', default=[])
    p.add_argument('--ann_ids', nargs='*', type=int, default=[], help='chi cac annotation nay')
    p.add_argument('--only_multi', action='store_true', help='chi anh co >= 2 nguoi (cho de nham)')
    p.add_argument('--stride', type=int, default=1, help='lay 1 anh moi N anh')
    p.add_argument('--max_people', type=int, default=20)
    p.add_argument('--prompts', nargs='+', default=['box', 'box_op', 'box_op_neg'],
                   choices=['box', 'box_op', 'box_op_neg'])
    p.add_argument('--sam', default='sam2', choices=['sam2', 'sam1'])
    p.add_argument('--sam_model', default='facebook/sam2.1-hiera-large', help='sam2: ten HuggingFace')
    p.add_argument('--sam_ckpt', default='', help='sam1: file checkpoint')
    p.add_argument('--sam_type', default='vit_h', help='sam1: vit_h | vit_l | vit_b')
    p.add_argument('--dilate', type=float, default=0.04, help='no mask theo ti le canh lon bbox')
    p.add_argument('--fill', default='gray', choices=['gray', 'blur'], help='to phan ngoai mask')
    p.add_argument('--flip', action='store_true', help='TTA: trung binh voi anh lat ngang')
    p.add_argument('--out_dir', default='debug_sam')
    # cho build_model cua extract_halpe26_3dpw
    p.add_argument('--mode', default='performance', choices=['performance', 'balanced', 'lightweight'])
    p.add_argument('--onnx', default='')
    p.add_argument('--input_size', type=int, nargs=2, default=None)
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    return p.parse_args()


class Sam:
    def __init__(self, args):
        self.kind = args.sam
        if self.kind == 'sam2':
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            self.pred = SAM2ImagePredictor.from_pretrained(args.sam_model, device=args.device)
        else:
            from segment_anything import sam_model_registry, SamPredictor
            m = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt).to(args.device)
            self.pred = SamPredictor(m)

    def set_image(self, img_bgr):
        self.pred.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    def mask(self, box, pos=None, neg=None):
        pts, lab = None, None
        if pos is not None or neg is not None:
            P = [] if pos is None else list(pos)
            N = [] if neg is None else list(neg)
            if P or N:
                pts = np.asarray(P + N, dtype=np.float32)
                lab = np.asarray([1] * len(P) + [0] * len(N), dtype=np.int32)
        masks, scores, _ = self.pred.predict(point_coords=pts, point_labels=lab, box=np.asarray(box, dtype=np.float32),
                                             multimask_output=False)
        return np.asarray(masks[0]).astype(bool), float(np.asarray(scores).reshape(-1)[0])


def op_points(ann):
    op = op_joints(ann)
    if op is None:
        return []
    return [tuple(p[:2]) for p in op if p[2] > OPENPOSE_THR]


def masked_image(img, m, box, args):
    k = max(3, int(args.dilate * max(box[2] - box[0], box[3] - box[1])) | 1)
    md = cv2.dilate(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    bg = np.full_like(img, 114) if args.fill == 'gray' else cv2.GaussianBlur(img, (51, 51), 0)
    out = img.copy()
    out[~md] = bg[~md]
    return out


def rtm(model, img, box, flip):
    k, s = model(img, bboxes=[box])
    k, s = np.asarray(k)[0], np.asarray(s)[0]
    if not flip:
        return k, s
    W = img.shape[1]
    fb = [W - 1 - box[2], box[1], W - 1 - box[0], box[3]]
    kf, sf = model(img[:, ::-1].copy(), bboxes=[fb])
    kf, sf = np.asarray(kf)[0].copy(), np.asarray(sf)[0].copy()
    kf[:, 0] = W - 1 - kf[:, 0]
    for a, b in HALPE_FLIP:
        kf[[a, b]] = kf[[b, a]]
        sf[[a, b]] = sf[[b, a]]
    return (k + kf) / 2, (s + sf) / 2


def panel(img, title, box, kpts=None, scores=None, skel=None, thr=0.3, mask=None, op=None):
    p = img.copy()
    if mask is not None:
        ov = p.copy()
        ov[mask] = (0.5 * ov[mask] + 0.5 * np.array([255, 128, 0])).astype(np.uint8)
        p = ov
    cv2.rectangle(p, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 255), 2)
    if op is not None:
        draw(p, op[:, :2], op[:, 2], OPENPOSE18_SKELETON, OPENPOSE_THR)
    if kpts is not None:
        draw(p, kpts, scores, skel, thr)
    return p, title


def main():
    args = parse_args()
    ann_file = osp.join(args.data_root, f'3DPW_latest_{args.split}.json')
    if not osp.exists(ann_file):
        ann_file = osp.join(args.data_root, f'3DPW_{args.split}.json')
    with open(ann_file) as f:
        db = json.load(f)
    imgs = {im['id']: im for im in db['images']}
    by_img = defaultdict(list)
    for a in db['annotations']:
        by_img[a['image_id']].append(a)
    old = {}
    p_old = osp.join(args.data_root, f'3DPW_{args.split}_halpe26.json')
    if osp.isfile(p_old):
        with open(p_old) as f:
            old = json.load(f)
    keys = [k for k in by_img if (not args.seqs or imgs[k]['sequence'] in set(args.seqs))
            and (not args.only_multi or len(by_img[k]) >= 2)
            and (not args.ann_ids or any(a['id'] in set(args.ann_ids) for a in by_img[k]))]
    keys = keys[::max(1, args.stride)]
    print(f'{len(keys)} anh duoc chon | prompts {args.prompts} | flip {args.flip}')
    # SAM (torch) tao TRUOC, roi RTMPose (onnxruntime) KHONG preload_dlls: tranh nap 2 ban cuDNN cung luc
    sam = Sam(args)
    args.skip_preload = True
    model = build_model(args)
    os.makedirs(args.out_dir, exist_ok=True)
    n = 0
    for k in keys:
        if n >= args.max_people:
            break
        im = imgs[k]
        img = cv2.imread(osp.join(args.data_root, 'imageFiles', im['sequence'], im['file_name']))
        if img is None:
            continue
        sam.set_image(img)
        anns = by_img[k]
        for a in anns:
            if args.ann_ids and a['id'] not in set(args.ann_ids):
                continue
            if n >= args.max_people:
                break
            box = gt_box(a)
            panels = []
            o = old.get(str(a['id']))
            if o is not None:
                o = np.asarray(o, dtype=np.float32)
                panels.append(panel(img, 'Halpe cu (bbox GT)', box, o[:, :2], o[:, 2], HALPE26_SKELETON, HALPE_THR))
            pos = op_points(a)
            neg = [pt for b in anns if b is not a for pt in op_points(b)]
            for pr in args.prompts:
                m, sc = sam.mask(box, pos if pr != 'box' else None, neg if pr == 'box_op_neg' else None)
                kp, s = rtm(model, masked_image(img, m, box, args), box, args.flip)
                panels.append(panel(img, f'SAM {pr} ({sc:.2f}) + RTMPose', box, kp, s, HALPE26_SKELETON, HALPE_THR,
                                    mask=m))
            op = op_joints(a)
            panels.append(panel(img, 'OpenPose', box, op=op))
            # cat quanh bbox (noi 60% ngang) de de nhin
            H, W = img.shape[:2]
            bw = box[2] - box[0]
            x0, x1 = int(max(0, box[0] - 0.6 * bw)), int(min(W, box[2] + 0.6 * bw))
            y0, y1 = int(max(0, box[1] - 40)), int(min(H, box[3] + 40))
            crops = []
            for p, t in panels:
                c = np.ascontiguousarray(p[y0:y1, x0:x1])
                f = 480.0 / max(c.shape[0], 1)
                c = cv2.resize(c, None, fx=f, fy=f)
                draw(c, [], [], [], 0.3, title=t)
                crops.append(c)
            out = osp.join(args.out_dir, f'{args.split}_{im["sequence"]}_{a["id"]}_{im["file_name"]}')
            cv2.imwrite(out, np.concatenate(crops, axis=1))
            n += 1
    print(f'Da ve {n} nguoi -> {args.out_dir}')


if __name__ == '__main__':
    main()
