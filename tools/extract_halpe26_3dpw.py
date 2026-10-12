"""
Trich 2D keypoint Halpe-26 (co DINH DAU, co, hong) cho 3DPW bang RTMPose (rtmlib, ONNX).
Chay trong moi truong rieng `pose2d` (rtmlib + onnxruntime-gpu), KHONG can code FuseKin.

Dau vao : <data_root>/3DPW_latest_{split}.json  (annotation COCO-style ma FuseKin dang dung)
          <data_root>/imageFiles/<sequence>/<file_name>
Bbox    : bbox GT trong annotation (ann['bbox'], xywh) -> RTMPose tu mo rong bbox ben trong.
Dau ra  : <out_dir>/3DPW_{split}_halpe26.json = { "<ann_id>": [[x, y, score] x 26], ... }
          toa do pixel ANH GOC.

Thu tu 26 khop Halpe:
  0 Nose, 1 LEye, 2 REye, 3 LEar, 4 REar, 5 LShoulder, 6 RShoulder, 7 LElbow, 8 RElbow,
  9 LWrist, 10 RWrist, 11 LHip, 12 RHip, 13 LKnee, 14 RKnee, 15 LAnkle, 16 RAnkle,
  17 Head (dinh dau), 18 Neck, 19 Hip (giua hong), 20-25 ban chan

--vis_num N : ve N anh moi split vao --vis_dir, moi anh 3 o:
              [Halpe-26 goc] | [OpenPose cu cua FuseKin] | [H36M-17 sau khi doi (halpe2h36m)]

Vi du (chay tu thu muc goc FuseKin):
  conda activate pose2d
  python tools/extract_halpe26_3dpw.py --data_root data_final/PW3D --splits validation --vis_num 5
  python tools/extract_halpe26_3dpw.py --data_root data_final/PW3D --splits train validation test
"""
import argparse
import json
import os
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

HALPE26_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 19), (12, 19), (19, 18),   # chan, hong, than
    (18, 5), (18, 6), (5, 7), (7, 9), (6, 8), (8, 10),                       # vai, tay
    (18, 17), (0, 1), (0, 2), (1, 3), (2, 4), (18, 0),                       # dau, mat
    (15, 24), (15, 20), (15, 22), (16, 25), (16, 21), (16, 23),              # ban chan
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default=os.environ.get('JOTR_DATA_ROOT', 'data_final') + '/PW3D',
                   help='Thu muc PW3D (chua 3DPW_latest_*.json va imageFiles/)')
    p.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'])
    p.add_argument('--out_dir', default='', help='Mac dinh = data_root')
    p.add_argument('--mode', default='performance', choices=['performance', 'balanced', 'lightweight'],
                   help='Kich co model RTMPose-Halpe26 trong rtmlib (performance = lon nhat, chinh xac nhat)')
    p.add_argument('--onnx', default='', help='(Tuy chon) duong dan/URL file ONNX Halpe26, ghi de --mode')
    p.add_argument('--input_size', type=int, nargs=2, default=None, metavar=('W', 'H'),
                   help='(Di kem --onnx) kich thuoc dau vao model, vd 288 384')
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--vis_num', type=int, default=0, help='So anh ve kiem tra moi split (0 = tat)')
    p.add_argument('--vis_dir', default='debug_halpe26')
    return p.parse_args()


def build_model(args):
    if args.device == 'cuda':
        # Nap thu vien CUDA/cuDNN cai qua pip (nvidia-*-cu12) truoc khi tao session (onnxruntime >= 1.21)
        import onnxruntime as ort
        if hasattr(ort, 'preload_dlls'):
            try:
                ort.preload_dlls()
            except Exception as e:
                print(f'[canh bao] preload_dlls loi: {e}')
        print(f'[onnxruntime] {ort.__version__} | providers: {ort.get_available_providers()}')
    from rtmlib import RTMPose
    if args.onnx:
        if args.input_size is None:
            raise SystemExit('Dung --onnx thi phai kem --input_size W H')
        onnx, size = args.onnx, tuple(args.input_size)
    else:
        try:
            from rtmlib import BodyWithFeet   # BodyWithFeet = detector nguoi + RTMPose Halpe26
            m = BodyWithFeet.MODE[args.mode]
            onnx, size = m['pose'], tuple(m['pose_input_size'])
        except Exception as e:
            raise SystemExit(f'Khong lay duoc model Halpe26 tu rtmlib ({type(e).__name__}: {e}).\n'
                             f'Hay truyen --onnx <url hoac file .onnx> --input_size W H')
    print(f'[model] {onnx} | input_size={size} | device={args.device}')
    model = RTMPose(onnx, model_input_size=size, backend='onnxruntime', device=args.device)
    # Kiem tra that su chay GPU (onnxruntime tu lui ve CPU neu thieu CUDA, chi in canh bao)
    sess = getattr(model, 'session', None)
    if sess is not None:
        used = sess.get_providers()
        print(f'[model] providers dang dung: {used}')
        if args.device == 'cuda' and 'CUDAExecutionProvider' not in used:
            print('[CANH BAO] Dang chay CPU (rat cham). Kiem tra lai onnxruntime-gpu / CUDA 12 / cuDNN 9.')
    return model


# OpenPose COCO-18 (truong 'openpose_result' trong annotation 3DPW):
# 0 Nose, 1 Neck, 2 RSho, 3 RElb, 4 RWri, 5 LSho, 6 LElb, 7 LWri, 8 RHip, 9 RKnee, 10 RAnk,
# 11 LHip, 12 LKnee, 13 LAnk, 14 REye, 15 LEye, 16 REar, 17 LEar
OPENPOSE18_SKELETON = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
                       (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
OPENPOSE_THR = 0.05   # giong PW3D.conf_thr cua FuseKin

# H36M-17 (thu tu FuseKin): 0 Pelvis, 1 RHip, 2 RKnee, 3 RAnk, 4 LHip, 5 LKnee, 6 LAnk, 7 Torso,
# 8 Neck, 9 Nose, 10 Head_top, 11 LSho, 12 LElb, 13 LWri, 14 RSho, 15 RElb, 16 RWri
H36M_SKELETON = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
                 (8, 14), (14, 15), (15, 16), (8, 11), (11, 12), (12, 13)]


def halpe2h36m(k):
    """Halpe-26 (26, 3) -> H36M-17 (17, 3), GIONG ham halpe2h36m cua MotionBERT (lib/data/dataset_wild.py).
    Torso = trung diem Neck-Hip (confidence = min cua 2 khop)."""
    y = np.zeros((17, 3), dtype=np.float32)
    idx = {0: 19, 1: 12, 2: 14, 3: 16, 4: 11, 5: 13, 6: 15, 8: 18, 9: 0, 10: 17,
           11: 5, 12: 7, 13: 9, 14: 6, 15: 8, 16: 10}
    for dst, src in idx.items():
        y[dst] = k[src]
    y[7, :2] = (k[18, :2] + k[19, :2]) * 0.5
    y[7, 2] = min(k[18, 2], k[19, 2])
    return y


def draw(img, kpts, scores, skeleton, thr, highlight=(), title=''):
    for a, b in skeleton:
        if scores[a] > thr and scores[b] > thr:
            cv2.line(img, tuple(int(v) for v in kpts[a]), tuple(int(v) for v in kpts[b]), (255, 255, 0), 2)
    for j, ((x, y), s) in enumerate(zip(kpts, scores)):
        if s <= thr:
            continue
        color = (0, 0, 255) if j in highlight else (0, 255, 0)
        cv2.circle(img, (int(x), int(y)), 4, color, -1)
        cv2.putText(img, str(j), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if title:
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


def save_vis(img, anns, kpts, scores, out_path, thr=0.3):
    """3 o: [Halpe-26 goc] | [OpenPose cu (FuseKin dang dung)] | [H36M-17 sau khi doi]. Cat quanh cac nguoi."""
    a = img.copy(); b = img.copy(); c = img.copy()
    for ann, k, s in zip(anns, kpts, scores):
        x, y, w, h = [int(v) for v in ann['bbox']]
        for im in (a, b, c):
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
        draw(a, k, s, HALPE26_SKELETON, thr, highlight=(17, 18, 19))
        op = np.asarray(ann.get('openpose_result', []), dtype=np.float32).reshape(-1, 3)
        if len(op) >= 18:
            draw(b, op[:18, :2], op[:18, 2], OPENPOSE18_SKELETON, OPENPOSE_THR)
        h36 = halpe2h36m(np.concatenate([k, s[:, None]], axis=1))
        draw(c, h36[:, :2], h36[:, 2], H36M_SKELETON, thr, highlight=(7, 10))
    # cat vung chua tat ca nguoi (+ le) cho de nhin, roi moi ghi tieu de
    xs = [v for ann in anns for v in (ann['bbox'][0], ann['bbox'][0] + ann['bbox'][2])]
    ys = [v for ann in anns for v in (ann['bbox'][1], ann['bbox'][1] + ann['bbox'][3])]
    H, W = img.shape[:2]
    x0, x1 = max(0, int(min(xs)) - 50), min(W, int(max(xs)) + 50)
    y0, y1 = max(0, int(min(ys)) - 60), min(H, int(max(ys)) + 50)
    titles = ['Halpe-26 (do: 17 dinh dau, 18 co, 19 hong)', 'OpenPose cu', 'H36M-17 (do: 7 Torso, 10 Head_top)']
    crops = [np.ascontiguousarray(p[y0:y1, x0:x1]) for p in (a, b, c)]
    crops = [draw(cr, [], [], [], thr, title=t) for cr, t in zip(crops, titles)]
    cv2.imwrite(out_path, np.concatenate(crops, axis=1))


def run_split(model, args, split, out_dir):
    ann_file = osp.join(args.data_root, f'3DPW_latest_{split}.json')
    with open(ann_file) as f:
        db = json.load(f)
    imgs = {im['id']: im for im in db['images']}
    by_img = defaultdict(list)
    for ann in db['annotations']:
        by_img[ann['image_id']].append(ann)
    print(f'[{split}] {len(db["annotations"])} nguoi / {len(by_img)} anh')

    results, n_missing_img = {}, 0
    vis_ids = set(list(by_img.keys())[::max(1, len(by_img) // max(1, args.vis_num))][:args.vis_num]) if args.vis_num else set()
    for image_id, anns in tqdm(by_img.items(), desc=split):
        im = imgs[image_id]
        img_path = osp.join(args.data_root, 'imageFiles', im['sequence'], im['file_name'])
        img = cv2.imread(img_path)          # BGR, dung dinh dang rtmlib
        if img is None:
            n_missing_img += 1
            continue
        bboxes = []
        for ann in anns:
            x, y, w, h = [float(v) for v in ann['bbox']]
            bboxes.append([x, y, x + w, y + h])   # xywh -> xyxy
        kpts, scores = model(img, bboxes=bboxes)  # (N, 26, 2), (N, 26)
        kpts, scores = np.asarray(kpts), np.asarray(scores)
        assert kpts.shape[1] == 26, f'Model khong phai Halpe26: {kpts.shape}'
        for ann, k, s in zip(anns, kpts, scores):
            results[str(ann['id'])] = np.concatenate([k, s[:, None]], axis=1).round(3).tolist()

        if image_id in vis_ids:
            os.makedirs(args.vis_dir, exist_ok=True)
            save_vis(img, anns, kpts, scores,
                     osp.join(args.vis_dir, f'{split}_{im["sequence"]}_{im["file_name"]}'))

    out_path = osp.join(out_dir, f'3DPW_{split}_halpe26.json')
    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f'[{split}] luu {len(results)} nguoi -> {out_path} | anh khong doc duoc: {n_missing_img}')


def main():
    args = parse_args()
    out_dir = args.out_dir or args.data_root
    os.makedirs(out_dir, exist_ok=True)
    model = build_model(args)
    for split in args.splits:
        run_split(model, args, split, out_dir)


if __name__ == '__main__':
    main()
