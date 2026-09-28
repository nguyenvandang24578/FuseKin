"""check_overlay.py – Visual and numerical keypoint overlay check.

Claims to check
===============
1. Draw BOTH orig_joint_img (GT clean) AND inputs['joints'] (detector) on
   the same crop image in different colours so we can compare positions.
2. Un-normalise image using the actual mean/std from the dataset code.
3. Numerical checks: (a) keypoints mostly in [0,64) heatmap space,
   (b) average distance between GT and detector < 30px (crop-256).

Previous bugs
=============
- Only drew ONE set of keypoints (GT or detector), not both.
- Used hardcoded ImageNet mean/std (but dataset code uses ToTensor()/255
  without further normalisation — values are in [0,1]).
- PASS was unconditional — just meant "image saved".
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.config import update_config, cfg
from utils.jotr_dataset import get_train_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/train_init_mesh.yaml')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out_dir', type=str, default='logs/server_checks')
    parser.add_argument('--n_samples', type=int, default=16)
    return parser.parse_args()


def hm_to_pixel(xy, hm_shape, img_shape):
    """Convert heatmap coords [0, hm) to pixel coords [0, img_size)."""
    px = xy[0] / hm_shape[2] * img_shape[1]
    py = xy[1] / hm_shape[1] * img_shape[0]
    return px, py


def norm_to_pixel(xy, img_shape):
    """Convert [-1, 1] normalised coords to pixel [0, img_size)."""
    px = (xy[0] + 1) * 0.5 * img_shape[1]
    py = (xy[1] + 1) * 0.5 * img_shape[0]
    return px, py


def draw_kps(draw, kps, conf, color, hm_shape, img_shape, coord_type='heatmap'):
    """Draw keypoints on image. Returns list of pixel coords."""
    px_coords = []
    for i in range(kps.shape[0]):
        x, y = float(kps[i, 0]), float(kps[i, 1])
        c = float(conf[i]) if conf is not None else 1.0

        if coord_type == 'heatmap':
            px, py = hm_to_pixel((x, y), hm_shape, img_shape)
        elif coord_type == 'normalised':
            px, py = norm_to_pixel((x, y), img_shape)
        else:
            px, py = x, y

        px_coords.append((px, py))
        fill = color if c > 0.3 else 'gray'
        r = 3
        draw.ellipse((px-r, py-r, px+r, py+r), fill=fill, outline='white')
        draw.text((px+4, py-6), str(i), fill=fill)

    return px_coords


def main():
    args = parse_args()
    out_img_dir = os.path.join(args.out_dir, 'overlay')
    os.makedirs(out_img_dir, exist_ok=True)
    log_file = os.path.join(args.out_dir, 'check_overlay.log')

    def log(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== check_overlay ===\n")

    all_pass = True
    try:
        if os.path.exists(args.cfg):
            update_config(args.cfg)

        train_ds = get_train_dataset('3dpw-train', args)

        hm_shape = cfg.output_hm_shape  # (64, 64, 64)
        img_shape = cfg.input_img_shape  # (256, 256)

        gt_oob_count = 0
        det_oob_count = 0
        gt_total = 0
        distances = []

        log(f"Processing {args.n_samples} train samples...")
        log(f"Image shape: {img_shape}, Heatmap shape: {hm_shape}")
        log(f"\nDataset image normalisation: ToTensor()/255 → values in [0,1]")
        log(f"(No additional mean/std normalisation detected in PW3D.__getitem__)")

        for i in range(min(args.n_samples, len(train_ds))):
            inputs, targets, meta = train_ds[i]

            # ---- Reconstruct image ----
            # PW3D dataset: img = self.transform(img.astype(np.float32))/255.
            # transform = transforms.ToTensor() → (C,H,W) [0,1]
            # Then /255 → values in [0, 1/255] ... Actually ToTensor already
            # divides by 255 for uint8. But input is float32, so ToTensor just
            # permutes. Then /255 brings to [0, ~1].
            # So img values ≈ [0, 1].
            img_tensor = np.asarray(inputs['img'])  # (3, H, W)
            img_np = img_tensor.transpose(1, 2, 0)  # (H, W, 3)
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            draw = ImageDraw.Draw(img_pil)

            H, W = img_np.shape[:2]

            # ---- GT keypoints (orig_joint_img) ----
            # Shape: (17, 3) in heatmap space [0, 64)
            gt_kp = np.asarray(targets['orig_joint_img'])[:, :2]  # (17, 2)
            gt_conf = np.asarray(meta['orig_joint_trunc']).squeeze(-1)  # (17,)

            gt_px = draw_kps(draw, gt_kp, gt_conf, 'lime', hm_shape, (H, W),
                             coord_type='heatmap')

            # ---- Detector keypoints (inputs['joints']) ----
            # After Human36M17Dataset, joints are normalised to [-1,1]
            det_kp_raw = np.asarray(inputs['joints'])  # (17, 3) or (30, 3)
            det_kp = det_kp_raw[:17, :2]
            det_conf_raw = np.asarray(inputs.get('joints_mask',
                                                  np.ones((17, 1))))
            if det_conf_raw.ndim == 2:
                det_conf = det_conf_raw[:17].squeeze(-1)
            else:
                det_conf = det_conf_raw[:17]

            det_px = draw_kps(draw, det_kp, det_conf, 'cyan', hm_shape, (H, W),
                              coord_type='normalised')

            # ---- Numerical checks ----
            for j in range(min(17, len(gt_px))):
                gt_total += 1
                gx, gy = gt_kp[j]
                # GT should be in [0, 64)
                if gx < 0 or gx >= hm_shape[2] or gy < 0 or gy >= hm_shape[1]:
                    gt_oob_count += 1

                # Detector should be in [-1, 1]
                dx, dy = det_kp[j]
                if abs(dx) > 1.1 or abs(dy) > 1.1:
                    det_oob_count += 1

                # Distance in pixel-256 space
                gt_p = gt_px[j]
                det_p = det_px[j]
                dist = np.sqrt((gt_p[0] - det_p[0])**2 + (gt_p[1] - det_p[1])**2)
                distances.append(dist)

            img_pil.save(os.path.join(out_img_dir, f'overlay_{i:02d}.png'))

        # ---- Report ----
        log(f"\nGT keypoints out of [0,64): {gt_oob_count}/{gt_total}")
        log(f"Det keypoints out of [-1,1]: {det_oob_count}/{gt_total}")

        if gt_total > 0 and len(distances) > 0:
            dist_arr = np.array(distances)
            mean_dist = dist_arr.mean()
            median_dist = np.median(dist_arr)
            max_dist = dist_arr.max()
            log(f"\nGT-vs-Detector distance (px-{W}):")
            log(f"  mean: {mean_dist:.1f}px, median: {median_dist:.1f}px, "
                f"max: {max_dist:.1f}px")

            if mean_dist > 30:
                log(f"  WARNING: mean distance {mean_dist:.1f}px > 30px. "
                    f"Possible coordinate system mismatch!")
                all_pass = False

        gt_oob_frac = gt_oob_count / gt_total if gt_total > 0 else 0
        if gt_oob_frac > 0.2:
            log(f"  WARNING: {gt_oob_frac*100:.0f}% of GT kps out of bounds!")
            all_pass = False

        if all_pass:
            log(f"\nPASS - Numerical checks OK. Visual verification still needed "
                f"(check {out_img_dir}/ images: lime=GT, cyan=detector).")
        else:
            log(f"\nFAIL - Numerical checks indicate coordinate issues.")
            sys.exit(1)

    except Exception as e:
        import traceback
        log(f"FAIL - Exception occurred:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
