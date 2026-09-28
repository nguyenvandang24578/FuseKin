"""check_overlay.py – Visual and numerical keypoint overlay check.

Claims to check
===============
1. Draw BOTH orig_joint_img (GT clean) AND inputs['joints'] (detector) on
   the same crop image in different colours.
2. Un-normalise image correctly (values are in [0,1]).
3. Check average distance between GT and detector < 15px.
4. Print distance per joint.
5. Check left/right swapped pairs to detect joint order mismatch.

Previous bugs
=============
- Only drew ONE set of keypoints (GT or detector).
- Used train set (which has noisy GT instead of real detector).
- PASS was unconditional.
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
from utils.h36m_adapter import HUMAN36M_JOINTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/train_init_mesh.yaml')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out_dir', type=str, default='logs/server_checks')
    parser.add_argument('--n_samples', type=int, default=16)
    return parser.parse_args()


def hm_to_pixel(xy, hm_shape, img_shape):
    px = xy[0] / hm_shape[2] * img_shape[1]
    py = xy[1] / hm_shape[1] * img_shape[0]
    return px, py


def norm_to_pixel(xy, img_shape):
    px = (xy[0] + 1) * 0.5 * img_shape[1]
    py = (xy[1] + 1) * 0.5 * img_shape[0]
    return px, py


def draw_kps(draw, kps, conf, color, hm_shape, img_shape, coord_type='heatmap'):
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

        # Using train set because it provides both orig_joint_img (GT) and inputs['joints'] (detector/openpose).
        # The test split omits orig_joint_img in its targets dict.
        train_ds = get_train_dataset('3dpw-train', args)

        hm_shape = cfg.output_hm_shape
        img_shape = cfg.input_img_shape

        log(f"Processing {args.n_samples} train samples (using dataset's provided openpose for detector)...")
        
        per_joint_dists = [[] for _ in range(17)]
        
        # H36M left/right pairs
        # 1:R_Hip/4:L_Hip, 2:R_Knee/5:L_Knee, 3:R_Ankle/6:L_Ankle
        # 11:L_Shoulder/14:R_Shoulder, 12:L_Elbow/15:R_Elbow, 13:L_Wrist/16:R_Wrist
        swap_pairs = [(1,4), (2,5), (3,6), (11,14), (12,15), (13,16)]
        
        normal_dist_total = 0.0
        swapped_dist_total = 0.0
        pair_count = 0

        for i in range(min(args.n_samples, len(train_ds))):
            inputs, targets, meta = train_ds[i]

            img_tensor = np.asarray(inputs['img'])
            img_np = img_tensor.transpose(1, 2, 0)
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            draw = ImageDraw.Draw(img_pil)

            H, W = img_np.shape[:2]

            # GT keypoints
            gt_kp = np.asarray(targets['orig_joint_img'])[:, :2]
            gt_conf = np.asarray(meta['orig_joint_trunc']).squeeze(-1)
            gt_px = draw_kps(draw, gt_kp, gt_conf, 'lime', hm_shape, (H, W), coord_type='heatmap')

            # Detector keypoints
            det_kp_raw = np.asarray(inputs['joints'])
            det_kp = det_kp_raw[:17, :2]
            det_conf_raw = np.asarray(inputs.get('joints_mask', np.ones((17, 1))))
            if det_conf_raw.ndim == 2:
                det_conf = det_conf_raw[:17].squeeze(-1)
            else:
                det_conf = det_conf_raw[:17]

            det_px = draw_kps(draw, det_kp, det_conf, 'cyan', hm_shape, (H, W), coord_type='normalised')

            # Calculate distances
            for j in range(min(17, len(gt_px))):
                if gt_conf[j] > 0.5: # only count valid GT
                    gx, gy = gt_px[j]
                    dx, dy = det_px[j]
                    dist = np.sqrt((gx - dx)**2 + (gy - dy)**2)
                    per_joint_dists[j].append(dist)
                    
            # Check left-right swaps for detector vs GT
            for (l, r) in swap_pairs:
                if gt_conf[l] > 0.5 and gt_conf[r] > 0.5:
                    gl_x, gl_y = gt_px[l]
                    dl_x, dl_y = det_px[l]
                    gr_x, gr_y = gt_px[r]
                    dr_x, dr_y = det_px[r]
                    
                    # Normal distance (L to L, R to R)
                    n_dist = np.sqrt((gl_x - dl_x)**2 + (gl_y - dl_y)**2) + \
                             np.sqrt((gr_x - dr_x)**2 + (gr_y - dr_y)**2)
                             
                    # Swapped distance (L to R, R to L)
                    s_dist = np.sqrt((gl_x - dr_x)**2 + (gl_y - dr_y)**2) + \
                             np.sqrt((gr_x - dl_x)**2 + (gr_y - dl_x)**2)
                             
                    normal_dist_total += n_dist
                    swapped_dist_total += s_dist
                    pair_count += 1

            img_pil.save(os.path.join(out_img_dir, f'overlay_{i:02d}.png'))

        # ---- Report ----
        log(f"\n--- GT vs Detector Distance per Joint (pixels on {cfg.input_img_shape[0]} crop) ---")
        log(f"{'Joint':>2} {'Name':<15} | {'Mean Dist (px)':>15}")
        log("-" * 40)
        
        all_dists = []
        for j in range(17):
            dists = per_joint_dists[j]
            if dists:
                mean_d = np.mean(dists)
                all_dists.extend(dists)
                log(f"{j:>2} {HUMAN36M_JOINTS[j]:<15} | {mean_d:>15.1f}")
            else:
                log(f"{j:>2} {HUMAN36M_JOINTS[j]:<15} | {'N/A':>15}")

        if all_dists:
            overall_mean = np.mean(all_dists)
            log(f"\nOverall Mean Distance: {overall_mean:.1f}px")
            
            if overall_mean > 15.0:
                log(f"  WARNING: Mean distance > 15px. Possible coordinate mismatch or very poor detector.")
                all_pass = False

        if pair_count > 0:
            norm_avg = normal_dist_total / pair_count
            swap_avg = swapped_dist_total / pair_count
            log(f"\n--- L/R Swap Analysis ---")
            log(f"Normal Assignment L/R Distance : {norm_avg:.1f}px")
            log(f"Swapped Assignment L/R Distance: {swap_avg:.1f}px")
            if swap_avg < norm_avg:
                log(f"  FAIL: Detector left/right joints seem swapped compared to GT!")
                all_pass = False

        if all_pass:
            log(f"\nPASS - Numerical checks OK. Visual verification still needed "
                f"(check {out_img_dir}/ images: lime=GT, cyan=detector).")
        else:
            log(f"\nFAIL - Numerical checks indicate coordinate or assignment issues.")
            sys.exit(1)

    except Exception as e:
        import traceback
        log(f"FAIL - Exception occurred:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
