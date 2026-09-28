import os
import sys
import json
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.config import update_config, cfg
from utils.jotr_dataset import get_train_dataset, get_test_dataset
from utils.smpl import SMPL

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='experiment/mesh_3dpw.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dir', type=str, default='logs/server_checks')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = os.path.join(args.out_dir, 'check_mask_3dpw.log')
    json_file = os.path.join(args.out_dir, 'mask_stats.json')

    def log(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=== check_mask_3dpw ===\n")

    try:
        if os.path.exists(args.cfg):
            update_config(args.cfg)
        
        # Load dataset
        train_ds = get_train_dataset('3dpw-train', args)
        test_ds = get_test_dataset('3dpw', args)
        smpl = SMPL()

        stats = {'train': {}, 'test': {}}
        
        for ds_name, ds in [('train', train_ds), ('test', test_ds)]:
            log(f"Processing {ds_name} dataset...")
            mask_sum = None
            total_samples = 0
            
            for i in tqdm(range(len(ds))):
                inputs, targets, meta = ds[i]
                mask = meta['fit_param_valid'] # (72,)
                mask_24 = mask.reshape(24, 3)[:, 0] # (24,)
                
                if mask_sum is None:
                    mask_sum = mask_24.copy()
                else:
                    mask_sum += mask_24
                total_samples += 1

            stats[ds_name]['total'] = total_samples
            stats[ds_name]['joints'] = {}
            for j_idx, joint_name in enumerate(smpl.joints_name):
                valid_count = float(mask_sum[j_idx])
                ratio = valid_count / total_samples if total_samples > 0 else 0
                stats[ds_name]['joints'][joint_name] = {
                    'valid': valid_count,
                    'total': total_samples,
                    'ratio': ratio
                }
                
                if ratio < 1.0:
                    status = "ALL SAMPLES" if ratio == 0 else "PARTIAL"
                    log(f"  [{ds_name}] {joint_name}: mask=0 ({status}) - {valid_count}/{total_samples} valid")

        with open(json_file, 'w') as f:
            json.dump(stats, f, indent=4)
        
        log(f"Wrote stats to {json_file}")
        log("PASS - Mask statistics collected successfully.")
    except Exception as e:
        import traceback
        log(f"FAIL - Exception occurred:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
