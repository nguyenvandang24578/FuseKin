import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'lib'))
sys.path.insert(0, ROOT_DIR)

from core.config import cfg, update_config
from utils.jotr_dataset import get_test_dataset
from utils.jotr_evaluation import evaluate_3dpw_subset
from models.ARTS import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ARTS with JOTR 3DPW subsets')
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=None)
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    state_dict = {
        key.removeprefix('module.'): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)


def main():
    args = parse_args()
    if args.cfg:
        update_config(args.cfg)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(
        num_joint=17,
        embed_dim=cfg.MODEL.hpe_dim,
        depth=cfg.MODEL.hpe_dep,
    ).to(device)
    load_checkpoint(model, args.checkpoint, device)

    batch_size = args.batch_size or cfg.TEST.batch_size
    results = {}
    for subset in ('3dpw', '3dpw-crowd', '3dpw-pc', '3dpw-oc'):
        dataset = get_test_dataset(subset, args)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.DATASET.workers,
            pin_memory=True,
        )
        results[subset] = evaluate_3dpw_subset(model, dataset, loader, device=device)
        print(subset, results[subset])


if __name__ == '__main__':
    main()