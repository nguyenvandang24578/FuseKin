import os, sys
sys.path.append('./lib')
sys.path.append('./')
import argparse
import yaml
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# FuseKin imports
from lib.core.config import update_config, cfg
from lib.utils.jotr_dataset import get_train_dataset
from models.DSTformer import DSTformer
from MotionBERT.lib.model.loss import loss_mpjpe, n_mpjpe, loss_velocity, \
    loss_limb_var, loss_limb_gt, loss_angle, loss_angle_velocity

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config/finetune_motionbert_pw3d.yaml", help="Path to the unified config file (DATASET + MOTIONBERT).")
    parser.add_argument('--pretrained', default='', type=str, help='pretrained checkpoint path (e.g. MotionBERT/checkpoint/pretrain/MB_release.bin)')
    parser.add_argument('--gpu', type=str, default='0,1', help='assign multi-gpus by comma concat, e.g. "0,1" or "0,1,2,3"')
    opts = parser.parse_args()
    return opts


def train_epoch(args, mb_cfg, model, train_loader, optimizer, device):
    model.train()
    losses = {
        '3d_pos': 0.0, '3d_scale': 0.0, '3d_velocity': 0.0,
        'lv': 0.0, 'lg': 0.0, 'angle': 0.0, 'angle_velocity': 0.0, 'total': 0.0
    }
    n_batches = 0

    for i, (inputs_b, targets_b, meta_b) in tqdm(enumerate(train_loader), total=len(train_loader)):
        joints_2d = inputs_b['joints'].to(device)       # (B, 17, 2)
        joints_mask = inputs_b['joints_mask'].to(device) # (B, 17, 1)
        target_3d = targets_b['orig_joint_cam'].to(device) # (B, 17, 3)

        # Prepare MotionBERT input (append mask as confidence)
        pose2d_3ch = torch.cat([joints_2d, joints_mask], dim=-1) # (B, 17, 3)
        
        # Simulate temporal sequence of length 243
        mb_input = pose2d_3ch.unsqueeze(1).repeat(1, mb_cfg.maxlen, 1, 1) # (B, 243, 17, 3)
        target_3d_seq = target_3d.unsqueeze(1).repeat(1, mb_cfg.maxlen, 1, 1) # (B, 243, 17, 3)
        
        with torch.no_grad():
            # Root relative (same as MotionBERT train.py line 168)
            mb_input[..., :2] = mb_input[..., :2] - mb_input[:, :, 0:1, :2]
            target_3d_seq = target_3d_seq - target_3d_seq[:, :, 0:1, :]

        # Forward pass
        predicted_3d = model(mb_input) # (B, 243, 17, 3)

        optimizer.zero_grad()

        # ============================================================
        # Loss computation — 100% faithful to MotionBERT train.py L178-191
        # ============================================================
        loss_3d_pos   = loss_mpjpe(predicted_3d, target_3d_seq)
        loss_3d_scale = n_mpjpe(predicted_3d, target_3d_seq)
        loss_3d_vel   = loss_velocity(predicted_3d, target_3d_seq)
        loss_lv       = loss_limb_var(predicted_3d)
        loss_lg       = loss_limb_gt(predicted_3d, target_3d_seq)
        loss_a        = loss_angle(predicted_3d, target_3d_seq)
        loss_av       = loss_angle_velocity(predicted_3d, target_3d_seq)

        loss_total = loss_3d_pos + \
                     mb_cfg.lambda_scale       * loss_3d_scale + \
                     mb_cfg.lambda_3d_velocity * loss_3d_vel + \
                     mb_cfg.lambda_lv          * loss_lv + \
                     mb_cfg.lambda_lg          * loss_lg + \
                     mb_cfg.lambda_a           * loss_a  + \
                     mb_cfg.lambda_av          * loss_av

        loss_total.backward()
        optimizer.step()

        # Logging
        losses['3d_pos']          += loss_3d_pos.item()
        losses['3d_scale']        += loss_3d_scale.item()
        losses['3d_velocity']     += loss_3d_vel.item()
        losses['lv']              += loss_lv.item()
        losses['lg']              += loss_lg.item()
        losses['angle']           += loss_a.item()
        losses['angle_velocity']  += loss_av.item()
        losses['total']           += loss_total.item()
        n_batches += 1

    # Average over batches
    for k in losses:
        losses[k] /= n_batches
    return losses

def main():
    opts = parse_args()
    
    # Configure GPUs
    if opts.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu)
        print(f"Work on GPU(s): {opts.gpu}")

    # Load Unified Config — update_config handles both DATASET and MOTIONBERT sections
    update_config(opts.cfg)
    
    # MotionBERT hyperparameters from the MOTIONBERT section
    mb_cfg = cfg.MOTIONBERT

    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (Total GPUs active: {num_gpus})")

    # Prepare Dataset
    print("Loading 3DPW dataset...")
    train_dataset = get_train_dataset('3dpw-train', opts)
        
    # Create DataLoader with MotionBERT batch size
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=mb_cfg.batch_size,
        shuffle=True,
        num_workers=cfg.DATASET.workers,
        pin_memory=True
    )

    # Initialize MotionBERT (same as train.py L250-258)
    print("Initializing MotionBERT (DSTformer)...")
    model = DSTformer(
        dim_in=3, 
        dim_out=3, 
        dim_feat=mb_cfg.dim_feat, 
        dim_rep=mb_cfg.dim_rep, 
        depth=mb_cfg.depth, 
        num_heads=mb_cfg.num_heads, 
        mlp_ratio=mb_cfg.mlp_ratio, 
        norm_layer=nn.LayerNorm, 
        maxlen=mb_cfg.maxlen, 
        num_joints=mb_cfg.num_joints, 
        att_fuse=mb_cfg.att_fuse
    )

    # Wrap with DataParallel BEFORE loading weights (same as train.py L257-258)
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        print(f"Model wrapped in DataParallel on {num_gpus} GPU(s)")

    # Load Pretrained (same as train.py L260-272)
    if opts.pretrained:
        print(f"Loading pretrained weights from {opts.pretrained}")
        checkpoint = torch.load(opts.pretrained, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_pos'], strict=True)
    
    # Optimizer (same as train.py L288-289)
    lr = mb_cfg.learning_rate
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, 
        weight_decay=mb_cfg.weight_decay
    )
    lr_decay = mb_cfg.lr_decay
    
    os.makedirs('experiment/finetune_motionbert', exist_ok=True)

    # Training Loop (same as train.py L311-374)
    print(f"Starting finetuning for {mb_cfg.epochs} epochs...")
    min_loss = float('inf')
    
    for epoch in range(mb_cfg.epochs):
        start_time = time.time()
        
        losses = train_epoch(opts, mb_cfg, model, train_loader, optimizer, device)
        
        elapsed = (time.time() - start_time) / 60
        print(f"[Epoch {epoch+1}/{mb_cfg.epochs}] Time: {elapsed:.2f}m | LR: {lr:.6f} | "
              f"3d_pos: {losses['3d_pos']:.6f} | scale: {losses['3d_scale']:.6f} | "
              f"velocity: {losses['3d_velocity']:.6f} | total: {losses['total']:.6f}")
        
        # Decay learning rate exponentially (same as train.py L360-362)
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay

        # Save Checkpoints (same as train.py L364-374)
        chk_path_latest = "experiment/finetune_motionbert/latest_epoch.bin"
        torch.save({
            'epoch': epoch + 1,
            'lr': lr,
            'optimizer': optimizer.state_dict(),
            'model_pos': model.state_dict(),
            'min_loss': min_loss
        }, chk_path_latest)

        if (epoch + 1) % mb_cfg.checkpoint_frequency == 0:
            chk_path = f"experiment/finetune_motionbert/epoch_{epoch+1}.bin"
            torch.save({
                'epoch': epoch + 1,
                'lr': lr,
                'optimizer': optimizer.state_dict(),
                'model_pos': model.state_dict(),
                'min_loss': min_loss
            }, chk_path)
            
        if losses['3d_pos'] < min_loss:
            min_loss = losses['3d_pos']
            best_path = "experiment/finetune_motionbert/best_epoch.bin"
            torch.save({
                'epoch': epoch + 1,
                'lr': lr,
                'optimizer': optimizer.state_dict(),
                'model_pos': model.state_dict(),
                'min_loss': min_loss
            }, best_path)
            print(f"--> Saved new best model (MPJPE={min_loss:.6f}) to {best_path}")

    print("Finetuning completed.")

if __name__ == '__main__':
    main()
