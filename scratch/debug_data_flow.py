"""
Debug script to trace the data flow and identify unit/shape mismatches
causing MPJPE=575mm in Teacher training.

Run from FuseKin root:
    python scratch/debug_data_flow.py --cfg config/train_teacher.yml --gpu 0
"""
import os, sys
sys.path.append('./lib')
sys.path.append('./')
import argparse
import torch
import numpy as np
from core.config import cfg, update_config

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='config/train_teacher.yml')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
update_config(args.cfg)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from utils.jotr_dataset import get_train_dataset, get_test_dataset

print("=" * 60)
print("CONFIG DUMP (relevant)")
print("=" * 60)
print(f"  MODEL.name          = {cfg.MODEL.name}")
print(f"  DATASET.train_list  = {cfg.DATASET.train_list}")
print(f"  DATASET.test_list   = {cfg.DATASET.test_list}")
print(f"  DATASET.input_joint_set  = {cfg.DATASET.input_joint_set}")
print(f"  DATASET.target_joint_set = {cfg.DATASET.target_joint_set}")
print(f"  DATASET.seqlen      = {cfg.DATASET.seqlen}")
print(f"  output_hm_shape     = {cfg.output_hm_shape}")
print()

# ===== 1. TRAIN DATASET =====
print("=" * 60)
print("1. TRAIN DATASET: Loading first sample...")
print("=" * 60)
train_ds = get_train_dataset(cfg.DATASET.train_list[0], args)
print(f"  Dataset type: {type(train_ds)}")
print(f"  Dataset len:  {len(train_ds)}")
print(f"  joints_name:  {train_ds.joints_name}")
print(f"  joint_num:    {train_ds.joint_num}")
print()

inputs, targets, meta = train_ds[0]

print("--- INPUTS ---")
for k, v in inputs.items():
    v = np.asarray(v) if not isinstance(v, np.ndarray) else v
    print(f"  inputs['{k}']: shape={v.shape}, dtype={v.dtype}, "
          f"min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}")

print("\n--- TARGETS ---")
for k, v in targets.items():
    v = np.asarray(v) if not isinstance(v, np.ndarray) else v
    print(f"  targets['{k}']: shape={v.shape}, dtype={v.dtype}, "
          f"min={v.min():.6f}, max={v.max():.6f}, mean={v.mean():.6f}")

print("\n--- META ---")
for k, v in meta.items():
    v = np.asarray(v) if not isinstance(v, np.ndarray) else v
    if v.ndim > 0:
        print(f"  meta['{k}']: shape={v.shape}, dtype={v.dtype}, "
              f"min={v.min():.4f}, max={v.max():.4f}")
    else:
        print(f"  meta['{k}']: value={v}")

# ===== 2. KEY CHECK: units of targets =====
print()
print("=" * 60)
print("2. UNIT CHECK — orig_joint_cam vs fit_joint_cam")
print("=" * 60)
orig_cam = np.asarray(targets['orig_joint_cam'])
fit_cam = np.asarray(targets['fit_joint_cam'])
print(f"  orig_joint_cam: shape={orig_cam.shape}")
print(f"    Range: [{orig_cam.min():.6f}, {orig_cam.max():.6f}]")
print(f"    Abs mean: {np.abs(orig_cam).mean():.6f}")
print(f"    Sample values (first 3 joints):")
for j in range(min(3, orig_cam.shape[0])):
    print(f"      joint[{j}] = {orig_cam[j]}")
print(f"  fit_joint_cam: shape={fit_cam.shape}")
print(f"    Range: [{fit_cam.min():.6f}, {fit_cam.max():.6f}]")
print(f"    Abs mean: {np.abs(fit_cam).mean():.6f}")
print(f"    Sample values (first 3 joints):")
for j in range(min(3, fit_cam.shape[0])):
    print(f"      joint[{j}] = {fit_cam[j]}")

if np.abs(orig_cam).mean() > 10:
    print("  >>> WARNING: orig_joint_cam seems in MILLIMETERS (mean > 10)")
elif np.abs(orig_cam).mean() < 5:
    print("  >>> INFO: orig_joint_cam seems in METERS (mean < 5)")
else:
    print("  >>> INFO: orig_joint_cam in ambiguous range")

# ===== 3. TEST DATASET =====
print()
print("=" * 60)
print("3. TEST DATASET: Loading first sample...")
print("=" * 60)
test_ds = get_test_dataset(cfg.DATASET.test_list[0], args)
print(f"  Dataset type: {type(test_ds)}")
print(f"  Dataset len:  {len(test_ds)}")
t_inputs, t_targets, t_meta = test_ds[0]

print("--- TEST INPUTS ---")
for k, v in t_inputs.items():
    v = np.asarray(v) if not isinstance(v, np.ndarray) else v
    if v.ndim > 0:
        print(f"  inputs['{k}']: shape={v.shape}, dtype={v.dtype}, "
              f"min={v.min():.4f}, max={v.max():.4f}")
    else:
        print(f"  inputs['{k}']: {v}")

print("\n--- TEST TARGETS ---")
for k, v in t_targets.items():
    v = np.asarray(v) if not isinstance(v, np.ndarray) else v
    print(f"  targets['{k}']: shape={v.shape}, dtype={v.dtype}, "
          f"min={v.min():.6f}, max={v.max():.6f}, mean(abs)={np.abs(v).mean():.6f}")

if 'smpl_mesh_cam' in t_targets:
    mesh = np.asarray(t_targets['smpl_mesh_cam'])
    print(f"\n  >>> TEST mesh (smpl_mesh_cam) abs_mean = {np.abs(mesh).mean():.6f}")
    if np.abs(mesh).mean() > 10:
        print("  >>> mesh unit = MILLIMETERS")
    else:
        print("  >>> mesh unit = METERS")

# ===== 4. TRAINING LOSS CHECK: simulate Teacher_Trainer loss computation =====
print()
print("=" * 60)
print("4. SIMULATED LOSS CHECK")
print("   Compare loss inputs vs GT in Teacher_Trainer")
print("=" * 60)

# In Teacher_Trainer, we do:
#   input_pose = inputs['joints'][:, smpl_to_h36m_idx]
#   BUT inputs['joints'] is ALREADY Human36M-17 (converted by Human36M17Dataset)!
#   So smpl_to_h36m_idx mapping is DOUBLE-APPLIED!

print("\n  [CRITICAL CHECK] Does Teacher_Trainer double-convert joints?")
print(f"    Dataset joints_name = {train_ds.joints_name}")
print(f"    joint_num = {train_ds.joint_num}")

# Teacher_Trainer builds smpl_to_h36m_idx like this:
smpl_joints = train_ds.joints_name
h36m_joints = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')

print(f"\n    smpl_joints (from dataset) = {smpl_joints}")
print(f"    h36m_joints (hardcoded)    = {h36m_joints}")

if smpl_joints == h36m_joints:
    print("\n  >>> smpl_joints == h36m_joints !!!")
    print("  >>> This means smpl_to_h36m_idx = [0, 1, 2, ..., 16] (identity)")
    print("  >>> Slicing is redundant but NOT HARMFUL")
else:
    try:
        idx_map = [smpl_joints.index(name) for name in h36m_joints]
        print(f"\n    smpl_to_h36m_idx = {idx_map}")
        # Check if this is identity
        if idx_map == list(range(17)):
            print("  >>> Identity mapping — slicing is redundant but safe")
        else:
            print("  >>> NON-IDENTITY mapping! Data will be reshuffled!")
    except ValueError as e:
        print(f"\n  >>> ERROR: Joint name not found: {e}")
        print("  >>> This would cause a crash!")

# ===== 5. EVALUATION: check unit mismatch =====
print()
print("=" * 60)
print("5. EVALUATION UNIT CHECK")
print("   In evaluate(), results are computed as:")
print("   mesh_error = np.sqrt(...) * 1000  (meter -> mm)")
print("   So evaluation expects mesh in METERS.")
print("=" * 60)

# Teacher's RegressorSpin outputs SMPL vertices.
# SMPL outputs are typically in METERS.
# But the LOSS computes L1 between pred_pose (from J_regressor * mesh)
# and gt_fit_joint_cam. What unit is gt_fit_joint_cam?
print(f"\n  gt_fit_joint_cam abs_mean = {np.abs(fit_cam).mean():.6f}")
print(f"  gt_orig_joint_cam abs_mean = {np.abs(orig_cam).mean():.6f}")

# In PW3D dataset __getitem__ (train):
#   orig_joint_cam = smpl_joint_cam (from SMPL output, raw)
#   Then: orig_joint_cam = np.dot(rot_aug_mat, ...) / 1000  # mm -> m
#   And:  smpl_joint_cam = smpl_joint_cam - root then /1000  # mm -> m
print("\n  PW3D dataset converts to METERS (/ 1000):")
print("    orig_joint_cam /= 1000 (line 331)")
print("    fit_joint_cam  /= 1000 (line 343)")

# So GT is in METERS, and SMPL output is in METERS.
# Loss should be fine unit-wise...unless MotionBERT output mixes things up.

# ===== 6. Check what MotionBERT 3D output looks like vs GT =====
print()
print("=" * 60)
print("6. MotionBERT OUTPUT SCALE CHECK")
print("   MotionBERT predicts 3D joints. What scale?")
print("   Check if its output is mm or m.")
print("=" * 60)

# MotionBERT was trained on Human3.6M (in mm typically)
# or in normalized coordinates. Need to check.
print("  MotionBERT usually outputs in mm (H3.6M convention).")
print("  But Teacher uses this output as joints input to cross-attention,")
print("  then feeds fused features to RegressorSpin.")
print("  RegressorSpin internally does SMPL forward → output in meters.")
print()
print("  The question is: is the LOSS comparing things in the same unit?")
print("  loss_smpl_joint_cam = L1(pred_pose, gt_fit_joint_cam)")
print(f"    gt_fit_joint_cam is in {'METERS' if np.abs(fit_cam).mean() < 5 else 'MM'}")
print("    pred_pose = J_regressor @ pred_mesh (from SMPL = METERS)")
print("  >>> If both in METERS, loss is OK unit-wise.")

# ===== 7. Double slicing check =====
print()
print("=" * 60)
print("7. DOUBLE SLICING CHECK")
print("   Teacher_Trainer does: inputs['joints'][:, smpl_to_h36m_idx]")
print("   But Human36M17Dataset already converted 30 joints → 17 joints")
print("=" * 60)

# In Teacher_Trainer, config says input_joint_set='human36' and target_joint_set='human36'
# Dataset already wraps PW3D in Human36M17Dataset which outputs 17 joints
# Then Teacher_Trainer does: input_pose = inputs['joints'][:, self.smpl_to_h36m_idx]
# Since joints_name is already h36m (17 joints), smpl_to_h36m_idx is identity [0..16]
# So this is harmless

# BUT for targets: 
# gt_orig_joint_cam = targets['orig_joint_cam'][:, smpl_to_h36m_idx]
# targets['orig_joint_cam'] has been converted to 17 h36m joints by dataset
# smpl_to_h36m_idx is identity → OK

# BUT WAIT: targets like 'orig_joint_cam', 'fit_joint_cam' are converted by
# Human36M17Dataset.__getitem__ which calls _convert_joint().
# Is this correct for 3D camera coords?

print("\n  Human36M17Dataset converts ALL joint keys:")
print("    JOINT_KEYS = {'orig_joint_img', 'fit_joint_img', 'orig_joint_cam', 'fit_joint_cam', ...}")
print("  This converts 30 SMPL joints → 17 H36M joints")
print()

# The original PW3D returns fit_joint_cam as root-relative, /1000 (meters)
# with 30 SMPL joints. Human36M17Dataset selects 17 of them.
# Then Teacher_Trainer slices again with smpl_to_h36m_idx (identity).
# Should be OK.

# ===== 8. THE REAL PROBLEM CHECK: RegressorSpin output vs GT format =====
print()
print("=" * 60)
print("8. THE REAL CHECK: pred vs GT alignment")
print("=" * 60)

# In Teacher_Trainer:
#   pred_mesh = model_output['smpl_mesh_cam']  → SMPL vertices (B, 6890, 3) in METERS
#   pred_pose = J_regressor @ pred_mesh        → joints from mesh (B, 17, 3) in METERS
#   gt_fit_joint_cam                           → (B, 17, 3) in METERS, root-relative
#
#   BUT: pred_pose is NOT root-relative!
#   RegressorSpin does SMPL forward and returns raw vertices.
#   The vertices are in camera space, NOT root-relative.
#   So pred_pose has absolute camera coords, while gt_fit_joint_cam is root-relative!

print("  pred_mesh = model_output['smpl_mesh_cam']")  
print("  pred_pose = J_regressor @ pred_mesh")
print("  gt_fit_joint_cam = root-relative, meters (from dataset)")
print()
print("  >>> QUESTION: Is pred_pose root-relative or absolute?")
print("  >>> RegressorSpin → SMPL → absolute camera coords")
print("  >>> gt_fit_joint_cam → root-relative (root subtracted in PW3D dataset)")
print("  >>> IF NOT ROOT-RELATIVE, LOSS IS COMPUTING GARBAGE!")
print()

# In the reference (Trainer, non-teacher):
#   pred_pose also = J_regressor @ pred_mesh
#   Same issue exists... unless Multimodel handles this differently.
# Let's check if the reference Trainer also has this issue or handles it.
print("  Check reference Trainer (ARTS mode):")
print("    Same formula: pred_pose = J_regressor @ pred_mesh")
print("    Same gt: gt_fit_joint_cam (root-relative, meters)")
print("    If reference works with MPJPE ~80mm, then pred_mesh must be root-relative too")
print("    OR the loss doesn't contribute much due to weighting")

# ===== 9. Check if '3dpw-train' exists as a train dataset key =====
print()
print("=" * 60)
print("9. TRAIN DATASET KEY CHECK")
print("=" * 60)
print(f"  cfg.DATASET.train_list = {cfg.DATASET.train_list}")
print(f"  TRAIN_DATASETS keys (in jotr_dataset.py): {['Human36M', 'MuCo', 'MSCOCO', 'CrowdPose']}")
if cfg.DATASET.train_list[0] not in ['Human36M', 'MuCo', 'MSCOCO', 'CrowdPose', 'PW3D']:
    print(f"  >>> '{cfg.DATASET.train_list[0]}' is NOT in TRAIN_DATASETS!")
    print("  >>> But in get_train_dataset, it uses TRAIN_DATASETS dict")
    print("  >>> This would CRASH unless there's special handling for '3dpw-train'!")

# Check if '3dpw-train' is handled differently
print()
print("  get_train_dataset code: return Human36M17Dataset(TRAIN_DATASETS[name](..., 'train'))")
print(f"  For name='{cfg.DATASET.train_list[0]}', this would fail if key not in dict")

# Actually wait - in base.py prepare_network, get_dataloader calls
# get_jotr_train_dataset. But '3dpw-train' is not in TRAIN_DATASETS!
# It's actually handled by... let me check

print()
print("=" * 60)
print("DONE. Review output above for data/unit mismatches.")
print("=" * 60)
