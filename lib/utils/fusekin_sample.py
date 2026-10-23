"""
lib/utils/fusekin_sample.py
Mau train FuseKin (Teacher / Student, CO anh) THONG NHAT cho Human36M / MuCo / MSCOCO / CrowdPose
(cfg.DATASET.fusekin_unified). Tra ve DUNG bo key cua nhanh train 3DPW (data_final/PW3D/dataset.py) de cac bo
tron chung duoc va Teacher_Trainer / Student_Trainer doc nhu 3DPW:

  inputs : img, joints (17, 3) [x, y, conf] khung output_hm_shape (Halpe-26 -> H36M-17), joints_mask (17, 1), cam_bbox (3,)
  targets: orig_joint_img (17, 3) GT 2D (chieu tu mesh), orig_joint_cam = fit_joint_cam (17, 3) met root-relative
           (H36M regressor x mesh SMPL, giong 3DPW / evaluate), pose_param (72,), shape_param (10,), smpl_mesh_cam (6890, 3)
  meta   : img_path, img2bb_trans, orig_joint_valid, orig_joint_trunc, fit_param_valid (72,), fit_joint_trunc,
           is_valid_fit, is_3D

Quy uoc camera: 3D theo camera THAT (H36M / MuCo). MSCOCO / CrowdPose: pseudo-GT NeuralAnnot fit voi camera AO dat
tai tam bbox ("nhin thang") -> use_view_rot xoay 3D + goc root sang camera uoc luong est_cam (giong lifter,
utils/lift_pseudo3d.py) de 5 bo (ke ca 3DPW test) cung quy uoc. GT 2D van chieu bang camera NeuralAnnot (dung pixel).
"""
import cv2
import numpy as np

from core.config import cfg
from utils.transforms import cam2pixel
from utils.cam_bbox import cam_bbox_feat, view_rot

NO_PARAM_JOINTS = ('L_Ankle', 'R_Ankle', 'L_Toe', 'R_Toe', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand')   # giong 3DPW


def fusekin_train_sample(ds, img, img2bb_trans, rot, do_flip, bbox, img_shape, img_path,
                         xy, mask, mesh_cam, smpl_pose, smpl_shape, proj_cam, feat_cam, use_view_rot=False):
    """
    ds        : dataset (can h36m_joint_regressor, smpl, joints_name)
    xy, mask  : 2D Halpe (17, 2) khung output_hm_shape (da lat + affine), conf (17, 1)
    mesh_cam  : (6890, 3) MET, he camera proj_cam, DA LAT (get_smpl_coord cua tung dataset)
    smpl_pose : (72,) axis-angle, goc root trong cung he camera (da lat); smpl_shape (10,)
    proj_cam  : camera chieu mesh_cam ra dung pixel anh (da lat) -> GT 2D
    feat_cam  : camera tinh cam_bbox (camera that, hoac est_cam cho MSCOCO / CrowdPose)
    """
    h36m = np.dot(ds.h36m_joint_regressor, mesh_cam).astype(np.float32)               # (17, 3) met
    root = h36m[0:1]                                                                   # Pelvis

    # --- GT 2D: chieu khop H36M-17 cua mesh ra anh -> khung crop -> khung output_hm_shape ---
    j2d = cam2pixel(h36m, proj_cam['focal'], proj_cam['princpt'])
    xy1 = np.concatenate((j2d[:, :2], np.ones_like(j2d[:, :1])), 1)
    j2d[:, :2] = np.dot(img2bb_trans, xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
    j2d[:, 0] = j2d[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
    j2d[:, 1] = j2d[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
    trunc = ((j2d[:, 0] >= 0) * (j2d[:, 0] < cfg.output_hm_shape[2]) *
             (j2d[:, 1] >= 0) * (j2d[:, 1] < cfg.output_hm_shape[1])).reshape(-1, 1).astype(np.float32)

    # --- 3D: root-relative -> (xoay camera ao -> camera uoc luong) -> xoay augmentation ---
    a = np.deg2rad(-rot)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]], dtype=np.float32)
    if use_view_rot:
        dx, dy, _ = cam_bbox_feat(bbox, feat_cam, img_shape, do_flip, 0.0)            # tam bbox (da lat), chua xoay aug
        R = np.dot(R, view_rot(dx, dy))
    joint_cam = np.dot(h36m - root, R.T).astype(np.float32)
    mesh = np.dot(mesh_cam - root, R.T).astype(np.float32)
    pose = np.array(smpl_pose, dtype=np.float32).reshape(-1, 3).copy()
    root_rot, _ = cv2.Rodrigues(pose[0])
    pose[0] = cv2.Rodrigues(np.dot(R, root_rot))[0].reshape(3)

    param_valid = np.ones((ds.smpl.orig_joint_num, 3), dtype=np.float32)
    for name in NO_PARAM_JOINTS:
        param_valid[ds.joints_name.index(name)] = 0

    inputs = {'img': img,
              'joints': np.concatenate([xy, mask], axis=1).astype(np.float32),
              'joints_mask': mask.astype(np.float32),
              'cam_bbox': cam_bbox_feat(bbox, feat_cam, img_shape, do_flip, rot)}
    targets = {'orig_joint_img': j2d.astype(np.float32), 'orig_joint_cam': joint_cam, 'fit_joint_cam': joint_cam.copy(),
               'pose_param': pose.reshape(-1), 'shape_param': np.asarray(smpl_shape, dtype=np.float32),
               'smpl_mesh_cam': mesh}
    meta = {'img_path': img_path, 'img2bb_trans': img2bb_trans.astype(np.float32),
            'orig_joint_valid': np.ones((17, 1), dtype=np.float32), 'orig_joint_trunc': trunc,
            'fit_param_valid': param_valid.reshape(-1), 'fit_joint_trunc': np.ones((17, 1), dtype=np.float32),
            'is_valid_fit': 1.0, 'is_3D': 1.0}
    return inputs, targets, meta


def keep_unified(datalist, name, param_key='smpl_param'):
    """fusekin_unified: chi giu mau co SMPL (pseudo-)GT va Halpe-26 (Teacher can GT 3D cho moi mau)."""
    out = [d for d in datalist if d.get(param_key) is not None and d.get('halpe_h36m') is not None]
    print(f'[{name} fusekin_unified] giu {len(out)} / {len(datalist)} mau (co SMPL + Halpe-26)')
    return out
