import numpy as np


HUMAN36M_JOINTS = (
    'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle',
    'Torso', 'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
    'R_Shoulder', 'R_Elbow', 'R_Wrist',
)


def convert_smpl_to_human36m(joints, joint_mask, smpl_joints_name):
    """Convert JOTR's post-processed SMPL joint arrays to Human36M-17.

    This adapter intentionally performs name-based reordering only. Missing
    joints remain zero and are marked invalid in the returned mask.
    """
    joints = np.asarray(joints)
    joint_mask = np.asarray(joint_mask)
    output_shape = (len(HUMAN36M_JOINTS),) + joints.shape[1:]
    converted_joints = np.zeros(output_shape, dtype=np.float32)
    converted_mask = np.zeros((len(HUMAN36M_JOINTS),) + joint_mask.shape[1:], dtype=np.float32)

    for target_idx, joint_name in enumerate(HUMAN36M_JOINTS):
        if joint_name not in smpl_joints_name:
            continue
        source_idx = smpl_joints_name.index(joint_name)
        converted_joints[target_idx] = joints[source_idx]
        converted_mask[target_idx] = joint_mask[source_idx]

    return converted_joints, converted_mask


def h36m_joint_indices(names):
    """Doi danh sach ten khop H36M-17 sang chi so. Bao loi neu ten sai (tranh go nham trong yml)."""
    idx = []
    for n in names:
        if n not in HUMAN36M_JOINTS:
            raise ValueError(f"Khop '{n}' khong co trong H36M-17: {HUMAN36M_JOINTS}")
        idx.append(HUMAN36M_JOINTS.index(n))
    return idx


def drop_joints_2d(xy, conf, drop_idx):
    """Xoa cac khop drop_idx khoi dau vao MotionBERT: toa do = 0, confidence = 0.
    xy: (..., 17, 2) tensor, conf: (..., 17, 1) tensor. Tra ve ban sao, khong sua tai cho."""
    if not drop_idx:
        return xy, conf
    xy = xy.clone()
    conf = conf.clone()
    xy[..., drop_idx, :] = 0
    conf[..., drop_idx, :] = 0
    return xy, conf


def convert_sample_inputs(inputs, smpl_joints_name):
    """Return a copy of a JOTR input dict with Human36M-17 2D joints."""
    converted = dict(inputs)
    joints, joint_mask = convert_smpl_to_human36m(
        inputs['joints'],
        inputs['joints_mask'],
        smpl_joints_name,
    )
    converted['joints'] = joints
    converted['joints_mask'] = joint_mask
    return converted