import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from core.config import cfg
from data_final.CrowdPose.dataset import CrowdPose
from data_final.Human36M.dataset import Human36M
from data_final.MSCOCO.dataset import MSCOCO
from data_final.MuCo.dataset import MuCo
from data_final.PW3D.dataset import PW3D
from utils.h36m_adapter import HUMAN36M_JOINTS, convert_smpl_to_human36m


TRAIN_DATASETS = {
    'Human36M': Human36M,
    'MuCo': MuCo,
    'MSCOCO': MSCOCO,
    'CrowdPose': CrowdPose,
}


class Human36M17Dataset(Dataset):
    JOINT_KEYS = {
        'orig_joint_img', 'fit_joint_img', 'orig_joint_cam', 'fit_joint_cam',
        'lift_pose3d', 'reg_pose3d',
    }
    MASK_KEYS = {
        'orig_joint_valid', 'orig_joint_trunc', 'fit_joint_trunc',
        'lift_pose3d_valid', 'reg_pose3d_valid',
    }
    # Cac key giu lai khi cfg.DATASET.lift_only (train/eval MotionBERT)
    LIFT_INPUT_KEYS = {'img', 'joints', 'joints_mask'}
    LIFT_TARGET_KEYS_TRAIN = {'orig_joint_img', 'orig_joint_cam'}
    LIFT_TARGET_KEYS_EVAL = {'smpl_mesh_cam', 'orig_joint_cam'}   # 3DPW: mesh; Human36M test: joint mocap
    LIFT_META_KEYS = {'orig_joint_valid', 'orig_joint_trunc', 'is_3D'}

    def __init__(self, dataset):
        self.dataset = dataset
        self.joint_num = len(HUMAN36M_JOINTS)
        self.joints_name = HUMAN36M_JOINTS
        self.joint_regressor_human36 = dataset.h36m_joint_regressor
        self.mesh_model = dataset.smpl
        self.vertex_num = dataset.vertex_num

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        # Guard against infinite recursion: if `dataset` itself is missing
        # (e.g. during unpickling before __init__ restores it), don't recurse.
        if name == 'dataset':
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def _convert_joint(self, value):
        """Reorder joints to H36M-17. Arrays already in H36M-17 pass through;
        source-joint arrays (matching the wrapped dataset) are name-mapped."""
        array = np.asarray(value)
        n = array.shape[0]
        if n == len(HUMAN36M_JOINTS):
            return array.astype(np.float32)
        if n == len(self.dataset.joints_name):
            zeros = np.zeros_like(array, dtype=np.float32)
            return convert_smpl_to_human36m(array, zeros, self.dataset.joints_name)[0]
        raise ValueError(
            f"Unexpected joint count {n}; expected {len(HUMAN36M_JOINTS)} (H36M) "
            f"or {len(self.dataset.joints_name)} (source)."
        )

    def _convert_mask(self, value):
        array = np.asarray(value)
        n = array.shape[0]
        if n == len(HUMAN36M_JOINTS):
            return array.astype(np.float32)
        if n == len(self.dataset.joints_name):
            zeros = np.zeros_like(array, dtype=np.float32)
            return convert_smpl_to_human36m(zeros, array, self.dataset.joints_name)[1]
        raise ValueError(
            f"Unexpected mask joint count {n}; expected {len(HUMAN36M_JOINTS)} (H36M) "
            f"or {len(self.dataset.joints_name)} (source)."
        )

    def __getitem__(self, index):
        inputs, targets, meta = self.dataset[index]
        inputs, targets, meta = dict(inputs), dict(targets), dict(meta)
        joints = np.asarray(inputs['joints'])
        if joints.shape[0] == len(HUMAN36M_JOINTS):
            # Dataset da tra thang H36M-17 (cfg.DATASET.lift_only) -> giu nguyen
            joints = joints.astype(np.float32)
            joints_mask = np.asarray(inputs['joints_mask']).astype(np.float32).reshape(-1, 1)
        else:
            joints, joints_mask = convert_smpl_to_human36m(
                inputs['joints'], inputs['joints_mask'], self.dataset.joints_name,
            )
        # PW3D tra 3 cot (x, y, conf OpenPose), H36M/MuCo/COCO/CrowdPose tra 2 cot -> collate loi khi tron.
        # Them cot thu 3 = mask cho bo 2 cot de moi dataset cung shape (17, 3). Model chi dung [..., :2] + joints_mask.
        if joints.shape[-1] == 2:
            joints = np.concatenate([joints, joints_mask.reshape(-1, 1)], axis=-1).astype(np.float32)
        inputs['joints'], inputs['joints_mask'] = joints, joints_mask
        inputs['joints'][..., 0] = inputs['joints'][..., 0] / cfg.output_hm_shape[2] * 2 - 1
        inputs['joints'][..., 1] = inputs['joints'][..., 1] / cfg.output_hm_shape[1] * 2 - 1
        # Khớp không hợp lệ (mask=0) có toạ độ rác (đến ~-3.6 sau chuẩn hóa) -> đặt về 0.
        # Mask vẫn được truyền riêng làm confidence cho MotionBERT.
        inputs['joints'] = inputs['joints'] * (inputs['joints_mask'] > 0)
        for key in self.JOINT_KEYS:
            if key in targets:
                targets[key] = self._convert_joint(targets[key])
        for key in self.MASK_KEYS:
            if key in meta:
                meta[key] = self._convert_mask(meta[key])
        if cfg.DATASET.lift_only:
            # Giu dung mot bo key chung cho moi dataset de collate duoc khi tron (PW3D tra nhieu key hon H36M/MuCo).
            is_train = getattr(self.dataset, 'data_split', '') == 'train'
            target_keys = self.LIFT_TARGET_KEYS_TRAIN if is_train else self.LIFT_TARGET_KEYS_EVAL
            inputs = {k: v for k, v in inputs.items() if k in self.LIFT_INPUT_KEYS}
            targets = {k: v for k, v in targets.items() if k in target_keys}
            meta = {k: v for k, v in meta.items() if k in self.LIFT_META_KEYS}
        return inputs, targets, meta


def get_train_dataset(name, args):
    if '3dpw' in name or name == 'PW3D':
        # PW3D chỉ dùng để train với split '3dpw-train'.
        if name not in ('3dpw-train', 'PW3D'):
            raise ValueError(
                f"get_train_dataset không hỗ trợ '{name}' cho train. "
                f"Dùng '3dpw-train' (các split '3dpw'/'3dpw-pc'/... là split test)."
            )
        return Human36M17Dataset(PW3D(transforms.ToTensor(), data_name='3dpw-train'))
    else:
        # Các dataset còn lại thì truyền 'train'
        return Human36M17Dataset(TRAIN_DATASETS[name](transforms.ToTensor(), 'train'))



def get_test_dataset(name, args):
    # name: '3dpw' | '3dpw-pc' | '3dpw-oc' | '3dpw-crowd' | '3dpw-val' | 'Human36M'
    if name == 'Human36M':
        # Split test cua H36M (S9, S11, camera 4). Hien chi ho tro khi cfg.DATASET.lift_only=True.
        if not cfg.DATASET.lift_only:
            raise ValueError("Test 'Human36M' hien chi ho tro cfg.DATASET.lift_only=True (finetune MotionBERT).")
        return Human36M17Dataset(Human36M(transforms.ToTensor(), 'test'))
    if name == 'MuCo':
        raise ValueError("MuCo khong co split test (chi co train). Bo 'MuCo' khoi test_list.")
    return Human36M17Dataset(PW3D(transforms.ToTensor(), data_name=name))