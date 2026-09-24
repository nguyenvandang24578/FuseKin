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
        inputs['joints'], inputs['joints_mask'] = convert_smpl_to_human36m(
            inputs['joints'], inputs['joints_mask'], self.dataset.joints_name,
        )
        inputs['joints'][..., 0] = inputs['joints'][..., 0] / cfg.output_hm_shape[2] * 2 - 1
        inputs['joints'][..., 1] = inputs['joints'][..., 1] / cfg.output_hm_shape[1] * 2 - 1
        for key in self.JOINT_KEYS:
            if key in targets:
                targets[key] = self._convert_joint(targets[key])
        for key in self.MASK_KEYS:
            if key in meta:
                meta[key] = self._convert_mask(meta[key])
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
    return Human36M17Dataset(PW3D(transforms.ToTensor(), data_name=name))