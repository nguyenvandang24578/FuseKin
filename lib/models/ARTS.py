"""
ARTS: vo boc Teacher / Student FuseKin.
  - Teacher: anh + GT 3D (H36M-17, met, root-relative) -> mesh.
  - Student: anh + 2D Halpe -> DiffLifter (dong bang) -> joint 3D -> mesh.
Backbone ResNet-50 (SPIN) va DiffLifter luon dong bang.
"""
import os

import torch
import torch.nn as nn

from core.config import cfg
from models.backbones.resnet import ResNetBackbone
from models.diff_lifter import DiffLifter
from models.teacher_student import Teacher, Student

os.environ.setdefault("WANDB_MODE", "offline")


class ARTS(nn.Module):
    def __init__(self, num_joint, embed_dim):
        super().__init__()
        self.mode = cfg.MODEL.name
        if self.mode not in ('teacher', 'student'):
            raise ValueError(f"MODEL.name khong hop le: {self.mode}. Chon 'teacher' hoac 'student'.")

        self.backbone = ResNetBackbone(cfg.MODEL.resnet_type)
        # Nap trong so pretrain cho backbone ('spin' | 'imagenet' | '' = ngau nhien).
        # Khi nap checkpoint da train, load_model_weights ghi de backbone (va chuan hoa anh) bang ban trong checkpoint.
        bb_src = cfg.MODEL.get('backbone_pretrained', '')
        if bb_src:
            self.backbone.load_pretrained(
                bb_src,
                spin_checkpoint=cfg.MODEL.get('spin_checkpoint', 'data_final/base_data/spin_model_checkpoint.pth.tar'),
            )
        else:
            print('[CANH BAO] Backbone ResNet KHONG nap pretrain (cfg.MODEL.backbone_pretrained rong) -> dac trung anh ngau nhien.')

        # Student: DiffLifter dong bang (2D -> 3D). Teacher khong can lifter.
        self.use_diff = self.mode == 'student'
        self.diff_lifter = self._build_diff_lifter() if self.use_diff else None
        # Gate spread (chi Student): std K gia thuyet DiffLifter lam dau vao gate rieng tung khop o moi block fusion.
        self.diff_spread_gate = self.mode == 'student' and bool(cfg.MODEL.get('difflift_spread_gate', False))
        if self.mode == 'student' and cfg.MODEL.get('difflift_spread_gate_zero', False) and not self.diff_spread_gate:
            raise ValueError('difflift_spread_gate_zero can bat difflift_spread_gate')

        if self.mode == 'teacher':
            print('Dang goi mo hinh teacher (GT 3D + PE 2D Halpe)')
            self.smpl_model = Teacher(num_joint=num_joint, embed_dim=embed_dim, depth=3,
                                      use_img_probe=bool(cfg.MODEL.get('img_probe', False)))
        else:
            print('Dang goi mo hinh student (DiffLifter)')
            self.smpl_model = Student(num_joint=num_joint, embed_dim=embed_dim, depth=3,
                                      spread_gate=self.diff_spread_gate,
                                      spread_gate_zero=bool(cfg.MODEL.get('difflift_spread_gate_zero', False)),
                                      spread_gate_hidden=int(cfg.MODEL.get('difflift_spread_gate_hidden', 64)))
            if self.diff_spread_gate:
                print(f'===> Spread gate: 2*sigmoid, init=1, zero_std={cfg.MODEL.difflift_spread_gate_zero}, '
                      f'hidden={cfg.MODEL.difflift_spread_gate_hidden}')
            self.privileged_joint_head = nn.Linear(embed_dim, 3)

        self.freeze_backbone_and_lifter()

    def _build_diff_lifter(self):
        """DiffLifter tu checkpoint finetune_difflifter.py (model_state = trong so EMA, difflift_cfg = kien truc)."""
        path = cfg.MODEL.get('difflift_checkpoint', '')
        assert path, "Student can cfg.MODEL.difflift_checkpoint"
        ck = torch.load(path, map_location='cpu', weights_only=False)
        dc = ck['difflift_cfg']
        m = DiffLifter(num_joints=17, embed_dim=dc['embed_dim'], depth=dc['depth'], num_heads=dc['num_heads'],
                       timesteps=dc['timesteps'], sampling_timesteps=int(cfg.MODEL.get('difflift_steps', 1)),
                       num_proposals=int(cfg.MODEL.get('difflift_k', 1)), scale=dc['scale'], drop_path_rate=0.0,
                       gcn_attn=bool(dc.get('gcn_attn', False)))
        missing = m.load_weights(ck['model_state'])
        print(f"Load DiffLifter ({path}) epoch {ck.get('epoch')} | S={m.sampling_timesteps} K={m.num_proposals}"
              + (f" | [CANH BAO] checkpoint khong co camera: thieu {missing}" if missing else ''))
        if bool(dc.get('soft_conf', False)) != bool(cfg.DATASET.lift_soft_conf):
            print(f"[CANH BAO] DiffLifter train voi soft_conf={dc.get('soft_conf')} nhung "
                  f"DATASET.lift_soft_conf={cfg.DATASET.lift_soft_conf} -> conf dau vao khac luc train lifter")
        return m

    def freeze_backbone_and_lifter(self):
        for module in (self.backbone, self.diff_lifter):
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if self.diff_lifter is not None:
            self.diff_lifter.eval()   # luon eval: DiffLifter.forward chon nhanh suy dien theo self.training
        return self

    def get_image_features(self, image):
        feature_map, global_feature = self.backbone(image)
        global_feature = global_feature.reshape(image.shape[0], 1, -1)
        return feature_map, global_feature

    def lift_diffusion(self, pose_2d, joints_mask, cam_bbox, all_hyp=False, random_noise=False):
        """DiffLifter: pose_2d (B,17,2|3) khung crop [-1,1] (Human36M17Dataset), joints_mask (B,17,1) conf,
        cam_bbox (B,3). Dau vao giong het finetune_difflifter.evaluate. Tra (B,17,3) met, root-relative
        (trung binh K gia thuyet), hoac ca K gia thuyet (B,K,17,3) neu all_hyp."""
        p2 = pose_2d[:, 0] if pose_2d.dim() == 4 else pose_2d
        xy = p2[..., :2].float()
        conf = joints_mask.to(xy.device).float().reshape(xy.shape[0], xy.shape[1], 1)
        xy = xy * (conf > 0)                                  # khop mask = 0 -> (0, 0) giong luc train lifter
        cam = cam_bbox.to(xy.device).float()[:, None, :].expand(-1, xy.shape[1], -1)
        gen = None
        if not random_noise:
            gen = torch.Generator(device=xy.device)
            gen.manual_seed(0)                                # nhieu DDIM co dinh -> ket qua lap lai duoc
        mean_pose, _, hyp = self.diff_lifter(torch.cat([xy, conf, cam], dim=-1), generator=gen)
        if all_hyp:
            return hyp - hyp[:, :, 0:1, :]
        return mean_pose - mean_pose[:, 0:1, :]

    def _diff_hyp(self, pose_2d, joints_mask, cam_bbox, steps, k, random_noise):
        """K gia thuyet DiffLifter voi S / K tam thoi (khong doi cau hinh S, K chinh). (B, K, 17, 3) root-rel."""
        dl = self.diff_lifter
        old = (dl.sampling_timesteps, dl.num_proposals)
        dl.sampling_timesteps, dl.num_proposals = int(steps), int(k)
        try:
            return self.lift_diffusion(pose_2d, joints_mask, cam_bbox, all_hyp=True, random_noise=random_noise)
        finally:
            dl.sampling_timesteps, dl.num_proposals = old

    @staticmethod
    def _pe2d_kwargs(pose_2d, joints_mask=None):
        """Toa do 2D Halpe + mask nhi phan cho PE vi tri trong fusion."""
        assert pose_2d is not None, 'Fusion can pose_2d (2D Halpe) cho PE vi tri khop'
        p2 = pose_2d[:, 0] if pose_2d.dim() == 4 else pose_2d            # (B, J, 2|3)
        if joints_mask is not None:
            m = (joints_mask > 0).to(p2.dtype)                             # nhi phan (soft conf -> 0/1)
        elif p2.shape[-1] >= 3:
            m = (p2[..., 2:3] > 0).to(p2.dtype)                            # cot 3 = conf / mask
        else:
            m = None
        return dict(joints_2d=p2[..., :2], joints_2d_mask=m)

    def forward_teacher(self, image, gt_pose_3d, is_train, pose_2d=None, joints_mask=None):
        # gt_pose_3d: (B, 17, 3), met, root-relative (da xu ly o Trainer)
        with torch.no_grad():
            feature_map, _ = self.get_image_features(image)
        result = self.smpl_model(
            joints=gt_pose_3d,
            img_feats=feature_map,
            is_train=is_train,
            return_features=True,
            **self._pe2d_kwargs(pose_2d, joints_mask),
        )
        result['teacher_input_joints'] = gt_pose_3d
        return result

    def forward_student(self, image, pose_2d, is_train, joints_mask=None, cam_bbox=None):
        spread = None
        with torch.no_grad():
            feature_map, _ = self.get_image_features(image)
            assert cam_bbox is not None and joints_mask is not None, \
                "Student (DiffLifter) can joints_mask + inputs['cam_bbox']"
            pose_3d = self.lift_diffusion(pose_2d, joints_mask, cam_bbox)
            if self.diff_spread_gate:
                # Std (B,17,3) met chi dieu kien gate, khong cong vao token.
                hk = self._diff_hyp(pose_2d, joints_mask, cam_bbox, cfg.MODEL.get('difflift_spread_steps', 5),
                                    cfg.MODEL.get('difflift_spread_k', 20), random_noise=self.training)
                spread = hk.std(dim=1)
            pose_3d = pose_3d - pose_3d[:, 0:1, :]            # root-relative

        result = self.smpl_model(
            joints=pose_3d,
            img_feats=feature_map,
            is_train=is_train,
            return_features=True,
            joint_spread=spread,
            **self._pe2d_kwargs(pose_2d, joints_mask),
        )
        result['privileged_3d'] = self.privileged_joint_head(result['feat_joint'])
        result['lifted_joints_3d'] = pose_3d   # joint dua vao Student (root-relative)
        if self.diff_spread_gate:
            result['joint_spread_3d'] = spread   # Std that de inspector kiem tra, ke ca ban zero-control.
        return result

    def forward(self, image, joints, is_train=True, joints_mask=None, pose_2d=None, cam_bbox=None):
        """Teacher: joints = GT 3D, pose_2d = 2D Halpe (cho PE). Student: joints = 2D Halpe, cam_bbox bat buoc."""
        if self.mode == 'teacher':
            return self.forward_teacher(image, joints, is_train, pose_2d=pose_2d, joints_mask=joints_mask)
        return self.forward_student(image, joints, is_train, joints_mask=joints_mask, cam_bbox=cam_bbox)


def get_model(num_joint, embed_dim, depth=None):
    return ARTS(num_joint, embed_dim)
