"""
ARTS: vo boc Student FuseKin (phien ban chinh thuc duy nhat tu 2026-11-11).
  anh + 2D detector -> Refiner 2D day (dong bang) -> ImgLifter (dong bang, K x S, gop mean) -> pose 3D
  -> Student (lib/models/teacher_student.py) -> mesh.
Backbone ResNet-50 (SPIN): fine-tune lr nho + ghep heatmap 2D o stride 4 ([A], cfg.MODEL.v3_bb_*).
Teacher / DiffLifter cu da bo khoi ARTS (code cu o _to_delete/bak_cleanup_v7_20261111/).
build_diff_lifter / diff_lift / make_heatmaps van giu o day vi Refiner, script eval va tools con import.
"""
import os

import torch
import torch.nn as nn

from core.config import cfg
from models.backbones.resnet import ResNetBackbone
from models.diff_lifter import DiffLifter
from models.teacher_student import Student

os.environ.setdefault("WANDB_MODE", "offline")


def make_heatmaps(pose_2d, joints_mask, size=64, sigma=2.0):
    """[A] Heatmap Gauss tu 2D [-1,1] (B,J,2|3) x conf (B,J,1) -> (B, J, size, size). Khop mask = 0 -> 0."""
    p2 = pose_2d[:, 0] if pose_2d.dim() == 4 else pose_2d
    xy = (p2[..., :2].float() + 1.0) * 0.5 * size - 0.5                        # tam o pixel cua luoi size x size
    conf = joints_mask.reshape(xy.shape[0], xy.shape[1]).float().clamp(0, 1)
    g = torch.arange(size, device=xy.device, dtype=xy.dtype)
    gx = torch.exp(-(g[None, None, :] - xy[..., 0:1]) ** 2 / (2 * sigma ** 2))    # (B, J, W)
    gy = torch.exp(-(g[None, None, :] - xy[..., 1:2]) ** 2 / (2 * sigma ** 2))    # (B, J, H)
    return gy[..., :, None] * gx[..., None, :] * conf[..., None, None]


# ---- Tien ich DiffLifter cu: KHONG dung trong Student, chi de script eval / tools so sanh voi lifter cu ----
def build_diff_lifter(path, steps=1, k=1):
    """DiffLifter tu checkpoint finetune_difflifter.py (model_state = trong so EMA, difflift_cfg = kien truc)."""
    assert path, "Can duong dan checkpoint DiffLifter"
    ck = torch.load(path, map_location='cpu', weights_only=False)
    dc = ck['difflift_cfg']
    m = DiffLifter(num_joints=17, embed_dim=dc['embed_dim'], depth=dc['depth'], num_heads=dc['num_heads'],
                   timesteps=dc['timesteps'], sampling_timesteps=int(steps), num_proposals=int(k),
                   scale=dc['scale'], drop_path_rate=0.0, gcn_attn=bool(dc.get('gcn_attn', False)))
    missing = m.load_weights(ck['model_state'])
    print(f"Load DiffLifter ({path}) epoch {ck.get('epoch')} | S={m.sampling_timesteps} K={m.num_proposals}"
          + (f" | [CANH BAO] checkpoint khong co camera: thieu {missing}" if missing else ''))
    if bool(dc.get('soft_conf', False)) != bool(cfg.DATASET.lift_soft_conf):
        print(f"[CANH BAO] DiffLifter train voi soft_conf={dc.get('soft_conf')} nhung "
              f"DATASET.lift_soft_conf={cfg.DATASET.lift_soft_conf} -> conf dau vao khac luc train lifter")
    return m


def diff_lift(diff_lifter, pose_2d, joints_mask, cam_bbox, all_hyp=False, random_noise=False, steps=None, k=None):
    """DiffLifter: pose_2d (B,17,2|3) khung crop [-1,1], joints_mask (B,17,1) conf, cam_bbox (B,3).
    Tra (B,17,3) met root-relative (trung binh K gia thuyet), hoac (B,K,17,3) neu all_hyp."""
    p2 = pose_2d[:, 0] if pose_2d.dim() == 4 else pose_2d
    xy = p2[..., :2].float()
    conf = joints_mask.to(xy.device).float().reshape(xy.shape[0], xy.shape[1], 1)
    xy = xy * (conf > 0)                                  # khop mask = 0 -> (0, 0) giong luc train lifter
    cam = cam_bbox.to(xy.device).float()[:, None, :].expand(-1, xy.shape[1], -1)
    gen = None
    if not random_noise:
        gen = torch.Generator(device=xy.device)
        gen.manual_seed(0)
    old = (diff_lifter.sampling_timesteps, diff_lifter.num_proposals)
    if steps is not None:
        diff_lifter.sampling_timesteps = int(steps)
    if k is not None:
        diff_lifter.num_proposals = int(k)
    try:
        mean_pose, _, hyp = diff_lifter(torch.cat([xy, conf, cam], dim=-1), generator=gen)
    finally:
        diff_lifter.sampling_timesteps, diff_lifter.num_proposals = old
    if all_hyp:
        return hyp - hyp[:, :, 0:1, :]
    return mean_pose - mean_pose[:, 0:1, :]


class ARTS(nn.Module):
    def __init__(self, num_joint, embed_dim):
        super().__init__()
        self.mode = 'student'
        if cfg.MODEL.name != 'student':
            raise ValueError(f"MODEL.name = {cfg.MODEL.name}: chi con ho tro 'student' (Teacher da bo 2026-11-11).")
        assert cfg.MODEL.get('imglift_checkpoint', ''), 'Student can MODEL.imglift_checkpoint (Refiner day + ImgLifter)'

        self.backbone = ResNetBackbone(cfg.MODEL.resnet_type)
        # [A]: fine-tune backbone (lr nho, xem base.prepare_network) + ghep heatmap 2D vao stride 4
        self.bb_finetune = bool(cfg.MODEL.v3_bb_finetune)
        self.bb_heatmap = bool(cfg.MODEL.v3_bb_heatmap)
        if self.bb_heatmap:
            self.backbone.enable_heatmap_fusion(17)
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

        # Refiner 2D day + ImgLifter (dong bang)
        from utils.img_lifter_tools import load_img_lifter, load_frozen_refiner
        cpu = torch.device('cpu')
        self.img_lifter, ck = load_img_lifter(cfg.MODEL.imglift_checkpoint, cpu)
        self.refiner2d = load_frozen_refiner(cfg.MODEL.get('imglift_refiner_checkpoint', '') or ck['refiner_ckpt'], cpu)
        self.student_2d_from_refiner = bool(cfg.MODEL.get('student_2d_from_refiner', False))
        print(f'===> ImgLifter {cfg.MODEL.imglift_checkpoint} (K={cfg.MODEL.imglift_K}, S={cfg.MODEL.imglift_S}) | '
              f'2D Student tu Refiner: {self.student_2d_from_refiner} | bb_finetune={self.bb_finetune} '
              f'bb_heatmap={self.bb_heatmap}')

        self.smpl_model = Student(num_joint=num_joint, embed_dim=embed_dim, depth=3,
                                  hgcn_freeze_adj=bool(cfg.MODEL.get('hgcn_freeze_adj', False)))
        # GIU TAM: base.py van doc output 'privileged_3d' (loss tat bang v7_no_privileged); bo khi don base.py
        self.privileged_joint_head = nn.Linear(embed_dim, 3)

        self.freeze_backbone_and_lifter()

    def freeze_backbone_and_lifter(self):
        # Backbone: dong bang tru khi v3_bb_finetune (BN giu thong ke cu, xem train()); module ghep heatmap luon train.
        for module in ((None if self.bb_finetune else self.backbone), self.img_lifter, self.refiner2d):
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()
        if self.backbone.heat_fuse is not None:
            for param in self.backbone.heat_fuse.parameters():
                param.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.img_lifter.eval()                                  # dong bang: luon eval
        self.refiner2d.eval()
        return self

    @torch.no_grad()
    def lift_imglift(self, image, pose_2d, joints_mask, cam_bbox):
        """anh + 2D detector -> Refiner 2D day -> ImgLifter (K gia thuyet x S buoc, gop mean).
        Tra pose_3d (B,17,3) met, xy Refiner 17 khop (B,17,2) [-1,1]."""
        from models.img_lifter import ImgLifter, aggregate
        from utils.img_lifter_tools import refiner_inputs, lifter_det
        B = image.shape[0]
        p2 = pose_2d[:, 0] if pose_2d.dim() == 4 else pose_2d
        jm = joints_mask.reshape(B, 17).float()
        nc = int(getattr(self.img_lifter, 'num_ctx', 0))
        xy, log_b, f8, fj = refiner_inputs(self.refiner2d, image, p2[..., :2].float(), jm, 17 + nc)
        cond7 = ImgLifter.make_cond7(xy, log_b, lifter_det(jm, nc), cam_bbox.float())
        gen = None
        if not self.training:                                   # test: nhieu co dinh -> lap lai duoc
            gen = torch.Generator(device=image.device)
            gen.manual_seed(0)
        hyp = self.img_lifter.sample(cond7, fj, f8, K=int(cfg.MODEL.imglift_K), S=int(cfg.MODEL.imglift_S),
                                     eta=1.0, generator=gen)
        return aggregate(hyp, xy, log_b, 'mean'), xy[:, :17]

    def get_image_features(self, image):
        feature_map, global_feature = self.backbone(image)
        global_feature = global_feature.reshape(image.shape[0], 1, -1)
        return feature_map, global_feature

    def image_feats(self, image, pose_2d=None, joints_mask=None):
        """Feature F8 cho Student. Co gradient neu backbone duoc train ([A]) hoac co module ghep heatmap."""
        heat = None
        if self.bb_heatmap:
            assert pose_2d is not None and joints_mask is not None, 'v3_bb_heatmap can pose_2d + joints_mask'
            heat = make_heatmaps(pose_2d, joints_mask, size=image.shape[-1] // 4, sigma=float(cfg.MODEL.v3_heatmap_sigma))
        grad = self.bb_finetune or self.bb_heatmap
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            out = self.backbone(image, heatmap=heat)
        return out[0]

    @staticmethod
    def _pe2d_kwargs(pose_2d, joints_mask=None):
        """Toa do 2D + mask nhi phan cho PE vi tri trong fusion."""
        assert pose_2d is not None, 'Fusion can pose_2d cho PE vi tri khop'
        p2 = pose_2d[:, 0] if pose_2d.dim() == 4 else pose_2d            # (B, J, 2|3)
        if joints_mask is not None:
            m = (joints_mask > 0).to(p2.dtype)                             # nhi phan (soft conf -> 0/1)
        elif p2.shape[-1] >= 3:
            m = (p2[..., 2:3] > 0).to(p2.dtype)                            # cot 3 = conf / mask
        else:
            m = None
        return dict(joints_2d=p2[..., :2], joints_2d_mask=m)

    def forward(self, image, joints, is_train=True, joints_mask=None, pose_2d=None, cam_bbox=None):
        """joints = 2D detector (B, 17, 2|3) [-1,1]; joints_mask (B, 17, 1); cam_bbox (B, 3) bat buoc.
        pose_2d: khong dung (giu cho tuong thich loi goi cu)."""
        assert cam_bbox is not None and joints_mask is not None, "Student can joints_mask + inputs['cam_bbox']"
        pose_2d = joints
        pose_3d, xy_ref = self.lift_imglift(image, pose_2d, joints_mask, cam_bbox)
        pose_3d = pose_3d - pose_3d[:, 0:1, :]
        if self.student_2d_from_refiner:
            # 2D Student = 2D Refiner (da dien khop detector sot), mask = 1 moi khop
            ones = torch.ones_like(xy_ref[..., :1])
            pose_2d = torch.cat([xy_ref, ones], dim=-1)
            joints_mask = ones
        img_feats = self.image_feats(image, pose_2d, joints_mask)
        result = self.smpl_model(joints=pose_3d, img_feats=img_feats, is_train=is_train, return_features=True,
                                 **self._pe2d_kwargs(pose_2d, joints_mask))
        result['privileged_3d'] = self.privileged_joint_head(result['feat_joint'])
        result['lifted_joints_3d'] = pose_3d   # pose 3D dua vao Student (root-relative)
        return result


def get_model(num_joint, embed_dim, depth=None):
    return ARTS(num_joint, embed_dim)
