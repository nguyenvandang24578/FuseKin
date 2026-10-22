import os
from functools import partial

import torch
import torch.nn as nn

from core.config import cfg
from models import Multimodel
from models.backbones.resnet import ResNetBackbone
from models.DSTformer import DSTformer
from models.diff_lifter import DiffLifter
from models.hyp_selector import HypSelector
from models.teacher_student import Teacher, Student   # model end-to-end moi (khong con SPIN)
from utils.h36m_adapter import drop_joints_2d, h36m_joint_indices

os.environ.setdefault("WANDB_MODE", "offline")

MOTIONBERT_CONFIG = dict(
    dim_in=3,
    dim_out=3,
    dim_feat=512,
    dim_rep=512,
    depth=5,
    num_heads=8,
    mlp_ratio=2,
    maxlen=243,
    num_joints=17,
    att_fuse=True,
)
NUM_FRAMES = MOTIONBERT_CONFIG["maxlen"]


class ARTS(nn.Module):
    def __init__(self, num_joint, embed_dim):
        super().__init__()
        self.mode = cfg.MODEL.name

        self.backbone = ResNetBackbone(cfg.MODEL.resnet_type)
        # Nap trong so pretrain cho backbone ('spin' | 'imagenet' | '' = ngau nhien nhu cu).
        # Khi nap checkpoint da train, load_model_weights se ghi de backbone (va chuan hoa) bang ban trong checkpoint.
        bb_src = cfg.MODEL.get('backbone_pretrained', '')
        if bb_src:
            self.backbone.load_pretrained(
                bb_src,
                spin_checkpoint=cfg.MODEL.get('spin_checkpoint', 'data_final/base_data/spin_model_checkpoint.pth.tar'),
            )
        else:
            print('[CANH BAO] Backbone ResNet KHONG nap pretrain (cfg.MODEL.backbone_pretrained rong) -> dac trung anh ngau nhien.')

        # Lấy hpe_dim từ config để hỗ trợ MotionBERT-Lite (256) hoặc Full (512)
        mb_dim = cfg.MODEL.get('hpe_dim', 512)
        mb_mlp_ratio = cfg.MODEL.get('mlp_ratio', 4 if mb_dim == 256 else 2)

        # Override dim_feat và mlp_ratio theo config
        mb_config = MOTIONBERT_CONFIG.copy()
        mb_config['dim_feat'] = mb_dim
        mb_config['dim_rep'] = 512  # dim_rep luôn là 512 kể cả bản Lite
        mb_config['mlp_ratio'] = mb_mlp_ratio

        # Teacher chi can MotionBERT khi bat (A): tron GT voi joint lift (cfg.MODEL.teacher_lift_alpha_max > 0)
        self.teacher_lift = (self.mode == "teacher" and cfg.MODEL.get('teacher_lift_alpha_max', 0.0) > 0)
        # Goi y nen (PLAN_TEACHER_HINT.md): T-hint nhan joint MotionBERT GIONG Student + goi y z tu GT;
        # S-hint = Student tu doan z.
        self.teacher_hint = (self.mode == "teacher" and bool(cfg.MODEL.get('teacher_hint', False)))
        self.student_hint = (self.mode == "student" and bool(cfg.MODEL.get('student_hint', False)))
        self.hint_dim = int(cfg.MODEL.get('hint_dim', 16))
        self.hint_noise = float(cfg.MODEL.get('hint_noise', 0.1))
        self.hint_off_prob = float(cfg.MODEL.get('hint_off_prob', 0.3))
        # Bo lift: 'pmce' (GraphormerNet co dac trung anh) thay MotionBERT khi Student.
        self.use_pmce = (self.mode == "student" and cfg.MODEL.get('pose_lifter', 'motionbert') == 'pmce')
        self.pmce_lifter = None
        # Bo lift: 'diffusion' (DiffLifter, dong bang) thay MotionBERT khi Student.
        self.use_diff = (self.mode == "student" and cfg.MODEL.get('pose_lifter', 'motionbert') == 'diffusion')
        self.diff_lifter = None
        # Chon gia thuyet DiffLifter theo tung khop bang anh (co hoc, KHONG dong bang)
        self.diff_select = self.use_diff and bool(cfg.MODEL.get('difflift_select', False))
        self.hyp_selector = HypSelector(num_joints=num_joint, img_dim=2048,
                                        dim=int(cfg.MODEL.get('difflift_select_dim', 256))) \
            if self.diff_select else None
        if self.use_pmce:
            from models.PoseEstimation import get_model as _get_pmce_model
            self.pose_lifter = None   # khong dung MotionBERT khi lifter = PMCE
            self.pmce_lifter = _get_pmce_model(
                num_joint=num_joint, embed_dim=cfg.MODEL.get('pmce_dim', 256),
                depth=cfg.MODEL.get('pmce_dep', 3), pretrained=False)
            self._load_pmce_weights()
        elif self.use_diff:
            self.pose_lifter = None   # khong dung MotionBERT khi lifter = DiffLifter
            self.diff_lifter = self._build_diff_lifter()
        elif self.mode != "teacher" or self.teacher_lift or self.teacher_hint:
            self.pose_lifter = DSTformer(
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                **mb_config,
            )
            self.load_pose_lifter_weights()
        else:
            self.pose_lifter = None

        # ---- Co kien truc v2 (PLAN_STUDENT_V2.md), DA CHOT 2026-10-13: mac dinh BAT trong config.py ----
        # CHU Y KD: Teacher dung cho KD duoc dung tu CUNG cfg -> nhan cung cac co nay. T-hint v2 + S-hint v2 -> khop.
        # Neu dung lai Teacher kien truc cu thi phai tat cac co v2 (checkpoint cu khong co cac module moi).
        self.joint_pe2d = bool(cfg.MODEL.get('joint_pe2d', False))
        arch_kw = dict(
            joint_pe2d=self.joint_pe2d,
            pe2d_num_freq=int(cfg.MODEL.get('pe2d_num_freq', 6)),
            fusion_joint_self_attn=bool(cfg.MODEL.get('fusion_joint_self_attn', False)),
            pose_head_mode=cfg.MODEL.get('pose_head_mode', 'vposer'),
            pose_query_bone=bool(cfg.MODEL.get('pose_query_bone', False)),
            shape_attn_mode=cfg.MODEL.get('shape_attn_mode', 'global'),
        )
        if self.mode in ("teacher", "student") and (self.joint_pe2d or arch_kw['fusion_joint_self_attn']
                                                    or arch_kw['pose_head_mode'] != 'vposer'
                                                    or arch_kw['shape_attn_mode'] != 'global'):
            print(f"===> Kien truc v2: {arch_kw}")

        if self.mode == "teacher" and self.teacher_hint:
            # Cung kien truc Student (bo ma hoa joint 'noisy') + HintEncoder -> Student khoi tao duoc TOAN BO tu Teacher
            print(f"Đang gọi mô hình teacher T-hint (k={self.hint_dim}, noise={self.hint_noise}, off={self.hint_off_prob})")
            self.smpl_model = Student(num_joint=num_joint, embed_dim=embed_dim, depth=3,
                                      hint_dim=self.hint_dim, hint_mode='teacher', **arch_kw)
        elif self.mode == "teacher":
            if self.joint_pe2d:
                print('===> Teacher GT + joint_pe2d: PE vi tri khop lay tu 2D Halpe (nhieu), truyen qua pose_2d')
            print("Đang gọi mô hình teacher")
            self.smpl_model = Teacher(num_joint=num_joint, embed_dim=embed_dim, depth=3,
                                      use_img_probe=bool(cfg.MODEL.get('img_probe', False)), **arch_kw)
        elif self.mode == "student":
            print("Đang gọi mô hình student" + (f" S-hint (k={self.hint_dim})" if self.student_hint else ""))
            self.smpl_model = Student(num_joint=num_joint, embed_dim=embed_dim, depth=3,
                                      hint_dim=self.hint_dim if self.student_hint else 0,
                                      hint_mode='student' if self.student_hint else 'none', **arch_kw)
            self.privileged_joint_head = nn.Linear(embed_dim, 3)
        elif self.mode == "ARTS":
            self.pose_mesh_coevo = Multimodel.get_model(num_joint, embed_dim)
        else:
            raise ValueError(f"Mode không hợp lệ: {self.mode}. Chọn teacher, student hoặc ARTS.")

        self.freeze_backbone_and_pose_lifter()

    def load_pose_lifter_weights(self):
        path = cfg.MODEL.get("motionbert_pretrained", "")
        if not path:
            print("Cảnh báo: chưa có weight pretrained cho MotionBERT.")
            return

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = {}
        for name, weight in checkpoint["model_pos"].items():
            state_dict[name.replace("module.", "", 1)] = weight

        missing, unexpected = self.pose_lifter.load_state_dict(state_dict, strict=False)
        print(f"Load MotionBERT xong. Thiếu: {missing}. Thừa: {unexpected}.")

    def _load_pmce_weights(self):
        path = cfg.MODEL.get("pmce_checkpoint", "")
        if not path:
            print("[CANH BAO] pose_lifter='pmce' nhung cfg.MODEL.pmce_checkpoint rong -> PMCE khoi tao ngau nhien.")
            return
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = self.pmce_lifter.load_state_dict(sd, strict=False)
        print(f"Load PMCE ({path}) xong. Thieu: {len(missing)} {missing[:3]}. Thua: {len(unexpected)} {unexpected[:3]}.")

    def _build_diff_lifter(self):
        """DiffLifter tu checkpoint finetune_difflifter.py (model_state = trong so EMA, difflift_cfg = kien truc)."""
        path = cfg.MODEL.get('difflift_checkpoint', '')
        assert path, "pose_lifter='diffusion' can cfg.MODEL.difflift_checkpoint"
        ck = torch.load(path, map_location='cpu', weights_only=False)
        dc = ck['difflift_cfg']
        m = DiffLifter(num_joints=17, embed_dim=dc['embed_dim'], depth=dc['depth'], num_heads=dc['num_heads'],
                       timesteps=dc['timesteps'], sampling_timesteps=int(cfg.MODEL.get('difflift_steps', 1)),
                       num_proposals=int(cfg.MODEL.get('difflift_k', 1)), scale=dc['scale'], drop_path_rate=0.0)
        missing = m.load_weights(ck['model_state'])
        print(f"Load DiffLifter ({path}) epoch {ck.get('epoch')} | S={m.sampling_timesteps} K={m.num_proposals}"
              + (f" | [CANH BAO] checkpoint khong co camera: thieu {missing}" if missing else ''))
        if bool(dc.get('soft_conf', False)) != bool(cfg.DATASET.lift_soft_conf):
            print(f"[CANH BAO] DiffLifter train voi soft_conf={dc.get('soft_conf')} nhung "
                  f"DATASET.lift_soft_conf={cfg.DATASET.lift_soft_conf} -> conf dau vao khac luc train lifter")
        return m

    def freeze_backbone_and_pose_lifter(self):
        for module in (self.backbone, self.pose_lifter, getattr(self, 'pmce_lifter', None),
                       getattr(self, 'diff_lifter', None)):
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if self.pose_lifter is not None:
            self.pose_lifter.eval()
        if getattr(self, 'pmce_lifter', None) is not None:
            self.pmce_lifter.eval()
        if getattr(self, 'diff_lifter', None) is not None:
            self.diff_lifter.eval()   # luon eval: DiffLifter.forward chon nhanh suy dien theo self.training
        return self

    def get_image_features(self, image):
        feature_map, global_feature = self.backbone(image)
        global_feature = global_feature.reshape(image.shape[0], 1, -1)
        return feature_map, global_feature

    def lift_2d_to_3d(self, pose_2d, joints_mask=None):
        if pose_2d.dim() == 3:
            pose_2d = pose_2d.unsqueeze(1)

        first_frame = pose_2d[:, 0]
        xy = first_frame[..., :2]

        # CRITICAL: MotionBERT goc (MB_ft_h36m.yaml) KHONG tru root khoi input 2D, chi tru root
        # o 3D target/loss (rootrel: True). FuseKin truoc day tru ca input 2D -> mat tin hieu vi
        # tri tuyet doi trong khung hinh. Dieu khien boi cfg.MODEL.motionbert_2d_rootrel de so sanh.
        if cfg.MODEL.get('motionbert_2d_rootrel', True):
            xy = xy - xy[:, 0:1, :]

        # MotionBERT's 3rd input channel was finetuned on the real per-joint
        # validity mask (see main/finetune_motionbert.py), not a constant.
        if joints_mask is not None:
            confidence = joints_mask.to(dtype=xy.dtype, device=xy.device)
        elif first_frame.shape[-1] >= 3:
            confidence = first_frame[..., 2:3]
        else:
            confidence = torch.ones_like(xy[..., :1])
        # Xoa cac khop giong luc finetune (cfg.MODEL.motionbert_drop_joints, mac dinh [] = khong xoa)
        drop_names = cfg.MODEL.get('motionbert_drop_joints', [])
        if drop_names:
            xy, confidence = drop_joints_2d(xy, confidence, h36m_joint_indices(drop_names))
        pose_xyc = torch.cat([xy, confidence], dim=-1)

        # Truyền trực tiếp 1 frame tĩnh (F=1) vào thay vì lặp lại NUM_FRAMES lần
        single_frame = pose_xyc.unsqueeze(1)  # (B, 1, 17, 3)
        pose_3d = self.pose_lifter(single_frame)
        # Checkpoint finetune trong FuseKin xuat MET (root-rel). Luu y: ban goc MotionBERT
        # (MB_release, MB_ft_h36m) xuat toa do pixel chuan hoa, KHONG phai met.
        return pose_3d[:, 0]  # (B, 17, 3) mét

    def lift_pmce(self, joints_raw, joints_mask, global_feature):
        """PMCE (GraphormerNet): joints_raw (B,17,2) 2D anh goc chuan hoa kieu PMCE (KHONG tru root),
        global_feature (B,1,2048) dac trung anh SPIN. Nhan ban 1 anh thanh 16 frame (giong PMCE goc). Tra ve MET."""
        SEQ = 16
        x = joints_raw[..., :2].float()
        if joints_mask is not None:
            x = x * (joints_mask.to(x.device) > 0)
        xs = x.unsqueeze(1).repeat(1, SEQ, 1, 1)                       # (B, T, 17, 2)
        feat = global_feature.reshape(global_feature.shape[0], -1).unsqueeze(1).repeat(1, SEQ, 1)  # (B, T, 2048)
        pred = self.pmce_lifter(xs, feat)                             # (B, 17, 3) milimet
        return pred / 1000.0                                          # -> met

    def lift_diffusion(self, pose_2d, joints_mask, cam_bbox, all_hyp=False):
        """DiffLifter: pose_2d (B,17,2|3) khung crop [-1,1] (Human36M17Dataset), joints_mask (B,17,1) conf,
        cam_bbox (B,3). Dau vao giong het finetune_difflifter.evaluate. Tra (B,17,3) met, root-relative
        (trung binh K gia thuyet), hoac ca K gia thuyet (B,K,17,3) neu all_hyp."""
        p2 = pose_2d[:, 0] if pose_2d.dim() == 4 else pose_2d
        xy = p2[..., :2].float()
        conf = joints_mask.to(xy.device).float().reshape(xy.shape[0], xy.shape[1], 1)
        xy = xy * (conf > 0)                                  # khop mask = 0 -> (0, 0) giong luc train lifter
        cam = cam_bbox.to(xy.device).float()[:, None, :].expand(-1, xy.shape[1], -1)
        gen = None
        if not (self.training and self.diff_select):          # chon gia thuyet luc train: nhieu ngau nhien (da dang)
            gen = torch.Generator(device=xy.device)
            gen.manual_seed(0)                                # nhieu DDIM co dinh -> ket qua lap lai duoc
        mean_pose, _, hyp = self.diff_lifter(torch.cat([xy, conf, cam], dim=-1), generator=gen)
        if all_hyp:
            return hyp - hyp[:, :, 0:1, :]
        return mean_pose - mean_pose[:, 0:1, :]

    def _pe2d_kwargs(self, pose_2d, joints_mask=None):
        """Toa do 2D (CUNG 2D dua vao MotionBERT) + mask cho PE vi tri trong fusion (PLAN_STUDENT_V2).
        Tra {} khi joint_pe2d tat -> goi smpl_model nhu cu."""
        if not self.joint_pe2d:
            return {}
        p2 = pose_2d[:, 0] if pose_2d.dim() == 4 else pose_2d            # (B, J, 2|3)
        if joints_mask is not None:
            m = (joints_mask > 0).to(p2.dtype)                             # nhi phan (soft conf -> 0/1)
        elif p2.shape[-1] >= 3:
            m = (p2[..., 2:3] > 0).to(p2.dtype)                            # cot 3 = conf / mask
        else:
            m = None
        return dict(joints_2d=p2[..., :2], joints_2d_mask=m)

    def forward_teacher_hint(self, image, gt_pose_3d, is_train, pose_2d, joints_mask=None, hint_on=None):
        """T-hint: joint vao = MotionBERT lift (GIONG Student); goi y z = HintEncoder([GT, GT - lift]).
        Luc train: cong nhieu N(0, hint_noise^2) vao z + tat goi y (z = 0) voi xac suat hint_off_prob.
        hint_on (B,) 1/0 ghi de viec tat/bat (vd danh gia 'khong goi y' = toan 0). Eval mac dinh: bat, khong nhieu."""
        assert pose_2d is not None, 'T-hint can pose_2d (2D dau vao MotionBERT, giong Student)'
        with torch.no_grad():
            feature_map, _ = self.get_image_features(image)
            lifted = self.lift_2d_to_3d(pose_2d, joints_mask=joints_mask)
            lifted = (lifted - lifted[:, 0:1, :]).to(gt_pose_3d.dtype)      # root-relative, met

        B = image.shape[0]
        z = self.smpl_model.hint_encoder(gt_pose_3d.float(), lifted.float())   # (B, k), co grad
        if is_train and self.hint_noise > 0:
            z = z + torch.randn_like(z) * self.hint_noise                   # gioi han so bit moi chieu
        if hint_on is None:
            if is_train and self.hint_off_prob > 0:
                hint_on = (torch.rand(B, device=z.device) >= self.hint_off_prob).to(z.dtype)
            else:
                hint_on = torch.ones(B, device=z.device, dtype=z.dtype)
        hint_on = hint_on.to(device=z.device, dtype=z.dtype).reshape(B)
        z = z * hint_on[:, None]

        result = self.smpl_model(
            joints=lifted,
            img_feats=feature_map,
            is_train=is_train,
            return_features=True,
            hint_z=z,
            **self._pe2d_kwargs(pose_2d, joints_mask),
        )
        result['teacher_input_joints'] = lifted
        result['lifted_joints_3d'] = lifted
        result['hint_on'] = hint_on
        return result

    def forward_teacher(self, image, gt_pose_3d, is_train, pose_2d=None, joints_mask=None, lift_alpha=None,
                        hint_on=None):
        if self.teacher_hint:
            return self.forward_teacher_hint(image, gt_pose_3d, is_train, pose_2d,
                                             joints_mask=joints_mask, hint_on=hint_on)
        with torch.no_grad():
            feature_map, _ = self.get_image_features(image)

            # gt_pose_3d: (B, 17, 3), met, root-relative (da xu ly o Trainer)
            joints_in = gt_pose_3d
            lifted = None
            if self.teacher_lift and pose_2d is not None and lift_alpha is not None:
                # (A) Tron GT voi joint MotionBERT lift: input = (1-a)*GT + a*lift  (a=0: GT sach, a=1: nhu Student)
                lifted = self.lift_2d_to_3d(pose_2d, joints_mask=joints_mask)
                lifted = (lifted - lifted[:, 0:1, :]).to(gt_pose_3d.dtype)     # root-relative
                a = torch.as_tensor(lift_alpha, dtype=gt_pose_3d.dtype, device=gt_pose_3d.device)
                a = a.reshape(-1, 1, 1) if a.dim() > 0 else a.reshape(1, 1, 1)
                joints_in = (1.0 - a) * gt_pose_3d + a * lifted

        if self.joint_pe2d:
            assert pose_2d is not None, 'Teacher GT + joint_pe2d can pose_2d (2D Halpe): truyen pose_2d, joints_mask'
        result = self.smpl_model(
            joints=joints_in,
            img_feats=feature_map,
            is_train=is_train,
            return_features=True,
            **self._pe2d_kwargs(pose_2d, joints_mask),
        )
        result['teacher_input_joints'] = joints_in      # (B, 17, 3) input thuc su dua vao Teacher
        if lifted is not None:
            result['lifted_joints_3d'] = lifted         # (B, 17, 3) joint MotionBERT lift
        return result

    def forward_student(self, image, pose_2d, is_train, gt_pose_3d=None, alpha=1.0, joints_mask=None, pose_2d_raw=None,
                        cam_bbox=None):
        hyp = None
        with torch.no_grad():
            feature_map, global_feature = self.get_image_features(image)
            if self.use_pmce:
                assert pose_2d_raw is not None, "lifter=PMCE can pose_2d_raw (inputs['joints_raw'])"
                pose_3d = self.lift_pmce(pose_2d_raw, joints_mask, global_feature)
            elif self.use_diff:
                assert cam_bbox is not None and joints_mask is not None, \
                    "lifter=diffusion can joints_mask + inputs['cam_bbox']"
                if self.diff_select:
                    hyp = self.lift_diffusion(pose_2d, joints_mask, cam_bbox, all_hyp=True)   # (B,K,17,3)
                    pose_3d = hyp.mean(dim=1)                                                 # chi de log
                else:
                    pose_3d = self.lift_diffusion(pose_2d, joints_mask, cam_bbox)
            else:
                # MotionBERT finetune ra milimet, lift_2d_to_3d đã đổi về mét (root-relative bên dưới)
                pose_3d = self.lift_2d_to_3d(pose_2d, joints_mask=joints_mask)
            # student_3d_rootrel=True (mac dinh): tru root de fusion nhan toa do relative.
            # student_3d_rootrel=False: giu nguyen toa do tuyet doi trong cam-space, model hoc context day du.
            if cfg.MODEL.get('student_3d_rootrel', True):
                pose_3d = pose_3d - pose_3d[:, 0:1, :]        # root-relative

        sel_w = None
        if hyp is not None:
            # Chon gia thuyet theo tung khop bang anh (CO gradient vao hyp_selector; hyp, feature_map khong grad)
            p2 = pose_2d[:, 0] if pose_2d.dim() == 4 else pose_2d
            pose_3d, sel_w = self.hyp_selector(hyp, feature_map, p2[..., :2], joints_mask)

        result = self.smpl_model(
            joints=pose_3d,
            img_feats=feature_map,
            is_train=is_train,
            return_features=True,
            **self._pe2d_kwargs(pose_2d, joints_mask),
        )
        result['privileged_3d'] = self.privileged_joint_head(result['feat_joint'])
        result['lifted_joints_3d'] = pose_3d   # joint dua vao student (tuyet doi hoac root-rel tuy flag)
        if sel_w is not None:
            result['hyp_select_3d'] = pose_3d  # pose da chon (co grad) -> loss phu trong Student_Trainer
            result['hyp_select_w'] = sel_w     # (B, K, 17) trong so softmax theo K
        return result

    def forward_arts(self, image, pose_input, is_train, use_gt_3d=False,
                     gt_pose_6d=None, kp2d=None, kp_conf=None,
                     pose_valid_mask=None, gt_joints_3d=None, joints_mask=None):
        with torch.no_grad():
            ft_map, global_feature = self.get_image_features(image)

            if not use_gt_3d:
                # Output MotionBERT đã được đổi về mét trong lift_2d_to_3d
                pose_3d = self.lift_2d_to_3d(pose_input, joints_mask=joints_mask)
                pose_3d = pose_3d - pose_3d[:, 0:1, :]        # root-relative
            else:
                # Dùng trực tiếp GT 3D (đã là đơn vị Mét và root-relative từ Trainer)
                pose_3d = pose_input

        output = self.pose_mesh_coevo(
            pose_3d,
            ft_map,
            is_train=is_train,
            gt_pose_6d=gt_pose_6d,
            kp2d=kp2d,
            kp_conf=kp_conf,
            pose_valid_mask=pose_valid_mask
        )
        # MotionBERT outputs 3D joints in millimeters; convert to meters so
        # joint_img shares the unit of the GT joints used in the training loss.
        output["joint_img"] = pose_3d
        return output

    def forward(self, image, joints, is_train=True, use_gt_3d=False,
                gt_pose_6d=None, kp2d=None, kp_conf=None,
                pose_valid_mask=None, gt_joints_3d=None, alpha=1.0, joints_mask=None,
                pose_2d=None, lift_alpha=None, hint_on=None, pose_2d_raw=None, cam_bbox=None):
        if self.mode == "teacher":
            return self.forward_teacher(image, joints, is_train, pose_2d=pose_2d,
                                        joints_mask=joints_mask, lift_alpha=lift_alpha, hint_on=hint_on)
        if self.mode == "student":
            return self.forward_student(image, joints, is_train, gt_pose_3d=gt_joints_3d, alpha=alpha,
                                        joints_mask=joints_mask, pose_2d_raw=pose_2d_raw, cam_bbox=cam_bbox)
        return self.forward_arts(
            image, joints, is_train, use_gt_3d,
            gt_pose_6d=gt_pose_6d, kp2d=kp2d, kp_conf=kp_conf,
            pose_valid_mask=pose_valid_mask, gt_joints_3d=gt_joints_3d,
            joints_mask=joints_mask,
        )


def get_model(num_joint, embed_dim, depth=None):
    return ARTS(num_joint, embed_dim)