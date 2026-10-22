import os
import os.path as osp
import sys
import shutil
import yaml
from easydict import EasyDict as edict
import datetime


def init_dirs(dir_list):
    for dir in dir_list:
        if os.path.exists(dir) and os.path.isdir(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)


def _argv_value(flag):
    """Doc gia tri cua --flag tu sys.argv (config.py chay luc import, truoc argparse cua script).
    Ho tro '--flag value' va '--flag=value'."""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + '='):
            return a.split('=', 1)[1]
    return None


cfg = edict()


""" Directory """
cfg.cur_dir = osp.dirname(os.path.abspath(__file__))
cfg.root_dir = osp.join(cfg.cur_dir, '../../')
cfg.data_dir = './data'
cfg.smpl_path = osp.join(cfg.root_dir, 'smplpytorch')
cfg.mano_dir = osp.join(cfg.root_dir, 'manopth')
# ---- Ten thu muc thi nghiem ----
# Thu tu uu tien:
#   1) --exp_name <ten>              -> experiment/<ten>
#   2) bien moi truong FUSEKIN_EXP   -> experiment/<ten>
#   3) mac dinh: <ten file cfg>_<MM-DD_HH-MM-SS>  (vd train_teacher_gt_noshape_10-11_00-05-12)
# Script KHONG phai main/train.py (eval, chan doan...) -> experiment/_eval/<ten script>_<thoi gian>
# de khong lam roi thu muc experiment/.
# An toan: neu --exp_name trung voi thu muc DA CO -> bao loi, KHONG xoa (tru khi --resume_training thi dung lai).
VN_TZ = datetime.timezone(datetime.timedelta(hours=7))
_stamp = datetime.datetime.now(tz=VN_TZ).strftime('%m-%d_%H-%M-%S')
_script = osp.splitext(osp.basename(sys.argv[0]))[0] if sys.argv and sys.argv[0] else 'interactive'
_is_train = (_script == 'train')
_exp_name = _argv_value('--exp_name') or os.environ.get('FUSEKIN_EXP', '')
_resume = '--resume_training' in sys.argv
_cfg_arg = _argv_value('--cfg')

if _exp_name:
    save_folder_path = osp.join('experiment', _exp_name)
elif _is_train:
    _stem = osp.splitext(osp.basename(_cfg_arg))[0] if _cfg_arg else 'exp'
    save_folder_path = osp.join('experiment', f'{_stem}_{_stamp}')
else:
    save_folder_path = osp.join('experiment', '_eval', f'{_script}_{_stamp}')

cfg.exp_name = osp.basename(save_folder_path)
cfg.output_dir = osp.join(cfg.root_dir, save_folder_path)
cfg.graph_dir = osp.join(cfg.output_dir, 'graph')
cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
cfg.res_dir = osp.join(cfg.output_dir, 'result')
cfg.checkpoint_dir = osp.join(cfg.output_dir, 'checkpoint')

print("Experiment Data on {}".format(cfg.output_dir))
if osp.isdir(cfg.output_dir) and os.listdir(cfg.output_dir):
    if _resume:
        print(f"[config] --resume_training: dung lai thu muc da co, KHONG xoa: {cfg.output_dir}")
        for _d in (cfg.graph_dir, cfg.vis_dir, cfg.checkpoint_dir):
            os.makedirs(_d, exist_ok=True)
    elif _exp_name:
        raise SystemExit(f"[config] Thu muc thi nghiem '{cfg.output_dir}' DA TON TAI va khong rong.\n"
                         f"         Doi --exp_name khac, xoa thu muc cu bang tay, hoac them --resume_training.")
    else:
        # Mac dinh co giay -> gan nhu khong trung; neu trung van khong xoa, them hau to.
        cfg.output_dir = cfg.output_dir + '_' + str(os.getpid())
        cfg.exp_name = osp.basename(cfg.output_dir)
        cfg.graph_dir = osp.join(cfg.output_dir, 'graph')
        cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
        cfg.res_dir = osp.join(cfg.output_dir, 'result')
        cfg.checkpoint_dir = osp.join(cfg.output_dir, 'checkpoint')
        print(f"[config] Thu muc trung -> dung {cfg.output_dir}")
        os.makedirs(cfg.output_dir)
        init_dirs([cfg.graph_dir, cfg.vis_dir, cfg.checkpoint_dir])
else:
    os.makedirs(osp.dirname(osp.normpath(cfg.output_dir)), exist_ok=True)
    init_dirs([cfg.output_dir, cfg.graph_dir, cfg.vis_dir, cfg.checkpoint_dir])

""" Dataset """
cfg.DATASET = edict()
cfg.DATASET.train_list = ['Human36M', 'MuCo', 'MSCOCO', 'CrowdPose']
cfg.DATASET.test_list = ['3dpw', '3dpw-crowd', '3dpw-pc', '3dpw-oc']
cfg.DATASET.input_joint_set = 'human36'
cfg.DATASET.target_joint_set = 'human36'
cfg.DATASET.workers = 16
cfg.DATASET.use_gt_input = False
cfg.DATASET.seqlen = 1
cfg.DATASET.stride = 1
cfg.DATASET.noise = 0
# True: che do chi phuc vu train/eval lifter (MotionBERT). Dataset KHONG doc anh (img = tensor rong),
#   KHONG tinh SMPL (H36M, MuCo), tra thang H36M-17 (khong di vong qua SMPL 29 nen khong mat khop).
#   False (mac dinh): hanh vi cu.
cfg.DATASET.lift_only = False
# Nguon 2D dau vao cua 3DPW:
#   'openpose' (mac dinh, hanh vi cu): truong openpose_result trong annotation, thieu Torso/Head_top.
#   'halpe26': file <PW3D>/3DPW_{split}_halpe26.json do tools/extract_halpe26_3dpw.py sinh ra (RTMPose Halpe-26),
#              doi sang H36M-17 du 17 khop (Head_top = dinh dau detector, Torso = trung diem Neck-Hip).
cfg.DATASET.pw3d_2d_source = 'openpose'
# Chi danh gia 3dpw-oc / 3dpw-pc tren cac sequence thuoc TEST (bo sequence trung train/val de tranh ro ri). Mac dinh TAT.
cfg.DATASET.pw3d_ocpc_test_only = False
# Nguong score de coi 1 khop Halpe-26 la hop le (mask = 1)
cfg.DATASET.halpe26_conf_thr = 0.3
# joints_mask MEM: khop hop le (score > nguong, nam trong crop) mang gia tri = score Halpe (cat ve [0, 1]) thay vi 1.
# Khop khong hop le van = 0. Ap dung cho Halpe-26 cua H36M / MuCo / 3DPW. Mac dinh TAT (mask nhi phan nhu cu).
cfg.DATASET.lift_soft_conf = False
# 3DPW test: tra them GT 2D (targets['orig_joint_img'] H36M-17, khung output_hm_shape) + meta orig_joint_valid/trunc.
# Chi de phan tich sai so 2D dau vao; evaluate() khong dung. Mac dinh TAT = hanh vi cu.
cfg.DATASET.pw3d_test_gt2d = False
# Tat augmentation xoay (giu scale/flip). Bat khi lifter = PMCE (2D anh goc khong xoay) de khop 3D GT.
cfg.DATASET.no_rot_aug = False
# Nguon 2D dau vao cua Human36M (train + test):
#   'gt_posefix' (mac dinh, giong JOTR): train = GT 2D + nhieu PoseFix, test = GT 2D sach
#   'halpe26'   : Human36M_{split}_halpe26.npz tu tools/extract_halpe26_h36m.py (nguong = halpe26_conf_thr)
cfg.DATASET.h36m_2d_source = 'gt_posefix'
# Thu muc chua Human36M_{split}_halpe26.npz ('' = <jotr_data_root>/Human36M)
cfg.DATASET.h36m_halpe26_dir = ''
# Nguon 2D dau vao cua MuCo (train): 'gt_posefix' (mac dinh, JOTR) | 'halpe26' (MuCo_train_halpe26.npz
#   tu tools/extract_halpe26_muco.py, nguong = halpe26_conf_thr)
cfg.DATASET.muco_2d_source = 'gt_posefix'
# Thu muc chua MuCo_train_halpe26.npz ('' = <jotr_data_root>/MuCo)
cfg.DATASET.muco_halpe26_dir = ''
cfg.DATASET.jotr_data_root = os.environ.get(
    'JOTR_DATA_ROOT',
    osp.abspath(osp.join(cfg.root_dir, 'data_final')),
)
cfg.use_gt_info = True
# Truoc day Human36M/MuCo/MSCOCO/CrowdPose doc cfg.update_bbox nhung key nay chua khai bao -> AttributeError.
# False: dung bbox da xu ly trong load_data (hanh vi cua 3DCrowdNet/JOTR).
cfg.update_bbox = False
cfg.input_img_shape = (256, 256)
cfg.output_hm_shape = (64, 64, 64)
cfg.bbox_3d_size = 2
cfg.focal = (5000, 5000)
cfg.princpt = (cfg.input_img_shape[1] / 2, cfg.input_img_shape[0] / 2)
cfg.camera_3d_size = cfg.bbox_3d_size  # alias for get_camera_trans

# Copy camera params to cfg.DATASET for Multimodel access
cfg.DATASET.focal = cfg.focal
cfg.DATASET.princpt = cfg.princpt
cfg.DATASET.camera_3d_size = cfg.camera_3d_size
cfg.DATASET.FORCE_FULL_FIT_MASK_3DPW = False  # Override fit_param_valid to all-ones for 3DPW

###############SMPL mean data#################
cfg.DATASET.BASE_DATA_DIR = 'data_final/base_data'
cfg.human_model_path = osp.join(cfg.root_dir, 'data_final', 'base_data', 'human_model_files')
##############################################

""" Model """
cfg.MODEL = edict()
cfg.MODEL.name = 'ARTS'
cfg.MODEL.resnet_type = 50
cfg.MODEL.freeze_backbone = True
cfg.MODEL.pretrained_backbone = False   # khong anh huong ResNetBackbone cua ARTS; dung backbone_pretrained ben duoi
# Trong so pretrain cho ResNetBackbone: 'spin' (ResNet-50 cua SPIN/HMR) | 'imagenet' (torchvision) | '' (ngau nhien)
# Bat cai nay thi backbone cung tu chuan hoa anh kieu ImageNet (mean/std) truoc khi trich dac trung.
cfg.MODEL.backbone_pretrained = ''
cfg.MODEL.spin_checkpoint = 'data_final/base_data/spin_model_checkpoint.pth.tar'
cfg.MODEL.hpe_dim = 256
cfg.MODEL.hpe_dep = 3
cfg.MODEL.joint_dim = 64
cfg.MODEL.vertx_dim = 64
cfg.MODEL.input_shape = (384, 288)
cfg.MODEL.normal_loss_weight = 1e-1
cfg.MODEL.edge_loss_weight = 20
cfg.MODEL.joint_loss_weight = 1e-3
cfg.MODEL.shape_loss_weight = 0.06
cfg.MODEL.pose_loss_weight = 0.06
cfg.MODEL.posenet_pretrained = False
cfg.MODEL.motionbert_pretrained = ''
cfg.MODEL.TEACHER = './experiment/teacher/checkpoint'
cfg.MODEL.STUDENT = './experiment/multimodel/best_epoch.bin'
cfg.MODEL.kd_weight = 1.0
cfg.MODEL.motionbert_pretrained = './experiment/finetune_motionbert/best_epoch.bin'
# True (mac dinh, hanh vi cu): tru root khoi 2D TRUOC khi dua vao MotionBERT (ca luc finetune
#   lan luc dung that trong ARTS.lift_2d_to_3d) -> model chi thay HINH DANG tuong doi, mat tin
#   hieu vi tri nguoi trong khung hinh.
# False: dung dung convention goc cua MotionBERT (xem MB_ft_h36m.yaml: rootrel chi ap dung cho
#   3D target/loss, KHONG ap dung cho input 2D) -> model duoc giu lai tin hieu vi tri tuyet doi
#   trong khung hinh, dung nhu luc pretrain MB_release.
cfg.MODEL.motionbert_2d_rootrel = False
# Cac khop H36M-17 luon bi xoa (toa do 0, mask 0) truoc khi dua vao MotionBERT, ca luc finetune lan luc
# dung that (ARTS.lift_2d_to_3d). OpenPose cua 3DPW khong co 'Torso', 'Head_top' nen checkpoint train tren
# H36M/MuCo can xoa 2 khop nay de dau vao giong 3DPW. Giong motionbert_2d_rootrel: PHAI khop voi checkpoint.
# [] (mac dinh): khong xoa gi (hanh vi cu).
cfg.MODEL.motionbert_drop_joints = []

# Bo lift 2D->3D cho Student: 'motionbert' (mac dinh) | 'pmce' (GraphormerNet, co dac trung anh SPIN global)
#   | 'diffusion' (DiffLifter, main/finetune_difflifter.py; xem difflift_* ben duoi).
# PMCE nhan 2D pixel ANH GOC chuan hoa X/w*2-[1,h/w] (inputs['joints_raw']) + dac trung anh; tra 3D (mm->m).
# PMCE 2D KHONG xoay theo augmentation -> khi dung pmce PHAI dat DATASET.no_rot_aug: True de 3D GT khong xoay.
cfg.MODEL.pose_lifter = 'motionbert'
cfg.MODEL.pmce_checkpoint = ''   # checkpoint PMCE (model_state_dict / epoch_*.bin cua finetune_pmce.py)
cfg.MODEL.pmce_dim = 256   # embed_dim cua PMCE GraphormerNet (TACH khoi hpe_dim cua fusion)
cfg.MODEL.pmce_dep = 3
# DiffLifter (pose_lifter = 'diffusion'): dong bang, nhan [x, y, conf] khung crop [-1, 1] + inputs['cam_bbox'].
# Kien truc doc tu difflift_cfg trong checkpoint. DATASET.lift_soft_conf PHAI giong luc train lifter (r25_cam: True).
cfg.MODEL.difflift_checkpoint = ''
cfg.MODEL.difflift_steps = 1   # S buoc DDIM; 1 = du doan 1 buoc (tot nhat cho MPJPE, diag muc F)
cfg.MODEL.difflift_k = 1       # K gia thuyet, gop trung binh
# Chon gia thuyet THEO TUNG KHOP bang anh (lib/models/hyp_selector.py, cach A'). Mac dinh TAT.
#   Bat: dat difflift_steps / difflift_k = so buoc / so gia thuyet sinh ra (vd 5 / 20). Luc train nhieu DDIM ngau nhien.
#   Loss phu: MPJPE(pose da chon, GT 3D) * difflift_select_w (cong vao loss Student).
# (1) Augmentation bang gia thuyet: LUC TRAIN, moi mau voi xac suat difflift_aug_prob thay pose S=1 bang
#     1 gia thuyet ngau nhien (difflift_aug_steps buoc DDIM). Luc test van S=1. 0 = TAT.
cfg.MODEL.difflift_aug_prob = 0.0
cfg.MODEL.difflift_aug_steps = 5
# (2) Do trai gia thuyet: sinh difflift_spread_k gia thuyet (difflift_spread_steps buoc) -> std theo truc tung khop
#     (B, 17, 3) -> Linear (khoi tao 0) -> cong vao token khop truoc fusion. Pose dua vao van S=1. Mac dinh TAT.
cfg.MODEL.difflift_spread = False
cfg.MODEL.difflift_spread_steps = 5
cfg.MODEL.difflift_spread_k = 20
cfg.MODEL.difflift_select = False
cfg.MODEL.difflift_select_dim = 256
cfg.MODEL.difflift_select_w = 1.0

# ---- Teacher: ep Teacher phai dung anh ----
# (A) Tron GT voi joint MotionBERT lift tu 2D: input = (1-a)*GT + a*lift, a ~ U(0, alpha_max) moi mau luc train.
#     alpha_max = 0 -> tat (hanh vi cu, Teacher nhan GT sach).
cfg.MODEL.teacher_lift_alpha_max = 0.0
#     alpha co dinh luc eval; None -> dung alpha_max. 0.0 -> eval voi GT sach.
cfg.MODEL.teacher_lift_alpha_eval = None
#     Ti le mau luc train nhan GT SACH (a = 0) thay vi a ~ U(0, alpha_max). 0.0 = tat (hanh vi cu).
cfg.MODEL.teacher_lift_clean_prob = 0.0
# (C) Head tham do: doan joint 3D CHI tu token anh dau vao fusion (do/ep nhanh anh mang thong tin tu the).
cfg.MODEL.img_probe = False
cfg.MODEL.img_probe_w = 0.1
# (B) Che/nhieu mot phan khop GT luc train (da co trong Teacher_Trainer._corrupt_joints).
cfg.MODEL.teacher_joint_corrupt_prob = 0.0
cfg.MODEL.teacher_joint_corrupt_mode = 'zero'          # 'zero' | 'noise'
cfg.MODEL.teacher_joint_corrupt_noise_std = 0.15
cfg.MODEL.teacher_joint_corrupt_warmup_epochs = 5
# Trong so loss tham so shape (beta) trong Teacher_Trainer. Beta GT cua 3DPW thuoc SMPL theo gioi tinh,
# model dung SMPL neutral -> beta GT co the day sai kich thuoc co the. 1.0 = nhu cu; 0 = tat han.
cfg.MODEL.shape_param_loss_w = 1.0
# ---- Student: KD / privileged (dung trong Student_Trainer, mac dinh nhu trong code) ----
cfg.MODEL.kd_w_proj = 1.0
cfg.MODEL.kd_w_joint = 1.0
cfg.MODEL.kd_w_global = 1.0
cfg.MODEL.priv_beta = 0.05
cfg.MODEL.init_student_from_teacher = True
# False -> Student "from scratch": khong Teacher, khong KD, khong khoi tao tu Teacher (baseline de so voi KD)
cfg.MODEL.student_use_kd = True
# True (mac dinh, hanh vi cu): tru root khoi 3D lifted TRUOC khi dua vao fusion (FuseKin them vao khi port).
# False: giu nguyen toa do tuyet doi trong cam-space, giong ARTS goc (chi chia 1000, khong tru root);
#   GT van la root-relative -> model tu hoc cach bo qua global translation (nhu ARTS goc thiet ke).
cfg.MODEL.student_3d_rootrel = True
# ---- Che do KD (Student_Trainer) ----
# 'legacy': nhu cu. 'feat': chi KD feature (E1) [+ trong so theo nhieu khop (E2)] [+ InfoNCE (E3)],
#   loss = kd_hard_w*hard + kd_lambda*KD + privileged; KHONG KD output, KHONG adaptive weight.
cfg.MODEL.kd_mode = 'legacy'
cfg.MODEL.kd_feat_dist = 'mse'       # 'mse': MSE(Linear(LN(s)), LN(t)) qua projector | 'cos': nhu cu
cfg.MODEL.kd_lambda = 1.0
cfg.MODEL.kd_hard_w = 1.0
# E2: trong so KD tung khop = clamp(loi_input_khop / tau, wmin, wmax), loi input = |MotionBERT - GT| (met)
#     tau <= 0: TUONG DOI, chia cho loi trung binh cua batch (nen dung: MotionBERT tren 3DPW-train sai rat it)
cfg.MODEL.kd_noise_weight = False
cfg.MODEL.kd_noise_tau = 0.0
cfg.MODEL.kd_noise_wmin = 0.2
cfg.MODEL.kd_noise_wmax = 3.0
# E3: InfoNCE Student->Teacher trong batch (token khop sau fusion + feature toan cuc). 0 = tat.
cfg.MODEL.kd_nce_w = 0.0
cfg.MODEL.kd_nce_temp = 0.1
# Distill phan du (v1): feat_joint_in (bo ma hoa joint, TRUOC fusion) -> feat_joint SAU fusion cua Teacher. 0 = tat.
cfg.MODEL.kd_res_w = 0.0
# Che do distill phan du: 'v1' = nhu tren (dung kd_res_w);
#   'v2' = KD feature (joint/global/InfoNCE) tinh tren 1 luot fusion phu voi token joint DA DETACH
#          -> gradient KD khong vao bo ma hoa joint, chi vao fusion (duong anh). Hard loss van train tat ca.
cfg.MODEL.kd_res_mode = 'v1'
# ---- Goi y nen (Privileged Hint), xem PLAN_TEACHER_HINT.md ----
# Teacher T-hint: anh + joint MotionBERT (GIONG Student) + goi y z (hint_dim so) nen tu [GT, GT - lift].
#   Dung luong goi y bi gioi han: hint_dim nho + nhieu N(0, hint_noise^2) tren z luc train
#   + tat goi y (z = 0) voi xac suat hint_off_prob -> Teacher buoc phai tu fuse anh + joint nhieu.
cfg.MODEL.teacher_hint = False
cfg.MODEL.hint_dim = 16
cfg.MODEL.hint_noise = 0.1
cfg.MODEL.hint_off_prob = 0.3
# Khoi tao T-hint tu checkpoint Teacher GT (moi phan trung ten/shape; bo ma hoa joint khac nen bo qua). '' = khong.
cfg.MODEL.hint_init_from = ''
# Student S-hint: cung kien truc T-hint, tu doan goi y z_hat (fusion 2 luot). Dung voi kd_mode: 'hint'.
cfg.MODEL.student_hint = False
cfg.MODEL.kd_hint_w = 1.0        # MSE(z_hat, z_Teacher)
cfg.MODEL.kd_hint_feat_w = 0.0   # MSE(LN(feat Student), LN(feat Teacher co goi y)), khong projector
# Nhieu 2D luc train cho Student (truoc MotionBERT). Toa do [-1,1] (1.0 ~ 128 px trong crop 256). 0 = tat.
cfg.MODEL.s2d_noise_prob = 0.0        # ti le mau bi lam nhieu
cfg.MODEL.s2d_noise_jitter = 0.02     # std lech nho moi khop (~2.5 px)
cfg.MODEL.s2d_noise_big_prob = 0.15   # xac suat 1 khop bi lech lon
cfg.MODEL.s2d_noise_big_std = 0.2     # std lech lon (~25 px)
cfg.MODEL.s2d_noise_swap_prob = 0.2   # xac suat doi trai/phai 1 cap khop
cfg.MODEL.s2d_noise_drop_prob = 0.05  # xac suat mat 1 khop (mask = 0)
# ---- Kien truc v2 (PLAN_STUDENT_V2.md) - DA CHOT 2026-10-13, MAC DINH BAT ----
# Kien truc cu (config/archive/*.yml) can dat lai: joint_pe2d False, fusion_joint_self_attn False,
# pose_head_mode 'vposer', pose_query_bone False, shape_attn_mode 'global' (checkpoint cu cung vay).
# Luu y: Teacher GT (teacher_hint False) KHONG ho tro joint_pe2d -> phai tat khi chay lai Teacher GT cu.
# PE vi tri 2D Fourier DUNG CHUNG cho token khop (theo 2D detect, nhan joints_mask) va o anh (theo tam o).
cfg.MODEL.joint_pe2d = True
cfg.MODEL.pe2d_num_freq = 6
# Self-attn giua 17 token khop trong moi block fusion (truoc cross-attn).
cfg.MODEL.fusion_joint_self_attn = True
# 'per_joint': moi token -> delta 6D cua khop do + mean pose (v2) | 'vposer': trung binh 24 token -> z (32) -> VPoser (cu).
cfg.MODEL.pose_head_mode = 'per_joint'
# Neo query pose vao xuong: query_j += Linear(concat(ctx[cha_j], ctx[con_j])). Chi dung voi 'per_joint'.
cfg.MODEL.pose_query_bone = True
# 'tokens': shape token cross-attn [ctx 17 khop, token anh] (v2) | 'global': cross-attn 1 key global_ft (cu).
cfg.MODEL.shape_attn_mode = 'tokens'
""" Train Detail """
cfg.TRAIN = edict()
cfg.TRAIN.print_freq = 20
cfg.TRAIN.batch_size = 32
cfg.TRAIN.shuffle = True
cfg.TRAIN.begin_epoch = 1
cfg.TRAIN.end_epoch = 20
cfg.TRAIN.curriculum_epochs = 15.0
cfg.TRAIN.edge_loss_start = 2
cfg.TRAIN.scheduler = 'cosine'
cfg.TRAIN.lr = 1e-4
cfg.TRAIN.backbone_lr_scale = 0.1
cfg.TRAIN.lr_step = [5, 10, 15]
cfg.TRAIN.lr_factor = 0.95
cfg.TRAIN.warmup_epochs = 1
cfg.TRAIN.min_lr = 1e-6
cfg.TRAIN.optimizer = 'adam'
cfg.TRAIN.wandb = False

""" Augmentation """
cfg.AUG = edict()
cfg.AUG.flip = False
cfg.AUG.rotate_factor = 0

""" MotionBERT Finetuning """
cfg.MOTIONBERT = edict()
cfg.MOTIONBERT.finetune = True
cfg.MOTIONBERT.epochs = 60
cfg.MOTIONBERT.checkpoint_frequency = 10
cfg.MOTIONBERT.batch_size = 32
cfg.MOTIONBERT.dropout = 0.0
cfg.MOTIONBERT.learning_rate = 0.0002
cfg.MOTIONBERT.weight_decay = 0.01
cfg.MOTIONBERT.lr_decay = 0.99
cfg.MOTIONBERT.maxlen = 243
cfg.MOTIONBERT.dim_feat = 512
cfg.MOTIONBERT.mlp_ratio = 2
cfg.MOTIONBERT.depth = 5
cfg.MOTIONBERT.dim_rep = 512
cfg.MOTIONBERT.num_heads = 8
cfg.MOTIONBERT.att_fuse = True
cfg.MOTIONBERT.num_joints = 17
cfg.MOTIONBERT.lambda_3d_velocity = 20.0
cfg.MOTIONBERT.lambda_scale = 0.5
cfg.MOTIONBERT.lambda_lv = 0.0
cfg.MOTIONBERT.lambda_lg = 0.0
cfg.MOTIONBERT.lambda_a = 0.0
cfg.MOTIONBERT.lambda_av = 0.0
# File khoi tao (vd MB_release.bin, MB_ft_h36m.bin). '' -> dung --pretrained tren dong lenh (hanh vi cu).
cfg.MOTIONBERT.pretrained = ''
# Thu muc luu checkpoint + log. Mac dinh giu duong dan cu.
cfg.MOTIONBERT.save_dir = 'experiment/finetune_motionbert'
# Tap dung de CHON best checkpoint. '3dpw' (mac dinh, hanh vi cu) = chon tren test -> ro ri test.
#   Nen dung '3dpw-val' (3DPW validation, OpenPose, tach roi train/test).
cfg.MOTIONBERT.val_set = '3dpw'
# So iteration moi epoch. 0 = chay het dataloader (hanh vi cu). Dung khi tron dataset lon (H36M).
cfg.MOTIONBERT.iters_per_epoch = 0
# Khi tron nhieu dataset: True = lay mau can bang (bo nho bi lap lai cho bang bo lon, make_same_len).
#   False = noi thang cac bo, moi epoch duyet moi mau dung 1 lan (ti le theo kich thuoc that).
cfg.MOTIONBERT.balance_datasets = True
# Xac suat moi mau thay 2D nhieu bang GT 2D sach (targets['orig_joint_img']). 0 = tat (hanh vi cu).
#   Dung de 1 checkpoint lift duoc ca 2D nhieu (Student) lan GT 2D (Teacher GT 2D).
cfg.MOTIONBERT.gt2d_prob = 0.0
# Xac suat xoa ngau nhien tung khop hop le cua dau vao 2D luc train (gia lap detector bo sot khop). 0 = tat.
cfg.MOTIONBERT.joint_drop_prob = 0.0
# Checkpoint dung cho lan test cuoi: 'best' (theo val_set, nhu cu) | 'latest' (epoch cuoi, khong chon theo val --
# dung khi val_set la tap khong duoc phep dung de chon model, vd 3dpw-val vi 3DPW-Crowd lay tu split validation)
cfg.MOTIONBERT.final_ckpt = 'best'

""" PMCE (GraphormerNet lifter, main/finetune_pmce.py + tools/extract_img_feat.py) """
cfg.PMCE = edict()
cfg.PMCE.pmce_ckpt = ''            # checkpoint khoi tao (finetune); '' + from_scratch = train tu dau
cfg.PMCE.from_scratch = False      # True: khoi tao ngau nhien, bo qua pmce_ckpt
cfg.PMCE.train_sets = ['h36m-train', 'muco-train', '3dpw-train']   # ten trong utils/pmce_data.FEAT_SETS
cfg.PMCE.val_sets = ['3dpw-val']   # danh gia moi epoch; val_sets[0] dung chon best
cfg.PMCE.test_sets = ['3dpw-test', 'h36m-test']
cfg.PMCE.test_every = 1            # danh gia test_sets moi N epoch (epoch cuoi luon test)
cfg.PMCE.feat_dir = 'data_final/feat_cache'   # dac trung anh tinh san (tools/extract_img_feat.py)
cfg.PMCE.save_dir = 'experiment/finetune_pmce_halpe26'
cfg.PMCE.epochs = 15
cfg.PMCE.lr = 0.00005
cfg.PMCE.lr_decay = 0.95           # nhan lr moi epoch
cfg.PMCE.weight_decay = 0.0
cfg.PMCE.batch_size = 128
cfg.PMCE.workers = 16
cfg.PMCE.final_ckpt = 'latest'     # test cuoi: 'latest' (epoch cuoi) | 'best' (theo val_sets[0])

""" DiffLifter (single-frame diffusion 2D->3D, main/finetune_difflifter.py) """
cfg.DIFFLIFT = edict()
cfg.DIFFLIFT.embed_dim = 256           # chieu an Transformer (D3DP goc = 512)
cfg.DIFFLIFT.depth = 4                 # so Spatial Attention blocks (D3DP goc = 8)
cfg.DIFFLIFT.num_heads = 8
cfg.DIFFLIFT.timesteps = 1000          # T diffusion timesteps khi train
cfg.DIFFLIFT.sampling_timesteps = 5   # S DDIM steps khi inference
cfg.DIFFLIFT.num_proposals = 5        # K hypotheses khi inference
cfg.DIFFLIFT.scale = 1.0              # he so scale toa do 3D truoc khi them noise
cfg.DIFFLIFT.val_set = '3dpw-val'     # tap chon best checkpoint
cfg.DIFFLIFT.save_dir = 'experiment/finetune_difflifter'
cfg.DIFFLIFT.epochs = 100
cfg.DIFFLIFT.lr = 3e-4
cfg.DIFFLIFT.lr_decay = 0.99          # nhan lr moi epoch
cfg.DIFFLIFT.weight_decay = 0.01
cfg.DIFFLIFT.batch_size = 512
cfg.DIFFLIFT.save_freq = 10           # luu checkpoint moi N epoch
cfg.DIFFLIFT.iters_per_epoch = 0      # 0 = chay het dataloader
cfg.DIFFLIFT.print_freq = 50
cfg.DIFFLIFT.balance_datasets = True  # can bang dataset khi tron nhieu nguon
cfg.DIFFLIFT.resume = ''              # checkpoint de resume tu do
# Occlusion augmentation
cfg.DIFFLIFT.joint_drop_prob = 0.1    # xac suat drop tung khop hop le
cfg.DIFFLIFT.limb_drop_prob = 0.1     # xac suat drop ca chi (tay/chan)
cfg.DIFFLIFT.jitter_std = 0.02        # std nhieu 2D (toa do chuan hoa [-1,1])
# Che khuat GIONG THAT (mac dinh TAT): voi xac suat occ_aug_prob moi chi (cang+ban chan / cang+ban tay),
# 2D bi dat SAI CHO (dich ca doan / lay vi tri chi doi xung / dinh vao khop goc) hoac xoa, score bi ha ve gan nguong.
cfg.DIFFLIFT.occ_aug_prob = 0.0
cfg.DIFFLIFT.occ_shift_std = 0.15     # do lech 2D (toa do [-1,1]) khi dat sai cho
# EMA trong so (mac dinh TAT = 0). Bat (vd 0.999) -> eval/luu bang trong so EMA, on dinh hon cho diffusion.
cfg.DIFFLIFT.ema_decay = 0.0
# Danh gia them trong luc train (khong anh huong chon best, best van theo val_set)
cfg.DIFFLIFT.epoch_eval_sets = []     # danh gia MOI epoch, vd ['3dpw']
cfg.DIFFLIFT.full_eval_sets = []      # danh gia moi full_eval_freq epoch, vd ['Human36M', '3dpw-oc', '3dpw-pc']
cfg.DIFFLIFT.full_eval_freq = 5
# Ti le tron dataset moi epoch (theo thu tu train_list), vd [0.4, 0.4, 0.2]. [] = TAT (dung balance_datasets).
# Bat -> lay mau co hoan lai theo ti le, khong lap 3DPW-train ~14 lan/epoch nhu balance.
cfg.DIFFLIFT.dataset_ratio = []
cfg.DIFFLIFT.samples_per_epoch = 0    # so mau moi epoch khi dung dataset_ratio (0 = tong so mau cac tap)
cfg.DIFFLIFT.drop_rate = 0.0          # dropout trong attention/MLP cua denoiser (chong overfit)
cfg.DIFFLIFT.drop_path_rate = 0.1     # stochastic depth (truoc day co dinh 0.1)

""" Diffusion (SMPL_HyperDiff) """
cfg.DIFF = edict()
cfg.DIFF.num_timesteps = 1000
cfg.DIFF.sampling_timesteps = 10
cfg.DIFF.ddim_eta = 0.0
cfg.DIFF.dim_feat = 256
cfg.DIFF.dim_rep = 512
cfg.DIFF.n_layers = 4
cfg.DIFF.num_heads = 8
cfg.DIFF.mlp_ratio = 4.0
cfg.DIFF.drop_path_rate = 0.1
cfg.DIFF.layer_scale_init = 1e-5
cfg.DIFF.loss_type = 'mse'          # 'mse' or 'l1'
cfg.DIFF.p_kp_dropout = 0.05       # condition-dropout probability
cfg.DIFF.t_sampling = 'uniform'    # 'uniform' or 'sqrt' (bias toward large t)
cfg.DIFF.eval_kp_mode = 'detector' # 'gt', 'noisy_gt', 'detector'

cfg.DIFF.NOISE = edict()
# PLACEHOLDER sigmas — calibrate with real detector errors on val 3DPW
cfg.DIFF.NOISE.sigma_per_joint = [
    0.02, 0.03, 0.04, 0.06,   # Pelvis, R_Hip, R_Knee, R_Ankle
    0.03, 0.04, 0.06,          # L_Hip, L_Knee, L_Ankle
    0.02, 0.03, 0.04, 0.04,   # Torso, Neck, Nose, Head
    0.03, 0.05, 0.07,          # L_Shoulder, L_Elbow, L_Wrist
    0.03, 0.05, 0.07,          # R_Shoulder, R_Elbow, R_Wrist
]
cfg.DIFF.NOISE.limb_sigma = 0.02
cfg.DIFF.NOISE.p_big_err = 0.02
cfg.DIFF.NOISE.p_drop = 0.10
cfg.DIFF.NOISE.k = 5.0             # conf decay rate with error
cfg.DIFF.NOISE.use_continuous_conf = False
cfg.DIFF.NOISE.warmup_epochs = 0   # 0 = no curriculum
cfg.DIFF.NOISE.warmup_scale_start = 0.3

""" Loss """
cfg.LOSS = edict()
cfg.LOSS.diff_w = 1.0
cfg.LOSS.w_proj = 1.0              # body_joint_proj weight
cfg.LOSS.w_shape = 1.0             # smpl_shape weight
cfg.LOSS.w_joint_cam = 0.0         # OFF by default
cfg.LOSS.w_mesh = 0.0              # OFF by default
cfg.LOSS.w_pose = 0.0              # OFF by default
cfg.LOSS.t_min_for_smpl = 500      # only compute aux SMPL losses when t >= this
cfg.LOSS.DETACH_POSE_FOR_PROJ = True # Detach pose (but not shape) for 2D projection loss

""" Model (continued) """
cfg.MODEL.REFINER = 'hypergcn'    # 'hypergcn' or 'diffusion'

""" Test Detail """
cfg.TEST = edict()
cfg.TEST.batch_size = 64
cfg.TEST.shuffle = False
cfg.TEST.vis = False
cfg.TEST.weight_path = './experiment/pretrained/mesh_3dpw.pth.tar'
# Dataset dung de chon checkpoint best. '' = trung binh moi DATASET.test_list (nhu cu).
# Vd '3dpw-val' voi test_list ['3dpw-val', '3dpw']: chon tren val, test van chay va in ra moi epoch.
cfg.TEST.select_dataset = ''
# Test 1 LAN sau khi train xong (main/train.py nap best.pth.tar roi danh gia tren cac tap nay). [] = khong test cuoi.
# Vd test_list ['3dpw-val'] (moi epoch chi val) + final_test_list ['3dpw'].
cfg.TEST.final_test_list = []
# Teacher T-hint: moi epoch co danh gia them dong KHONG goi y (z = 0) khong. False = chi dong CO goi y (dong chon best).
# Test cuoi luon in ca 2 dong.
cfg.TEST.hint_eval_nohint = False


def _update_dict(cfg_sub, v, prefix=''):
    """Recursively update *cfg_sub* (an edict) with values from dict *v*.

    Supports nested edicts: if a value in *v* is a dict AND the corresponding
    key in *cfg_sub* is also an edict, recurse.  Otherwise overwrite directly.
    """
    for vk, vv in v.items():
        if vk not in cfg_sub:
            raise ValueError("{}{} not exist in config.py".format(prefix, vk))
        if isinstance(vv, dict) and isinstance(cfg_sub[vk], edict):
            _update_dict(cfg_sub[vk], vv, prefix='{}{}.'.format(prefix, vk))
        else:
            cfg_sub[vk] = vv


def update_config(config_file):
    exp_config = None
    # Luu ban sao file yml vao thu muc thi nghiem de biet run nay dung config nao.
    try:
        shutil.copyfile(config_file, osp.join(cfg.output_dir, osp.basename(config_file)))
    except Exception as e:
        print(f"[config] Khong sao chep duoc {config_file} vao {cfg.output_dir}: {e}")
    with open(config_file) as f:
        exp_config = edict(yaml.safe_load(f))
        for k, v in exp_config.items():
            if k in cfg:
                if isinstance(v, dict):
                    _update_dict(cfg[k], v, prefix='{}.'.format(k))
                else:
                    if k == 'SCALES':
                        cfg[k][0] = (tuple(v))
                    else:
                        cfg[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))

