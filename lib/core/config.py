import os
import os.path as osp
import shutil
import yaml
from easydict import EasyDict as edict
import datetime


def init_dirs(dir_list):
    for dir in dir_list:
        if os.path.exists(dir) and os.path.isdir(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)


cfg = edict()


""" Directory """
cfg.cur_dir = osp.dirname(os.path.abspath(__file__))
cfg.root_dir = osp.join(cfg.cur_dir, '../../')
cfg.data_dir = './data'
cfg.smpl_path = osp.join(cfg.root_dir, 'smplpytorch')
cfg.mano_dir = osp.join(cfg.root_dir, 'manopth')
KST = datetime.timezone(datetime.timedelta(hours=8))
save_folder = 'exp_' + str(datetime.datetime.now(tz=KST))[5:-16]
save_folder = save_folder.replace(" ", "_")
save_folder = save_folder.replace(":", "_")
save_folder_path = 'experiment/{}'.format(save_folder)

cfg.output_dir = osp.join(cfg.root_dir, save_folder_path)
cfg.graph_dir = osp.join(cfg.output_dir, 'graph')
cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
cfg.res_dir = osp.join(cfg.output_dir, 'result')
cfg.checkpoint_dir = osp.join(cfg.output_dir, 'checkpoint')

print("Experiment Data on {}".format(cfg.output_dir))
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
cfg.MODEL.teacher_joint_corrupt_warmup_epochs = 0
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
# Xac suat moi mau thay 2D nhieu bang GT 2D sach (targets['orig_joint_img']). 0 = tat (hanh vi cu).
#   Dung de 1 checkpoint lift duoc ca 2D nhieu (Student) lan GT 2D (Teacher GT 2D).
cfg.MOTIONBERT.gt2d_prob = 0.0
# Xac suat xoa ngau nhien tung khop hop le cua dau vao 2D luc train (gia lap detector bo sot khop). 0 = tat.
cfg.MOTIONBERT.joint_drop_prob = 0.0

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

