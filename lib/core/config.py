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
cfg.DATASET.jotr_data_root = os.environ.get(
    'JOTR_DATA_ROOT',
    osp.abspath(osp.join(cfg.root_dir, 'data_final')),
)
cfg.use_gt_info = True
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
cfg.MODEL.pretrained_backbone = False
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
cfg.MODEL.STUDENT = './experiment/student/checkpoint/best.pth.tar'
cfg.MODEL.kd_weight = 1.0
cfg.MODEL.motionbert_pretrained = './experiment/finetune_motionbert/best_epoch.bin'
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

