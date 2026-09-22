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
cfg.smpl_dir = osp.join(cfg.root_dir, 'smplpytorch')
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
    osp.abspath(osp.join(cfg.root_dir, '..', '..', 'JOTR', 'data')),
)
cfg.input_img_shape = (256, 256)
cfg.output_hm_shape = (64, 64, 64)
cfg.bbox_3d_size = 2
cfg.focal = (5000, 5000)
cfg.princpt = (cfg.input_img_shape[1] / 2, cfg.input_img_shape[0] / 2)

###############SMPL mean data#################
cfg.DATASET.BASE_DATA_DIR = 'data/base_data'
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
cfg.MODEL.posenet_path = './experiment/pretrained/pose_3dpw.pth.tar'


""" Train Detail """
cfg.TRAIN = edict()
cfg.TRAIN.print_freq = 20
cfg.TRAIN.batch_size = 32
cfg.TRAIN.shuffle = True
cfg.TRAIN.begin_epoch = 1
cfg.TRAIN.end_epoch = 20
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

""" Test Detail """
cfg.TEST = edict()
cfg.TEST.batch_size = 64
cfg.TEST.shuffle = False
cfg.TEST.vis = False
cfg.TEST.weight_path = './experiment/pretrained/mesh_3dpw.pth.tar'


def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in cfg[k]:
            cfg[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.safe_load(f))
        for k, v in exp_config.items():
            if k in cfg:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        cfg[k][0] = (tuple(v))
                    else:
                        cfg[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))
