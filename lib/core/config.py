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
cfg.smpl_path = osp.join(cfg.root_dir, 'smplpytorch')
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
cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
cfg.checkpoint_dir = osp.join(cfg.output_dir, 'checkpoint')

print("Experiment Data on {}".format(cfg.output_dir))
if osp.isdir(cfg.output_dir) and os.listdir(cfg.output_dir):
    if _resume:
        print(f"[config] --resume_training: dung lai thu muc da co, KHONG xoa: {cfg.output_dir}")
        for _d in (cfg.vis_dir, cfg.checkpoint_dir):
            os.makedirs(_d, exist_ok=True)
    elif _exp_name:
        raise SystemExit(f"[config] Thu muc thi nghiem '{cfg.output_dir}' DA TON TAI va khong rong.\n"
                         f"         Doi --exp_name khac, xoa thu muc cu bang tay, hoac them --resume_training.")
    else:
        # Mac dinh co giay -> gan nhu khong trung; neu trung van khong xoa, them hau to.
        cfg.output_dir = cfg.output_dir + '_' + str(os.getpid())
        cfg.exp_name = osp.basename(cfg.output_dir)
        cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
        cfg.checkpoint_dir = osp.join(cfg.output_dir, 'checkpoint')
        print(f"[config] Thu muc trung -> dung {cfg.output_dir}")
        os.makedirs(cfg.output_dir)
        init_dirs([cfg.vis_dir, cfg.checkpoint_dir])
else:
    os.makedirs(osp.dirname(osp.normpath(cfg.output_dir)), exist_ok=True)
    init_dirs([cfg.output_dir, cfg.vis_dir, cfg.checkpoint_dir])

""" Dataset """
cfg.DATASET = edict()
cfg.DATASET.train_list = ['Human36M', 'MuCo', 'MSCOCO', 'CrowdPose']
cfg.DATASET.test_list = ['3dpw', '3dpw-crowd', '3dpw-pc', '3dpw-oc']
cfg.DATASET.workers = 16
# True: che do chi phuc vu train/eval lifter (DiffLifter). Dataset KHONG doc anh (img = tensor rong),
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
# Mau train FuseKin (Teacher / Student, co anh) THONG NHAT cho Human36M / MuCo / MSCOCO / CrowdPose
# (lib/utils/fusekin_sample.py): cung bo key voi 3DPW train, 2D Halpe-26, GT 3D / mesh tu SMPL NeuralAnnot,
# MSCOCO / CrowdPose xoay sang camera uoc luong. Can h36m_2d_source / muco_2d_source = 'halpe26'. Mac dinh TAT.
cfg.DATASET.fusekin_unified = False
# 3DPW test: tra them GT 2D (targets['orig_joint_img'] H36M-17, khung output_hm_shape) + meta orig_joint_valid/trunc.
# Chi de phan tich sai so 2D dau vao; evaluate() khong dung. Mac dinh TAT = hanh vi cu.
cfg.DATASET.pw3d_test_gt2d = False
# Tat augmentation xoay (giu scale/flip). Chi dung cho chan doan lifter.
cfg.DATASET.no_rot_aug = False
# Mask khop Halpe SAI bang GT 2D (chi tap TRAIN Human36M / MuCo / MSCOCO / CrowdPose): khop Halpe hop le lech
# GT 2D > nguong (ty le canh crop, vd 0.1) -> mask = 0 (toa do ve 0). Khop khong co GT tuong ung (vd Torso cua COCO)
# khong kiem tra. Ap dung ca lifter (lift_only) va Student (fusekin_unified). 0 = TAT (hanh vi cu).
cfg.DATASET.halpe_gt_mask_thr = 0.0
# Hau to file Halpe-26 cua 3DPW (test / OC / PC / Crowd): '' = 3DPW_{tag}_halpe26.json (trich bang bbox GT, cu);
# '_opbox' = trich bang bbox tu khop OpenPose cua dung nguoi (tools/extract_halpe26_3dpw.py --bbox_source openpose_best)
cfg.DATASET.pw3d_halpe_suffix = ''
# THU NGHIEM ORACLE (2026-10-29): thay 2D dau vao (Halpe) bang GT 2D (orig_joint_img, khung hm) o MOI bo train / test,
# mask = khop GT trong khung crop (conf 1). Dung cho ca lifter (lift_only) va Student. Tu bat pw3d_test_gt2d.
# GT 2D: H36M / MuCo = khop mocap chieu; MSCOCO / CrowdPose / 3DPW = H36M-17 hoi quy tu mesh SMPL chieu. Crop van tu Halpe.
cfg.DATASET.input_2d_gt = False
# Kieu JOTR (2026-10-30): 'all' = oracle (moi tap GT 2D); 'train' = CHI tap train GT 2D (+ nhieu duoi), tap test
# dung detector theo pw3d_2d_source ('openpose' -> 3DPW / OC / PC OpenPose, Crowd Higher-HRNet, nhu JOTR / 3DCrowdNet).
cfg.DATASET.input_2d_gt_split = 'all'
# Chi giu cac khop GT 2D nay o dau vao train (khop detector test khong co -> mask 0, vd Head_top voi OpenPose). [] = het.
cfg.DATASET.gt2d_input_joints = []
# Nhieu tren GT 2D (chi tap train): rung (toa do [-1, 1]; 0.03 ~ 4 px crop 256), mat khop, dao trai / phai tung cap
cfg.DATASET.gt2d_jitter_std = 0.0
cfg.DATASET.gt2d_drop_prob = 0.0
cfg.DATASET.gt2d_lr_swap_prob = 0.0
# Torso dau vao = trung diem Neck-Pelvis (mask = Neck * Pelvis) cho MOI tap (train GT + test detector) -> 2 ben cung cach
cfg.DATASET.input_torso_mid = False
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

# Tham so camera ao dung trong model (get_camera_trans / get_coord) va loss chieu 2D
cfg.DATASET.focal = cfg.focal
cfg.DATASET.princpt = cfg.princpt
cfg.DATASET.camera_3d_size = cfg.camera_3d_size

""" Model """
cfg.MODEL = edict()
cfg.MODEL.name = 'teacher'             # 'teacher' (anh + GT 3D) | 'student' (anh + 2D -> DiffLifter)
cfg.MODEL.resnet_type = 50
# Trong so pretrain cho ResNetBackbone: 'spin' (ResNet-50 cua SPIN/HMR) | 'imagenet' (torchvision) | '' (ngau nhien)
# Bat cai nay thi backbone cung tu chuan hoa anh kieu ImageNet (mean/std) truoc khi trich dac trung.
cfg.MODEL.backbone_pretrained = ''
cfg.MODEL.spin_checkpoint = 'data_final/base_data/spin_model_checkpoint.pth.tar'
cfg.MODEL.hpe_dim = 256                # chieu token fusion (embed_dim)
cfg.MODEL.TEACHER = './experiment/teacher/checkpoint'   # checkpoint Teacher (Student KD / khoi tao)

# ---- DiffLifter cua Student (dong bang): nhan [x, y, conf] khung crop [-1, 1] + inputs['cam_bbox'] ----
# Kien truc doc tu difflift_cfg trong checkpoint. DATASET.lift_soft_conf PHAI giong luc train lifter (r25_cam: True).
cfg.MODEL.difflift_checkpoint = ''
cfg.MODEL.difflift_steps = 1   # S buoc DDIM; 1 = du doan 1 buoc (tot nhat cho MPJPE, diag muc F)
cfg.MODEL.difflift_k = 1       # K gia thuyet, gop trung binh
# Gate dieu chinh cap nhat tu anh theo tung khop. Mac dinh TAT.
# g=2*sigmoid(MLP([LN(joint),LN(image_update),log1p(std*10)])), khoi tao g=1; std tu K gia thuyet rieng
# (difflift_spread_steps / difflift_spread_k), pose chinh van S1/K1.
cfg.MODEL.difflift_spread_gate = False
cfg.MODEL.difflift_spread_gate_zero = False   # Doi chung: chi thay std dau vao gate bang 0, van sinh K.
cfg.MODEL.difflift_spread_gate_hidden = 64
cfg.MODEL.difflift_spread_steps = 5
cfg.MODEL.difflift_spread_k = 20

# ---- Teacher: ep Teacher phai dung anh ----
# Head tham do: doan joint 3D CHI tu token anh dau vao fusion (do/ep nhanh anh mang thong tin tu the).
cfg.MODEL.img_probe = False
cfg.MODEL.img_probe_w = 0.1
# Che (dat 0) mot phan khop GT luc train (Teacher_Trainer._corrupt_joints), tang dan trong warmup epoch dau.
cfg.MODEL.teacher_joint_corrupt_prob = 0.0
cfg.MODEL.teacher_joint_corrupt_warmup_epochs = 5
# Trong so loss tham so shape (beta). Beta GT cua 3DPW thuoc SMPL theo gioi tinh,
# model dung SMPL neutral -> beta GT co the day sai kich thuoc co the. 1.0 = binh thuong; 0 = tat han.
cfg.MODEL.shape_param_loss_w = 1.0

# ---- Student: KD feature / privileged (Student_Trainer) ----
# loss = kd_hard_w*hard + kd_lambda*KD + privileged. KD o 3 diem noi (trong so 0 = tat diem do).
# False -> Student khong Teacher: khong KD, khong khoi tao tu Teacher (baseline)
cfg.MODEL.student_use_kd = True
cfg.MODEL.init_student_from_teacher = True
cfg.MODEL.kd_w_proj = 1.0
cfg.MODEL.kd_w_joint = 1.0
cfg.MODEL.kd_w_global = 1.0
cfg.MODEL.priv_beta = 0.05
cfg.MODEL.kd_feat_dist = 'mse'       # 'mse': MSE(Linear(LN(s)), LN(t)) qua projector | 'cos': 1 - cos(LN(s), LN(t))
# KD quan he: thay loss joint bang MSE ma tran cosine LN token sau fusion; global giu nguyen.
cfg.MODEL.kd_joint_relation = False
# Chi log norm gradient tren tham so fusion o batch dau moi epoch, khong doi gradient optimizer.
cfg.MODEL.kd_joint_relation_log_grad = False
cfg.MODEL.kd_lambda = 1.0
cfg.MODEL.kd_hard_w = 1.0

# ---- FuseKin-v3 (2026-10-26): moi module mac dinh TAT -> kien truc / hanh vi v2 ----
# [A] Backbone: fine-tune ResNet (lr = TRAIN.lr * v3_bb_lr_scale; BN giu thong ke) va/hoac ghep heatmap 2D Halpe
#     vao feature stride 4 (Conv3x3 khoi tao 0, 3DCrowdNet). Heatmap Gauss sigma tinh theo o 64x64.
cfg.MODEL.v3_bb_finetune = False
cfg.MODEL.v3_bb_lr_scale = 0.1
cfg.MODEL.v3_bb_heatmap = False
cfg.MODEL.v3_heatmap_sigma = 2.0
# [B] Feature anh F16 (layer3) + F8 (layer4) lay tai khop 2D -> cong vao token khop (Linear khoi tao 0).
cfg.MODEL.v3_joint_feat = False
# [B'] Bias attention theo so hop tren khung xuong trong self-attn giua cac khop (BlockGCN), khoi tao 0.
cfg.MODEL.v3_hop_bias = False
# [C] Head phan bo do sau tung khop (PandaPose): phan loai v3_depth_bins bin trong [-R, R] met (tuong doi pelvis),
#     CE mem (sigma 1 bin) trong so v3_depth_loss_w; do sau ky vong + entropy cong vao token khop (khoi tao 0).
cfg.MODEL.v3_depth_head = False
cfg.MODEL.v3_depth_bins = 64
cfg.MODEL.v3_depth_range = 1.0
cfg.MODEL.v3_depth_loss_w = 0.1
# [D] Tinh chinh lap kieu ReFit: so buoc (0 = tat), GRU tung tham so (24 goc + shape + camera), feedback = feature
#     tai khop da chieu (cua so 3x3 tren F16) + sai lech 2D Halpe x conf + cam_bbox; dropout theo khop; loss moi buoc
#     trong so gamma^(so buoc con lai).
cfg.MODEL.v3_refine_iters = 0
cfg.MODEL.v3_refine_feat_dim = 32
cfg.MODEL.v3_refine_hidden = 32
cfg.MODEL.v3_refine_dropout = 0.25
cfg.MODEL.v3_refine_gamma = 0.85
# [E] Loss khop 3D co do bat dinh hoc duoc tung khop (|e| / sigma + log sigma) va nhieu 2D luc train Student
#     (ti le mau v3_pose2d_noise_prob; 0 = tat): jitter std (toa do [-1,1]) + mat khop.
cfg.MODEL.v3_joint_uncert = False
cfg.MODEL.v3_pose2d_noise_prob = 0.0
cfg.MODEL.v3_pose2d_noise_std = 0.02
cfg.MODEL.v3_pose2d_drop_prob = 0.05

""" Train Detail """
cfg.TRAIN = edict()
cfg.TRAIN.print_freq = 20
cfg.TRAIN.batch_size = 32
cfg.TRAIN.shuffle = True
# Tron nhieu bo train FuseKin theo ti le (theo thu tu DATASET.train_list), vd [0.25, 0.25, 0.25, 0.25].
# [] = TAT (cach cu: MultipleDatasets keo moi bo bang bo lon nhat). samples_per_epoch: so mau / epoch (0 = tong so mau).
cfg.TRAIN.dataset_ratio = []
cfg.TRAIN.samples_per_epoch = 0
# Cach tron bo train (2026-10-29). 'ratio' = nhu tren (dataset_ratio + samples_per_epoch, lay mau co hoan lai);
# 'full' = moi epoch duyet HET hop 4 bo, moi mau dung 1 lan, xao tron (ti le theo kich thuoc bo; bo qua dataset_ratio);
# 'jotr' = kieu 3DCrowdNet / JOTR: nhom 3D (Human36M, MuCo) va nhom 2D (MSCOCO, CrowdPose) keo bang nhau (50 / 50),
#          trong moi nhom ti le theo kich thuoc bo.
cfg.TRAIN.mix_mode = 'ratio'
cfg.TRAIN.begin_epoch = 1
cfg.TRAIN.end_epoch = 20
cfg.TRAIN.scheduler = 'cosine'
cfg.TRAIN.lr = 1e-4
cfg.TRAIN.lr_step = [5, 10, 15]
cfg.TRAIN.lr_factor = 0.95
cfg.TRAIN.warmup_epochs = 1
cfg.TRAIN.min_lr = 1e-6
cfg.TRAIN.optimizer = 'adam'
cfg.TRAIN.wandb = False

""" DiffLifter (single-frame diffusion 2D->3D, main/finetune_difflifter.py) """
cfg.DIFFLIFT = edict()
# Chi eval_difflifter_3dpw.py dung: GT/loader/noise nhu Student, mac dinh TAT de giu lenh cu.
cfg.DIFFLIFT.eval_student_input = False
cfg.DIFFLIFT.embed_dim = 256           # chieu an Transformer (D3DP goc = 512)
cfg.DIFFLIFT.depth = 4                 # so Spatial Attention blocks (D3DP goc = 8)
cfg.DIFFLIFT.num_heads = 8
cfg.DIFFLIFT.gcn_attn = False          # GCN skeleton + attention song song; TAT = backbone cu
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
# Nhieu "bat nham nguoi" (mac dinh TAT): voi xac suat wp_aug_prob moi mau, 1 doan dau chi (co tay | khuyu+co tay |
# co chan | goi+co chan; tay chiem wp_arm_prob) bi DOI ca doan di wp_shift_min..wp_shift_max (toa do [-1,1], 1 = 128 px
# crop 256) theo huong ngau nhien, GIU NGUYEN conf -> giong loi that cua Halpe o canh dong nguoi (khop sai, conf cao).
cfg.DIFFLIFT.wp_aug_prob = 0.0
cfg.DIFFLIFT.wp_shift_min = 0.3125     # 40 px
cfg.DIFFLIFT.wp_shift_max = 0.9375     # 120 px
cfg.DIFFLIFT.wp_arm_prob = 0.7
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
# Cuoi moi epoch: MPJPE S=1 tren N mau CO DINH cua tung bo train (giu augmentation train, seed co dinh
# -> so duoc giua cac epoch). 0 = TAT.
cfg.DIFFLIFT.train_eval_samples = 0
# Chi so chon best tren val_set: 'mpjpe' (mac dinh, hanh vi cu) | 'pa_mpjpe'
cfg.DIFFLIFT.select_metric = 'mpjpe'

""" Selector: chon gia thuyet DiffLifter theo nhom khop bang anh (main/train_selector.py, eval_selector.py) """
# Chi script Selector doc nhom nay; train.py / Teacher / Student khong dung -> khong anh huong run cu.
cfg.SELECTOR = edict()
cfg.SELECTOR.K = 20                   # so gia thuyet DiffLifter (cong pose mac dinh S1/K1 -> K + 1 ung vien)
cfg.SELECTOR.steps = 5                # so buoc DDIM khi sinh gia thuyet
cfg.SELECTOR.use_flip = True          # moi chi them 4 trang thai lat do sau (doan tren / duoi)
# Checkpoint chua 'diff_lifter.*' (vd best.pth.tar cua Student) de ghi de trong so lifter; '' = dung difflift_checkpoint
cfg.SELECTOR.lifter_from_ckpt = ''
cfg.SELECTOR.dim = 256
cfg.SELECTOR.heads = 8
cfg.SELECTOR.enc_layers = 2
cfg.SELECTOR.dec_layers = 2
cfg.SELECTOR.dropout = 0.1
cfg.SELECTOR.tau = 0.02               # met: muc tieu mem softmax(-loi_nhom / tau)
cfg.SELECTOR.ce_w = 1.0               # trong so CE mem
cfg.SELECTOR.exp_err_w = 0.0          # trong so sai so ky vong sum(softmax * loi) (cm); 0 = tat (ban v1)
# Prior pose mac dinh: danh dau phuong an 0 (S1/K1, khong lat) + bias logit khoi tao = gia tri nay. 0 = tat (ban v1).
cfg.SELECTOR.base_prior_init = 0.0
cfg.SELECTOR.lr = 1e-4
cfg.SELECTOR.weight_decay = 1e-4
cfg.SELECTOR.epochs = 20
cfg.SELECTOR.lr_step = [14, 18]
cfg.SELECTOR.lr_factor = 0.1
# Danh gia nhanh moi epoch (chi log + best tham khao). '' = tat. 3dpw-val chua 2 sequence Crowd -> neu bao Crowd thi
# dung last.pth.tar, khong dung best.
cfg.SELECTOR.val_set = ''
cfg.SELECTOR.val_max_samples = 2000

""" Test Detail """
cfg.TEST = edict()
cfg.TEST.batch_size = 64
cfg.TEST.shuffle = False
# Dataset dung de chon checkpoint best. '' = trung binh moi DATASET.test_list (nhu cu).
# Vd '3dpw-val' voi test_list ['3dpw-val', '3dpw']: chon tren val, test van chay va in ra moi epoch.
cfg.TEST.select_dataset = ''
# Test 1 LAN sau khi train xong (main/train.py nap best.pth.tar roi danh gia tren cac tap nay). [] = khong test cuoi.
# Vd test_list ['3dpw-val'] (moi epoch chi val) + final_test_list ['3dpw'].
cfg.TEST.final_test_list = []
# Luu them checkpoint/best_<tap>.pth.tar cho MOI tap trong DATASET.test_list (theo MPJPE tap do), ngoai best.pth.tar.
# Giong cach JOTR / DPMesh phat hanh 1 checkpoint / tap. Mac dinh tat.
cfg.TEST.save_best_per_set = False


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

