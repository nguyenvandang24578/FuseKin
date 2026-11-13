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
# Refiner 2D (2026-11-03): mau train fusekin_unified (nguon 2D gt_posefix) tra them targets['joint_img_clean']
# (17, 2) = GT 2D TRUOC PoseFix, khung output_hm_shape, cung dinh nghia khop voi 2D dau vao. Khong doi 2D dau vao.
# Mac dinh TAT (bo key giu nguyen nhu cu).
cfg.DATASET.return_clean_2d = False
# Nhan 2D DAY (2026-11-04): file json danh sach dinh SMPL (tools/select_dense_vertices.py, vd config/dense48_fps.json).
# Bat -> mau fusekin_unified va 3DPW test (khi pw3d_test_gt2d) tra them targets['dense_img'] (N, 2) khung
# output_hm_shape + meta['dense_trunc'] (N, 1). '' = TAT (bo key nhu cu).
cfg.DATASET.dense2d_vertex_file = ''
# CHI DE VE KIEM TRA (tools/vis_dense2d.py): mau fusekin_unified / 3DPW test (pw3d_test_gt2d) tra them
# targets['mesh_img'] (6890, 2) = toan bo mesh GT chieu vao crop. Mac dinh TAT (khong dung khi train).
cfg.DATASET.debug_mesh2d = False
# Bo meta 'raw_smpl_*' (chi PW3D train tra) de tron 3dpw-train voi 4 bo fusekin_unified ma collate khong loi
# (2026-11-04, ImgLifter day). Mac dinh TAT.
cfg.DATASET.drop_raw_smpl_stats = False
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
# KIEU JOTR (viet lai 2026-10-31 de test lifter gt2dnoise, mac dinh TAT = hanh vi cu):
#   input_2d_gt_split: 'all' (oracle, moi bo) | 'train' (chi bo train dung GT 2D; test giu detector)
#   gt2d_input_joints: [] = du 17 khop; danh sach ten -> khop khac mask 0 (vd bo Head_top)
#   gt2d_jitter_std / gt2d_drop_prob / gt2d_lr_swap_prob: nhieu CHI o bo train (toa do [-1, 1]; bo khop; dao cap trai/phai)
#   input_torso_mid: Torso = trung diem Neck-Pelvis (mask = ca 2) o MOI bo (train + test)
cfg.DATASET.input_2d_gt_split = 'all'
cfg.DATASET.gt2d_input_joints = []
cfg.DATASET.gt2d_jitter_std = 0.0
cfg.DATASET.gt2d_drop_prob = 0.0
cfg.DATASET.gt2d_lr_swap_prob = 0.0
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
# Nguon 2D dau vao cua MSCOCO / CrowdPose o nhanh LIFTER (lift_only, train) (2026-10-31):
#   'halpe26'    (mac dinh, hanh vi cu): Halpe-26 detect san (<name>_train_halpe26.npz)
#   'gt_posefix' (giong JOTR dataloader/MSCOCO.py, CrowdPose.py): keypoint annotation + pelvis -> PoseFix
#                (near_joints IoU >= 0.1), mask = khop GT hop le & trong crop. Chi giu nguoi co NeuralAnnot (nhan 3D).
cfg.DATASET.coco_2d_source = 'halpe26'
cfg.DATASET.crowdpose_2d_source = 'halpe26'
# jotr_strict (mac dinh TAT): dong bo 100% voi JOTR o cac cho FuseKin da doi:
#   - MuCo lifter: dau vao doi sang H36M-17 theo TEN GOC MuCo (khong alias Thorax->Neck, Spine->Torso, Head->Nose),
#     giong JOTR MuCo -> SMPL-30 -> H36M-17 (Neck / Torso / Nose mask 0). Nhan 3D van dung alias.
#   - CrowdPose: loc NeuralAnnot nhu JOTR (fit_err < fitting_thr -> bo).
cfg.DATASET.jotr_strict = False
# Nhan xac suat 4 loai loi PoseFix (jitter, miss, inversion, swap) cua utils/posefix.synthesize_pose (mac dinh 1.0 = JOTR).
# Vd 0.5: loi nhe hon cho lifter (lifter khong co anh de sua loi). Ap dung cho moi bo dung gt_posefix.
cfg.DATASET.posefix_scale = 1.0
# Ban PoseFix vector hoa (utils/posefix.synthesize_pose_fast): cung thu tu goi random -> giong tung bit ban goc, nhanh hon.
# Chi bat sau khi tools/check_posefix_fast.py bao GIONG TUNG BIT.
cfg.DATASET.posefix_fast = False
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
# Student v4 (2026-11-05): thay DiffLifter bang Refiner 2D day + ImgLifter (ca hai dong bang). '' = TAT (DiffLifter cu).
cfg.MODEL.imglift_checkpoint = ''          # vd experiment/img_lifter_dense48_3dpw_markov/checkpoint/last.pth.tar
cfg.MODEL.imglift_refiner_checkpoint = ''  # '' = refiner luu trong checkpoint ImgLifter
# FuseKin v8 (2026-11-13, nhanh git v8-e2e, nguoi dung duyet): mang END-TO-END anh lam chinh (lib/models/fusekin_v8.py).
# True -> ARTS.get_model tra V8Net thay ARTS (KHONG nap Refiner / ImgLifter / Student cu). Mac dinh TAT.
cfg.MODEL.v8_e2e = False
cfg.MODEL.v8_dim = 256          # so kenh token (encoder / decoder)
cfg.MODEL.v8_enc_layers = 3     # lop transformer encoder tren 64 token anh
cfg.MODEL.v8_dec_layers = 3     # lop decoder (moi lop hoi quy phan du SMPL + khop 3D)
cfg.MODEL.v8_joint_drop = 0.1   # che ngau nhien khop 2D dau vao luc train (giong JOTR)
cfg.MODEL.imglift_K = 5                    # so gia thuyet, gop trung binh
cfg.MODEL.imglift_S = 2                    # so buoc DDIM
# True: 2D cho Student (heatmap [A], PE, [B], feedback [D]) = 17 khop 2D cua Refiner (mask = 1 moi khop) thay 2D detector
cfg.MODEL.student_2d_from_refiner = False
# Student v5 (2026-11-07): giai ma 3D tu ctx (sau HyperGCN) -> vector xuong cha -> con -> MLP cong vao query SMPL.
# v5_ctx_loss_w: trong so loss smooth-L1 ctx_3d vs GT (nhu privileged). Mac dinh TAT.
cfg.MODEL.v5_bone_query = False
cfg.MODEL.v5_ctx_loss_w = 1.0
# Student v6 (2026-11-08): mesh neo vao hinh hoc qua IK giai tich (HybrIK). Mac dinh TAT.
#   pose17 = lift + delta17(ctx) [ctx_3d, loss v5_ctx_loss_w] -> J29 = M x pose17 + delta29(query) -> IK (beta, twist)
#   -> goc SMPL -> get_coord nhu cu. Moi delta / twist khoi tao 0 -> luc dau = baseline "lifter + M + IK" (§19).
#   v6_m_path: anh xa M (29 x 17) tu tools/fit_h36m_to_smpl29.py. v6_j29_loss_w: smooth-L1 J29 vs GT29 (tu mesh GT).
#   v6_delta17 / v6_delta29 / v6_twist: bat / tat tung phan (ablation); False = giu co dinh (delta 0, twist 0).
cfg.MODEL.v6_ik = False
cfg.MODEL.v6_m_path = 'data_final/h36m17_to_smpl29.npy'
cfg.MODEL.v6_delta17 = True
cfg.MODEL.v6_delta29 = True
cfg.MODEL.v6_twist = True
cfg.MODEL.v6_j29_loss_w = 1.0
# Khoa ma tran sieu canh p_adj / b_adj cua HyperGCN (khong sua hypergraph.py). Ly do: §17 (v4 tong cot ve 0 -> BN no).
cfg.MODEL.hgcn_freeze_adj = False
# Student v7 (2026-11-09): mesh NEO theo pose lifter, Student chi doan phan BE MAT. Mac dinh TAT.
#   Pose : J29 = M x pose3d (than giu nguyen) + delta 7 diem dau chi {10,11,22,23,24,27,28} (body_tok) -> IK
#          twist tu part_tok (doan chi chua xuong). v7_delta17: cho Student sua 17 khop (CHI de ablation).
#   Shape: beta = beta_xuong (Gauss-Newton tu do dai 16 xuong H36M cua pose3d, khong tham so) + dbeta (body_tok, anh)
#   Loss : bo loss tham so pose (v7_no_pose_param_loss), bo privileged (v7_no_privileged), J29 chi 7 diem dau chi,
#          + v7_dbeta_w * TB(dbeta^2). Dung chung v6_m_path, hgcn_freeze_adj. v7_no_hgcn: bo qua HyperGCN (ablation).
cfg.MODEL.v7_anchor = False
cfg.MODEL.v7_part_readout = True       # doc token sieu canh (10 doan chi, 5 chi/than) sau HyperGCN
cfg.MODEL.v7_twist = True
cfg.MODEL.v7_delta_limb = True
cfg.MODEL.v7_dbeta = True
cfg.MODEL.v7_delta17 = False
cfg.MODEL.v7_no_hgcn = False
cfg.MODEL.v7_bone_gn_iters = 3          # so buoc Gauss-Newton giai beta tu do dai xuong
cfg.MODEL.v7_bone_beta_reg = 1e-4       # lambda * |beta|^2 trong Gauss-Newton (residual tinh bang met; J^T J ~ 1e-3)
cfg.MODEL.v7_dbeta_w = 0.01
cfg.MODEL.v7_j29_loss_w = 1.0
cfg.MODEL.v7_no_pose_param_loss = False
cfg.MODEL.v7_no_privileged = False
# HyperGCN tren 24 token query SMPL (2026-11-12, lib/models/hypergraph_smpl.py): sau pose_context_attn, truoc head be mat
#   (twist, delta dau chi) -> du doan trong cung chuoi chi nhat quan. Sieu canh 10 part / 5 body bam sat H36M cua
#   HyperDiff. Layer scale 1e-5 -> luc dau = khong co. Ma tran sieu canh khoa theo hgcn_freeze_adj. Mac dinh TAT.
cfg.MODEL.smpl_hgcn = False
cfg.MODEL.smpl_hgcn_layers = 3
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
# EMA trong so (mac dinh TAT = 0). Bat (vd 0.999) -> eval/luu bang trong so EMA, on dinh hon cho diffusion.
cfg.DIFFLIFT.ema_decay = 0.0
# Danh gia them trong luc train (khong anh huong chon best, best van theo val_set)
cfg.DIFFLIFT.epoch_eval_sets = []     # danh gia MOI epoch, vd ['3dpw']
cfg.DIFFLIFT.full_eval_sets = []      # danh gia moi full_eval_freq epoch, vd ['Human36M', '3dpw-oc', '3dpw-pc']
cfg.DIFFLIFT.full_eval_freq = 5
# Ti le tron dataset moi epoch (theo thu tu train_list), vd [0.4, 0.4, 0.2]. [] = TAT (dung balance_datasets).
# Bat -> lay mau co hoan lai theo ti le, khong lap 3DPW-train ~14 lan/epoch nhu balance.
cfg.DIFFLIFT.dataset_ratio = []
cfg.DIFFLIFT.samples_per_epoch = 0
# Cach tron cac bo train cua lifter: 'ratio' (mac dinh: dataset_ratio / balance_datasets nhu cu) |
# 'jotr': giong JOTR MultipleDatasets([3D (H36M + MuCo), 2D (COCO + CrowdPose)], make_same_len=True)
#   -> 50% mau tu nhom 3D, 50% nhom 2D, trong nhom ti le theo so mau; lay mau co hoan lai, samples_per_epoch mau / epoch
cfg.DIFFLIFT.mix_mode = 'ratio'    # so mau moi epoch khi dung dataset_ratio (0 = tong so mau cac tap)
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

""" Refiner 2D (2026-11-03): anh + 2D detector -> 2D da sua cua DUNG nguoi + do bat dinh tung khop """
# Chi main/train_refiner2d.py, main/eval_refiner2d.py doc nhom nay; train.py / Teacher / Student khong dung.
# Du lieu, batch, ti le bo, lich lr dung chung cfg.DATASET / cfg.TRAIN (base.get_dataloader); thu muc = --exp_name.
cfg.REFINE2D = edict()
cfg.REFINE2D.backbone_pretrained = 'spin'   # 'spin' | 'imagenet'
# Muc tieu 2D: 'clean' = GT annotation truoc PoseFix (can DATASET.return_clean_2d) | 'mesh' = H36M-17 tinh tu mesh GT
# chieu vao crop (targets orig_joint_img, mask orig_joint_trunc) -> cung dinh nghia khop voi GT 3D benchmark
cfg.REFINE2D.target = 'clean'
# Refiner day: so diem 2D day du doan them sau 17 khop (can DATASET.dense2d_vertex_file, target 'mesh'). 0 = TAT.
cfg.REFINE2D.num_dense = 0
# Khoi tao tu checkpoint Refiner khac (vd ban 17 khop): nap moi tensor cung shape; lop 'final' chep 17 kenh dau,
# kenh diem day khoi tao moi. '' = khong.
cfg.REFINE2D.init_ckpt = ''
cfg.REFINE2D.bb_lr_scale = 0.1             # lr backbone = TRAIN.lr * bb_lr_scale (head dung TRAIN.lr)
cfg.REFINE2D.bb_bn_eval = True             # backbone giu thong ke BN pretrain (giong [A]); head BN train binh thuong
cfg.REFINE2D.heat_sigma = 2.0              # sigma Gauss (o luoi 64) cho heatmap dau vao va phan bo muc tieu
cfg.REFINE2D.deconv_ch = 256               # so kenh 3 lop deconv (8x8 -> 64x64)
cfg.REFINE2D.w_coord = 1.0                 # L1 toa do soft-argmax (don vi o luoi 64)
cfg.REFINE2D.w_heat = 0.1                  # CE phan bo softmax heatmap vs Gauss muc tieu (chuan hoa)
cfg.REFINE2D.w_nll = 0.1                   # Laplace NLL cho log_b (loi detach, dac trung vao head detach)
cfg.REFINE2D.input_drop_prob = 0.1         # bo ngau nhien khop dau vao (mask 0) luc train -> hoc dien khop thieu
cfg.REFINE2D.weight_decay = 1e-4
cfg.REFINE2D.eval_sets = ['3dpw', '3dpw-pc', '3dpw-crowd']   # danh gia nhanh moi epoch (chi log, KHONG chon ckpt)
cfg.REFINE2D.eval_max_samples = 3000       # so mau dau moi tap khi danh gia nhanh (0 = toan bo)
cfg.REFINE2D.far_ratio = 0.2               # khop "xa": loi > far_ratio x kich thuoc nguoi (giong compare_2d_sources)

""" ImgLifter (2026-11-04): diffusion 2D -> 3D CO ANH, dung Refiner 2D dong bang (lib/models/img_lifter.py) """
# Chi main/train_img_lifter.py, main/eval_img_lifter.py doc nhom nay; train.py / Teacher / Student / DiffLifter cu khong dung.
# Du lieu, batch, lich lr dung chung cfg.DATASET / cfg.TRAIN (base.get_dataloader); thu muc = --exp_name.
cfg.IMGLIFT = edict()
cfg.IMGLIFT.refiner_ckpt = ''          # checkpoint Refiner 2D (target mesh), vd experiment/refiner2d_jotr_mesh/checkpoint/last.pth.tar
cfg.IMGLIFT.embed_dim = 512
cfg.IMGLIFT.depth = 8
cfg.IMGLIFT.num_heads = 8
cfg.IMGLIFT.mlp_ratio = 2.0
cfg.IMGLIFT.drop_path = 0.1
cfg.IMGLIFT.timesteps = 1000
cfg.IMGLIFT.scale = 3.0                # nhan pose 3D (met) truoc khi them nhieu, giong DiffLifter cu
cfg.IMGLIFT.train_xy_noise = 0.5       # nhieu Gauss std = he so x b cong vao xy Refiner luc train (0 = tat)
cfg.IMGLIFT.input_drop_prob = 0.1      # bo ngau nhien khop dau vao Refiner (det = 0) luc train
cfg.IMGLIFT.weight_decay = 0.01
cfg.IMGLIFT.ema_decay = 0.999
cfg.IMGLIFT.eval_sets = ['3dpw', '3dpw-pc', '3dpw-crowd']   # danh gia nhanh moi epoch (chi log)
cfg.IMGLIFT.eval_max_samples = 3000
cfg.IMGLIFT.eval_K = 5                 # so gia thuyet / buoc DDIM khi danh gia nhanh
cfg.IMGLIFT.eval_S = 2
cfg.IMGLIFT.eta = 1.0                  # DDIM eta
cfg.IMGLIFT.jpma_max_b = 3.0           # gop jpma: chi chon theo 2D o khop co b <= nguong (o luoi 64)
# ImgLifter day (2026-11-04): so token NGU CANH 2D lay tu diem day cua Refiner day (sau 17 khop), vd 48.
# Can IMGLIFT.refiner_ckpt la Refiner co num_dense >= num_ctx. Token ngu canh chi mang dieu kien, khong co dau ra 3D.
# 0 = TAT (ImgLifter 17 khop nhu cu).
cfg.IMGLIFT.num_ctx = 0
# Lop Markov (2026-11-05, y tuong DiffusionPose AAAI-26): moi block truyen tin giua 17 token khop theo 16 xuong
# H36M (sau self-attn). markov_gate_b: nhan cong sigmoid(a (log_b - c)) -> khop mo ho nhan nhieu tu khop ke.
# Loss giai phau tren x0: w_bone x L1 do dai xuong (met) + w_angle x (cos goc 2 xuong noi tiep)^2. Tat ca TAT mac dinh.
cfg.IMGLIFT.markov_layer = False
cfg.IMGLIFT.markov_gate_b = False
cfg.IMGLIFT.w_bone = 0.0
cfg.IMGLIFT.w_angle = 0.0
# Ban sua theo DiffusionPose; mac dinh TAT de giu ket qua cua YAML/checkpoint cu.
cfg.IMGLIFT.markov_iters = 0             # 0: duong cu; >0: so vong thong diep co huong, khoi tao bang 1
cfg.IMGLIFT.markov_gate_positive = False # True: a = softplus(a_raw) + 1e-6, cong luon tang theo log_b
cfg.IMGLIFT.markov_msg_norm = False     # True: chuan hoa RMS tin nhan moi vong (can markov_iters > 0)
cfg.IMGLIFT.bone_paper = False           # True: cong thuc (4), var GT tren 16 xuong tung pose; khong don vi
# Fine-tune Refiner (backbone ResNet + head) CUNG LUC khi train ImgLifter (2026-11-12, nguoi dung duyet). Mac dinh TAT.
# Loss 3D cua lifter chay vao backbone CHI qua f8 (xy / log_b detach; feat_j von detach trong Refiner) + giu loss 2D
# cua Refiner (target mesh, 17 khop + diem day, giong train_refiner2d.py) x refiner_loss_w.
# lr head Refiner = TRAIN.lr x refiner_lr_scale; lr backbone = lr head x REFINE2D.bb_lr_scale.
# Moi epoch luu Refiner (EMA) canh checkpoint lifter va ghi duong dan vao ck['refiner_ckpt'] -> eval / Student nap tu dong.
cfg.IMGLIFT.refiner_finetune = False
cfg.IMGLIFT.refiner_lr_scale = 0.5      # TRAIN.lr 2e-4 -> head 1e-4, backbone 1e-5 (bb_lr_scale 0.1)
cfg.IMGLIFT.refiner_loss_w = 1.0
# KD o lifter (2026-11-12, nguoi dung duyet). Mac dinh TAT (kd_teacher_ckpt = '').
# Teacher = checkpoint train_img_lifter train bang GT 2D (Refiner cua no lay tu ck['refiner_ckpt']), DONG BANG.
# Moi batch: teacher nhan GT 2D (targets orig_joint_img, mask orig_joint_trunc x orig_joint_valid), student nhan 2D
# nhieu nhu cu, CUNG buoc t va CUNG nhieu. Loss KD = MSE giua token an 17 khop (lop cuoi, truoc head), ca hai qua
# layer_norm khong tham so, trung binh theo khop va kenh, chi mau is_valid_fit, x kd_w, cong vao loss lifter.
cfg.IMGLIFT.kd_teacher_ckpt = ''
cfg.IMGLIFT.kd_w = 0.05
# InfoNCE trong KD (2026-11-12). kd_nce_w 0 = TAT. Chi mau is_valid_fit. kd_nce_mode:
#   'sample' (GIONG lop SimCLR trong DPMesh/common/nets/loss.py): moi MAU 1 vector = 17 token an noi lai (flatten),
#            NT-Xent DOI XUNG: duong = (student_b, teacher_b); am = 2(n - 1) vector con lai cua ca student lan teacher.
#   'joint'  (ban dau cua Claude): anchor = token student (b, j), duong = teacher (b, j), am = moi token teacher khac.
# kd_nce_tau: lop SimCLR cua DPMesh mac dinh 1.0 nhung cho goi khong co trong code phat hanh -> khong biet gia tri ho dung.
cfg.IMGLIFT.kd_nce_w = 0.0
cfg.IMGLIFT.kd_nce_tau = 0.07
cfg.IMGLIFT.kd_nce_mode = 'sample'
# KD tren f8 cua Refiner (2026-11-12, nguoi dung duyet 23:01) = cho 2D duoc tiem vao backbone (tuong ung NKR cua DPMesh):
# f8 student (2D nhieu) phai giong f8 teacher (GT 2D). KHONG phu thuoc buoc nhieu t; gradient vao backbone Refiner
# -> can refiner_finetune. kd_f8_w x TB theo o 8x8 cua ||chuan hoa L2 theo kenh (f8_s) - (f8_t)||^2 (= 2 - 2 cos, trong [0, 4]).
# kd_nce_feat: InfoNCE tinh tren 'hidden' (token an lifter, co duong tat qua t) | 'f8' (1 vector / mau = f8 flatten,
# NT-Xent doi xung nhu SimCLR cua DPMesh; kd_nce_mode khong ap dung).
cfg.IMGLIFT.kd_f8_w = 0.0
cfg.IMGLIFT.kd_nce_feat = 'hidden'
# Khoi tao lifter (+ Refiner neu refiner_finetune) tu last.pth.tar cua run khac (raw + EMA), KHONG nap optimizer.
# '' = train tu dau (nhu cu). Bo qua khi --resume_training.
cfg.IMGLIFT.init_ckpt = ''

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

