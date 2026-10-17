import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import Counter
import copy
import numpy as np
import pickle
import models
from data_final.dataset import MultipleDatasets
from core.config import cfg
from core.loss import JOTRCoordLoss, JOTRParamLoss, AutomaticWeightedLoss, CoordLoss
from funcs_utils import get_optimizer, load_checkpoint, get_scheduler, count_parameters, lr_check
from utils.jotr_dataset import get_test_dataset as get_jotr_test_dataset
from utils.jotr_dataset import get_train_dataset as get_jotr_train_dataset
from utils.jotr_evaluation import evaluate_3dpw_subset
from utils.transforms import cam2pixel
from models.smpl_hyperdiff import axis_angle_to_rot6d, make_noisy_kp2d

def get_dataloader(args, dataset_names, is_train):
    dataset_split = 'TRAIN' if is_train else 'TEST'
    if is_train:
        batch_per_dataset = cfg[dataset_split].batch_size // len(dataset_names)
    else:
        batch_per_dataset = cfg[dataset_split].batch_size
        
    dataset_list, dataloader_list = [], []

    print(f"==> Preparing {dataset_split} Dataloader...")
    if is_train:
        dataset_list = [get_jotr_train_dataset(name, args) for name in dataset_names]
    else:
        dataset_list = [get_jotr_test_dataset(name, args) for name in dataset_names]
    for name, dataset in zip(dataset_names, dataset_list):
        print("# of {} {} data: {}".format(dataset_split, name, len(dataset)))
        dataloader = DataLoader(dataset,
                                batch_size=batch_per_dataset,
                                shuffle=cfg[dataset_split].shuffle,
                                num_workers=cfg.DATASET.workers,
                                pin_memory=False)
        dataloader_list.append(dataloader)

    if not is_train:
        return dataset_list, dataloader_list
    else:
        trainset_loader = MultipleDatasets(dataset_list, make_same_len=True)
        batch_generator = DataLoader(dataset=trainset_loader, \
                          batch_size=batch_per_dataset * len(dataset_names), \
                          shuffle=cfg[dataset_split].shuffle, \
                          num_workers=cfg.DATASET.workers, pin_memory=False)
        return dataset_list, batch_generator

_RESUME_CKPT = None   # checkpoint dang resume (de Trainer nap AWL / projector / optimizer sau khi tao du nhom)


def _restore_resume_state(trainer, student=False):
    """Nap AWL, projector KD (Student) va optimizer tu checkpoint dang resume, SAU khi Trainer da them du nhom tham so."""
    global _RESUME_CKPT
    ck = _RESUME_CKPT
    _RESUME_CKPT = None   # giai phong bo nho GPU sau khi nap xong
    if ck is None:
        print('===> [resume] khong co checkpoint de nap AWL/optimizer')
        return
    if 'awl_state_dict' in ck:
        trainer.awl.load_state_dict(ck['awl_state_dict'])
        print('===> [resume] da nap AWL')
    if student and ck.get('projector_state_dict'):
        psd = ck['projector_state_dict']
        keys = [k for k in ('proj', 'joint', 'global', 'res') if f'{k}.1.weight' in psd]   # giu dung thu tu luc tao
        mods = {}
        for k in keys:
            C = psd[f'{k}.1.weight'].shape[0]
            mods[k] = torch.nn.Sequential(torch.nn.LayerNorm(C, elementwise_affine=False), torch.nn.Linear(C, C))
        trainer.feat_projector = torch.nn.ModuleDict(mods).cuda()
        trainer.feat_projector.load_state_dict(psd)
        trainer.optimizer.add_param_group({'params': trainer.feat_projector.parameters(),
                                           'lr': trainer.optimizer.param_groups[0]['lr'], 'weight_decay': 0})
        print(f'===> [resume] da nap projector KD: {keys}')
    try:
        trainer.optimizer.load_state_dict(ck['optim_state_dict'])
        for state in trainer.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        print(f'===> [resume] da nap optimizer ({len(trainer.optimizer.param_groups)} nhom), '
              f'lr = {trainer.optimizer.param_groups[0]["lr"]:.2e}')
    except (ValueError, KeyError) as e:
        print(f'===> [resume] CANH BAO: khong nap duoc optimizer ({str(e)[:100]}) -> dung optimizer moi')


def prepare_network(args, load_dir='', is_train=True):
    dataset_names = cfg.DATASET.train_list if is_train else cfg.DATASET.test_list
    dataset_list, dataloader = get_dataloader(args, dataset_names, is_train)
    model, criterion, optimizer, lr_scheduler = None, None, None, None
    loss_history, test_error_history = [], {'surface': [], 'joint': []}

    main_dataset = dataset_list[0]
    J_regressor = eval(f'torch.Tensor(main_dataset.joint_regressor_{cfg.DATASET.input_joint_set})')
    if is_train or load_dir:
        print(f"==> Preparing {cfg.MODEL.name} MODEL...")
        if cfg.MODEL.name in ['ARTS', 'teacher', 'student']:
            model = models.ARTS.get_model(num_joint=17, embed_dim=cfg.MODEL.hpe_dim, depth=cfg.MODEL.hpe_dep)
        elif cfg.MODEL.name == 'PoseEst':
            model = models.PoseEstimation.get_model(num_joint=main_dataset.joint_num, embed_dim=cfg.MODEL.hpe_dim, depth=cfg.MODEL.hpe_dep, pretrained=False)
        print('# of model parameters: {}'.format(count_parameters(model)))
        model = model.cuda()

    if is_train:
        criterion = None
        optimizer = get_optimizer(model=model)
        lr_scheduler = get_scheduler(optimizer=optimizer)

    if load_dir and (not is_train or args.resume_training):
        print('==> Loading checkpoint')
        if cfg.MODEL.name in ['teacher', 'student', 'ARTS']:
            model = load_model_weights(model, load_dir)
            checkpoint = torch.load(load_dir, map_location='cuda', pickle_module=_PickleShim, weights_only=False)
        else:
            checkpoint = load_checkpoint(load_dir=load_dir, pick_best=(cfg.MODEL.name == 'PoseEst'))
            model.load_state_dict(checkpoint['model_state_dict'])

        if is_train:
            # Teacher/Student them nhom tham so (AWL, projector KD) vao optimizer SAU khi tao -> o day so nhom
            # chua khop -> hoan nap optimizer, Trainer se nap lai sau khi tao du nhom (xem _restore_resume_state).
            global _RESUME_CKPT
            _RESUME_CKPT = checkpoint
            try:
                optimizer.load_state_dict(checkpoint['optim_state_dict'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            except (ValueError, KeyError) as e:
                print(f'===> Hoan nap optimizer (se nap sau khi them AWL/projector): {str(e)[:80]}')
            curr_lr = 0.0

            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']

            lr_state = checkpoint['scheduler_state_dict']
            # update lr_scheduler
            lr_state['milestones'], lr_state['gamma'] = Counter(cfg.TRAIN.lr_step), cfg.TRAIN.lr_factor
            lr_scheduler.load_state_dict(lr_state)

            loss_history = checkpoint['train_log']
            test_error_history = checkpoint['test_log']
            # AWL state will be loaded in Trainer.__init__ after AWL is created
            cfg.TRAIN.begin_epoch = checkpoint['epoch'] + 1
            print('===> resume from epoch {:d}, current lr: {:.0e}, milestones: {}, lr factor: {:.0e}'
                  .format(cfg.TRAIN.begin_epoch, curr_lr, lr_state['milestones'], lr_state['gamma']))

    return dataloader, dataset_list, model, criterion, optimizer, lr_scheduler, loss_history, test_error_history


class Trainer:
    def __init__(self, args, load_dir):
        self.batch_generator, self.dataset_list, self.model, self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history\
            = prepare_network(args, load_dir=load_dir, is_train=True)

        self.main_dataset = self.dataset_list[0]
        self.print_freq = cfg.TRAIN.print_freq

        self.J_regressor = eval(f'torch.Tensor(self.main_dataset.joint_regressor_{cfg.DATASET.target_joint_set}).cuda()')

        # The dataset wrapper already emits H36M-17 GT and 2D inputs. The model's
        # auxiliary outputs (joint_proj / joint_cam) come from get_coord in
        # SMPL-30 joint order, so build a reduction index from the underlying
        # SMPL joint set to map those outputs to H36M-17.
        h36m_joints = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        smpl30_joints = self.main_dataset.mesh_model.joints_name
        self.h36m_from_smpl30 = [smpl30_joints.index(name) for name in h36m_joints]

        self.model = torch.nn.DataParallel(self.model).cuda()

        self.jotr_coord_loss = JOTRCoordLoss()
        self.jotr_param_loss = JOTRParamLoss()
        self.coordLoss = CoordLoss(has_valid=True)

        # KD loss: Cosine + MSE
        # self.kd_weight = getattr(cfg.MODEL, 'kd_weight', 0.5)

        # 5 losses now (including mesh_loss)
        self.awl = AutomaticWeightedLoss(5).cuda()
        self.optimizer.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})
        # Resume: nap AWL + optimizer (sau khi da them nhom AWL vao optimizer)
        if getattr(args, 'resume_training', False) and load_dir:
            _restore_resume_state(self)

        if cfg.TRAIN.wandb:
            wandb.init(config=cfg,
                   project=cfg.MODEL.name,
                   name='ARTS/' + cfg.output_dir.split('/')[-1],
                   dir=cfg.output_dir,
                   job_type="training",
                   reinit=True)

    def train(self, epoch):
        self.model.train()
        use_diffusion = getattr(cfg.MODEL, 'REFINER', 'hypergcn') == 'diffusion'

        # Noise curriculum warmup
        warmup_ep = getattr(cfg.DIFF.NOISE, 'warmup_epochs', 0)
        if warmup_ep > 0 and epoch <= warmup_ep:
            warmup_scale = cfg.DIFF.NOISE.warmup_scale_start + \
                (1.0 - cfg.DIFF.NOISE.warmup_scale_start) * (epoch / warmup_ep)
        else:
            warmup_scale = 1.0

        lr_check(self.optimizer, epoch)
        running_loss = 0.0
        running_mpjpe = 0.0
        mpjpe_count = 0
        batch_generator = tqdm(self.batch_generator)
        for i, (inputs, targets, meta) in enumerate(batch_generator):
            # convert to cuda
            input_image = inputs['img'].cuda().float()
            input_pose = inputs['joints'].cuda().float()
            gt_orig_joint_cam = targets['orig_joint_cam'].cuda()
            gt_fit_joint_cam = targets['fit_joint_cam'].cuda()
            orig_joint_valid = meta['orig_joint_valid'].cuda()
            fit_joint_trunc = meta['fit_joint_trunc'].cuda()

            gt_smplpose = targets['pose_param'].cuda()
            gt_smplshape = targets['shape_param'].cuda()
            gt_mesh_cam = targets['smpl_mesh_cam'].cuda()
            is_3d = meta['is_3D'].cuda()
            is_valid_fit = meta['is_valid_fit'].cuda()

            # ---- Prepare diffusion inputs ----
            gt_pose_6d = None
            kp2d = None
            kp_conf = None
            pose_valid_mask = None

            if use_diffusion:
                # Convert GT axis-angle (B,72) -> 6D (B,24,6)
                B = gt_smplpose.shape[0]
                gt_pose_6d = axis_angle_to_rot6d(
                    gt_smplpose.reshape(-1, 3)
                ).reshape(B, 24, 6)

                # GT 2D keypoints: heatmap [0,64) -> normalise to [-1,1]
                kp2d_gt = targets['orig_joint_img'].cuda()[:, :, :2].clone()
                kp2d_gt[:, :, 0] = kp2d_gt[:, :, 0] / cfg.output_hm_shape[2] * 2 - 1
                kp2d_gt[:, :, 1] = kp2d_gt[:, :, 1] / cfg.output_hm_shape[1] * 2 - 1
                kp_conf_gt = meta['orig_joint_trunc'].cuda().squeeze(-1).float()

                # Apply noise augmentation to condition
                kp2d, kp_conf, _ = make_noisy_kp2d(
                    kp2d_gt, kp_conf_gt,
                    noise_cfg=cfg.DIFF.NOISE,
                    warmup_scale=warmup_scale,
                )

                # Joint validity mask for loss: fit_param_valid (B,72) -> (B,24)
                fit_pv = meta['fit_param_valid'].cuda() * is_valid_fit[:, None]
                pose_valid_mask = fit_pv.reshape(B, 24, 3)[:, :, 0]

            # # ---- Forward (pass GT 3D joints for teacher KD) ----
            # gt_pose_input = gt_fit_joint_cam - gt_fit_joint_cam[:, 0:1, :]
            
            model_output = self.model(
                input_image, input_pose, is_train=True, use_gt_3d=False,
                gt_pose_6d=gt_pose_6d, kp2d=kp2d, kp_conf=kp_conf,
                pose_valid_mask=pose_valid_mask,
                # gt_joints_3d=gt_pose_input,  # GT joints for teacher KD
            )

            pred_mesh = model_output['smpl_mesh_cam']
            pred_smplpose = model_output['smpl_pose']
            pred_smplshape = model_output['smpl_shape']
            cam_param = model_output['cam_param']
            
            pose3d_pred = model_output.get('joint_img', None)
            if pose3d_pred is not None:
                mpjpe_error = torch.mean(torch.sqrt(torch.sum((pose3d_pred - gt_fit_joint_cam) ** 2, dim=-1))) * 1000
                running_mpjpe += float(mpjpe_error.detach().item())
                mpjpe_count += 1

            
            pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh)
            pred_pose = pred_pose - pred_pose[:, 0:1, :]

            # ---- Compute individual losses ----
            loss_smpl_joint_cam = self.jotr_coord_loss(
                pred_pose, gt_fit_joint_cam,
                fit_joint_trunc * is_valid_fit[:, None, None]
            ).mean()

            # 2D projection loss (cam training signal)
            # Use smpl_mesh_cam_proj which optionally detaches pose but keeps shape attached
            cam = model_output['cam_param']
            scale = cam[:, 0:1, None]
            trans = cam[:, 1:3, None].transpose(1, 2)
            pred_pose_17 = torch.matmul(self.J_regressor[None, :, :], model_output['smpl_mesh_cam_proj'])
            proj_2d = scale * pred_pose_17[:, :, :2] + trans
            proj_pixel = (proj_2d + 1.0) * 0.5 * cfg.input_img_shape[0]
            pred_joint_proj = proj_pixel * (cfg.output_hm_shape[1] / cfg.input_img_shape[0])
            
            loss_body_joint_proj = self.jotr_coord_loss(
                pred_joint_proj,
                targets['orig_joint_img'].cuda()[:, :, :2],
                meta['orig_joint_trunc'].cuda()
            ).mean()

            fit_pose_valid = meta['fit_param_valid'].cuda() * is_valid_fit[:, None]
            fit_shape_valid = is_valid_fit[:, None]

            
            smpl_pose_loss = self.jotr_param_loss(pred_smplpose, gt_smplpose, fit_pose_valid).mean()
            smpl_shape_loss = self.jotr_param_loss(pred_smplshape, gt_smplshape, fit_shape_valid).mean()
            mesh_loss = self.coordLoss(pred_mesh, gt_mesh_cam, is_valid_fit[:, None, None])

            # ---- Combine losses ----
            if use_diffusion:
                diff_loss = model_output['diff_loss']
                loss = cfg.LOSS.diff_w * diff_loss
                loss = loss + cfg.LOSS.w_proj  * loss_body_joint_proj
                loss = loss + cfg.LOSS.w_shape * smpl_shape_loss
                if cfg.LOSS.w_joint_cam > 0:
                    loss = loss + cfg.LOSS.w_joint_cam * loss_smpl_joint_cam
                if cfg.LOSS.w_mesh > 0:
                    loss = loss + cfg.LOSS.w_mesh * mesh_loss
                if cfg.LOSS.w_pose > 0:
                    loss = loss + cfg.LOSS.w_pose * smpl_pose_loss
            else:
                loss_dict = {
                    'smpl_joint_cam': loss_smpl_joint_cam,
                    'smpl_pose': smpl_pose_loss,
                    'smpl_shape': smpl_shape_loss,
                    'body_joint_proj': loss_body_joint_proj,
                    'mesh_loss': mesh_loss,
                }
                loss_dict = self.awl(loss_dict)
                loss = sum(loss_dict.values())
                diff_loss = torch.zeros(1, device=input_image.device)

            # # ---- KD Loss: L_NKR = L_Cosine + L_MSE ----
            # kd_loss = torch.zeros(1, device=input_image.device).squeeze()
            # if 's_feat_joint' in model_output and 't_feat_joint' in model_output:
            #     s_feat_joint = model_output['s_feat_joint']
            #     t_feat_joint = model_output['t_feat_joint'].detach()
            #     s_feat_img = model_output['s_feat_img']
            #     t_feat_img = model_output['t_feat_img'].detach()

            #     # MSE component
            #     mse_joint = F.mse_loss(s_feat_joint, t_feat_joint)
            #     mse_img = F.mse_loss(s_feat_img, t_feat_img)

            #     # Cosine component
            #     cos_joint = F.cosine_similarity(s_feat_joint.flatten(1), t_feat_joint.flatten(1), dim=-1)
            #     cos_img = F.cosine_similarity(s_feat_img.flatten(1), t_feat_img.flatten(1), dim=-1)
                
            #     cosine_loss_joint = (1.0 - cos_joint).mean()
            #     cosine_loss_img = (1.0 - cos_img).mean()

            #     kd_loss = (mse_joint + mse_img) + 0.1 * (cosine_loss_joint + cosine_loss_img)

            # loss = loss + self.kd_weight * kd_loss

            # update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log
            running_loss += float(loss.detach().item())
            if cfg.TRAIN.wandb:
                log_dict = {
                    'train_loss/smpl_joint_cam': loss_smpl_joint_cam.detach(),
                    'train_loss/smpl_pose': smpl_pose_loss.detach(),
                    'train_loss/smpl_shape': smpl_shape_loss.detach(),
                    'train_loss/body_joint_proj': loss_body_joint_proj.detach(),
                    'train_loss/mesh': mesh_loss.detach(),
                    # 'train_loss/kd': kd_loss.detach(),
                }
                if use_diffusion:
                    log_dict['train_loss/diff'] = diff_loss.detach()
                wandb.log(log_dict)

            if i % self.print_freq == 0:
                total_loss = loss.detach()
                desc = (f'Epoch{epoch}_({i}/{len(batch_generator)}) => '
                        f'proj2d: {loss_body_joint_proj.item():.3f} '
                        f'smpl3d: {loss_smpl_joint_cam.item():.3f} '
                        f'shape: {smpl_shape_loss.item():.3f} '
                        f'mesh: {mesh_loss.item():.3f} ')
                        # f'kd: {kd_loss.item():.4f} ')
                if use_diffusion:
                    desc += f'diff: {diff_loss.item():.4f} '
                desc += f'tl: {total_loss.item():.3f}'
                batch_generator.set_description(desc)

        self.loss_history.append(running_loss / len(batch_generator))
        print(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')
        if mpjpe_count > 0:
            print(f'Epoch{epoch} Train MPJPE Error (mm): {running_mpjpe / mpjpe_count:.4f}')

class Tester:
    def __init__(self, args, load_dir=''):
        self.val_loaders, self.val_datasets, self.model, _, _, _, _, _ = \
            prepare_network(args, load_dir=load_dir, is_train=False)
        self.print_freq = cfg.TRAIN.print_freq
        if self.model:
            self.model = torch.nn.DataParallel(self.model).cuda()
        self.surface_error = 9999.9
        self.joint_error = 9999.9

    def test(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()
        results = {}
        for dataset_name, dataset, loader in zip(
                cfg.DATASET.test_list, self.val_datasets, self.val_loaders):
            metrics = evaluate_3dpw_subset(self.model, dataset, loader)
            results[dataset_name] = metrics
            print(
                f'{dataset_name}: MPJPE={metrics["mpjpe"]:.2f}, '
                f'PA-MPJPE={metrics["pa_mpjpe"]:.2f}, '
                f'MPVPE={metrics["mpvpe"]:.2f}'
            )

        if results:
            self.joint_error = sum(item['mpjpe'] for item in results.values()) / len(results)
            self.surface_error = sum(item['mpvpe'] for item in results.values()) / len(results)
        return results
class Teacher_Trainer:
    def __init__(self, args, load_dir):
        self.batch_generator, self.dataset_list, self.model, self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history\
            = prepare_network(args, load_dir=load_dir, is_train=True)

        self.main_dataset = self.dataset_list[0]
        self.print_freq = cfg.TRAIN.print_freq

        self.J_regressor = eval(f'torch.Tensor(self.main_dataset.joint_regressor_{cfg.DATASET.target_joint_set}).cuda()')

        # The dataset wrapper already emits H36M-17 GT and 2D inputs.
        # h36m_from_smpl30 dung de lay 17 khop H36M tu 30 khop joint_proj cua model.
        h36m_joints = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        smpl30_joints = self.main_dataset.mesh_model.joints_name
        self.h36m_from_smpl30 = [smpl30_joints.index(name) for name in h36m_joints]

        # ---- Goi y nen (T-hint, PLAN_TEACHER_HINT.md): khoi tao tu Teacher GT (tru bo ma hoa joint + nhanh goi y) ----
        self.teacher_hint = bool(cfg.MODEL.get('teacher_hint', False))
        if self.teacher_hint:
            init_from = cfg.MODEL.get('hint_init_from', '')
            if init_from and not getattr(args, 'resume_training', False):
                load_partial_weights(self.model, init_from, tag='T-hint <- Teacher GT')
            print(f"===> Teacher T-hint: k={cfg.MODEL.get('hint_dim', 16)}, noise={cfg.MODEL.get('hint_noise', 0.1)}, "
                  f"off_prob={cfg.MODEL.get('hint_off_prob', 0.3)} | joint vao = MotionBERT (giong Student)")

        self.model = torch.nn.DataParallel(self.model).cuda()

        self.jotr_coord_loss = JOTRCoordLoss()
        self.jotr_param_loss = JOTRParamLoss()
        self.coordLoss = CoordLoss(has_valid=True)
        self.awl = AutomaticWeightedLoss(5).cuda()  # 5 losses: joint_cam, pose, shape, mesh, body_joint_proj
        self.optimizer.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})

        # ---- Ep Teacher phai dung anh: che/nhieu mot phan khop GT luc train ----
        # (xem content_ablation_test.py: Teacher goc content-blind vi GT sach khong
        #  bao gio can anh de giam loss; mesh_loss/smpl_joint_cam/smpl_pose van dung
        #  pass KHONG detach nen van ep duoc nhanh pose phai dung anh khi khop bi che).
        self.joint_corrupt_prob = cfg.MODEL.get('teacher_joint_corrupt_prob', 0.0)
        self.joint_corrupt_mode = cfg.MODEL.get('teacher_joint_corrupt_mode', 'zero')  # 'zero' | 'noise'
        self.joint_corrupt_noise_std = cfg.MODEL.get('teacher_joint_corrupt_noise_std', 0.15)
        self.joint_corrupt_warmup_epochs = cfg.MODEL.get('teacher_joint_corrupt_warmup_epochs', 0)
        if self.joint_corrupt_prob > 0:
            print(f'===> Teacher joint-corruption BAT: prob={self.joint_corrupt_prob}, '
                  f'mode={self.joint_corrupt_mode}, warmup={self.joint_corrupt_warmup_epochs} epoch')

        # ---- (A) tron GT voi joint MotionBERT lift, (C) head tham do anh ----
        self.lift_alpha_max = float(cfg.MODEL.get('teacher_lift_alpha_max', 0.0))
        self.lift_clean_prob = float(cfg.MODEL.get('teacher_lift_clean_prob', 0.0))
        # T-hint dung kien truc Student (khong co head tham do anh) -> tat img_probe
        self.img_probe = bool(cfg.MODEL.get('img_probe', False)) and not bool(cfg.MODEL.get('teacher_hint', False))
        self.img_probe_w = float(cfg.MODEL.get('img_probe_w', 0.1))
        if self.lift_alpha_max > 0:
            print(f'===> Teacher (A) tron GT voi MotionBERT lift: alpha ~ U(0, {self.lift_alpha_max}), '
                  f'ti le mau GT sach (alpha = 0) = {self.lift_clean_prob}')
        if self.img_probe:
            print(f'===> Teacher (C) head tham do anh BAT: weight={self.img_probe_w}')

        # ---- Loss tham so shape (beta) ----
        # Beta GT cua 3DPW thuoc SMPL theo GIOI TINH, con model dung SMPL NEUTRAL -> cung beta cho ra co the
        # KHAC (vposer_floor_test: neutral + beta GT = 37.7 mm, te hon beta = 0 = 27.3 mm). Loss nay co the
        # day sai kich thuoc co the. 1.0 = nhu cu; 0 = bo han (shape chi hoc qua loss joint/mesh).
        self.shape_loss_w = float(cfg.MODEL.get('shape_param_loss_w', 1.0))
        if self.shape_loss_w != 1.0:
            print(f'===> Teacher loss smpl_shape: trong so = {self.shape_loss_w}'
                  + (' (TAT - khong dua vao AWL)' if self.shape_loss_w <= 0 else ''))

        # Resume: nap AWL + optimizer (sau khi da them nhom AWL vao optimizer)
        if getattr(args, 'resume_training', False) and load_dir:
            _restore_resume_state(self)

        if cfg.TRAIN.wandb:
            wandb.init(config=cfg,
                   project=cfg.MODEL.name,
                   name='ARTS/' + cfg.output_dir.split('/')[-1],
                   dir=cfg.output_dir,
                   job_type="training",
                   reinit=True)

    def _corrupt_joints(self, joints3d, epoch):
        """joints3d: (B,17,3) GT root-relative, mon vi met. Tra ve BAN SAO da bi che/nhieu
        mot phan khop (joint 0 = root luon giu nguyen vi da = 0 mot cach tam thuong).
        CHI dung lam INPUT cho model — moi loss van so voi joints3d GOC (sach), nen
        model buoc phai dung anh de doan dung lai cac khop bi che.
        """
        if self.joint_corrupt_prob <= 0:
            return joints3d
        B, J, _ = joints3d.shape
        prob = self.joint_corrupt_prob
        if self.joint_corrupt_warmup_epochs > 0:
            prob = prob * min(1.0, float(epoch) / self.joint_corrupt_warmup_epochs)
        if prob <= 0:
            return joints3d
        mask = (torch.rand(B, J, 1, device=joints3d.device) < prob).float()
        mask[:, 0, :] = 0.0  # khong che root (= 0 mot cach tam thuong, che cung vo nghia)
        corrupted = joints3d.clone()
        if self.joint_corrupt_mode == 'zero':
            corrupted = corrupted * (1.0 - mask)
        else:  # 'noise': cong nhieu Gauss manh vao dung cac khop bi chon
            noise = torch.randn_like(corrupted) * self.joint_corrupt_noise_std
            corrupted = corrupted + noise * mask
        return corrupted

    def train(self, epoch):
        self.model.train()

        lr_check(self.optimizer, epoch)
        running_loss = 0.0
        run_extra = {}   # loss tham do anh / MPJPE tham do / sai so input (A), tich luy theo epoch
        batch_generator = tqdm(self.batch_generator)
        for i, (inputs, targets, meta) in enumerate(batch_generator):
            # convert to cuda
            input_image = inputs['img'].cuda().float()
            gt_orig_joint_cam = targets['orig_joint_cam'].cuda() # Đây là tọa độ 3D thực tế đo được từ các cảm biến
            gt_fit_joint_cam = targets['fit_joint_cam'].cuda() # tọa độ 3D sinh ra từ smpl prj
            gt_mesh_cam = targets['smpl_mesh_cam'].cuda()
            orig_joint_valid = meta['orig_joint_valid'].cuda() # mask, = 0 thì k tính loss
            fit_joint_trunc = meta['fit_joint_trunc'].cuda() # mask

            gt_smplpose = targets['pose_param'].cuda()
            gt_smplshape = targets['shape_param'].cuda()
            is_3d = meta['is_3D'].cuda()
            is_valid_fit = meta['is_valid_fit'].cuda()

            # Teacher dùng GT 3D gốc (mét, root-relative) từ orig_joint_cam, không chuẩn hóa
            teacher_gt_pose3d = gt_orig_joint_cam - gt_orig_joint_cam[:, 0:1, :]
            # INPUT dua vao model co the bi che/nhieu mot phan khop (xem _corrupt_joints);
            # moi loss ben duoi van dung GT SACH (gt_fit_joint_cam / gt_smplpose / gt_mesh_cam...)
            # nen model buoc phai doc anh de bu lai phan khop bi che.
            teacher_model_input = self._corrupt_joints(teacher_gt_pose3d, epoch)
            fwd_kwargs = {}
            if cfg.MODEL.get('joint_pe2d', False):
                # Teacher GT + PE2D: PE vi tri khop lay tu 2D Halpe (nhieu), cung nguon voi Student
                fwd_kwargs = dict(pose_2d=inputs['joints'].cuda().float(),
                                  joints_mask=inputs['joints_mask'].cuda().float())
            if self.lift_alpha_max > 0:
                # (A) input = (1-a)*GT + a*MotionBERT_lift, a ~ U(0, alpha_max) rieng tung mau
                lift_alpha = torch.rand(input_image.shape[0], device=input_image.device) * self.lift_alpha_max
                if self.lift_clean_prob > 0:
                    # Mot phan mau nhan GT SACH (a = 0) de Teacher giu duoc do chinh xac khi dau vao sach
                    clean = torch.rand(input_image.shape[0], device=input_image.device) < self.lift_clean_prob
                    lift_alpha = torch.where(clean, torch.zeros_like(lift_alpha), lift_alpha)
                fwd_kwargs = dict(pose_2d=inputs['joints'].cuda().float(),
                                  joints_mask=inputs['joints_mask'].cuda().float(),
                                  lift_alpha=lift_alpha)
            if self.teacher_hint:
                print("ĐANG SỬ DỤNG HINT CHO TEACHER")
                # T-hint: joint vao = MotionBERT(2D) nhu Student, GT (sach) chi di vao bo nen goi y
                teacher_model_input = teacher_gt_pose3d
                fwd_kwargs = dict(pose_2d=inputs['joints'].cuda().float(),
                                  joints_mask=inputs['joints_mask'].cuda().float())
            model_output = self.model(input_image, teacher_model_input, is_train=True, **fwd_kwargs)

            pred_mesh = model_output['smpl_mesh_cam']
            pred_smplpose = model_output['smpl_pose']
            pred_smplshape = model_output['smpl_shape']

            # Regress H36M joints from the predicted SMPL mesh.
            pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh)
            pred_pose = pred_pose - pred_pose[:, 0:1, :]
            loss_smpl_joint_cam = self.jotr_coord_loss(
                pred_pose,
                gt_fit_joint_cam,
                fit_joint_trunc * is_valid_fit[:, None, None]
            ).mean()
            fit_pose_valid = meta['fit_param_valid'].cuda() * is_valid_fit[:, None]
            fit_shape_valid = is_valid_fit[:, None]
            smpl_pose_loss = self.jotr_param_loss(pred_smplpose, gt_smplpose, fit_pose_valid).mean()
            smpl_shape_loss = self.jotr_param_loss(pred_smplshape, gt_smplshape, fit_shape_valid).mean()
            mesh_loss = self.coordLoss(pred_mesh, gt_mesh_cam, is_valid_fit[:, None, None])

            # 2D projection loss: dung phep chieu phoi canh cua chinh model (focal/princpt trong cfg.DATASET),
            # toa do he heatmap. joint_proj_det da detach pose neu cfg.LOSS.DETACH_POSE_FOR_PROJ bat.
            gt_orig_joint_img = targets['orig_joint_img'].cuda()
            orig_joint_trunc = meta['orig_joint_trunc'].cuda()
            pred_joint_proj = model_output['joint_proj_det'][:, self.h36m_from_smpl30, :2]
            loss_body_joint_proj = self.jotr_coord_loss(
                pred_joint_proj,
                gt_orig_joint_img[:, :, :2],
                orig_joint_trunc
            ).mean()

            loss_dict = {
                'smpl_joint_cam': loss_smpl_joint_cam,
                'smpl_pose': smpl_pose_loss,
                'smpl_shape': smpl_shape_loss,
                'mesh_loss': mesh_loss,
                'body_joint_proj': loss_body_joint_proj,
            }
            if self.shape_loss_w <= 0:
                # Bo han khoi AWL (neu giu voi gia tri 0, AWL day trong so cua no ve vo cung -> de NaN)
                loss_dict.pop('smpl_shape')
            elif self.shape_loss_w != 1.0:
                loss_dict['smpl_shape'] = loss_dict['smpl_shape'] * self.shape_loss_w
            loss_dict = self.awl(loss_dict)
            loss = sum(loss_dict.values())

            # ---- (C) loss head tham do anh + (A) log sai so cua input thuc su dua vao Teacher ----
            extra_stats = {}
            valid_joint = fit_joint_trunc
            if valid_joint.dim() == 3:
                valid_joint = valid_joint.squeeze(-1)
            valid_joint = valid_joint * is_valid_fit[:, None]
            n_valid_tot = valid_joint.sum().clamp_min(1.0)
            if self.img_probe:
                probe_pred = model_output['img_probe_joints']
                probe_err = F.smooth_l1_loss(probe_pred, teacher_gt_pose3d, reduction='none', beta=0.05).mean(-1)  # (B, J)
                probe_loss = (probe_err * valid_joint).sum() / n_valid_tot
                loss = loss + self.img_probe_w * probe_loss
                with torch.no_grad():
                    extra_stats['probe_loss'] = probe_loss.item()
                    extra_stats['probe_mpjpe_mm'] = (((probe_pred - teacher_gt_pose3d).norm(dim=-1) * 1000.0)
                                                     * valid_joint).sum().item() / n_valid_tot.item()
            if self.lift_alpha_max > 0 or self.teacher_hint:
                with torch.no_grad():
                    in_err = (model_output['teacher_input_joints'] - teacher_gt_pose3d).norm(dim=-1) * 1000.0
                    extra_stats['input_err_mm'] = (in_err * valid_joint).sum().item() / n_valid_tot.item()
            if self.teacher_hint:
                with torch.no_grad():
                    extra_stats['hint_on'] = model_output['hint_on'].float().mean().item()     # ~ 1 - off_prob
                    on = model_output['hint_on'].float()
                    z_abs = model_output['hint_z'].abs().mean(-1)                           # (B,)
                    extra_stats['z_abs'] = ((z_abs * on).sum() / on.sum().clamp_min(1.0)).item()
            for k_, v_ in extra_stats.items():
                run_extra[k_] = run_extra.get(k_, 0.0) + v_
            if extra_stats and cfg.TRAIN.wandb:
                wandb.log({f'train_loss/{k_}': v_ for k_, v_ in extra_stats.items()})
            if extra_stats and i % self.print_freq == 0:
                batch_generator.set_postfix_str(' '.join(f'{k_}: {v_:.3f}' for k_, v_ in extra_stats.items()))

            # update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # log
            running_loss += float(loss.detach().item())
            if cfg.TRAIN.wandb:
                wandb.log(
                    {
                        'train_loss/smpl_joint_cam': loss_smpl_joint_cam.detach(),
                        'train_loss/smpl_pose': smpl_pose_loss.detach(),
                        'train_loss/smpl_shape': smpl_shape_loss.detach(),
                        'train_loss/mesh': mesh_loss.detach(),
                        'train_loss/body_joint_proj': loss_body_joint_proj.detach(),
                    }
                )

            if i % self.print_freq == 0:
                total_loss = loss.detach()
                batch_generator.set_description(
                    f'Epoch{epoch}_({i}/{len(batch_generator)}) => '
                    f'smpl3d: {loss_smpl_joint_cam.item():.3f} '
                    f'smpl: {(smpl_pose_loss + smpl_shape_loss).item():.3f} '
                    f'mesh: {mesh_loss.item():.3f} '
                    f'proj: {loss_body_joint_proj.item():.3f} '
                    f'tl: {total_loss.item():.3f}'
                )

        self.loss_history.append(running_loss / len(batch_generator))
        print(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')
        if run_extra:
            print('  ' + ' | '.join(f'{k_}: {v_ / len(batch_generator):.4f}' for k_, v_ in run_extra.items()))
        # Trong so hieu dung cua AWL (0.5 / p^2) theo tung loss (giong Student_Trainer)
        if hasattr(self.awl, 'keys'):
            awl_w = {k: (0.5 / (p.detach() ** 2)).item() for k, p in zip(self.awl.keys, self.awl.params)}
            print('  AWL trong so: ' + ' | '.join(f'{k}: {v:.3f}' for k, v in awl_w.items()))
            if cfg.TRAIN.wandb:
                wandb.log({f'awl_w/{k}': v for k, v in awl_w.items()})

def _select_errors(results):
    """(joint_error, surface_error) dung de chon checkpoint best (main/train.py).
    cfg.TEST.select_dataset rong -> trung binh moi test_list (nhu cu); khac rong -> CHI dataset do (vd '3dpw-val')."""
    sel = cfg.TEST.get('select_dataset', '')
    if sel:
        assert sel in results, f"TEST.select_dataset='{sel}' khong nam trong DATASET.test_list {list(results)}"
        chosen = {sel: results[sel]}
    else:
        chosen = results
    joint_error = sum(item['mpjpe'] for item in chosen.values()) / len(chosen)
    surface_error = sum(item['mpvpe'] for item in chosen.values()) / len(chosen)
    print(f'  -> chon best theo: {list(chosen)} | MPJPE {joint_error:.2f} | MPVPE {surface_error:.2f}')
    return joint_error, surface_error


class Teacher_Tester:
    def __init__(self, args, load_dir=''):
        self.val_loaders, self.val_datasets, self.model, _, _, _, _, _ = \
            prepare_network(args, load_dir=load_dir, is_train=False)
        self.print_freq = cfg.TRAIN.print_freq
        if self.model:
            self.model = torch.nn.DataParallel(self.model).cuda()
        self.surface_error = 9999.9
        self.joint_error = 9999.9
        self.teacher_hint = bool(cfg.MODEL.get('teacher_hint', False))
        # T-hint: moi epoch chi danh gia dong CO goi y (dong chon best) tru khi TEST.hint_eval_nohint = True.
        # Test cuoi trong main/train.py dat eval_nohint = True de in ca 2 dong.
        self.eval_nohint = bool(cfg.TEST.get('hint_eval_nohint', False))

    def test(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()
        results = {}
        for dataset_name, dataset, loader in zip(
                cfg.DATASET.test_list, self.val_datasets, self.val_loaders):
            metrics = evaluate_3dpw_subset(self.model, dataset, loader)
            results[dataset_name] = metrics
            print(
                f'{dataset_name}: MPJPE={metrics["mpjpe"]:.2f}, '
                f'PA-MPJPE={metrics["pa_mpjpe"]:.2f}, '
                f'MPVPE={metrics["mpvpe"]:.2f}' + (' [co goi y]' if self.teacher_hint else '')
            )
            if self.teacher_hint and self.eval_nohint:
                # T-hint: do them KHONG goi y (z = 0) = hoan canh Student. Chon best van theo ban CO goi y.
                m0 = evaluate_3dpw_subset(self.model, dataset, loader, teacher_hint_on=False)
                print(
                    f'{dataset_name}: MPJPE={m0["mpjpe"]:.2f}, '
                    f'PA-MPJPE={m0["pa_mpjpe"]:.2f}, '
                    f'MPVPE={m0["mpvpe"]:.2f} [KHONG goi y] | goi y giup: '
                    f'{m0["mpjpe"] - metrics["mpjpe"]:+.2f} mm MPJPE'
                )
                if cfg.TRAIN.wandb and wandb.run is not None:
                    wandb.log({f'test_nohint/{dataset_name}_mpjpe': m0['mpjpe'],
                               f'test_nohint/{dataset_name}_pa_mpjpe': m0['pa_mpjpe'],
                               f'test_nohint/{dataset_name}_mpvpe': m0['mpvpe']})

        if results:
            self.joint_error, self.surface_error = _select_errors(results)
        return results
import pickle

class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # ckpt lưu bằng NumPy 2.x, môi trường đang là NumPy 1.x
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)

class _PickleShim:
    Unpickler = _NumpyCompatUnpickler
    load = pickle.load
    Pickler = pickle.Pickler
    dump = pickle.dump
def load_model_weights(model, ckpt_path, skip_prefixes=('pose_lifter.',)):
    ckpt = torch.load(ckpt_path, map_location='cpu',
                      pickle_module=_PickleShim, weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in ckpt:
                state = ckpt[k]
                break

    # 1) bỏ prefix 'module.' (DataParallel)
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}

    # 2) ckpt cũ lưu module này là 'teacher_model.', code hiện tại là 'smpl_model.'
    old_p, new_p = 'teacher_model.', 'smpl_model.'
    state = {(new_p + k[len(old_p):] if k.startswith(old_p) else k): v
             for k, v in state.items()}

    # 3) lọc: bỏ pose_lifter, bỏ key lệch shape
    own = model.state_dict()
    filtered, skipped, shape_bad = {}, [], []
    for k, v in state.items():
        if k.startswith(skip_prefixes):
            skipped.append(k)
        elif k in own and own[k].shape != v.shape:
            shape_bad.append((k, tuple(v.shape), tuple(own[k].shape)))
        else:
            filtered[k] = v

    # 4) nạp
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    missing = [k for k in missing if not k.startswith(skip_prefixes)]

    # 4b) Chuan hoa anh cua backbone (backbone.input_mean / input_std) chi co trong checkpoint moi.
    #     Checkpoint cu (backbone ngau nhien, KHONG chuan hoa) khong co 2 key nay -> dat ve khong chuan hoa
    #     de giu dung hanh vi luc train, bat ke config hien tai co bat backbone_pretrained hay khong.
    norm_keys = ('backbone.input_mean', 'backbone.input_std')
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'set_input_normalization'):
        if not any(k in state for k in norm_keys):
            model.backbone.set_input_normalization(imagenet=False)
            print('[load_model_weights] checkpoint khong co chuan hoa anh cho backbone -> dung anh [0,1] nhu luc train cu')
        else:
            print(f'[load_model_weights] backbone dung chuan hoa tu checkpoint: '
                  f'mean={model.backbone.input_mean.flatten().tolist()}')
    missing = [k for k in missing if k not in norm_keys]

    print(f'[Teacher] loaded {len(filtered)} tensors, skipped {len(skipped)} {skip_prefixes}')
    print(f'[Teacher] missing={len(missing)}, unexpected={len(unexpected)}, '
          f'shape_mismatch={len(shape_bad)}')
    if shape_bad:
        print('  shape mismatch (5 đầu):', shape_bad[:5])
    if missing:
        print('  MISSING   (10 đầu):', sorted(missing)[:10])
    if unexpected:
        print('  UNEXPECTED(10 đầu):', sorted(unexpected)[:10])

    # 5) dừng ngay nếu teacher nạp chưa đủ
    # (Teacher phai duoc train bang Teacher_Trainer moi; ckpt SPIN cu se thieu HyperGCN/heads -> assert o day)
    assert not shape_bad, f'Có key lệch shape: {shape_bad[:3]}'
    missing_core = [k for k in missing if k.startswith(('smpl_model.', 'backbone.'))]
    hint = ''
    if any('fusion.projector_student.' in k for k in missing_core):
        hint = ' -> Model la STUDENT nhung checkpoint khong co encoder cua Student (co the la checkpoint TEACHER).'
    elif any('fusion.joint_proj.' in k or 'fusion.norm_joint_proj.' in k for k in missing_core):
        hint = ' -> Model la TEACHER nhung checkpoint khong co encoder cua Teacher (co the la checkpoint STUDENT).'
    assert not missing_core, \
        f'Checkpoint {ckpt_path} chưa nạp đủ smpl_model/backbone{hint} Thiếu (5 đầu): {missing_core[:5]}'
    return model


def load_partial_weights(model, ckpt_path, skip_prefixes=('pose_lifter.',), tag='partial'):
    """Nap moi tensor TRUNG TEN va TRUNG SHAPE tu checkpoint (khong assert thieu key).
    Dung de khoi tao T-hint tu Teacher GT: bo ma hoa joint khac ten (joint_proj <-> projector_student),
    nhanh goi y (hint_*) moi -> giu khoi tao ban dau."""
    ckpt = torch.load(ckpt_path, map_location='cpu', pickle_module=_PickleShim, weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in ckpt:
                state = ckpt[k]
                break
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    own = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if not k.startswith(skip_prefixes) and k in own and own[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    missing = [k for k in missing if not k.startswith(skip_prefixes)]
    print(f'===> [{tag}] nap {len(filtered)} tensor tu {ckpt_path} | thieu {len(missing)} | bo qua '
          f'{len(state) - len(filtered)} (khac ten/shape hoac pose_lifter)')
    groups = sorted({'.'.join(k.split('.')[:3]) for k in missing})
    if groups:
        print(f'     thieu (nhom): {groups[:12]}')
    return model


class Student_Trainer:
    def __init__(self, args, load_dir):
        self.batch_generator, self.dataset_list, self.model, self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history\
            = prepare_network(args, load_dir=load_dir, is_train=True)

        self.main_dataset = self.dataset_list[0]
        self.print_freq = cfg.TRAIN.print_freq

        self.J_regressor = eval(f'torch.Tensor(self.main_dataset.joint_regressor_{cfg.DATASET.target_joint_set}).cuda()')
#---------------------------------------------------------------------------------
        # student_use_kd = False -> baseline "from scratch": KHONG dung Teacher (khong KD, khong khoi tao tu Teacher),
        # chi train bang hard loss + privileged loss (privileged so voi GT 3D, khong lien quan Teacher).
        self.use_kd = bool(cfg.MODEL.get('student_use_kd', True))
        self.kd_weight = cfg.MODEL.get('kd_weight', 1.0)
        resume = hasattr(args, 'resume_training') and args.resume_training

        if self.use_kd:
            teacher_ckpt = cfg.MODEL.get('TEACHER', '')
            assert teacher_ckpt, 'Cần đặt cfg.MODEL.TEACHER (checkpoint của Teacher_Trainer)'

            from models.ARTS import ARTS
            old_mode = cfg.MODEL.name
            cfg.MODEL.name = "teacher"
            hpe_dim = cfg.MODEL.get('hpe_dim', 512)
            self.teacher = ARTS(num_joint=self.main_dataset.joint_num, embed_dim=hpe_dim)
            cfg.MODEL.name = old_mode
            load_model_weights(self.teacher, teacher_ckpt)

            # Student va Teacher cung kien truc (chi khac bo ma hoa joint) -> khoi tao Student tu Teacher.
            # Key khac nhau (fusion.joint_proj / norm_joint_proj <-> fusion.projector_student) se bi bo qua boi strict=False.
            if not resume and cfg.MODEL.get('init_student_from_teacher', True):
                missing, unexpected = self.model.smpl_model.load_state_dict(
                    self.teacher.smpl_model.state_dict(), strict=False)
                print(f'===> Student smpl_model khoi tao tu Teacher | '
                      f'missing={len(missing)} (bo ma hoa joint cua student), unexpected={len(unexpected)}')

            self.teacher = torch.nn.DataParallel(self.teacher).cuda()
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
        else:
            self.teacher = None
            print('===> Student FROM SCRATCH: khong Teacher, khong KD, khong khoi tao tu Teacher '
                  '(loss = 0.5*hard + privileged)')

        # ---------------- Che do KD ----------------
        # 'legacy': nhu cu (cosine P/J/G + KD output hyper/pose/root/latent, loss = 0.5*hard + 0.5*kd*adaptive + priv)
        # 'feat'  : KD CHI o feature (E1) + trong so theo do nhieu tung khop (E2) + InfoNCE (E3):
        #           loss = kd_hard_w*hard + kd_lambda*KD_feat + priv
        self.kd_mode = cfg.MODEL.get('kd_mode', 'legacy')
        self.kd_feat_dist = cfg.MODEL.get('kd_feat_dist', 'mse')            # 'mse' (qua projector) | 'cos'
        self.kd_lambda = float(cfg.MODEL.get('kd_lambda', 1.0))
        self.kd_hard_w = float(cfg.MODEL.get('kd_hard_w', 1.0))
        self.kd_noise_weight = bool(cfg.MODEL.get('kd_noise_weight', False))
        self.kd_noise_tau = float(cfg.MODEL.get('kd_noise_tau', 0.05))      # met: loi input 50 mm -> trong so 1
        self.kd_noise_wmin = float(cfg.MODEL.get('kd_noise_wmin', 0.2))
        self.kd_noise_wmax = float(cfg.MODEL.get('kd_noise_wmax', 3.0))
        self.kd_nce_w = float(cfg.MODEL.get('kd_nce_w', 0.0))
        self.kd_nce_temp = float(cfg.MODEL.get('kd_nce_temp', 0.1))
        # Distill phan du (v1): bo ma hoa joint (TRUOC fusion) phai tu xap xi feat_joint SAU fusion cua Teacher;
        # phan con thieu do fusion (anh) bu -> anh bi ep sua loi do joint nhieu. 0 = tat.
        self.kd_res_w = float(cfg.MODEL.get('kd_res_w', 0.0))
        # Distill phan du (v2): KD feature tinh tren 1 luot fusion PHU voi token joint DA DETACH ->
        # gradient KD chi vao fusion (cross-attn toi anh, img_proj, FFN), KHONG vao bo ma hoa joint.
        # Muc tieu Teacher phai dat duoc qua DUONG ANH (v1 lam nguoc lai: joint encoder tu 'lam sach', fusion bot dung anh).
        self.kd_res_mode = str(cfg.MODEL.get('kd_res_mode', 'v1'))
        assert self.kd_res_mode in ('v1', 'v2'), f'kd_res_mode khong hop le: {self.kd_res_mode}'
        # Goi y nen (S-hint, PLAN_TEACHER_HINT.md), kd_mode = 'hint':
        #   loss = kd_hard_w*hard + kd_lambda*(kd_hint_w*MSE(z_hat, z_T) + kd_hint_feat_w*KD_feat_truc_tiep) + priv
        self.teacher_is_hint = bool(cfg.MODEL.get('teacher_hint', False))
        self.student_hint = bool(cfg.MODEL.get('student_hint', False))
        self.kd_hint_w = float(cfg.MODEL.get('kd_hint_w', 1.0))
        self.kd_hint_feat_w = float(cfg.MODEL.get('kd_hint_feat_w', 0.0))
        if self.use_kd and self.kd_mode == 'hint':
            assert self.student_hint and self.teacher_is_hint, \
                "kd_mode 'hint' can student_hint: True va teacher_hint: True (TEACHER phai la checkpoint T-hint)"
            print(f'===> KD mode = hint | k={cfg.MODEL.get("hint_dim", 16)} | kd_hint_w={self.kd_hint_w} | '
                  f'kd_hint_feat_w={self.kd_hint_feat_w} | lambda={self.kd_lambda} | hard_w={self.kd_hard_w}')
        # Nhieu 2D luc train (truoc MotionBERT) - chi Student, chi luc train. 0 = tat.
        self.s2d_prob = float(cfg.MODEL.get('s2d_noise_prob', 0.0))
        self.s2d_jitter = float(cfg.MODEL.get('s2d_noise_jitter', 0.02))
        self.s2d_big_prob = float(cfg.MODEL.get('s2d_noise_big_prob', 0.15))
        self.s2d_big_std = float(cfg.MODEL.get('s2d_noise_big_std', 0.2))
        self.s2d_swap_prob = float(cfg.MODEL.get('s2d_noise_swap_prob', 0.2))
        self.s2d_drop_prob = float(cfg.MODEL.get('s2d_noise_drop_prob', 0.05))
        if self.s2d_prob > 0:
            print(f'===> Nhieu 2D luc train: prob={self.s2d_prob} jitter={self.s2d_jitter} '
                  f'big(p={self.s2d_big_prob}, std={self.s2d_big_std}) swap={self.s2d_swap_prob} drop={self.s2d_drop_prob}')
        if self.use_kd and self.kd_mode == 'feat' and self.kd_res_mode == 'v2':
            print('===> Distill phan du (v2): KD joint/global/InfoNCE tinh voi token joint DETACH '
                  '(gradient KD chi vao fusion / duong anh)')
        if self.use_kd and self.kd_mode == 'feat' and self.kd_res_w > 0:
            print(f'===> Distill phan du (v1): kd_res_w={self.kd_res_w} '
                  f'(feat_joint_in -> feat_joint SAU fusion cua Teacher)')
        self.feat_projector = None   # tao luc batch dau tien (biet so kenh); train.py tu luu 'projector_state_dict'
        if self.use_kd and self.kd_mode == 'feat':
            print(f'===> KD mode = feat | dist={self.kd_feat_dist} | lambda={self.kd_lambda} | hard_w={self.kd_hard_w} | '
                  f'w_proj={cfg.MODEL.get("kd_w_proj", 1.0)} w_joint={cfg.MODEL.get("kd_w_joint", 1.0)} '
                  f'w_global={cfg.MODEL.get("kd_w_global", 1.0)}')
            print(f'     E2 trong so theo nhieu tung khop: {self.kd_noise_weight} '
                  f'(tau={self.kd_noise_tau} m, w in [{self.kd_noise_wmin}, {self.kd_noise_wmax}])')
            print(f'     E3 InfoNCE: w={self.kd_nce_w}, temp={self.kd_nce_temp}')
            print('     KD output (hyper/pose/root/latent) va adaptive weight: TAT')
        h36m_joints = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        smpl30_joints = self.main_dataset.mesh_model.joints_name
        self.h36m_from_smpl30 = [smpl30_joints.index(name) for name in h36m_joints]

        self.model = torch.nn.DataParallel(self.model).cuda()

        self.jotr_coord_loss = JOTRCoordLoss()
        self.jotr_param_loss = JOTRParamLoss()
        self.coordLoss = CoordLoss(has_valid=True)

        self.awl = AutomaticWeightedLoss(5).cuda()
        self.optimizer.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})

        # Loss tham so shape (beta): xem giai thich o Teacher_Trainer. 1.0 = nhu cu; 0 = tat han.
        self.shape_loss_w = float(cfg.MODEL.get('shape_param_loss_w', 1.0))
        if self.shape_loss_w != 1.0:
            print(f'===> Student loss smpl_shape: trong so = {self.shape_loss_w}'
                  + (' (TAT - khong dua vao AWL)' if self.shape_loss_w <= 0 else ''))

        # Resume: nap AWL + projector KD + optimizer (sau khi da them nhom AWL; projector tao lai tu checkpoint)
        if getattr(args, 'resume_training', False) and load_dir:
            _restore_resume_state(self, student=True)

        if cfg.TRAIN.wandb:
            wandb.init(config=cfg,
                   project=cfg.MODEL.name,
                   name='ARTS/' + cfg.output_dir.split('/')[-1],
                   dir=cfg.output_dir,
                   job_type="training",
                   reinit=True)

    # ---------------- Nhieu 2D luc train ----------------
    _LR_PAIRS = ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16))   # H36M-17: phai/trai

    def _corrupt_2d(self, pose2d, mask):
        """pose2d (B,J,2|3) toa do [-1,1]; mask (B,J,1). Tra ban sao bi lam nhieu cho ~s2d_prob mau:
        jitter nho moi khop + lech lon vai khop + doi trai/phai 1 cap + mat khop (mask=0)."""
        B, J = pose2d.shape[:2]
        dev = pose2d.device
        p = pose2d.clone()
        m = mask.clone()
        sel = (torch.rand(B, device=dev) < self.s2d_prob).float()[:, None, None]          # mau bi nhieu
        xy = p[..., :2]
        noise = torch.randn_like(xy) * self.s2d_jitter
        big = (torch.rand(B, J, 1, device=dev) < self.s2d_big_prob).float()
        noise = noise + big * torch.randn_like(xy) * self.s2d_big_std
        xy_new = xy + noise * sel * (m > 0).float()
        # doi trai/phai 1 cap ngau nhien
        swap = (torch.rand(B, device=dev) < self.s2d_swap_prob) & (sel[:, 0, 0] > 0)
        if swap.any():
            pair = torch.randint(len(self._LR_PAIRS), (B,), device=dev)
            for k, (a, b) in enumerate(self._LR_PAIRS):
                idx = swap & (pair == k)
                if idx.any():
                    tmp = xy_new[idx, a].clone()
                    xy_new[idx, a] = xy_new[idx, b]
                    xy_new[idx, b] = tmp
        # mat khop (khong dong root)
        drop = (torch.rand(B, J, 1, device=dev) < self.s2d_drop_prob).float() * sel
        drop[:, 0] = 0
        m = m * (1.0 - drop)
        xy_new = xy_new * (m > 0).float()
        p[..., :2] = xy_new
        if p.shape[-1] >= 3:
            p[..., 2:3] = p[..., 2:3] * (m > 0).float()
        return p, m

    # ---------------- KD feature (kd_mode = 'feat') ----------------
    def _ensure_projectors(self, s_out):
        """Tao projector (LN khong affine -> Linear khoi tao identity) cho moi diem noi, lan dau goi."""
        if self.feat_projector is not None:
            return
        mods = {}
        for key, out_key in (('proj', 'feat_joint_in'), ('joint', 'feat_joint'), ('global', 'feat_global'),
                             ('res', 'feat_joint_in')):
            C = s_out[out_key].shape[-1]
            lin = torch.nn.Linear(C, C)
            with torch.no_grad():
                lin.weight.copy_(torch.eye(C))
                lin.bias.zero_()
            mods[key] = torch.nn.Sequential(torch.nn.LayerNorm(C, elementwise_affine=False), lin)
        self.feat_projector = torch.nn.ModuleDict(mods).cuda()
        self.optimizer.add_param_group({'params': self.feat_projector.parameters(),
                                        'lr': self.optimizer.param_groups[0]['lr'], 'weight_decay': 0})
        print(f'===> Tao feat_projector: ' + ', '.join(f'{k}: {m[1].in_features}' for k, m in mods.items()))

    def _kd_dist(self, key, s, t):
        """Khoang cach Student->Teacher theo kenh cuoi, tra ve shape s.shape[:-1].
        'cos': 1 - cos(LN(s), LN(t)) (nhu cu, de bao hoa ~0).
        'mse': MSE(Linear(LN(s)), LN(t)) -> khong bao hoa, projector hoc them do lon/xoay."""
        t = F.layer_norm(t.detach(), t.shape[-1:])
        if self.kd_feat_dist == 'cos':
            return 1.0 - F.cosine_similarity(F.layer_norm(s, s.shape[-1:]), t, dim=-1)
        return ((self.feat_projector[key](s) - t) ** 2).mean(-1)

    def _info_nce(self, key, s, t, mask=None):
        """InfoNCE Student->Teacher trong batch: cung mau = duong, mau khac = am (DPMesh NKR).
        s, t: (B, C) hoac (B, J, C); mask (B, J) cho dang token."""
        B = s.shape[0]
        if B < 2:
            return s.new_zeros(())
        zs = self.feat_projector[key](s)
        zt = F.layer_norm(t.detach(), t.shape[-1:])
        if mask is not None:
            zs = zs * mask[..., None]
            zt = zt * mask[..., None]
        zs = F.normalize(zs.reshape(B, -1), dim=-1)
        zt = F.normalize(zt.reshape(B, -1), dim=-1)
        logits = zs @ zt.t() / self.kd_nce_temp
        return F.cross_entropy(logits, torch.arange(B, device=s.device))

    def train(self, epoch):
        self.model.train()
        if self.teacher is not None:
            self.teacher.eval()

        lr_check(self.optimizer, epoch)
        # beta của smooth L1 cho privileged loss (giả định đơn vị mét; nếu GT là mm thì tăng lên)
        priv_beta = cfg.MODEL.get('priv_beta', 0.05)
        # KD chỉ ở 3 điểm nối (đặt trọng số 0 để tắt một số hạng):
        #   proj  : đầu ra bộ mã hóa joint (student) <-> bộ mã hóa joint GT (teacher)
        #   joint : token joint sau fusion (đầu vào của HyperGCN / bộ giải mã pose)
        #   global: feature toàn cục sau fusion (đầu vào của cam_head / nhánh shape)
        w_proj = cfg.MODEL.get('kd_w_proj', 1.0)
        w_joint = cfg.MODEL.get('kd_w_joint', 1.0)
        w_global = cfg.MODEL.get('kd_w_global', 1.0)
        # Trừ mean theo kênh (LayerNorm không affine) để cosine không bị thành phần trung bình chung làm gần 1 một cách tầm thường
        _center = lambda x: F.layer_norm(x, x.shape[-1:])
        _cos_dist = lambda s, t: 1.0 - F.cosine_similarity(_center(s), _center(t.detach()), dim=-1)

        running_loss = 0.0
        running = {}   # tích lũy các số hạng để in / log theo epoch
        batch_generator = tqdm(self.batch_generator)
        for i, (inputs, targets, meta) in enumerate(batch_generator):
            # convert to cuda
            input_image = inputs['img'].cuda().float()
            input_pose2d = inputs['joints'].cuda().float()
            joints_mask = inputs['joints_mask'].cuda().float()
            if self.s2d_prob > 0:
                # Nhieu 2D TRUOC MotionBERT (MotionBERT da thuoc long 3DPW-train -> joint gan dung neu khong nhieu)
                input_pose2d, joints_mask = self._corrupt_2d(input_pose2d, joints_mask)
            gt_orig_joint_cam = targets['orig_joint_cam'].cuda()
            gt_fit_joint_cam = targets['fit_joint_cam'].cuda()
            fit_joint_trunc = meta['fit_joint_trunc'].cuda()

            gt_smplpose = targets['pose_param'].cuda()
            gt_smplshape = targets['shape_param'].cuda()
            gt_mesh_cam = targets['smpl_mesh_cam'].cuda()
            is_valid_fit = meta['is_valid_fit'].cuda()

            gt_pose_input = gt_orig_joint_cam - gt_orig_joint_cam[:, 0:1, :]

            # Feed 2D pose to model (which routes to MotionBERT / PMCE in Student mode)
            input_pose2d_raw = inputs['joints_raw'].cuda().float() if 'joints_raw' in inputs else None
            model_output = self.model(
                input_image, input_pose2d, is_train=True,
                gt_joints_3d=gt_pose_input, joints_mask=joints_mask, pose_2d_raw=input_pose2d_raw
            )

            pred_mesh = model_output['smpl_mesh_cam']
            pred_smplpose = model_output['smpl_pose']
            pred_smplshape = model_output['smpl_shape']

            # Regress H36M joints from the predicted SMPL mesh.
            pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh)
            pred_pose_rootrel = pred_pose - pred_pose[:, 0:1, :]

            # ---------- mask khớp hợp lệ ----------
            valid_joint = fit_joint_trunc
            if valid_joint.dim() == 3:
                valid_joint = valid_joint.squeeze(-1)
            valid_joint = valid_joint * is_valid_fit[:, None]
            n_valid = valid_joint.sum(-1).clamp_min(1.0)       # (B,); mẫu không có khớp hợp lệ -> loss = 0

            # ---------- teacher forward (GT 3D làm đầu vào, không grad) ----------
            # Khi khong dung KD: t_out rong -> moi so hang KD = 0 (cac nhanh .get() ben duoi tu bo qua)
            if self.use_kd:
                with torch.no_grad():
                    if self.teacher_is_hint:
                        # T-hint: cung joint MotionBERT voi Student (cung 2D, cung mask), goi y tu GT, eval = khong nhieu
                        t_out = self.teacher(input_image, gt_pose_input, is_train=False,
                                             pose_2d=input_pose2d, joints_mask=joints_mask)
                    else:
                        t_kw = (dict(pose_2d=input_pose2d, joints_mask=joints_mask)
                                if cfg.MODEL.get('joint_pe2d', False) else {})
                        t_out = self.teacher(input_image, gt_pose_input, is_train=False, **t_kw)
            else:
                t_out = {}

            # ---------- KD: 3 điểm nối ----------
            zeros_b = torch.zeros(input_image.shape[0], device=input_image.device)
            feat_mode = self.use_kd and self.kd_mode == 'feat'
            hint_mode = self.use_kd and self.kd_mode == 'hint'
            kd_nce = zeros_b.sum()
            kd_res_ps = zeros_b
            kd_hint_ps = zeros_b
            kd_hfeat_ps = zeros_b
            noise_w_mean = 1.0
            # loi joint dau vao Student (MotionBERT) so voi GT, mm - de kiem tra muc nhieu
            with torch.no_grad():
                _lerr = torch.norm(model_output['lifted_joints_3d'] - gt_pose_input, dim=-1)
                lift_err_mm = float(((_lerr * valid_joint).sum() / valid_joint.sum().clamp_min(1.0)).item() * 1000)
            if feat_mode:
                self._ensure_projectors(model_output)
                # E2: trong so tung khop theo do sai cua joint dau vao Student (MotionBERT) so voi GT.
                #     Khop sai nhieu -> KD manh hon -> Student phai dung anh de keo ve dac trung cua Teacher.
                if self.kd_noise_weight:
                    in_err = torch.norm(model_output['lifted_joints_3d'].detach() - gt_pose_input, dim=-1)   # (B,J) met
                    if self.kd_noise_tau > 0:
                        scale = self.kd_noise_tau                                       # tuyet doi (met)
                    else:
                        # tuong doi: chia cho loi TB cua batch (MotionBERT tren 3DPW-train sai rat it ~9 mm,
                        # nguong tuyet doi se dua gan nhu moi khop ve wmin)
                        scale = ((in_err * valid_joint).sum() / valid_joint.sum().clamp_min(1.0)).clamp_min(1e-4)
                    jw = (in_err / scale).clamp(self.kd_noise_wmin, self.kd_noise_wmax)
                else:
                    jw = torch.ones_like(valid_joint)
                jw_valid = jw * valid_joint
                sw = jw_valid.sum(-1) / n_valid                                                # (B,) trong so mau
                noise_w_mean = float(sw.mean().item())

                kd_proj_ps = (self._kd_dist('proj', model_output['feat_joint_in'], t_out['feat_joint_in'])
                              * jw_valid).sum(-1) / n_valid
                # Nguon feature cho KD: v1 = output fusion binh thuong; v2 = luot fusion phu voi joint DETACH
                kd_src = model_output
                if self.kd_res_mode == 'v2':
                    _net = self.model.module if hasattr(self.model, 'module') else self.model
                    img_o, jnt_o = _net.smpl_model.fusion.cfcer(model_output['feat_img_in'],
                                                               model_output['feat_joint_in'].detach())
                    kd_src = {'feat_joint': jnt_o,
                              'feat_global': torch.cat([img_o.mean(dim=1), jnt_o.mean(dim=1)], dim=1)}
                kd_joint_ps = (self._kd_dist('joint', kd_src['feat_joint'], t_out['feat_joint'])
                               * jw_valid).sum(-1) / n_valid
                kd_global_ps = self._kd_dist('global', kd_src['feat_global'], t_out['feat_global']) * sw
                # Distill phan du (v1): joint-only (truoc fusion) -> feat_joint SAU fusion cua Teacher
                if self.kd_res_w > 0:
                    kd_res_ps = (self._kd_dist('res', model_output['feat_joint_in'], t_out['feat_joint'])
                                 * jw_valid).sum(-1) / n_valid
                # E3: InfoNCE tren token khop sau fusion va feature toan cuc
                if self.kd_nce_w > 0:
                    kd_nce = (self._info_nce('joint', kd_src['feat_joint'], t_out['feat_joint'], mask=valid_joint)
                              + self._info_nce('global', kd_src['feat_global'], t_out['feat_global']))
            elif hint_mode:
                # S-hint: (1) tu doan goi y z_hat ~ z_Teacher; (2) KD feature TRUC TIEP (cung kien truc, cung khong gian
                # -> khong can projector). (2) luon tinh de log J/G, chi vao loss khi kd_hint_feat_w > 0.
                kd_hint_ps = ((model_output['hint_pred'] - t_out['hint_z'].detach()) ** 2).mean(-1)   # (B,)
                _ln = lambda x: F.layer_norm(x, x.shape[-1:])
                kd_joint_ps = (((_ln(model_output['feat_joint']) - _ln(t_out['feat_joint'].detach())) ** 2).mean(-1)
                               * valid_joint).sum(-1) / n_valid
                kd_global_ps = ((_ln(model_output['feat_global']) - _ln(t_out['feat_global'].detach())) ** 2).mean(-1)
                kd_hfeat_ps = kd_joint_ps + kd_global_ps
                kd_proj_ps = zeros_b
            elif self.use_kd:
                kd_proj_ps = (_cos_dist(model_output['feat_joint_in'], t_out['feat_joint_in'])
                              * valid_joint).sum(-1) / n_valid                              # (B,)
                kd_joint_ps = (_cos_dist(model_output['feat_joint'], t_out['feat_joint'])
                               * valid_joint).sum(-1) / n_valid                             # (B,)
                kd_global_ps = _cos_dist(model_output['feat_global'], t_out['feat_global'])  # (B,)
            else:
                kd_proj_ps = kd_joint_ps = kd_global_ps = zeros_b

            # ---------- KD: sâu hơn (GCN / Transformer pose) ----------
            kd_hyper_ps = 0.0
            if not (feat_mode or hint_mode) and model_output.get('feat_hyper') is not None and t_out.get('feat_hyper') is not None:
                kd_hyper_ps = (_cos_dist(model_output['feat_hyper'], t_out['feat_hyper']) * valid_joint).sum(-1) / n_valid
            
            kd_pose_ps = 0.0
            if not (feat_mode or hint_mode) and model_output.get('feat_pose') is not None and t_out.get('feat_pose') is not None:
                kd_pose_ps = _cos_dist(model_output['feat_pose'], t_out['feat_pose']).mean(-1)
            
            kd_root_ps = 0.0
            if not (feat_mode or hint_mode) and model_output.get('root_pose_6d') is not None and t_out.get('root_pose_6d') is not None:
                kd_root_ps = F.l1_loss(model_output['root_pose_6d'], t_out['root_pose_6d'].detach(), reduction='none').mean(-1)
                
            kd_latent_ps = 0.0
            if not (feat_mode or hint_mode) and model_output.get('pose_latent') is not None and t_out.get('pose_latent') is not None:
                kd_latent_ps = F.l1_loss(model_output['pose_latent'], t_out['pose_latent'].detach(), reduction='none').mean(-1)

            kd_per_sample = (w_proj * kd_proj_ps + w_joint * kd_joint_ps + w_global * kd_global_ps 
                             + kd_hyper_ps + kd_pose_ps + kd_root_ps + kd_latent_ps)

            # ---------- privileged reconstruction (student -> GT 3D) ----------
            privileged_pred = model_output['privileged_3d']
            privileged_error = F.smooth_l1_loss(
                privileged_pred, gt_pose_input, reduction='none', beta=priv_beta).mean(-1)
            privileged_per_sample = (privileged_error * valid_joint).sum(-1) / n_valid

            # Hard samples receive more KD, but the weight is bounded and detached.
            # So sánh cùng domain (mét) để pose_gap có ý nghĩa
            gt_fit_rootrel = gt_fit_joint_cam - gt_fit_joint_cam[:, 0:1, :]
            pose_gap = (pred_pose_rootrel - gt_fit_rootrel).abs().mean(-1)
            pose_gap = (pose_gap * valid_joint).sum(-1) / n_valid
            adaptive_beta = self.kd_weight * min(1.0, float(epoch) / 10.0)

            # Giá trị trước clamp luôn >= 1 nên chỉ cần chặn trên
            adaptive_weight = torch.clamp(
                1.0 + adaptive_beta * pose_gap / pose_gap.detach().mean().clamp_min(1e-6),
                max=2.0
            ).detach()

            if feat_mode:
                # E1: trong so co dinh, khong adaptive (adaptive chi de log = 1)
                adaptive_weight = torch.ones_like(adaptive_weight)
                kd_loss = (kd_per_sample + self.kd_res_w * kd_res_ps).mean() + self.kd_nce_w * kd_nce
            elif hint_mode:
                adaptive_weight = torch.ones_like(adaptive_weight)
                kd_loss = (self.kd_hint_w * kd_hint_ps + self.kd_hint_feat_w * kd_hfeat_ps).mean()
            else:
                kd_loss = (adaptive_weight * kd_per_sample).mean()
            privileged_loss = privileged_per_sample.mean()

            # ---------- hard loss ----------
            loss_smpl_joint_cam = self.jotr_coord_loss(
                pred_pose_rootrel,
                gt_fit_joint_cam,
                fit_joint_trunc * is_valid_fit[:, None, None]
            ).mean()

            gt_orig_joint_img = targets['orig_joint_img'].cuda()
            orig_joint_trunc = meta['orig_joint_trunc'].cuda()

            # Loss body_joint_proj (2D): dung phep chieu phoi canh cua chinh model (focal/princpt trong cfg.DATASET),
            # toa do he heatmap. joint_proj_det da detach pose neu cfg.LOSS.DETACH_POSE_FOR_PROJ bat.
            pred_joint_proj = model_output['joint_proj_det'][:, self.h36m_from_smpl30, :2]
            loss_body_joint_proj = self.jotr_coord_loss(
                pred_joint_proj,
                gt_orig_joint_img[:, :, :2],
                orig_joint_trunc
            ).mean()

            fit_pose_valid = meta['fit_param_valid'].cuda() * is_valid_fit[:, None]
            fit_shape_valid = is_valid_fit[:, None]
            smpl_pose_loss = self.jotr_param_loss(pred_smplpose, gt_smplpose, fit_pose_valid).mean()
            smpl_shape_loss = self.jotr_param_loss(pred_smplshape, gt_smplshape, fit_shape_valid).mean()
            mesh_loss = self.coordLoss(pred_mesh, gt_mesh_cam, is_valid_fit[:, None, None])

            loss_dict = {
                'smpl_joint_cam': loss_smpl_joint_cam,
                'smpl_pose': smpl_pose_loss,
                'smpl_shape': smpl_shape_loss,
                'body_joint_proj': loss_body_joint_proj,
                'mesh_loss': mesh_loss,
            }
            if self.shape_loss_w <= 0:
                loss_dict.pop('smpl_shape')   # bo han khoi AWL (tranh trong so AWL -> vo cung)
            elif self.shape_loss_w != 1.0:
                loss_dict['smpl_shape'] = loss_dict['smpl_shape'] * self.shape_loss_w
            loss_dict = self.awl(loss_dict)
            hard_loss = sum(loss_dict.values())
            if feat_mode or hint_mode:
                loss = self.kd_hard_w * hard_loss + self.kd_lambda * kd_loss + privileged_loss
            else:
                loss = 0.5 * hard_loss + 0.5 * kd_loss + privileged_loss
            # loss = hard_loss

            # update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log (khoảng cách cosine đã center: 0 = giống teacher hoàn toàn)
            running_loss += float(loss.detach().item())
            
            # Helper để tính mean an toàn cho tensor hoặc số thực
            def _get_val(x):
                if isinstance(x, torch.Tensor):
                    return x.mean().item()
                return float(x)
                
            step_stats = {
                'kd_proj': kd_proj_ps.mean().item(),
                'kd_joint': kd_joint_ps.mean().item(),
                'kd_global': kd_global_ps.mean().item(),
                'kd_hyper': _get_val(kd_hyper_ps),
                'kd_pose': _get_val(kd_pose_ps),
                'kd_root': _get_val(kd_root_ps),
                'kd_latent': _get_val(kd_latent_ps),
                'privileged_3d': privileged_loss.item(),
                'adaptive_kd_weight': adaptive_weight.mean().item(),
                'kd_nce': float(kd_nce.item()) if torch.is_tensor(kd_nce) else float(kd_nce),
                'kd_noise_w': noise_w_mean,
                'kd_res': kd_res_ps.mean().item(),
                'lift_err_mm': lift_err_mm,
                'kd_hint': kd_hint_ps.mean().item(),
                'kd_hfeat': kd_hfeat_ps.mean().item(),
                'hint_abs': (model_output['hint_pred'].detach().abs().mean().item()
                             if model_output.get('hint_pred') is not None else 0.0),
            }
            for k, v in step_stats.items():
                running[k] = running.get(k, 0.0) + v
            if cfg.TRAIN.wandb:
                wandb.log(
                    {
                        'train_loss/smpl_joint_cam': loss_smpl_joint_cam.item(),
                        'train_loss/smpl_pose': smpl_pose_loss.item(),
                        'train_loss/smpl_shape': smpl_shape_loss.item(),
                        'train_loss/mesh': mesh_loss.item(),
                        'train_loss/body_joint_proj': loss_body_joint_proj.item(),
                        'train_loss/kd_feat': kd_loss.item(),
                        'train_loss/kd_proj': step_stats['kd_proj'],
                        'train_loss/kd_joint': step_stats['kd_joint'],
                        'train_loss/kd_global': step_stats['kd_global'],
                        'train_loss/kd_hyper': step_stats['kd_hyper'],
                        'train_loss/kd_pose': step_stats['kd_pose'],
                        'train_loss/kd_root': step_stats['kd_root'],
                        'train_loss/kd_latent': step_stats['kd_latent'],
                        'train_loss/privileged_3d': step_stats['privileged_3d'],
                        'train_loss/adaptive_kd_weight': step_stats['adaptive_kd_weight'],
                        'train_loss/kd_nce': step_stats['kd_nce'],
                        'train_loss/kd_noise_w': step_stats['kd_noise_w'],
                        'train_loss/kd_res': step_stats['kd_res'],
                        'train_loss/lift_err_mm': step_stats['lift_err_mm'],
                        'train_loss/kd_hint': step_stats['kd_hint'],
                        'train_loss/kd_hfeat': step_stats['kd_hfeat'],
                        'train_loss/hint_abs': step_stats['hint_abs'],
                        'train_loss/hard_total': hard_loss.item(),
                        'train_loss/total': loss.item(),
                    }
                )

            if i % self.print_freq == 0:
                batch_generator.set_description(
                    f'Epoch{epoch}_({i}/{len(batch_generator)}) => '
                    f'smpl3d: {loss_smpl_joint_cam.item():.3f} '
                    f'smpl: {(smpl_pose_loss + smpl_shape_loss).item():.3f} '
                    f'mesh: {mesh_loss.item():.3f} '
                    f'proj2d: {loss_body_joint_proj.item():.3f} '
                    f'kd: {kd_loss.item():.3f} '
                    f'[P {step_stats["kd_proj"]:.3f} J {step_stats["kd_joint"]:.3f} G {step_stats["kd_global"]:.3f} '
                    f'N {step_stats["kd_nce"]:.3f} R {step_stats["kd_res"]:.3f} nw {step_stats["kd_noise_w"]:.2f} '
                    f'H {step_stats["kd_hint"]:.3f} |z| {step_stats["hint_abs"]:.2f}] '
                    f'lift: {step_stats["lift_err_mm"]:.0f}mm '
                    f'priv: {step_stats["privileged_3d"]:.3f} '
                    f'w: {step_stats["adaptive_kd_weight"]:.2f} '
                    f'tl: {loss.detach().item():.3f}'
                )

        n_batch = len(batch_generator)
        self.loss_history.append(running_loss / n_batch)
        avg = {k: v / n_batch for k, v in running.items()}
        print(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f} | '
              f'KD dist proj: {avg["kd_proj"]:.4f} | joint: {avg["kd_joint"]:.4f} | global: {avg["kd_global"]:.4f}')
        print(f'  Privileged 3D: {avg["privileged_3d"]:.4f} | Adaptive weight: {avg["adaptive_kd_weight"]:.4f} | '
              f'InfoNCE: {avg["kd_nce"]:.4f} | noise w: {avg["kd_noise_w"]:.3f}')
        print(f'  Residual (joint-only -> Teacher): {avg["kd_res"]:.4f} | KD fused (J): {avg["kd_joint"]:.4f} | '
              f'loi joint dau vao (MotionBERT): {avg["lift_err_mm"]:.1f} mm')
        if self.use_kd and self.kd_mode == 'hint':
            print(f'  Goi y: MSE(z_hat, z_T) {avg["kd_hint"]:.4f} | |z_hat| {avg["hint_abs"]:.3f} | '
                  f'KD feat truc tiep (J+G) {avg["kd_hfeat"]:.4f}')

        # Trong so hieu dung cua AWL (0.5 / p^2) theo tung loss: moi run hoc 1 bo khac nhau -> log de doc ablation
        awl_w = {}
        if hasattr(self, 'awl') and hasattr(self.awl, 'keys'):
            awl_w = {k: (0.5 / (p.detach() ** 2)).item() for k, p in zip(self.awl.keys, self.awl.params)}
            print('  AWL trong so: ' + ' | '.join(f'{k}: {v:.3f}' for k, v in awl_w.items()))

        if cfg.TRAIN.wandb:
            wandb.log({f'epoch_metric/{k}': v for k, v in avg.items()})
            if awl_w:
                wandb.log({f'awl_w/{k}': v for k, v in awl_w.items()})
class Student_Tester:
    def __init__(self, args, load_dir=''):
        self.val_loaders, self.val_datasets, self.model, _, _, _, _, _ = \
            prepare_network(args, load_dir=load_dir, is_train=False)
        self.print_freq = cfg.TRAIN.print_freq
        if self.model:
            self.model = torch.nn.DataParallel(self.model).cuda()
        self.surface_error = 9999.9
        self.joint_error = 9999.9

    def test(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()
        results = {}
        for dataset_name, dataset, loader in zip(
                cfg.DATASET.test_list, self.val_datasets, self.val_loaders):
            metrics = evaluate_3dpw_subset(self.model, dataset, loader)
            results[dataset_name] = metrics
            print(
                f'{dataset_name}: MPJPE={metrics["mpjpe"]:.2f}, '
                f'PA-MPJPE={metrics["pa_mpjpe"]:.2f}, '
                f'MPVPE={metrics["mpvpe"]:.2f}'
            )

        if results:
            self.joint_error, self.surface_error = _select_errors(results)
        return results

class LiftTrainer:
    def __init__(self, args, load_dir):
        self.batch_generator, self.dataset_list, self.model, self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history \
            = prepare_network(args, load_dir=load_dir, is_train=True)

        self.loss = self.loss[0]
        self.main_dataset = self.dataset_list[0]
        self.num_joint = self.main_dataset.joint_num
        self.print_freq = cfg.TRAIN.print_freq

        self.model = self.model.cuda()

        if cfg.TRAIN.wandb:
            wandb.init(config=cfg,
                   project=cfg.MODEL.name,
                   name='PoseEst/' + cfg.output_dir.split('/')[-1],
                   dir=cfg.output_dir,
                   job_type="training",
                   reinit=True)

    def train(self, epoch):
        self.model.train()

        lr_check(self.optimizer, epoch)

        running_loss = 0.0
        batch_generator = tqdm(self.batch_generator)
        for i, (img_joint, cam_joint, joint_valid, img_features) in enumerate(batch_generator):
            img_joint, cam_joint = img_joint.cuda().float(), cam_joint.cuda().float()
            joint_valid = joint_valid.cuda().float()
            img_features = img_features.cuda().float()

            pred_joint = self.model(img_joint, img_features)
            pred_joint = pred_joint.view(-1, self.num_joint, 3)

            loss = self.loss(pred_joint, cam_joint, joint_valid)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += float(loss.detach().item())
            if cfg.TRAIN.wandb:
                wandb_loss = loss.detach()
                wandb.log(
                    {
                        'train_loss/total_loss': wandb_loss
                    }
                )

            if i % self.print_freq == 0:
                batch_generator.set_description(f'Epoch{epoch}_({i}/{len(self.batch_generator)}) => '
                                                f'total loss: {loss.detach():.4f} ')

        self.loss_history.append(running_loss / len(self.batch_generator))

        print(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')


class LiftTester:
    def __init__(self, args, load_dir=''):
        self.val_loader, self.val_dataset, self.model, _, _, _, _, _ = \
            prepare_network(args, load_dir=load_dir, is_train=False)
        self.val_dataset = self.val_dataset[0]
        self.val_loader = self.val_loader[0]

        self.num_joint = self.val_dataset.joint_num
        self.print_freq = cfg.TRAIN.print_freq

        if self.model:
            self.model = self.model.cuda()

        # initialize error value
        self.surface_error = 9999.9
        self.joint_error = 9999.9

    def test(self, epoch, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()
        

        result = []
        joint_error = 0.0
        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        loader = tqdm(self.val_loader)
        with torch.no_grad():
            for i, (img_joint, cam_joint, _, img_features) in enumerate(loader):
                img_joint, cam_joint = img_joint.cuda().float(), cam_joint.cuda().float()
                img_features = img_features.cuda().float()

                pred_joint = self.model(img_joint, img_features)
                pred_joint = pred_joint.view(-1, self.num_joint, 3)

                mpjpe = self.val_dataset.compute_joint_err(pred_joint, cam_joint)
                joint_error += mpjpe

                if i % self.print_freq == 0:
                    loader.set_description(f'{eval_prefix}({i}/{len(self.val_loader)}) => joint error: {mpjpe:.4f}')

                # Final Evaluation
                if (epoch == 0 or epoch == cfg.TRAIN.end_epoch):
                    pred_joint, target_joint = pred_joint.detach().cpu().numpy(), cam_joint.detach().cpu().numpy()
                    for j in range(len(pred_joint)):
                        out = {}
                        out['joint_coord'], out['joint_coord_target'] = pred_joint[j], target_joint[j]
                        result.append(out)

        self.joint_error = joint_error / len(self.val_loader)
        print(f'{eval_prefix}MPJPE: {self.joint_error:.4f}')

        if cfg.TRAIN.wandb:
                wandb_error = self.joint_error
                wandb.log(
                    {
                        'epoch': epoch,
                        'error/MPJPE': wandb_error
                    }
                )

        # Final Evaluation
        if (epoch == 0 or epoch == cfg.TRAIN.end_epoch):
            self.val_dataset.evaluate_joint(result)