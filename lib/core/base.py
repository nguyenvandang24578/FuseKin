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
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
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

        # ---- Discriminative LR: fusion params get 1/10 of base LR ----
        fusion_lr_scale = getattr(cfg.MODEL, 'fusion_lr_scale', 0.1)
        fusion_param_ids = set()
        # Collect fusion parameter IDs (through DataParallel → module)
        for p in self.model.module.pose_mesh_coevo.fusion.parameters():
            fusion_param_ids.add(id(p))

        # Rebuild optimizer param groups: split fusion vs rest
        base_lr = self.optimizer.defaults['lr']
        non_fusion_params = []
        fusion_params = []
        for pg in self.optimizer.param_groups:
            for p in pg['params']:
                if id(p) in fusion_param_ids:
                    fusion_params.append(p)
                else:
                    non_fusion_params.append(p)

        # Clear existing param groups and re-add with discriminative LR
        self.optimizer.param_groups.clear()
        self.optimizer.add_param_group({'params': non_fusion_params, 'lr': base_lr})
        self.optimizer.add_param_group({'params': fusion_params, 'lr': base_lr * fusion_lr_scale})
        print(f'[Trainer] Discriminative LR: fusion={base_lr * fusion_lr_scale:.1e}, rest={base_lr:.1e}')

        self.jotr_coord_loss = JOTRCoordLoss()
        self.jotr_param_loss = JOTRParamLoss()
        self.coordLoss = CoordLoss(has_valid=True)
        # 5 losses now (including mesh_loss)
        self.awl = AutomaticWeightedLoss(5).cuda()
        self.optimizer.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})
        # Restore AWL weights if resuming
        if hasattr(args, 'resume_training') and args.resume_training:
            import os
            from funcs_utils import load_checkpoint as _load_ckpt
            try:
                ckpt = _load_ckpt(load_dir=os.path.join(cfg.checkpoint_dir), pick_best=False)
                if 'awl_state_dict' in ckpt:
                    self.awl.load_state_dict(ckpt['awl_state_dict'])
                    print('===> AWL weights restored from checkpoint')
            except Exception as e:
                print(f'===> Could not restore AWL weights: {e}')

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

            # ---- Forward ----
            gt_pose_input = gt_fit_joint_cam - gt_fit_joint_cam[:, 0:1, :]
            model_output = self.model(
                input_image, gt_pose_input, is_train=True, use_gt_3d=True,
                gt_pose_6d=gt_pose_6d, kp2d=kp2d, kp_conf=kp_conf,
                pose_valid_mask=pose_valid_mask,
            )

            pred_mesh = model_output['smpl_mesh_cam']
            pred_smplpose = model_output['smpl_pose']
            pred_smplshape = model_output['smpl_shape']
            cam_param = model_output['cam_param']

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
                if use_diffusion:
                    desc += f'diff: {diff_loss.item():.4f} '
                desc += f'tl: {total_loss.item():.3f}'
                batch_generator.set_description(desc)

        self.loss_history.append(running_loss / len(batch_generator))
        print(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')

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

        # The dataset wrapper already emits H36M-17 GT and 2D inputs. Kept for
        # parity with the other trainers (Teacher uses only H36M-17 targets).
        h36m_joints = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        smpl30_joints = self.main_dataset.mesh_model.joints_name
        self.h36m_from_smpl30 = [smpl30_joints.index(name) for name in h36m_joints]

        self.model = torch.nn.DataParallel(self.model).cuda()

        self.jotr_coord_loss = JOTRCoordLoss()
        self.jotr_param_loss = JOTRParamLoss()
        self.coordLoss = CoordLoss(has_valid=True)
        self.awl = AutomaticWeightedLoss(4).cuda()  # 4 losses: joint_cam, pose, shape, mesh
        self.optimizer.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})
        # Restore AWL weights if resuming
        if hasattr(args, 'resume_training') and args.resume_training:
            import os
            from funcs_utils import load_checkpoint as _load_ckpt
            try:
                ckpt = _load_ckpt(load_dir=os.path.join(cfg.checkpoint_dir), pick_best=False)
                if 'awl_state_dict' in ckpt:
                    self.awl.load_state_dict(ckpt['awl_state_dict'])
                    print('===> AWL weights restored from checkpoint')
            except Exception as e:
                print(f'===> Could not restore AWL weights: {e}')

        if cfg.TRAIN.wandb:
            wandb.init(config=cfg,
                   project=cfg.MODEL.name,
                   name='ARTS/' + cfg.output_dir.split('/')[-1],
                   dir=cfg.output_dir,
                   job_type="training",
                   reinit=True)

    def train(self, epoch):
        self.model.train()

        lr_check(self.optimizer, epoch)
        running_loss = 0.0
        batch_generator = tqdm(self.batch_generator)
        for i, (inputs, targets, meta) in enumerate(batch_generator):
            # convert to cuda
            input_image = inputs['img'].cuda().float()
            gt_orig_joint_cam = targets['orig_joint_cam'].cuda() #ÄÃ¢y lÃ  tá»a Ä‘á»™ 3D thá»±c táº¿ Ä‘o Ä‘Æ°á»£c tá»« cÃ¡c cáº£m biáº¿n
            gt_fit_joint_cam = targets['fit_joint_cam'].cuda() # tá»a Ä‘á»™ 3D sinh ra tá»« smpl prj
            gt_mesh_cam = targets['smpl_mesh_cam'].cuda()
            orig_joint_valid = meta['orig_joint_valid'].cuda() #mask, = 0 thÃ¬ k tÃ­nh loss
            fit_joint_trunc = meta['fit_joint_trunc'].cuda() # mask
            
            gt_smplpose = targets['pose_param'].cuda()
            gt_smplshape = targets['shape_param'].cuda()
            is_3d = meta['is_3D'].cuda()
            is_valid_fit = meta['is_valid_fit'].cuda()
            
            teacher_gt_pose3d = gt_fit_joint_cam  # (B, 17, 3), meters, absolute
            model_output = self.model(input_image, teacher_gt_pose3d, is_train=True)

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
            loss_dict = {
                'smpl_joint_cam': loss_smpl_joint_cam,
                'smpl_pose': smpl_pose_loss,
                'smpl_shape': smpl_shape_loss,
                'mesh_loss': mesh_loss,
            }
            loss_dict = self.awl(loss_dict)
            loss = sum(loss_dict.values())

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
                    }
                )

            if i % self.print_freq == 0:
                total_loss = loss.detach()
                batch_generator.set_description(
                    f'Epoch{epoch}_({i}/{len(batch_generator)}) => '
                    f'smpl3d: {loss_smpl_joint_cam.item():.3f} '
                    f'smpl: {(smpl_pose_loss + smpl_shape_loss).item():.3f} '
                    f'mesh: {mesh_loss.item():.3f} '
                    f'tl: {total_loss.item():.3f}'
                )

        self.loss_history.append(running_loss / len(batch_generator))
        print(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')

class Teacher_Tester:
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
    assert not shape_bad, f'Có key lệch shape: {shape_bad[:3]}'
    assert not [k for k in missing if k.startswith(('smpl_model.', 'backbone.'))], \
        f'Teacher chưa nạp đủ smpl_model/backbone: {missing[:5]}'
    return model
class Student_Trainer:
    def __init__(self, args, load_dir):
        self.batch_generator, self.dataset_list, self.model, self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history\
            = prepare_network(args, load_dir=load_dir, is_train=True)

        self.main_dataset = self.dataset_list[0]
        self.print_freq = cfg.TRAIN.print_freq

        self.J_regressor = eval(f'torch.Tensor(self.main_dataset.joint_regressor_{cfg.DATASET.target_joint_set}).cuda()')
#---------------------------------------------------------------------------------
        teacher_ckpt = cfg.MODEL.get('TEACHER', '')
        assert teacher_ckpt, 'Cần đặt cfg.MODEL.TEACHER (checkpoint của Teacher_Trainer)'
        self.kd_weight = cfg.MODEL.get('kd_weight', 1.0)

        self.teacher = copy.deepcopy(self.model)   # cùng kiến trúc với student
        self.teacher.mode = 'teacher'              # forward() sẽ chạy forward_teacher
        load_model_weights(self.teacher, teacher_ckpt)

        resume = hasattr(args, 'resume_training') and args.resume_training
        if not resume:
            self.model.smpl_model.load_state_dict(self.teacher.smpl_model.state_dict())
            self.model.backbone.load_state_dict(self.teacher.backbone.state_dict())
            print('===> Student smpl_model + backbone initialized from Teacher')

        self.teacher = torch.nn.DataParallel(self.teacher).cuda()
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
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
        
        hpe_dim = cfg.MODEL.get('hpe_dim', 512)
        self.feat_projector = torch.nn.Sequential(
            torch.nn.Linear(hpe_dim, hpe_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(hpe_dim * 2, hpe_dim)
        ).cuda()
        
        self.awl = AutomaticWeightedLoss(5).cuda()
        self.optimizer.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})
        self.optimizer.add_param_group({'params': self.feat_projector.parameters(), 'weight_decay': 0})
        
        # Restore AWL weights if resuming
        if hasattr(args, 'resume_training') and args.resume_training:
            import os
            from funcs_utils import load_checkpoint as _load_ckpt
            try:
                ckpt = _load_ckpt(load_dir=os.path.join(cfg.checkpoint_dir), pick_best=False)
                if 'awl_state_dict' in ckpt:
                    self.awl.load_state_dict(ckpt['awl_state_dict'])
                    print('===> AWL weights restored from checkpoint')
                if 'projector_state_dict' in ckpt:
                    self.feat_projector.load_state_dict(ckpt['projector_state_dict'])
                    print('===> Projector weights restored from checkpoint')
            except Exception as e:
                print(f'===> Could not restore AWL/Projector weights: {e}')

        if cfg.TRAIN.wandb:
            wandb.init(config=cfg,
                   project=cfg.MODEL.name,
                   name='ARTS/' + cfg.output_dir.split('/')[-1],
                   dir=cfg.output_dir,
                   job_type="training",
                   reinit=True)

    def train(self, epoch):
        self.model.train()
        self.feat_projector.train()

        lr_check(self.optimizer, epoch)
        running_loss = 0.0
        running_cos_joint = 0.0
        running_cos_img = 0.0
        batch_generator = tqdm(self.batch_generator)
        for i, (inputs, targets, meta) in enumerate(batch_generator):
            # convert to cuda
            input_image = inputs['img'].cuda().float()
            input_pose2d = inputs['joints'].cuda().float()
            gt_orig_joint_cam = targets['orig_joint_cam'].cuda() 
            gt_fit_joint_cam = targets['fit_joint_cam'].cuda() 
            orig_joint_valid = meta['orig_joint_valid'].cuda() 
            fit_joint_trunc = meta['fit_joint_trunc'].cuda() 
            
            gt_smplpose = targets['pose_param'].cuda()
            gt_smplshape = targets['shape_param'].cuda()
            gt_mesh_cam = targets['smpl_mesh_cam'].cuda()
            is_3d = meta['is_3D'].cuda()
            is_valid_fit = meta['is_valid_fit'].cuda()
            
            # print(f"Input Pose2D Shape: {input_pose2d.shape}")
            # print(f"input_image shape: {input_image.shape}")
            # print(f"gt_orig_joint_cam shape: {gt_orig_joint_cam.shape}")
            # print(f"gt_fit_joint_cam shape: {gt_fit_joint_cam.shape}")
            # print(f"orig_joint_valid shape: {orig_joint_valid.shape}")
            # print(f"fit_joint_trunc shape: {fit_joint_trunc.shape}")
            # print(f"gt_smplpose shape: {gt_smplpose.shape}")
            # print(f"gt_smplshape shape: {gt_smplshape.shape}")
            # print(f"is_3d shape: {is_3d.shape}")
            # print(f"is_valid_fit shape: {is_valid_fit.shape}")
            # Feed 2D pose to model (which routes to MotionBERT in Student mode)
            model_output = self.model(input_image, input_pose2d, is_train=True)

            pred_mesh = model_output['smpl_mesh_cam']
            pred_smplpose = model_output['smpl_pose']
            pred_smplshape = model_output['smpl_shape']

            # print(f"pred_mesh shape: {pred_mesh.shape}")
            # print(f"pred_smplpose shape: {pred_smplpose.shape}")
            # print(f"pred_smplshape shape: {pred_smplshape.shape}")
            # Regress H36M joints from the predicted SMPL mesh.
            pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh)
            pred_pose_rootrel = pred_pose - pred_pose[:, 0:1, :]

            # ---------- teacher forward (GT 3D làm đầu vào, không grad) ----------
            with torch.no_grad():
                t_out = self.teacher(input_image, gt_fit_joint_cam, is_train=False)

            # ---------- loss mềm: student bắt chước teacher (tầng feature) ----------
            # Trích xuất đặc trưng Joint (17 khớp)
            s_feat_joint = model_output['feat']
            t_feat_joint = t_out['feat'].detach()
            
            # Trích xuất đặc trưng RGB (Token ảnh)
            s_feat_img = model_output['feat_img']
            t_feat_img = t_out['feat_img'].detach()
            
            # 1. Tính MSE Loss cho từng loại
            mse_joint = F.mse_loss(s_feat_joint, t_feat_joint)
            mse_img = F.mse_loss(s_feat_img, t_feat_img)
            
            # 2. Tính Cosine Loss cho từng loại
            # Cần flatten về (B, -1) để so sánh hướng tổng thể của tensor
            cos_joint = F.cosine_similarity(s_feat_joint.flatten(1), t_feat_joint.flatten(1), dim=-1)
            cos_img = F.cosine_similarity(s_feat_img.flatten(1), t_feat_img.flatten(1), dim=-1)
            
            cosine_loss_joint = (1.0 - cos_joint).mean()
            cosine_loss_img = (1.0 - cos_img).mean()
            
            # 3. Tổng hợp Knowledge Distillation Loss
            # Trọng số 0.1 cho Cosine để cân bằng với MSE
            kd_loss = (mse_joint + mse_img) + 0.1 * (cosine_loss_joint + cosine_loss_img)

            # ---------------------------------------------------------

            loss_smpl_joint_cam = self.jotr_coord_loss(
                pred_pose_rootrel,
                gt_fit_joint_cam,
                fit_joint_trunc * is_valid_fit[:, None, None]
            ).mean()

            # Lấy thêm outputs từ model
            # pred_joint_proj = model_output.get('joint_proj')
            
            gt_orig_joint_img = targets['orig_joint_img'].cuda()
            orig_joint_trunc = meta['orig_joint_trunc'].cuda()
            gt_orig_joint_cam_30 = targets['orig_joint_cam'].cuda()
            orig_joint_valid_30 = meta['orig_joint_valid'].cuda()
            # print(f"pred_joint_proj shape: {pred_joint_proj.shape}")
            # print(f"gt_orig_joint_img shape: {gt_orig_joint_img.shape}")
            # print(f"orig_joint_trunc shape: {orig_joint_trunc.shape}")
            # print(f"gt_orig_joint_cam_30 shape: {gt_orig_joint_cam_30.shape}")
            # print(f"orig_joint_valid_30 shape: {orig_joint_valid_30.shape}")
            # Tính loss body_joint_proj (2D)
            # Dùng Weak Perspective Projection của SPIN (từ cam_param: scale, tx, ty)
            cam = model_output['cam_param'] # (B, 3)
            scale = cam[:, 0:1, None]
            trans = cam[:, 1:3, None].transpose(1, 2) # (B, 1, 2)
            
            # Lấy 17 khớp 3D (Root-relative)
            pred_pose_17 = torch.matmul(self.J_regressor[None, :, :], pred_mesh)
            
            # Chiếu 3D xuống 2D (tọa độ normalize [-1, 1])
            proj_2d = scale * pred_pose_17[:, :, :2] + trans
            
            # Đổi từ [-1, 1] sang tọa độ pixel của ảnh input (ví dụ: 256x256)
            proj_pixel = (proj_2d + 1.0) * 0.5 * cfg.input_img_shape[0]
            
            # Chuẩn hóa về tọa độ heatmap (ví dụ: 64x64)
            pred_joint_proj = proj_pixel * (cfg.output_hm_shape[1] / cfg.input_img_shape[0])
            
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
            loss_dict = self.awl(loss_dict)
            hard_loss = sum(loss_dict.values())
            loss = 0.5  * hard_loss + 0.5 * kd_loss
            # update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # log
            running_loss += float(loss.detach().item())
            running_cos_joint += float(cos_joint.mean().detach().item())
            running_cos_img += float(cos_img.mean().detach().item())
            if cfg.TRAIN.wandb:
                wandb.log(
                    {
                        'train_loss/smpl_joint_cam': loss_smpl_joint_cam.item(),
                        'train_loss/smpl_pose': smpl_pose_loss.item(),
                        'train_loss/smpl_shape': smpl_shape_loss.item(),
                        'train_loss/mesh': mesh_loss.item(),
                        'train_loss/kd_feat': kd_loss.item(),
                        'train_loss/hard_total': hard_loss.item(),
                        'train_loss/total': loss.item(),
                    }
                )

            if i % self.print_freq == 0:
                total_loss = loss.detach()
                batch_generator.set_description(
                    f'Epoch{epoch}_({i}/{len(batch_generator)}) => '
                    f'smpl3d: {loss_smpl_joint_cam.item():.3f} '
                    f'smpl: {(smpl_pose_loss + smpl_shape_loss).item():.3f} '
                    f'mesh: {mesh_loss.item():.3f} '
                    f'kd_feat: {kd_loss.item():.3f} '
                    f'tl: {total_loss.item():.3f}'
                )

        self.loss_history.append(running_loss / len(batch_generator))
        avg_cos_joint = running_cos_joint / len(batch_generator)
        avg_cos_img = running_cos_img / len(batch_generator)
        print(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f} | Cosine Joint: {avg_cos_joint:.4f} | Cosine IMG: {avg_cos_img:.4f}')

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
            self.joint_error = sum(item['mpjpe'] for item in results.values()) / len(results)
            self.surface_error = sum(item['mpvpe'] for item in results.values()) / len(results)
        return results
