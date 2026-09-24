import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import Counter
import copy

import models
from data_final.dataset import MultipleDatasets
from core.config import cfg
from core.loss import JOTRCoordLoss, JOTRParamLoss, AutomaticWeightedLoss
from funcs_utils import get_optimizer, load_checkpoint, get_scheduler, count_parameters, lr_check
from utils.jotr_dataset import get_test_dataset as get_jotr_test_dataset
from utils.jotr_dataset import get_train_dataset as get_jotr_train_dataset
from utils.jotr_evaluation import evaluate_3dpw_subset
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

    if is_train:
        criterion = None
        optimizer = get_optimizer(model=model)
        lr_scheduler = get_scheduler(optimizer=optimizer)

    if load_dir and (not is_train or args.resume_training):
        print('==> Loading checkpoint')
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

        self.jotr_coord_loss = JOTRCoordLoss()
        self.jotr_param_loss = JOTRParamLoss()
        # 4 losses: body_joint_cam dropped (joint_img comes from the frozen
        # MotionBERT lifter, so a loss on it has zero gradient).
        self.awl = AutomaticWeightedLoss(4).cuda()
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
            input_pose = inputs['joints'].cuda().float() # keypoint 2D Ä‘áº§u vÃ o
            gt_orig_joint_cam = targets['orig_joint_cam'].cuda() #ÄÃ¢y lÃ  tá»a Ä‘á»™ 3D thá»±c táº¿ Ä‘o Ä‘Æ°á»£c tá»« cÃ¡c cáº£m biáº¿n
            gt_fit_joint_cam = targets['fit_joint_cam'].cuda() # tá»a Ä‘á»™ 3D sinh ra tá»« smpl prj
            orig_joint_valid = meta['orig_joint_valid'].cuda() #mask, = 0 thÃ¬ k tÃ­nh loss
            fit_joint_trunc = meta['fit_joint_trunc'].cuda() # mask
            
            gt_smplpose = targets['pose_param'].cuda()
            gt_smplshape = targets['shape_param'].cuda()
            is_3d = meta['is_3D'].cuda()
            is_valid_fit = meta['is_valid_fit'].cuda()
            
            model_output = self.model(input_image, input_pose, is_train=True)

            pred_mesh = model_output['smpl_mesh_cam']
            pred_smplpose = model_output['smpl_pose']
            pred_smplshape = model_output['smpl_shape']

            # Regress H36M-17 joints from the predicted mesh, root-relative to
            # match gt_fit_joint_cam (the dataset subtracts the pelvis).
            pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh)
            pred_pose = pred_pose - pred_pose[:, 0:1, :]

            # NOTE: no body_joint_cam loss here. In ARTS mode joint_img is the
            # output of the frozen MotionBERT lifter (computed under no_grad), so
            # a loss on it would produce zero gradient — it is intentionally dropped.
            loss_smpl_joint_cam = self.jotr_coord_loss(
                pred_pose, gt_fit_joint_cam, fit_joint_trunc * is_valid_fit[:, None, None]).mean()

            # 2D projection loss. joint_proj comes from get_coord in SMPL-30
            # order, so reduce it to H36M-17 before comparing to the GT.
            pred_joint_proj = model_output.get('joint_proj')
            gt_orig_joint_img = targets['orig_joint_img'].cuda()
            orig_joint_trunc = meta['orig_joint_trunc'].cuda()
            if pred_joint_proj is not None:
                if pred_joint_proj.shape[1] == 30:
                    pred_joint_proj = pred_joint_proj[:, self.h36m_from_smpl30]
                loss_body_joint_proj = self.jotr_coord_loss(
                    pred_joint_proj,
                    gt_orig_joint_img[:, :, :2],
                    orig_joint_trunc
                ).mean()
            else:
                loss_body_joint_proj = torch.tensor(0.0).cuda()

            fit_pose_valid = meta['fit_param_valid'].cuda() * is_valid_fit[:, None]
            fit_shape_valid = is_valid_fit[:, None]
            smpl_pose_loss = self.jotr_param_loss(pred_smplpose, gt_smplpose, fit_pose_valid).mean()
            smpl_shape_loss = self.jotr_param_loss(pred_smplshape, gt_smplshape, fit_shape_valid).mean()
            loss_dict = {
                'smpl_joint_cam': loss_smpl_joint_cam,
                'smpl_pose': smpl_pose_loss,
                'smpl_shape': smpl_shape_loss,
                'body_joint_proj': loss_body_joint_proj,
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
                        'train_loss/body_joint_proj': loss_body_joint_proj.detach(),
                    }
                )

            if i % self.print_freq == 0:
                total_loss = loss.detach()
                batch_generator.set_description(
                    f'Epoch{epoch}_({i}/{len(batch_generator)}) => '
                    f'proj2d: {loss_body_joint_proj.item():.3f} '
                    f'smpl3d: {loss_smpl_joint_cam.item():.3f} '
                    f'smpl: {(smpl_pose_loss + smpl_shape_loss).item():.3f} '
                    f'tl: {total_loss.item():.3f}'
                )

        self.loss_history.append(running_loss / len(batch_generator))
        for i, pg in enumerate(self.optimizer.param_groups):
            group_name = ['SPIN', 'Fresh'][i] if i < 2 else f'Group{i}'
            grads = [p.grad.norm().item() for p in pg['params'] if p.grad is not None]
            if grads:
                print(f"  [{group_name}] grad_norm={sum(grads) / len(grads):.4f}")
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
        self.awl = AutomaticWeightedLoss(3).cuda()
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
            orig_joint_valid = meta['orig_joint_valid'].cuda() #mask, = 0 thÃ¬ k tÃ­nh loss
            fit_joint_trunc = meta['fit_joint_trunc'].cuda() # mask
            
            gt_smplpose = targets['pose_param'].cuda()
            gt_smplshape = targets['shape_param'].cuda()
            is_3d = meta['is_3D'].cuda()
            is_valid_fit = meta['is_valid_fit'].cuda()
            
            teacher_gt_pose3d = gt_fit_joint_cam  # (B, 17, 3), meters, root-relative
            model_output = self.model(input_image, teacher_gt_pose3d, is_train=True)

            pred_mesh = model_output['smpl_mesh_cam']
            pred_smplpose = model_output['smpl_pose']
            pred_smplshape = model_output['smpl_shape']

            # Regress H36M joints from the predicted SMPL mesh.
            # gt_fit_joint_cam is root-relative (the dataset subtracts the
            # pelvis before returning it), so normalize the prediction in the
            # same coordinate system before computing the joint loss.
            pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh)
            pred_pose_rootrel = pred_pose - pred_pose[:, 0:1, :]

            # Coordinate-system diagnostic: print once per epoch, on the first
            # batch, so we can verify that GT and prediction use the same origin.
            if i == 0:
                with torch.no_grad():
                    gt_root_abs = gt_fit_joint_cam[:, 0, :].abs().mean().item()
                    pred_root_abs = pred_pose[:, 0, :].abs().mean().item()
                    pred_rootrel_abs = pred_pose_rootrel[:, 0, :].abs().mean().item()
                    gt_mean_abs = gt_fit_joint_cam.abs().mean().item()
                    pred_mean_abs = pred_pose.abs().mean().item()
                    pred_rootrel_mean_abs = pred_pose_rootrel.abs().mean().item()
                    valid_mask = fit_joint_trunc * is_valid_fit[:, None, None]
                    valid_ratio = valid_mask.float().mean().item()
                    valid_count = valid_mask.sum().item()
                    # Dataset raw SMPL stats. raw/fit ratio should stay near 1.0
                    # after fixing the old duplicate /1000 scale conversion.
                    raw_std = meta['raw_smpl_rootrel_std']
                    raw_mean_abs = meta['raw_smpl_rootrel_mean_abs']
                    raw_bone_median = meta['raw_smpl_bone_median']
                    raw_mesh_std = meta['raw_smpl_mesh_std']
                    raw_joint_std = meta['raw_smpl_joint_std']
                    raw_trans = meta['raw_smpl_trans']
                    raw_to_fit_ratio = raw_std.float().mean().item() / max(
                        gt_fit_joint_cam.std().item(), 1e-12
                    )
                    if not torch.isfinite(pred_pose).all():
                        raise FloatingPointError(
                            'Non-finite values detected in pred_pose during Teacher training.'
                        )
                    if not torch.isfinite(gt_fit_joint_cam).all():
                        raise FloatingPointError(
                            'Non-finite values detected in gt_fit_joint_cam.'
                        )

            loss_smpl_joint_cam = self.jotr_coord_loss(
                pred_pose_rootrel,
                gt_fit_joint_cam,
                fit_joint_trunc * is_valid_fit[:, None, None]
            ).mean()
            fit_pose_valid = meta['fit_param_valid'].cuda() * is_valid_fit[:, None]
            fit_shape_valid = is_valid_fit[:, None]
            smpl_pose_loss = self.jotr_param_loss(pred_smplpose, gt_smplpose, fit_pose_valid).mean()
            smpl_shape_loss = self.jotr_param_loss(pred_smplshape, gt_smplshape, fit_shape_valid).mean()
            loss_dict = {
                'smpl_joint_cam': loss_smpl_joint_cam,
                'smpl_pose': smpl_pose_loss,
                'smpl_shape': smpl_shape_loss,
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
                    }
                )

            if i % self.print_freq == 0:
                total_loss = loss.detach()
                batch_generator.set_description(
                    f'Epoch{epoch}_({i}/{len(batch_generator)}) => '
                    f'smpl3d: {loss_smpl_joint_cam.item():.3f} '
                    f'smpl: {(smpl_pose_loss + smpl_shape_loss).item():.3f} '
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
def load_model_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt
    if isinstance(ckpt, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in ckpt:
                state = ckpt[k]
                break
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'[Teacher] loaded: missing={len(missing)}, unexpected={len(unexpected)}')
    if missing:
        print('  missing (5 key đầu):', missing[:5])
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
            # Khởi tạo student từ trọng số teacher (chỉ module smpl_model)
            self.model.smpl_model.load_state_dict(self.teacher.smpl_model.state_dict())
            print('===> Student smpl_model initialized from Teacher')

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
        self.awl = AutomaticWeightedLoss(3).cuda()
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
            input_pose2d = inputs['joints'].cuda().float()
            gt_orig_joint_cam = targets['orig_joint_cam'].cuda() 
            gt_fit_joint_cam = targets['fit_joint_cam'].cuda() 
            orig_joint_valid = meta['orig_joint_valid'].cuda() 
            fit_joint_trunc = meta['fit_joint_trunc'].cuda() 
            
            gt_smplpose = targets['pose_param'].cuda()
            gt_smplshape = targets['shape_param'].cuda()
            is_3d = meta['is_3D'].cuda()
            is_valid_fit = meta['is_valid_fit'].cuda()
            # Feed 2D pose to model (which routes to MotionBERT in Student mode)
            model_output = self.model(input_image, input_pose2d, is_train=True)

            pred_mesh = model_output['smpl_mesh_cam']
            pred_smplpose = model_output['smpl_pose']
            pred_smplshape = model_output['smpl_shape']
            # Regress H36M joints from the predicted SMPL mesh.
            pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh)
            pred_pose_rootrel = pred_pose - pred_pose[:, 0:1, :]

            # ---------- teacher forward (GT 3D làm đầu vào, không grad) ----------
            with torch.no_grad():
                t_out = self.teacher(input_image, gt_fit_joint_cam, is_train=False)
                t_mesh = t_out['smpl_mesh_cam']
                t_pose = torch.matmul(self.J_regressor[None, :, :], t_mesh)
                t_pose_rootrel = t_pose - t_pose[:, 0:1, :]
                t_mesh_rootrel = t_mesh - t_pose[:, 0:1, :]

            # ---------- loss mềm: student bắt chước teacher ----------
            kd_joint = (pred_pose_rootrel - t_pose_rootrel).abs().mean()
            kd_pose = (pred_smplpose - t_out['smpl_pose']).abs().mean()
            kd_shape = (pred_smplshape - t_out['smpl_shape']).abs().mean()
            kd_loss = kd_joint + kd_pose + kd_shape

            # ---------------------------------------------------------

            loss_smpl_joint_cam = self.jotr_coord_loss(
                pred_pose_rootrel,
                gt_fit_joint_cam,
                fit_joint_trunc * is_valid_fit[:, None, None]
            ).mean()

            # Lấy thêm outputs từ model
            pred_joint_proj = model_output.get('joint_proj')
            
            gt_orig_joint_img = targets['orig_joint_img'].cuda()
            orig_joint_trunc = meta['orig_joint_trunc'].cuda()
            gt_orig_joint_cam_30 = targets['orig_joint_cam'].cuda()
            orig_joint_valid_30 = meta['orig_joint_valid'].cuda()
            # Tính loss body_joint_proj (2D)
            loss_body_joint_proj = self.jotr_coord_loss(
                pred_joint_proj, 
                gt_orig_joint_img[:, :, :2], 
                orig_joint_trunc
            ).mean() if pred_joint_proj is not None else torch.tensor(0.0).cuda()

            fit_pose_valid = meta['fit_param_valid'].cuda() * is_valid_fit[:, None]
            fit_shape_valid = is_valid_fit[:, None]
            smpl_pose_loss = self.jotr_param_loss(pred_smplpose, gt_smplpose, fit_pose_valid).mean()
            smpl_shape_loss = self.jotr_param_loss(pred_smplshape, gt_smplshape, fit_shape_valid).mean()
            loss_dict = {
                'smpl_joint_cam': loss_smpl_joint_cam,
                'smpl_pose': smpl_pose_loss,
                'smpl_shape': smpl_shape_loss,
            }
            loss_dict = self.awl(loss_dict)
            hard_loss = sum(loss_dict.values())
            loss = hard_loss + self.kd_weight * kd_loss
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
                        'train_loss/body_joint_proj': loss_body_joint_proj.detach()
                    }
                )

            if i % self.print_freq == 0:
                total_loss = loss.detach()
                batch_generator.set_description(
                    f'Epoch{epoch}_({i}/{len(batch_generator)}) => '
                    f'proj2d: {loss_body_joint_proj.item():.3f} '
                    f'smpl3d: {loss_smpl_joint_cam.item():.3f} '
                    f'smpl: {(smpl_pose_loss + smpl_shape_loss).item():.3f} '
                    f'tl: {total_loss.item():.3f}'
                )

        self.loss_history.append(running_loss / len(batch_generator))
        print(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')

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
