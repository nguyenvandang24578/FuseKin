import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import Counter

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
    batch_per_dataset = cfg[dataset_split].batch_size // len(dataset_names)
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
        if cfg.MODEL.name == 'ARTS':
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
        
        # Mapping from SMPL 30 joints (Dataloader output) to H36M 17 joints (ARTS input)
        smpl_joints = self.main_dataset.joints_name
        h36m_joints = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.smpl_to_h36m_idx = [smpl_joints.index(name) for name in h36m_joints]

        self.model = torch.nn.DataParallel(self.model).cuda()

        self.jotr_coord_loss = JOTRCoordLoss()
        self.jotr_param_loss = JOTRParamLoss()
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
            except:
                pass

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
        for i, pg in enumerate(self.optimizer.param_groups):
            group_name = ['SPIN', 'Fresh'][i] if i < 2 else f'Group{i}'
            print(f"  [{group_name}] lr={pg['lr']:.2e} | params={sum(p.numel() for p in pg['params']):,}")
        running_loss = 0.0
        batch_generator = tqdm(self.batch_generator)
        for i, (inputs, targets, meta) in enumerate(batch_generator):
            # convert to cuda
            input_image = inputs['img'].cuda().float()
            # Slice 30 joints -> 17 joints matching ARTS expected input
            input_pose = inputs['joints'][:, self.smpl_to_h36m_idx].cuda().float()
            gt_orig_joint_cam = targets['orig_joint_cam'][:, self.smpl_to_h36m_idx].cuda()
            gt_fit_joint_cam = targets['fit_joint_cam'][:, self.smpl_to_h36m_idx].cuda()
            orig_joint_valid = meta['orig_joint_valid'][:, self.smpl_to_h36m_idx].cuda()
            fit_joint_trunc = meta['fit_joint_trunc'][:, self.smpl_to_h36m_idx].cuda()
            
            gt_smplpose = targets['pose_param'].cuda()
            gt_smplshape = targets['shape_param'].cuda()
            is_3d = meta['is_3D'].cuda()
            is_valid_fit = meta['is_valid_fit'].cuda()
            
            model_output = self.model(input_image, input_pose, is_train=True)

            pred_joint_img = model_output['joint_img']
            pred_mesh = model_output['smpl_mesh_cam']
            pred_smplpose = model_output['smpl_pose']
            pred_smplshape = model_output['smpl_shape']

            pred_pose = torch.matmul(self.J_regressor[None, :, :], pred_mesh)
            
            loss_body_joint_cam = 5 * self.jotr_coord_loss(
                pred_joint_img, gt_orig_joint_cam, orig_joint_valid * is_3d[:, None, None]).mean()
            loss_smpl_joint_cam = self.jotr_coord_loss(
                pred_pose, gt_fit_joint_cam, fit_joint_trunc * is_valid_fit[:, None, None]).mean()
            fit_pose_valid = meta['fit_param_valid'].cuda() * is_valid_fit[:, None]
            fit_shape_valid = is_valid_fit[:, None]
            smpl_pose_loss = self.jotr_param_loss(pred_smplpose, gt_smplpose, fit_pose_valid).mean()
            smpl_shape_loss = self.jotr_param_loss(pred_smplshape, gt_smplshape, fit_shape_valid).mean()
            loss_dict = {
                'body_joint_cam': loss_body_joint_cam,
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
                        'train_loss/body_joint_cam': loss_body_joint_cam.detach(),
                        'train_loss/smpl_joint_cam': loss_smpl_joint_cam.detach(),
                        'train_loss/smpl_pose': smpl_pose_loss.detach(),
                        'train_loss/smpl_shape': smpl_shape_loss.detach(),
                    }
                )

            if i % self.print_freq == 0:
                total_loss = loss.detach()
                batch_generator.set_description(
                    f'Epoch{epoch}_({i}/{len(batch_generator)}) => '
                    f'body3d: {loss_body_joint_cam.item():.3f} '
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

class LiftTrainer:
    def __init__(self, args, load_dir):
        self.batch_generator, self.dataset_list, self.model, self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history \
            = prepare_network(args, load_dir=load_dir, is_train=True)

        self.loss = self.loss[0]
        self.main_dataset = self.dataset_list[0]
        self.num_joint = self.main_dataset.joint_num
        # self.num_joint = 16
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
            pred_joint = pred_joint.view(-1, cfg.DATASET.seqlen, self.num_joint, 3)
            cam_joint = cam_joint.view(-1, cfg.DATASET.seqlen, self.num_joint, 3)

            mpjpe_loss = self.loss(pred_joint, cam_joint, joint_valid)
            
            loss = mpjpe_loss

            self.optimizer.zero_grad()
            loss.backward()  
            # loss.sum().backward()
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
        # self.num_joint = 16
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
                pred_joint = pred_joint.view(-1, cfg.DATASET.seqlen, self.num_joint, 3)

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