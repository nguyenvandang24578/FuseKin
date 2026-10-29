"""
Huan luyen / danh gia Teacher va Student FuseKin.
  - Teacher_Trainer: anh + GT 3D (che ngau nhien mot phan khop) -> mesh.
  - Student_Trainer: anh + 2D -> DiffLifter -> mesh; KD feature tu Teacher (tuy chon).
  - Tester: danh gia dataset.evaluate tren cfg.DATASET.test_list.
Loss mesh va loss chieu 2D dung quy uoc khop H36M-17 hoi quy tu mesh, giong GT va dataset.evaluate.
"""
import pickle
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from core.config import cfg
from core.loss import JOTRCoordLoss, JOTRParamLoss, AutomaticWeightedLoss, CoordLoss
from data_final.dataset import MultipleDatasets
from funcs_utils import get_optimizer, get_scheduler, count_parameters, lr_check
from models.teacher_student import LEGACY_KEY_PARTS
from utils.joint_relation import joint_relation_matrix
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
        mix = cfg.TRAIN.get('mix_mode', 'ratio')
        if mix in ('full', 'jotr'):
            # Duyet het du lieu moi epoch (khong lay mau co hoan lai), xao tron
            if mix == 'full':
                trainset_loader = MultipleDatasets(dataset_list, make_same_len=False)
            else:
                g3 = [ds for n, ds in zip(dataset_names, dataset_list) if n in ('Human36M', 'MuCo')]
                g2 = [ds for n, ds in zip(dataset_names, dataset_list) if n not in ('Human36M', 'MuCo')]
                assert g3 and g2, f"mix_mode 'jotr' can ca bo 3D (Human36M / MuCo) lan bo 2D: {dataset_names}"
                trainset_loader = MultipleDatasets([MultipleDatasets(g3, make_same_len=False),
                                                    MultipleDatasets(g2, make_same_len=False)], make_same_len=True)
            n_tot = len(trainset_loader)
            for name, ds in zip(dataset_names, dataset_list):
                print(f'# [{mix}] {name}: {len(ds)} mau')
            bs = batch_per_dataset * len(dataset_names)
            print(f'# [{mix}] mau / epoch: {n_tot} -> {n_tot // bs} iter / epoch (batch {bs})')
            batch_generator = DataLoader(dataset=trainset_loader, batch_size=bs, shuffle=True,
                                         num_workers=cfg.DATASET.workers, pin_memory=False)
            return dataset_list, batch_generator
        ratio = list(cfg.TRAIN.get('dataset_ratio', []) or [])
        if ratio:
            # Tron theo ti le: noi cac bo (khong lap), moi mau trong so = ti le bo / so mau bo
            assert len(ratio) == len(dataset_list), f'TRAIN.dataset_ratio {ratio} phai co {len(dataset_list)} phan tu'
            trainset_loader = MultipleDatasets(dataset_list, make_same_len=False)
            rs = np.asarray(ratio, dtype=np.float64) / float(sum(ratio))
            weights = np.concatenate([np.full(len(ds), r / len(ds)) for ds, r in zip(dataset_list, rs)])
            n_ep = int(cfg.TRAIN.get('samples_per_epoch', 0)) or len(trainset_loader)
            sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(weights), num_samples=n_ep,
                                                             replacement=True)
            for name, ds, r in zip(dataset_names, dataset_list, rs):
                print(f'# ti le {name}: {r:.2f} ({len(ds)} mau) -> ~{r * n_ep / len(ds):.2f} lan / epoch')
            print(f'# mau / epoch: {n_ep}')
            batch_generator = DataLoader(dataset=trainset_loader,
                                         batch_size=batch_per_dataset * len(dataset_names), sampler=sampler,
                                         num_workers=cfg.DATASET.workers, pin_memory=False)
            return dataset_list, batch_generator
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


def _make_optimizer(model):
    """Nhu get_optimizer; neu v3_bb_finetune: backbone (tru module ghep heatmap) co lr = TRAIN.lr * v3_bb_lr_scale
    (khoa 'lr_scale' trong param group, lr_warmup cung ton trong)."""
    if not cfg.MODEL.v3_bb_finetune:
        return get_optimizer(model=model)
    assert cfg.TRAIN.optimizer == 'adam', 'v3_bb_finetune hien chi ho tro optimizer adam'
    bb, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb if name.startswith('backbone.') and not name.startswith('backbone.heat_fuse.') else rest).append(p)
    scale = float(cfg.MODEL.v3_bb_lr_scale)
    print(f'===> Optimizer: {len(rest)} tensor lr={cfg.TRAIN.lr:.1e} | backbone {len(bb)} tensor lr x{scale}')
    return torch.optim.Adam([{'params': rest},
                             {'params': bb, 'lr': cfg.TRAIN.lr * scale, 'lr_scale': scale}], lr=cfg.TRAIN.lr)


def prepare_network(args, load_dir='', is_train=True):
    dataset_names = cfg.DATASET.train_list if is_train else cfg.DATASET.test_list
    dataset_list, dataloader = get_dataloader(args, dataset_names, is_train)
    model, criterion, optimizer, lr_scheduler = None, None, None, None
    loss_history, test_error_history = [], {'surface': [], 'joint': []}

    if is_train or load_dir:
        print(f"==> Preparing {cfg.MODEL.name} MODEL...")
        model = models.ARTS.get_model(num_joint=17, embed_dim=cfg.MODEL.hpe_dim)
        print('# of model parameters: {}'.format(count_parameters(model)))
        model = model.cuda()

    if is_train:
        optimizer = _make_optimizer(model)
        lr_scheduler = get_scheduler(optimizer=optimizer)

    if load_dir and (not is_train or args.resume_training):
        print('==> Loading checkpoint')
        model = load_model_weights(model, load_dir)
        checkpoint = torch.load(load_dir, map_location='cuda', pickle_module=_PickleShim, weights_only=False)

        if is_train:
            # Trainer them nhom tham so (AWL, projector KD) vao optimizer SAU khi tao -> o day so nhom
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
            lr_state['milestones'], lr_state['gamma'] = Counter(cfg.TRAIN.lr_step), cfg.TRAIN.lr_factor
            lr_scheduler.load_state_dict(lr_state)

            loss_history = checkpoint['train_log']
            test_error_history = checkpoint['test_log']
            cfg.TRAIN.begin_epoch = checkpoint['epoch'] + 1
            print('===> resume from epoch {:d}, current lr: {:.0e}, milestones: {}, lr factor: {:.0e}'
                  .format(cfg.TRAIN.begin_epoch, curr_lr, lr_state['milestones'], lr_state['gamma']))

    return dataloader, dataset_list, model, criterion, optimizer, lr_scheduler, loss_history, test_error_history


# ============================================================
# Loss cung quy uoc khop voi GT (H36M-17 hoi quy tu mesh)
# ============================================================
def _h36m_proj_heatmap(J_h36m, mesh_abs):
    """H36M-17 hoi quy tu mesh TUYET DOI (camera) -> chieu phoi canh giong get_coord -> toa do output_hm_shape."""
    j = torch.matmul(J_h36m[None, :, :], mesh_abs)
    x = j[:, :, 0] / (j[:, :, 2] + 1e-4) * cfg.DATASET.focal[0] + cfg.DATASET.princpt[0]
    y = j[:, :, 1] / (j[:, :, 2] + 1e-4) * cfg.DATASET.focal[1] + cfg.DATASET.princpt[1]
    x = x / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
    y = y / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
    return torch.stack((x, y), 2)


def _mesh_for_loss(J_h36m, pred_mesh, mesh_abs=None):
    """Mesh pred cho loss mesh, cung goc voi GT: GT = mesh_tuyet_doi - J_h36m[0] @ mesh_tuyet_doi (pelvis H36M).
    Model tra mesh tru pelvis SMPL (lech pelvis H36M ~38 mm) -> doi goc sang pelvis H36M.
    Dung mesh tuyet doi (smpl_mesh_cam_render) neu co de khop tuyet doi (tong hang pelvis H36M = 0.9996)."""
    if mesh_abs is not None:
        return mesh_abs - torch.matmul(J_h36m[None, 0:1, :], mesh_abs)
    return pred_mesh - torch.matmul(J_h36m[None, 0:1, :], pred_mesh)


def _proj_for_loss(J_h36m, model_output):
    """2D pred cho loss chieu: H36M-17 hoi quy tu mesh tuyet doi (pose detach luc train), giong GT 2D
    (= chieu J_h36m @ mesh). Truoc 2026-10-25 dung khop SMPL-30 chon theo ten -> lech TB 68 mm."""
    return _h36m_proj_heatmap(J_h36m, model_output['smpl_mesh_cam_render_det'])


def _smpl_hard_losses(trainer, model_output, targets, meta, gt_fit_joint_cam, fit_joint_trunc, is_valid_fit):
    """5 loss voi GT (dung chung Teacher / Student). Tra (loss_dict chua qua AWL, pred_pose_rootrel)."""
    pred_mesh = model_output['smpl_mesh_cam']
    # H36M-17 hoi quy tu mesh pred, root-relative (giong GT fit_joint_cam)
    pred_pose = torch.matmul(trainer.J_regressor[None, :, :], pred_mesh)
    pred_pose = pred_pose - pred_pose[:, 0:1, :]
    jvalid = fit_joint_trunc * is_valid_fit[:, None, None]
    coord = trainer.jotr_coord_loss(pred_pose, gt_fit_joint_cam, jvalid)              # (B, J, 3)
    log_sigma = _joint_log_sigma(trainer)
    if log_sigma is None:
        loss_smpl_joint_cam = coord.mean()
    else:
        # [E] Do bat dinh tung khop. KHONG dua NLL (|e| / sigma + log sigma, co the AM) vao AWL: AWL gia dinh loss >= 0,
        # loss am -> 0.5 / p^2 no tung (log epoch 6-11: trong so smpl_joint_cam 43 -> 7.5e5) -> train vo.
        # Cach moi: (1) AWL nhan L1 co trong so khop w = exp(-log_sigma) (detach, chuan hoa trung binh = 1) -> luon >= 0,
        # cung thang do voi v2; (2) log_sigma hoc rieng bang NLL tren loi detach (grad chi vao log_sigma),
        # cong NGOAI AWL qua khoa 'joint_sigma_nll' (xem _apply_awl).
        vj = jvalid.reshape(coord.shape[0], -1)
        err_j = coord.mean(-1)                                                       # (B, J)
        w = torch.exp(-log_sigma.detach())
        w = w / w.mean()
        loss_smpl_joint_cam = (err_j * w[None]).mean()
        nll = (err_j.detach() * torch.exp(-log_sigma)[None] + log_sigma[None]) * vj
        sigma_nll = nll.sum() / vj.sum().clamp_min(1.0)
    fit_pose_valid = meta['fit_param_valid'].cuda() * is_valid_fit[:, None]
    fit_shape_valid = is_valid_fit[:, None]
    smpl_pose_loss = trainer.jotr_param_loss(model_output['smpl_pose'], targets['pose_param'].cuda(),
                                             fit_pose_valid).mean()
    smpl_shape_loss = trainer.jotr_param_loss(model_output['smpl_shape'], targets['shape_param'].cuda(),
                                              fit_shape_valid).mean()
    mesh_loss = trainer.coordLoss(_mesh_for_loss(trainer.J_regressor, pred_mesh, model_output['smpl_mesh_cam_render']),
                                  targets['smpl_mesh_cam'].cuda(), is_valid_fit[:, None, None])
    # Loss chieu 2D: phep chieu phoi canh cua model (focal/princpt trong cfg.DATASET), toa do he heatmap
    loss_body_joint_proj = trainer.jotr_coord_loss(
        _proj_for_loss(trainer.J_regressor, model_output),
        targets['orig_joint_img'].cuda()[:, :, :2],
        meta['orig_joint_trunc'].cuda()).mean()
    loss_dict = {
        'smpl_joint_cam': loss_smpl_joint_cam,
        'smpl_pose': smpl_pose_loss,
        'smpl_shape': smpl_shape_loss,
        'mesh_loss': mesh_loss,
        'body_joint_proj': loss_body_joint_proj,
    }
    if log_sigma is not None:
        loss_dict['joint_sigma_nll'] = sigma_nll      # [E] cong ngoai AWL
    return loss_dict, pred_pose


def _joint_log_sigma(trainer):
    net = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    return getattr(net.smpl_model, 'joint_log_sigma', None)


def _hard_losses(trainer, model_output, targets, meta, gt_fit_joint_cam, fit_joint_trunc, is_valid_fit):
    """5 loss voi GT cho uoc luong cuoi; neu co tinh chinh lap [D]: cong ca cac buoc truoc voi trong so
    gamma^(so buoc con lai), chuan hoa theo tong trong so (ReFit giam sat moi buoc)."""
    final, _ = _smpl_hard_losses(trainer, model_output, targets, meta, gt_fit_joint_cam, fit_joint_trunc, is_valid_fit)
    hist = model_output.get('refine_history')
    if not hist:
        return final
    g = float(cfg.MODEL.v3_refine_gamma)
    T = len(hist)
    tot = dict(final)
    wsum = 1.0
    for i, out in enumerate(hist):
        w = g ** (T - i)
        li, _ = _smpl_hard_losses(trainer, out, targets, meta, gt_fit_joint_cam, fit_joint_trunc, is_valid_fit)
        for k in tot:
            tot[k] = tot[k] + w * li[k]
        wsum += w
    return {k: v / wsum for k, v in tot.items()}


def _depth_loss(model_output, gt_rootrel, valid_joint):
    """[C] CE mem cho phan bo do sau tung khop: muc tieu Gauss quanh do sau GT (sigma = 1 bin). Tra (loss, mm)."""
    logits = model_output['depth_logits'].float()                                   # (B, J, bins)
    bins, R = logits.shape[-1], float(cfg.MODEL.v3_depth_range)
    centers = torch.linspace(-R, R, bins, device=logits.device)
    z = gt_rootrel[..., 2].float().clamp(-R, R)
    width = 2 * R / (bins - 1)
    target = torch.softmax(-((centers[None, None] - z[..., None]) / width) ** 2 / 2, -1)
    ce = -(target * torch.log_softmax(logits, -1)).sum(-1)                         # (B, J)
    v = valid_joint.float()
    loss = (ce * v).sum() / v.sum().clamp_min(1.0)
    with torch.no_grad():
        err = ((model_output['depth_exp'].float() - z).abs() * v).sum() / v.sum().clamp_min(1.0) * 1000
    return loss, float(err)


def _noisy_2d(pose2d, mask):
    """[E] Nhieu 2D luc train Student (tren ti le v3_pose2d_noise_prob mau): jitter cac khop hop le
    (std v3_pose2d_noise_std, toa do [-1,1]) + mat khop ngau nhien (v3_pose2d_drop_prob, khong mat pelvis)."""
    p = float(cfg.MODEL.v3_pose2d_noise_prob)
    if p <= 0:
        return pose2d, mask
    B, J = pose2d.shape[:2]
    sel = (torch.rand(B, 1, 1, device=pose2d.device) < p).float()
    m = mask.reshape(B, J, 1)
    valid = (m > 0).float()
    xy = pose2d[..., :2] + torch.randn_like(pose2d[..., :2]) * float(cfg.MODEL.v3_pose2d_noise_std) * valid * sel
    drop = (torch.rand(B, J, 1, device=pose2d.device) < float(cfg.MODEL.v3_pose2d_drop_prob)).float() * sel
    drop[:, 0] = 0
    m = m * (1 - drop)
    out = pose2d.clone()
    out[..., :2] = xy * (m > 0).float()
    if out.shape[-1] >= 3:
        out[..., 2:3] = out[..., 2:3] * (m > 0).float()
    return out, m.reshape(mask.shape)


def _apply_awl(trainer, loss_dict):
    """Trong so shape (shape_param_loss_w) roi AWL. Tra (tong loss, dict loss da qua AWL)."""
    loss_dict = dict(loss_dict)
    sigma_nll = loss_dict.pop('joint_sigma_nll', None)   # [E] khong qua AWL (co the am)
    if trainer.shape_loss_w <= 0:
        loss_dict.pop('smpl_shape')   # bo han khoi AWL (giu voi gia tri 0 -> trong so AWL ve vo cung -> NaN)
    elif trainer.shape_loss_w != 1.0:
        loss_dict['smpl_shape'] = loss_dict['smpl_shape'] * trainer.shape_loss_w
    for k, v in loss_dict.items():   # AWL chi dung voi loss >= 0 (xem experiment.md muc 38)
        if float(v.detach()) < 0:
            raise RuntimeError(f'Loss {k} = {float(v):.4f} < 0 dua vao AWL -> trong so AWL se no')
    loss_dict = trainer.awl(loss_dict)
    total = sum(loss_dict.values())
    if sigma_nll is not None:
        total = total + sigma_nll
    return total, loss_dict


def _log_awl(trainer):
    """In / log trong so hieu dung cua AWL (0.5 / p^2) theo tung loss."""
    if not hasattr(trainer.awl, 'keys'):
        return
    awl_w = {k: (0.5 / (p.detach() ** 2)).item() for k, p in zip(trainer.awl.keys, trainer.awl.params)}
    print('  AWL trong so: ' + ' | '.join(f'{k}: {v:.3f}' for k, v in awl_w.items()))
    bad = {k: v for k, v in awl_w.items() if v > 10.0}
    if bad:   # binh thuong ~0.3-3; > 10 = dau hieu loss dua vao AWL bi am / sai thang (su co 2026-10-26)
        print(f'  [CANH BAO] trong so AWL bat thuong {bad} -> kiem tra loss tuong ung (phai >= 0)')
    if cfg.TRAIN.wandb:
        wandb.log({f'awl_w/{k}': v for k, v in awl_w.items()})


def _init_common(trainer, args, load_dir):
    """Phan khoi tao chung Teacher / Student: du lieu, model, loss, AWL, trong so shape."""
    trainer.batch_generator, trainer.dataset_list, trainer.model, trainer.loss, trainer.optimizer, \
        trainer.lr_scheduler, trainer.loss_history, trainer.error_history = \
        prepare_network(args, load_dir=load_dir, is_train=True)
    trainer.main_dataset = trainer.dataset_list[0]
    trainer.print_freq = cfg.TRAIN.print_freq
    # H36M-17 regressor (17, 6890): GT 3D / 2D va dataset.evaluate cung dung regressor nay
    trainer.J_regressor = torch.Tensor(trainer.main_dataset.joint_regressor_human36).cuda()
    trainer.jotr_coord_loss = JOTRCoordLoss()
    trainer.jotr_param_loss = JOTRParamLoss()
    trainer.coordLoss = CoordLoss(has_valid=True)
    # Loss tham so shape (beta): beta GT 3DPW thuoc SMPL theo GIOI TINH, model dung SMPL NEUTRAL -> co the
    # day sai kich thuoc co the. 1.0 = binh thuong; 0 = bo han (shape chi hoc qua loss joint/mesh).
    trainer.shape_loss_w = float(cfg.MODEL.get('shape_param_loss_w', 1.0))
    if trainer.shape_loss_w != 1.0:
        print(f'===> loss smpl_shape: trong so = {trainer.shape_loss_w}'
              + (' (TAT - khong dua vao AWL)' if trainer.shape_loss_w <= 0 else ''))


def _init_wandb():
    if cfg.TRAIN.wandb:
        wandb.init(config=cfg, project=cfg.MODEL.name, name='ARTS/' + cfg.output_dir.split('/')[-1],
                   dir=cfg.output_dir, job_type="training", reinit=True)


class Teacher_Trainer:
    def __init__(self, args, load_dir):
        _init_common(self, args, load_dir)
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.awl = AutomaticWeightedLoss(5).cuda()  # 5 loss: joint_cam, pose, shape, mesh, body_joint_proj
        self.optimizer.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})

        # ---- Ep Teacher phai dung anh: che (dat 0) mot phan khop GT luc train ----
        # Moi loss van so voi GT SACH -> model buoc phai doc anh de doan lai khop bi che.
        self.joint_corrupt_prob = float(cfg.MODEL.get('teacher_joint_corrupt_prob', 0.0))
        self.joint_corrupt_warmup_epochs = int(cfg.MODEL.get('teacher_joint_corrupt_warmup_epochs', 0))
        if self.joint_corrupt_prob > 0:
            print(f'===> Teacher che khop: prob={self.joint_corrupt_prob}, warmup={self.joint_corrupt_warmup_epochs} epoch')
        # ---- Head tham do anh: doan joint 3D chi tu token anh ----
        self.img_probe = bool(cfg.MODEL.get('img_probe', False))
        self.img_probe_w = float(cfg.MODEL.get('img_probe_w', 0.1))
        if self.img_probe:
            print(f'===> Teacher head tham do anh BAT: weight={self.img_probe_w}')

        # Resume: nap AWL + optimizer (sau khi da them nhom AWL vao optimizer)
        if getattr(args, 'resume_training', False) and load_dir:
            _restore_resume_state(self)
        _init_wandb()

    def _corrupt_joints(self, joints3d, epoch):
        """joints3d: (B,17,3) GT root-relative, met. Tra BAN SAO da che (dat 0) mot phan khop
        (khong che root vi root = 0 san). Chi dung lam INPUT; moi loss van so voi GT sach."""
        if self.joint_corrupt_prob <= 0:
            return joints3d
        B, J, _ = joints3d.shape
        prob = self.joint_corrupt_prob
        if self.joint_corrupt_warmup_epochs > 0:
            prob = prob * min(1.0, float(epoch) / self.joint_corrupt_warmup_epochs)
        if prob <= 0:
            return joints3d
        mask = (torch.rand(B, J, 1, device=joints3d.device) < prob).float()
        mask[:, 0, :] = 0.0
        return joints3d * (1.0 - mask)

    def train(self, epoch):
        self.model.train()
        lr_check(self.optimizer, epoch)
        running_loss = 0.0
        run_extra = {}   # loss / MPJPE head tham do anh, tich luy theo epoch
        batch_generator = tqdm(self.batch_generator)
        for i, (inputs, targets, meta) in enumerate(batch_generator):
            input_image = inputs['img'].cuda().float()
            gt_orig_joint_cam = targets['orig_joint_cam'].cuda()     # H36M-17 tu mesh GT, met, root-relative
            gt_fit_joint_cam = targets['fit_joint_cam'].cuda()
            fit_joint_trunc = meta['fit_joint_trunc'].cuda()
            is_valid_fit = meta['is_valid_fit'].cuda()

            teacher_gt_pose3d = gt_orig_joint_cam - gt_orig_joint_cam[:, 0:1, :]
            teacher_model_input = self._corrupt_joints(teacher_gt_pose3d, epoch)
            # PE vi tri khop lay tu 2D Halpe (nhieu), cung nguon voi Student
            model_output = self.model(input_image, teacher_model_input, is_train=True,
                                      pose_2d=inputs['joints'].cuda().float(),
                                      joints_mask=inputs['joints_mask'].cuda().float(),
                                      cam_bbox=inputs['cam_bbox'].cuda().float() if 'cam_bbox' in inputs else None)

            loss_terms = _hard_losses(self, model_output, targets, meta, gt_fit_joint_cam, fit_joint_trunc,
                                      is_valid_fit)
            loss, _ = _apply_awl(self, loss_terms)

            extra_stats = {}
            if 'depth_logits' in model_output:
                vj = (fit_joint_trunc.squeeze(-1) if fit_joint_trunc.dim() == 3 else fit_joint_trunc) * is_valid_fit[:, None]
                d_loss, d_mm = _depth_loss(model_output, teacher_gt_pose3d, vj)
                loss = loss + float(cfg.MODEL.v3_depth_loss_w) * d_loss
                extra_stats['depth_loss'] = d_loss.item()
                extra_stats['depth_err_mm'] = d_mm
            if self.img_probe:
                valid_joint = fit_joint_trunc.squeeze(-1) if fit_joint_trunc.dim() == 3 else fit_joint_trunc
                valid_joint = valid_joint * is_valid_fit[:, None]
                n_valid_tot = valid_joint.sum().clamp_min(1.0)
                probe_pred = model_output['img_probe_joints']
                probe_err = F.smooth_l1_loss(probe_pred, teacher_gt_pose3d, reduction='none', beta=0.05).mean(-1)
                probe_loss = (probe_err * valid_joint).sum() / n_valid_tot
                loss = loss + self.img_probe_w * probe_loss
                with torch.no_grad():
                    extra_stats['probe_loss'] = probe_loss.item()
                    extra_stats['probe_mpjpe_mm'] = (((probe_pred - teacher_gt_pose3d).norm(dim=-1) * 1000.0)
                                                     * valid_joint).sum().item() / n_valid_tot.item()
            for k_, v_ in extra_stats.items():
                run_extra[k_] = run_extra.get(k_, 0.0) + v_
            if extra_stats and cfg.TRAIN.wandb:
                wandb.log({f'train_loss/{k_}': v_ for k_, v_ in extra_stats.items()})
            if extra_stats and i % self.print_freq == 0:
                batch_generator.set_postfix_str(' '.join(f'{k_}: {v_:.3f}' for k_, v_ in extra_stats.items()))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += float(loss.detach().item())
            if cfg.TRAIN.wandb:
                wandb.log({
                    'train_loss/smpl_joint_cam': loss_terms['smpl_joint_cam'].detach(),
                    'train_loss/smpl_pose': loss_terms['smpl_pose'].detach(),
                    'train_loss/smpl_shape': loss_terms['smpl_shape'].detach(),
                    'train_loss/mesh': loss_terms['mesh_loss'].detach(),
                    'train_loss/body_joint_proj': loss_terms['body_joint_proj'].detach(),
                })
            if i % self.print_freq == 0:
                batch_generator.set_description(
                    f'Epoch{epoch}_({i}/{len(batch_generator)}) => '
                    f'smpl3d: {loss_terms["smpl_joint_cam"].item():.3f} '
                    f'smpl: {(loss_terms["smpl_pose"] + loss_terms["smpl_shape"]).item():.3f} '
                    f'mesh: {loss_terms["mesh_loss"].item():.3f} '
                    f'proj: {loss_terms["body_joint_proj"].item():.3f} '
                    f'tl: {loss.detach().item():.3f}'
                )

        self.loss_history.append(running_loss / len(batch_generator))
        print(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')
        if run_extra:
            print('  ' + ' | '.join(f'{k_}: {v_ / len(batch_generator):.4f}' for k_, v_ in run_extra.items()))
        _log_awl(self)


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


class Tester:
    """Danh gia Teacher / Student tren cfg.DATASET.test_list (dataset.evaluate, 14 khop, mm)."""
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
        for dataset_name, dataset, loader in zip(cfg.DATASET.test_list, self.val_datasets, self.val_loaders):
            metrics = evaluate_3dpw_subset(self.model, dataset, loader)
            results[dataset_name] = metrics
            print(f'{dataset_name}: MPJPE={metrics["mpjpe"]:.2f}, PA-MPJPE={metrics["pa_mpjpe"]:.2f}, '
                  f'MPVPE={metrics["mpvpe"]:.2f}')
        if results:
            self.joint_error, self.surface_error = _select_errors(results)
        return results


Teacher_Tester = Tester
Student_Tester = Tester


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


def validate_spread_gate_checkpoint(model, checkpoint):
    # Real/zero co cung shape tham so, can metadata de tranh nap nham YAML.
    if getattr(model, 'mode', None) != 'student':
        return
    # Khong de loader strict=False bo qua nhanh spread cu roi bao ket qua sai.
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ('model_state_dict', 'state_dict', 'model'):
            if key in checkpoint:
                state = checkpoint[key]
                break
    if isinstance(state, dict) and any(key.startswith(('spread_proj.', 'module.spread_proj.')) for key in state):
        raise ValueError('Checkpoint cong embedding spread cu khong con ho tro; dung source/YAML cua run cu')
    metadata = checkpoint.get('spread_gate_config') if isinstance(checkpoint, dict) else None
    if not getattr(model, 'diff_spread_gate', False):
        if metadata is not None:
            raise ValueError('Checkpoint gate can YAML da bat difflift_spread_gate')
        return
    gate = model.smpl_model.fusion.cfcer.blocks[0].image_gate
    expected = {'zero_std': gate.zero_spread, 'hidden': gate.net[0].out_features}
    if metadata != expected:
        raise ValueError(f'Checkpoint gate/YAML khong khop: checkpoint={metadata}, YAML={expected}')


def load_model_weights(model, ckpt_path, skip_prefixes=('pose_lifter.', 'pmce_lifter.', 'hyp_selector.')):
    """Nap checkpoint Teacher / Student. Bo qua key cua module da bo (LEGACY_KEY_PARTS, lifter cu)."""
    ckpt = torch.load(ckpt_path, map_location='cpu', pickle_module=_PickleShim, weights_only=False)
    validate_spread_gate_checkpoint(model, ckpt)
    state = ckpt
    if isinstance(ckpt, dict):
        for k in ('model_state_dict', 'state_dict', 'model'):
            if k in ckpt:
                state = ckpt[k]
                break

    # 1) bo prefix 'module.' (DataParallel)
    state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
    # 2) ckpt cu luu module nay la 'teacher_model.', code hien tai la 'smpl_model.'
    old_p, new_p = 'teacher_model.', 'smpl_model.'
    state = {(new_p + k[len(old_p):] if k.startswith(old_p) else k): v for k, v in state.items()}

    # 3) loc: bo lifter cu, module kien truc cu, key lech shape
    own = model.state_dict()
    filtered, skipped, legacy, shape_bad = {}, [], [], []
    for k, v in state.items():
        if k.startswith(skip_prefixes):
            skipped.append(k)
        elif k.startswith('smpl_model.') and any(p in k for p in LEGACY_KEY_PARTS):
            legacy.append(k)
        elif k in own and own[k].shape != v.shape:
            shape_bad.append((k, tuple(v.shape), tuple(own[k].shape)))
        else:
            filtered[k] = v

    # 4) nap
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    missing = [k for k in missing if not k.startswith(skip_prefixes)]

    # 4b) Chuan hoa anh cua backbone (backbone.input_mean / input_std) chi co trong checkpoint moi.
    #     Checkpoint cu (backbone ngau nhien, KHONG chuan hoa) khong co 2 key nay -> dat ve khong chuan hoa.
    norm_keys = ('backbone.input_mean', 'backbone.input_std')
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'set_input_normalization'):
        if not any(k in state for k in norm_keys):
            model.backbone.set_input_normalization(imagenet=False)
            print('[load_model_weights] checkpoint khong co chuan hoa anh cho backbone -> dung anh [0,1] nhu luc train cu')
        else:
            print(f'[load_model_weights] backbone dung chuan hoa tu checkpoint: '
                  f'mean={model.backbone.input_mean.flatten().tolist()}')
    missing = [k for k in missing if k not in norm_keys]

    print(f'[load_model_weights] {ckpt_path}: nap {len(filtered)} tensor, bo {len(skipped)} (lifter cu) + '
          f'{len(legacy)} (module kien truc cu)')
    print(f'[load_model_weights] missing={len(missing)}, unexpected={len(unexpected)}, shape_mismatch={len(shape_bad)}')
    if shape_bad:
        print('  shape mismatch (5 dau):', shape_bad[:5])
    if missing:
        print('  MISSING   (10 dau):', sorted(missing)[:10])
    if unexpected:
        print('  UNEXPECTED(10 dau):', sorted(unexpected)[:10])

    # 5) dung ngay neu nap chua du
    assert not shape_bad, f'Co key lech shape: {shape_bad[:3]}'
    missing_core = [k for k in missing if k.startswith(('smpl_model.', 'backbone.'))]
    hint = ''
    if any('fusion.projector_student.' in k for k in missing_core):
        hint = ' -> Model la STUDENT nhung checkpoint khong co encoder cua Student (co the la checkpoint TEACHER).'
    elif any('fusion.joint_proj.' in k or 'fusion.norm_joint_proj.' in k for k in missing_core):
        hint = ' -> Model la TEACHER nhung checkpoint khong co encoder cua Teacher (co the la checkpoint STUDENT).'
    assert not missing_core, \
        f'Checkpoint {ckpt_path} chua nap du smpl_model/backbone{hint} Thieu (5 dau): {missing_core[:5]}'
    return model


class Student_Trainer:
    def __init__(self, args, load_dir):
        _init_common(self, args, load_dir)
        # student_use_kd = False -> baseline: KHONG dung Teacher (khong KD, khong khoi tao tu Teacher),
        # chi train bang hard loss + privileged loss (so voi GT 3D).
        self.use_kd = bool(cfg.MODEL.get('student_use_kd', True))
        resume = bool(getattr(args, 'resume_training', False))

        if self.use_kd:
            teacher_ckpt = cfg.MODEL.get('TEACHER', '')
            assert teacher_ckpt, 'Can dat cfg.MODEL.TEACHER (checkpoint cua Teacher_Trainer)'
            from models.ARTS import ARTS
            old_mode = cfg.MODEL.name
            cfg.MODEL.name = 'teacher'
            self.teacher = ARTS(num_joint=17, embed_dim=cfg.MODEL.hpe_dim)
            cfg.MODEL.name = old_mode
            load_model_weights(self.teacher, teacher_ckpt)

            # Student va Teacher cung kien truc (chi khac bo ma hoa joint, gate) -> khoi tao Student tu Teacher.
            if not resume and cfg.MODEL.get('init_student_from_teacher', True):
                missing, unexpected = self.model.smpl_model.load_state_dict(
                    self.teacher.smpl_model.state_dict(), strict=False)
                # Chi cho phep khac bo ma hoa joint, gate va head tham do anh cua Teacher.
                bad_missing = [k for k in missing if not k.startswith('fusion.projector_student.')
                               and not (k.startswith('fusion.cfcer.blocks.') and '.image_gate.' in k)]
                bad_unexpected = [k for k in unexpected
                                  if not k.startswith(('fusion.joint_proj.', 'fusion.norm_joint_proj.', 'probe_'))]
                if bad_missing or bad_unexpected:
                    raise RuntimeError(f'Khoi tao Student tu Teacher sai kien truc: '
                                       f'missing={bad_missing[:5]}, unexpected={bad_unexpected[:5]}')
                print(f'===> Student smpl_model khoi tao tu Teacher | missing={len(missing)} '
                      f'(joint encoder/gate Student), unexpected={len(unexpected)} (joint encoder/probe Teacher)')
                if cfg.MODEL.v3_bb_finetune or cfg.MODEL.v3_bb_heatmap:
                    # [A] Teacher da fine-tune backbone / heat_fuse -> smpl_model cua Teacher quen feature cua
                    # backbone DO; Student phai nhan dung backbone nay, khong giu SPIN goc.
                    self.model.backbone.load_state_dict(self.teacher.backbone.state_dict(), strict=True)
                    print('===> [A] Student backbone (+ heat_fuse) khoi tao tu Teacher')

            self.teacher = torch.nn.DataParallel(self.teacher).cuda()
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
        else:
            self.teacher = None
            print('===> Student KHONG Teacher: khong KD, khong khoi tao tu Teacher (loss = kd_hard_w*hard + privileged)')

        # ---------------- KD feature ----------------
        # loss = kd_hard_w*hard + kd_lambda*KD_feat + privileged
        # KD o 3 diem noi: proj (token joint sau encoder), joint (token joint sau fusion), global (feature toan cuc).
        self.kd_feat_dist = cfg.MODEL.get('kd_feat_dist', 'mse')            # 'mse' (qua projector) | 'cos'
        self.kd_lambda = float(cfg.MODEL.get('kd_lambda', 1.0))
        self.kd_hard_w = float(cfg.MODEL.get('kd_hard_w', 1.0))
        self.w_proj = float(cfg.MODEL.get('kd_w_proj', 1.0))
        self.w_joint = float(cfg.MODEL.get('kd_w_joint', 1.0))
        self.w_global = float(cfg.MODEL.get('kd_w_global', 1.0))
        self.kd_joint_relation = bool(cfg.MODEL.get('kd_joint_relation', False))
        self.kd_joint_relation_log_grad = bool(cfg.MODEL.get('kd_joint_relation_log_grad', False))
        if self.kd_joint_relation:
            if not self.use_kd:
                raise ValueError('kd_joint_relation can student_use_kd=True')
            if self.w_joint <= 0:
                raise ValueError('kd_joint_relation can kd_w_joint > 0')
        self.feat_projector = None   # tao luc batch dau tien (biet so kenh); train.py tu luu 'projector_state_dict'
        if self.use_kd:
            print(f'===> KD feat | dist={self.kd_feat_dist} | lambda={self.kd_lambda} | hard_w={self.kd_hard_w} | '
                  f'w_proj={self.w_proj} w_joint={self.w_joint} w_global={self.w_global}')
            if self.kd_joint_relation:
                print('     KD joint = relation: MSE cosine(LN(token)), bo duong cheo/cap khop GT khong hop le; '
                      'khong qua projector joint, global KD giu nguyen')
                print(f'     Log gradient tai fusion (batch dau moi epoch): {self.kd_joint_relation_log_grad}')

        self.model = torch.nn.DataParallel(self.model).cuda()
        self.awl = AutomaticWeightedLoss(5).cuda()
        self.optimizer.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})

        # Resume: nap AWL + projector KD + optimizer (sau khi da them nhom AWL; projector tao lai tu checkpoint)
        if resume and load_dir:
            _restore_resume_state(self, student=True)
        _init_wandb()

    # ---------------- KD feature ----------------
    def _ensure_projectors(self, s_out):
        """Tao projector (LN khong affine -> Linear khoi tao identity) cho moi diem noi, lan dau goi."""
        if self.feat_projector is not None:
            return
        mods = {}
        for key, out_key in (('proj', 'feat_joint_in'), ('joint', 'feat_joint'), ('global', 'feat_global')):
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

    def _kd_joint_relation(self, s, t, valid_joint):
        """MSE quan he token trong cung mau; tra loss va tong trong so cap, shape (B,).
        Mask la GT nhu loss joint cu, khong phai confidence detector. Ca i->j va j->i
        deu tinh; chia dung so cap nen tuong duong trung binh tam giac tren.
        """
        if s.ndim != 3 or t.ndim != 3 or s.shape[:2] != t.shape[:2]:
            raise ValueError('KD relation can token Student/Teacher (B,J,C) voi cung B,J')
        if valid_joint.shape != s.shape[:2]:
            raise ValueError('Mask KD relation phai co shape (B,J)')
        # Tinh FP32 de on dinh LN/cosine; ep kieu van giu gradient Student.
        rs = joint_relation_matrix(s)
        with torch.no_grad():
            rt = joint_relation_matrix(t.detach())
            mask = valid_joint.detach().to(device=s.device, dtype=torch.float32)
            off_diag = 1.0 - torch.eye(s.shape[1], device=s.device, dtype=torch.float32)
            pair_mask = mask[:, :, None] * mask[:, None, :] * off_diag
            pair_count = pair_mask.sum(dim=(1, 2))
        # Mau co duoi 2 khop hop le: tu so=0, loss=0, khong chia cho 0.
        per_sample = (((rs - rt) ** 2) * pair_mask).sum(dim=(1, 2)) / pair_count.clamp_min(1.0)
        return per_sample, pair_count

    def train(self, epoch):
        self.model.train()
        if self.teacher is not None:
            self.teacher.eval()
        lr_check(self.optimizer, epoch)
        # beta cua smooth L1 cho privileged loss (don vi met)
        priv_beta = cfg.MODEL.get('priv_beta', 0.05)

        running_loss = 0.0
        running = {}   # tich luy cac so hang de in / log theo epoch
        batch_generator = tqdm(self.batch_generator)
        for i, (inputs, targets, meta) in enumerate(batch_generator):
            input_image = inputs['img'].cuda().float()
            input_pose2d = inputs['joints'].cuda().float()
            joints_mask = inputs['joints_mask'].cuda().float()
            input_pose2d, joints_mask = _noisy_2d(input_pose2d, joints_mask)       # [E] (0 = tat)
            gt_orig_joint_cam = targets['orig_joint_cam'].cuda()
            gt_fit_joint_cam = targets['fit_joint_cam'].cuda()
            fit_joint_trunc = meta['fit_joint_trunc'].cuda()
            is_valid_fit = meta['is_valid_fit'].cuda()
            gt_pose_input = gt_orig_joint_cam - gt_orig_joint_cam[:, 0:1, :]
            cam_bbox = inputs['cam_bbox'].cuda().float()

            model_output = self.model(input_image, input_pose2d, is_train=True, joints_mask=joints_mask,
                                      cam_bbox=cam_bbox)

            # ---------- mask khop hop le ----------
            valid_joint = fit_joint_trunc.squeeze(-1) if fit_joint_trunc.dim() == 3 else fit_joint_trunc
            valid_joint = valid_joint * is_valid_fit[:, None]
            n_valid = valid_joint.sum(-1).clamp_min(1.0)       # (B,); mau khong co khop hop le -> loss = 0

            # loi joint dau vao Student (DiffLifter) so voi GT, mm
            with torch.no_grad():
                _lerr = torch.norm(model_output['lifted_joints_3d'] - gt_pose_input, dim=-1)
                lift_err_mm = float(((_lerr * valid_joint).sum() / valid_joint.sum().clamp_min(1.0)).item() * 1000)

            # ---------- KD feature (Teacher nhan GT 3D, khong grad) ----------
            zeros_b = torch.zeros(input_image.shape[0], device=input_image.device)
            kd_relation_pairs = zeros_b
            if self.use_kd:
                with torch.no_grad():
                    t_out = self.teacher(input_image, gt_pose_input, is_train=False,
                                         pose_2d=input_pose2d, joints_mask=joints_mask, cam_bbox=cam_bbox)
                self._ensure_projectors(model_output)
                sw = valid_joint.sum(-1) / n_valid                                             # (B,) 1 | 0
                kd_proj_ps = (self._kd_dist('proj', model_output['feat_joint_in'], t_out['feat_joint_in'])
                              * valid_joint).sum(-1) / n_valid
                if self.kd_joint_relation:
                    kd_joint_ps, kd_relation_pairs = self._kd_joint_relation(
                        model_output['feat_joint'], t_out['feat_joint'], valid_joint)
                else:
                    kd_joint_ps = (self._kd_dist('joint', model_output['feat_joint'], t_out['feat_joint'])
                                   * valid_joint).sum(-1) / n_valid
                kd_global_ps = self._kd_dist('global', model_output['feat_global'], t_out['feat_global']) * sw
            else:
                kd_proj_ps = kd_joint_ps = kd_global_ps = zeros_b
            kd_loss = (self.w_proj * kd_proj_ps + self.w_joint * kd_joint_ps + self.w_global * kd_global_ps).mean()

            # ---------- privileged reconstruction (Student -> GT 3D) ----------
            privileged_error = F.smooth_l1_loss(
                model_output['privileged_3d'], gt_pose_input, reduction='none', beta=priv_beta).mean(-1)
            privileged_loss = ((privileged_error * valid_joint).sum(-1) / n_valid).mean()

            # ---------- hard loss ----------
            loss_terms = _hard_losses(self, model_output, targets, meta, gt_fit_joint_cam, fit_joint_trunc,
                                      is_valid_fit)
            hard_loss, _ = _apply_awl(self, loss_terms)
            loss = self.kd_hard_w * hard_loss + self.kd_lambda * kd_loss + privileged_loss
            depth_stats = {}
            if 'depth_logits' in model_output:
                d_loss, d_mm = _depth_loss(model_output, gt_pose_input, valid_joint)
                loss = loss + float(cfg.MODEL.v3_depth_loss_w) * d_loss
                depth_stats = {'depth_loss': d_loss.item(), 'depth_err_mm': d_mm}

            # Do gradient tai tham so fusion (KD relation): feature gather cua DataParallel la cac nhanh rieng,
            # khong dung grad theo feature joint gather de tranh log nham bang 0.
            relation_grad_stats = {}
            if self.kd_joint_relation and self.kd_joint_relation_log_grad and i == 0:
                grad_terms = {
                    'kd_rel_grad_joint': self.kd_lambda * self.w_joint * kd_joint_ps.mean(),
                    'kd_rel_grad_global': self.kd_lambda * self.w_global * kd_global_ps.mean(),
                    'kd_rel_grad_hard': self.kd_hard_w * hard_loss,
                }
                fusion_params = tuple(p for p in self.model.module.smpl_model.fusion.parameters() if p.requires_grad)
                if not fusion_params:
                    raise RuntimeError('Log gradient KD relation can tham so fusion dang train')
                for name, term in grad_terms.items():
                    grads = (torch.autograd.grad(term, fusion_params, retain_graph=True, allow_unused=True)
                             if term.requires_grad else ())
                    norms = [grad.detach().float().norm() for grad in grads if grad is not None]
                    relation_grad_stats[name] = torch.stack(norms).norm().item() if norms else 0.0
                    del grads, norms
                print(f'\nKD relation grad@fusion (da nhan trong so): '
                      f'J={relation_grad_stats["kd_rel_grad_joint"]:.3e} '
                      f'G={relation_grad_stats["kd_rel_grad_global"]:.3e} '
                      f'hard={relation_grad_stats["kd_rel_grad_hard"]:.3e}; '
                      f'cap/mau={kd_relation_pairs.mean().item():.1f}')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += float(loss.detach().item())

            step_stats = {
                'kd_proj': kd_proj_ps.mean().item(),
                'kd_joint': kd_joint_ps.mean().item(),
                'kd_global': kd_global_ps.mean().item(),
                'privileged_3d': privileged_loss.item(),
                'lift_err_mm': lift_err_mm,
                **depth_stats,
            }
            if self.kd_joint_relation:
                step_stats['kd_joint_relation'] = step_stats['kd_joint']
                step_stats['kd_relation_pairs'] = kd_relation_pairs.mean().item()
            gates = [layer['spread_gate'].detach() for layer in model_output.get('feat_layers', [])
                     if 'spread_gate' in layer]
            if gates:
                gate_values = torch.stack(gates)
                step_stats.update({'gate_mean': gate_values.mean().item(),
                                   'gate_std': gate_values.std(unbiased=False).item(),
                                   'gate_min': gate_values.min().item(), 'gate_max': gate_values.max().item()})
            for k, v in step_stats.items():
                running[k] = running.get(k, 0.0) + v
            if cfg.TRAIN.wandb:
                wandb.log({
                    'train_loss/smpl_joint_cam': loss_terms['smpl_joint_cam'].item(),
                    'train_loss/smpl_pose': loss_terms['smpl_pose'].item(),
                    'train_loss/smpl_shape': loss_terms['smpl_shape'].item(),
                    'train_loss/mesh': loss_terms['mesh_loss'].item(),
                    'train_loss/body_joint_proj': loss_terms['body_joint_proj'].item(),
                    'train_loss/kd_feat': kd_loss.item(),
                    'train_loss/hard_total': hard_loss.item(),
                    'train_loss/total': loss.item(),
                    **{f'train_loss/{k}': v for k, v in step_stats.items()},
                    **{f'train_loss/{k}': v for k, v in relation_grad_stats.items()},
                })

            if i % self.print_freq == 0:
                gate_text = f'g: {step_stats["gate_mean"]:.3f}+/-{step_stats["gate_std"]:.3f} ' if gates else ''
                batch_generator.set_description(
                    f'Epoch{epoch}_({i}/{len(batch_generator)}) => '
                    f'smpl3d: {loss_terms["smpl_joint_cam"].item():.3f} '
                    f'smpl: {(loss_terms["smpl_pose"] + loss_terms["smpl_shape"]).item():.3f} '
                    f'mesh: {loss_terms["mesh_loss"].item():.3f} '
                    f'proj2d: {loss_terms["body_joint_proj"].item():.3f} '
                    f'kd: {kd_loss.item():.3f} '
                    f'[P {step_stats["kd_proj"]:.3f} J {step_stats["kd_joint"]:.3f} G {step_stats["kd_global"]:.3f}] '
                    f'{gate_text}'
                    f'lift: {lift_err_mm:.0f}mm '
                    + (f'depth: {depth_stats["depth_err_mm"]:.0f}mm ' if depth_stats else '') +
                    f'priv: {step_stats["privileged_3d"]:.3f} '
                    f'tl: {loss.detach().item():.3f}'
                )

        n_batch = len(batch_generator)
        self.loss_history.append(running_loss / n_batch)
        avg = {k: v / n_batch for k, v in running.items()}
        print(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f} | '
              f'KD dist proj: {avg["kd_proj"]:.4f} | joint: {avg["kd_joint"]:.4f} | global: {avg["kd_global"]:.4f}')
        print(f'  Privileged 3D: {avg["privileged_3d"]:.4f} | loi joint dau vao (DiffLifter): {avg["lift_err_mm"]:.1f} mm'
              + (f' | do sau [C]: loss {avg["depth_loss"]:.4f}, loi {avg["depth_err_mm"]:.1f} mm' if 'depth_loss' in avg else ''))
        if self.kd_joint_relation:
            print(f'  KD joint RELATION (thay MSE feature): {avg["kd_joint_relation"]:.6f} | '
                  f'cap hop le/mau: {avg["kd_relation_pairs"]:.1f}; global KD giu nguyen')
        if 'gate_mean' in avg:
            print(f'  Spread gate: mean={avg["gate_mean"]:.4f} | std={avg["gate_std"]:.4f} | '
                  f'min/max TB batch={avg["gate_min"]:.4f}/{avg["gate_max"]:.4f} | '
                  f'zero_std={cfg.MODEL.get("difflift_spread_gate_zero", False)}')
        _log_awl(self)
        if cfg.TRAIN.wandb:
            wandb.log({f'epoch_metric/{k}': v for k, v in avg.items()})
