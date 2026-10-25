import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from geometry import batch_rodrigues


class JOTRCoordLoss(nn.Module):
    def forward(self, coord_out, coord_gt, valid, is_3d=None):
        loss = F.l1_loss(coord_out, coord_gt, reduction='none') * valid
        if is_3d is not None:
            loss[:, :, 2:] = loss[:, :, 2:] * is_3d[:, None, None].float()
        return loss


class JOTRParamLoss(nn.Module):
    def forward(self, param_out, param_gt, valid):
        return F.l1_loss(param_out, param_gt, reduction='none') * valid

class CoordLoss(nn.Module):
    def __init__(self, has_valid=False):
        super(CoordLoss, self).__init__()

        self.has_valid = has_valid
        self.criterion = nn.L1Loss(reduction='mean')

    def forward(self, pred, target, target_valid):
        if self.has_valid:
            pred, target = pred * target_valid, target * target_valid

        loss = self.criterion(pred, target)

        return loss

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape) - 1), axis=1)


def acc_error(predicted, target):
    """
    Calculates acceleration error:
         1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    """
    accel_gt = target[:-2] - 2 * target[1:-1] + target[2:]
    accel_pred = predicted[:-2] - 2 * predicted[1:-1] + predicted[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    return np.mean(normed, axis=1)


def jpe(predicted, target):
    """
    per-joint position error
    """
    assert predicted.shape == target.shape
    return np.linalg.norm(predicted - target, axis=len(target.shape) - 1)


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation
    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=1)


# PyTorch-based errors (for losses)

def loss_mpjpe(predicted, target): # b t j 3
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

def loss_2d_mpjpe(predicted, target):
    #b t j 2
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted-target,dim=len(target.shape)-1))


def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape) - 1))


def loss_2d_weighted(predicted, target, conf):
    assert predicted.shape == target.shape
    predicted_2d = predicted[:, :, :, :2]
    target_2d = target[:, :, :, :2]
    diff = (predicted_2d - target_2d) * conf
    return torch.mean(torch.norm(diff, dim=-1))


def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    norm_predicted = torch.mean(torch.sum(predicted ** 2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target * predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return loss_mpjpe(scale * predicted, target)


def weighted_bonelen_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.001 * torch.pow(predict_3d_length - gt_3d_length, 2).mean()
    return loss_length


def weighted_boneratio_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.1 * torch.pow((predict_3d_length - gt_3d_length) / gt_3d_length, 2).mean()
    return loss_length


def get_limb_lens(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = [
        [0, 1], [0, 2], [1, 3], [2, 4],     # Mặt
        [5, 6], [5, 7], [7, 9],             # Tay trái
        [6, 8], [8, 10],                    # Tay phải
        [5, 11], [6, 12], [11, 12],         # Thân
        [11, 13], [13, 15],                 # Chân trái
        [12, 14], [14, 16]                  # Chân phải
    ]
    limbs = x[:, :, limbs_id, :]
    limbs = limbs[:, :, :, 0, :] - limbs[:, :, :, 1, :]
    limb_lens = torch.norm(limbs, dim=-1)
    return limb_lens


def loss_limb_var(x):
    '''
        Input: (N, T, 17, 3)
    '''
    if x.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    limb_lens = get_limb_lens(x)
    limb_lens_var = torch.var(limb_lens, dim=1)
    limb_loss_var = torch.mean(limb_lens_var)
    return limb_loss_var


def loss_limb_gt(x, gt):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_lens_x = get_limb_lens(x)
    limb_lens_gt = get_limb_lens(gt)  # (N, T, 16)
    return nn.L1Loss()(limb_lens_x, limb_lens_gt)


def loss_velocity(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    if predicted.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    velocity_predicted = predicted[:, 1:] - predicted[:, :-1]
    velocity_target = target[:, 1:] - target[:, :-1]
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))


def loss_joint(predicted, target):
    assert predicted.shape == target.shape
    return nn.L1Loss()(predicted, target)


def get_angles(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 16)
    '''
    limbs_id = [
        [5, 7],   # 0: Vai trái -> Khuỷu tay trái
        [7, 9],   # 1: Khuỷu tay trái -> Cổ tay trái
        [6, 8],   # 2: Vai phải -> Khuỷu tay phải
        [8, 10],  # 3: Khuỷu tay phải -> Cổ tay phải
        [11, 13], # 4: Háng trái -> Đầu gối trái
        [13, 15], # 5: Đầu gối trái -> Cổ chân trái
        [12, 14], # 6: Háng phải -> Đầu gối phải
        [14, 16], # 7: Đầu gối phải -> Cổ chân phải
        [5, 11],  # 8: Vai trái -> Háng trái
        [6, 12],  # 9: Vai phải -> Háng phải
        [5, 6],   # 10: Vai trái -> Vai phải
        [11, 12]  # 11: Háng trái -> Háng phải
    ]
    angle_id = [
        [0, 1],   # Góc khuỷu tay trái (giữa xương 0 và 1)
        [2, 3],   # Góc khuỷu tay phải (giữa xương 2 và 3)
        [4, 5],   # Góc đầu gối trái (giữa xương 4 và 5)
        [6, 7],   # Góc đầu gối phải (giữa xương 6 và 7)
        [0, 8],   # Góc nách trái (Vai-Khuỷu và Vai-Háng)
        [2, 9],   # Góc nách phải
        [4, 11],  # Góc háng trái
        [6, 11]   # Góc háng phải
    ]
    eps = 1e-7
    limbs = x[:, :, limbs_id, :]
    limbs = limbs[:, :, :, 0, :] - limbs[:, :, :, 1, :]
    angles = limbs[:, :, angle_id, :]
    angle_cos = F.cosine_similarity(angles[:, :, :, 0, :], angles[:, :, :, 1, :], dim=-1)
    return torch.acos(angle_cos.clamp(-1 + eps, 1 - eps))


def loss_angle(x, gt):
    '''
        Input: (N, T, 17, 3), (N, T, 17, 3)
    '''
    limb_angles_x = get_angles(x)
    limb_angles_gt = get_angles(gt)
    return nn.L1Loss()(limb_angles_x, limb_angles_gt)


def loss_angle_velocity(x, gt):
    """
    Mean per-angle velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert x.shape == gt.shape
    if x.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    x_a = get_angles(x)
    gt_a = get_angles(gt)
    x_av = x_a[:, 1:] - x_a[:, :-1]
    gt_av = gt_a[:, 1:] - gt_a[:, :-1]
    return nn.L1Loss()(x_av, gt_av)

def miloss(x,gt):
    assert x.shape == gt.shape

    x_log = F.log_softmax(x,dim=-1)
    y = F.softmax(gt,dim=-1)
    kl = nn.KLDivLoss(reduction='batchmean')
    out = kl(x_log,y)
    return out

def pck(pred, gt, threshold=150.0):
    """Percentage of Correct Keypoints at threshold (e.g., 150mm)"""
    assert pred.shape == gt.shape
    error = np.linalg.norm(pred - gt, axis=-1)  # (N, J)
    correct = (error < threshold).astype(np.float32)
    return 100.0 * np.mean(correct)

def auc(pred, gt, max_threshold=150.0, num_steps=31):
    """Compute AUC for PCK curve from 0 to max_threshold"""
    thresholds = np.linspace(0, max_threshold, num_steps)
    pck_list = [pck(pred, gt, t) for t in thresholds]
    return np.trapz(pck_list, thresholds) / max_threshold  # normalize
class LaplacianLoss(nn.Module):
    def __init__(self, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = 6890  # SMPL
        self.nf = faces.shape[0]
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            laplacian[i, :] /= (laplacian[i, i] + 1e-8)

        self.register_buffer('laplacian', torch.from_numpy(laplacian).cuda().float())

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.cat([torch.matmul(self.laplacian, x[i])[None, :, :] for i in range(batch_size)], 0)

        x = x.pow(2).sum(2)
        if self.average:
            return x.sum() / batch_size
        else:
            return x.mean()


class NormalVectorLoss(nn.Module):
    def __init__(self, face):
        super(NormalVectorLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt):
        face = torch.LongTensor(self.face).cuda()

        v1_out = coord_out[:, face[:, 1], :] - coord_out[:, face[:, 0], :]
        v1_out = F.normalize(v1_out, p=2, dim=2)  # L2 normalize to make unit vector
        v2_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 0], :]
        v2_out = F.normalize(v2_out, p=2, dim=2)  # L2 normalize to make unit vector
        v3_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 1], :]
        v3_out = F.normalize(v3_out, p=2, dim=2)  # L2 nroamlize to make unit vector

        v1_gt = coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 0], :]
        v1_gt = F.normalize(v1_gt, p=2, dim=2)  # L2 normalize to make unit vector
        v2_gt = coord_gt[:, face[:, 2], :] - coord_gt[:, face[:, 0], :]
        v2_gt = F.normalize(v2_gt, p=2, dim=2)  # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2)  # L2 normalize to make unit vector

        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))
        loss = torch.cat((cos1, cos2, cos3), 1)
        return loss.mean()


class EdgeLengthLoss(nn.Module):
    def __init__(self, face):
        super(EdgeLengthLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt):
        face = torch.LongTensor(self.face).cuda()

        d1_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 1], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))

        d1_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
 
        diff1 = torch.abs(d1_out - d1_gt)
        diff2 = torch.abs(d2_out - d2_gt)
        diff3 = torch.abs(d3_out - d3_gt)
        loss = torch.cat((diff1, diff2, diff3), 1)
        return loss.mean()
    
class SMPLLoss(nn.Module):
    def __init__(self):
        super(SMPLLoss, self).__init__()
        self.criterion_regr = nn.MSELoss().cuda()

    def forward(self, pred_rotmat, pred_betas, gt_pose, gt_betas, mask_3d=None):
        pred_rotmat_valid = batch_rodrigues(pred_rotmat.reshape(-1,3)).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1,3)).reshape(-1, 24, 3, 3)
        pred_betas_valid = pred_betas
        gt_betas_valid = gt_betas
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
            if mask_3d is None:
                loss_regr_pose = loss_regr_pose.mean()
                loss_regr_betas = loss_regr_betas.mean()
            else:
                loss_regr_pose = (loss_regr_pose * mask_3d.unsqueeze(-1).unsqueeze(-1)).mean()
                loss_regr_betas = (loss_regr_betas * mask_3d).mean()
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).cuda()
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).cuda()
        return loss_regr_pose, loss_regr_betas
    
class PoseLoss(nn.Module):
    def __init__(self):
        super(PoseLoss, self).__init__()
        self.criterion_regr = nn.MSELoss().cuda()

    def forward(self, pred_rotmat, gt_pose, mask_3d=None):
        pred_rotmat_valid = batch_rodrigues(pred_rotmat.reshape(-1,3)).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1,3)).reshape(-1, 24, 3, 3)
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            if mask_3d is None:
                loss_regr_pose = loss_regr_pose.mean()
            else:
                loss_regr_pose = (loss_regr_pose * mask_3d.unsqueeze(-1).unsqueeze(-1)).mean()
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).cuda()
        return loss_regr_pose
    
class LimbLengthError(nn.Module):
    """ Limb Length Loss: to let the """
    def __init__(self):
        super(LimbLengthError, self).__init__()
        self.CONNECTIVITY_DICT = [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 16), (9, 16), (8, 12), (11, 12), (10, 11), (8, 13), (13, 14), (14, 15)]

    def forward(self, keypoints_3d_pred, keypoints_3d_gt):
        # (b, 17, 3)

        error = 0
        for (joint0, joint1) in self.CONNECTIVITY_DICT:
            limb_pred = keypoints_3d_pred[:, joint0] - keypoints_3d_pred[:, joint1]
            limb_gt = keypoints_3d_gt[:, joint0] - keypoints_3d_gt[:, joint1]
            if isinstance(limb_pred, np.ndarray):
                limb_pred = torch.from_numpy(limb_pred)
                limb_gt = torch.from_numpy(limb_gt)
            limb_length_pred = torch.norm(limb_pred, dim = 1)
            limb_length_gt = torch.norm(limb_gt, dim = 1)
            error += torch.abs(limb_length_pred - limb_length_gt).mean().cpu()

        return float(error)/len(self.CONNECTIVITY_DICT)

def get_loss(faces):
    loss = CoordLoss(has_valid=True), NormalVectorLoss(faces), EdgeLengthLoss(faces), \
           CoordLoss(has_valid=True), CoordLoss(has_valid=True), CoordLoss(has_valid=True),\
           SMPLLoss(), PoseLoss()

    return loss

class AutomaticWeightedLoss(nn.Module):
    """Automatically weighted multi-task loss (from JOTR).
    Learns a weight for each loss term to balance them.
    """
    def __init__(self, num=4):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True, dtype=torch.float32)
        self.params = nn.Parameter(params)

    def forward(self, loss_dict):
        if not hasattr(self, 'keys'):
            self.keys = sorted(list(loss_dict.keys()))
        for i, key in enumerate(self.keys):
            loss_dict[key] = 0.5 / (self.params[i] ** 2) * loss_dict[key] + torch.log(1 + self.params[i] ** 2)
        return loss_dict
