"""Loss dung trong Teacher_Trainer / Student_Trainer."""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
