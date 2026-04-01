import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSE(nn.Module):
    def __init__(self):
        super(MaskedMSE, self).__init__()
        self.l2_loss = nn.MSELoss()

    def forward(self, pred, gt, mask=None):
        # 将所有所有的输入都经过squeeze函数
        pred = torch.squeeze(pred)
        gt = torch.squeeze(gt)
        mask = torch.squeeze(mask) if mask is not None else None
        if mask is not None:
            valid_mask = mask == 1
            masked_pred = pred[valid_mask]
            masked_gt = gt[valid_mask]

            loss = self.l2_loss(masked_pred, masked_gt)
        else:
            loss = self.l2_loss(pred, gt)

        return loss


class masked_nll_loss(nn.Module):
    def __init__(self, weight, num_class):
        super(masked_nll_loss, self).__init__()
        self.weight = weight
        self.num_class = num_class

    def forward(self, pred, gt, mask=None):
        pred = torch.squeeze(pred)
        gt = torch.squeeze(gt)
        mask = torch.squeeze(mask) if mask is not None else None
        if mask is not None:
            valid_mask = mask == 1
            masked_pred = pred[valid_mask]
            masked_gt = gt[valid_mask]
            masked_pred = masked_pred.contiguous().view(-1, self.num_class)
            masked_gt = masked_gt.view(-1, 1)[:, 0]
            loss = F.nll_loss(masked_pred, masked_gt, self.weight)

        else:
            pred = pred.contiguous().view(-1, self.num_class)
            gt = gt.view(-1, 1)[:, 0]
            loss = F.nll_loss(pred, gt, self.weight)

        return loss

