import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pre_score, gt_score, train_mask):
        pre_score = pre_score.contiguous().view(pre_score.size()[0], -1)
        gt_score = gt_score.contiguous().view(gt_score.size()[0], -1)
        train_mask = train_mask.contiguous().view(train_mask.size()[0], -1)
        pre_score = pre_score * train_mask
        gt_score = gt_score * train_mask
        a = torch.sum(pre_score * gt_score, 1)
        b = torch.sum(pre_score * pre_score, 1) + self.eps
        c = torch.sum(gt_score * gt_score, 1) + self.eps
        d = (2 * a) / (b + c)
        dice_loss = torch.mean(d)
        return 1 - dice_loss


class BalanceCrossEntropyLoss(nn.Module):
    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                mask: torch.Tensor):
        '''
        Args:
            pred: shape :math:`(N, H, W)`, the prediction of network
            gt: shape :math:`(N, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        '''
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()),
                            int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(
            pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) /\
            (positive_count + negative_count + self.eps)
        return balance_loss


class MaskL1Loss(nn.Module):
    def __init__(self):
        super(MaskL1Loss, self).__init__()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor,
                mask: torch.Tensor):
        mask_sum = mask.sum()
        if mask_sum.item() == 0:
            return mask_sum, dict(l1_loss=mask_sum)
        else:
            loss = (torch.abs(pred - gt) * mask).sum() / mask_sum
            return loss, dict(loss_l1=loss)


class DBLoss(nn.Module):
    def __init__(self, l1_scale=10, bce_scale=1, eps=1e-6):
        super().__init__()
        self.bce_loss = BalanceCrossEntropyLoss()
        self.dice_loss = DiceLoss(eps)
        self.l1_loss = MaskL1Loss()
        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def forward(self, pred_batch, gt_batch):
        bce_loss = self.bce_loss(pred_batch['binary'][:, 0],
                                 gt_batch['gt'], gt_batch['mask'])
        metrics = dict(loss_bce=bce_loss)
        if 'thresh' in pred_batch:
            l1_loss, l1_metric = self.l1_loss(pred_batch['thresh'][:, 0],
                                              gt_batch['thresh_map'],
                                              gt_batch['thresh_mask'])
            dice_loss = self.dice_loss(pred_batch['thresh_binary'][:, 0],
                                       gt_batch['gt'], gt_batch['mask'])
            metrics['loss_thresh'] = dice_loss
            loss = dice_loss + self.l1_scale * l1_loss + self.bce_scale * bce_loss
            metrics.update(**l1_metric)
        else:
            loss = bce_loss
        return loss, metrics
