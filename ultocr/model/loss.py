import torch
import torch.nn as nn
import torch.nn.functional as F


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
        loss = F.binary_cross_entropy(
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
            return mask_sum
        else:
            loss = (torch.abs(pred - gt) * mask).sum() / mask_sum
            return loss


class DBLoss(nn.Module):
    def __init__(self, alpha=1., beta=10., reduction='mean', negative_ratio=3, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.negative_ratio = negative_ratio
        self.eps = eps
        self.ohem_loss = BalanceCrossEntropyLoss(negative_ratio, eps)
        self.l1_loss = MaskL1Loss()
        self.dice_loss = DiceLoss(eps)

    def forward(self, preds, gts):
        """
        :param preds: probability map (Ls), binary map (Lb), threshold map (Lt)
        :param gts: prob map, binary map
        :return: prob_loss, threshold_loss, binary_loss, prob_threshold_loss, total_loss
        total_loss = Ls + alpha * Lb + beta * Lt
        """
        assert preds.dim() == 4
        assert gts.dim() == 4
        prob_map = preds[:, 0, :, :]
        threshold_map = preds[:, 1, :, :]
        if preds.size(1) == 3:
            appro_binary_map = preds[:, 2, :, :]
        prob_gt_map = gts[0, :, :, :]
        supervision_mask = gts[1, :, :, :]  # 0/1
        threshold_gt_map = gts[2, :, :, :]  # 0.3/0.7
        text_area_gt_map = gts[3, :, :, :]  # 0/1

        # loss
        prob_loss = self.ohem_loss(prob_map, prob_gt_map, supervision_mask)
        threshold_loss = self.l1_loss(threshold_map, threshold_gt_map, text_area_gt_map)
        prob_threshold_loss = prob_loss + self.beta * threshold_loss
        if preds.size(1) == 3:
            binary_loss = self.dice_loss(appro_binary_map, prob_gt_map, supervision_mask)
            total_loss = prob_threshold_loss + self.alpha * binary_loss
            return total_loss
        else:
            return prob_threshold_loss
