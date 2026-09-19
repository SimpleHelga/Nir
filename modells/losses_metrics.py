import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HybridLoss(nn.Module):
    """CE + Dice + Focal для M3-Net++ (с deep supervision)."""
    def __init__(self, num_classes=22, ignore_index=255, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_weights = class_weights
        self.dice_weight = 2.5
        self.focal_weight = 0.5
        self.ce_weight = 1.0

    def _masked_mean(self, tensor, mask):
        """Усреднение только по не-маскированным элементам. Если все 0 — вернуть 0."""
        if mask.sum() == 0:
            return torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
        return (tensor * mask).sum() / mask.sum()

    def ce_loss(self, pred, target):
        # reduction='none' + ручное усреднение — защита от nan на полностью ignore-батчах
        ce = F.cross_entropy(pred, target, weight=self.class_weights,
                             ignore_index=self.ignore_index, reduction='none')
        mask = (target != self.ignore_index).float()
        return self._masked_mean(ce, mask)

    def dice_loss(self, pred, target, smooth=1.0):
        pred = F.softmax(pred, dim=1)
        target_safe = target.clone()
        target_safe[target == self.ignore_index] = 0
        target_one_hot = F.one_hot(target_safe, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        mask = (target != self.ignore_index).unsqueeze(1).float()
        pred = pred * mask
        target_one_hot = target_one_hot * mask

        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def focal_loss(self, pred, target, gamma=2.0, alpha=0.25):
        ce = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        mask = (target != self.ignore_index).float()
        pt = torch.exp(-ce)
        focal = alpha * (1 - pt) ** gamma * ce
        return self._masked_mean(focal, mask)

    def forward(self, outputs, target):
        if isinstance(outputs, dict):
            out = outputs['out']
            aux = outputs['aux']
            loss = self.ce_weight * self.ce_loss(out, target) + \
                   self.dice_weight * self.dice_loss(out, target) + \
                   self.focal_weight * self.focal_loss(out, target)
            loss += 0.4 * (self.ce_weight * self.ce_loss(aux, target) +
                           self.dice_weight * self.dice_loss(aux, target))
            return loss
        else:
            return self.ce_weight * self.ce_loss(outputs, target) + \
                self.dice_weight * self.dice_loss(outputs, target) + \
                self.focal_weight * self.focal_loss(outputs, target)


class HoVerNetLoss(nn.Module):
    """CE + Dice + MSE(hv) + BCE(boundary) для HoVer-Net."""
    def __init__(self, num_classes=22, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def _masked_mean(self, tensor, mask):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
        return (tensor * mask).sum() / mask.sum()

    def ce_loss(self, pred, target):
        ce = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        mask = (target != self.ignore_index).float()
        return self._masked_mean(ce, mask)

    def dice_loss(self, pred, target, smooth=1.0):
        pred = F.softmax(pred, dim=1)
        target_safe = target.clone()
        target_safe[target == self.ignore_index] = 0
        target_one_hot = F.one_hot(target_safe, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        mask = (target != self.ignore_index).unsqueeze(1).float()
        pred = pred * mask
        target_one_hot = target_one_hot * mask

        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def forward(self, outputs, target, hv_target=None, boundary_target=None):
        semantic = outputs['semantic']
        hv_map = outputs['hv_map']
        boundary = outputs['boundary']

        loss = self.ce_loss(semantic, target) + self.dice_loss(semantic, target)

        if hv_target is not None:
            loss += 0.5 * self.mse(hv_map, hv_target)
        if boundary_target is not None:
            loss += 0.5 * self.bce(boundary, boundary_target)
        return loss


def compute_iou_dice(pred, target, num_classes=22, ignore_index=255):
    """Mean IoU и Dice по классам 1..21 (ignore_index пропускается)."""
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()

    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    ious, dices = [], []
    for cls in range(1, num_classes):
        pred_cls = (pred == cls).astype(np.float32)
        target_cls = (target == cls).astype(np.float32)
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        if union == 0:
            continue
        ious.append(intersection / union)
        dices.append(2 * intersection / (pred_cls.sum() + target_cls.sum() + 1e-8))

    return (np.mean(ious) if ious else 0.0), (np.mean(dices) if dices else 0.0)