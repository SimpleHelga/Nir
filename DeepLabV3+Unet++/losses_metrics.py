import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

EPSILON = 1e-5
NUM_CLASSES = 22
IGNORE_INDEX = 255

def compute_class_weights(masks_dir: str, num_classes: int = NUM_CLASSES,
                          ignore_index: int = IGNORE_INDEX,
                          max_weight: float = 5.0) -> Optional[torch.Tensor]:
    print(f"⚖️  Подсчёт весов классов из {masks_dir} ...")
    files = [f for f in os.listdir(masks_dir) if f.lower().endswith('.png')]
    if len(files) == 0:
        print("   Маски не найдены, веса не используются.")
        return None

    counts = np.zeros(num_classes, dtype=np.float64)
    total_pixels = 0

    for f in files:
        mask = cv2.imread(os.path.join(masks_dir, f), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        m = mask.copy()
        m[m == 0] = ignore_index
        m[m == 7] = ignore_index
        valid_mask = (m != ignore_index) & (m >= 1) & (m <= 21)
        if not valid_mask.any():
            continue
        labels = m[valid_mask]
        unique, cnt = np.unique(labels, return_counts=True)
        for u, c in zip(unique, cnt):
            if 1 <= u <= 21:
                counts[u] += c
                total_pixels += c

    if total_pixels == 0:
        print("   Нет валидных пикселей для вычисления весов.")
        return None

    present_classes = counts[1:21] > 0
    num_present = present_classes.sum()
    if num_present == 0:
        return None

    freq = counts[1:21] / total_pixels
    weights_raw = 1.0 / np.sqrt(freq + EPSILON)
    weights_raw = np.clip(weights_raw, 0.5, max_weight)
    mean_w = weights_raw[present_classes].mean()
    weights_norm = weights_raw / mean_w
    weights = np.ones(num_classes, dtype=np.float32)
    weights[1:21] = weights_norm

    w_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"   Базовые веса классов 1..21: {w_tensor[1:21].numpy().round(3).tolist()}")
    return w_tensor

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=IGNORE_INDEX, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX,
                 softmax: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.softmax = softmax

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.float()
        if self.softmax:
            inputs = F.softmax(inputs, dim=1)

        targets = targets.clone()
        invalid = (targets >= self.num_classes) & (targets != self.ignore_index)
        targets[invalid] = self.ignore_index
        targets[targets < 0] = self.ignore_index

        targets_safe = targets.clone()
        targets_safe[targets == self.ignore_index] = 0
        targets_one_hot = F.one_hot(targets_safe, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            ignore_mask = (targets == self.ignore_index)
            targets_one_hot[ignore_mask.unsqueeze(1).expand_as(targets_one_hot)] = 0
            inputs = inputs.clone()
            inputs[ignore_mask.unsqueeze(1).expand_as(inputs)] = 0

        intersection = (inputs * targets_one_hot).sum(dim=(0, 2, 3))
        union = inputs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))

        valid = torch.ones(self.num_classes, dtype=torch.bool, device=inputs.device)
        valid[0] = False
        if self.ignore_index < self.num_classes:
            valid[self.ignore_index] = False
        valid = valid & (union > 0)

        dice_score = torch.zeros_like(intersection)
        dice_score[valid] = (2.0 * intersection[valid] + EPSILON) / (union[valid] + EPSILON)

        if valid.sum() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        return 1.0 - dice_score[valid].mean()

class CombinedLoss(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX,
                 ce_weight: float = 0.3, dice_weight: float = 0.7,
                 class_weights: Optional[torch.Tensor] = None,
                 use_focal: bool = True, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        if use_focal:
            self.ce_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma,
                                     ignore_index=ignore_index, weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)

        self.dice_loss = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.float()
        targets = targets.clone().long()
        invalid = (targets >= self.num_classes) & (targets != self.ignore_index)
        targets[invalid] = self.ignore_index
        targets[targets < 0] = self.ignore_index

        loss_ce = self.ce_loss(inputs, targets)
        loss_dice = self.dice_loss(inputs, targets)

        loss = self.ce_weight * loss_ce + self.dice_weight * loss_dice
        return torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)

class MetricsAccumulator:
    def __init__(self, num_classes: int, ignore_index: int):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.inter = torch.zeros(num_classes, dtype=torch.float64)
        self.union = torch.zeros(num_classes, dtype=torch.float64)
        self._valid_mask = torch.ones(num_classes, dtype=torch.bool)
        self._valid_mask[0] = False
        if ignore_index < num_classes:
            self._valid_mask[ignore_index] = False

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if preds.dim() == 4 and preds.size(1) > 1:
            preds = torch.argmax(preds, dim=1)

        ps = preds.clone()
        ts = targets.clone()
        ps[ps == self.ignore_index] = 0
        ts[ts == self.ignore_index] = 0

        p_one_hot = F.one_hot(ps, self.num_classes).permute(0, 3, 1, 2).float()
        t_one_hot = F.one_hot(ts, self.num_classes).permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            ignore_mask = (targets == self.ignore_index)
            p_one_hot[ignore_mask.unsqueeze(1).expand_as(p_one_hot)] = 0
            t_one_hot[ignore_mask.unsqueeze(1).expand_as(t_one_hot)] = 0

        intersection = (p_one_hot * t_one_hot).sum(dim=(0, 2, 3))
        union = p_one_hot.sum(dim=(0, 2, 3)) + t_one_hot.sum(dim=(0, 2, 3)) - intersection

        self.inter += intersection.cpu().double()
        self.union += union.cpu().double()

    def get_metrics(self) -> Dict:
        iou = self.inter / (self.union + EPSILON)
        dice = (2.0 * self.inter) / (self.union + self.inter + EPSILON)

        present = self.union > 0
        valid = self._valid_mask & present

        if valid.sum() == 0:
            return {
                'mean_iou': 0.0,
                'mean_dice': 0.0,
                'per_class_iou': ['N/A'] * self.num_classes,
                'per_class_dice': ['N/A'] * self.num_classes,
            }

        return {
            'mean_iou': iou[valid].mean().item(),
            'mean_dice': dice[valid].mean().item(),
            'per_class_iou': [f"{v:.3f}" if valid[i] else "N/A" for i, v in enumerate(iou)],
            'per_class_dice': [f"{v:.3f}" if valid[i] else "N/A" for i, v in enumerate(dice)],
        }