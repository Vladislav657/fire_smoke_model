import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        batch_size, num_classes = logits.shape[0], logits.shape[1]

        # Применяем softmax по классам для получения вероятностей
        probs = nn.functional.softmax(logits, dim=1)

        # Подготавливаем тензоры: [batch_size, num_classes, -1]
        probs_flat = probs.view(batch_size, num_classes, -1)
        targets_flat = targets.view(batch_size, num_classes, -1)

        # Вычисляем пересечение и суммы для каждого класса
        intersection = (probs_flat * targets_flat).sum(2)  # [batch_size, num_classes]
        cardinality = (probs_flat + targets_flat).sum(2)  # [batch_size, num_classes]

        # Вычисляем Dice score для каждого класса
        dice_scores = (2. * intersection + self.smooth) / (cardinality + self.smooth)  # [batch_size, num_classes]
        dice_scores = dice_scores.mean(1)
        dice_loss = 1. - dice_scores.mean()  # [batch_size]
        return dice_loss
