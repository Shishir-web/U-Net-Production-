# src/train.py
import torch
import torch.nn as nn
from tqdm import tqdm


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds   = torch.softmax(preds, dim=1)
        num_cls = preds.shape[1]
        targets_one_hot = torch.zeros_like(preds).scatter_(
            1, targets.unsqueeze(1), 1)
        intersection = (preds * targets_one_hot).sum(dim=(2,3))
        union        = preds.sum(dim=(2,3)) + targets_one_hot.sum(dim=(2,3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """BCE + Dice — standard for segmentation tasks."""
    def __init__(self):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        return self.ce(preds, targets) + self.dice(preds, targets)


def iou_score(preds, targets, num_classes=7):
    preds = preds.argmax(dim=1)
    ious  = []
    for cls in range(num_classes):
        pred_cls   = (preds == cls)
        target_cls = (targets == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union        = (pred_cls | target_cls).sum().float()
        if union == 0:
            continue
        ious.append((intersection / union).item())
    return sum(ious) / len(ious) if ious else 0.0


def train_model(model, train_loader, val_loader,
                epochs=30, lr=1e-3, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    model     = model.to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)

    history = {'train_loss': [], 'val_iou': [], 'val_dice': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for imgs, masks in tqdm(train_loader,
                                desc=f"Epoch {epoch+1}/{epochs}",
                                leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        total_iou = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                total_iou += iou_score(preds, masks)

        avg_loss = running_loss / len(train_loader)
        avg_iou  = total_iou   / len(val_loader)
        history['train_loss'].append(avg_loss)
        history['val_iou'].append(avg_iou)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | "
              f"Val IoU: {avg_iou:.4f}")

    return history