import torch
import numpy as np
import matplotlib.pyplot as plt
import os

COLORS = [
    [0,   255, 255],  # urban
    [255, 255, 0],    # agriculture
    [255, 0,   255],  # rangeland
    [0,   255, 0],    # forest
    [0,   0,   255],  # water
    [255, 255, 255],  # barren
    [0,   0,   0],    # unknown
]

def label_to_rgb(label):
    h, w  = label.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, rgb in enumerate(COLORS):
        color[label == cls] = rgb
    return color


def plot_predictions(model, val_loader, device, n=4,
                     save_dir='./assets'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    imgs, masks = next(iter(val_loader))
    imgs, masks = imgs.to(device), masks.to(device)

    with torch.no_grad():
        preds = model(imgs).argmax(dim=1).cpu().numpy()

    imgs  = imgs.cpu().numpy()
    masks = masks.cpu().numpy()

    fig, axes = plt.subplots(n, 3, figsize=(12, n * 4))
    for i in range(n):
        img = imgs[i].transpose(1,2,0)
        img = (img * np.array([0.229,0.224,0.225]) +
                     np.array([0.485,0.456,0.406]))
        img = np.clip(img, 0, 1)
        axes[i,0].imshow(img);          axes[i,0].set_title('Image')
        axes[i,1].imshow(label_to_rgb(masks[i]));
        axes[i,1].set_title('Ground Truth')
        axes[i,2].imshow(label_to_rgb(preds[i]));
        axes[i,2].set_title('Prediction')
        for ax in axes[i]: ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'predictions.png'), dpi=150)
    print("Saved assets/predictions.png")


def plot_comparison(unet_history, plain_history,
                    save_dir='./assets'):
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(unet_history['train_loss'],  label='U-Net',     color='#2563eb')
    ax1.plot(plain_history['train_loss'], label='Plain Enc', color='#dc2626',
             linestyle='--')
    ax1.set_title('Training Loss'); ax1.set_xlabel('Epoch')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(unet_history['val_iou'],  label='U-Net IoU',     color='#2563eb')
    ax2.plot(plain_history['val_iou'], label='Plain Enc IoU', color='#dc2626',
             linestyle='--')
    ax2.set_title('Validation IoU'); ax2.set_xlabel('Epoch')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison.png'), dpi=150)
    print("Saved assets/comparison.png")