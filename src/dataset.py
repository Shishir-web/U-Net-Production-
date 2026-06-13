# src/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# DeepGlobe has 7 land cover classes
CLASSES = [
    'urban', 'agriculture', 'rangeland',
    'forest', 'water', 'barren', 'unknown'
]

# RGB values for each class in the mask
CLASS_RGB = {
    (0,   255, 255): 0,  # urban
    (255, 255, 0):   1,  # agriculture
    (255, 0,   255): 2,  # rangeland
    (0,   255, 0):   3,  # forest
    (0,   0,   255): 4,  # water
    (255, 255, 255): 5,  # barren
    (0,   0,   0):   6,  # unknown
}

def rgb_mask_to_label(mask_rgb):
    """Convert RGB mask to single-channel label map."""
    h, w = mask_rgb.shape[:2]
    label = np.zeros((h, w), dtype=np.int64)
    for rgb, idx in CLASS_RGB.items():
        match = np.all(mask_rgb == np.array(rgb), axis=-1)
        label[match] = idx
    return label


class DeepGlobeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.images    = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith('_sat.jpg')
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')

        image = np.array(Image.open(
            os.path.join(self.image_dir, img_name)).convert('RGB'))
        mask  = np.array(Image.open(
            os.path.join(self.mask_dir, mask_name)).convert('RGB'))
        mask  = rgb_mask_to_label(mask)

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug['image']
            mask  = torch.tensor(aug['mask'], dtype=torch.long)
        else:
            image = torch.tensor(image.transpose(2,0,1),
                                 dtype=torch.float32) / 255.0
            mask  = torch.tensor(mask, dtype=torch.long)

        return image, mask


def get_transforms(img_size=256):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(p=0.3),
        A.Normalize(mean=(0.485,0.456,0.406),
                    std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485,0.456,0.406),
                    std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])
    return train_transform, val_transform


def get_loaders(data_dir='./data/deepglobe',
                batch_size=8, img_size=256, val_split=0.15):
    train_tf, val_tf = get_transforms(img_size)

    full_dataset = DeepGlobeDataset(
        image_dir=os.path.join(data_dir, 'train'),
        mask_dir =os.path.join(data_dir, 'train'),  # same folder
        transform=train_tf
    )
    val_size   = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # Apply val transform to val split
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)
    return train_loader, val_loader