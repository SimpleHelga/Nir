import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple

# Правильные названия классов из gtruth_codes_512.tsv
CLASS_NAMES = [
    'outside_roi',          # 0 -> будет замаплен в ignore_index
    'tumor',                # 1
    'stroma',               # 2
    'lymphocytic_infiltrate', # 3
    'necrosis_or_debris',   # 4
    'glandular_secretions', # 5
    'blood',                # 6
    'exclude',              # 7 -> тоже игнорируем
    'metaplasia_NOS',       # 8
    'fat',                  # 9
    'plasma_cells',         # 10
    'other_immune_infiltrate', # 11
    'mucoid_material',      # 12
    'normal_acinus_or_duct', # 13
    'lymphatics',           # 14
    'undetermined',         # 15
    'nerve',                # 16
    'skin_adnexa',          # 17
    'blood_vessel',         # 18
    'angioinvasion',        # 19
    'dcis',                 # 20
    'other'                 # 21
]

CLASS_MAPPING = {
    0: 255,   # outside_roi -> ignore
    7: 255,   # exclude -> ignore
}

class BCSSDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, augment: bool = False,
                 img_size: Tuple[int, int] = (512, 512), ignore_index: int = 255):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.ignore_index = ignore_index

        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            raise FileNotFoundError(f"Папки не найдены: {images_dir} или {masks_dir}")

        self.images = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.png', '.tif', '.jpeg', '.tiff'))
        ])
        if len(self.images) == 0:
            raise FileNotFoundError(f"Нет изображений в {images_dir}")

        self._validate_pairs()
        print(f"✓ Загружено {len(self.images)} изображений из {images_dir}")
        self._diagnose_first_mask()
        self.transform = self._get_train_transform() if augment else self._get_val_transform()

    def _validate_pairs(self):
        missing = 0
        for img_name in self.images:
            base = os.path.splitext(img_name)[0]
            mask_path = os.path.join(self.masks_dir, base + '.png')
            if not os.path.exists(mask_path):
                mask_path = os.path.join(self.masks_dir, img_name)
            if not os.path.exists(mask_path):
                missing += 1
        if missing > 0:
            print(f"⚠️ {missing}/{len(self.images)} масок не найдено!")

    def _diagnose_first_mask(self):
        if len(self.images) == 0:
            return
        img_name = self.images[0]
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.masks_dir, base_name + '.png')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.masks_dir, img_name)
        if not os.path.exists(mask_path):
            return

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return
        uniq = np.unique(mask)
        print(f"🔍 Диагностика маски ({img_name}): уникальные значения = {uniq.tolist()}")
        if 0 in uniq:
            pct = (mask == 0).sum() / mask.size * 100
            print(f"   → Класс 0 (outside ROI): {pct:.1f}% пикселей")
        if 7 in uniq:
            pct = (mask == 7).sum() / mask.size * 100
            print(f"   → Класс 7 (exclude): {pct:.1f}% пикселей (будет проигнорирован)")
        if 255 in uniq:
            pct = (mask == 255).sum() / mask.size * 100
            print(f"   → Класс 255 (border/ignore): {pct:.1f}% пикселей")

    def _get_train_transform(self) -> A.Compose:
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
            ], p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1),
                rotate=(-15, 15),
                interpolation=cv2.INTER_NEAREST,
                p=0.5
            ),
            # 🎨 НОВЫЕ STAIN AUGMENTATIONS (имитируют разные лаборатории)
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=15,
                p=0.6
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            A.Resize(height=self.img_size[0], width=self.img_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    def _get_val_transform(self) -> A.Compose:
        return A.Compose([
            A.Resize(height=self.img_size[0], width=self.img_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        base_name = os.path.splitext(img_name)[0]

        mask_path = os.path.join(self.masks_dir, base_name + '.png')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.masks_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Не удалось прочитать маску: {mask_path}")

        # Маппинг 0->255, 7->255
        mapped_mask = mask.copy()
        for orig, target in CLASS_MAPPING.items():
            mapped_mask[mask == orig] = target

        # Остальные значения должны быть в диапазоне 1..21, иначе тоже в ignore
        mapped_mask = np.where(
            (mapped_mask > 21) & (mapped_mask != self.ignore_index),
            self.ignore_index,
            mapped_mask
        )

        augmented = self.transform(image=image, mask=mapped_mask)
        image_tensor = augmented['image']
        mask_tensor = augmented['mask'].long()

        # Принудительная очистка от артефактов 0, возникших при аугментациях
        mask_tensor = torch.where(
            (mask_tensor == 0) | (mask_tensor > 21),
            torch.tensor(self.ignore_index, device=mask_tensor.device),
            mask_tensor
        )

        if mask_tensor.dim() == 3 and mask_tensor.size(0) == 1:
            mask_tensor = mask_tensor.squeeze(0)

        return image_tensor, mask_tensor

def get_dataloaders(data_root: str, batch_size: int = 2, num_workers: int = 0,
                    img_size: Tuple[int, int] = (512, 512)) -> Tuple[DataLoader, DataLoader]:
    train_ds = BCSSDataset(
        os.path.join(data_root, 'train_512'),
        os.path.join(data_root, 'train_mask_512'),
        augment=True, img_size=img_size)
    val_ds = BCSSDataset(
        os.path.join(data_root, 'val_512'),
        os.path.join(data_root, 'val_mask_512'),
        augment=False, img_size=img_size)

    common_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if num_workers > 0:
        common_kwargs.update({
            'persistent_workers': True,
            'prefetch_factor': 2,
        })

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common_kwargs)
    return train_loader, val_loader