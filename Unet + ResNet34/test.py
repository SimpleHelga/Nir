import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from dataset import get_dataloaders, CLASS_NAMES
from models import UNet, count_parameters
from losses_metrics import CombinedLoss, compute_class_weights

def analyze_dataset(data_root: str, num_classes: int = 22, ignore_index: int = 255):
    print("\n" + "=" * 70)
    print("🔬 АНАЛИЗ ДАТАСЕТА (BCSS: правильные классы 1..21, ignore = 255)")
    print("=" * 70)

    mask_dirs = {
        'train': os.path.join(data_root, 'train_mask_512'),
        'val': os.path.join(data_root, 'val_mask_512'),
    }

    for split, mask_dir in mask_dirs.items():
        if not os.path.exists(mask_dir):
            print(f"❌ Папка не найдена: {mask_dir}")
            continue

        files = [f for f in os.listdir(mask_dir) if f.lower().endswith('.png')]
        print(f"\n📁 {split.upper()}: {len(files)} масок")

        if len(files) == 0:
            continue

        counts = np.zeros(num_classes, dtype=np.float64)  # 0..21
        ignore_count = 0
        total_pixels = 0

        for f in files:
            mask = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            # Применяем маппинг как в dataset (0->255, 7->255)
            m = mask.copy()
            m[m == 0] = ignore_index
            m[m == 7] = ignore_index

            uniq, cnts = np.unique(m, return_counts=True)
            for u, c in zip(uniq, cnts):
                if u == ignore_index:
                    ignore_count += c
                elif 1 <= u <= 21:
                    counts[u] += c
                else:
                    # Любое другое значение (например, >21 или <0) тоже считаем ignore
                    ignore_count += c
            total_pixels += m.size

        print(f"\n   Распределение активных классов (1..21):")
        for cls in range(1, num_classes):
            pct = counts[cls] / total_pixels * 100 if total_pixels > 0 else 0
            bar = "█" * int(pct / 2)
            name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"cls_{cls}"
            print(f"   {cls:2d} {name:22s}: {pct:6.2f}% {bar}")

        ignore_pct = ignore_count / total_pixels * 100 if total_pixels > 0 else 0
        print(f"\n   Игнорируемые пиксели (0,7 и прочие): {ignore_pct:.2f}%")
        print(f"   Всего пикселей: {total_pixels}")

def test_forward_pass(data_root: str, batch_size: int = 2, num_workers: int = 0):
    print("\n" + "=" * 70)
    print("🧪 ТЕСТ FORWARD/BACKWARD PASS (исправленные классы)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Устройство: {device}")

    train_loader, _ = get_dataloaders(data_root, batch_size=batch_size, num_workers=num_workers)
    images, masks = next(iter(train_loader))
    images, masks = images.to(device), masks.to(device)

    print(f"\n📦 Батч: images={images.shape}, masks={masks.shape}")
    unique_vals = torch.unique(masks).tolist()
    print(f"   Уникальные значения в масках (после датасета): {unique_vals}")

    # Проверяем, что нет 0 и 7
    if 0 in unique_vals:
        print(f"   ❌ ОШИБКА: класс 0 присутствует! Маппинг 0->255 не сработал.")
    elif 7 in unique_vals:
        print(f"   ❌ ОШИБКА: класс 7 присутствует! Должен быть заменён на 255.")
    else:
        print(f"   ✅ OK: классы 0 и 7 отсутствуют.")

    # Дополнительно: проверяем, что все значения либо 1..21, либо 255
    invalid = [v for v in unique_vals if v not in range(1,22) and v != 255]
    if invalid:
        print(f"   ⚠️ ВНИМАНИЕ: обнаружены недопустимые значения: {invalid}")
    else:
        print(f"   ✅ OK: все значения маски корректны (1..21 или 255)")

    model = UNet(n_channels=3, num_classes=22, bilinear=True).to(device)
    print(f"\n📐 Модель (U-Net): {count_parameters(model):,} параметров")

    class_weights = compute_class_weights(
        os.path.join(data_root, "train_mask_512"),
        num_classes=22,
        ignore_index=255
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = CombinedLoss(
        num_classes=22,
        ignore_index=255,
        class_weights=class_weights
    ).to(device)

    with torch.no_grad():
        outputs = model(images)
        loss = criterion(outputs, masks)
        print(f"\n📉 Loss (no grad): {loss.item():.4f}, finite={torch.isfinite(loss).item()}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    outputs = model(images)
    loss = criterion(outputs, masks)
    loss.backward()

    total_grad_norm = 0.0
    has_nan_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            g_norm = param.grad.norm().item()
            total_grad_norm += g_norm ** 2
            if torch.isnan(param.grad).any():
                has_nan_grad = True
                print(f"   ❌ NaN grad in {name}")
    total_grad_norm = total_grad_norm ** 0.5

    print(f"\n📈 Backward: loss={loss.item():.4f}")
    print(f"   Норма градиентов: {total_grad_norm:.4f}")
    print(f"   NaN в градиентах: {'❌ ДА' if has_nan_grad else '✅ Нет'}")

    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)
        pred_unique = torch.unique(preds).tolist()
        print(f"\n🎯 Argmax предсказаний: уникальные = {pred_unique}")

    if torch.isfinite(loss) and not has_nan_grad and (0 not in unique_vals) and (7 not in unique_vals):
        print(f"\n✅ Тест ПРОЙДЕН: всё работает корректно.")
    else:
        print(f"\n❌ Тест ПРОВАЛЕН. Проверьте маппинг и аугментации.")

def main():
    data_root = "C:/Users/Home/PycharmProjects/Nir/archive/BCSS_512"
    analyze_dataset(data_root)
    test_forward_pass(data_root)

    print("\n" + "=" * 70)
    print("💻 СИСТЕМА")
    print("=" * 70)
    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA доступен:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA версия:    {torch.version.cuda}")
        print(f"GPU:            {torch.cuda.get_device_name(0)}")

if __name__ == '__main__':
    main()