import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# === ВАЖНО: Добавляем пути к модулям из подпапок ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "DeepLabV3+Unet++"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Unet + ResNet34"))

from modells.models import get_model, count_parameters
from dataset import CLASS_NAMES, CLASS_MAPPING

# === КОНФИГУРАЦИЯ ===
CONFIG = {
    # Пути к чекпоинтам
    "checkpoints": {
        "unet": os.path.join(PROJECT_ROOT, "Unet + ResNet34", "checkpoints", "best_model_unet.pth"),  # ← ИСПРАВЛЕНО
        "deeplab": os.path.join(PROJECT_ROOT, "DeepLabV3+Unet++", "checkpoints", "best_model_deeplab.pth"),
        "unetpp": os.path.join(PROJECT_ROOT, "DeepLabV3+Unet++", "checkpoints", "best_model_unetpp.pth"),
    },

    "data_root": os.path.join(PROJECT_ROOT, "archive", "BCSS_512"),
    "val_images_dir": os.path.join(PROJECT_ROOT, "archive", "BCSS_512", "val_512"),
    "val_masks_dir": os.path.join(PROJECT_ROOT, "archive", "BCSS_512", "val_mask_512"),
    "output_dir": os.path.join(PROJECT_ROOT, "ensemble_results"),

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "focus_classes": [1, 2, 3, 4, 9],
    "tta": True,
    "model_weights": {"unet": 1.0, "deeplab": 1.0, "unetpp": 1.0},

    "models": [
        {"name": "unet", "arch": "unet", "encoder": "resnet34", "num_classes": 22},
        {"name": "deeplab", "arch": "deeplab", "encoder": "efficientnet-b3", "num_classes": 22},
        {"name": "unetpp", "arch": "unetpp", "encoder": "resnet50", "num_classes": 22},
    ]
}


def get_val_transform():
    return Compose([
        Resize(height=512, width=512),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


@torch.no_grad()
def tta_predict(model, tensor_img, device, use_tta=True):
    if not use_tta:
        return torch.softmax(model(tensor_img), dim=1)

    probs = []
    probs.append(torch.softmax(model(tensor_img), dim=1))

    out = model(torch.flip(tensor_img, dims=[3]))
    probs.append(torch.flip(torch.softmax(out, dim=1), dims=[3]))

    out = model(torch.flip(tensor_img, dims=[2]))
    probs.append(torch.flip(torch.softmax(out, dim=1), dims=[2]))

    out = model(torch.rot90(tensor_img, k=1, dims=[2, 3]))
    probs.append(torch.rot90(torch.softmax(out, dim=1), k=3, dims=[2, 3]))

    aug = torch.flip(torch.rot90(tensor_img, k=1, dims=[2, 3]), dims=[3])
    out = model(aug)
    out = torch.rot90(torch.flip(torch.softmax(out, dim=1), dims=[3]), k=3, dims=[2, 3])
    probs.append(out)

    return torch.stack(probs, dim=0).mean(dim=0)


def load_models(device):
    models = {}

    for m_cfg in CONFIG["models"]:
        name = m_cfg["name"]
        ckpt_path = CONFIG["checkpoints"].get(name)

        if ckpt_path is None or not os.path.exists(ckpt_path):
            print(f"⚠️  Чекпоинт не найден для {name}: {ckpt_path}")
            print(f"   Проверьте имя файла в папке checkpoints!")
            continue

        print(f"📦 Загрузка {name.upper()} ({m_cfg['arch']} + {m_cfg['encoder']})...")

        # Создаём модель с ПРАВИЛЬНОЙ архитектурой
        model = get_model(
            num_classes=m_cfg["num_classes"],
            encoder=m_cfg["encoder"],
            architecture=m_cfg["arch"]  # ← ВАЖНО: передаём архитектуру!
        ).to(device)

        print(f"   Параметры: {count_parameters(model):,}")

        # Загружаем веса
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            best_dice = ckpt.get('best_val_dice', 0)
            epoch = ckpt.get('epoch', '?')
            print(f"   ✅ Загружено (Epoch {epoch}, Best Dice: {best_dice:.4f})")
        else:
            model.load_state_dict(ckpt)
            print(f"   ✅ Загружено (без метаданных)")

        model.eval()
        models[name] = model

    if len(models) < 2:
        raise RuntimeError(f"❌ Загружено только {len(models)} моделей. Нужно минимум 2.")

    return models


def compute_metrics(pred_mask, gt_mask, focus_classes):
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in focus_classes}
    valid = (gt_mask != 255)
    pred_v = pred_mask[valid]
    gt_v = gt_mask[valid]

    for c in focus_classes:
        pred_c = (pred_v == c)
        gt_c = (gt_v == c)
        stats[c]["tp"] += (pred_c & gt_c).sum().item()
        stats[c]["fp"] += (pred_c & ~gt_c).sum().item()
        stats[c]["fn"] += (~pred_c & gt_c).sum().item()

    return stats


def main():
    print("=" * 70)
    print("🎭 АНСАМБЛЕВАЯ ОЦЕНКА (Soft Voting + TTA)")
    print("=" * 70)
    print(f"📁 Проект: {PROJECT_ROOT}")
    print(f"️  Устройство: {CONFIG['device']}")
    print(f"🎯 TTA: {'✅ ON' if CONFIG['tta'] else ' OFF'}")

    torch.backends.cudnn.benchmark = True
    device = torch.device(CONFIG["device"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # 1. Загрузка моделей
    print("\n" + "=" * 70)
    print("📦 ЗАГРУЗКА МОДЕЛЕЙ")
    print("=" * 70)
    models = load_models(device)
    print(f"\n🎯 Ансамбль из {len(models)} моделей: {list(models.keys())}")

    # 2. Подготовка данных
    transform = get_val_transform()
    img_dir = CONFIG["val_images_dir"]
    mask_dir = CONFIG["val_masks_dir"]

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"❌ Папка с изображениями не найдена: {img_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"❌ Папка с масками не найдена: {mask_dir}")

    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"\n Найдено {len(img_files)} валидационных изображений")

    # 3. Инициализация аккумуляторов
    all_stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CONFIG["focus_classes"]}
    all_preds_flat, all_gts_flat = [], []

    # 4. Цикл инференса
    print("\n" + "=" * 70)
    print("🔍 ИНФЕРЕНС С TTA")
    print("=" * 70)

    for fname in tqdm(img_files, desc=" Ансамбль + TTA"):
        img_path = os.path.join(img_dir, fname)
        base = os.path.splitext(fname)[0]
        mask_path = os.path.join(mask_dir, f"{base}.png")

        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, fname)
        if not os.path.exists(mask_path):
            continue

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        for orig, target in CLASS_MAPPING.items():
            gt_mask[gt_mask == orig] = target
        gt_mask = np.where((gt_mask > 21) & (gt_mask != 255), 255, gt_mask)

        tensor_img = transform(image=img)['image'].unsqueeze(0).to(device)

        # Ансамблирование
        ensemble_probs = torch.zeros((1, 22, 512, 512), device=device)
        total_weight = 0.0

        for name, model in models.items():
            prob = tta_predict(model, tensor_img, device, CONFIG["tta"])
            w = CONFIG["model_weights"].get(name, 1.0)
            ensemble_probs += prob * w
            total_weight += w

        ensemble_probs /= (total_weight + 1e-8)
        pred_mask = torch.argmax(ensemble_probs, dim=1).squeeze(0).cpu().numpy()

        img_stats = compute_metrics(pred_mask, gt_mask, CONFIG["focus_classes"])
        for c in CONFIG["focus_classes"]:
            all_stats[c]["tp"] += img_stats[c]["tp"]
            all_stats[c]["fp"] += img_stats[c]["fp"]
            all_stats[c]["fn"] += img_stats[c]["fn"]

        valid = (gt_mask != 255)
        all_preds_flat.append(pred_mask[valid])
        all_gts_flat.append(gt_mask[valid])

    # 5. Вывод таблицы
    print("\n" + "=" * 80)
    print("📊 ФИНАЛЬНЫЕ МЕТРИКИ АНСАМБЛЯ (TTA + Soft Voting)")
    print("=" * 80)
    print(f"{'Класс':<20} {'IoU':<10} {'Dice':<10} {'Precision':<12} {'Recall':<10}")
    print("-" * 80)

    ious, dices = [], []
    for c in CONFIG["focus_classes"]:
        tp = all_stats[c]["tp"]
        fp = all_stats[c]["fp"]
        fn = all_stats[c]["fn"]

        iou = tp / (tp + fp + fn + 1e-8)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)

        ious.append(iou)
        dices.append(dice)

        class_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        print(f"{c} ({class_name:<15}) {iou:<10.4f} {dice:<10.4f} {prec:<12.4f} {rec:<10.4f}")

    print("-" * 80)
    print(f"{'MEAN':<20} {np.mean(ious):<10.4f} {np.mean(dices):<10.4f}")
    print("=" * 80)

    # 6. Сохранение CSV
    csv_path = os.path.join(CONFIG["output_dir"], "ensemble_metrics.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Класс', 'ID', 'IoU', 'Dice', 'Precision', 'Recall', 'TP', 'FP', 'FN'])

        for c in CONFIG["focus_classes"]:
            tp = all_stats[c]["tp"]
            fp = all_stats[c]["fp"]
            fn = all_stats[c]["fn"]
            iou = tp / (tp + fp + fn + 1e-8)
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            class_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
            writer.writerow([class_name, c, f"{iou:.4f}", f"{dice:.4f}", f"{prec:.4f}", f"{rec:.4f}", tp, fp, fn])

        writer.writerow(['MEAN', '-', f"{np.mean(ious):.4f}", f"{np.mean(dices):.4f}", '-', '-', '-', '-', '-'])

    print(f"\n💾 CSV сохранён: {csv_path}")

    # 7. Confusion Matrix
    all_p = np.concatenate(all_preds_flat)
    all_g = np.concatenate(all_gts_flat)

    labels = [0] + CONFIG["focus_classes"]
    label_names = ['Other/BG'] + [CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}" for c in
                                  CONFIG["focus_classes"]]

    cm = confusion_matrix(all_g, all_p, labels=labels)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Recall (Normalized)'})
    plt.title('Ensemble Confusion Matrix (Soft Voting + TTA)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Ground Truth', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "ensemble_confusion_matrix.png"), dpi=150)
    plt.close()

    print(f"💾 Confusion Matrix: {os.path.join(CONFIG['output_dir'], 'ensemble_confusion_matrix.png')}")

    # 8. Сравнение
    print("\n" + "=" * 80)
    print(" СРАВНЕНИЕ: Одиночные модели vs Ансамбль")
    print("=" * 80)
    print("Одиночные модели:")
    print("  U-Net:     Mean Dice = 0.7935")
    print("  DeepLabV3+: Mean Dice = 0.7953")
    print("  U-Net++:   Mean Dice = 0.7852")
    print(f"\n🎯 Ансамбль: Mean Dice = {np.mean(dices):.4f}")
    print("=" * 80)

    print("\n✅ Оценка ансамбля завершена!")


if __name__ == "__main__":
    main()