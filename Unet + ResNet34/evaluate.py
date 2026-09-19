import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix

from models import get_model

# === КОНФИГУРАЦИЯ ===
CONFIG = {
    "checkpoint_path": "checkpoints/best_model_unet.pth",
    "images_dir": "C:/Users/Home/PycharmProjects/Nir/archive/BCSS_512/val_512",
    "masks_dir": "C:/Users/Home/PycharmProjects/Nir/archive/BCSS_512/val_mask_512",
    "output_dir": "evaluation_results",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "focus_classes": [1, 2, 3, 4, 9],  # Tumor, Stroma, Lymph, Necrosis, Fat
}

CLASS_NAMES = {
    1: 'Tumor',
    2: 'Stroma',
    3: 'Lymphocytic',
    4: 'Necrosis',
    9: 'Fat',
}

CLASS_MAPPING_GT = {0: 255, 7: 255}  # как в dataset.py


def get_val_transform():
    return Compose([
        Resize(height=512, width=512),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


@torch.no_grad()
def tta_predict(model, tensor_img):
    """TTA: 5 преобразований + усреднение softmax вероятностей"""
    probs_list = []

    # 1. Original
    out = torch.softmax(model(tensor_img), dim=1)
    probs_list.append(out)

    # 2. Horizontal Flip
    out = model(torch.flip(tensor_img, dims=[3]))
    probs_list.append(torch.flip(torch.softmax(out, dim=1), dims=[3]))

    # 3. Vertical Flip
    out = model(torch.flip(tensor_img, dims=[2]))
    probs_list.append(torch.flip(torch.softmax(out, dim=1), dims=[2]))

    # 4. Rotate 90
    out = model(torch.rot90(tensor_img, k=1, dims=[2, 3]))
    probs_list.append(torch.rot90(torch.softmax(out, dim=1), k=3, dims=[2, 3]))

    # 5. H-Flip + Rotate 90
    aug = torch.flip(torch.rot90(tensor_img, k=1, dims=[2, 3]), dims=[3])
    out = model(aug)
    out = torch.rot90(torch.flip(torch.softmax(out, dim=1), dims=[3]), k=3, dims=[2, 3])
    probs_list.append(out)

    avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)
    return torch.argmax(avg_probs, dim=1).squeeze(0).cpu().numpy()


def compute_metrics(pred_mask, gt_mask, focus_classes):
    """Возвращает TP, FP, FN для каждого целевого класса"""
    results = {}
    for cls in focus_classes:
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)

        tp = np.logical_and(pred_cls, gt_cls).sum()
        fp = np.logical_and(pred_cls, ~gt_cls).sum()
        fn = np.logical_and(~pred_cls, gt_cls).sum()

        results[cls] = {'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}
    return results


def print_results_table(accumulated_stats):
    """Красивый вывод итоговой таблицы метрик"""
    print("\n" + "=" * 85)
    print("📊 ИТОГОВЫЕ МЕТРИКИ ПО 5 КЛЮЧЕВЫМ КЛАССАМ (TTA + Val Set)")
    print("=" * 85)
    print(
        f"{'Класс':<15} {'IoU':<10} {'Dice':<10} {'Precision':<12} {'Recall':<10} {'TP (px)':<12} {'FP (px)':<12} {'FN (px)':<12}")
    print("-" * 85)

    ious, dices = [], []
    for cls, stats in accumulated_stats.items():
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']

        iou = tp / (tp + fp + fn + 1e-8)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        ious.append(iou)
        dices.append(dice)

        name = f"{cls} ({CLASS_NAMES[cls]})"
        print(
            f"{name:<15} {iou:<10.4f} {dice:<10.4f} {precision:<12.4f} {recall:<10.4f} {tp:<12,d} {fp:<12,d} {fn:<12,d}")

    print("-" * 85)
    print(f"{'MEAN':<15} {np.mean(ious):<10.4f} {np.mean(dices):<10.4f}")
    print("=" * 85)
    return ious, dices


def plot_confusion_matrix(pred_flat, gt_flat, focus_classes, output_dir):
    """Строит и сохраняет матрицу ошибок для 5 классов + фон"""
    # Оставляем только интересующие классы, остальное -> 0 (Other/Background)
    labels = [0] + focus_classes
    label_names = ['Other/BG'] + [CLASS_NAMES[c] for c in focus_classes]

    # Фильтруем: если пиксель не из labels, превращаем в 0
    pred_filtered = np.where(np.isin(pred_flat, labels), pred_flat, 0)
    gt_filtered = np.where(np.isin(gt_flat, labels), gt_flat, 0)

    cm = confusion_matrix(gt_filtered, pred_filtered, labels=labels)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={'label': 'Recall (normalized)'})
    plt.title('Confusion Matrix (TTA, Val Set) — Focus 5 Classes', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Ground Truth', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"💾 Confusion Matrix сохранена: {os.path.join(output_dir, 'confusion_matrix.png')}")


def main():
    print("=" * 70)
    print("📈 ОЦЕНКА КАЧЕСТВА: per-class метрики + Confusion Matrix")
    print("=" * 70)

    device = torch.device(CONFIG["device"])
    print(f"🖥️ Устройство: {device}")

    # 1. Загрузка модели
    model = get_model(num_classes=22, encoder="resnet34").to(device)
    ckpt = torch.load(CONFIG["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"✅ Модель: Epoch {ckpt['epoch']}, Best Dice: {ckpt['best_val_dice']:.4f}")

    # 2. Сбор файлов
    img_files = sorted([f for f in os.listdir(CONFIG["images_dir"])
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"📁 Изображений для оценки: {len(img_files)}")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    transform = get_val_transform()

    # 3. Аккумуляторы статистики
    stats = {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in CONFIG["focus_classes"]}
    all_preds, all_gts = [], []

    # 4. Цикл оценки
    for filename in tqdm(img_files, desc="🔍 Оценка с TTA"):
        img_path = os.path.join(CONFIG["images_dir"], filename)
        base = os.path.splitext(filename)[0]
        mask_path = os.path.join(CONFIG["masks_dir"], base + '.png')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(CONFIG["masks_dir"], filename)

        if not os.path.exists(mask_path):
            continue

        # Читаем изображение и GT маску
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Применяем тот же маппинг, что и в dataset.py
        for orig, target in CLASS_MAPPING_GT.items():
            gt_mask[gt_mask == orig] = target
        gt_mask = np.where((gt_mask > 21) & (gt_mask != 255), 255, gt_mask)

        # Инференс
        transformed = transform(image=img_rgb)
        tensor_img = transformed['image'].unsqueeze(0).to(device)
        pred_mask = tta_predict(model, tensor_img)

        # Маска валидных пикселей (исключаем ignore_index=255)
        valid = (gt_mask != 255)
        pred_valid = pred_mask[valid]
        gt_valid = gt_mask[valid]

        # Обновляем статистику
        img_stats = compute_metrics(pred_valid, gt_valid, CONFIG["focus_classes"])
        for cls in CONFIG["focus_classes"]:
            stats[cls]['tp'] += img_stats[cls]['tp']
            stats[cls]['fp'] += img_stats[cls]['fp']
            stats[cls]['fn'] += img_stats[cls]['fn']

        all_preds.append(pred_valid)
        all_gts.append(gt_valid)

    # 5. Вывод таблицы
    ious, dices = print_results_table(stats)

    # 6. Confusion Matrix
    plot_confusion_matrix(
        np.concatenate(all_preds),
        np.concatenate(all_gts),
        CONFIG["focus_classes"],
        CONFIG["output_dir"]
    )

    # 7. Сохранение CSV для отчёта
    csv_path = os.path.join(CONFIG["output_dir"], "metrics_table.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Class,ID,IoU,Dice,Precision,Recall,TP,FP,FN\n")
        for cls in CONFIG["focus_classes"]:
            tp, fp, fn = stats[cls]['tp'], stats[cls]['fp'], stats[cls]['fn']
            iou = tp / (tp + fp + fn + 1e-8)
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f.write(f"{CLASS_NAMES[cls]},{cls},{iou:.4f},{dice:.4f},{prec:.4f},{rec:.4f},{tp},{fp},{fn}\n")
        f.write(f"MEAN,,-,{np.mean(ious):.4f},{np.mean(dices):.4f},-,-,-,-\n")
    print(f"💾 CSV таблица: {csv_path}")
    print("\n✅ Оценка завершена!")


if __name__ == "__main__":
    main()