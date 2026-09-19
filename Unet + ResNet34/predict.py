import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

from models import get_model

# === КОНФИГУРАЦИЯ ===
CONFIG = {
    "checkpoint_path": "checkpoints/best_model_unet.pth",
    "input_dir": "C:/Users/Home/PycharmProjects/Nir/archive/BCSS_512/val_512",
    # Поменяйте на test_512 при необходимости
    "output_dir": "predictions",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # 5 ключевых классов: Tumor, Stroma, Lymphocytic, Necrosis, Fat
    "focus_classes": [1, 2, 3, 4, 9],
}

# Цветовая карта для визуализации (RGB)
CLASS_COLORS = {
    0: (0, 0, 0),  # Фон / Игнорируемые / Остальные
    1: (255, 0, 0),  # Tumor - Красный
    2: (0, 255, 0),  # Stroma - Зеленый
    3: (0, 0, 255),  # Lymphocytic - Синий
    4: (255, 255, 0),  # Necrosis - Желтый
    9: (255, 0, 255),  # Fat - Фиолетовый
}

CLASS_NAMES_MAP = {
    0: 'Other/Ignore', 1: 'Tumor', 2: 'Stroma',
    3: 'Lymphocytic', 4: 'Necrosis', 9: 'Fat'
}


def get_val_transform():
    """Точная копия валидационных трансформов из train.py"""
    return Compose([
        Resize(height=512, width=512),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})


@torch.no_grad()
def tta_predict(model, tensor_img, device):
    """
    TTA: 5 преобразований + усреднение логитов.
    Работает с тензором формы [1, 3, H, W]
    """
    logits_list = []

    # 1. Original
    out = model(tensor_img)
    logits_list.append(out)

    # 2. Horizontal Flip
    out = model(torch.flip(tensor_img, dims=[3]))
    logits_list.append(torch.flip(out, dims=[3]))

    # 3. Vertical Flip
    out = model(torch.flip(tensor_img, dims=[2]))
    logits_list.append(torch.flip(out, dims=[2]))

    # 4. Rotate 90 (CCW)
    out = model(torch.rot90(tensor_img, k=1, dims=[2, 3]))
    logits_list.append(torch.rot90(out, k=3, dims=[2, 3]))  # Inverse: k=3

    # 5. H-Flip + Rotate 90
    aug = torch.flip(torch.rot90(tensor_img, k=1, dims=[2, 3]), dims=[3])
    out = model(aug)
    out = torch.rot90(torch.flip(out, dims=[3]), k=3, dims=[2, 3])
    logits_list.append(out)

    # Усредняем логиты и берем argmax
    avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)
    pred_mask = torch.argmax(avg_logits, dim=1).squeeze(0).cpu().numpy()
    return pred_mask


def visualize_and_save(pred_mask, orig_img_bgr, save_path, filename):
    """Создает цветную маску, оверлей и plot с легендой"""
    h, w = pred_mask.shape

    # Фильтруем: оставляем только 5 целевых классов, остальные -> 0
    focus_mask = np.zeros_like(pred_mask)
    for cls in CONFIG["focus_classes"]:
        focus_mask[pred_mask == cls] = cls

    # Генерируем RGB маску
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        color_mask[focus_mask == cls] = color

    # Альфа-блендинг для наложения
    overlay = cv2.addWeighted(orig_img_bgr, 0.7, color_mask, 0.3, 0)

    # Сохраняем результаты
    cv2.imwrite(os.path.join(save_path, f"mask_{filename}"), color_mask)
    cv2.imwrite(os.path.join(save_path, f"overlay_{filename}"), overlay)

    # Генерируем красивый plot с легендой
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    patches = [mpatches.Patch(color=np.array(c) / 255, label=CLASS_NAMES_MAP[k])
               for k, c in CLASS_COLORS.items()]
    plt.legend(handles=patches, loc='upper right', fontsize=10)
    plt.axis('off')
    plt.title(f"TTA Prediction: {filename}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"plot_{filename}"), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("🔮 ИНФЕРЕНС С TTA + ФОКУС НА 5 КЛАССАХ")
    print("=" * 60)

    device = torch.device(CONFIG["device"])
    print(f"🖥️ Устройство: {device}")

    # 1. Загрузка модели
    print("📦 Загрузка модели...")
    model = get_model(num_classes=22, encoder="resnet34").to(device)
    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Модель загружена (Epoch {checkpoint['epoch']}, Best Dice: {checkpoint['best_val_dice']:.4f})")

    # 2. Подготовка
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    img_files = [f for f in os.listdir(CONFIG["input_dir"]) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"📁 Найдено изображений: {len(img_files)}")

    transform = get_val_transform()

    # 3. Цикл инференса
    for filename in tqdm(img_files, desc="🚀 Инференс с TTA"):
        img_path = os.path.join(CONFIG["input_dir"], filename)
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Нормализация и тензор
        transformed = transform(image=img_rgb)
        tensor_img = transformed['image'].unsqueeze(0).to(device)

        # TTA предсказание
        pred_mask = tta_predict(model, tensor_img, device)

        # Визуализация
        base_name = os.path.splitext(filename)[0] + ".png"
        visualize_and_save(pred_mask, img_bgr, CONFIG["output_dir"], base_name)

    print(f"\n💾 Результаты сохранены в: {os.path.abspath(CONFIG['output_dir'])}")
    print("✅ Готово!")


if __name__ == "__main__":
    main()