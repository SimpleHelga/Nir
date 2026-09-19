"""
Менеджер моделей для сегментации тканей
"""
import os
import sys
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Добавляем пути к модулям
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)  # для пакета modells при запуске не из корня проекта
sys.path.insert(0, os.path.join(PROJECT_ROOT, "DeepLabV3+Unet++"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "Unet + ResNet34"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "modells"))

from modells.models import get_model


class ModelManager:
    """Менеджер для загрузки и использования моделей сегментации"""
    
    CLASS_NAMES = [
        'outside_roi',          # 0
        'tumor',                # 1
        'stroma',               # 2
        'lymphocytic_infiltrate', # 3
        'necrosis_or_debris',   # 4
        'glandular_secretions', # 5
        'blood',                # 6
        'exclude',              # 7
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
    
    # Цветовая карта для визуализации классов
    CLASS_COLORS = {
        0: (0, 0, 0),           # outside_roi - черный
        1: (255, 0, 0),         # tumor - красный
        2: (255, 165, 0),       # stroma - оранжевый
        3: (0, 255, 0),         # lymphocytic_infiltrate - зеленый
        4: (128, 0, 128),       # necrosis - фиолетовый
        5: (0, 255, 255),       # glandular_secretions - голубой
        6: (200, 0, 0),         # blood - темно-красный
        7: (64, 64, 64),        # exclude - темно-серый
        8: (255, 255, 0),       # metaplasia - желтый
        9: (192, 192, 0),       # fat - оливковый
        10: (0, 128, 128),      # plasma_cells - темно-голубой
        11: (128, 128, 0),      # other_immune - темно-желтый
        12: (255, 192, 203),    # mucoid_material - розовый
        13: (0, 255, 128),      # normal_acinus - светло-зеленый
        14: (128, 255, 0),      # lymphatics - зелено-желтый
        15: (128, 128, 128),    # undetermined - серый
        16: (100, 149, 237),    # nerve - голубовато
        17: (218, 112, 214),    # skin_adnexa - орхидея
        18: (255, 20, 147),     # blood_vessel - глубокий розовый
        19: (220, 20, 60),      # angioinvasion - алый
        20: (210, 105, 30),     # dcis - шоколадный
        21: (169, 169, 169),    # other - темный серый
    }
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.model_configs = {
            "unet": {
                "arch": "unet",
                "encoder": "resnet34",
                "ckpt": os.path.join(PROJECT_ROOT, "Unet + ResNet34", "checkpoints", "best_model_unet.pth")
            },
            "deeplab": {
                "arch": "deeplab",
                "encoder": "efficientnet-b3",
                "ckpt": os.path.join(PROJECT_ROOT, "DeepLabV3+Unet++", "checkpoints", "best_model_deeplab.pth")
            },
            "unetpp": {
                "arch": "unetpp",
                "encoder": "resnet50",
                "ckpt": os.path.join(PROJECT_ROOT, "DeepLabV3+Unet++", "checkpoints", "best_model_unetpp.pth")
            },
        }
    
    def load_model(self, model_name: str) -> bool:
        """Загрузить модель"""
        if model_name in self.models:
            return True
            
        if model_name not in self.model_configs:
            print(f"❌ Неизвестная модель: {model_name}")
            return False
        
        cfg = self.model_configs[model_name]
        
        if not os.path.exists(cfg["ckpt"]):
            print(f"❌ Чекпоинт не найден: {cfg['ckpt']}")
            return False
        
        try:
            print(f"📦 Загрузка {model_name}...")
            model = get_model(
                num_classes=22,
                encoder=cfg["encoder"],
                architecture=cfg["arch"]
            ).to(self.device)
            
            checkpoint = torch.load(cfg["ckpt"], map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            self.models[model_name] = model
            print(f"✅ Модель {model_name} загружена")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки {model_name}: {e}")
            return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """Загрузить все модели"""
        results = {}
        for model_name in self.model_configs.keys():
            results[model_name] = self.load_model(model_name)
        return results
    
    def get_transform(self):
        """Получить трансформацию для инпута"""
        return A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    @torch.no_grad()
    def predict_single(self, image_path: str, model_name: str = "unet", 
                      use_tta: bool = False) -> Optional[np.ndarray]:
        """
        Предсказание маски для одного изображения
        
        Args:
            image_path: путь к изображению
            model_name: имя модели
            use_tta: использовать Test-Time Augmentation
            
        Returns:
            Массив предсказаний (H, W) с индексами классов
        """
        if model_name not in self.models:
            if not self.load_model(model_name):
                return None
        
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Не удалось загрузить изображение: {image_path}")
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]
        
        # Трансформация
        transform = self.get_transform()
        transformed = transform(image=image)
        tensor_img = transformed['image'].unsqueeze(0).to(self.device)
        
        model = self.models[model_name]
        
        if use_tta:
            probs = self._tta_predict(model, tensor_img)
        else:
            output = model(tensor_img)
            probs = torch.softmax(output, dim=1)
        
        # Получить маску (индекс класса с максимальной вероятностью)
        mask = torch.argmax(probs[0], dim=0).cpu().numpy()
        
        # Вернуть к оригинальному размеру
        mask = cv2.resize(mask.astype(np.uint8), (original_shape[1], original_shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    @torch.no_grad()
    def predict_ensemble(self, image_path: str, use_tta: bool = False,
                        weights: Optional[Dict[str, float]] = None) -> Optional[np.ndarray]:
        """
        Предсказание с ансамблем всех моделей
        
        Args:
            image_path: путь к изображению
            use_tta: использовать TTA
            weights: веса моделей (если None, то равные)
            
        Returns:
            Маска с голосованием ансамбля
        """
        if weights is None:
            weights = {name: 1.0 for name in self.model_configs.keys()}
        
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]
        
        # Трансформация
        transform = self.get_transform()
        transformed = transform(image=image)
        tensor_img = transformed['image'].unsqueeze(0).to(self.device)
        
        # Собрать вероятности от всех моделей
        all_probs = []
        all_weights = []
        
        for model_name, weight in weights.items():
            if model_name not in self.models:
                if not self.load_model(model_name):
                    continue
            
            model = self.models[model_name]
            
            if use_tta:
                probs = self._tta_predict(model, tensor_img)
            else:
                output = model(tensor_img)
                probs = torch.softmax(output, dim=1)
            
            all_probs.append(probs)
            all_weights.append(weight)
        
        if not all_probs:
            return None
        
        # Взвешенное усреднение
        all_probs = torch.stack(all_probs, dim=0)  # (n_models, 1, 22, 512, 512)
        weights_tensor = torch.tensor(all_weights, device=self.device).view(-1, 1, 1, 1, 1)
        
        avg_probs = (all_probs * weights_tensor).sum(dim=0) / sum(all_weights)
        
        # Получить маску
        mask = torch.argmax(avg_probs[0], dim=0).cpu().numpy()
        
        # Вернуть к оригинальному размеру
        mask = cv2.resize(mask.astype(np.uint8), (original_shape[1], original_shape[0]),
                         interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    @torch.no_grad()
    def _tta_predict(self, model, tensor_img):
        """Test-Time Augmentation: применяет 5 трансформаций и усредняет"""
        probs = []
        
        # Оригинальное изображение
        probs.append(torch.softmax(model(tensor_img), dim=1))
        
        # Горизонтальный флип
        out = model(torch.flip(tensor_img, dims=[3]))
        probs.append(torch.flip(torch.softmax(out, dim=1), dims=[3]))
        
        # Вертикальный флип
        out = model(torch.flip(tensor_img, dims=[2]))
        probs.append(torch.flip(torch.softmax(out, dim=1), dims=[2]))
        
        # Ротация 90 градусов
        out = model(torch.rot90(tensor_img, k=1, dims=[2, 3]))
        probs.append(torch.rot90(torch.softmax(out, dim=1), k=3, dims=[2, 3]))
        
        # Комбинация флипа и ротации
        aug = torch.flip(torch.rot90(tensor_img, k=1, dims=[2, 3]), dims=[3])
        out = model(aug)
        out = torch.rot90(torch.flip(torch.softmax(out, dim=1), dims=[3]), k=3, dims=[2, 3])
        probs.append(out)
        
        return torch.stack(probs, dim=0).mean(dim=0)
    
    def mask_to_image(self, mask: np.ndarray) -> np.ndarray:
        """Конвертировать маску классов в RGB изображение"""
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.CLASS_COLORS.items():
            colored_mask[mask == class_id] = color
        
        return colored_mask
    
    def overlay_mask(self, image: np.ndarray, mask: np.ndarray, 
                    alpha: float = 0.5) -> np.ndarray:
        """Наложить маску на изображение"""
        colored_mask = self.mask_to_image(mask)
        return cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    def get_class_name(self, class_id: int) -> str:
        """Получить название класса по ID"""
        if 0 <= class_id < len(self.CLASS_NAMES):
            return self.CLASS_NAMES[class_id]
        return f"Unknown ({class_id})"
