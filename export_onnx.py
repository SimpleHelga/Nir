import os
import torch
import sys

# Добавляем пути к модулям
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DeepLabV3+Unet++"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Unet + ResNet34"))

from modells.models import get_model

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "unet": {
        "ckpt": os.path.join(PROJECT_ROOT, "Unet + ResNet34", "checkpoints", "best_model_unet.pth"),
        "arch": "unet",
        "encoder": "resnet34",
        "output": os.path.join(PROJECT_ROOT, "exported_models", "unet_resnet34.onnx"),
    },
    "deeplab": {
        "ckpt": os.path.join(PROJECT_ROOT, "DeepLabV3+Unet++", "checkpoints", "best_model_deeplab.pth"),
        "arch": "deeplab",
        "encoder": "efficientnet-b3",
        "output": os.path.join(PROJECT_ROOT, "exported_models", "deeplab_efficientnet_b3.onnx"),
    },
    "unetpp": {
        "ckpt": os.path.join(PROJECT_ROOT, "DeepLabV3+Unet++", "checkpoints", "best_model_unetpp.pth"),
        "arch": "unetpp",
        "encoder": "resnet50",
        "output": os.path.join(PROJECT_ROOT, "exported_models", "unetpp_resnet50.onnx"),
    },
}


def export_model(name, config):
    print(f"\n{'=' * 60}")
    print(f"📦 Экспорт {name.upper()} ({config['arch']} + {config['encoder']})")
    print(f"{'=' * 60}")

    # Загрузка модели
    device = torch.device("cpu")
    model = get_model(
        num_classes=22,
        encoder=config["encoder"],
        architecture=config["arch"]
    ).to(device)

    # Загрузка весов
    ckpt = torch.load(config["ckpt"], map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f"✅ Модель загружена")

    # Создание dummy input
    dummy_input = torch.randn(1, 3, 512, 512, device=device)

    # Экспорт в ONNX
    os.makedirs(os.path.dirname(config["output"]), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        config["output"],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )

    # Проверка размера
    size_mb = os.path.getsize(config["output"]) / (1024 * 1024)
    print(f"💾 Сохранено: {config['output']}")
    print(f"📏 Размер: {size_mb:.2f} MB")
    print(f"✅ Экспорт завершён!")


def main():
    print("=" * 60)
    print("🎯 ЭКСПОРТ МОДЕЛЕЙ В ONNX")
    print("=" * 60)

    for name, config in MODELS.items():
        if not os.path.exists(config["ckpt"]):
            print(f"⚠️  Чекпоинт не найден: {config['ckpt']}")
            continue
        export_model(name, config)

    print("\n" + "=" * 60)
    print("✅ ВСЕ МОДЕЛИ ЭКСПОРТИРОВАНЫ!")
    print("=" * 60)
    print("\n📁 Экспортированные модели:")
    for name, config in MODELS.items():
        if os.path.exists(config["output"]):
            size_mb = os.path.getsize(config["output"]) / (1024 * 1024)
            print(f"  • {name}: {config['output']} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()