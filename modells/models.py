# models.py
import torch
import segmentation_models_pytorch as smp

# Попробуем импортировать кастомные модели, но не требуем их
try:
    from other_models import M3NetPlusPlus, HoverNet, DMFNet, DenseUNet, LightweightUNet
    CUSTOM_MODELS_AVAILABLE = True
except ImportError:
    CUSTOM_MODELS_AVAILABLE = False
    print("⚠️ Кастомные модели недоступны - используем только SMP архитектуры")


def get_model(num_classes: int = 22, encoder: str = "resnet34", architecture: str = "unet") -> torch.nn.Module:
    arch = architecture.lower().strip()

    # SMP модели (всегда доступны)
    if arch == "unet":
        return smp.Unet(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=num_classes)
    elif arch in ("deeplab", "deeplabv3plus", "deeplabv3+"):
        return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=num_classes)
    elif arch in ("unetpp", "unet++", "unetplusplus"):
        return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=num_classes)

    # Кастомные модели (если доступны)
    elif CUSTOM_MODELS_AVAILABLE:
        if arch == "m3net":
            return M3NetPlusPlus(n_channels=3, n_classes=num_classes)
        elif arch == "hovernet":
            return HoverNet(n_channels=3, n_classes=num_classes)
        elif arch == "dmfnet":
            return DMFNet(n_channels=3, n_classes=num_classes)
        elif arch == "denseunet":
            return DenseUNet(n_channels=3, n_classes=num_classes)
        elif arch == "lightweight":
            return LightweightUNet(n_channels=3, n_classes=num_classes)
        else:
            raise ValueError(f"❌ Неизвестная архитектура: {architecture}")
    else:
        raise ValueError(f"❌ Неизвестная архитектура: {architecture}. "
                        f"Доступны: unet, deeplab, unetpp")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)