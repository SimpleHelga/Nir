import torch
import segmentation_models_pytorch as smp


def get_model(num_classes: int = 22, encoder: str = "resnet34",
              architecture: str = "unet") -> torch.nn.Module:
    """
    Универсальная фабрика моделей для ансамбля.

    Поддерживаемые архитектуры:
      - "unet"      : классический U-Net
      - "deeplab"   : DeepLabV3+ (atrous convolutions)
      - "unetpp"    : U-Net++ (dense skip connections)
    """
    arch = architecture.lower().strip()

    if arch == "unet":
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    elif arch in ("deeplab", "deeplabv3plus", "deeplabv3+"):
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    elif arch in ("unetpp", "unet++", "unetplusplus"):
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    else:
        raise ValueError(f"❌ Неизвестная архитектура: {architecture}. "
                         f"Доступны: unet, deeplab, unetpp")

    return model


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)