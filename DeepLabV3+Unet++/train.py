import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import time
import argparse
import torch
import torch.optim as optim
from torch.amp import GradScaler
from tqdm import tqdm
import csv
from datetime import datetime
from modells.losses_metrics import CombinedLoss, MetricsAccumulator, compute_class_weights
from models import get_model, count_parameters
from dataset import get_dataloaders

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

CONFIG = {
    "data_root": "C:/Users/Home/PycharmProjects/Nir/archive/BCSS_512",
    "img_size": (512, 512),
    "batch_size": 8,
    "num_workers": 2,
    "num_classes": 22,
    "ignore_index": 255,
    "epochs": 80,
    "learning_rate": 1e-3,
    "encoder_lr_mult": 0.1,
    "weight_decay": 1e-5,
    "patience": 25,
    "checkpoint_dir": "checkpoints",
    "log_dir": "logs",
    "seed": 42,
    "use_amp": True,
    "grad_clip": 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--arch", type=str, default="unet",
                        choices=["unet", "deeplab", "unetpp"],
                        help="Архитектура модели (unet, deeplab, unetpp)")
    parser.add_argument("--encoder", type=str, default="resnet34",
                        help="Имя энкодера (resnet34, efficientnet-b3, resnet50 и т.д.)")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, epoch):
    model.train()
    running_loss = 0.0
    num_batches = 0
    num_skipped = 0
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False, unit="batch")

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        valid_pixels = (masks != CONFIG["ignore_index"]).sum()
        if valid_pixels == 0:
            num_skipped += 1
            continue

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)

        if not torch.isfinite(loss):
            num_skipped += 1
            if use_amp and scaler is not None:
                scaler.update()
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG["grad_clip"])
            optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if num_skipped > 0:
        print(f"   ⚠️ Пропущено батчей: {num_skipped}/{len(train_loader)}")
    return running_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, val_loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    accum = MetricsAccumulator(CONFIG["num_classes"], CONFIG["ignore_index"])

    for images, masks in tqdm(val_loader, desc="Val", leave=False, unit="batch"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
        loss = criterion(outputs, masks)

        running_loss += loss.item()
        accum.update(outputs, masks)

    return running_loss / len(val_loader), accum.get_metrics()


def main():
    args = parse_args()
    arch_name = args.arch
    encoder_name = args.encoder

    print("=" * 70)
    print(f"🚀 ОБУЧЕНИЕ: {arch_name.upper()} + {encoder_name}")
    print("=" * 70)

    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = CONFIG["use_amp"] and device.type == "cuda"
    print(f"🖥️  Устройство: {device} | AMP: {'✅ ON' if use_amp else '⛔ OFF'}")

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    train_loader, val_loader = get_dataloaders(
        data_root=CONFIG["data_root"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        img_size=CONFIG["img_size"]
    )

    model = get_model(
        architecture=arch_name,
        encoder=encoder_name,
        num_classes=CONFIG["num_classes"]
    ).to(device)
    print(f"📐 Модель: {model.__class__.__name__} ({count_parameters(model):,} параметров)")

    class_weights = compute_class_weights(
        os.path.join(CONFIG["data_root"], "train_mask_512"),
        num_classes=CONFIG["num_classes"],
        ignore_index=CONFIG["ignore_index"],
        max_weight=10.0
    )
    if class_weights is not None:
        rare_multiplier = 5.0
        rare_indices = list(range(5, CONFIG["num_classes"]))
        class_weights[rare_indices] *= rare_multiplier
        class_weights = torch.clamp(class_weights, max=10.0)
        class_weights = class_weights.to(device)
        print(f"   Веса классов после увеличения (max=10): {class_weights[1:].tolist()}")

    criterion = CombinedLoss(
        num_classes=CONFIG["num_classes"],
        ignore_index=CONFIG["ignore_index"],
        ce_weight=0.3,
        dice_weight=0.7,
        class_weights=class_weights,
        use_focal=False
    ).to(device)

    if hasattr(model, 'encoder'):
        encoder_params = list(model.encoder.parameters())
        decoder_params = [p for n, p in model.named_parameters() if not n.startswith('encoder')]
        optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': CONFIG["learning_rate"] * CONFIG["encoder_lr_mult"], 'name': 'encoder'},
            {'params': decoder_params, 'lr': CONFIG["learning_rate"], 'name': 'decoder'}
        ], weight_decay=CONFIG["weight_decay"])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(CONFIG["log_dir"], f"train_log_{arch_name}_{timestamp}.csv")
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_Mean_IoU', 'Val_Mean_Dice', 'LR'])

    print(f"\n📊 Лог: {log_path}")
    print(f"📦 Батчей за эпоху: {len(train_loader)}")
    print(f"⏳ Эпох: {CONFIG['epochs']} | Patience: {CONFIG['patience']}\n")

    best_val_dice = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, epoch
        )

        val_loss, metrics = validate(model, val_loader, criterion, device, use_amp)
        val_iou, val_dice = metrics['mean_iou'], metrics['mean_dice']

        scheduler.step()

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        per_class_iou = metrics['per_class_iou'][1:22]
        print(f"\nEpoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} |  "
              f"IoU: {val_iou:.4f} | Dice: {val_dice:.4f} |  "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        print(f"   Per-class IoU (1-21): {per_class_iou}")

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_iou, val_dice, current_lr])

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], f"best_model_{arch_name}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_dice': best_val_dice,
                'config': CONFIG,
                'architecture': arch_name,
                'encoder': encoder_name
            }, checkpoint_path)
            print(f"💾 Сохранена лучшая модель (Epoch {epoch}, Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1
            print(f"   EarlyStopping: {patience_counter}/{CONFIG['patience']}")
            if patience_counter >= CONFIG["patience"]:
                print(f"\n⏹️ Early stopping на эпохе {epoch}")
                break

    print("\n" + "=" * 70)
    print(f"✅ ГОТОВО! {arch_name.upper()} | Лучший Val Dice: {best_val_dice:.4f} (Epoch {best_epoch})")
    print("=" * 70)


if __name__ == '__main__':
    main()