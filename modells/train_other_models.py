#!/usr/bin/env python3
"""
Обучение M3-Net++ и HoVer-Net на BCSS.
CSV-логи: Epoch,Train_Loss,Val_Loss,Val_Mean_IoU,Val_Mean_Dice,LR
"""

import os
import time
import argparse
import csv
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset import BCSSDataset, get_dataloaders
from other_models import M3NetPlusPlus, HoVerNetAdapted
from losses_metrics import HybridLoss, HoVerNetLoss, compute_iou_dice


# ---------- Wrapper для HoVer-Net (генерация hv_map + boundary) ----------
class HVWrapper(Dataset):
    def __init__(self, base_dataset: Dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def _compute_hv_boundary(self, mask: np.ndarray):
        h, w = mask.shape
        hv = np.zeros((2, h, w), dtype=np.float32)
        boundary = np.zeros((h, w), dtype=np.float32)

        classes = np.unique(mask)
        for cls in classes:
            if cls == 255:
                continue
            binary = (mask == cls).astype(np.float32)
            gx = np.zeros_like(binary)
            gy = np.zeros_like(binary)
            gx[:, 1:-1] = binary[:, 2:] - binary[:, :-2]
            gy[1:-1, :] = binary[2:, :] - binary[:-2, :]
            hv[0] += gx
            hv[1] += gy
            boundary += (np.abs(gx) + np.abs(gy)) > 0

        boundary = np.clip(boundary, 0, 1).astype(np.float32)
        return hv, boundary

    def __getitem__(self, idx):
        img, mask = self.base[idx]
        mask_np = mask.cpu().numpy()
        hv, boundary = self._compute_hv_boundary(mask_np)
        return img, mask.long(), torch.from_numpy(hv), torch.from_numpy(boundary).float()


# ---------- Аргументы ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Train M3-Net++ or HoVer-Net")
    parser.add_argument("--arch", type=str, default="m3net",
                        choices=["m3net", "hovernet"],
                        help="Архитектура (m3net, hovernet)")
    parser.add_argument("--encoder", type=str, default="resnet50",
                        choices=["resnet34", "resnet50"],
                        help="Энкодер (resnet34, resnet50)")
    return parser.parse_args()


CONFIG = {
    "data_root": "C:/Users/Home/PycharmProjects/Nir/archive/BCSS_512",
    "img_size": (512, 512),
    "batch_size": 8,
    "num_workers": 4,
    "num_classes": 22,
    "ignore_index": 255,
    "epochs": 50,               # ← больше эпох
    "learning_rate": 5e-4,      # ← ниже (было 1e-3)
    "encoder_lr_mult": 0.1,
    "weight_decay": 1e-3,       # ← выше (было 5e-4)
    "patience": 15,             # ← больше терпения
    "checkpoint_dir": "checkpoints",
    "log_dir": "logs",
    "seed": 42,
    "use_amp": True,
    "grad_clip": 1.0,
}


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, epoch, model_type):
    model.train()
    running_loss = 0.0
    num_batches = 0
    num_skipped = 0
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False, unit="batch")

    for batch in pbar:
        if model_type == "hovernet":
            images, masks, hv, boundary = batch
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            hv = hv.to(device, non_blocking=True)
            boundary = boundary.to(device, non_blocking=True).unsqueeze(1)

            valid_pixels = (masks != CONFIG["ignore_index"]).sum()
            if valid_pixels == 0:
                num_skipped += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, masks, hv, boundary)
        else:
            images, masks = batch
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
def validate(model, val_loader, criterion, device, use_amp, model_type):
    model.eval()
    running_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    count = 0
    num_skipped = 0

    for batch in tqdm(val_loader, desc="Val", leave=False, unit="batch"):
        if model_type == "hovernet":
            images, masks, hv, boundary = batch
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            hv = hv.to(device, non_blocking=True)
            boundary = boundary.to(device, non_blocking=True).unsqueeze(1)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
            loss = criterion(outputs, masks, hv, boundary)
            pred = outputs["semantic"]
        else:
            images, masks = batch
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
            pred = outputs["out"] if isinstance(outputs, dict) else outputs
            loss = criterion(outputs, masks)

        # ← НОВОЕ: пропускаем батчи без полезных пикселей
        valid_pixels = (masks != CONFIG["ignore_index"]).sum()
        if valid_pixels == 0:
            num_skipped += 1
            continue

        running_loss += loss.item()
        iou, dice = compute_iou_dice(pred, masks, num_classes=CONFIG["num_classes"], ignore_index=CONFIG["ignore_index"])
        total_iou += iou
        total_dice += dice
        count += 1

    if num_skipped > 0:
        print(f"   ⚠️ Val: пропущено пустых батчей: {num_skipped}")
    return running_loss / max(count, 1), total_iou / max(count, 1), total_dice / max(count, 1)

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

    # --- DataLoaders ---
    if arch_name == "hovernet":
        train_ds = BCSSDataset(
            os.path.join(CONFIG["data_root"], "train_512"),
            os.path.join(CONFIG["data_root"], "train_mask_512"),
            augment=True, img_size=CONFIG["img_size"])
        val_ds = BCSSDataset(
            os.path.join(CONFIG["data_root"], "val_512"),
            os.path.join(CONFIG["data_root"], "val_mask_512"),
            augment=False, img_size=CONFIG["img_size"])

        train_ds = HVWrapper(train_ds)
        val_ds = HVWrapper(val_ds)

        train_loader = DataLoader(
            train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
            num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True)
        val_loader = DataLoader(
            val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
            num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=False)
    else:
        train_loader, val_loader = get_dataloaders(
            data_root=CONFIG["data_root"],
            batch_size=CONFIG["batch_size"],
            num_workers=CONFIG["num_workers"],
            img_size=CONFIG["img_size"]
        )

    # --- Model ---
    if arch_name == "m3net":
        model = M3NetPlusPlus(num_classes=CONFIG["num_classes"], backbone=encoder_name)
        criterion = HybridLoss(num_classes=CONFIG["num_classes"], ignore_index=CONFIG["ignore_index"])
    elif arch_name == "hovernet":
        model = HoVerNetAdapted(num_classes=CONFIG["num_classes"], backbone=encoder_name)
        criterion = HoVerNetLoss(num_classes=CONFIG["num_classes"], ignore_index=CONFIG["ignore_index"])
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")

    model = model.to(device)
    print(f"📐 Модель: {model.__class__.__name__} ({sum(p.numel() for p in model.parameters()):,} параметров)")

    # --- Optimizer (encoder с меньшим lr) ---
    if hasattr(model, "enc1"):
        encoder_params = []
        for name in ["enc1", "enc2", "enc3", "enc4", "enc5"]:
            if hasattr(model, name):
                encoder_params.extend(list(getattr(model, name).parameters()))
        decoder_params = [p for n, p in model.named_parameters() if not n.startswith("enc")]
        optimizer = optim.AdamW([
            {"params": encoder_params, "lr": CONFIG["learning_rate"] * CONFIG["encoder_lr_mult"], "name": "encoder"},
            {"params": decoder_params, "lr": CONFIG["learning_rate"], "name": "decoder"}
        ], weight_decay=CONFIG["weight_decay"])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    # --- CosineAnnealingLR ---
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    # --- CSV Log ---
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
            model, train_loader, criterion, optimizer, scaler, device, use_amp, epoch, arch_name
        )

        val_loss, val_iou, val_dice = validate(model, val_loader, criterion, device, use_amp, arch_name)

        scheduler.step()

        epoch_time = time.time() - start_time
        lr_enc = optimizer.param_groups[0]["lr"]
        lr_dec = optimizer.param_groups[1]["lr"]
        current_lr = lr_dec

        print(f"\nEpoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} |  "
              f"IoU: {val_iou:.4f} | Dice: {val_dice:.4f} |  "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_iou, val_dice, current_lr])

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], f"best_model_{arch_name}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_dice": best_val_dice,
                "config": CONFIG,
                "architecture": arch_name,
                "encoder": encoder_name
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