import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
from typing import Optional, Dict

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSlider, QFileDialog, QMessageBox,
    QCheckBox, QTabWidget, QTextEdit,
    QGridLayout, QGroupBox, QDoubleSpinBox, QSizePolicy,
    QTableWidget, QTableWidgetItem, QListWidget, QListWidgetItem,
    QDialog, QFormLayout, QLineEdit, QDialogButtonBox, QAbstractItemView,
    QDateEdit, QInputDialog
)
from PyQt6.QtGui import QImage, QPainter, QPen, QColor, QFont, QShortcut, QKeySequence
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF, QDate

try:
    from gui.model_manager import ModelManager
    from gui import database as db
except ImportError:
    from model_manager import ModelManager
    import database as db

os.environ.setdefault(
    "NIRSEG_DATA_DIR",
    str(Path(__file__).resolve().parents[1] / "database")
)



# Русские подписи классов для сводки по исследованию
CLASS_RU = {
    "tumor": "опухоль",
    "stroma": "строма",
    "lymphocytic": "лимфоциты",
    "necrosis": "некроз",
    "fat": "жир",
    "other": "прочее",
}

# === ONNX-бэкенд (быстрый инференс без загрузки PyTorch-весов) ===
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ onnxruntime не установлен — инференс через PyTorch")

import torch  # noqa: E402  (нужен для TTA-преобразований)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONNX_FILES = {
    "unet": "unet_resnet34.onnx",
    "deeplab": "deeplab_efficientnet_b3.onnx",
    "unetpp": "unetpp_resnet50.onnx",
}

# === Стиль приложения ===
APP_STYLESHEET = """
* {
    font-family: 'Noto Sans', 'Segoe UI', 'Cantarell', sans-serif;
    font-size: 13px;
    color: #E6EDF3;
}

QMainWindow, QDialog { background: #0B0D12; }
QWidget { background: transparent; }

QGroupBox {
    background: #11151D;
    border: 1px solid #2A303B;
    border-radius: 12px;
    margin: 0;
    padding: 0;
}

QPushButton {
    background: #11151D;
    border: 1px solid #343B48;
    border-radius: 10px;
    padding: 8px 14px;
    min-height: 20px;
    color: #E6EDF3;
}
QPushButton:hover {
    background: #171C25;
    border-color: #FCEE0A;
}
QPushButton:pressed {
    background: #242A18;
    border-color: #D7CE00;
}
QPushButton:disabled {
    color: #5F6877;
    background: #171B22;
    border-color: #2A303B;
}
QPushButton:focus { border-color: #FCEE0A; }

QPushButton#primary {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #FCEE0A, stop:1 #D6CC00);
    color: #11151D;
    border: none;
    font-weight: 700;
    font-size: 14px;
    padding: 12px 16px;
    border-radius: 12px;
}
QPushButton#primary:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #FFF45C, stop:1 #E9DF22);
}
QPushButton#primary:pressed {
    background: #B8AE00;
}
QPushButton#primary:disabled {
    background: #d8dae4;
    color: #687384;
}

QComboBox, QLineEdit, QDateEdit, QDoubleSpinBox, QTextEdit {
    background: #11151D;
    border: 1px solid #343B48;
    border-radius: 8px;
    padding: 6px 10px;
    min-height: 22px;
    selection-background-color: #242A18;
    selection-color: #E6EDF3;
}
QComboBox:hover, QLineEdit:hover, QDateEdit:hover,
QDoubleSpinBox:hover, QTextEdit:hover {
    border-color: #8F8A20;
}
QLineEdit:focus, QDateEdit:focus, QDoubleSpinBox:focus, QTextEdit:focus {
    border: 2px solid #FCEE0A;
    padding: 5px 9px;
}

QComboBox {
    min-height: 26px;
    padding: 5px 34px 5px 10px;
}
QComboBox:focus {
    border: 1px solid #FCEE0A;
}
QComboBox::drop-down {
    border: none;
    width: 26px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #FCEE0A;
    margin-right: 10px;
}
QComboBox QAbstractItemView {
    border: 1px solid #343B48;
    border-radius: 8px;
    background: #11151D;
    selection-background-color: #FCEE0A;
    selection-color: #11151D;
    padding: 4px;
    outline: 0;
}
QComboBox QAbstractItemView::item {
    min-height: 28px;
    padding: 4px 10px;
    border-radius: 6px;
}
QComboBox QAbstractItemView::item:selected {
    background: #FCEE0A;
    color: #11151D;
}

QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    border: none;
    background: #171C25;
    border-radius: 4px;
    width: 18px;
    margin: 2px;
}
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background: #242A18;
}

QSlider::groove:horizontal {
    height: 6px;
    border-radius: 3px;
    background: #2A303B;
}
QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
    background: #11151D;
    border: 2px solid #FCEE0A;
}
QSlider::handle:horizontal:hover {
    background: #242A18;
    border-color: #D6CC00;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #D7CE00, stop:1 #FCEE0A);
    border-radius: 4px;
}

QCheckBox { spacing: 10px; color: #E6EDF3; }
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    margin-top: 1px;
    border-radius: 6px;
    border: 2px solid #4A5362;
    background: #11151D;
}
QCheckBox::indicator:hover { border-color: #D7CE00; }
QCheckBox::indicator:checked {
    background: #FCEE0A;
    border-color: #FCEE0A;
    image: none;
}
QCheckBox::indicator:checked:hover {
    background: #D6CC00;
    border-color: #D6CC00;
}

QTabWidget::pane {
    border: 1px solid #2A303B;
    border-radius: 12px;
    background: #11151D;
    top: -1px;
}
QTabBar::tab {
    padding: 9px 18px;
    margin-right: 6px;
    border: 1px solid #2A303B;
    border-bottom: none;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    background: #171C25;
    color: #8B95A5;
}
QTabBar::tab:hover:!selected {
    background: #202631;
    color: #E6EDF3;
}
QTabBar::tab:selected {
    background: #11151D;
    color: #FCEE0A;
    font-weight: 700;
    border-color: #2A303B;
}

QTableWidget {
    border: 1px solid #2A303B;
    border-radius: 10px;
    gridline-color: #222832;
    background: #11151D;
    alternate-background-color: #0F1319;
    selection-background-color: #242A18;
    selection-color: #E6EDF3;
}
QHeaderView::section {
    background: #171C25;
    border: none;
    border-bottom: 1px solid #2A303B;
    border-right: 1px solid #2A303B;
    padding: 8px;
    font-weight: 700;
    color: #5a5f73;
}
QTableWidget::item { padding: 6px; }
QTableWidget::item:selected {
    background: #242A18;
    color: #E6EDF3;
    border-radius: 4px;
}

QListWidget {
    border: 1px solid #2A303B;
    border-radius: 10px;
    background: #11151D;
    padding: 4px;
}
QListWidget::item {
    padding: 6px 10px;
    border-radius: 8px;
    margin: 2px 0;
}
QListWidget::item:hover { background: #171C25; }
QListWidget::item:selected {
    background: #242A18;
    color: #E6EDF3;
}

QStatusBar {
    background: #11151D;
    border-top: 1px solid #2A303B;
    font-weight: 600;
    padding: 2px 8px;
}

QToolTip {
    background: #E6EDF3;
    color: #11151D;
    border: none;
    padding: 6px 10px;
    border-radius: 8px;
}

QScrollBar:vertical {
    background: transparent;
    width: 12px;
    margin: 4px 2px 4px 2px;
    border-radius: 6px;
}
QScrollBar::handle:vertical {
    background: #46505F;
    min-height: 30px;
    border-radius: 6px;
}
QScrollBar::handle:vertical:hover { background: #697586; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }

QScrollBar:horizontal {
    background: transparent;
    height: 12px;
    margin: 2px 4px 2px 4px;
    border-radius: 6px;
}
QScrollBar::handle:horizontal {
    background: #46505F;
    min-width: 30px;
    border-radius: 6px;
}
QScrollBar::handle:horizontal:hover { background: #697586; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }

QMessageBox { background: #0B0D12; }
QMessageBox QLabel { color: #E6EDF3; font-size: 13px; }
QMessageBox QPushButton { min-width: 90px; padding: 8px 16px; }

QMenu {
    background: #11151D;
    border: 1px solid #2A303B;
    border-radius: 10px;
    padding: 6px;
}
QMenu::item { padding: 6px 24px 6px 20px; border-radius: 6px; }
QMenu::item:selected { background: #242A18; }
QMenu::separator { height: 1px; background: #2A303B; margin: 4px 8px; }

QLabel#cyber_status {
    color: #39FF88;
    font-weight: 800;
    letter-spacing: 1px;
}
QPushButton#primary {
    border: 1px solid #FCEE0A;
    box-shadow: none;
}
QGroupBox:hover {
    border-color: #3A424F;
}
QTabBar::tab:selected {
    border-top: 2px solid #FCEE0A;
}
QHeaderView::section {
    color: #FCEE0A;
    letter-spacing: 0.5px;
}

QGroupBox#activeStudyCard {
    background: #11151D;
    border: 1px solid #FCEE0A;
    border-radius: 10px;
}
QLabel#studyHeader {
    color: #FCEE0A;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}
QLabel#activeStudyLabel {
    color: #F2F4F7;
    font-size: 12px;
    font-weight: 600;
}
QPushButton#studyButton {
    background: #171C25;
    color: #00E5FF;
    border: 1px solid #00E5FF;
    border-radius: 7px;
    padding: 6px 12px;
    min-height: 18px;
    font-size: 10px;
    font-weight: 800;
}
QPushButton#studyButton:hover {
    background: #0D2A30;
    border-color: #5CF2FF;
    color: #5CF2FF;
}
QPushButton#studyButton:pressed {
    background: #12343A;
}
"""


def seg_logits(model_output):
    """Логиты сегментации из выхода модели (тензор или dict кастомных моделей)."""
    if isinstance(model_output, dict):
        for key in ("out", "semantic", "logits"):
            if key in model_output:
                return model_output[key]
        raise ValueError(f"Нет сегментационного выхода: {list(model_output.keys())}")
    return model_output


def _row_get(row, key, default=""):
    """Безопасное чтение поля из sqlite3.Row / dict / None."""
    try:
        value = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if value is None else value


def make_card(title_text: str, spacing: int = 10):
    """Карточка-группа: заголовок — обычный QLabel, первый элемент layout."""
    group = QGroupBox()
    layout = QVBoxLayout(group)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(spacing)

    header = QLabel(title_text)
    header.setStyleSheet(
        "color: #FCEE0A; font-weight: 700;"
        "background: #0B0D12; border-radius: 6px; padding: 3px 10px;")
    layout.addWidget(header)

    return group, layout


class ZoomableImageView(QWidget):
    """Изображение с зумом и панорамированием.

    Колесо мыши — зум к курсору; ЛКМ + движение — панорамирование;
    двойной клик — вписать в окно; fit() — вписать; actual_size() — 1:1.
    """
    MIN_SCALE, MAX_SCALE, ZOOM_STEP = 0.05, 16.0, 1.25

    def __init__(self, placeholder: str = "Нет изображения", parent=None):
        super().__init__(parent)
        self._qimage: Optional[QImage] = None
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._placeholder = placeholder
        self._is_fitted = True
        self._dragging = False
        self._drag_start = QPointF()
        self._offset_start = QPointF()
        self.setMinimumSize(200, 200)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def set_image(self, image_rgb: Optional[np.ndarray], keep_view: bool = False):
        """Показать изображение. keep_view=True — заменить кадр, не сбрасывая
        зум/панорамирование (используется при перерисовке оверлея слайдером)."""
        if image_rgb is None:
            self.reset()
            return
        h, w = image_rgb.shape[:2]
        data = np.ascontiguousarray(image_rgb)
        self._qimage = QImage(data.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        if not keep_view:
            self.fit()
        self.update()

    def reset(self, placeholder: Optional[str] = None):
        if placeholder is not None:
            self._placeholder = placeholder
        self._qimage = None
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._is_fitted = True
        self.update()

    def zoom_in(self):
        self._zoom_at(QPointF(self.width() / 2, self.height() / 2), self.ZOOM_STEP)

    def zoom_out(self):
        self._zoom_at(QPointF(self.width() / 2, self.height() / 2), 1.0 / self.ZOOM_STEP)

    def fit(self):
        if self._qimage is None:
            return
        iw, ih = self._qimage.width(), self._qimage.height()
        ww, wh = max(self.width(), 1), max(self.height(), 1)
        self._scale = min(ww / iw, wh / ih) * 0.98
        self._offset = QPointF(ww / 2, wh / 2)
        self._is_fitted = True
        self.update()

    def actual_size(self):
        if self._qimage is None:
            return
        self._scale = 1.0
        self._offset = QPointF(self.width() / 2, self.height() / 2)
        self._is_fitted = False
        self.update()

    @property
    def has_image(self) -> bool:
        return self._qimage is not None

    def _zoom_at(self, pos: QPointF, factor: float):
        if self._qimage is None:
            return
        new_scale = float(np.clip(self._scale * factor, self.MIN_SCALE, self.MAX_SCALE))
        if new_scale == self._scale:
            return
        img_pt = (pos - self._offset) / self._scale
        self._offset = pos - img_pt * new_scale
        self._scale = new_scale
        self._is_fitted = False
        self.update()

    def _image_rect(self) -> QRectF:
        iw = self._qimage.width() * self._scale
        ih = self._qimage.height() * self._scale
        return QRectF(self._offset.x() - iw / 2, self._offset.y() - ih / 2, iw, ih)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._qimage is None:
            p.setPen(QPen(QColor("#394352"), 1, Qt.PenStyle.DashLine))
            p.drawRoundedRect(1, 1, self.width() - 2, self.height() - 2, 10, 10)
            p.setPen(QColor("#718096"))
            f = QFont()
            f.setPointSize(10)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return
        p.drawImage(self._image_rect(), self._qimage)
        label = f"{self._scale * 100:.0f}%"
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(15, 23, 42, 140))
        p.drawRoundedRect(8, 8, p.fontMetrics().horizontalAdvance(label) + 16, 22, 6, 6)
        p.setPen(QColor("#F2F4F7"))
        p.drawText(16, 24, label)

    def wheelEvent(self, e):
        if self._qimage is None:
            return
        f = self.ZOOM_STEP if e.angleDelta().y() > 0 else 1.0 / self.ZOOM_STEP
        self._zoom_at(e.position(), f)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._qimage is not None:
            self._dragging = True
            self._drag_start = e.position()
            self._offset_start = QPointF(self._offset)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._dragging:
            self._offset = self._offset_start + (e.position() - self._drag_start)
            self._is_fitted = False
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, e):
        if self._qimage is not None:
            self.fit()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._is_fitted:
            self.fit()


class LazyModelManager(ModelManager):
    """Ленивая загрузка моделей + ONNX-бэкенд."""

    def __init__(self, device: Optional[str] = None):
        super().__init__(device)
        self._onnx_sessions: dict = {}

    def _onnx_session(self, name: str):
        if not ONNX_AVAILABLE or name not in ONNX_FILES:
            return None
        if name not in self._onnx_sessions:
            path = PROJECT_ROOT / "exported_models" / ONNX_FILES[name]
            if not path.exists():
                self._onnx_sessions[name] = None
            else:
                try:
                    self._onnx_sessions[name] = ort.InferenceSession(
                        str(path), providers=["CPUExecutionProvider"])
                    print(f"⚡ ONNX-модель {name} готова ({path.name})")
                except Exception as e:
                    print(f"⚠️ ONNX {name} не загрузился ({e}); будет PyTorch")
                    self._onnx_sessions[name] = None
        return self._onnx_sessions[name]

    def available_models(self) -> dict:
        return {name: {"torch": os.path.exists(cfg["ckpt"]),
                       "onnx": self._onnx_session(name) is not None}
                for name, cfg in self.model_configs.items()}

    def load_model(self, model_name: str) -> bool:
        """Офлайн-загрузка PyTorch-модели (без скачивания ImageNet-весов)."""
        if model_name in self.models:
            return True
        cfg = self.model_configs.get(model_name)
        if cfg is None or not os.path.exists(cfg["ckpt"]):
            return False
        try:
            import segmentation_models_pytorch as smp
            builders = {"unet": smp.Unet, "deeplab": smp.DeepLabV3Plus,
                        "unetpp": smp.UnetPlusPlus}
            builder = builders[cfg["arch"]]
            model = builder(encoder_name=cfg["encoder"], encoder_weights=None,
                            in_channels=3, classes=22)
            ckpt = torch.load(cfg["ckpt"], map_location=self.device, weights_only=False)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) \
                and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state)
            model.eval().to(self.device)
            self.models[model_name] = model
            print(f"✅ Модель {model_name} загружена (офлайн, из чекпоинта)")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки {model_name}: {e}")
            return False

    def _forward(self, name: str, tensor_img: "torch.Tensor") -> "torch.Tensor":
        """Прямой проход: ONNX если есть, иначе ленивая загрузка PyTorch."""
        sess = self._onnx_session(name)
        if sess is not None:
            out = sess.run(None, {"input": tensor_img.cpu().numpy().astype("float32")})[0]
            return torch.from_numpy(out).to(tensor_img.device)
        if name not in self.models and not self.load_model(name):
            raise RuntimeError(f"Модель '{name}' недоступна: нет ни чекпоинта, ни ONNX")
        return seg_logits(self.models[name](tensor_img))

    @torch.no_grad()
    def _tta_probs(self, forward_fn, tensor_img):
        """TTA: 5 преобразований, усреднение softmax."""
        f = lambda t: torch.softmax(forward_fn(t), dim=1)
        probs = [f(tensor_img)]
        out = forward_fn(torch.flip(tensor_img, dims=[3]))
        probs.append(torch.flip(torch.softmax(out, 1), dims=[3]))
        out = forward_fn(torch.flip(tensor_img, dims=[2]))
        probs.append(torch.flip(torch.softmax(out, 1), dims=[2]))
        out = forward_fn(torch.rot90(tensor_img, 1, dims=[2, 3]))
        probs.append(torch.rot90(torch.softmax(out, 1), 3, dims=[2, 3]))
        aug = torch.flip(torch.rot90(tensor_img, 1, dims=[2, 3]), dims=[3])
        out = forward_fn(aug)
        probs.append(torch.rot90(torch.flip(torch.softmax(out, 1), dims=[3]), 3, dims=[2, 3]))
        return torch.stack(probs, dim=0).mean(dim=0)

    @torch.no_grad()
    def predict_single(self, image_path: str, model_name: str = "unet",
                       use_tta: bool = False) -> Optional[np.ndarray]:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Не удалось загрузить изображение: {image_path}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]

        transformed = self.get_transform()(image=image)
        tensor_img = transformed['image'].unsqueeze(0).to(self.device)

        fwd = lambda t: self._forward(model_name, t)
        probs = self._tta_probs(fwd, tensor_img) if use_tta \
            else torch.softmax(fwd(tensor_img), dim=1)

        mask = torch.argmax(probs[0], dim=0).cpu().numpy()
        return cv2.resize(mask.astype(np.uint8), (original_shape[1], original_shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    @torch.no_grad()
    def predict_ensemble(self, image_path: str, use_tta: bool = False,
                         weights: Optional[Dict[str, float]] = None) -> Optional[np.ndarray]:
        image = cv2.imread(image_path)
        if image is None:
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]

        transformed = self.get_transform()(image=image)
        tensor_img = transformed['image'].unsqueeze(0).to(self.device)

        if weights is None:
            weights = {name: 1.0 for name in self.model_configs}

        all_probs, all_weights = [], []
        for model_name, weight in weights.items():
            fwd = lambda t, n=model_name: self._forward(n, t)
            probs = self._tta_probs(fwd, tensor_img) if use_tta \
                else torch.softmax(fwd(tensor_img), dim=1)
            all_probs.append(probs)
            all_weights.append(weight)

        w = torch.tensor(all_weights, device=self.device,
                         dtype=all_probs[0].dtype).view(-1, 1, 1, 1, 1)
        avg = (torch.stack(all_probs) * w).sum(dim=0) / sum(all_weights)

        mask = torch.argmax(avg[0], dim=0).cpu().numpy()
        return cv2.resize(mask.astype(np.uint8), (original_shape[1], original_shape[0]),
                          interpolation=cv2.INTER_NEAREST)


class PredictionWorker(QThread):
    """Рабочий поток для предсказания (без блокировки UI).

    ВАЖНО: поток НЕ трогает базу данных — sqlite-соединение принадлежит
    главному потоку. Сохранение предсказания выполняется в _on_prediction_finished.
    """
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, model_manager: ModelManager, image_path: str,
                 model_name: str, use_tta: bool, use_ensemble: bool,
                 weights: dict):
        super().__init__()
        self.model_manager = model_manager
        self.image_path = image_path
        self.model_name = model_name
        self.use_tta = use_tta
        self.use_ensemble = use_ensemble
        self.weights = weights
        self.result_mask = None
        self.inference_ms: Optional[int] = None

    def run(self):
        try:
            self.progress.emit("🔄 Загрузка модели...")
            t0 = time.monotonic()

            if self.use_ensemble:
                self.progress.emit("🤖 Запуск ансамбля...")
                self.result_mask = self.model_manager.predict_ensemble(
                    self.image_path, use_tta=self.use_tta, weights=self.weights)
            else:
                self.progress.emit(f"🤖 Предсказание ({self.model_name})...")
                self.result_mask = self.model_manager.predict_single(
                    self.image_path, model_name=self.model_name, use_tta=self.use_tta)

            self.inference_ms = int((time.monotonic() - t0) * 1000)

            if self.result_mask is None:
                self.error.emit("❌ Ошибка при предсказании")
            else:
                self.progress.emit("✅ Готово!")
                self.finished.emit()

        except Exception as e:
            self.error.emit(f"❌ Ошибка: {str(e)}")


# ======================================================================
# Диалоги (пациент / исследование)
# ======================================================================

class PatientDialog(QDialog):
    """Диалог создания пациента."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Новый пациент")
        self.setMinimumWidth(380)

        form = QFormLayout(self)

        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("Иванов Иван Иванович")
        form.addRow("ФИО:", self.edit_name)

        self.edit_card = QLineEdit()
        self.edit_card.setPlaceholderText("Номер медицинской карты")
        form.addRow("Карта:", self.edit_card)

        self.edit_birth = QDateEdit(QDate(1980, 1, 1))
        self.edit_birth.setCalendarPopup(True)
        self.edit_birth.setDisplayFormat("dd.MM.yyyy")
        self.edit_birth.setMaximumDate(QDate.currentDate())
        form.addRow("Дата рождения:", self.edit_birth)

        self.combo_sex = QComboBox()
        self.combo_sex.addItems(["м", "ж"])
        form.addRow("Пол:", self.combo_sex)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def data(self) -> dict:
        return {
            "full_name": self.edit_name.text().strip(),
            "card_number": self.edit_card.text().strip(),
            "birth_date": self.edit_birth.date().toString("yyyy-MM-dd"),
            "sex": self.combo_sex.currentText(),
        }


class StudyDialog(QDialog):
    """Диалог создания исследования для выбранного пациента."""

    def __init__(self, patient_label: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Новое исследование")
        self.setMinimumWidth(400)

        form = QFormLayout(self)

        patient = QLabel(patient_label)
        patient.setWordWrap(True)
        form.addRow("Пациент:", patient)

        self.edit_date = QDateEdit(QDate.currentDate())
        self.edit_date.setCalendarPopup(True)
        self.edit_date.setDisplayFormat("dd.MM.yyyy")
        form.addRow("Дата исследования:", self.edit_date)

        self.edit_localization = QLineEdit()
        self.edit_localization.setPlaceholderText("Например: молочная железа, ВНК квадрант")
        form.addRow("Локализация:", self.edit_localization)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def data(self) -> dict:
        return {
            "study_date": self.edit_date.date().toString("yyyy-MM-dd"),
            "localization": self.edit_localization.text().strip(),
        }


class SegmentationApp(QMainWindow):
    """Главное окно приложения для сегментации"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("◈ CYBER MED AI — Сегментация тканей")
        self.setGeometry(100, 100, 1400, 900)

        self.model_manager = LazyModelManager()
        self.current_image_path = None
        self.current_mask = None
        self.current_image_rgb = None
        self.prediction_thread = None

        # --- База данных (init_db идемпотентен: существующая база откроется) ---
        try:
            self.db_conn = db.init_db()
        except Exception as e:
            QMessageBox.critical(
                None, "Ошибка базы данных",
                f"Не удалось открыть/мигрировать базу:\n{db.db_path()}\n\n{e}\n\n"
                "Если база создавалась старой версией приложения и пуста — "
                "удалите файл nir.db и запустите снова.")
            raise SystemExit(1)

        self.model_ids = self._register_models_in_db()

        # Пользователь для audit_log: аутентификации нет — берём первого в БД
        u = self.db_conn.execute("SELECT id FROM users ORDER BY id LIMIT 1").fetchone()
        self.current_user_id = u["id"] if u else None

        self.current_study_id = None
        self.current_image_id = None
        self._studies_cache = []      # строки list_studies (актуальны после refresh)
        self._history_cache = {}      # pred_id -> строка предсказания из истории
        self._last_predict_params = {}

        self.init_ui()

        # Постоянный индикатор пути к БД в строке состояния
        db_label = QLabel(f" БД: {db.db_path()} ")
        db_label.setStyleSheet("color: #8B95A5; padding: 0 8px;")
        self.statusBar().addPermanentWidget(db_label)

        self.refresh_studies_table()
        self._update_models_status()

        # «Вписать» без кнопки: хоткей Ctrl+0 (вью активной вкладки)
        self._sc_fit = QShortcut(QKeySequence("Ctrl+0"), self)
        self._sc_fit.activated.connect(lambda: self._active_view().fit())

    # ------------------------------------------------------------------
    # Статус
    # ------------------------------------------------------------------

    def _status(self, text: str, color: str = "green"):
        sb = self.statusBar()
        sb.showMessage(text)
        sb.setStyleSheet(f"QStatusBar {{ color: {color}; font-weight: bold; }}")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_panel = self.create_control_panel()
        left_panel.setMinimumWidth(340)
        main_layout.addWidget(left_panel, 0)

        right_panel = self.create_image_panel()
        main_layout.addWidget(right_panel, 1)

        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)

    # ------------------------------------------------------------------
    # Модели в БД
    # ------------------------------------------------------------------

    def _register_models_in_db(self) -> dict:
        """Регистрирует модели в БД (register_model идемпотентен по name+version).

        Версия = mtime чекпоинта: переобучили модель — появится новая версия,
        старые предсказания останутся привязаны к старой (трассируемость).
        """
        ids = {}
        for name, cfg in self.model_manager.model_configs.items():
            version = "v1"
            if os.path.exists(cfg["ckpt"]):
                version = str(int(os.path.getmtime(cfg["ckpt"])))
            ids[name] = db.register_model(
                self.db_conn, name=name, version=version,
                arch=str(cfg.get("arch", "")), encoder=str(cfg.get("encoder", "")),
                ckpt_path=str(cfg.get("ckpt", "")),
            )
        # псевдо-запись для режима ансамбля
        ids["ensemble"] = db.register_model(
            self.db_conn, name="ensemble", version="v1",
            arch="soft-voting",
            encoder="+".join(self.model_manager.model_configs.keys()),
        )
        return ids

    def _current_model_key(self) -> str:
        """Ключ модели ('unet'/'deeplab'/'unetpp'/'ensemble') по комбобоксам."""
        if self.combo_model.currentIndex() == 1:
            return "ensemble"
        return ["unet", "deeplab", "unetpp"][self.combo_single.currentIndex()]

    # ------------------------------------------------------------------
    # Левая панель управления
    # ------------------------------------------------------------------

    def create_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        # === ЗАГОЛОВОК ===
        title = QLabel("◈ CYBER MED AI")
        title_font = QFont()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # === ЗАГРУЗКА ИЗОБРАЖЕНИЯ ===
        group_load, load_layout = make_card("📁 Входное изображение")

        self.btn_load = QPushButton("Загрузить изображение")
        self.btn_load.clicked.connect(self.load_image)
        load_layout.addWidget(self.btn_load)

        self.label_image_info = QLabel("Изображение не загружено")
        self.label_image_info.setStyleSheet("color: gray; font-size: 11px;")
        self.label_image_info.setWordWrap(True)
        load_layout.addWidget(self.label_image_info)

        layout.addWidget(group_load)

        # === ВЫБОР МОДЕЛИ ===
        group_model, model_layout = make_card("🤖 Выбор модели")

        lbl_model = QLabel("Модель:")
        lbl_model.setStyleSheet("color: #8B95A5;")
        model_layout.addWidget(lbl_model)

        self.combo_model = QComboBox()
        self.combo_model.addItems(["Одна модель", "Ансамбль (все модели)"])
        self.combo_model.setMinimumHeight(34)
        self.combo_model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.combo_model.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Fixed)
        model_layout.addWidget(self.combo_model)

        self.label_single_model = QLabel("Архитектура:")
        self.label_single_model.setStyleSheet("color: #8B95A5;")
        model_layout.addWidget(self.label_single_model)

        self.combo_single = QComboBox()
        self.combo_single.addItems([
            "U-Net + ResNet34",
            "DeepLab + EfficientNet-B3",
            "U-Net++ + ResNet50",
        ])
        self.combo_single.setMinimumHeight(34)
        self.combo_single.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.combo_single.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Fixed)
        model_layout.addWidget(self.combo_single)

        # Веса для ансамбля
        self.label_weights = QLabel("Веса ансамбля:")
        self.label_weights.setStyleSheet("color: #8B95A5;")
        self.label_weights.setVisible(False)
        model_layout.addWidget(self.label_weights)

        weights_layout = QGridLayout()
        weights_layout.setHorizontalSpacing(10)
        weights_layout.setVerticalSpacing(6)
        weights_layout.setColumnStretch(1, 1)

        self.label_w_unet = QLabel("U-Net:")
        self.label_w_unet.setVisible(False)
        weights_layout.addWidget(self.label_w_unet, 0, 0)
        self.weight_unet = QDoubleSpinBox()
        self.weight_unet.setRange(0.0, 10.0)
        self.weight_unet.setSingleStep(0.1)
        self.weight_unet.setValue(1.0)
        self.weight_unet.setVisible(False)
        weights_layout.addWidget(self.weight_unet, 0, 1)

        self.label_w_deeplab = QLabel("DeepLab:")
        self.label_w_deeplab.setVisible(False)
        weights_layout.addWidget(self.label_w_deeplab, 1, 0)
        self.weight_deeplab = QDoubleSpinBox()
        self.weight_deeplab.setRange(0.0, 10.0)
        self.weight_deeplab.setSingleStep(0.1)
        self.weight_deeplab.setValue(1.0)
        self.weight_deeplab.setVisible(False)
        weights_layout.addWidget(self.weight_deeplab, 1, 1)

        self.label_w_unetpp = QLabel("U-Net++:")
        self.label_w_unetpp.setVisible(False)
        weights_layout.addWidget(self.label_w_unetpp, 2, 0)
        self.weight_unetpp = QDoubleSpinBox()
        self.weight_unetpp.setRange(0.0, 10.0)
        self.weight_unetpp.setSingleStep(0.1)
        self.weight_unetpp.setValue(1.0)
        self.weight_unetpp.setVisible(False)
        weights_layout.addWidget(self.weight_unetpp, 2, 1)

        model_layout.addLayout(weights_layout)

        self.combo_model.currentTextChanged.connect(self.on_model_type_changed)

        layout.addWidget(group_model)

        # === ОПЦИИ ПРЕДСКАЗАНИЯ ===
        group_options, options_layout = make_card("⚙️ Опции предсказания")

        self.check_tta = QCheckBox("Использовать TTA (Аугментация в тестировании)")
        self.check_tta.setChecked(False)
        options_layout.addWidget(self.check_tta)

        tta_info = QLabel("TTA: 5× трансформаций → точнее, но медленнее")
        tta_info.setStyleSheet("color: #8B95A5; font-size: 11px;")
        tta_info.setWordWrap(True)
        options_layout.addWidget(tta_info)

        layout.addWidget(group_options)

        # === КНОПКА PREDICT ===
        self.btn_predict = QPushButton("  Предсказать")
        self.btn_predict.setObjectName("primary")
        self.btn_predict.setToolTip("Запустить сегментацию (Ctrl+P)")
        self.btn_predict.setShortcut("Ctrl+P")
        self.btn_predict.clicked.connect(self.predict)
        self.btn_predict.setEnabled(False)
        layout.addWidget(self.btn_predict)

        # === ВИЗУАЛИЗАЦИЯ ===
        group_viz, viz_layout = make_card("🎨 Визуализация")

        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Прозрачность:"))
        self.slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_alpha.setMinimum(0)
        self.slider_alpha.setMaximum(100)
        self.slider_alpha.setValue(50)
        alpha_layout.addWidget(self.slider_alpha)
        self.label_alpha = QLabel("50%")
        alpha_layout.addWidget(self.label_alpha)
        self.slider_alpha.valueChanged.connect(self.on_alpha_changed)
        viz_layout.addLayout(alpha_layout)

        self.check_show_mask = QCheckBox("Показать цветную маску")
        self.check_show_mask.setChecked(True)
        self.check_show_mask.stateChanged.connect(self.update_display)
        viz_layout.addWidget(self.check_show_mask)

        self.check_show_overlay = QCheckBox("Показать наложение")
        self.check_show_overlay.setChecked(False)
        self.check_show_overlay.stateChanged.connect(self.update_display)
        viz_layout.addWidget(self.check_show_overlay)

        layout.addWidget(group_viz)

        # === ЭКСПОРТ ===
        group_save, save_layout = make_card("💾 Экспорт")

        self.btn_save_mask = QPushButton("Сохранить маску")
        self.btn_save_mask.clicked.connect(self.save_mask)
        self.btn_save_mask.setEnabled(False)
        save_layout.addWidget(self.btn_save_mask)

        self.btn_save_overlay = QPushButton("Сохранить наложение")
        self.btn_save_overlay.clicked.connect(self.save_overlay)
        self.btn_save_overlay.setEnabled(False)
        save_layout.addWidget(self.btn_save_overlay)

        layout.addWidget(group_save)

        layout.addStretch()

        return panel

    # ------------------------------------------------------------------
    # Правая панель
    # ------------------------------------------------------------------

    def create_image_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # === АКТИВНОЕ ИССЛЕДОВАНИЕ — верхний правый угол ===
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 4)
        top_row.setSpacing(10)

        top_row.addStretch(1)

        study_card = QGroupBox()
        study_card.setObjectName("activeStudyCard")
        study_layout = QHBoxLayout(study_card)
        study_layout.setContentsMargins(12, 8, 10, 8)
        study_layout.setSpacing(10)

        study_title = QLabel("◈ ИССЛЕДОВАНИЕ")
        study_title.setObjectName("studyHeader")
        study_layout.addWidget(study_title)

        self.label_study = QLabel("Не выбрано")
        self.label_study.setObjectName("activeStudyLabel")
        self.label_study.setWordWrap(True)
        self.label_study.setMinimumWidth(260)
        self.label_study.setMaximumWidth(430)
        study_layout.addWidget(self.label_study, 1)

        self.btn_open_patients = QPushButton("ПАЦИЕНТЫ")
        self.btn_open_patients.setObjectName("studyButton")
        self.btn_open_patients.setToolTip("Открыть пациентов и исследования")
        self.btn_open_patients.clicked.connect(self.open_patients_tab)
        study_layout.addWidget(self.btn_open_patients)

        top_row.addWidget(study_card)
        layout.addLayout(top_row)

        # Кнопки-лупа
        zoom_row = QHBoxLayout()
        for text, slot, tip, key in [
                ("🔍+", lambda: self._active_view().zoom_in(), "Приблизить", "Ctrl++"),
                ("🔍−", lambda: self._active_view().zoom_out(), "Отдалить", "Ctrl+-"),
                ("1:1", lambda: self._active_view().actual_size(), "Реальный размер", "Ctrl+1")]:
            b = QPushButton(text)
            b.setToolTip(f"{tip} ({key})")
            b.setShortcut(key)
            b.setMaximumWidth(120)
            b.clicked.connect(slot)
            zoom_row.addWidget(b)
        zoom_row.addStretch()
        layout.addLayout(zoom_row)

        tabs = QTabWidget()

        # === Вкладка 1: Исходное изображение ===
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)

        self.view_original = ZoomableImageView(
            "Исходное изображение\n(Загрузите изображение)")
        tab1_layout.addWidget(QLabel("📷 Исходное изображение"), 0)
        tab1_layout.addWidget(self.view_original, 1)

        tabs.addTab(tab1, "Исходное")

        # === Вкладка 2: Наложение + легенда ===
        tab2 = QWidget()
        tab2_layout = QHBoxLayout(tab2)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.view_result = ZoomableImageView(
            "Результат сегментации\n(Нет предсказания)")
        left_layout.addWidget(QLabel("🎨 Результат сегментации"), 0)
        left_layout.addWidget(self.view_result, 1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.legend_display = QTextEdit()
        self.legend_display.setReadOnly(True)
        self.legend_display.setHtml(self.create_legend())
        self.legend_display.setMaximumWidth(420)
        right_layout.addWidget(self.legend_display)

        tab2_layout.addWidget(left_panel, 3)
        tab2_layout.addWidget(right_panel, 1)

        tabs.addTab(tab2, "Наложение")
        self.tab_overlay = tab2

        # === Вкладка 3: История предсказаний ===
        tab_hist = QWidget()
        tab_hist_layout = QVBoxLayout(tab_hist)
        tab_hist_layout.setSpacing(10)
        tab_hist_layout.setContentsMargins(12, 12, 12, 12)

        self.label_history_info = QLabel("История: изображение не загружено")
        self.label_history_info.setWordWrap(True)
        self.label_history_info.setStyleSheet(
            "background: #171C25; padding: 8px; border-radius: 8px; color: #00E5FF;")
        tab_hist_layout.addWidget(self.label_history_info)

        self.list_history = QListWidget()
        self.list_history.itemClicked.connect(self.on_history_item_clicked)
        tab_hist_layout.addWidget(self.list_history, 1)

        hist_hint = QLabel("Клик по записи — открыть маску предсказания "
                           "(показ произойдёт на вкладке «Наложение»)")
        hist_hint.setStyleSheet("color: #8B95A5; font-size: 11px;")
        hist_hint.setWordWrap(True)
        tab_hist_layout.addWidget(hist_hint)

        tabs.addTab(tab_hist, "🕘 История")
        self.tab_history = tab_hist

        # === Вкладка 4: Пациенты и исследования (БД) ===
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)

        search_row = QHBoxLayout()
        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("🔍 Поиск по ФИО или номеру карты…")
        self.edit_search.textChanged.connect(self.refresh_studies_table)
        search_row.addWidget(self.edit_search, 1)

        self.btn_new_patient = QPushButton("Новый пациент")
        self.btn_new_patient.clicked.connect(self.create_patient_dialog)
        search_row.addWidget(self.btn_new_patient)

        self.btn_new_study = QPushButton("Новое исследование")
        self.btn_new_study.clicked.connect(self.create_study_dialog)
        search_row.addWidget(self.btn_new_study)
        tab3_layout.addLayout(search_row)

        self.table_studies = QTableWidget(0, 6)
        self.table_studies.setHorizontalHeaderLabels(
            ["ID", "Пациент", "Карта", "Дата", "Локализация", "Препаратов"])
        self.table_studies.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_studies.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table_studies.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_studies.setAlternatingRowColors(True)
        self.table_studies.verticalHeader().setVisible(False)
        self.table_studies.horizontalHeader().setStretchLastSection(True)
        self.table_studies.setColumnWidth(0, 60)
        self.table_studies.setColumnWidth(2, 110)
        self.table_studies.setColumnWidth(3, 100)
        self.table_studies.doubleClicked.connect(lambda _: self.on_activate_clicked())
        tab3_layout.addWidget(self.table_studies, 1)

        btn_row = QHBoxLayout()
        self.btn_activate_study = QPushButton("Открыть исследование")
        self.btn_activate_study.clicked.connect(self.on_activate_clicked)
        btn_row.addWidget(self.btn_activate_study)
        btn_row.addStretch()
        tab3_layout.addLayout(btn_row)

        self.label_study_summary = QLabel("Сводка: — (откройте исследование)")
        self.label_study_summary.setWordWrap(True)
        self.label_study_summary.setStyleSheet(
            "background: #171C25; padding: 8px; border-radius: 8px; color: #00E5FF;")
        tab3_layout.addWidget(self.label_study_summary)

        tabs.addTab(tab3, "🧑‍⚕️ Пациенты")
        self.tabs = tabs
        self.tab_patients = tab3

        layout.addWidget(tabs)

        return panel

    def create_legend(self, mask: Optional[np.ndarray] = None) -> str:
        """Легенда классов в HTML с процентами по площади."""
        html = "<h3>Легенда классов</h3><table border='1' cellpadding='4' width='100%'>"
        html += "<tr><th>Цвет</th><th>Класс</th><th>%</th></tr>"

        if mask is not None and mask.size > 0:
            counts = np.bincount(mask.ravel(), minlength=len(self.model_manager.CLASS_NAMES))
            total_pixels = mask.size
        else:
            counts = np.zeros(len(self.model_manager.CLASS_NAMES), dtype=np.float64)
            total_pixels = 1

        for class_id in range(len(self.model_manager.CLASS_NAMES)):
            name = self.model_manager.CLASS_NAMES[class_id]
            color = self.model_manager.CLASS_COLORS.get(class_id, (128, 128, 128))
            color_hex = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
            percentage = (counts[class_id] / total_pixels * 100.0) if total_pixels else 0.0

            html += "<tr>"
            html += f"<td style='background-color: {color_hex}; width: 22px; height: 18px; border: 1px solid rgba(255,255,255,0.35);'></td>"
            html += f"<td>{name}</td>"
            html += f"<td>{percentage:.2f}%</td>"
            html += "</tr>"

        html += "</table>"
        return html

    def _update_models_status(self):
        try:
            avail = self.model_manager.available_models()
        except Exception as e:
            self._status(f"❌ Ошибка проверки моделей: {e}", "red")
            return
        ok = [n for n, v in avail.items() if v["torch"] or v["onnx"]]
        if ok:
            onnx_ok = sum(1 for n in ok if avail[n]["onnx"])
            extra = f", из них ONNX: {onnx_ok}" if ONNX_AVAILABLE else ""
            self._status(f"✅ Готово. Модели ({len(ok)}{extra}) загрузятся при первом предсказании")
        else:
            self._status("❌ Не найдены ни чекпоинты, ни ONNX-модели", "red")

    def on_model_type_changed(self):
        is_ensemble = self.combo_model.currentIndex() == 1

        self.label_single_model.setVisible(not is_ensemble)
        self.combo_single.setVisible(not is_ensemble)
        self.label_weights.setVisible(is_ensemble)
        for w in (self.weight_unet, self.weight_deeplab, self.weight_unetpp,
                  self.label_w_unet, self.label_w_deeplab, self.label_w_unetpp):
            w.setVisible(is_ensemble)

    def on_alpha_changed(self):
        alpha_value = self.slider_alpha.value()
        self.label_alpha.setText(f"{alpha_value}%")

        if self.current_image_rgb is not None and self.current_mask is not None:
            self.update_display()

    def _active_view(self) -> ZoomableImageView:
        """Вью активной вкладки (для лупы и Ctrl+0)."""
        if self.tabs.currentWidget() is self.tab_overlay:
            return self.view_result
        return self.view_original

    def open_patients_tab(self):
        self.tabs.setCurrentWidget(self.tab_patients)

    # ------------------------------------------------------------------
    # Пациенты и исследования (БД)
    # ------------------------------------------------------------------

    def create_patient_dialog(self):
        dlg = PatientDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        d = dlg.data()
        if not d["full_name"] or not d["card_number"]:
            QMessageBox.warning(self, "Недостаточно данных",
                                "ФИО и номер карты обязательны.")
            return
        try:
            # ВНИМАНИЕ: у db.add_patient первым аргументом card_number, потом full_name
            pid = db.add_patient(
                self.db_conn,
                card_number=d["card_number"],
                full_name=d["full_name"],
                birth_date=d["birth_date"],
                sex=d["sex"].upper(),        # 'м'/'ж' -> 'М'/'Ж' (иначе запишется NULL)
                user_id=self.current_user_id,
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка БД",
                                 f"Не удалось создать пациента:\n{e}")
            return
        self.refresh_studies_table()
        self._status(f"Пациент #{pid} создан. Теперь создайте исследование.")

    def create_study_dialog(self):
        patients = db.find_patient(self.db_conn, "")   # '' -> LIKE '%%' -> все
        if not patients:
            QMessageBox.information(self, "Нет пациентов",
                                    "Сначала создайте пациента.")
            return
        names = [f"{p['full_name']} — карта {p['card_number']}" for p in patients]
        name, ok = QInputDialog.getItem(self, "Пациент", "Выберите пациента:",
                                        names, 0, False)
        if not ok:
            return
        patient = patients[names.index(name)]

        dlg = StudyDialog(f"{patient['full_name']} (карта {patient['card_number']})", self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        d = dlg.data()
        try:
            sid = db.add_study(self.db_conn, patient_id=patient["id"],
                               study_date=d["study_date"],
                               localization=d["localization"],
                               user_id=self.current_user_id)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка БД",
                                 f"Не удалось создать исследование:\n{e}")
            return
        self.refresh_studies_table()
        self._activate_study(sid)
        self._status(f"Исследование #{sid} создано и активно")

    def refresh_studies_table(self):
        """Список исследований. В db.list_studies нет параметра поиска —
        фильтруем в Python (заодно работает регистр кириллицы)."""
        self._studies_cache = db.list_studies(self.db_conn)
        q = self.edit_search.text().strip().lower() if hasattr(self, "edit_search") else ""
        rows = self._studies_cache
        if q:
            rows = [r for r in rows
                    if q in str(r["patient_name"]).lower()
                    or q in str(r["card_number"]).lower()]

        t = self.table_studies
        t.setRowCount(0)
        for r in rows:
            row = t.rowCount()
            t.insertRow(row)
            values = [str(r["id"]), r["patient_name"], r["card_number"],
                      r["study_date"], r["localization"] or "—", str(r["n_slides"])]
            for col, v in enumerate(values):
                t.setItem(row, col, QTableWidgetItem(v))

    def on_activate_clicked(self):
        idx = self.table_studies.currentRow()
        if idx < 0:
            QMessageBox.information(self, "Исследование",
                                    "Выберите исследование в таблице.")
            return
        self._activate_study(int(self.table_studies.item(idx, 0).text()))

    def _activate_study(self, study_id: int):
        """Активировать исследование (строка берётся из кэша list_studies)."""
        row = next((r for r in self._studies_cache if r["id"] == study_id), None)
        if row is None:
            return
        self.current_study_id = study_id
        self.label_study.setText(
            f"{row['patient_name']} (карта {row['card_number']})\n"
            f"Исследование #{row['id']} от {row['study_date']}\n"
            f"{row['localization'] or 'локализация не указана'}")
        self._update_study_summary()
        self._status(f"Активно исследование #{study_id} — можно загружать изображения")

    def _update_study_summary(self):
        """Сводка по исследованию: средние доли тканей по последним предсказаниям.
        В dict от db.get_study_summary есть служебный ключ n_images."""
        if self.current_study_id is None:
            self.label_study_summary.setText("Сводка: — (откройте исследование)")
            return
        s = db.get_study_summary(self.db_conn, self.current_study_id)
        if not s.get("n_images"):
            self.label_study_summary.setText(
                f"Сводка по исследованию #{self.current_study_id}: предсказаний нет")
            return
        parts = [f"{CLASS_RU.get(k, k)}: {v:.1f}%"
                 for k, v in s.items() if k != "n_images"]
        self.label_study_summary.setText(
            f"Сводка по {s['n_images']} изобр.: " + "; ".join(parts))

    # ------------------------------------------------------------------
    # Изображения (БД)
    # ------------------------------------------------------------------

    def _get_or_create_slide(self, study_id: int, image_path: str) -> int:
        """Слайд по имени файла в рамках исследования: найти или создать.
        Поиск — read-only SELECT; создание — db.add_slide."""
        filename = Path(image_path).name
        row = self.db_conn.execute(
            "SELECT id FROM slides WHERE study_id = ? AND filename = ? ORDER BY id LIMIT 1",
            (study_id, filename),
        ).fetchone()
        if row:
            return row["id"]
        return db.add_slide(self.db_conn, study_id, filename,
                            user_id=self.current_user_id)

    def load_image(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self, "Загрузить изображение", "",
            "Image Files (*.jpg *.jpeg *.png *.tif *.tiff);;All Files (*)")
        if not image_path:
            return
        self._import_image(image_path)

    def _import_image(self, image_path: str):
        """Импорт изображения в активное исследование и показ в UI."""
        image = cv2.imread(image_path)
        if image is None:
            QMessageBox.critical(self, "Ошибка",
                                 f"Ошибка загрузки изображения: {image_path}")
            return

        if self.current_study_id is None:
            QMessageBox.information(
                self, "Нужно исследование",
                "Изображение привязывается к исследованию пациента.\n\n"
                "Сначала создайте или откройте исследование на вкладке «Пациенты».")
            self.open_patients_tab()
            return

        try:
            slide_id = self._get_or_create_slide(self.current_study_id, image_path)
            image_id = db.import_image(self.db_conn, slide_id, image_path,
                                       user_id=self.current_user_id)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка БД",
                                 f"Не удалось импортировать изображение:\n{e}")
            return

        row = db.get_image(self.db_conn, image_id)
        self.current_image_id = image_id
        # Предсказание гоняем по копии в каталоге данных, а не по исходному файлу
        self.current_image_path = str(db.data_root() / row["path"])
        self.current_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.current_mask = None

        self.view_original.set_image(self.current_image_rgb)
        self.view_result.reset("Результат сегментации\n(Нет предсказания)")
        self.legend_display.setHtml(self.create_legend())
        self.label_image_info.setText(
            f"ID {image_id} · {row['width']}×{row['height']} · "
            f"sha256 {str(row['sha256'])[:12]}…")
        self.btn_predict.setEnabled(True)
        self.btn_save_mask.setEnabled(False)
        self.btn_save_overlay.setEnabled(False)

        self.refresh_history()
        self._update_study_summary()
        self._status(f"Изображение #{image_id} импортировано в исследование "
                     f"#{self.current_study_id}")

    # ------------------------------------------------------------------
    # Предсказание
    # ------------------------------------------------------------------

    def predict(self):
        if self.current_image_id is None or self.current_image_rgb is None:
            return
        use_ensemble = self.combo_model.currentIndex() == 1
        weights = {"unet": self.weight_unet.value(),
                   "deeplab": self.weight_deeplab.value(),
                   "unetpp": self.weight_unetpp.value()} if use_ensemble else {}

        self._last_predict_params = {
            "mode": "ensemble" if use_ensemble else "single",
            "model_key": self._current_model_key(),
            "tta": self.check_tta.isChecked(),
        }

        self.btn_predict.setEnabled(False)
        self.prediction_thread = PredictionWorker(
            self.model_manager, self.current_image_path,
            model_name=self._current_model_key(),
            use_tta=self.check_tta.isChecked(),
            use_ensemble=use_ensemble, weights=weights)
        self.prediction_thread.progress.connect(lambda s: self._status(s, "#FCEE0A"))
        self.prediction_thread.error.connect(self._on_prediction_error)
        self.prediction_thread.finished.connect(self._on_prediction_finished)
        self.prediction_thread.start()

    def _on_prediction_error(self, msg: str):
        self.btn_predict.setEnabled(True)
        self._status(msg, "red")
        QMessageBox.critical(self, "Ошибка предсказания", msg)

    def _on_prediction_finished(self):
        """Сохранение в БД — строго в главном потоке (sqlite-соединение
        принадлежит ему). Маску/оверлей на диск и class_areas пишет сама БД."""
        worker = self.prediction_thread
        mask = worker.result_mask
        if mask is None:
            self.btn_predict.setEnabled(True)
            return

        params = self._last_predict_params
        try:
            pred_id = db.save_prediction(
                self.db_conn,
                image_id=self.current_image_id,
                model_id=self.model_ids[params.get("model_key", "unet")],
                mask=mask,
                overlay=self._build_overlay(
                    self.current_image_rgb, mask,
                    self.slider_alpha.value() / 100.0),
                mode=params.get("mode", "single"),
                tta=params.get("tta", False),
                inference_time=worker.inference_ms,
                user_id=self.current_user_id,
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка БД",
                                 f"Предсказание не сохранено:\n{e}")
            self.btn_predict.setEnabled(True)
            return

        self.current_mask = mask
        self.update_display()
        self.legend_display.setHtml(self.create_legend(mask))
        self.btn_save_mask.setEnabled(True)
        self.btn_save_overlay.setEnabled(True)
        self.btn_predict.setEnabled(True)
        self.tabs.setCurrentWidget(self.tab_overlay)
        self.refresh_history()
        self._update_study_summary()
        self._status(f"Предсказание #{pred_id} сохранено "
                     f"({worker.inference_ms} мс, {params.get('mode')})")

    # ------------------------------------------------------------------
    # Визуализация
    # ------------------------------------------------------------------

    def _palette(self) -> np.ndarray:
        """Палитра классов (n, 3) из model_manager.CLASS_COLORS."""
        n = len(self.model_manager.CLASS_NAMES)
        size = max(n, 22)  # модели выдают до 22 классов
        palette = np.zeros((size, 3), dtype=np.uint8)
        for cid, color in self.model_manager.CLASS_COLORS.items():
            if 0 <= int(cid) < size:
                palette[int(cid)] = color
        return palette

    def _colorize(self, mask: np.ndarray) -> np.ndarray:
        """Цветная маска (RGB) по палитре классов."""
        palette = self._palette()
        safe = np.clip(mask, 0, palette.shape[0] - 1)
        return palette[safe]

    def _build_overlay(self, image_rgb: np.ndarray, mask: np.ndarray,
                       alpha: float = 0.5) -> np.ndarray:
        """Наложение цветной маски на изображение (RGB uint8)."""
        color = self._colorize(mask).astype(np.float32)
        base = image_rgb.astype(np.float32)
        return (base * (1.0 - alpha) + color * alpha).astype(np.uint8)

    def update_display(self):
        """Перерисовать вкладку «Наложение» согласно чекбоксам и прозрачности.
        keep_view=True — зум/панорамирование не сбрасываются."""
        if self.current_image_rgb is None:
            return
        if self.current_mask is None:
            self.view_result.reset("Результат сегментации\n(Нет предсказания)")
            return

        show_mask = self.check_show_mask.isChecked()
        show_overlay = self.check_show_overlay.isChecked()

        if show_overlay:
            img = self._build_overlay(self.current_image_rgb, self.current_mask,
                                      self.slider_alpha.value() / 100.0)
            self.view_result.set_image(img, keep_view=True)
        elif show_mask:
            self.view_result.set_image(self._colorize(self.current_mask), keep_view=True)
        else:
            self.view_result.set_image(self.current_image_rgb, keep_view=True)

    # ------------------------------------------------------------------
    # История предсказаний (БД)
    # ------------------------------------------------------------------

    def refresh_history(self):
        """История предсказаний активного изображения (db.get_prediction_history)."""
        self.list_history.clear()
        self._history_cache = {}
        if self.current_image_id is None:
            self.label_history_info.setText("История: изображение не загружено")
            return
        rows = db.get_prediction_history(self.db_conn, self.current_image_id)
        self.label_history_info.setText(
            f"История предсказаний изображения #{self.current_image_id}: "
            f"{len(rows)} шт.")
        for r in rows:
            text = (f"#{r['id']} · {r['model_name']} {r['model_version']} · "
                    f"{r['mode']}{' +TTA' if r['tta'] else ''} · {r['created_at']}")
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, r["id"])
            self.list_history.addItem(item)
            self._history_cache[r["id"]] = r   # все поля строки — для открытия

    def on_history_item_clicked(self, item: QListWidgetItem):
        pred_id = item.data(Qt.ItemDataRole.UserRole)
        row = self._history_cache.get(pred_id)
        if row is None:
            return
        mask = cv2.imread(str(db.data_root() / row["mask_path"]),
                          cv2.IMREAD_UNCHANGED)
        if mask is None:
            QMessageBox.warning(self, "Файл не найден",
                                f"Маска отсутствует на диске:\n{row['mask_path']}")
            return
        self.current_mask = mask
        self.update_display()
        self.legend_display.setHtml(self.create_legend(mask))
        self.btn_save_mask.setEnabled(True)
        self.btn_save_overlay.setEnabled(True)
        self.tabs.setCurrentWidget(self.tab_overlay)
        self._status(f"Открыто предсказание #{pred_id} "
                     f"({row['model_name']} {row['model_version']})")

    # ------------------------------------------------------------------
    # Экспорт
    # ------------------------------------------------------------------

    def save_mask(self):
        if self.current_mask is None:
            return
        default = f"mask_img{self.current_image_id or 0}.png"
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить маску", default,
                                              "PNG (*.png);;All Files (*)")
        if not path:
            return
        if cv2.imwrite(path, self.current_mask.astype(np.uint8)):
            self._status(f"Маска сохранена: {path}")
        else:
            QMessageBox.critical(self, "Ошибка", f"Не удалось записать файл:\n{path}")

    def save_overlay(self):
        if self.current_mask is None or self.current_image_rgb is None:
            return
        default = f"overlay_img{self.current_image_id or 0}.png"
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить наложение", default,
                                              "PNG (*.png);;All Files (*)")
        if not path:
            return
        overlay = self._build_overlay(self.current_image_rgb, self.current_mask,
                                      self.slider_alpha.value() / 100.0)
        if cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
            self._status(f"Наложение сохранено: {path}")
        else:
            QMessageBox.critical(self, "Ошибка", f"Не удалось записать файл:\n{path}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CYBER MED AI")
    app.setStyleSheet(APP_STYLESHEET)
    window = SegmentationApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()