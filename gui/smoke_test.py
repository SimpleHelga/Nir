"""
Смоук-тест интеграции GUI + БД (без экрана и без инференса моделей).

Запуск из корня проекта:
    QT_QPA_PLATFORM=offscreen .venv/bin/python gui/smoke_test.py

Проверяет сценарий врача: пациент -> исследование -> импорт изображения ->
предсказание (маска подставляется) -> запись в БД -> история -> сводка.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # корень проекта

import cv2
import numpy as np

# Каталог данных -> во временную папку, чтобы тест не трогал реальные данные
import gui.database as db
TMP = Path(tempfile.mkdtemp(prefix="nirseg_test_"))
db.data_root = lambda: TMP  # type: ignore[assignment]

from PyQt6.QtWidgets import QApplication  # noqa: E402

QApplication.setApplicationName("NirSeg smoke test")

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

app = QApplication([])

from gui.mainwindow import SegmentationApp  # noqa: E402

window = SegmentationApp()
window.show()

# --- 1. Пациент и исследование (как через диалоги, но без UI) ---
patient_id = db.add_patient(window.db_conn, "К-2026/0001", "Тестова Тест Тестовна",
                            "1980-01-01", "Ж")
study_id = db.add_study(window.db_conn, patient_id, "2026-09-02",
                        "левая молочная железа", "протоковая карцинома in situ")
window._set_active_study(study_id)
assert window.current_study_id == study_id
assert "Тестова" in window.label_study.text()
print("✅ 1. Пациент и исследование созданы, активное исследование установлено")

# --- 2. Импорт изображения (минуя файловый диалог) ---
test_img = TMP / "input.png"
cv2.imwrite(str(test_img), np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
window._import_image(str(test_img))
assert window.current_image_id is not None
assert Path(window.current_image_path).exists()  # работаем с копией в data/
assert (TMP / "data" / f"study_{study_id}" / "images" / "input.png").exists()
print("✅ 2. Изображение импортировано в каталог данных и привязано к исследованию")

# --- 3. «Предсказание»: подставляем маску и вызываем сохранение в БД ---
window._last_predict_params = {"model_name": "unet", "use_ensemble": False, "use_tta": True}
window.current_mask = np.random.choice([1, 2, 3, 4, 9], size=(512, 512)).astype(np.uint8)
pred_id = window._save_prediction_to_db()
assert pred_id is not None, "предсказание не сохранилось в БД"

# ансамбль — вторая запись
window._last_predict_params = {"model_name": None, "use_ensemble": True, "use_tta": True}
window.current_mask = np.random.choice([1, 2, 3, 4, 9], size=(512, 512)).astype(np.uint8)
pred_id2 = window._save_prediction_to_db()
assert pred_id2 is not None
print(f"✅ 3. Предсказания сохранены в БД (№{pred_id}, №{pred_id2})")

# --- 4. История предсказаний в UI ---
assert window.list_history.count() == 2, f"ожидалось 2 записи, есть {window.list_history.count()}"
print("✅ 4. Список истории предсказаний обновился (2 записи)")

# --- 5. Открытие маски из истории ---
item = window.list_history.item(0)  # самая свежая (ансамбль)
window.on_history_item_clicked(item)
assert window.current_mask is not None
print("✅ 5. Маска из истории открыта")

# --- 6. Сводка по исследованию ---
summary = db.get_study_summary(window.db_conn, study_id)
assert summary["n_images"] == 1
assert "Сводка" in window.label_study_summary.text()
print(f"✅ 6. Сводка по исследованию: {window.label_study_summary.text()}")

# --- 7. Таблица исследований и поиск ---
window.refresh_studies_table()
assert window.table_studies.rowCount() == 1
window.edit_search.setText("нет такого пациента")
window.refresh_studies_table()
assert window.table_studies.rowCount() == 0
window.edit_search.setText("К-2026")
window.refresh_studies_table()
assert window.table_studies.rowCount() == 1
print("✅ 7. Таблица исследований и поиск по карте работают")

# --- 8. Аудит ---
n_audit = window.db_conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
assert n_audit >= 4  # patient, study, image, 2 predictions (slide создаётся без аудита)
print(f"✅ 8. Audit log: {n_audit} записей")

# --- 9. Повторный запуск приложения не ломает схему (миграции идемпотентны) ---
conn2 = db.init_db(TMP / "nir.db")
assert conn2.execute("PRAGMA user_version").fetchone()[0] >= 1
print("✅ 9. Повторная инициализация БД безопасна")

window.close()
print("\n🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ — GUI и БД работают вместе.")
