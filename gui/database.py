"""
База данных приложения (SQLite) — реализация логической схемы.

Сущности:
    users, patients, studies, slides, images, predictions,
    doctor_edits (правки врача), reports (заключения), audit_log
    + models (реестр версий моделей — обязательна, т.к. predictions.model_id
    ссылается на неё; на исходной диаграмме она пропущена).

Принципы:
  * В БД — только метаданные и ОТНОСИТЕЛЬНЫЕ пути к файлам.
  * Файлы лежат в data/study_<id>/... внутри каталога данных; каталог можно
    переносить/бэкапить целиком вместе с базой.
  * Каждое предсказание ссылается на конкретную версию модели (трассируемость).
  * Любое изменение данных пишется в audit_log АТОМАРНО с самой операцией.
  * Схема после миграций сверяется с эталоном (verify_schema) — несоответствие
    обнаруживается при старте, а не на первом INSERT.

Зависимости: стандартная библиотека + opencv-python + numpy.
Проверка: python -m gui.database
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

APP_NAME = "NirSeg"


# ---------------------------------------------------------------------------
# Каталог данных и подключение
# ---------------------------------------------------------------------------

def data_root() -> Path:
    """Каталог данных приложения.

    Порядок определения:
      1) переменная окружения NIRSEG_DATA_DIR (тесты/демо/переносимый режим);
      2) платформенный каталог: %APPDATA%/NirSeg или ~/.local/share/NirSeg.
    """
    env = os.environ.get("NIRSEG_DATA_DIR")
    if env:
        root = Path(env)
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        root = base / APP_NAME
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        root = base / APP_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def db_path() -> Path:
    return data_root() / "nir.db"


def connect(path: Optional[Path] = None) -> sqlite3.Connection:
    """Подключение: foreign keys, WAL, таймаут ожидания блокировки.

    ВАЖНО: соединение sqlite3 нельзя использовать из нескольких потоков
    одновременно. Для фоновых задач (инференс, экспорт) открывайте отдельное
    подключение в этом потоке через connect().
    """
    conn = sqlite3.connect(path or db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


# ---------------------------------------------------------------------------
# Транзакции и аудит
# ---------------------------------------------------------------------------

@contextmanager
def tx(conn: sqlite3.Connection):
    """Атомарная транзакция: основная операция + audit-записи фиксируются
    (или откатываются) вместе. Не вкладывать друг в друга."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def audit(conn: sqlite3.Connection, action: str, entity: str,
          entity_id: Optional[int], user_id: Optional[int] = None,
          details: Optional[Dict[str, Any]] = None) -> None:
    """INSERT в audit_log. Вызывать ВНУТРИ tx(), рядом с основной операцией."""
    conn.execute(
        "INSERT INTO audit_log (user_id, action, entity, entity_id, details) "
        "VALUES (?,?,?,?,?)",
        (user_id, action, entity, entity_id,
         json.dumps(details, ensure_ascii=False) if details else None),
    )


# ---------------------------------------------------------------------------
# Служебное
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _stamp() -> str:
    """Штамп для имён файлов: с микросекундами — исключает коллизии имён
    при двух предсказаниях/правках за одну секунду."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _study_dir(conn: sqlite3.Connection, study_id: int) -> Path:
    d = data_root() / "data" / f"study_{study_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _rel(path: Path) -> str:
    """Относительный путь файла внутри каталога данных (для хранения в БД)."""
    return path.relative_to(data_root()).as_posix()


# ---------------------------------------------------------------------------
# Миграции
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2

_MIGRATIONS: Dict[int, str] = {
    # v1 — исходная схема (не менять: существующие БД уже на ней)
    1: """
    CREATE TABLE users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        login         TEXT UNIQUE NOT NULL,
        full_name     TEXT NOT NULL,
        role          TEXT NOT NULL CHECK (role IN ('doctor', 'admin')),
        password_hash TEXT NOT NULL,
        created_at    TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE patients (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        card_number TEXT UNIQUE NOT NULL,
        full_name   TEXT NOT NULL,
        birth_date  TEXT,
        sex         TEXT CHECK (sex IN ('М', 'Ж', '')),
        created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE studies (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id   INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
        study_date   TEXT NOT NULL,
        localization TEXT,
        diagnosis    TEXT,
        doctor_id    INTEGER REFERENCES users(id),
        notes        TEXT,
        created_at   TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE slides (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        study_id   INTEGER NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
        filename   TEXT NOT NULL,
        stain      TEXT DEFAULT 'H&E',
        mpp        REAL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE images (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        slide_id   INTEGER NOT NULL REFERENCES slides(id) ON DELETE CASCADE,
        path       TEXT NOT NULL,
        width      INTEGER NOT NULL,
        height     INTEGER NOT NULL,
        sha256     TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE models (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        name       TEXT NOT NULL,
        arch       TEXT,
        encoder    TEXT,
        version    TEXT NOT NULL,
        ckpt_path  TEXT,
        mean_dice  REAL,
        is_active  INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE (name, version)
    );

    CREATE TABLE predictions (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id      INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
        model_id      INTEGER NOT NULL REFERENCES models(id),
        mode          TEXT NOT NULL CHECK (mode IN ('single', 'ensemble')),
        use_tta       INTEGER NOT NULL DEFAULT 0,
        mask_path     TEXT NOT NULL,
        overlay_path  TEXT,
        class_areas   TEXT NOT NULL,
        inference_ms  INTEGER,
        created_by    INTEGER REFERENCES users(id),
        created_at    TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE corrections (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
        mask_path     TEXT NOT NULL,
        note          TEXT,
        corrected_by  INTEGER REFERENCES users(id),
        created_at    TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE reports (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        study_id   INTEGER NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
        file_path  TEXT NOT NULL,
        created_by INTEGER REFERENCES users(id),
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE audit_log (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id   INTEGER REFERENCES users(id),
        action    TEXT NOT NULL,
        entity    TEXT NOT NULL,
        entity_id INTEGER,
        details   TEXT,
        ts        TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE INDEX idx_studies_patient ON studies(patient_id);
    CREATE INDEX idx_slides_study    ON slides(study_id);
    CREATE INDEX idx_images_slide    ON images(slide_id);
    CREATE INDEX idx_predictions_img ON predictions(image_id);
    """,

    # v2 — приведение к логической схеме:
    #   corrections -> doctor_edits (+ имена полей как на диаграмме),
    #   predictions: use_tta -> tta, inference_ms -> inference_time,
    #   reports: created_by -> user_id, + колонка result,
    #   audit_log: ts -> created_at,
    #   индексы на внешние ключи (SQLite не строит их сам — без них каскады
    #   и JOIN'ы делают full scan). Требуется SQLite >= 3.25.
    2: """
    ALTER TABLE corrections RENAME TO doctor_edits;
    ALTER TABLE doctor_edits RENAME COLUMN note TO comment;
    ALTER TABLE doctor_edits RENAME COLUMN corrected_by TO user_id;

    ALTER TABLE predictions RENAME COLUMN use_tta TO tta;
    ALTER TABLE predictions RENAME COLUMN inference_ms TO inference_time;

    ALTER TABLE reports RENAME COLUMN created_by TO user_id;
    ALTER TABLE reports ADD COLUMN result TEXT;

    ALTER TABLE audit_log RENAME COLUMN ts TO created_at;

    CREATE INDEX idx_doctor_edits_pred ON doctor_edits(prediction_id);
    CREATE INDEX idx_reports_study     ON reports(study_id);
    CREATE INDEX idx_predictions_model ON predictions(model_id);
    CREATE INDEX idx_images_sha        ON images(sha256);
    CREATE INDEX idx_studies_doctor    ON studies(doctor_id);
    CREATE INDEX idx_audit_entity      ON audit_log(entity, entity_id);
    """,
}

# Эталон колонок после всех миграций — сверяется в verify_schema().
_EXPECTED_COLUMNS: Dict[str, set] = {
    "users":         {"id", "login", "full_name", "role", "password_hash", "created_at"},
    "patients":      {"id", "card_number", "full_name", "birth_date", "sex", "created_at"},
    "studies":       {"id", "patient_id", "study_date", "localization", "diagnosis",
                      "doctor_id", "notes", "created_at"},
    "slides":        {"id", "study_id", "filename", "stain", "mpp", "created_at"},
    "images":        {"id", "slide_id", "path", "width", "height", "sha256", "created_at"},
    "models":        {"id", "name", "arch", "encoder", "version", "ckpt_path",
                      "mean_dice", "is_active", "created_at"},
    "predictions":   {"id", "image_id", "model_id", "mode", "tta", "mask_path",
                      "overlay_path", "class_areas", "inference_time",
                      "created_by", "created_at"},
    "doctor_edits":  {"id", "prediction_id", "user_id", "mask_path", "comment", "created_at"},
    "reports":       {"id", "study_id", "user_id", "file_path", "result", "created_at"},
    "audit_log":     {"id", "user_id", "action", "entity", "entity_id", "details", "created_at"},
}


def verify_schema(conn: sqlite3.Connection) -> None:
    """Проверяет, что все таблицы содержат ожидаемые колонки.

    Ловит рассинхронизацию кода и схемы при старте, а не на первом INSERT.
    Лишние колонки не считаются ошибкой (прямая совместимость).
    """
    for table, need in _EXPECTED_COLUMNS.items():
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if not cols:
            raise RuntimeError(f"Таблица '{table}' отсутствует в базе")
        missing = need - cols
        if missing:
            raise RuntimeError(
                f"Таблица '{table}': отсутствуют колонки {sorted(missing)}")


def init_db(path: Optional[Path] = None) -> sqlite3.Connection:
    """Создать/мигрировать базу, проверить схему и вернуть подключение.

    Каждая миграция выполняется атомарно (BEGIN ... COMMIT): при ошибке
    база остаётся на предыдущей версии, а не в наполовину изменённом виде.
    """
    conn = connect(path)
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"База создана более новой версией приложения "
            f"(schema v{current} > v{SCHEMA_VERSION})")
    for version in range(current + 1, SCHEMA_VERSION + 1):
        script = f"BEGIN;\n{_MIGRATIONS[version]}\nPRAGMA user_version = {version};\nCOMMIT;"
        try:
            conn.executescript(script)
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
    verify_schema(conn)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Пользователи / пациенты / исследования
# ---------------------------------------------------------------------------

def add_user(conn: sqlite3.Connection, login: str, full_name: str, role: str,
             password_hash: str) -> int:
    """Создаёт пользователя. password_hash — готовый хеш (bcrypt/argon2),
    эту функцию не интересует способ хеширования."""
    if role not in ("doctor", "admin"):
        raise ValueError(f"Неизвестная роль: {role}")
    with tx(conn):
        cur = conn.execute(
            "INSERT INTO users (login, full_name, role, password_hash) VALUES (?,?,?,?)",
            (login, full_name, role, password_hash),
        )
        audit(conn, "create", "users", cur.lastrowid)
    return cur.lastrowid


def add_patient(conn: sqlite3.Connection, card_number: str, full_name: str,
                birth_date: str = "", sex: str = "",
                user_id: Optional[int] = None) -> int:
    if sex not in ("М", "Ж"):
        sex = None  # неизвестный пол -> NULL, а не пустая строка
    with tx(conn):
        cur = conn.execute(
            "INSERT INTO patients (card_number, full_name, birth_date, sex) VALUES (?,?,?,?)",
            (card_number, full_name, birth_date, sex),
        )
        audit(conn, "create", "patients", cur.lastrowid, user_id)
    return cur.lastrowid


def find_patient(conn: sqlite3.Connection, query: str) -> List[sqlite3.Row]:
    """Поиск пациента по подстроке номера карты или ФИО.

    Спецсимволы LIKE (% и _) экранируются. Учтите: lower() в SQLite работает
    только для ASCII — для регистронезависимого поиска по кириллице нужна
    нормализация при записи или ICU-коллация.
    """
    q = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    like = f"%{q}%"
    return conn.execute(
        """SELECT * FROM patients
           WHERE card_number LIKE ? ESCAPE '\\' OR full_name LIKE ? ESCAPE '\\'
           ORDER BY full_name""",
        (like, like),
    ).fetchall()


def add_study(conn: sqlite3.Connection, patient_id: int, study_date: str,
              localization: str = "", diagnosis: str = "",
              doctor_id: Optional[int] = None, notes: str = "",
              user_id: Optional[int] = None) -> int:
    if conn.execute("SELECT 1 FROM patients WHERE id = ?",
                    (patient_id,)).fetchone() is None:
        raise ValueError(f"Пациент {patient_id} не найден")
    with tx(conn):
        cur = conn.execute(
            """INSERT INTO studies (patient_id, study_date, localization, diagnosis,
                                    doctor_id, notes)
               VALUES (?,?,?,?,?,?)""",
            (patient_id, study_date, localization, diagnosis, doctor_id, notes),
        )
        audit(conn, "create", "studies", cur.lastrowid, user_id)
    return cur.lastrowid


def delete_study(conn: sqlite3.Connection, study_id: int,
                 user_id: Optional[int] = None) -> None:
    """Удаляет исследование: строки (каскадно, вместе с предсказаниями и
    правками) + каталог с файлами на диске."""
    if conn.execute("SELECT 1 FROM studies WHERE id = ?",
                    (study_id,)).fetchone() is None:
        raise ValueError(f"Исследование {study_id} не найдено")
    with tx(conn):
        conn.execute("DELETE FROM studies WHERE id = ?", (study_id,))
        audit(conn, "delete", "studies", study_id, user_id)
    # файлы удаляем только после успешного коммита
    shutil.rmtree(data_root() / "data" / f"study_{study_id}", ignore_errors=True)


# ---------------------------------------------------------------------------
# Слайды и изображения
# ---------------------------------------------------------------------------

def add_slide(conn: sqlite3.Connection, study_id: int, filename: str,
              stain: str = "H&E", mpp: Optional[float] = None,
              user_id: Optional[int] = None) -> int:
    with tx(conn):
        cur = conn.execute(
            "INSERT INTO slides (study_id, filename, stain, mpp) VALUES (?,?,?,?)",
            (study_id, filename, stain, mpp),
        )
        audit(conn, "create", "slides", cur.lastrowid, user_id)
    return cur.lastrowid


def import_image(conn: sqlite3.Connection, slide_id: int, src_path: str | Path,
                 user_id: Optional[int] = None) -> int:
    """Копирует изображение в каталог данных и регистрирует в БД.

    * Дедупликация по sha256: повторный импорт того же файла в тот же слайд
      возвращает существующий id, ничего не копируя.
    * ВАЖНО: cv2.imread не читает WSI/SVS — для них вырежьте тайл/уровень
      через openslide и передайте сюда готовый PNG/TIFF.
    """
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(src)
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {src}")
    h, w = img.shape[:2]

    slide = conn.execute("SELECT study_id FROM slides WHERE id = ?",
                         (slide_id,)).fetchone()
    if slide is None:
        raise ValueError(f"Слайд {slide_id} не найден")

    digest = _sha256_file(src)
    dup = conn.execute(
        "SELECT id FROM images WHERE slide_id = ? AND sha256 = ?",
        (slide_id, digest),
    ).fetchone()
    if dup:
        return dup["id"]

    dest_dir = _study_dir(conn, slide["study_id"]) / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():  # другое содержимое под тем же именем — не перезаписываем
        dest = dest_dir / f"{src.stem}_{_stamp()}{src.suffix}"
    shutil.copy2(src, dest)

    with tx(conn):
        cur = conn.execute(
            "INSERT INTO images (slide_id, path, width, height, sha256) VALUES (?,?,?,?,?)",
            (slide_id, _rel(dest), w, h, digest),
        )
        audit(conn, "create", "images", cur.lastrowid, user_id, {"file": dest.name})
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Модели и предсказания
# ---------------------------------------------------------------------------

def register_model(conn: sqlite3.Connection, name: str, version: str,
                   arch: str = "", encoder: str = "", ckpt_path: str = "",
                   mean_dice: Optional[float] = None, is_active: int = 1) -> int:
    """Регистрирует версию модели. Повторный вызов возвращает существующий id
    (идемпотентно, безопасно при гонке из параллельного потока)."""
    row = conn.execute(
        "SELECT id FROM models WHERE name = ? AND version = ?", (name, version)
    ).fetchone()
    if row:
        return row["id"]
    try:
        with tx(conn):
            cur = conn.execute(
                """INSERT INTO models (name, version, arch, encoder, ckpt_path,
                                       mean_dice, is_active)
                   VALUES (?,?,?,?,?,?,?)""",
                (name, version, arch, encoder, ckpt_path, mean_dice, is_active),
            )
            new_id = cur.lastrowid
            audit(conn, "create", "models", new_id, None,
                  {"name": name, "version": version})
        return new_id
    except sqlite3.IntegrityError:
        # гонка: другой поток зарегистрировал первым — берём его id
        row = conn.execute(
            "SELECT id FROM models WHERE name = ? AND version = ?", (name, version)
        ).fetchone()
        if row is None:
            raise
        return row["id"]


def compute_class_areas(mask: np.ndarray) -> Dict[str, float]:
    """Доли площадей классов (в %) по 5 диагностическим классам + прочее."""
    total = mask.size
    areas = {}
    for cid, name in [(1, "tumor"), (2, "stroma"), (3, "lymphocytic"),
                      (4, "necrosis"), (9, "fat")]:
        areas[name] = round(float((mask == cid).sum()) / total * 100.0, 2)
    areas["other"] = round(100.0 - sum(areas.values()), 2)
    return areas


def save_prediction(conn: sqlite3.Connection, image_id: int, model_id: int,
                    mask: np.ndarray, overlay: Optional[np.ndarray] = None,
                    mode: str = "single", tta: bool = False,
                    inference_time: Optional[int] = None,
                    user_id: Optional[int] = None) -> int:
    """Сохраняет маску (и оверлей) на диск и запись предсказания в БД.

    inference_time — длительность инференса в МИЛЛИСЕКУНДАХ.
    """
    row = conn.execute(
        """SELECT s.id AS study_id
           FROM images i
           JOIN slides sl ON sl.id = i.slide_id
           JOIN studies s ON s.id = sl.study_id
           WHERE i.id = ?""",
        (image_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Изображение {image_id} не найдено")

    pred_dir = _study_dir(conn, row["study_id"]) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()

    mask_file = pred_dir / f"img{image_id}_model{model_id}_{stamp}_mask.png"
    cv2.imwrite(str(mask_file), mask.astype(np.uint8))

    overlay_file = None
    if overlay is not None:
        overlay_file = pred_dir / f"img{image_id}_model{model_id}_{stamp}_overlay.png"
        cv2.imwrite(str(overlay_file), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    with tx(conn):
        cur = conn.execute(
            """INSERT INTO predictions
               (image_id, model_id, mode, tta, mask_path, overlay_path,
                class_areas, inference_time, created_by)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                image_id, model_id, mode, int(tta),
                _rel(mask_file),
                _rel(overlay_file) if overlay_file else None,
                json.dumps(compute_class_areas(mask), ensure_ascii=False),
                inference_time, user_id,
            ),
        )
        audit(conn, "create", "predictions", cur.lastrowid, user_id,
              {"model_id": model_id, "mode": mode, "tta": tta})
    return cur.lastrowid


def add_doctor_edit(conn: sqlite3.Connection, prediction_id: int,
                    corrected_mask: np.ndarray, comment: str = "",
                    user_id: Optional[int] = None) -> int:
    """Правка врача (таблица doctor_edits): сохраняется как НОВАЯ маска,
    оригинальное предсказание не изменяется — полная история правок."""
    pred = conn.execute(
        "SELECT mask_path FROM predictions WHERE id = ?", (prediction_id,)
    ).fetchone()
    if pred is None:
        raise ValueError(f"Предсказание {prediction_id} не найдено")

    old_mask = data_root() / pred["mask_path"]
    corr_file = old_mask.parent / f"{old_mask.stem}_corrected_{_stamp()}.png"
    cv2.imwrite(str(corr_file), corrected_mask.astype(np.uint8))

    with tx(conn):
        cur = conn.execute(
            """INSERT INTO doctor_edits (prediction_id, user_id, mask_path, comment)
               VALUES (?,?,?,?)""",
            (prediction_id, user_id, _rel(corr_file), comment),
        )
        audit(conn, "create", "doctor_edits", cur.lastrowid, user_id)
    return cur.lastrowid


def list_doctor_edits(conn: sqlite3.Connection, prediction_id: int) -> List[sqlite3.Row]:
    """История правок врача по предсказанию (для GUI)."""
    return conn.execute(
        """SELECT de.*, u.full_name AS editor
           FROM doctor_edits de LEFT JOIN users u ON u.id = de.user_id
           WHERE de.prediction_id = ?
           ORDER BY de.created_at DESC, de.id DESC""",
        (prediction_id,),
    ).fetchall()


# ---------------------------------------------------------------------------
# Заключения (reports)
# ---------------------------------------------------------------------------

def add_report(conn: sqlite3.Connection, study_id: int, file_path: str,
               result: str = "", user_id: Optional[int] = None) -> int:
    """Регистрирует заключение.

    file_path — путь к файлу (PDF/DOCX/TXT): относительный внутри каталога
    данных или абсолютный. Файл должен существовать. result — текст
    заключения (поля 'result' на логической схеме).
    """
    p = Path(file_path)
    if not p.is_absolute():
        p = data_root() / p
    if not p.exists():
        raise FileNotFoundError(f"Файл заключения не найден: {p}")

    if conn.execute("SELECT 1 FROM studies WHERE id = ?",
                    (study_id,)).fetchone() is None:
        raise ValueError(f"Исследование {study_id} не найдено")

    rel = p.relative_to(data_root()).as_posix() if p.is_relative_to(data_root()) else str(p)
    with tx(conn):
        cur = conn.execute(
            "INSERT INTO reports (study_id, user_id, file_path, result) VALUES (?,?,?,?)",
            (study_id, user_id, rel, result),
        )
        audit(conn, "create", "reports", cur.lastrowid, user_id)
    return cur.lastrowid


def list_reports(conn: sqlite3.Connection, study_id: int) -> List[sqlite3.Row]:
    return conn.execute(
        """SELECT r.*, u.full_name AS author
           FROM reports r LEFT JOIN users u ON u.id = r.user_id
           WHERE r.study_id = ?
           ORDER BY r.created_at DESC, r.id DESC""",
        (study_id,),
    ).fetchall()


# ---------------------------------------------------------------------------
# Выборки для GUI
# ---------------------------------------------------------------------------

def list_studies(conn: sqlite3.Connection,
                 patient_id: Optional[int] = None) -> List[sqlite3.Row]:
    """Список исследований с числом препаратов и временем последнего предсказания."""
    sql = """
        SELECT st.*, p.full_name AS patient_name, p.card_number,
               (SELECT COUNT(*) FROM slides sl WHERE sl.study_id = st.id) AS n_slides,
               (SELECT MAX(pr.created_at) FROM predictions pr
                JOIN images i ON i.id = pr.image_id
                JOIN slides sl ON sl.id = i.slide_id
                WHERE sl.study_id = st.id) AS last_prediction_at
        FROM studies st JOIN patients p ON p.id = st.patient_id
    """
    params: tuple = ()
    if patient_id is not None:
        sql += " WHERE st.patient_id = ?"
        params = (patient_id,)
    sql += " ORDER BY st.created_at DESC, st.id DESC"
    return conn.execute(sql, params).fetchall()


def list_study_images(conn: sqlite3.Connection, study_id: int) -> List[sqlite3.Row]:
    """Все изображения исследования — для навигации в GUI."""
    return conn.execute(
        """SELECT i.*, sl.filename AS slide_filename, sl.stain
           FROM images i JOIN slides sl ON sl.id = i.slide_id
           WHERE sl.study_id = ? ORDER BY i.id""",
        (study_id,),
    ).fetchall()


def get_study_summary(conn: sqlite3.Connection, study_id: int) -> Dict[str, Any]:
    """Средние доли тканей по последнему предсказанию каждого изображения
    исследования (основа для заключения)."""
    rows = conn.execute(
        """
        WITH latest AS (
            SELECT image_id, MAX(id) AS pred_id
            FROM predictions
            GROUP BY image_id
        )
        SELECT pr.class_areas
        FROM predictions pr
        JOIN latest l ON l.pred_id = pr.id
        JOIN images i ON i.id = pr.image_id
        JOIN slides sl ON sl.id = i.slide_id
        WHERE sl.study_id = ?
        """,
        (study_id,),
    ).fetchall()

    if not rows:
        return {"n_images": 0}

    sums: Dict[str, float] = {}
    for r in rows:
        for k, v in json.loads(r["class_areas"]).items():
            sums[k] = sums.get(k, 0.0) + v
    n = len(rows)
    return {"n_images": n, **{k: round(v / n, 2) for k, v in sums.items()}}


def get_prediction_history(conn: sqlite3.Connection, image_id: int) -> List[sqlite3.Row]:
    return conn.execute(
        """SELECT pr.*, m.name AS model_name, m.version AS model_version
           FROM predictions pr JOIN models m ON m.id = pr.model_id
           WHERE pr.image_id = ?
           ORDER BY pr.created_at DESC, pr.id DESC""",
        (image_id,),
    ).fetchall()


def get_image(conn: sqlite3.Connection, image_id: int) -> sqlite3.Row:
    return conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()


# ---------------------------------------------------------------------------
# Демонстрация / самопроверка: python -m gui.database
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    conn = init_db()
    print("База создана:", db_path())
    conn.close()