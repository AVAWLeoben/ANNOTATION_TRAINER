from __future__ import annotations

import hashlib
import math
import random
import sqlite3
import struct
import sys
import tempfile
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from PySide6.QtCore import Qt, QTimer, QUrl, Signal, QRectF
from PySide6.QtGui import QColor, QFont, QKeySequence, QMouseEvent, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtMultimedia import QSoundEffect
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_TEXT_FILES = ("classes.txt", "class_names.txt", "names.txt", "obj.names")
YAML_FILES = ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml")
DB_FILENAME = ".yolo_trainer_stats.sqlite3"
APP_VERSION = "4.3-shortcuts-2026-08-10"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ------------------------------ Data model ---------------------------------


@dataclass
class YoloBox:
    class_id: int
    cx: float
    cy: float
    width: float
    height: float
    solved: bool = False

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains(self, x: float, y: float) -> bool:
        left = self.cx - self.width / 2
        top = self.cy - self.height / 2
        right = self.cx + self.width / 2
        bottom = self.cy + self.height / 2
        return left <= x <= right and top <= y <= bottom


@dataclass
class Sample:
    image_path: Path
    label_path: Path
    boxes: list[YoloBox]


@dataclass
class Dataset:
    root: Path
    classes: list[str]
    samples: list[Sample]
    class_source: str
    missing_labels: int = 0

    @property
    def total_objects(self) -> int:
        return sum(len(sample.boxes) for sample in self.samples)


@dataclass
class TrainingTask:
    name: str
    path: Path
    image_count: int
    class_count: int
    class_source: str


# ----------------------------- Dataset loading -----------------------------


def _parse_yolo_label(label_path: Path) -> list[YoloBox]:
    boxes: list[YoloBox] = []

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = label_path.read_text(encoding="latin-1").splitlines()

    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{label_path} line {line_number}: expected 5 YOLO detection values "
                f"(class cx cy width height), got {len(parts)}."
            )

        try:
            class_id = int(float(parts[0]))
            cx, cy, width, height = map(float, parts[1:])
        except ValueError as exc:
            raise ValueError(
                f"{label_path} line {line_number}: invalid numeric annotation."
            ) from exc

        if class_id < 0:
            raise ValueError(f"{label_path} line {line_number}: negative class id.")

        vals = (cx, cy, width, height)
        if any(not math.isfinite(v) for v in vals):
            raise ValueError(f"{label_path} line {line_number}: non-finite coordinate.")
        if width <= 0 or height <= 0:
            raise ValueError(f"{label_path} line {line_number}: box has zero/negative size.")

        boxes.append(
            YoloBox(
                class_id=class_id,
                cx=min(1.0, max(0.0, cx)),
                cy=min(1.0, max(0.0, cy)),
                width=min(1.0, max(0.0, width)),
                height=min(1.0, max(0.0, height)),
            )
        )

    return boxes


def _candidate_label_paths(root: Path, image_path: Path) -> list[Path]:
    """Return plausible YOLO label paths for mixed and split layouts."""
    candidates: list[Path] = [image_path.with_suffix(".txt")]

    try:
        relative = image_path.relative_to(root)
    except ValueError:
        relative = image_path

    parts = list(relative.parts)
    for i, part in enumerate(parts[:-1]):
        if part.lower() == "images":
            replaced = parts.copy()
            replaced[i] = "labels"
            candidates.append((root.joinpath(*replaced)).with_suffix(".txt"))

    images_root = root / "images"
    labels_root = root / "labels"
    try:
        relative_to_images = image_path.relative_to(images_root)
    except ValueError:
        relative_to_images = None
    if relative_to_images is not None:
        candidates.append((labels_root / relative_to_images).with_suffix(".txt"))

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _load_classes(root: Path) -> tuple[list[str], str]:
    for filename in CLASS_TEXT_FILES:
        path = root / filename
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="latin-1")
            names = [line.strip() for line in text.splitlines() if line.strip()]
            if names:
                return names, filename

    for filename in YAML_FILES:
        path = root / filename
        if not path.is_file():
            continue

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        names = data.get("names")
        if isinstance(names, list):
            parsed = [str(name) for name in names]
            if parsed:
                return parsed, filename
        if isinstance(names, dict):
            try:
                parsed = [str(names[key]) for key in sorted(names, key=lambda x: int(x))]
            except (TypeError, ValueError):
                parsed = [str(value) for _, value in sorted(names.items(), key=lambda kv: str(kv[0]))]
            if parsed:
                return parsed, filename

    return [], "not found"


def discover_tasks(training_root: Path) -> list[TrainingTask]:
    """Every immediate child directory of TRAINING_DATA is one training task."""
    root = training_root.expanduser().resolve()
    if not root.is_dir():
        return []

    tasks: list[TrainingTask] = []
    for child in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.casefold()):
        image_count = sum(
            1 for p in child.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if image_count == 0:
            continue
        classes, source = _load_classes(child)
        tasks.append(
            TrainingTask(
                name=child.name,
                path=child,
                image_count=image_count,
                class_count=len(classes),
                class_source=source,
            )
        )
    return tasks


def load_dataset(root: Path) -> Dataset:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset root does not exist: {root}")

    image_paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError("No supported images found below the selected folder.")

    samples: list[Sample] = []
    class_ids: set[int] = set()
    missing_labels = 0

    for image_path in image_paths:
        label_path: Optional[Path] = None
        for candidate in _candidate_label_paths(root, image_path):
            if candidate.is_file():
                label_path = candidate
                break

        if label_path is None:
            missing_labels += 1
            continue

        boxes = _parse_yolo_label(label_path)
        if not boxes:
            continue

        class_ids.update(box.class_id for box in boxes)
        samples.append(Sample(image_path=image_path, label_path=label_path, boxes=boxes))

    if not samples:
        raise ValueError(
            "Images were found, but no non-empty matching YOLO detection labels were found.\n\n"
            "Supported layouts include:\n"
            "  TASK/images/... + TASK/labels/...\n"
            "  or image.jpg + image.txt in the same folder."
        )

    classes, source = _load_classes(root)
    max_class_id = max(class_ids) if class_ids else -1

    if not classes:
        classes = [f"class_{i}" for i in range(max_class_id + 1)]
        source = "generated placeholders"
    elif max_class_id >= len(classes):
        for class_id in range(len(classes), max_class_id + 1):
            classes.append(f"class_{class_id}")
        source += " + generated placeholders"

    return Dataset(
        root=root,
        classes=classes,
        samples=samples,
        class_source=source,
        missing_labels=missing_labels,
    )


# ------------------------------- Statistics --------------------------------


class StatsDB:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def close(self) -> None:
        self.conn.close()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS trainees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trainee_id INTEGER NOT NULL,
                task_name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                score INTEGER NOT NULL DEFAULT 0,
                correct_answers INTEGER NOT NULL DEFAULT 0,
                mistakes INTEGER NOT NULL DEFAULT 0,
                completed INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (trainee_id) REFERENCES trainees(id)
            );

            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                trainee_id INTEGER NOT NULL,
                task_name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                target_class_id INTEGER NOT NULL,
                target_class_name TEXT NOT NULL,
                selected_class_id INTEGER NOT NULL,
                selected_class_name TEXT NOT NULL,
                correct INTEGER NOT NULL,
                attempted_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                FOREIGN KEY (trainee_id) REFERENCES trainees(id)
            );

            CREATE INDEX IF NOT EXISTS idx_attempts_worker_task
                ON attempts(trainee_id, task_name);
            CREATE INDEX IF NOT EXISTS idx_attempts_worker_task_class
                ON attempts(trainee_id, task_name, target_class_id);
            """
        )
        self.conn.commit()

    def ensure_default_trainee(self) -> None:
        if not self.trainees():
            self.add_trainee("Default Trainee")

    def trainees(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT id, name FROM trainees ORDER BY name COLLATE NOCASE"))

    def add_trainee(self, name: str) -> int:
        name = name.strip()
        if not name:
            raise ValueError("Worker name cannot be empty.")
        try:
            cursor = self.conn.execute(
                "INSERT INTO trainees(name, created_at) VALUES (?, ?)", (name, now_iso())
            )
            self.conn.commit()
            return int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise ValueError(f'A worker named "{name}" already exists.') from exc

    def start_session(self, trainee_id: int, task_name: str) -> int:
        cursor = self.conn.execute(
            "INSERT INTO sessions(trainee_id, task_name, started_at) VALUES (?, ?, ?)",
            (trainee_id, task_name, now_iso()),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def finish_session(
        self,
        session_id: int,
        score: int,
        correct_answers: int,
        mistakes: int,
        completed: bool,
    ) -> None:
        self.conn.execute(
            """
            UPDATE sessions
            SET ended_at=?, score=?, correct_answers=?, mistakes=?, completed=?
            WHERE id=?
            """,
            (now_iso(), score, correct_answers, mistakes, int(completed), session_id),
        )
        self.conn.commit()

    def record_attempt(
        self,
        *,
        session_id: int,
        trainee_id: int,
        task_name: str,
        image_path: str,
        target_class_id: int,
        target_class_name: str,
        selected_class_id: int,
        selected_class_name: str,
        correct: bool,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO attempts(
                session_id, trainee_id, task_name, image_path,
                target_class_id, target_class_name,
                selected_class_id, selected_class_name,
                correct, attempted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                trainee_id,
                task_name,
                image_path,
                target_class_id,
                target_class_name,
                selected_class_id,
                selected_class_name,
                int(correct),
                now_iso(),
            ),
        )
        self.conn.commit()

    def worker_task_summary(self, trainee_id: int, task_name: str) -> dict[str, object]:
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS attempts,
                   COALESCE(SUM(correct), 0) AS correct
            FROM attempts
            WHERE trainee_id=? AND task_name=?
            """,
            (trainee_id, task_name),
        ).fetchone()
        attempts = int(row["attempts"])
        correct = int(row["correct"])
        accuracy = (correct / attempts) if attempts else None
        return {"attempts": attempts, "correct": correct, "accuracy": accuracy}

    def class_stats(self, trainee_id: int, task_name: str) -> list[dict[str, object]]:
        rows = self.conn.execute(
            """
            SELECT target_class_id AS class_id,
                   MAX(target_class_name) AS class_name,
                   COUNT(*) AS attempts,
                   SUM(correct) AS correct
            FROM attempts
            WHERE trainee_id=? AND task_name=?
            GROUP BY target_class_id
            ORDER BY target_class_id
            """,
            (trainee_id, task_name),
        ).fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            attempts = int(row["attempts"])
            correct = int(row["correct"] or 0)
            result.append(
                {
                    "class_id": int(row["class_id"]),
                    "class_name": str(row["class_name"]),
                    "attempts": attempts,
                    "correct": correct,
                    "mistakes": attempts - correct,
                    "accuracy": (correct / attempts) if attempts else None,
                }
            )
        return result

    def class_difficulty(self, trainee_id: int, task_name: str) -> dict[int, float]:
        """Return per-class weights. More mistakes => larger future sampling weight."""
        weights: dict[int, float] = {}
        for row in self.class_stats(trainee_id, task_name):
            attempts = int(row["attempts"])
            accuracy = float(row["accuracy"] or 0.0)
            confidence = min(1.0, attempts / 5.0)
            weights[int(row["class_id"])] = 1.0 + (1.0 - accuracy) * 4.0 * confidence
        return weights


def make_stats_db(training_root: Path) -> StatsDB:
    """Prefer the training root; fall back to the user's home directory if read-only."""
    preferred = training_root / DB_FILENAME
    try:
        return StatsDB(preferred)
    except (OSError, sqlite3.Error):
        digest = hashlib.sha1(str(training_root).encode("utf-8")).hexdigest()[:10]
        fallback = Path.home() / ".yolo_annotation_trainer" / f"{training_root.name}_{digest}.sqlite3"
        return StatsDB(fallback)


# ------------------------------- Audio -------------------------------------


def _write_tone_sequence(path: Path, notes: list[tuple[float, float]], volume: float = 0.35) -> None:
    sample_rate = 44100
    frames: list[bytes] = []
    for frequency, duration in notes:
        count = max(1, int(sample_rate * duration))
        fade_samples = min(int(sample_rate * 0.012), count // 2)
        for i in range(count):
            envelope = 1.0
            if fade_samples:
                if i < fade_samples:
                    envelope = i / fade_samples
                elif i >= count - fade_samples:
                    envelope = (count - i - 1) / fade_samples
            sample = math.sin(2 * math.pi * frequency * i / sample_rate)
            value = int(32767 * volume * envelope * sample)
            frames.append(struct.pack("<h", value))

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(frames))


class SoundBank:
    def __init__(self, parent: QWidget) -> None:
        self._tempdir = tempfile.TemporaryDirectory(prefix="yolo_trainer_audio_")
        folder = Path(self._tempdir.name)
        success_path = folder / "success.wav"
        error_path = folder / "error.wav"
        _write_tone_sequence(success_path, [(659.25, 0.08), (783.99, 0.08), (1046.50, 0.16)])
        _write_tone_sequence(error_path, [(220.00, 0.10), (164.81, 0.18)], volume=0.25)

        self.success = QSoundEffect(parent)
        self.success.setSource(QUrl.fromLocalFile(str(success_path)))
        self.success.setVolume(0.55)
        self.error = QSoundEffect(parent)
        self.error.setSource(QUrl.fromLocalFile(str(error_path)))
        self.error.setVolume(0.45)

    def play_success(self) -> None:
        self.success.stop()
        self.success.play()

    def play_error(self) -> None:
        self.error.stop()
        self.error.play()


# ------------------------------- UI helpers --------------------------------


class CelebrationOverlay(QWidget):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.hide()
        self._tick = 0
        self._particles: list[tuple[float, float, float, float, int]] = []
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._advance)

    def celebrate(self) -> None:
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.show()
        self._tick = 0
        self._particles = []
        w = max(1, self.width())
        h = max(1, self.height())
        for _ in range(70):
            self._particles.append(
                (
                    random.uniform(w * 0.25, w * 0.75),
                    random.uniform(h * 0.25, h * 0.55),
                    random.uniform(-5.0, 5.0),
                    random.uniform(-11.0, -3.0),
                    random.randint(4, 11),
                )
            )
        self._timer.start()

    def _advance(self) -> None:
        self._tick += 1
        updated = []
        for x, y, vx, vy, size in self._particles:
            updated.append((x + vx, y + vy, vx, vy + 0.45, size))
        self._particles = updated
        self.update()
        if self._tick > 52:
            self._timer.stop()
            self.hide()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = [
            QColor("#ffd166"),
            QColor("#06d6a0"),
            QColor("#4cc9f0"),
            QColor("#f72585"),
            QColor("#b8f2e6"),
        ]
        for i, (x, y, _vx, _vy, size) in enumerate(self._particles):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(palette[i % len(palette)])
            painter.drawRoundedRect(QRectF(x, y, size, size * 1.7), 2, 2)

        if self._tick < 38:
            alpha = 255 if self._tick < 22 else max(0, 255 - (self._tick - 22) * 16)
            painter.setPen(QColor(255, 255, 255, alpha))
            painter.setFont(QFont("Arial", 28, QFont.Weight.Bold))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "✓  CORRECT!")


class ImageCanvas(QWidget):
    object_clicked = Signal(int)
    empty_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.sample: Optional[Sample] = None
        self.pixmap = QPixmap()
        self.display_rect = QRectF()
        self.show_targets = False
        self.class_names: list[str] = []
        self.feedback_box: Optional[int] = None
        self.feedback_correct: Optional[bool] = None
        self.locked_box: Optional[int] = None
        self.celebration = CelebrationOverlay(self)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.celebration.setGeometry(self.rect())

    def set_class_names(self, class_names: list[str]) -> None:
        self.class_names = list(class_names)
        self.update()

    def set_sample(self, sample: Sample) -> None:
        self.sample = sample
        self.pixmap = QPixmap(str(sample.image_path))
        self.feedback_box = None
        self.feedback_correct = None
        self.locked_box = None
        self.update()

    def clear_sample(self) -> None:
        self.sample = None
        self.pixmap = QPixmap()
        self.feedback_box = None
        self.feedback_correct = None
        self.locked_box = None
        self.update()

    def set_feedback(self, box_index: Optional[int], correct: Optional[bool]) -> None:
        self.feedback_box = box_index
        self.feedback_correct = correct
        self.update()

    def set_locked_box(self, box_index: Optional[int]) -> None:
        self.locked_box = box_index
        self.update()

    def _box_rect(self, box: YoloBox) -> QRectF:
        r = self.display_rect
        left = r.left() + (box.cx - box.width / 2) * r.width()
        top = r.top() + (box.cy - box.height / 2) * r.height()
        return QRectF(left, top, box.width * r.width(), box.height * r.height())

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0c1017"))

        if self.pixmap.isNull():
            painter.setPen(QColor("#8b96a8"))
            painter.setFont(QFont("Arial", 18))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Choose a training task to begin")
            self.display_rect = QRectF()
            return

        scaled = self.pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) / 2
        y = (self.height() - scaled.height()) / 2
        self.display_rect = QRectF(x, y, scaled.width(), scaled.height())
        painter.drawPixmap(int(x), int(y), scaled)

        if not self.sample:
            return

        if self.show_targets:
            # Answer/review mode: draw every ground-truth YOLO box and its true class name.
            for box in self.sample.boxes:
                rect = self._box_rect(box)
                color = QColor("#ffd166")
                painter.setPen(QPen(color, 3, Qt.PenStyle.SolidLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(rect)

                if 0 <= box.class_id < len(self.class_names):
                    class_name = self.class_names[box.class_id]
                else:
                    class_name = f"class_{box.class_id}"
                label = f"{box.class_id}: {class_name}"

                painter.setFont(QFont("Arial", 11, QFont.Weight.Bold))
                metrics = painter.fontMetrics()
                text_rect = metrics.boundingRect(label)
                pad_x, pad_y = 7, 4
                label_w = text_rect.width() + pad_x * 2
                label_h = text_rect.height() + pad_y * 2
                label_x = rect.left()
                label_y = max(self.display_rect.top(), rect.top() - label_h)
                if label_x + label_w > self.display_rect.right():
                    label_x = max(self.display_rect.left(), self.display_rect.right() - label_w)
                label_rect = QRectF(label_x, label_y, label_w, label_h)

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(12, 16, 23, 225))
                painter.drawRoundedRect(label_rect, 5, 5)
                painter.setPen(color)
                painter.drawText(
                    label_rect.adjusted(pad_x, 0, -pad_x, 0),
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                    label,
                )

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(46, 204, 113, 210), 3))
        for box in self.sample.boxes:
            if box.solved:
                painter.drawRect(self._box_rect(box))

        if self.locked_box is not None and 0 <= self.locked_box < len(self.sample.boxes):
            painter.setPen(QPen(QColor("#ff4d6d"), 4))
            painter.drawRect(self._box_rect(self.sample.boxes[self.locked_box]))

        if self.feedback_box is not None and 0 <= self.feedback_box < len(self.sample.boxes):
            color = QColor("#33e38e") if self.feedback_correct else QColor("#ff4d6d")
            painter.setPen(QPen(color, 5))
            painter.drawRect(self._box_rect(self.sample.boxes[self.feedback_box]))

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or not self.sample or self.display_rect.isNull():
            return

        pos = event.position()
        if not self.display_rect.contains(pos):
            self.empty_clicked.emit()
            return

        nx = (pos.x() - self.display_rect.left()) / self.display_rect.width()
        ny = (pos.y() - self.display_rect.top()) / self.display_rect.height()

        if self.locked_box is not None:
            box = self.sample.boxes[self.locked_box]
            if box.contains(nx, ny):
                self.object_clicked.emit(self.locked_box)
            else:
                self.empty_clicked.emit()
            return

        candidates = [
            (index, box)
            for index, box in enumerate(self.sample.boxes)
            if not box.solved and box.contains(nx, ny)
        ]
        if not candidates:
            self.empty_clicked.emit()
            return

        index, _ = min(candidates, key=lambda item: item[1].area)
        self.object_clicked.emit(index)


class ClassesDialog(QDialog):
    def __init__(self, classes: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit class names")
        self.resize(520, 520)
        layout = QVBoxLayout(self)
        info = QLabel("One class name per line. Line 1 = class ID 0, line 2 = class ID 1, etc.")
        info.setWordWrap(True)
        layout.addWidget(info)
        self.editor = QPlainTextEdit()
        self.editor.setPlainText("\n".join(classes))
        self.editor.setPlaceholderText("person\ncar\nforklift\npallet")
        layout.addWidget(self.editor, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def class_names(self) -> list[str]:
        return [line.strip() for line in self.editor.toPlainText().splitlines() if line.strip()]


class StatsDialog(QDialog):
    def __init__(
        self,
        worker_name: str,
        task_name: str,
        rows: list[dict[str, object]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Statistics — {worker_name}")
        self.resize(720, 520)
        layout = QVBoxLayout(self)

        heading = QLabel(f"{worker_name}  •  {task_name}")
        heading.setObjectName("dialogHeading")
        layout.addWidget(heading)

        if not rows:
            empty = QLabel("No recorded attempts for this worker and task yet.")
            empty.setObjectName("muted")
            layout.addWidget(empty)
        else:
            table = QTableWidget(len(rows), 5)
            table.setHorizontalHeaderLabels(["Class", "Attempts", "Correct", "Mistakes", "Accuracy"])
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            for col in range(1, 5):
                table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)

            for row_index, row in enumerate(rows):
                accuracy = row["accuracy"]
                values = [
                    f'{row["class_id"]}: {row["class_name"]}',
                    str(row["attempts"]),
                    str(row["correct"]),
                    str(row["mistakes"]),
                    f"{100 * float(accuracy):.1f}%" if accuracy is not None else "—",
                ]
                for column, value in enumerate(values):
                    table.setItem(row_index, column, QTableWidgetItem(value))
            layout.addWidget(table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class StatCard(QFrame):
    def __init__(self, title: str, value: str = "0") -> None:
        super().__init__()
        self.setObjectName("statCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("statTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("statValue")
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


# ------------------------------- Main window --------------------------------


class MainWindow(QMainWindow):
    def __init__(self, training_root: Path) -> None:
        super().__init__()
        self.setWindowTitle(f"YOLO Annotation Trainer — Category Menu — {APP_VERSION}")
        self.resize(1450, 920)

        self.training_root = training_root.expanduser().resolve()
        self.db: Optional[StatsDB] = None
        self.tasks: list[TrainingTask] = []
        self.dataset: Optional[Dataset] = None
        self.current_task: Optional[TrainingTask] = None
        self.trainee_id: Optional[int] = None
        self.trainee_name = ""
        self.session_id: Optional[int] = None
        self.session_image_limit = 0
        self.focus_weak_classes = True
        self.order: list[int] = []
        self.order_pos = 0
        self.session_total_objects = 0
        self.score = 0
        self.streak = 0
        self.correct_answers = 0
        self.mistakes = 0
        self.solved_objects = 0
        self.skipped_images = 0

        self.sound = SoundBank(self)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self._build_menu_page()
        self._build_training_page()
        self._install_training_shortcuts()
        self._apply_style()
        self.set_training_root(self.training_root, show_error=False)

    # --------------------------- Menu --------------------------------------

    def _build_menu_page(self) -> None:
        self.menu_page = QWidget()
        outer = QVBoxLayout(self.menu_page)
        outer.setContentsMargins(44, 34, 44, 34)
        outer.setSpacing(18)

        header = QHBoxLayout()
        header.setSpacing(18)
        heading = QVBoxLayout()
        heading.setSpacing(2)
        title = QLabel("ANNOTATION TRAINING")
        title.setObjectName("menuTitle")
        subtitle = QLabel("Choose a training category. Tasks are discovered automatically from TRAINING_DATA.")
        subtitle.setObjectName("subtitle")
        heading.addWidget(title)
        heading.addWidget(subtitle)
        header.addLayout(heading, 1)

        root_tools = QVBoxLayout()
        root_tools.setSpacing(6)
        root_caption = QLabel("TRAINING_DATA ROOT")
        root_caption.setObjectName("sectionTitle")
        root_caption.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.root_path_label = QLabel()
        self.root_path_label.setObjectName("rootBadge")
        self.root_path_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.root_path_label.setToolTip("Every immediate subfolder is shown as a training category.")
        root_tools.addWidget(root_caption)
        root_tools.addWidget(self.root_path_label)

        build_label = QLabel(f"BUILD  {APP_VERSION}")
        build_label.setObjectName("rootBadge")
        build_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        root_tools.addWidget(build_label)

        self.db_path_label = QLabel("SQLite: initializing…")
        self.db_path_label.setObjectName("rootBadge")
        self.db_path_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.db_path_label.setWordWrap(True)
        root_tools.addWidget(self.db_path_label)
        header.addLayout(root_tools)
        outer.addLayout(header)

        toolbar = QHBoxLayout()
        toolbar.addStretch(1)
        rescan = QPushButton("↻  Rescan TRAINING_DATA")
        rescan.setObjectName("secondaryButton")
        rescan.clicked.connect(self.scan_tasks)
        toolbar.addWidget(rescan)
        root_hint = QLabel("Categories are loaded only from the TRAINING_DATA folder shown above.")
        root_hint.setObjectName("muted")
        toolbar.addWidget(root_hint)
        outer.addLayout(toolbar)

        content = QHBoxLayout()
        content.setSpacing(18)

        task_card = QFrame()
        task_card.setObjectName("menuCard")
        task_layout = QVBoxLayout(task_card)
        task_layout.setContentsMargins(20, 20, 20, 20)
        task_layout.setSpacing(12)

        task_heading_row = QHBoxLayout()
        task_heading = QLabel("SELECT TRAINING CATEGORY")
        task_heading.setObjectName("sectionTitle")
        task_heading_row.addWidget(task_heading)
        task_heading_row.addStretch(1)
        self.task_count_label = QLabel("0 categories")
        self.task_count_label.setObjectName("muted")
        task_heading_row.addWidget(self.task_count_label)
        task_layout.addLayout(task_heading_row)

        self.task_scroll = QScrollArea()
        self.task_scroll.setWidgetResizable(True)
        self.task_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.task_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.task_scroll.setObjectName("taskScroll")
        self.task_grid_widget = QWidget()
        self.task_grid_widget.setObjectName("taskGridWidget")
        self.task_grid = QGridLayout(self.task_grid_widget)
        self.task_grid.setContentsMargins(2, 2, 8, 2)
        self.task_grid.setHorizontalSpacing(12)
        self.task_grid.setVerticalSpacing(12)
        self.task_grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.task_scroll.setWidget(self.task_grid_widget)
        task_layout.addWidget(self.task_scroll, 1)

        self.task_detail_label = QLabel("Select a category to begin.")
        self.task_detail_label.setObjectName("taskDetail")
        self.task_detail_label.setWordWrap(True)
        task_layout.addWidget(self.task_detail_label)
        content.addWidget(task_card, 7)

        setup_card = QFrame()
        setup_card.setObjectName("menuCard")
        setup_card.setMinimumWidth(320)
        setup_layout = QVBoxLayout(setup_card)
        setup_layout.setContentsMargins(20, 20, 20, 20)
        setup_layout.setSpacing(10)

        worker_heading = QLabel("WORKER")
        worker_heading.setObjectName("sectionTitle")
        setup_layout.addWidget(worker_heading)
        worker_row = QHBoxLayout()
        self.worker_combo = QComboBox()
        self.worker_combo.currentIndexChanged.connect(self._menu_selection_changed)
        worker_row.addWidget(self.worker_combo, 1)
        add_worker = QPushButton("+")
        add_worker.setFixedWidth(42)
        add_worker.setToolTip("Add worker profile")
        add_worker.clicked.connect(self.add_worker)
        worker_row.addWidget(add_worker)
        setup_layout.addLayout(worker_row)

        images_heading = QLabel("IMAGES PER SESSION")
        images_heading.setObjectName("sectionTitle")
        setup_layout.addWidget(images_heading)
        self.session_limit_spin = QSpinBox()
        self.session_limit_spin.setRange(0, 100000)
        self.session_limit_spin.setValue(0)
        self.session_limit_spin.setSpecialValueText("All images")
        self.session_limit_spin.setSuffix(" images")
        setup_layout.addWidget(self.session_limit_spin)

        self.focus_checkbox = QCheckBox("Prioritize classes this worker struggles with")
        self.focus_checkbox.setChecked(True)
        self.focus_checkbox.toggled.connect(self._update_worker_summary)
        setup_layout.addWidget(self.focus_checkbox)

        summary_heading = QLabel("PREVIOUS PERFORMANCE")
        summary_heading.setObjectName("sectionTitle")
        setup_layout.addWidget(summary_heading)
        self.worker_summary_label = QLabel("No previous attempts yet.")
        self.worker_summary_label.setObjectName("muted")
        self.worker_summary_label.setWordWrap(True)
        setup_layout.addWidget(self.worker_summary_label)

        self.menu_stats_button = QPushButton("View class statistics")
        self.menu_stats_button.setObjectName("secondaryButton")
        self.menu_stats_button.clicked.connect(self.show_stats)
        setup_layout.addWidget(self.menu_stats_button)

        setup_layout.addStretch(1)
        self.start_button = QPushButton("Select a category")
        self.start_button.setObjectName("startTrainingButton")
        self.start_button.setMinimumHeight(56)
        self.start_button.clicked.connect(self.start_selected_task)
        self.start_button.setEnabled(False)
        setup_layout.addWidget(self.start_button)
        content.addWidget(setup_card, 3)

        outer.addLayout(content, 1)
        self.stack.addWidget(self.menu_page)

        self.selected_task_index = -1
        self.task_buttons: list[QPushButton] = []
        self.task_button_group = QButtonGroup(self)
        self.task_button_group.setExclusive(True)

    def set_training_root(self, root: Path, show_error: bool = True) -> None:
        root = root.expanduser().resolve()

        # On normal startup we never force a native folder chooser.  If the
        # default TRAINING_DATA directory does not exist yet, create it when
        # possible and show the empty in-app category launcher instead.
        if not root.exists():
            try:
                root.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.training_root = root
                self.root_path_label.setText(str(root))
                self.tasks = []
                self.selected_task_index = -1
                self._clear_task_grid()
                self.task_count_label.setText("0 categories")
                self.start_button.setEnabled(False)
                self.start_button.setText("Select a category")
                self.task_detail_label.setText(
                    "TRAINING_DATA could not be created at this location. "
                    "Create or fix the TRAINING_DATA folder next to this application, then press Rescan."
                )
                if show_error:
                    QMessageBox.warning(self, "Training root unavailable", str(root))
                return

        if not root.is_dir():
            if show_error:
                QMessageBox.warning(self, "Invalid training root", f"Not a directory:\n{root}")
            return

        if self.db is not None:
            self.db.close()
        self.training_root = root
        self.db = make_stats_db(root)
        self.db.ensure_default_trainee()
        self.root_path_label.setText(str(root))
        self.db_path_label.setText(f"SQLite: {self.db.path}")
        self._reload_workers()
        self.scan_tasks()

    def _clear_task_grid(self) -> None:
        while self.task_grid.count():
            item = self.task_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                if isinstance(widget, QPushButton):
                    self.task_button_group.removeButton(widget)
                widget.deleteLater()
        self.task_buttons = []

    def scan_tasks(self) -> None:
        self.tasks = discover_tasks(self.training_root)
        self.selected_task_index = -1
        self._clear_task_grid()
        self.task_count_label.setText(
            f"{len(self.tasks)} categor{'y' if len(self.tasks) == 1 else 'ies'}"
        )

        for index, task in enumerate(self.tasks):
            display_name = task.name.replace("_", " ")
            classes_text = f"{task.class_count} classes" if task.class_count else "class names not found"
            source_text = task.class_source if task.class_source != "not found" else "Add classes.txt"
            button = QPushButton(
                f"{display_name}\n\n{task.image_count} images   •   {classes_text}\n{source_text}"
            )
            button.setObjectName("taskCardButton")
            button.setCheckable(True)
            button.setMinimumHeight(128)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(str(task.path))
            button.clicked.connect(
                lambda checked=False, task_index=index: self._select_task(task_index)
            )
            self.task_button_group.addButton(button)
            self.task_buttons.append(button)
            self.task_grid.addWidget(button, index // 2, index % 2)

        if self.tasks:
            self.task_buttons[0].setChecked(True)
            self._select_task(0)
        else:
            empty = QLabel(
                "No training categories found.\n\n"
                "Create folders directly inside TRAINING_DATA, for example:\n"
                "  FOOD_PACKAGING/\n"
                "  SHREDDER_SCRAP/\n\n"
                "Each category can contain classes.txt plus images/ and labels/."
            )
            empty.setObjectName("emptyTasks")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setWordWrap(True)
            empty.setMinimumHeight(260)
            self.task_grid.addWidget(empty, 0, 0, 1, 2)
            self.task_detail_label.setText(
                f"Watching: {self.training_root}\nAdd category folders there, then press Rescan TRAINING_DATA."
            )
            self.start_button.setEnabled(False)
            self.start_button.setText("Select a category")
            self.menu_stats_button.setEnabled(False)
            self._update_worker_summary()

    def _select_task(self, index: int) -> None:
        if not (0 <= index < len(self.tasks)):
            self.selected_task_index = -1
            self._menu_selection_changed()
            return
        self.selected_task_index = index
        if index < len(self.task_buttons) and not self.task_buttons[index].isChecked():
            self.task_buttons[index].setChecked(True)
        self._menu_selection_changed()

    def _reload_workers(self, preferred_id: Optional[int] = None) -> None:
        self.worker_combo.blockSignals(True)
        self.worker_combo.clear()
        if self.db:
            for row in self.db.trainees():
                self.worker_combo.addItem(str(row["name"]), int(row["id"]))
        self.worker_combo.blockSignals(False)

        if preferred_id is not None:
            for i in range(self.worker_combo.count()):
                if self.worker_combo.itemData(i) == preferred_id:
                    self.worker_combo.setCurrentIndex(i)
                    break
        self._menu_selection_changed()

    def add_worker(self) -> None:
        if not self.db:
            return
        name, ok = QInputDialog.getText(self, "Add worker", "Worker name:")
        if not ok:
            return
        try:
            worker_id = self.db.add_trainee(name)
        except ValueError as exc:
            QMessageBox.warning(self, "Could not add worker", str(exc))
            return
        self._reload_workers(preferred_id=worker_id)

    def _selected_task(self) -> Optional[TrainingTask]:
        if 0 <= self.selected_task_index < len(self.tasks):
            return self.tasks[self.selected_task_index]
        return None

    def _menu_selection_changed(self, *_args) -> None:
        task = self._selected_task()
        has_worker = self.worker_combo.currentIndex() >= 0
        ready = task is not None and has_worker
        self.start_button.setEnabled(ready)
        self.menu_stats_button.setEnabled(ready)
        if task:
            display_name = task.name.replace("_", " ")
            self.start_button.setText(f"Start {display_name}")
            self.task_detail_label.setText(
                f"Selected: {display_name}  •  {task.image_count} images  •  "
                f"{task.class_count or 'unknown'} classes  •  {task.class_source}"
            )
        else:
            self.start_button.setText("Select a category")
        self._update_worker_summary()

    def _update_worker_summary(self) -> None:
        if not self.db:
            return
        task = self._selected_task()
        trainee_id = self.worker_combo.currentData()
        if not task or trainee_id is None:
            self.worker_summary_label.setText("Choose a worker and task.")
            return

        summary = self.db.worker_task_summary(int(trainee_id), task.name)
        attempts = int(summary["attempts"])
        if not attempts:
            self.worker_summary_label.setText("No previous attempts yet. Random image selection will be used.")
            return

        accuracy = 100.0 * float(summary["accuracy"])
        class_rows = self.db.class_stats(int(trainee_id), task.name)
        weak = sorted(
            (r for r in class_rows if int(r["attempts"]) > 0),
            key=lambda r: (float(r["accuracy"]), -int(r["attempts"])),
        )[:3]
        weak_text = ", ".join(
            f'{r["class_name"]} {100 * float(r["accuracy"]):.0f}%' for r in weak
        )
        focus_text = " Weak-class prioritization is ON." if self.focus_checkbox.isChecked() else ""
        self.worker_summary_label.setText(
            f"{attempts} attempts • {accuracy:.1f}% accuracy"
            + (f"\nNeeds most practice: {weak_text}" if weak_text else "")
            + focus_text
        )

    def start_selected_task(self) -> None:
        task = self._selected_task()
        trainee_id = self.worker_combo.currentData()
        if not task or trainee_id is None or not self.db:
            return

        try:
            dataset = load_dataset(task.path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load training task", str(exc))
            return

        self.current_task = task
        self.dataset = dataset
        self.trainee_id = int(trainee_id)
        self.trainee_name = self.worker_combo.currentText()
        self.session_image_limit = int(self.session_limit_spin.value())
        self.focus_weak_classes = self.focus_checkbox.isChecked()

        self.class_combo.clear()
        self.class_combo.addItems(dataset.classes)
        self.canvas.set_class_names(dataset.classes)
        self.class_combo.setEnabled(True)
        self.edit_classes_button.setEnabled(True)
        self.training_title.setText(task.name)
        self.worker_label.setText(self.trainee_name)
        self._update_dataset_label()
        self.stack.setCurrentWidget(self.training_page)
        self.restart_session()

    # --------------------------- Training page -----------------------------

    def _build_training_page(self) -> None:
        self.training_page = QWidget()
        root_layout = QHBoxLayout(self.training_page)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(14)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(325)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(18, 18, 18, 18)
        side.setSpacing(12)

        top_row = QHBoxLayout()
        back = QPushButton("← Menu")
        back.setObjectName("secondaryButton")
        back.clicked.connect(self.back_to_menu)
        top_row.addWidget(back)
        top_row.addStretch(1)
        side.addLayout(top_row)

        self.training_title = QLabel("TRAINING TASK")
        self.training_title.setObjectName("title")
        self.training_title.setWordWrap(True)
        side.addWidget(self.training_title)
        self.worker_label = QLabel("")
        self.worker_label.setObjectName("workerBadge")
        side.addWidget(self.worker_label)

        self.dataset_label = QLabel("No dataset loaded")
        self.dataset_label.setObjectName("muted")
        self.dataset_label.setWordWrap(True)
        side.addWidget(self.dataset_label)

        side.addSpacing(8)
        class_title = QLabel("ACTIVE CLASS")
        class_title.setObjectName("sectionTitle")
        side.addWidget(class_title)
        self.class_combo = QComboBox()
        self.class_combo.setEnabled(False)
        side.addWidget(self.class_combo)
        self.edit_classes_button = QPushButton("Edit class names")
        self.edit_classes_button.setObjectName("secondaryButton")
        self.edit_classes_button.setEnabled(False)
        self.edit_classes_button.clicked.connect(self.edit_classes)
        side.addWidget(self.edit_classes_button)

        self.show_boxes_button = QPushButton("👁  Show correct boxes")
        self.show_boxes_button.setObjectName("answerRevealButton")
        self.show_boxes_button.setCheckable(True)
        self.show_boxes_button.setToolTip(
            "Reveal every ground-truth bounding box and its correct class name. "
            "Press H to toggle this answer view."
        )
        self.show_boxes_button.toggled.connect(self._toggle_target_boxes)
        side.addWidget(self.show_boxes_button)

        reveal_hint = QLabel("Answer aid: reveals the true boxes + class names (H)")
        reveal_hint.setObjectName("muted")
        reveal_hint.setWordWrap(True)
        side.addWidget(reveal_hint)

        self.skip_button = QPushButton("⏭  Skip image — revisit later")
        self.skip_button.setObjectName("skipImageButton")
        self.skip_button.setToolTip(
            "Move the current image to the end of this session without recording "
            "a correct or incorrect answer. Press S to skip."
        )
        self.skip_button.clicked.connect(self.skip_current_image)
        side.addWidget(self.skip_button)

        skip_hint = QLabel("Skip does not count as a mistake; the image returns later (S)")
        skip_hint.setObjectName("muted")
        skip_hint.setWordWrap(True)
        side.addWidget(skip_hint)

        side.addSpacing(10)
        stats_title = QLabel("SESSION")
        stats_title.setObjectName("sectionTitle")
        side.addWidget(stats_title)
        row1 = QHBoxLayout()
        self.score_card = StatCard("Score")
        self.streak_card = StatCard("Streak")
        row1.addWidget(self.score_card)
        row1.addWidget(self.streak_card)
        side.addLayout(row1)
        row2 = QHBoxLayout()
        self.correct_card = StatCard("Correct")
        self.accuracy_card = StatCard("Accuracy", "—")
        row2.addWidget(self.correct_card)
        row2.addWidget(self.accuracy_card)
        side.addLayout(row2)

        self.progress_label = QLabel("Progress: —")
        self.progress_label.setObjectName("muted")
        side.addWidget(self.progress_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        side.addWidget(self.progress)

        side.addStretch(1)
        hotkeys = QLabel("Hotkeys: 1–9 select classes • H shows answers • S skips/requeues image.")
        hotkeys.setObjectName("muted")
        hotkeys.setWordWrap(True)
        side.addWidget(hotkeys)

        stats_button = QPushButton("Worker class statistics")
        stats_button.setObjectName("secondaryButton")
        stats_button.clicked.connect(self.show_stats)
        side.addWidget(stats_button)
        restart = QPushButton("Restart / reshuffle")
        restart.setObjectName("secondaryButton")
        restart.clicked.connect(self.restart_session)
        side.addWidget(restart)
        root_layout.addWidget(sidebar)

        main = QFrame()
        main.setObjectName("mainPanel")
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(10)
        topbar = QHBoxLayout()
        self.image_label = QLabel("Image —")
        self.image_label.setObjectName("imageCounter")
        topbar.addWidget(self.image_label)
        topbar.addStretch(1)
        self.object_label = QLabel("Objects remaining: —")
        self.object_label.setObjectName("muted")
        topbar.addWidget(self.object_label)
        main_layout.addLayout(topbar)

        self.canvas = ImageCanvas()
        self.canvas.object_clicked.connect(self.grade_object)
        self.canvas.empty_clicked.connect(self.handle_empty_click)
        main_layout.addWidget(self.canvas, 1)

        self.feedback = QLabel("Choose a training task from the menu.")
        self.feedback.setObjectName("feedback")
        self.feedback.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feedback.setWordWrap(True)
        main_layout.addWidget(self.feedback)
        root_layout.addWidget(main, 1)
        self.stack.addWidget(self.training_page)

    def _weighted_order(self, indices: list[int], limit: int) -> list[int]:
        if not self.dataset or not self.db or self.trainee_id is None or not self.current_task:
            random.shuffle(indices)
            return indices[:limit] if limit else indices

        class_weights = self.db.class_difficulty(self.trainee_id, self.current_task.name)
        weighted: list[tuple[int, float]] = []
        for index in indices:
            sample = self.dataset.samples[index]
            sample_weight = max((class_weights.get(box.class_id, 1.0) for box in sample.boxes), default=1.0)
            # Efraimidis-Spirakis weighted random ordering without replacement.
            key = random.random() ** (1.0 / max(0.001, sample_weight))
            weighted.append((index, key))
        weighted.sort(key=lambda item: item[1], reverse=True)
        ordered = [index for index, _key in weighted]
        return ordered[:limit] if limit else ordered

    def restart_session(self) -> None:
        if not self.dataset or not self.db or self.trainee_id is None or not self.current_task:
            return

        self._finalize_active_session(completed=False)
        for sample in self.dataset.samples:
            for box in sample.boxes:
                box.solved = False

        indices = list(range(len(self.dataset.samples)))
        limit = self.session_image_limit
        if limit <= 0 or limit >= len(indices):
            limit = 0

        if self.focus_weak_classes:
            self.order = self._weighted_order(indices, limit)
        else:
            random.shuffle(indices)
            self.order = indices[:limit] if limit else indices

        self.order_pos = 0
        self.session_total_objects = sum(len(self.dataset.samples[i].boxes) for i in self.order)
        self.score = 0
        self.streak = 0
        self.correct_answers = 0
        self.mistakes = 0
        self.solved_objects = 0
        self.skipped_images = 0
        self.session_id = self.db.start_session(self.trainee_id, self.current_task.name)
        # Start every new session in quiz mode; answers stay hidden until explicitly revealed.
        if hasattr(self, "show_boxes_button"):
            self.show_boxes_button.setChecked(False)
        self._update_stats()
        self._show_current_sample()

    def _finalize_active_session(self, completed: bool) -> None:
        if self.db is not None and self.session_id is not None:
            self.db.finish_session(
                self.session_id,
                self.score,
                self.correct_answers,
                self.mistakes,
                completed,
            )
            self.session_id = None

    def back_to_menu(self) -> None:
        self._finalize_active_session(completed=False)
        self.canvas.clear_sample()
        self.stack.setCurrentWidget(self.menu_page)
        self._update_worker_summary()

    def _show_current_sample(self) -> None:
        if not self.dataset or not self.order:
            return
        if self.order_pos >= len(self.order):
            self._finish_session()
            return

        sample_index = self.order[self.order_pos]
        sample = self.dataset.samples[sample_index]
        self.canvas.set_sample(sample)
        self.image_label.setText(
            f"Image {self.order_pos + 1} / {len(self.order)}   •   {sample.image_path.name}"
        )
        self.feedback.setText("Choose the class, then click an object.")
        self.feedback.setStyleSheet("")
        self._update_remaining()

    def _toggle_target_boxes(self, checked: bool) -> None:
        self.canvas.show_targets = checked
        if hasattr(self, "show_boxes_button"):
            self.show_boxes_button.setText(
                "🙈  Hide correct boxes" if checked else "👁  Show correct boxes"
            )
        self.canvas.update()

    def _current_sample(self) -> Optional[Sample]:
        if not self.dataset or not self.order or self.order_pos >= len(self.order):
            return None
        return self.dataset.samples[self.order[self.order_pos]]

    def grade_object(self, box_index: int) -> None:
        sample = self._current_sample()
        if sample is None or not self.dataset or not self.current_task:
            return
        if self.class_combo.currentIndex() < 0:
            self.feedback.setText("Select a class first.")
            return

        box = sample.boxes[box_index]
        selected_class = self.class_combo.currentIndex()
        selected_name = self.class_combo.currentText()
        target_name = (
            self.dataset.classes[box.class_id]
            if 0 <= box.class_id < len(self.dataset.classes)
            else f"class_{box.class_id}"
        )
        correct = selected_class == box.class_id

        if self.db and self.session_id is not None and self.trainee_id is not None:
            try:
                relative_image = str(sample.image_path.relative_to(self.dataset.root))
            except ValueError:
                relative_image = str(sample.image_path)
            self.db.record_attempt(
                session_id=self.session_id,
                trainee_id=self.trainee_id,
                task_name=self.current_task.name,
                image_path=relative_image,
                target_class_id=box.class_id,
                target_class_name=target_name,
                selected_class_id=selected_class,
                selected_class_name=selected_name,
                correct=correct,
            )

        if correct:
            self.streak += 1
            bonus = min(20, max(0, self.streak - 1) * 2)
            gained = 10 + bonus
            self.score += gained
            self.correct_answers += 1
            self.solved_objects += 1
            box.solved = True
            self.canvas.set_feedback(box_index, True)
            self.canvas.set_locked_box(None)
            self.canvas.celebration.celebrate()
            self.sound.play_success()
            self.feedback.setText(f"Correct — {target_name}!  +{gained} points")
            self.feedback.setStyleSheet(
                "background:#10251b; border:1px solid #246b46; color:#7ff0b2; "
                "border-radius:10px; padding:10px; font-weight:700;"
            )
            self._update_stats()
            self._update_remaining()
            if all(item.solved for item in sample.boxes):
                QTimer.singleShot(800, self._advance_image)
            else:
                QTimer.singleShot(550, lambda: self.canvas.set_feedback(None, None))
        else:
            self.streak = 0
            self.mistakes += 1
            self.canvas.set_feedback(box_index, False)
            self.canvas.set_locked_box(box_index)
            self.sound.play_error()
            self.feedback.setText(
                f"Not {selected_name}. Change the active class and click the red object again."
            )
            self.feedback.setStyleSheet(
                "background:#2a1117; border:1px solid #7d2a3c; color:#ff96a8; "
                "border-radius:10px; padding:10px; font-weight:700;"
            )
            self._update_stats()

    def handle_empty_click(self) -> None:
        if self.canvas.locked_box is not None:
            self.feedback.setText("Finish the red object first: select another class and click it again.")
        else:
            self.feedback.setText("That click is not inside an unsolved annotated object.")

    def skip_current_image(self) -> None:
        """Defer the current image by moving it to the end of the session queue.

        Skipping is deliberately neutral: it does not write an attempt to SQLite,
        does not change the streak, and does not count as a mistake. Any objects
        already solved on this image remain solved when the image comes back.
        """
        sample = self._current_sample()
        if sample is None or not self.order:
            return

        # A fully solved image already has an automatic advance queued. Avoid a
        # second navigation action if the trainee hits S during the reward delay.
        if sample.boxes and all(box.solved for box in sample.boxes):
            self.feedback.setText("This image is already complete — advancing automatically.")
            return

        current_index = self.order.pop(self.order_pos)
        self.order.append(current_index)
        self.skipped_images += 1

        skipped_name = sample.image_path.name
        only_remaining = self.order_pos == len(self.order) - 1

        # set_sample() clears any red locked-box/error feedback on the deferred
        # image. The underlying solved flags are intentionally retained.
        self._show_current_sample()

        if only_remaining:
            self.feedback.setText(
                f"Skipped {skipped_name}, but it is the only remaining image, so it stays here. "
                f"Deferred skips this session: {self.skipped_images}."
            )
        else:
            self.feedback.setText(
                f"Skipped {skipped_name} — moved to the end of the session. "
                f"Deferred skips: {self.skipped_images}."
            )
        self.feedback.setStyleSheet(
            "background:#241d0d; border:1px solid #7b6425; color:#ffd978; "
            "border-radius:10px; padding:10px; font-weight:700;"
        )

    def _advance_image(self) -> None:
        self.order_pos += 1
        self._show_current_sample()

    def _update_remaining(self) -> None:
        sample = self._current_sample()
        if sample is None:
            self.object_label.setText("Objects remaining: —")
            return
        remaining = sum(not box.solved for box in sample.boxes)
        self.object_label.setText(f"Objects remaining: {remaining} / {len(sample.boxes)}")

    def _update_stats(self) -> None:
        self.score_card.set_value(str(self.score))
        self.streak_card.set_value(str(self.streak))
        self.correct_card.set_value(str(self.correct_answers))
        attempts = self.correct_answers + self.mistakes
        accuracy = (100.0 * self.correct_answers / attempts) if attempts else None
        self.accuracy_card.set_value(f"{accuracy:.0f}%" if accuracy is not None else "—")

        if self.session_total_objects:
            pct = int(100 * self.solved_objects / self.session_total_objects)
            self.progress.setValue(pct)
            self.progress_label.setText(
                f"Progress: {self.solved_objects} / {self.session_total_objects} objects"
            )
        else:
            self.progress.setValue(0)
            self.progress_label.setText("Progress: —")

    def _update_dataset_label(self) -> None:
        if not self.dataset:
            self.dataset_label.setText("No dataset loaded")
            return
        extra = (
            f" • {self.dataset.missing_labels} images without labels skipped"
            if self.dataset.missing_labels
            else ""
        )
        mode = "weak classes prioritized" if self.focus_weak_classes else "fully random"
        count_text = (
            f"random {min(self.session_image_limit, len(self.dataset.samples))} image subset"
            if 0 < self.session_image_limit < len(self.dataset.samples)
            else "all labeled images in randomized order"
        )
        self.dataset_label.setText(
            f"{len(self.dataset.samples)} labeled images • {self.dataset.total_objects} objects{extra}\n"
            f"Classes: {self.dataset.class_source}\nSession: {count_text} • {mode}"
        )

    def edit_classes(self) -> None:
        if not self.dataset:
            return
        dialog = ClassesDialog(self.dataset.classes, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        names = dialog.class_names()
        if not names:
            QMessageBox.warning(self, "No class names", "Enter at least one class name.")
            return
        max_id = max(box.class_id for sample in self.dataset.samples for box in sample.boxes)
        if len(names) <= max_id:
            QMessageBox.warning(
                self,
                "Not enough class names",
                f"The annotations contain class ID {max_id}, so at least {max_id + 1} names are required.",
            )
            return

        self.dataset.classes = names
        self.dataset.class_source = "classes.txt (edited in trainer)"
        (self.dataset.root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        old_index = self.class_combo.currentIndex()
        self.class_combo.clear()
        self.class_combo.addItems(names)
        self.canvas.set_class_names(names)
        if 0 <= old_index < len(names):
            self.class_combo.setCurrentIndex(old_index)
        self._update_dataset_label()

    def show_stats(self) -> None:
        if not self.db:
            return
        task = self.current_task if self.stack.currentWidget() is self.training_page else self._selected_task()
        trainee_id = self.trainee_id if self.stack.currentWidget() is self.training_page else self.worker_combo.currentData()
        trainee_name = self.trainee_name if self.stack.currentWidget() is self.training_page else self.worker_combo.currentText()
        if not task or trainee_id is None:
            return
        rows = self.db.class_stats(int(trainee_id), task.name)
        StatsDialog(str(trainee_name), task.name, rows, self).exec()

    def _finish_session(self) -> None:
        self.canvas.clear_sample()
        self.image_label.setText("Session complete")
        self.object_label.setText("Objects remaining: 0")
        self.progress.setValue(100)
        self.sound.play_success()
        attempts = self.correct_answers + self.mistakes
        accuracy = 100.0 * self.correct_answers / attempts if attempts else 0.0
        self.feedback.setText(
            f"Finished! Score {self.score} • {self.correct_answers} correct • "
            f"{self.mistakes} mistakes • {accuracy:.1f}% accuracy"
        )
        self.feedback.setStyleSheet(
            "background:#10251b; border:1px solid #246b46; color:#7ff0b2; "
            "border-radius:10px; padding:10px; font-weight:700;"
        )
        self._finalize_active_session(completed=True)
        QMessageBox.information(
            self,
            "Training complete",
            f"Worker: {self.trainee_name}\n"
            f"Task: {self.current_task.name if self.current_task else ''}\n\n"
            f"Score: {self.score}\n"
            f"Correct objects: {self.correct_answers}\n"
            f"Mistakes: {self.mistakes}\n"
            f"Accuracy: {accuracy:.1f}%\n\n"
            "The result has been saved to the worker statistics database.",
        )

    def _install_training_shortcuts(self) -> None:
        """Install shortcuts on the whole training page, not just the main window.

        Child widgets such as buttons and combo boxes normally consume key events before
        QMainWindow.keyPressEvent() sees them. QShortcut with WidgetWithChildrenShortcut
        keeps the training hotkeys active regardless of which training-page control has focus.
        """
        self.training_shortcuts: list[QShortcut] = []

        # 1..9 select the first nine classes. 0 selects class 10 when present.
        for key_text, class_index in [(str(n), n - 1) for n in range(1, 10)] + [("0", 9)]:
            shortcut = QShortcut(QKeySequence(key_text), self.training_page)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(
                lambda idx=class_index: self._select_class_by_shortcut(idx)
            )
            self.training_shortcuts.append(shortcut)

        show_boxes_shortcut = QShortcut(QKeySequence("H"), self.training_page)
        show_boxes_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        show_boxes_shortcut.activated.connect(self._toggle_boxes_shortcut)
        self.training_shortcuts.append(show_boxes_shortcut)

        skip_shortcut = QShortcut(QKeySequence("S"), self.training_page)
        skip_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        skip_shortcut.activated.connect(self.skip_current_image)
        self.training_shortcuts.append(skip_shortcut)

    def _select_class_by_shortcut(self, index: int) -> None:
        if self.stack.currentWidget() is not self.training_page or not self.class_combo.isEnabled():
            return
        if not (0 <= index < self.class_combo.count()):
            return
        self.class_combo.setCurrentIndex(index)
        shortcut_label = str(index + 1) if index < 9 else "0"
        self.feedback.setText(
            f"Active class [{shortcut_label}]: {self.class_combo.currentText()}"
        )

    def _toggle_boxes_shortcut(self) -> None:
        if self.stack.currentWidget() is self.training_page:
            self.show_boxes_button.toggle()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        # Fallback for unusual focus situations. Normal training hotkeys are handled by
        # QShortcut above so child widgets cannot swallow 1..9 / 0 / H / S.
        if self.stack.currentWidget() is self.training_page:
            text = event.text()
            if text in "1234567890":
                index = 9 if text == "0" else int(text) - 1
                self._select_class_by_shortcut(index)
                event.accept()
                return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._finalize_active_session(completed=False)
        if self.db is not None:
            self.db.close()
            self.db = None
        super().closeEvent(event)

    # --------------------------- Styling -----------------------------------

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #090d13;
                color: #edf3ff;
                font-family: Inter, Segoe UI, Arial;
                font-size: 14px;
            }
            QFrame#sidebar, QFrame#mainPanel, QFrame#menuCard {
                background: #111721;
                border: 1px solid #202a38;
                border-radius: 16px;
            }
            QLabel#menuTitle {
                font-size: 34px;
                font-weight: 900;
                color: #ffffff;
            }
            QLabel#title {
                font-size: 24px;
                font-weight: 800;
                color: #ffffff;
            }
            QLabel#subtitle, QLabel#muted { color: #8e9aae; }
            QLabel#pathLabel { color: #dbe5f5; font-weight: 600; }
            QLabel#workerBadge {
                background: #18243a;
                color: #a9bdff;
                border: 1px solid #2b4172;
                border-radius: 8px;
                padding: 6px 9px;
                font-weight: 700;
            }
            QLabel#sectionTitle {
                color: #91a0b8;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QLabel#dialogHeading, QLabel#imageCounter {
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#feedback {
                background: #0d131c;
                border: 1px solid #202a38;
                border-radius: 10px;
                padding: 10px;
                font-size: 15px;
                font-weight: 600;
            }
            QPushButton {
                background: #5b7cfa;
                border: none;
                border-radius: 10px;
                padding: 10px 12px;
                font-weight: 700;
                color: white;
            }
            QPushButton:hover { background: #6b89ff; }
            QPushButton:pressed { background: #4e6fe8; }
            QPushButton:disabled { background: #2a3240; color: #687487; }
            QPushButton#secondaryButton {
                background: #1b2431;
                border: 1px solid #2b3748;
            }
            QPushButton#secondaryButton:hover { background: #222d3d; }
            QPushButton#ghostButton {
                background: transparent;
                border: 1px solid #202a38;
                color: #8e9aae;
                padding: 8px 11px;
            }
            QPushButton#ghostButton:hover {
                background: #111721;
                border-color: #40567a;
                color: #dce5f4;
            }
            QPushButton#startTrainingButton {
                background: #5b7cfa;
                font-size: 15px;
                font-weight: 800;
                border-radius: 12px;
            }
            QPushButton#taskCardButton {
                background: #0d131c;
                color: #eaf1ff;
                border: 1px solid #243044;
                border-radius: 14px;
                padding: 18px;
                text-align: left;
                font-size: 15px;
                font-weight: 700;
            }
            QPushButton#taskCardButton:hover {
                background: #131c29;
                border-color: #4a6290;
            }
            QPushButton#taskCardButton:checked {
                background: #182747;
                border: 2px solid #5b7cfa;
                color: #ffffff;
            }
            QLabel#rootBadge {
                color: #b5c4db;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#taskDetail {
                background: #0d131c;
                border: 1px solid #202a38;
                border-radius: 10px;
                color: #9aa8bd;
                padding: 10px 12px;
            }
            QLabel#emptyTasks {
                color: #7f8ba0;
                font-size: 15px;
                border: 1px dashed #2b3748;
                border-radius: 14px;
                background: #0d131c;
                padding: 24px;
            }
            QScrollArea#taskScroll, QWidget#taskGridWidget {
                background: transparent;
                border: none;
            }
            QComboBox, QSpinBox {
                background: #0d131c;
                border: 1px solid #2b3748;
                border-radius: 10px;
                padding: 9px 10px;
                min-height: 22px;
            }
            QComboBox:hover, QSpinBox:hover { border-color: #5b7cfa; }
            QComboBox QAbstractItemView {
                background: #111721;
                selection-background-color: #5b7cfa;
                border: 1px solid #2b3748;
            }
            QPushButton#skipImageButton {
                background: #2a2314;
                color: #ffd978;
                border: 1px solid #6e5923;
                border-radius: 10px;
                padding: 10px 12px;
                font-weight: 800;
            }
            QPushButton#skipImageButton:hover {
                background: #372d17;
                border-color: #a0812c;
            }
            QPushButton#skipImageButton:pressed {
                background: #1e190f;
            }
            QPushButton#answerRevealButton {
                background: #2a2110;
                color: #ffd166;
                border: 1px solid #6d5521;
                border-radius: 10px;
                padding: 10px 12px;
                font-weight: 800;
                text-align: left;
            }
            QPushButton#answerRevealButton:hover {
                background: #352a13;
                border-color: #ffd166;
            }
            QPushButton#answerRevealButton:checked {
                background: #493814;
                border: 2px solid #ffd166;
                color: #fff2bf;
            }
            QCheckBox { color: #b7c0ce; spacing: 8px; }
            QProgressBar {
                background: #0d131c;
                border: none;
                border-radius: 5px;
                height: 10px;
            }
            QProgressBar::chunk { background: #5b7cfa; border-radius: 5px; }
            QFrame#statCard {
                background: #0d131c;
                border: 1px solid #202a38;
                border-radius: 10px;
            }
            QLabel#statTitle { color: #7f8ba0; font-size: 11px; }
            QLabel#statValue { color: #ffffff; font-size: 22px; font-weight: 800; }
            QPlainTextEdit, QTableWidget {
                background: #0d131c;
                border: 1px solid #2b3748;
                border-radius: 10px;
                padding: 8px;
                selection-background-color: #5b7cfa;
            }
            QHeaderView::section {
                background: #18212e;
                color: #aab6c8;
                padding: 7px;
                border: none;
                border-right: 1px solid #273445;
                font-weight: 700;
            }
            """
        )


def default_training_root() -> Path:
    """Use only TRAINING_DATA next to this Python file.

    This verified build intentionally has no native folder chooser, no environment-root
    override, and no command-line directory override. Dataset selection happens
    only through the in-app category cards discovered under TRAINING_DATA.
    """
    return Path(__file__).resolve().parent / "TRAINING_DATA"


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(f"YOLO Annotation Trainer {APP_VERSION}")
    root = default_training_root()
    print(f"[YOLO Annotation Trainer {APP_VERSION}]")
    print(f"Training root: {root}")
    print(f"SQLite database: {root / DB_FILENAME}")
    window = MainWindow(root)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
