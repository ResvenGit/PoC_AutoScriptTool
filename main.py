from __future__ import annotations

import logging
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, List

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QIntValidator,
    QPainter,
    QPen,
    QResizeEvent,
)
from PySide6.QtCore import QEvent, QTimer, QUrl, Qt, QRect, Signal

from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QSpacerItem,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QFontComboBox,
    QCheckBox,
    QHeaderView,
    QSizePolicy,
    QSpinBox,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoFrame, QVideoSink

from config import SubtitleConfig, load_config, save_config
from asr import transcribe


def format_time_name(seconds: float) -> str:
    """Returns time formatted with millisecond precision."""
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def parse_time_input(value: str) -> float | None:
    """Parses h:m:s(.ms) style input into total seconds."""
    parts = [part.strip() for part in value.split(":") if part.strip()]
    if not parts or len(parts) > 3:
        return None
    parts = ["0"] * (3 - len(parts)) + parts
    multipliers = (3600, 60, 1)
    try:
        return sum(float(part) * weight for part, weight in zip(parts, multipliers))
    except ValueError:
        return None


def build_cue_entry(
    source_start: float,
    source_end: float,
    text: str,
    offset: float = 0.0,
    overlay_meta: dict[str, Any] | None = None,
) -> dict:
    """Creates a cue entry storing both source timing and adjusted labels."""
    source_start = max(0.0, source_start)
    source_end = max(source_start, source_end)
    adjusted_start = max(0.0, source_start - offset)
    adjusted_end = max(adjusted_start, source_end - offset)
    duration = max(0.0, adjusted_end - adjusted_start)
    entry = {
        "source_start": source_start,
        "source_end": source_end,
        "start_seconds": adjusted_start,
        "end_seconds": adjusted_end,
        "start_label": format_time_name(adjusted_start),
        "end_label": format_time_name(adjusted_end),
        "duration_label": format_time_name(duration),
        "text": text,
        "char_count": len(text),
    }
    if overlay_meta:
        overlay_updates = {
            key: overlay_meta[key]
            for key in OVERLAY_METADATA_KEYS
            if key in overlay_meta
        }
        entry.update(overlay_updates)
    return entry


def split_segments(
    segments: List[dict],
    max_chars: int,
) -> List[dict]:
    split_result: List[dict] = []
    for segment in segments:
        word_list = [
            word
            for word in segment.get("words", [])
            if word.get("text")
        ]
        if not word_list:
            continue
        segment_text = " ".join(word["text"] for word in word_list).strip()
        if not segment_text:
            continue
        start = float(segment.get("start", word_list[0]["start"]))
        end = float(segment.get("end", word_list[-1]["end"]))
        if len(segment_text) <= max_chars:
            split_result.append({"start": start, "end": end, "text": segment_text})
            continue
        chunks: List[List[dict]] = []
        current_chunk: List[dict] = []
        current_length = 0
        for word in word_list:
            word_text = word["text"]
            additional = len(word_text) + (1 if current_chunk else 0)
            if current_chunk and current_length + additional > max_chars:
                chunks.append(current_chunk.copy())
                current_chunk = []
                current_length = 0
                additional = len(word_text)
            if current_chunk:
                current_length += 1 + len(word_text)
            else:
                current_length = len(word_text)
            current_chunk.append(word)
        if current_chunk:
            chunks.append(current_chunk.copy())

        min_chunk_len = max_chars / 3.0
        while len(chunks) >= 2:
            last_text = " ".join(word["text"] for word in chunks[-1]).strip()
            if len(last_text) >= min_chunk_len:
                break
            chunks[-2].extend(chunks[-1])
            chunks.pop()

        for chunk in chunks:
            chunk_text = " ".join(word["text"] for word in chunk)
            chunk_start = float(chunk[0]["start"])
            chunk_end = float(chunk[-1]["end"])
            split_result.append({"start": chunk_start, "end": chunk_end, "text": chunk_text})
    return split_result


DEFAULT_FONT_FAMILY = "Malgun Gothic"
DEFAULT_FONT_SIZE = 32
DEFAULT_FONT_COLOR = "#FFFFFF"
DEFAULT_OUTLINE_ENABLED = True
DEFAULT_OUTLINE_THICKNESS = 3
DEFAULT_OUTLINE_COLOR = "#000000"
DEFAULT_OVERLAY_MARGIN = 16
DEFAULT_OVERLAY_HEIGHT = 80
OVERLAY_METADATA_KEYS = [
    "font_family",
    "font_size",
    "font_color",
    "outline_enabled",
    "outline_thickness",
    "outline_color",
    "overlay_rect",
    "overlay_space",
    "show_overlay_box",
    "mute_preview",
]


def _rect_to_dict(rect: QRect) -> dict:
    return {"x": rect.x(), "y": rect.y(), "width": rect.width(), "height": rect.height()}


def _dict_to_rect(data: dict[str, int], bounds: QRect) -> QRect:
    x = int(data.get("x", DEFAULT_OVERLAY_MARGIN))
    y = int(data.get("y", bounds.height() - DEFAULT_OVERLAY_HEIGHT - DEFAULT_OVERLAY_MARGIN))
    width = int(data.get("width", max(100, bounds.width() - DEFAULT_OVERLAY_MARGIN * 2)))
    height = int(data.get("height", DEFAULT_OVERLAY_HEIGHT))
    rect = QRect(x, y, width, height)
    return rect.intersected(bounds)


class VideoPreviewWidget(QWidget):
    clicked = Signal()
    rectAdjusted = Signal(QRect)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sink: QVideoSink | None = None
        self._frame_image = None
        self._video_size = None
        self._video_target_rect = QRect()
        self._compositing_enabled = False
        self._subtitle_text = ""
        self._subtitle_style: dict[str, Any] = {
            "font_family": DEFAULT_FONT_FAMILY,
            "font_size": DEFAULT_FONT_SIZE,
            "font_color": DEFAULT_FONT_COLOR,
            "outline_enabled": DEFAULT_OUTLINE_ENABLED,
            "outline_thickness": DEFAULT_OUTLINE_THICKNESS,
            "outline_color": DEFAULT_OUTLINE_COLOR,
        }
        self._subtitle_rect_video: QRect | None = None
        self._wrap_cache_key = None
        self._wrap_cache_lines: list[str] | None = None

    def set_video_sink(self, sink: QVideoSink) -> None:
        if self._sink is not None:
            try:
                self._sink.videoFrameChanged.disconnect(self._on_video_frame_changed)
            except (TypeError, RuntimeError):
                pass
        self._sink = sink
        sink.videoFrameChanged.connect(self._on_video_frame_changed)

    def video_size(self):
        return self._video_size

    def video_target_rect(self) -> QRect:
        self._video_target_rect = self._compute_video_target_rect()
        return QRect(self._video_target_rect)

    def _compute_video_target_rect(self) -> QRect:
        if self._video_size is None:
            return QRect()
        video_w = self._video_size.width()
        video_h = self._video_size.height()
        if video_w <= 0 or video_h <= 0:
            return QRect()
        widget_w = max(1, self.width())
        widget_h = max(1, self.height())
        scale = min(widget_w / video_w, widget_h / video_h)
        target_w = max(1, int(round(video_w * scale)))
        target_h = max(1, int(round(video_h * scale)))
        x = (widget_w - target_w) // 2
        y = (widget_h - target_h) // 2
        return QRect(x, y, target_w, target_h)

    def map_video_rect_to_widget(self, rect: QRect) -> QRect | None:
        if self._video_size is None:
            return None
        target = self.video_target_rect()
        if target.isEmpty():
            return None
        video_w = self._video_size.width()
        video_h = self._video_size.height()
        if video_w <= 0 or video_h <= 0:
            return None
        sx = target.width() / video_w
        sy = target.height() / video_h
        x = target.x() + rect.x() * sx
        y = target.y() + rect.y() * sy
        w = rect.width() * sx
        h = rect.height() * sy
        return QRect(int(round(x)), int(round(y)), int(round(w)), int(round(h)))

    def map_widget_rect_to_video(self, rect: QRect) -> QRect | None:
        if self._video_size is None:
            return None
        target = self.video_target_rect()
        if target.isEmpty():
            return None
        video_w = self._video_size.width()
        video_h = self._video_size.height()
        if video_w <= 0 or video_h <= 0:
            return None
        sx = target.width() / video_w
        sy = target.height() / video_h
        if sx == 0 or sy == 0:
            return None
        x = (rect.x() - target.x()) / sx
        y = (rect.y() - target.y()) / sy
        w = rect.width() / sx
        h = rect.height() / sy
        return QRect(int(round(x)), int(round(y)), int(round(w)), int(round(h)))

    def enable_compositing(self, enabled: bool) -> None:
        self._compositing_enabled = enabled
        self.update()

    def set_subtitle(self, text: str, style: dict[str, Any], rect_video: QRect | None) -> None:
        self._subtitle_text = text or ""
        self._subtitle_style = dict(style or {})
        self._subtitle_rect_video = QRect(rect_video) if rect_video is not None else None
        self.update()

    def clear_subtitle(self) -> None:
        self._subtitle_text = ""
        self._subtitle_rect_video = None
        self.update()

    def _on_video_frame_changed(self, frame: QVideoFrame) -> None:
        try:
            self._frame_image = frame.toImage()
        except Exception:
            self._frame_image = None
        if self._frame_image is not None and not self._frame_image.isNull():
            self._video_size = self._frame_image.size()
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        return super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        try:
            painter.setRenderHints(
                QPainter.Antialiasing
                | QPainter.TextAntialiasing
                | QPainter.SmoothPixmapTransform
            )
            painter.fillRect(self.rect(), QColor(0, 0, 0))

            self._video_target_rect = self._compute_video_target_rect()
            if self._frame_image is not None and not self._frame_image.isNull():
                image = self._frame_image
                scaled = image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                x = (self.width() - scaled.width()) // 2
                y = (self.height() - scaled.height()) // 2
                self._video_target_rect = QRect(x, y, scaled.width(), scaled.height())
                painter.drawImage(x, y, scaled)

            if not self._compositing_enabled:
                return
            if not self._subtitle_text or self._subtitle_rect_video is None:
                return

            rect = self.map_video_rect_to_widget(self._subtitle_rect_video)
            if rect is None:
                return
            target = self._video_target_rect if not self._video_target_rect.isEmpty() else self.rect()
            rect = rect.intersected(target)
            if rect.isEmpty():
                return

            adjusted = self._expand_rect_upwards_to_fit(rect, target, self._subtitle_text, self._subtitle_style)
            if adjusted != rect:
                adjusted_video = self.map_widget_rect_to_video(adjusted)
                if adjusted_video is not None:
                    self.rectAdjusted.emit(adjusted_video)
                rect = adjusted

            self._draw_subtitle(painter, rect, self._subtitle_text, self._subtitle_style)
        finally:
            if painter.isActive():
                painter.end()

    def _expand_rect_upwards_to_fit(
        self,
        rect: QRect,
        bounds: QRect,
        text: str,
        style: dict[str, Any],
    ) -> QRect:
        padding = 8
        available_width = max(20, rect.width() - padding * 2)
        scale = 1.0
        if self._video_size is not None and not self._video_target_rect.isEmpty():
            video_w = self._video_size.width()
            if video_w > 0:
                scale = self._video_target_rect.width() / video_w
        font_size = int(style.get("font_size") or DEFAULT_FONT_SIZE)
        font = QFont(
            style.get("font_family") or DEFAULT_FONT_FAMILY,
            max(1, int(round(max(8, font_size) * scale))),
        )
        metrics = QFontMetrics(font)
        lines = self._wrap_lines_balanced(text, font, metrics, available_width)
        needed_height = len(lines) * metrics.lineSpacing() + padding * 2
        if needed_height <= rect.height():
            return rect
        y2 = rect.y() + rect.height()
        new_y1 = max(bounds.top(), y2 - needed_height)
        expanded = QRect(rect.x(), new_y1, rect.width(), y2 - new_y1)
        return expanded.intersected(bounds)

    def _wrap_lines_balanced(
        self,
        text: str,
        font: QFont,
        metrics: QFontMetrics,
        max_width: int,
    ) -> list[str]:
        cache_key = (
            text,
            font.family(),
            int(font.pointSize()),
            int(font.pixelSize()),
            int(font.weight()),
            bool(font.italic()),
            int(max_width),
        )
        if cache_key == self._wrap_cache_key and self._wrap_cache_lines is not None:
            return self._wrap_cache_lines

        paragraphs = text.splitlines() or [text]
        lines: list[str] = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                lines.append("")
                continue
            if " " in paragraph:
                tokens = paragraph.split()
                space_width = metrics.horizontalAdvance(" ")
                token_widths = [metrics.horizontalAdvance(token) for token in tokens]
                if any(width > max_width for width in token_widths):
                    lines.extend(self._wrap_by_chars_balanced(paragraph, font, metrics, max_width))
                    continue
                token_chars = [len(token) for token in tokens]
                line_count = self._min_line_count(token_widths, space_width, max_width)
                if line_count <= 1:
                    lines.append(" ".join(tokens))
                    continue
                breaks = self._optimal_breaks_fixed_lines(
                    token_widths,
                    token_chars,
                    space_width,
                    max_width,
                    line_count,
                )
                if breaks is None:
                    lines.extend(self._wrap_by_chars_balanced(paragraph, font, metrics, max_width))
                    continue
                idx = 0
                for next_idx in breaks:
                    lines.append(" ".join(tokens[idx:next_idx]))
                    idx = next_idx
                continue

            lines.extend(self._wrap_by_chars_balanced(paragraph, font, metrics, max_width))

        self._wrap_cache_key = cache_key
        self._wrap_cache_lines = lines
        return lines

    def _wrap_by_chars_balanced(self, text: str, font: QFont, metrics: QFontMetrics, max_width: int) -> list[str]:
        chars = list(text)
        char_widths = [metrics.horizontalAdvance(ch) for ch in chars]
        line_count = self._min_line_count(char_widths, 0, max_width)
        if line_count <= 1:
            return [text]
        breaks = self._optimal_breaks_fixed_lines(
            char_widths,
            [1] * len(chars),
            0,
            max_width,
            line_count,
        )
        if breaks is None:
            breaks = self._optimal_breaks(char_widths, 0, max_width)
        lines: list[str] = []
        idx = 0
        for next_idx in breaks:
            lines.append("".join(chars[idx:next_idx]))
            idx = next_idx
        return lines

    def _min_line_count(self, token_widths: list[int], space_width: int, max_width: int) -> int:
        if not token_widths:
            return 1
        if any(width > max_width for width in token_widths):
            return max(1, len(token_widths))
        lines = 1
        line_w = 0
        for width in token_widths:
            candidate = width if line_w == 0 else line_w + space_width + width
            if line_w and candidate > max_width:
                lines += 1
                line_w = width
            else:
                line_w = candidate
        return max(1, lines)

    def _optimal_breaks_fixed_lines(
        self,
        token_widths: list[int],
        token_chars: list[int],
        space_width: int,
        max_width: int,
        line_count: int,
    ) -> list[int] | None:
        n = len(token_widths)
        if n == 0:
            return []
        line_count = max(1, int(line_count))
        avg_chars = (sum(token_chars) / line_count) if line_count else float(sum(token_chars))
        inf = 10**18
        dp: list[list[float]] = [[float(inf)] * (line_count + 1) for _ in range(n + 1)]
        nxt: list[list[int]] = [[0] * (line_count + 1) for _ in range(n + 1)]
        dp[n][0] = 0.0
        for i in range(n - 1, -1, -1):
            for l in range(1, line_count + 1):
                line_w = 0
                line_chars = 0
                for j in range(i, n):
                    if j == i:
                        line_w = token_widths[j]
                    else:
                        line_w += space_width + token_widths[j]
                    if line_w > max_width and j > i:
                        break
                    if line_w > max_width and j == i:
                        continue
                    line_chars += token_chars[j]
                    remainder = dp[j + 1][l - 1]
                    if remainder >= float(inf):
                        continue
                    diff = line_chars - avg_chars
                    cost = diff * diff
                    candidate = cost + remainder
                    if candidate < dp[i][l]:
                        dp[i][l] = candidate
                        nxt[i][l] = j + 1
        if dp[0][line_count] >= float(inf):
            return None
        breaks: list[int] = []
        idx = 0
        remaining = line_count
        while idx < n and remaining > 0:
            next_idx = nxt[idx][remaining]
            if next_idx <= idx:
                next_idx = min(n, idx + 1)
            breaks.append(next_idx)
            idx = next_idx
            remaining -= 1
        if idx < n:
            breaks.append(n)
        return breaks

    def _optimal_breaks(self, token_widths: list[int], space_width: int, max_width: int) -> list[int]:
        n = len(token_widths)
        if n == 0:
            return [0]
        inf = 10**18
        dp = [inf] * (n + 1)
        nxt = [0] * (n + 1)
        dp[n] = 0
        for i in range(n - 1, -1, -1):
            line_w = 0
            for j in range(i, n):
                if j == i:
                    line_w = token_widths[j]
                else:
                    line_w += space_width + token_widths[j]
                if line_w > max_width and j > i:
                    break
                if line_w > max_width and j == i:
                    continue
                slack = max_width - line_w
                cost = slack * slack
                if j == n - 1:
                    cost = 0
                candidate = cost + dp[j + 1]
                if candidate < dp[i]:
                    dp[i] = candidate
                    nxt[i] = j + 1
        breaks: list[int] = []
        idx = 0
        while idx < n:
            next_idx = nxt[idx]
            if next_idx <= idx:
                next_idx = min(n, idx + 1)
            breaks.append(next_idx)
            idx = next_idx
        return breaks

    def _draw_subtitle(self, painter: QPainter, rect: QRect, text: str, style: dict[str, Any]) -> None:
        padding = 8
        font_family = style.get("font_family") or DEFAULT_FONT_FAMILY
        scale = 1.0
        if self._video_size is not None and not self._video_target_rect.isEmpty():
            video_w = self._video_size.width()
            if video_w > 0:
                scale = self._video_target_rect.width() / video_w
        font_size = int(style.get("font_size") or DEFAULT_FONT_SIZE)
        font = QFont(font_family, max(1, int(round(max(8, font_size) * scale))))
        painter.setFont(font)

        text_rect = rect.adjusted(padding, padding, -padding, -padding)
        metrics = QFontMetrics(font)
        max_width = max(20, text_rect.width())
        lines = self._wrap_lines_balanced(text, font, metrics, max_width)
        line_spacing = metrics.lineSpacing()
        baseline_last = text_rect.bottom() - metrics.descent()
        baseline_first = baseline_last - (len(lines) - 1) * line_spacing

        font_color = QColor(style.get("font_color") or DEFAULT_FONT_COLOR)
        outline_enabled = bool(style.get("outline_enabled", DEFAULT_OUTLINE_ENABLED))
        outline_thickness = int(style.get("outline_thickness") or DEFAULT_OUTLINE_THICKNESS)
        outline_thickness = max(0, int(round(outline_thickness * scale)))
        outline_color = QColor(style.get("outline_color") or DEFAULT_OUTLINE_COLOR)

        for idx, line in enumerate(lines):
            baseline = int(round(baseline_first + idx * line_spacing))
            if baseline < text_rect.top():
                continue
            line_width = metrics.horizontalAdvance(line)
            x = int(round(text_rect.x() + (text_rect.width() - line_width) / 2))
            if outline_enabled and outline_thickness > 0 and line:
                outline_pen = QPen(outline_color)
                painter.setPen(outline_pen)
                for dx in range(-outline_thickness, outline_thickness + 1):
                    for dy in range(-outline_thickness, outline_thickness + 1):
                        if dx == 0 and dy == 0:
                            continue
                        painter.drawText(x + dx, baseline + dy, line)
            painter.setPen(QPen(font_color))
            painter.drawText(x, baseline, line)


class SubtitleCreatorMainWindow(QMainWindow):
    _invoke_signal = Signal(object)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Subtitle Creator")
        self.config: SubtitleConfig = load_config()
        self._invoke_signal.connect(self._run_ui_callback)
        width = self.config.window_width or 860
        height = self.config.window_height or 660
        self.resize(width, height)
        self.cues: List[dict] = []
        self.raw_segments: List[dict] = []
        self.media_path = ""
        self.start_offset = max(0.0, self.config.start_offset)
        self._asr_start_time: float | None = None
        self._video_bounds_hint: QRect | None = None
        self._global_overlay_style: dict[str, Any] = {
            "font_family": self.config.overlay_font_family or DEFAULT_FONT_FAMILY,
            "font_size": self.config.overlay_font_size or DEFAULT_FONT_SIZE,
            "font_color": self.config.overlay_font_color or DEFAULT_FONT_COLOR,
            "outline_enabled": (
                self.config.overlay_outline_enabled
                if self.config.overlay_outline_enabled is not None
                else DEFAULT_OUTLINE_ENABLED
            ),
            "outline_thickness": (
                self.config.overlay_outline_thickness
                if self.config.overlay_outline_thickness is not None
                else DEFAULT_OUTLINE_THICKNESS
            ),
            "outline_color": self.config.overlay_outline_color or DEFAULT_OUTLINE_COLOR,
        }
        self._active_cue_index: int | None = None
        self._active_overlay_rect: QRect | None = None
        self._subtitle_active = False
        self._compositing_enabled = True
        self._active_subtitle_text = ""
        self._active_subtitle_style: dict[str, Any] = dict(self._global_overlay_style)
        self._show_overlay_box = True
        self._mute_preview = False
        self._global_overlay_rect_default: dict[str, int] | None = None
        self._global_overlay_rect_norm: dict[str, float] | None = self.config.overlay_rect_norm
        self._coord_edit_blocked = False
        self._dragging_overlay = False
        self._drag_start_pos = None
        self._drag_start_rect: QRect | None = None
        self._drag_mode: str | None = None
        self._pending_overlay_rect_apply = False
        self._continuous_preview = False
        self._build_ui()
        self._apply_font_controls(self._global_overlay_style)
        self._apply_preview_option_controls()
        self._global_overlay_rect_default = None
        self._active_overlay_rect = None
        self._coord_edit_blocked = True
        self.overlay_coord_inputs["x1"].setText("0")
        self.overlay_coord_inputs["y1"].setText("0")
        self.overlay_coord_inputs["x2"].setText("0")
        self.overlay_coord_inputs["y2"].setText("0")
        self._coord_edit_blocked = False
        self.overlay_position_label.setText(self._format_overlay_coords(None))
        self.video_sink.videoFrameChanged.connect(self._on_video_frame_changed)

    def _build_ui(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        video_row = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("비디오 파일을 선택하세요")
        video_row.addWidget(QLabel("비디오"))
        video_row.addWidget(self.video_path_edit)
        browse_btn = QPushButton("열기")
        browse_btn.clicked.connect(self._browse_video)
        video_row.addWidget(browse_btn)
        layout.addLayout(video_row)

        length_row = QHBoxLayout()
        self.char_length_edit = QLineEdit(str(self.config.subtitle_char_length))
        self.char_length_edit.setValidator(QIntValidator(10, 200))
        self.char_length_edit.editingFinished.connect(self._char_length_changed)
        length_row.addWidget(QLabel("최대 문자 수"))
        length_row.addWidget(self.char_length_edit)
        layout.addLayout(length_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        entry_row = QHBoxLayout()
        self.start_edit = QLineEdit()
        self.start_edit.setPlaceholderText("시작 (hh:mm:ss)")
        self.end_edit = QLineEdit()
        self.end_edit.setPlaceholderText("종료 (hh:mm:ss)")
        self.text_edit = QLineEdit()
        self.text_edit.setPlaceholderText("자막 내용을 입력하세요")
        add_btn = QPushButton("자막 추가")
        add_btn.clicked.connect(self._add_manual_cue)
        entry_row.addWidget(QLabel("Start"))
        entry_row.addWidget(self.start_edit)
        entry_row.addWidget(QLabel("End"))
        entry_row.addWidget(self.end_edit)
        entry_row.addWidget(self.text_edit, 2)
        entry_row.addWidget(add_btn)
        layout.addLayout(entry_row)

        layout.addWidget(QLabel("자막 목록"))
        self.cue_table = QTableWidget()
        self.cue_table.setColumnCount(5)
        self.cue_table.setHorizontalHeaderLabels(["Start", "End", "Duration", "Text", "Chars"])
        self.cue_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cue_table.setEditTriggers(
            QAbstractItemView.DoubleClicked
            | QAbstractItemView.SelectedClicked
            | QAbstractItemView.EditKeyPressed
        )
        header = self.cue_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.cue_table.cellClicked.connect(self._on_cue_selected)
        self.cue_table.cellChanged.connect(self._on_cue_cell_changed)
        layout.addWidget(self.cue_table, 1)

        controls = QHBoxLayout()
        self.remove_btn = QPushButton("선택 삭제")
        self.remove_btn.clicked.connect(self._remove_selected)
        self.remove_btn.setEnabled(False)
        controls.addSpacerItem(QSpacerItem(1, 1))
        controls.addWidget(self.remove_btn)
        layout.addLayout(controls)

        layout.addWidget(QLabel("미리보기"))
        self.video_widget = VideoPreviewWidget()
        self.video_widget.setMinimumHeight(240)
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_container = QWidget()
        container_layout = QVBoxLayout(self.video_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(self.video_widget)
        self.video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_container, 2)
        self.video_sink = QVideoSink(self)
        self.video_widget.set_video_sink(self.video_sink)
        self.video_widget.clicked.connect(self._on_preview_clicked)
        self.video_widget.rectAdjusted.connect(self._on_subtitle_rect_adjusted)
        preview_options_row = QHBoxLayout()
        preview_options_row.setSpacing(12)
        self.show_overlay_box_checkbox = QCheckBox("자막 영역 표시")
        self.show_overlay_box_checkbox.setChecked(True)
        self.show_overlay_box_checkbox.stateChanged.connect(self._on_preview_option_changed)
        preview_options_row.addWidget(self.show_overlay_box_checkbox)
        self.mute_preview_checkbox = QCheckBox("미리보기 음소거")
        self.mute_preview_checkbox.setChecked(False)
        self.mute_preview_checkbox.stateChanged.connect(self._on_preview_option_changed)
        preview_options_row.addWidget(self.mute_preview_checkbox)
        self.continuous_preview_checkbox = QCheckBox("미리보기 연속 재생")
        self.continuous_preview_checkbox.setChecked(False)
        self.continuous_preview_checkbox.stateChanged.connect(self._on_continuous_preview_changed)
        preview_options_row.addWidget(self.continuous_preview_checkbox)
        self.stop_preview_button = QPushButton("미리보기 멈춤")
        self.stop_preview_button.clicked.connect(self._on_stop_preview_clicked)
        preview_options_row.addWidget(self.stop_preview_button)
        preview_options_row.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addLayout(preview_options_row)
        self.preview_border = QWidget(self.video_widget)
        self.preview_border.setStyleSheet(
            "border: 2px solid #00cc00; background-color: transparent;"
        )
        self.preview_border.hide()
        self.preview_border.setCursor(Qt.OpenHandCursor)
        self.preview_border.setMouseTracking(True)
        self.preview_border.installEventFilter(self)
        self.video_container.installEventFilter(self)
        self.overlay_position_label = QLabel("오버레이 위치 (x1, y1) (x2, y2)")
        self.overlay_position_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.overlay_position_label)
        coord_layout = QHBoxLayout()
        coord_layout.setSpacing(6)
        self.overlay_coord_inputs: dict[str, QLineEdit] = {}
        coord_label = QLabel("좌표")
        coord_label.setFixedWidth(40)
        coord_layout.addWidget(coord_label)
        for key in ("x1", "y1", "x2", "y2"):
            group = QWidget()
            group_layout = QHBoxLayout(group)
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.setSpacing(4)
            key_label = QLabel(key)
            key_label.setFixedWidth(22)
            key_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            group_layout.addWidget(key_label)
            edit = QLineEdit()
            edit.setFixedWidth(80)
            edit.setValidator(QIntValidator(0, 9999))
            edit.editingFinished.connect(self._on_overlay_coord_changed)
            group_layout.addWidget(edit)
            coord_layout.addWidget(group)
            self.overlay_coord_inputs[key] = edit
        coord_layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum))
        layout.addLayout(coord_layout)
        font_controls_layout = QHBoxLayout()
        font_controls_layout.setSpacing(8)
        font_controls_layout.addWidget(QLabel("폰트"))
        self.font_combo = QFontComboBox()
        self.font_combo.currentFontChanged.connect(self._on_font_control_changed)
        font_controls_layout.addWidget(self.font_combo)
        font_controls_layout.addWidget(QLabel("크기"))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(10, 200)
        self.font_size_spin.valueChanged.connect(self._on_font_control_changed)
        font_controls_layout.addWidget(self.font_size_spin)
        font_controls_layout.addWidget(QLabel("글자 색"))
        self.font_color_edit = QLineEdit(DEFAULT_FONT_COLOR)
        self.font_color_edit.setFixedWidth(90)
        self.font_color_edit.editingFinished.connect(self._on_font_control_changed)
        font_controls_layout.addWidget(self.font_color_edit)
        font_color_btn = QPushButton("선택")
        font_color_btn.clicked.connect(
            lambda: self._choose_color(self.font_color_edit, DEFAULT_FONT_COLOR)
        )
        font_controls_layout.addWidget(font_color_btn)
        self.outline_checkbox = QCheckBox("외곽선")
        self.outline_checkbox.stateChanged.connect(self._on_font_control_changed)
        font_controls_layout.addWidget(self.outline_checkbox)
        font_controls_layout.addWidget(QLabel("두께"))
        self.outline_spin = QSpinBox()
        self.outline_spin.setRange(0, 20)
        self.outline_spin.valueChanged.connect(self._on_font_control_changed)
        font_controls_layout.addWidget(self.outline_spin)
        font_controls_layout.addWidget(QLabel("색"))
        self.outline_color_edit = QLineEdit(DEFAULT_OUTLINE_COLOR)
        self.outline_color_edit.setFixedWidth(90)
        self.outline_color_edit.editingFinished.connect(self._on_font_control_changed)
        font_controls_layout.addWidget(self.outline_color_edit)
        outline_color_btn = QPushButton("선택")
        outline_color_btn.clicked.connect(
            lambda: self._choose_color(self.outline_color_edit, DEFAULT_OUTLINE_COLOR)
        )
        font_controls_layout.addWidget(outline_color_btn)
        layout.addLayout(font_controls_layout)
        self.preview_status_label = QLabel("자막을 선택하면 영상과 음성이 재생됩니다.")
        layout.addWidget(self.preview_status_label)
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoSink(self.video_sink)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._pause_preview)

        layout.addWidget(QLabel("로그"))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, 1)

        export_row = QHBoxLayout()
        export_row.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.export_btn = QPushButton("영상 Export")
        self.export_btn.clicked.connect(self._export_video)
        export_row.addWidget(self.export_btn)
        layout.addLayout(export_row)

        self.setCentralWidget(root)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _position_overlay(self) -> QRect | None:
        if not self._subtitle_active:
            self.preview_border.hide()
            self.overlay_position_label.setText(self._format_overlay_coords(None))
            return None
        bounds = self._video_bounds()
        if bounds is None:
            self.preview_border.hide()
            return None
        rect = self._active_overlay_rect
        if rect is None:
            default = self._global_overlay_rect_default or self._initial_overlay_rect_dict()
            rect = _dict_to_rect(default, bounds)
            self._active_overlay_rect = rect
        return self._set_overlay_rect(rect, update_inputs=True, persist=False)

    def _set_overlay_rect(
        self,
        rect: QRect,
        update_inputs: bool = True,
        persist: bool = True,
    ) -> QRect:
        bounds = self._video_bounds()
        if bounds is None:
            self._active_overlay_rect = QRect(rect)
            self.preview_border.hide()
            self.overlay_position_label.setText(self._format_overlay_coords(None))
            return QRect(rect)
        rect = QRect(rect)
        rect = self._clamp_rect_to_bounds(rect, bounds)
        self._active_overlay_rect = rect
        widget_rect = self.video_widget.map_video_rect_to_widget(rect)
        if not self._show_overlay_box:
            self.preview_border.hide()
        elif widget_rect is None or widget_rect.isEmpty():
            self.preview_border.hide()
        else:
            self.preview_border.setGeometry(widget_rect)
            self.preview_border.show()
            self.preview_border.raise_()
        self.video_widget.set_subtitle(self._active_subtitle_text, self._active_subtitle_style, rect)
        coords_text = self._format_overlay_coords(rect)
        self.overlay_position_label.setText(coords_text)
        if update_inputs and not self._coord_edit_blocked:
            self._coord_edit_blocked = True
            self._update_coord_inputs(rect)
            self._coord_edit_blocked = False
        if persist and self._active_cue_index is not None:
            rect_dict = _rect_to_dict(rect)
            self._apply_overlay_rect_to_following(rect_dict)
        return rect

    def _show_preview_overlay(self, cue: dict) -> None:
        text = cue.get("text", "").strip()
        if not text:
            self._subtitle_active = False
            self.video_widget.clear_subtitle()
            self.preview_border.hide()
            self.overlay_position_label.setText(self._format_overlay_coords(None))
            return
        style = self._cue_style(cue)
        options = self._cue_options(cue)
        self._subtitle_active = True
        self._active_subtitle_text = text
        self._active_subtitle_style = dict(style)
        self._show_overlay_box = bool(options["show_overlay_box"])
        self._mute_preview = bool(options["mute_preview"])
        self._apply_preview_option_controls()
        self.video_widget.enable_compositing(self._compositing_enabled)
        bounds = self._video_bounds()
        if bounds is None:
            self.preview_border.hide()
            return
        self._ensure_cue_overlay_rect_video(cue)
        rect_source = cue.get("overlay_rect") or self._global_overlay_rect_default
        rect = _dict_to_rect(rect_source or self._initial_overlay_rect_dict(), bounds)
        self._set_overlay_rect(rect, update_inputs=True)

    def _video_bounds(self) -> QRect | None:
        size = self.video_widget.video_size()
        if size is None:
            return QRect(self._video_bounds_hint) if self._video_bounds_hint is not None else None
        width = int(getattr(size, "width", lambda: 0)())
        height = int(getattr(size, "height", lambda: 0)())
        if width <= 0 or height <= 0:
            return None
        return QRect(0, 0, width, height)

    def _video_rect_to_norm(self, rect: QRect, bounds: QRect) -> dict[str, float]:
        width = float(bounds.width()) if bounds.width() else 1.0
        height = float(bounds.height()) if bounds.height() else 1.0
        return {
            "x": rect.x() / width,
            "y": rect.y() / height,
            "width": rect.width() / width,
            "height": rect.height() / height,
        }

    def _norm_to_video_rect_dict(self, norm: dict[str, float], bounds: QRect) -> dict[str, int] | None:
        try:
            x = float(norm["x"])
            y = float(norm["y"])
            w = float(norm["width"])
            h = float(norm["height"])
        except Exception:
            return None
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.01, min(1.0, w))
        h = max(0.01, min(1.0, h))
        rect = QRect(
            int(round(x * bounds.width())),
            int(round(y * bounds.height())),
            int(round(w * bounds.width())),
            int(round(h * bounds.height())),
        )
        rect = self._clamp_rect_to_bounds(rect, bounds)
        return _rect_to_dict(rect)

    def _persist_overlay_style_to_config(self) -> None:
        self.config.overlay_font_family = str(self._global_overlay_style.get("font_family") or DEFAULT_FONT_FAMILY)
        self.config.overlay_font_size = int(self._global_overlay_style.get("font_size") or DEFAULT_FONT_SIZE)
        self.config.overlay_font_color = str(self._global_overlay_style.get("font_color") or DEFAULT_FONT_COLOR)
        self.config.overlay_outline_enabled = bool(self._global_overlay_style.get("outline_enabled", DEFAULT_OUTLINE_ENABLED))
        self.config.overlay_outline_thickness = int(
            self._global_overlay_style.get("outline_thickness") or DEFAULT_OUTLINE_THICKNESS
        )
        self.config.overlay_outline_color = str(self._global_overlay_style.get("outline_color") or DEFAULT_OUTLINE_COLOR)
        save_config(self.config)

    def _persist_overlay_rect_to_config(self, rect_video: QRect) -> None:
        bounds = self._video_bounds()
        if bounds is None:
            return
        if rect_video.isNull() or rect_video.width() <= 0 or rect_video.height() <= 0:
            return
        if (
            rect_video.x() == 0
            and rect_video.y() == 0
            and rect_video.width() == 0
            and rect_video.height() == 0
        ):
            return
        norm = self._video_rect_to_norm(rect_video, bounds)
        for key in norm:
            norm[key] = round(float(norm[key]), 6)
        self.config.overlay_rect_norm = norm
        self._global_overlay_rect_norm = dict(norm)
        save_config(self.config)

    def _apply_overlay_rect_to_following(self, rect_dict: dict[str, int]) -> None:
        start_index = self._active_cue_index if self._active_cue_index is not None else 0
        for cue in self.cues[start_index:]:
            cue["overlay_rect"] = dict(rect_dict)
            cue["overlay_space"] = "video"
        self._global_overlay_rect_default = dict(rect_dict)

    def _on_video_frame_changed(self, frame: QVideoFrame) -> None:
        size = frame.size()
        if size.isEmpty():
            return
        bounds = self._video_bounds()
        if bounds is None:
            return
        if self._global_overlay_rect_default is None:
            if self._global_overlay_rect_norm:
                rect_dict = self._norm_to_video_rect_dict(self._global_overlay_rect_norm, bounds)
                self._global_overlay_rect_default = rect_dict or self._initial_overlay_rect_dict()
            else:
                self._global_overlay_rect_default = self._initial_overlay_rect_dict()
        if self._pending_overlay_rect_apply and self._active_cue_index is None:
            rect_dict = self._global_overlay_rect_default or self._initial_overlay_rect_dict()
            rect = _dict_to_rect(rect_dict, bounds)
            self._active_overlay_rect = rect
            self._set_overlay_rect(rect, update_inputs=True, persist=False)
            self._pending_overlay_rect_apply = False
        if self._active_cue_index is not None and 0 <= self._active_cue_index < len(self.cues):
            cue = self.cues[self._active_cue_index]
            self._ensure_cue_overlay_rect_video(cue)
            rect_source = cue.get("overlay_rect") or self._global_overlay_rect_default
            rect = _dict_to_rect(rect_source or self._initial_overlay_rect_dict(), bounds)
            self._set_overlay_rect(rect, update_inputs=True, persist=False)

    def _ensure_cue_overlay_rect_video(self, cue: dict) -> None:
        bounds = self._video_bounds()
        if bounds is None:
            return
        rect_dict = cue.get("overlay_rect")
        if not isinstance(rect_dict, dict):
            cue["overlay_rect"] = dict(self._global_overlay_rect_default or self._initial_overlay_rect_dict())
            cue["overlay_space"] = "video"
            return
        try:
            rect = QRect(
                int(rect_dict.get("x", 0)),
                int(rect_dict.get("y", 0)),
                int(rect_dict.get("width", 0)),
                int(rect_dict.get("height", 0)),
            )
        except Exception:
            cue["overlay_rect"] = dict(self._global_overlay_rect_default or self._initial_overlay_rect_dict())
            cue["overlay_space"] = "video"
            return
        space = cue.get("overlay_space")
        if space == "video":
            cue["overlay_rect"] = _rect_to_dict(self._clamp_rect_to_bounds(rect, bounds))
            return
        widget_rect = QRect(rect)
        converted = self.video_widget.map_widget_rect_to_video(widget_rect)
        if converted is None:
            cue["overlay_rect"] = dict(self._global_overlay_rect_default or self._initial_overlay_rect_dict())
            cue["overlay_space"] = "video"
            return
        converted = self._clamp_rect_to_bounds(converted, bounds)
        cue["overlay_rect"] = _rect_to_dict(converted)
        cue["overlay_space"] = "video"

    def _on_preview_clicked(self) -> None:
        if not self._compositing_enabled:
            self._compositing_enabled = True
            self.video_widget.enable_compositing(True)
        if self._subtitle_active and self._active_overlay_rect is not None:
            self.video_widget.set_subtitle(
                self._active_subtitle_text,
                self._active_subtitle_style,
                self._active_overlay_rect,
            )

    def _on_preview_option_changed(self, *_) -> None:
        if self._active_cue_index is None:
            self._show_overlay_box = bool(self.show_overlay_box_checkbox.isChecked())
            self._mute_preview = bool(self.mute_preview_checkbox.isChecked())
        else:
            self._show_overlay_box = bool(self.show_overlay_box_checkbox.isChecked())
            self._mute_preview = bool(self.mute_preview_checkbox.isChecked())
            for cue in self.cues[self._active_cue_index:]:
                cue["show_overlay_box"] = self._show_overlay_box
                cue["mute_preview"] = self._mute_preview
        self.audio_output.setMuted(bool(self._mute_preview))
        if self._subtitle_active and self._active_overlay_rect is not None:
            self._set_overlay_rect(self._active_overlay_rect, update_inputs=False, persist=False)

    def _on_continuous_preview_changed(self, *_) -> None:
        self._continuous_preview = bool(self.continuous_preview_checkbox.isChecked())

    def _on_stop_preview_clicked(self) -> None:
        self._stop_preview()
        self.preview_status_label.setText("미리보기 멈춤")

    def _on_subtitle_rect_adjusted(self, rect: QRect) -> None:
        if not self._subtitle_active:
            return
        if self._active_cue_index is None:
            return
        rect_video = self._set_overlay_rect(rect, update_inputs=True, persist=True)
        self._persist_overlay_rect_to_config(rect_video)

    def _clamp_rect_to_bounds(self, rect: QRect, bounds: QRect) -> QRect:
        max_width = max(40, bounds.width())
        max_height = max(30, bounds.height())
        width = min(max(rect.width(), 20), max_width)
        height = min(max(rect.height(), 20), max_height)
        x = min(max(0, rect.x()), max(0, bounds.width() - width))
        y = min(max(0, rect.y()), max(0, bounds.height() - height))
        return QRect(x, y, width, height)

    def _update_coord_inputs(self, rect: QRect) -> None:
        self.overlay_coord_inputs["x1"].setText(str(rect.x()))
        self.overlay_coord_inputs["y1"].setText(str(rect.y()))
        self.overlay_coord_inputs["x2"].setText(str(rect.x() + rect.width()))
        self.overlay_coord_inputs["y2"].setText(str(rect.y() + rect.height()))

    def _rect_from_coord_inputs(self) -> QRect | None:
        try:
            x1 = int(self.overlay_coord_inputs["x1"].text())
            y1 = int(self.overlay_coord_inputs["y1"].text())
            x2 = int(self.overlay_coord_inputs["x2"].text())
            y2 = int(self.overlay_coord_inputs["y2"].text())
        except (ValueError, TypeError):
            return None
        width = max(20, x2 - x1)
        height = max(20, y2 - y1)
        return QRect(x1, y1, width, height)

    def _on_overlay_coord_changed(self) -> None:
        if self._coord_edit_blocked:
            return
        rect = self._rect_from_coord_inputs()
        if rect is None:
            return
        rect_video = self._set_overlay_rect(rect, update_inputs=False, persist=True)
        self._persist_overlay_rect_to_config(rect_video)

    def _on_font_control_changed(self, *_) -> None:
        style = self._style_from_controls()
        self._global_overlay_style.update(style)
        self._persist_overlay_style_to_config()
        start_index = self._active_cue_index if self._active_cue_index is not None else 0
        self._apply_style_to_cues(style, start_index)
        if self._active_cue_index is not None:
            self._show_preview_overlay(self.cues[self._active_cue_index])

    def _format_overlay_coords(self, rect: QRect | None) -> str:
        bounds = self._video_bounds()
        suffix = ""
        if bounds is not None:
            suffix = f" / 영상 {bounds.width()}x{bounds.height()}"
        if rect is None:
            return f"오버레이 위치 (x1, y1) (x2, y2){suffix}"
        x1, y1 = rect.x(), rect.y()
        x2 = x1 + rect.width()
        y2 = y1 + rect.height()
        return f"오버레이 위치 ({x1}, {y1}) ({x2}, {y2}){suffix}"

    def _cue_style(self, cue: dict) -> dict[str, Any]:
        return {
            "font_family": cue.get("font_family", self._global_overlay_style["font_family"]),
            "font_size": cue.get("font_size", self._global_overlay_style["font_size"]),
            "font_color": cue.get("font_color", self._global_overlay_style["font_color"]),
            "outline_enabled": cue.get("outline_enabled", self._global_overlay_style["outline_enabled"]),
            "outline_thickness": cue.get("outline_thickness", self._global_overlay_style["outline_thickness"]),
            "outline_color": cue.get("outline_color", self._global_overlay_style["outline_color"]),
        }

    def _cue_options(self, cue: dict) -> dict[str, Any]:
        return {
            "show_overlay_box": cue.get("show_overlay_box", self._show_overlay_box),
            "mute_preview": cue.get("mute_preview", self._mute_preview),
        }

    def _apply_preview_option_controls(self) -> None:
        widgets = [self.show_overlay_box_checkbox, self.mute_preview_checkbox]
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.show_overlay_box_checkbox.setChecked(bool(self._show_overlay_box))
            self.mute_preview_checkbox.setChecked(bool(self._mute_preview))
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def _overlay_meta_dict(self, rect: dict[str, int] | None = None) -> dict[str, Any]:
        rect_source = rect or self._global_overlay_rect_default or self._initial_overlay_rect_dict()
        overlay_rect = dict(rect_source)
        overlay_space = "video" if self._video_bounds() is not None else "widget"
        return {
            "font_family": self._global_overlay_style["font_family"],
            "font_size": self._global_overlay_style["font_size"],
            "font_color": self._global_overlay_style["font_color"],
            "outline_enabled": self._global_overlay_style["outline_enabled"],
            "outline_thickness": self._global_overlay_style["outline_thickness"],
            "outline_color": self._global_overlay_style["outline_color"],
            "overlay_rect": overlay_rect,
            "overlay_space": overlay_space,
            "show_overlay_box": self._show_overlay_box,
            "mute_preview": self._mute_preview,
        }

    def _cue_overlay_meta(self, cue: dict) -> dict[str, Any]:
        rect_source = cue.get("overlay_rect") or self._global_overlay_rect_default or self._initial_overlay_rect_dict()
        meta = self._cue_style(cue)
        meta["overlay_rect"] = dict(rect_source)
        meta["overlay_space"] = cue.get("overlay_space", "video")
        meta["show_overlay_box"] = cue.get("show_overlay_box", self._show_overlay_box)
        meta["mute_preview"] = cue.get("mute_preview", self._mute_preview)
        return meta

    def _style_from_controls(self) -> dict[str, Any]:
        font_family = self.font_combo.currentFont().family()
        font_size = self.font_size_spin.value()
        font_color = self._normalize_color(self.font_color_edit.text(), DEFAULT_FONT_COLOR)
        outline_enabled = bool(self.outline_checkbox.isChecked())
        outline_thickness = self.outline_spin.value()
        outline_color = self._normalize_color(self.outline_color_edit.text(), DEFAULT_OUTLINE_COLOR)
        self._decorate_color_input(self.font_color_edit)
        self._decorate_color_input(self.outline_color_edit)
        return {
            "font_family": font_family,
            "font_size": font_size,
            "font_color": font_color,
            "outline_enabled": outline_enabled,
            "outline_thickness": outline_thickness,
            "outline_color": outline_color,
        }

    def _apply_font_controls(self, style: dict[str, Any]) -> None:
        widgets = [
            self.font_combo,
            self.font_size_spin,
            self.font_color_edit,
            self.outline_checkbox,
            self.outline_spin,
            self.outline_color_edit,
        ]
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.font_combo.setCurrentFont(QFont(style["font_family"]))
            self.font_size_spin.setValue(style["font_size"])
            self.font_color_edit.setText(style["font_color"])
            self._decorate_color_input(self.font_color_edit)
            self.outline_checkbox.setChecked(bool(style["outline_enabled"]))
            self.outline_spin.setValue(style["outline_thickness"])
            self.outline_color_edit.setText(style["outline_color"])
            self._decorate_color_input(self.outline_color_edit)
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def _apply_style_to_cues(self, style: dict[str, Any], start_index: int) -> None:
        for cue in self.cues[start_index:]:
            cue.update(style)

    def _choose_color(self, target: QLineEdit, fallback: str) -> None:
        current = QColor(target.text())
        if not current.isValid():
            current = QColor(fallback)
        color = QColorDialog.getColor(current, self, "색상 선택")
        if not color.isValid():
            return
        target.setText(color.name())
        self._decorate_color_input(target)
        self._on_font_control_changed()

    def _decorate_color_input(self, edit: QLineEdit) -> None:
        color = QColor(edit.text())
        if not color.isValid():
            color = QColor(DEFAULT_FONT_COLOR)
        text_color = "#000000" if color.lightnessF() > 0.5 else "#ffffff"
        edit.setStyleSheet(f"background-color: {color.name()}; color: {text_color};")

    def _normalize_color(self, value: str, fallback: str) -> str:
        color = QColor(value)
        return color.name() if color.isValid() else fallback

    def _initial_overlay_rect_dict(self) -> dict[str, int]:
        bounds = self._video_bounds()
        if bounds is None:
            x = DEFAULT_OVERLAY_MARGIN
            y = DEFAULT_OVERLAY_MARGIN
            width = 640
            height = DEFAULT_OVERLAY_HEIGHT
            return {"x": x, "y": y, "width": width, "height": height}
        width = max(40, bounds.width() - DEFAULT_OVERLAY_MARGIN * 2)
        height = min(DEFAULT_OVERLAY_HEIGHT, max(30, bounds.height() - DEFAULT_OVERLAY_MARGIN * 2))
        x = DEFAULT_OVERLAY_MARGIN
        y = max(0, bounds.height() - height - DEFAULT_OVERLAY_MARGIN)
        return {"x": x, "y": y, "width": width, "height": height}

    def eventFilter(self, obj, event) -> bool:
        if obj is self.video_container and event.type() == QEvent.Resize:
            self._position_overlay()
        if obj is self.preview_border:
            if event.type() == QEvent.MouseMove and not self._dragging_overlay:
                self._update_preview_border_cursor(self._event_position(event))
                return False
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._dragging_overlay = True
                self._drag_mode = self._overlay_drag_mode(self._event_position(event))
                self._drag_start_pos = self.preview_border.mapToParent(self._event_position(event))
                self._drag_start_rect = QRect(self.preview_border.geometry())
                self.preview_border.setCursor(Qt.ClosedHandCursor)
                return True
            if event.type() == QEvent.MouseMove and self._dragging_overlay:
                pos = self.preview_border.mapToParent(self._event_position(event))
                delta = pos - self._drag_start_pos
                widget_rect = self._apply_drag_delta(QRect(self._drag_start_rect), delta)
                video_rect = self.video_widget.map_widget_rect_to_video(widget_rect)
                if video_rect is not None:
                    self._set_overlay_rect(video_rect, update_inputs=True, persist=False)
                return True
            if event.type() == QEvent.MouseButtonRelease and self._dragging_overlay:
                self._dragging_overlay = False
                self._drag_mode = None
                self.preview_border.setCursor(Qt.OpenHandCursor)
                if self._active_overlay_rect is not None:
                    self._apply_overlay_rect_to_following(_rect_to_dict(self._active_overlay_rect))
                    self._persist_overlay_rect_to_config(self._active_overlay_rect)
                return True
        return super().eventFilter(obj, event)

    def _overlay_drag_mode(self, pos) -> str:
        handle = 8
        w = self.preview_border.width()
        h = self.preview_border.height()
        near_left = pos.x() <= handle
        near_right = pos.x() >= w - handle
        near_top = pos.y() <= handle
        near_bottom = pos.y() >= h - handle
        if near_left and near_top:
            return "resize_tl"
        if near_right and near_top:
            return "resize_tr"
        if near_left and near_bottom:
            return "resize_bl"
        if near_right and near_bottom:
            return "resize_br"
        if near_left:
            return "resize_l"
        if near_right:
            return "resize_r"
        if near_top:
            return "resize_t"
        if near_bottom:
            return "resize_b"
        return "move"

    def _update_preview_border_cursor(self, pos) -> None:
        mode = self._overlay_drag_mode(pos)
        if mode in {"resize_tl", "resize_br"}:
            self.preview_border.setCursor(Qt.SizeFDiagCursor)
        elif mode in {"resize_tr", "resize_bl"}:
            self.preview_border.setCursor(Qt.SizeBDiagCursor)
        elif mode in {"resize_l", "resize_r"}:
            self.preview_border.setCursor(Qt.SizeHorCursor)
        elif mode in {"resize_t", "resize_b"}:
            self.preview_border.setCursor(Qt.SizeVerCursor)
        else:
            self.preview_border.setCursor(Qt.OpenHandCursor)

    def _apply_drag_delta(self, rect: QRect, delta) -> QRect:
        target = self.video_widget.video_target_rect()
        if target.isEmpty():
            target = self.video_widget.rect()
        min_w = 40
        min_h = 30
        mode = self._drag_mode or "move"
        if mode == "move":
            rect.translate(delta)
            rect.moveLeft(min(max(rect.left(), target.left()), target.right() - rect.width()))
            rect.moveTop(min(max(rect.top(), target.top()), target.bottom() - rect.height()))
            return rect

        if "l" in mode:
            rect.setLeft(rect.left() + delta.x())
        if "r" in mode:
            rect.setRight(rect.right() + delta.x())
        if "t" in mode:
            rect.setTop(rect.top() + delta.y())
        if "b" in mode:
            rect.setBottom(rect.bottom() + delta.y())
        rect = rect.normalized()

        if rect.width() < min_w:
            if "l" in mode:
                rect.setLeft(rect.right() - min_w)
            else:
                rect.setRight(rect.left() + min_w)
        if rect.height() < min_h:
            if "t" in mode:
                rect.setTop(rect.bottom() - min_h)
            else:
                rect.setBottom(rect.top() + min_h)

        rect.setLeft(max(rect.left(), target.left()))
        rect.setTop(max(rect.top(), target.top()))
        rect.setRight(min(rect.right(), target.right()))
        rect.setBottom(min(rect.bottom(), target.bottom()))

        if rect.width() < min_w:
            rect.setRight(min(target.right(), rect.left() + min_w))
        if rect.height() < min_h:
            rect.setBottom(min(target.bottom(), rect.top() + min_h))

        return rect

    def _event_position(self, event):
        if hasattr(event, "position"):
            return event.position().toPoint()
        return event.pos()

    def _log(self, message: str) -> None:
        self.log_view.append(message)
        self.status_bar.showMessage(message, 4000)

    def _set_progress(self, percent: int, message: str | None = None) -> None:
        self.progress_bar.setValue(percent)
        if message:
            self.status_bar.showMessage(message, 4000)

    def _browse_video(self) -> None:
        start_dir = self.config.last_media_dir or str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "영상 선택",
            start_dir,
            "Video Files (*.mp4 *.mkv *.mov *.avi);;모든 파일 (*)",
        )
        if not path:
            return
        self.config.last_media_dir = str(Path(path).parent)
        save_config(self.config)
        self.media_path = path
        self._video_bounds_hint = None
        self._global_overlay_rect_default = None
        self._active_overlay_rect = None
        self._pending_overlay_rect_apply = True
        try:
            _, ffprobe = self._find_ffmpeg_tools()
            if ffprobe:
                info = self._probe_media(path, ffprobe)
                width = int(info.get("width") or 0)
                height = int(info.get("height") or 0)
                if width > 0 and height > 0:
                    self._video_bounds_hint = QRect(0, 0, width, height)
        except Exception:
            self._video_bounds_hint = None
        self.video_path_edit.setText(path)
        self.player.setSource(QUrl.fromLocalFile(path))
        self._restore_overlay_rect_from_config_if_needed()
        self._log(f"영상 선택됨: {Path(path).name}")
        self._set_progress(5, "ASR 준비 중...")
        self._asr_start_time = time.perf_counter()
        threading.Thread(target=partial(self._run_asr, path), daemon=True).start()

    def _restore_overlay_rect_from_config_if_needed(self) -> None:
        bounds = self._video_bounds()
        if bounds is None:
            return
        if self._global_overlay_rect_default is None:
            if self._global_overlay_rect_norm:
                rect_dict = self._norm_to_video_rect_dict(self._global_overlay_rect_norm, bounds)
                self._global_overlay_rect_default = rect_dict or self._initial_overlay_rect_dict()
            else:
                self._global_overlay_rect_default = self._initial_overlay_rect_dict()
        if self._active_cue_index is not None:
            return
        try:
            current = {k: int(self.overlay_coord_inputs[k].text()) for k in ("x1", "y1", "x2", "y2")}
            all_zero = all(v == 0 for v in current.values())
        except Exception:
            all_zero = True
        if not all_zero and self._active_overlay_rect is not None:
            return
        rect_dict = self._global_overlay_rect_default or self._initial_overlay_rect_dict()
        rect = _dict_to_rect(rect_dict, bounds)
        self._active_overlay_rect = rect
        self._set_overlay_rect(rect, update_inputs=True, persist=False)

    def _find_ffmpeg_tools(self) -> tuple[str | None, str | None]:
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        return ffmpeg, ffprobe

    def _probe_media(self, path: str, ffprobe_path: str) -> dict[str, Any]:
        cmd = [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height:format=duration",
            "-of",
            "json",
            path,
        ]
        output = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace")
        data = json.loads(output or "{}")
        streams = data.get("streams") or []
        stream = streams[0] if streams else {}
        fmt = data.get("format") or {}
        duration = 0.0
        try:
            duration = float(fmt.get("duration") or 0.0)
        except Exception:
            duration = 0.0
        return {
            "codec_name": str(stream.get("codec_name") or ""),
            "width": int(stream.get("width") or 0),
            "height": int(stream.get("height") or 0),
            "duration": duration,
        }

    def _build_scripted_output_path(self, input_path: str) -> str:
        src = Path(input_path)
        base = src.with_name(f"{src.stem}_Scripted{src.suffix}")
        if not base.exists():
            return str(base)
        for idx in range(2, 1000):
            candidate = src.with_name(f"{src.stem}_Scripted_{idx}{src.suffix}")
            if not candidate.exists():
                return str(candidate)
        return str(base)

    def _ffmpeg_filter_escape_path(self, path: str) -> str:
        value = Path(path).resolve().as_posix()
        value = value.replace(":", "\\:")
        value = value.replace("'", "\\'")
        return value

    def _ass_time(self, seconds: float) -> str:
        total_cs = max(0, int(round(seconds * 100.0)))
        hours, remainder = divmod(total_cs, 360000)
        minutes, remainder = divmod(remainder, 6000)
        secs, cs = divmod(remainder, 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"

    def _ass_color(self, value: str, fallback: str = "&H00FFFFFF") -> str:
        text = (value or "").strip()
        if text.startswith("#") and len(text) == 7:
            try:
                r = int(text[1:3], 16)
                g = int(text[3:5], 16)
                b = int(text[5:7], 16)
                return f"&H00{b:02X}{g:02X}{r:02X}"
            except Exception:
                return fallback
        return fallback

    def _ass_escape_text(self, value: str) -> str:
        text = (value or "").replace("\r", "").strip()
        text = text.replace("\\", "\\\\")
        text = text.replace("{", "\\{").replace("}", "\\}")
        return text

    def _ass_font_size_from_qt(self, font_family: str, point_size: int) -> int:
        qt_font = QFont(font_family, max(1, max(8, int(point_size))))
        metrics = QFontMetrics(qt_font)
        size = int(round(metrics.ascent() + metrics.descent()))
        return max(1, size)

    def _build_ass_script(self, bounds: QRect) -> str:
        styles: dict[tuple, str] = {}
        style_lines: list[str] = []
        dialogue_lines: list[str] = []
        padding = 8
        def style_name_for(style: dict[str, Any]) -> str:
            font_size_base = int(style.get("font_size") or DEFAULT_FONT_SIZE)
            font_size_ass = self._ass_font_size_from_qt(
                style.get("font_family") or DEFAULT_FONT_FAMILY,
                font_size_base,
            )
            key = (
                style.get("font_family") or DEFAULT_FONT_FAMILY,
                font_size_ass,
                str(style.get("font_color") or DEFAULT_FONT_COLOR),
                bool(style.get("outline_enabled", DEFAULT_OUTLINE_ENABLED)),
                int(style.get("outline_thickness") or DEFAULT_OUTLINE_THICKNESS),
                str(style.get("outline_color") or DEFAULT_OUTLINE_COLOR),
            )
            if key in styles:
                return styles[key]
            name = f"S{len(styles) + 1}"
            styles[key] = name
            font_family, font_size, font_color, outline_enabled, outline_thickness, outline_color = key
            outline = outline_thickness if outline_enabled else 0
            style_lines.append(
                "Style: "
                f"{name},{font_family},{font_size},"
                f"{self._ass_color(font_color)},&H00000000,{self._ass_color(outline_color,'&H00000000')},&H00000000,"
                "0,0,0,0,100,100,0,0,1,"
                f"{outline},0,2,0,0,0,1"
            )
            return name

        for cue in self.cues:
            text = (cue.get("text") or "").strip()
            if not text:
                continue
            start = float(cue.get("source_start") or 0.0)
            end = float(cue.get("source_end") or start)
            if end <= start:
                continue
            style = self._cue_style(cue)
            rect_dict = cue.get("overlay_rect") or self._global_overlay_rect_default or self._initial_overlay_rect_dict()
            rect = _dict_to_rect(rect_dict, bounds)

            font_family = style.get("font_family") or DEFAULT_FONT_FAMILY
            font_size_base = int(style.get("font_size") or DEFAULT_FONT_SIZE)
            font_size_ass = self._ass_font_size_from_qt(font_family, font_size_base)
            font = QFont(font_family)
            font.setPixelSize(font_size_ass)
            metrics = QFontMetrics(font)

            y2 = rect.y() + rect.height()
            text_rect = rect.adjusted(padding, padding, -padding, -padding)
            max_width = max(20, text_rect.width())
            lines = self.video_widget._wrap_lines_balanced(text, font, metrics, max_width)
            needed_height = len(lines) * metrics.lineSpacing() + padding * 2
            if needed_height > rect.height():
                new_y1 = max(0, y2 - needed_height)
                rect = QRect(rect.x(), new_y1, rect.width(), y2 - new_y1).intersected(bounds)
                text_rect = rect.adjusted(padding, padding, -padding, -padding)

            clip_x1 = int(text_rect.x())
            clip_y1 = int(text_rect.y())
            clip_x2 = int(text_rect.x() + text_rect.width())
            clip_y2 = int(text_rect.y() + text_rect.height())
            pos_x = int(round(text_rect.x() + text_rect.width() / 2))
            pos_y = int(round(rect.y() + rect.height() - padding))

            ass_text = "\\N".join(self._ass_escape_text(line) for line in lines)
            override = f"{{\\an2\\pos({pos_x},{pos_y})\\clip({clip_x1},{clip_y1},{clip_x2},{clip_y2})}}"
            style_name = style_name_for(style)
            dialogue_lines.append(
                f"Dialogue: 0,{self._ass_time(start)},{self._ass_time(end)},{style_name},,0,0,0,,{override}{ass_text}"
            )

        script_info = "\n".join(
            [
                "[Script Info]",
                "ScriptType: v4.00+",
                f"PlayResX: {bounds.width()}",
                f"PlayResY: {bounds.height()}",
                "WrapStyle: 2",
                "ScaledBorderAndShadow: yes",
                "",
                "[V4+ Styles]",
                "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
                "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
                "Alignment,MarginL,MarginR,MarginV,Encoding",
            ]
        )
        events_header = "\n".join(
            [
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            ]
        )
        return "\n".join([script_info, *style_lines, events_header, *dialogue_lines, ""])

    def _select_video_encoder(self, codec_name: str) -> str:
        name = (codec_name or "").lower()
        if name in {"hevc", "h265"}:
            return "libx265"
        if name in {"vp9"}:
            return "libvpx-vp9"
        if name in {"vp8"}:
            return "libvpx"
        return "libx264"

    def _export_video(self) -> None:
        if not self.media_path:
            self._log("?ìƒ??? íƒ?˜ì? ?Šì•˜?µë‹ˆ??")
            return
        if not self.cues:
            self._log("?ë§‰ ëª©ë¡?? ?ŠìŠµ?ˆë‹¤.")
            return
        ffmpeg, ffprobe = self._find_ffmpeg_tools()
        if not ffmpeg or not ffprobe:
            self._log("ffmpeg/ffprobeë¥? ì°¾ì? ìˆ˜ ?ŠìŠµ?ˆë‹¤. PATHë¥? í™œì¸?´ì£¼ì„¸??")
            return
        try:
            info = self._probe_media(self.media_path, ffprobe)
        except Exception as exc:
            self._log(f"ffprobe ?¤íŒ¨: {exc}")
            return
        bounds = self._video_bounds()
        if bounds is None:
            width = int(info.get("width") or 0)
            height = int(info.get("height") or 0)
            if width <= 0 or height <= 0:
                self._log("??ìƒ í¬ê¸°ë¥? ì•Œ ìˆ˜ ?Šì–´ Exportë¥? ? í–‰? í•  ìˆ˜ ?ŠìŠµ?ˆë‹¤.")
                return
            bounds = QRect(0, 0, width, height)
        ass_text = self._build_ass_script(bounds)
        ass_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ass", mode="w", encoding="utf-8")
        ass_path = ass_file.name
        try:
            ass_file.write(ass_text)
            ass_file.close()
        except Exception:
            try:
                ass_file.close()
            except Exception:
                pass
            raise
        output_path = self._build_scripted_output_path(self.media_path)
        duration = float(info.get("duration") or 0.0)
        encoder = self._select_video_encoder(str(info.get("codec_name") or ""))
        self.export_btn.setEnabled(False)
        self._set_progress(0, "Export ì¤€ë¹? ì¤‘...")
        threading.Thread(
            target=self._run_export,
            args=(ffmpeg, ass_path, output_path, duration, encoder),
            daemon=True,
        ).start()

    def _run_export(
        self,
        ffmpeg_path: str,
        ass_path: str,
        output_path: str,
        duration_seconds: float,
        encoder: str,
    ) -> None:
        try:
            escaped_ass = self._ffmpeg_filter_escape_path(ass_path)
            filter_arg = f"subtitles=filename='{escaped_ass}':charenc=UTF-8"
            cmd = [
                ffmpeg_path,
                "-y",
                "-hide_banner",
                "-nostats",
                "-i",
                self.media_path,
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-vf",
                filter_arg,
                "-c:v",
                encoder,
                "-preset",
                "medium",
                "-crf",
                "18",
                "-c:a",
                "copy",
                "-progress",
                "pipe:1",
                output_path,
            ]
            self._schedule_ui(lambda: self._set_progress(0, "Export ì§„í–‰ ì¤‘..."))
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            last_out_time_ms: int | None = None
            last_line = ""
            if proc.stdout is not None:
                for raw in proc.stdout:
                    line = (raw or "").strip()
                    if line:
                        last_line = line
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    if key == "out_time_ms":
                        try:
                            last_out_time_ms = int(value)
                        except Exception:
                            last_out_time_ms = None
                    if duration_seconds > 0 and last_out_time_ms is not None:
                        percent = int(
                            max(
                                0.0,
                                min(
                                    99.0,
                                    (last_out_time_ms / 1_000_000.0) / duration_seconds * 100.0,
                                ),
                            )
                        )
                        self._schedule_ui(lambda p=percent: self._set_progress(p, "Export ì§„í–‰ ì¤‘..."))
            rc = proc.wait()
            if rc == 0:
                self._schedule_ui(lambda: self._set_progress(100, "Export ?„ë£Œ"))
                self._schedule_ui(lambda: self._log(f"Export ?„ë£Œ: {Path(output_path).name}"))
            else:
                self._schedule_ui(lambda: self._set_progress(0, "Export ?¤íŒ¨"))
                self._schedule_ui(lambda: self._log(f"Export ?¤íŒ¨ (code={rc}): {last_line}"))
        except Exception as exc:
            self._schedule_ui(lambda exc=exc: self._log(f"Export ?¤íŒ¨: {exc}"))
            self._schedule_ui(lambda: self._set_progress(0, "Export ?¤íŒ¨"))
        finally:
            try:
                Path(ass_path).unlink(missing_ok=True)
            except Exception:
                pass
            self._schedule_ui(lambda: self.export_btn.setEnabled(True))

    def _run_asr(self, path: str) -> None:
        self._schedule_ui(lambda: self._set_progress(15, "ASR 실행 중..."))
        self._schedule_ui(lambda: self._log("ASR을 시작합니다."))
        try:
            segments = transcribe(path)
        except Exception as exc:
            self._schedule_ui(lambda exc=exc: self._log(f"ASR 실패: {exc}"))
            self._schedule_ui(lambda: self._set_progress(0, "ASR 실패"))
            return
        self.raw_segments = segments
        self._schedule_ui(lambda: self._set_progress(60, "ASR 결과를 처리하는 중입니다..."))
        self._schedule_ui(self._refresh_from_asr)

    def _refresh_from_asr(self) -> None:
        self._set_progress(80, "자막을 재구성 중입니다...")
        self._reflow_cues()
        self._restore_overlay_rect_from_config_if_needed()
        self._set_progress(100, "ASR 완료, 자막 목록이 갱신되었습니다.")
        self._log("ASR 완료, 자막 목록이 갱신되었습니다.")
        if self._asr_start_time is not None:
            duration = time.perf_counter() - self._asr_start_time
            self._log(f"영상 선택부터 ASR 완료까지 {duration:.1f}초가 소요되었습니다.")
            self._asr_start_time = None


    def _schedule_ui(self, callback: Callable[[], None]) -> None:
        self._invoke_signal.emit(callback)

    def _run_ui_callback(self, callback: Callable[[], None]) -> None:
        callback()

    def _add_manual_cue(self) -> None:
        start_input = self.start_edit.text().strip()
        end_input = self.end_edit.text().strip()
        text = self.text_edit.text().strip()
        start_seconds = parse_time_input(start_input)
        end_seconds = parse_time_input(end_input)
        if start_seconds is None or end_seconds is None or not text:
            self.status_bar.showMessage("시간 또는 텍스트가 유효하지 않습니다.", 3000)
            return
        cue_entry = build_cue_entry(
            start_seconds,
            end_seconds,
            text,
            offset=self.start_offset,
            overlay_meta=self._overlay_meta_dict(),
        )
        self.cues.append(cue_entry)
        self._refresh_cue_table()
        self._refresh_preview()
        self.remove_btn.setEnabled(bool(self.cues))
        self.text_edit.clear()
        self.status_bar.showMessage("수동 자막이 추가되었습니다.", 2000)
        self._log(f"수동 자막 입력: {text[:60]}")

    def _refresh_cue_table(self) -> None:
        self.cue_table.blockSignals(True)
        self.cue_table.setRowCount(len(self.cues))
        for row, cue in enumerate(self.cues):
            start_item = QTableWidgetItem(cue["start_label"])
            start_item.setFlags(start_item.flags() & ~Qt.ItemIsEditable)
            self.cue_table.setItem(row, 0, start_item)
            end_item = QTableWidgetItem(cue["end_label"])
            end_item.setFlags(end_item.flags() & ~Qt.ItemIsEditable)
            self.cue_table.setItem(row, 1, end_item)
            duration_item = QTableWidgetItem(cue["duration_label"])
            duration_item.setFlags(duration_item.flags() & ~Qt.ItemIsEditable)
            self.cue_table.setItem(row, 2, duration_item)
            text_item = QTableWidgetItem(cue["text"])
            text_item.setFlags(text_item.flags() | Qt.ItemIsEditable)
            self.cue_table.setItem(row, 3, text_item)
            char_item = QTableWidgetItem(str(cue["char_count"]))
            char_item.setFlags(char_item.flags() & ~Qt.ItemIsEditable)
            self.cue_table.setItem(row, 4, char_item)
        self.cue_table.blockSignals(False)
        self.remove_btn.setEnabled(bool(self.cues))

    def _refresh_preview(self) -> None:
        count = len(self.cues)
        self.preview_status_label.setText(
            f"{count}개의 자막 목록이 준비되었습니다. 행을 클릭하면 해당 구간을 재생합니다."
        )

    def _on_cue_selected(self, row: int, column: int) -> None:
        if not (0 <= row < len(self.cues)):
            return
        if row == self._active_cue_index and self._is_preview_playing():
            return
        self._active_cue_index = row
        cue = self.cues[row]
        self._apply_font_controls(self._cue_style(cue))
        self.status_bar.showMessage(
            f"{cue['start_label']} - {cue['end_label']} ({cue['duration_label']}): {cue['text']}",
            3000,
        )
        self.preview_status_label.setText(
            f"재생 중: {cue['start_label']} - {cue['end_label']} ({cue['text'][:40]}...)"
        )
        self._play_cue_preview(cue)

    def _is_preview_playing(self) -> bool:
        if not self.preview_timer.isActive():
            return False
        try:
            return self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        except Exception:
            try:
                return self.player.state() == QMediaPlayer.PlayingState
            except Exception:
                return True

    def _play_cue_preview_by_index(self, row: int) -> None:
        if not (0 <= row < len(self.cues)):
            return
        cue = self.cues[row]
        self._active_cue_index = row
        try:
            self.cue_table.blockSignals(True)
            self.cue_table.selectRow(row)
            self.cue_table.setCurrentCell(row, 0)
        finally:
            self.cue_table.blockSignals(False)
        self._apply_font_controls(self._cue_style(cue))
        self.preview_status_label.setText(
            f"재생 중 {cue['start_label']} - {cue['end_label']} ({cue['text'][:40]}...)"
        )
        self._play_cue_preview(cue)

    def _play_cue_preview(self, cue: dict) -> None:
        if not self.media_path:
            self.preview_status_label.setText("영상을 선택하지 않았습니다.")
            return
        target_url = QUrl.fromLocalFile(self.media_path)
        if self.player.source() != target_url:
            self.player.setSource(target_url)
        self._stop_preview()
        self.audio_output.setMuted(bool(self._cue_options(cue)["mute_preview"]))
        position_ms = int(cue["source_start"] * 1000)
        duration_ms = max(100, int((cue["source_end"] - cue["source_start"]) * 1000))
        self.player.setPosition(position_ms)
        self.player.play()
        self.preview_timer.start(duration_ms + 200)
        self._show_preview_overlay(cue)

    def _pause_preview(self) -> None:
        self.player.pause()
        if self._continuous_preview and self._active_cue_index is not None:
            next_row = self._active_cue_index + 1
            if next_row < len(self.cues):
                self._play_cue_preview_by_index(next_row)
                return
        self.preview_status_label.setText("재생이 완료되었습니다.")

    def _stop_preview(self) -> None:
        self.preview_timer.stop()
        self.player.pause()

    def _offset_changed(self) -> None:
        if not hasattr(self, "offset_edit") or self.offset_edit is None:
            return
        text = self.offset_edit.text().strip()
        try:
            value = float(text) if text else 0.0
        except ValueError:
            value = self.config.start_offset
        value = max(0.0, value)
        self.start_offset = value
        self.config.start_offset = value
        save_config(self.config)
        self.offset_edit.setText(f"{value:.3f}")
        self._log(f"시작 오프셋을 {value:.3f}초로 설정했습니다.")
        if self.cues:
            self._stop_preview()
            self._reapply_offset()
            self.status_bar.showMessage("시작 오프셋을 적용해 자막을 재구성했습니다.", 3000)

    def _reapply_offset(self) -> None:
        if not self.cues:
            return
        previous_cues = list(self.cues)
        self.cues = [
            build_cue_entry(
                cue["source_start"],
                cue["source_end"],
                cue["text"],
                offset=self.start_offset,
                overlay_meta=self._cue_overlay_meta(cue),
            )
            for cue in previous_cues
        ]
        self._refresh_cue_table()
        self._refresh_preview()

    def _on_cue_cell_changed(self, row: int, column: int) -> None:
        if column != 3 or not (0 <= row < len(self.cues)):
            return
        item = self.cue_table.item(row, column)
        if item is None:
            return
        new_text = item.text()
        cue = self.cues[row]
        cue["text"] = new_text
        cue["char_count"] = len(new_text)
        self.cue_table.blockSignals(True)
        char_item = QTableWidgetItem(str(cue["char_count"]))
        char_item.setFlags(char_item.flags() & ~Qt.ItemIsEditable)
        self.cue_table.setItem(row, 4, char_item)
        self.cue_table.blockSignals(False)
        self._refresh_preview()
        self._log(f"자막 수정: {new_text[:60]}")

    def _remove_selected(self) -> None:
        selected_rows = {
            index.row() for index in self.cue_table.selectionModel().selectedRows()
        }
        if not selected_rows:
            return
        for row in sorted(selected_rows, reverse=True):
            if 0 <= row < len(self.cues):
                self.cues.pop(row)
        self._refresh_cue_table()
        self._refresh_preview()
        self._stop_preview()
        self.status_bar.showMessage("선택된 자막을 삭제했습니다.", 2000)
        self.remove_btn.setEnabled(bool(self.cues))
        self._log("선택된 자막을 삭제했습니다.")

    def _char_length_changed(self) -> None:
        text = self.char_length_edit.text().strip()
        try:
            value = int(text)
        except ValueError:
            self.char_length_edit.setText(str(self.config.subtitle_char_length))
            return
        value = max(10, min(value, 200))
        self.config.subtitle_char_length = value
        save_config(self.config)
        self.char_length_edit.setText(str(value))
        self._log(f"최대 문자 길이를 {value}자로 변경했습니다.")
        if self.raw_segments:
            self._reflow_cues()
            self.status_bar.showMessage("최대 문자 길이에 맞춰 자막을 재구성했습니다.", 3000)

    def _reflow_cues(self) -> None:
        max_chars = self.config.subtitle_char_length
        self.cues = [
            build_cue_entry(
                cue["start"],
                cue["end"],
                cue["text"],
                offset=self.start_offset,
                overlay_meta=self._overlay_meta_dict(),
            )
            for cue in split_segments(self.raw_segments, max_chars)
        ]
        self._refresh_cue_table()
        self._refresh_preview()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        new_width = event.size().width()
        new_height = event.size().height()
        if (
            self.config.window_width != new_width
            or self.config.window_height != new_height
        ):
            self.config.window_width = new_width
            self.config.window_height = new_height
            save_config(self.config)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = QApplication(sys.argv)
    window = SubtitleCreatorMainWindow()
    window.show()
    sys.exit(app.exec())



if __name__ == "__main__":
    main()
