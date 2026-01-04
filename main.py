from __future__ import annotations

import logging
import sys
import threading
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, List

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from PySide6.QtGui import QDoubleValidator, QIntValidator, QResizeEvent
from PySide6.QtCore import QEvent, QTimer, QUrl, Qt, Signal

from PySide6.QtWidgets import (
    QApplication,
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
    QHeaderView,
    QSizePolicy,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget

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
) -> dict:
    """Creates a cue entry storing both source timing and adjusted labels."""
    source_start = max(0.0, source_start)
    source_end = max(source_start, source_end)
    adjusted_start = max(0.0, source_start - offset)
    adjusted_end = max(adjusted_start, source_end - offset)
    duration = max(0.0, adjusted_end - adjusted_start)
    return {
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

        for chunk in chunks:
            chunk_text = " ".join(word["text"] for word in chunk)
            chunk_start = float(chunk[0]["start"])
            chunk_end = float(chunk[-1]["end"])
            split_result.append({"start": chunk_start, "end": chunk_end, "text": chunk_text})
    return split_result


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
        self._build_ui()

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

        offset_row = QHBoxLayout()
        self.offset_edit = QLineEdit(f"{self.start_offset:.3f}")
        offset_validator = QDoubleValidator(0.0, 3600.0, 3)
        offset_validator.setNotation(QDoubleValidator.StandardNotation)
        self.offset_edit.setValidator(offset_validator)
        self.offset_edit.editingFinished.connect(self._offset_changed)
        offset_row.addWidget(QLabel("시작 오프셋 (초)"))
        offset_row.addWidget(self.offset_edit)
        layout.addLayout(offset_row)

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
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(240)
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_widget, 2)
        self.preview_overlay = QLabel(self.video_widget)
        self.preview_overlay.setVisible(False)
        self.preview_overlay.setWordWrap(True)
        self.preview_overlay.setAlignment(Qt.AlignCenter | Qt.AlignBottom)
        self.preview_overlay.setStyleSheet(
            "color: white; background-color: rgba(0, 0, 0, 0.65); padding: 8px 12px; border-radius: 6px;"
        )
        self.preview_overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.video_widget.installEventFilter(self)
        self.preview_status_label = QLabel("자막을 선택하면 영상과 음성이 재생됩니다.")
        layout.addWidget(self.preview_status_label)
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._pause_preview)

        layout.addWidget(QLabel("로그"))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, 1)

        self.setCentralWidget(root)
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _position_overlay(self) -> None:
        if not self.preview_overlay.isVisible():
            return
        margin = 16
        max_height = 120
        parent_height = self.video_widget.height()
        height = min(max_height, parent_height - margin * 2)
        if height < 40:
            height = parent_height - margin
        self.preview_overlay.setGeometry(
            margin,
            max(margin, parent_height - height - margin),
            max(100, self.video_widget.width() - margin * 2),
            max(20, height),
        )

    def _show_preview_overlay(self, text: str) -> None:
        snippet = text.strip()
        if not snippet:
            self.preview_overlay.hide()
            return
        if len(snippet) > 120:
            snippet = snippet[:120].rstrip() + "…"
        self.preview_overlay.setText(snippet)
        self.preview_overlay.setVisible(True)
        self._position_overlay()

    def _hide_preview_overlay(self) -> None:
        self.preview_overlay.hide()

    def eventFilter(self, obj, event) -> bool:
        if obj is self.video_widget and event.type() == QEvent.Resize:
            self._position_overlay()
        return super().eventFilter(obj, event)

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
        self.video_path_edit.setText(path)
        self.player.setSource(QUrl.fromLocalFile(path))
        self._log(f"영상 선택됨: {Path(path).name}")
        self._set_progress(5, "ASR 준비 중...")
        self._asr_start_time = time.perf_counter()
        threading.Thread(target=partial(self._run_asr, path), daemon=True).start()

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
        cue_entry = build_cue_entry(start_seconds, end_seconds, text, offset=self.start_offset)
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
        cue = self.cues[row]
        self.status_bar.showMessage(
            f"{cue['start_label']} - {cue['end_label']} ({cue['duration_label']}): {cue['text']}",
            3000,
        )
        self.preview_status_label.setText(
            f"재생 중: {cue['start_label']} - {cue['end_label']} ({cue['text'][:40]}...)"
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
        position_ms = int(cue["source_start"] * 1000)
        duration_ms = max(100, int((cue["source_end"] - cue["source_start"]) * 1000))
        self.player.setPosition(position_ms)
        self.player.play()
        self.preview_timer.start(duration_ms + 200)
        self._show_preview_overlay(cue["text"])

    def _pause_preview(self) -> None:
        self.player.pause()
        self.preview_status_label.setText("재생이 완료되었습니다.")
        self._hide_preview_overlay()

    def _stop_preview(self) -> None:
        self.preview_timer.stop()
        self.player.pause()
        self._hide_preview_overlay()

    def _offset_changed(self) -> None:
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
            build_cue_entry(cue["start"], cue["end"], cue["text"], offset=self.start_offset)
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
