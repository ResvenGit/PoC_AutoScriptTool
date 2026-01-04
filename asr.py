from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

_model = None
logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from faster_whisper import WhisperModel


def _ensure_model() -> "WhisperModel":
    global _model
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "faster-whisper가 설치되어 있지 않습니다. `pip install faster-whisper`하고 "
            "다시 실행해 주세요."
        ) from exc

    if _model is None:
        model_dir = Path(__file__).resolve().parent / ".whisper_models"
        _model = WhisperModel(
            "medium",
            device="cuda",
            compute_type="float16",
            download_root=str(model_dir),
        )
    return _model


def _extract_word_text(word: Any) -> str:
    text = getattr(word, "text", None)
    if text:
        return str(text).strip()
    alt = getattr(word, "word", None)
    return str(alt).strip() if alt else ""


def transcribe(video_path: str) -> List[Dict[str, float]]:
    logger.info("ASR: 모델 준비 중...")
    model = _ensure_model()
    logger.info("ASR: 음성 변환 시작 중 %s", video_path)
    segments_iter, _ = model.transcribe(
        video_path,
        beam_size=3,
        word_timestamps=True,
    )
    segments = list(segments_iter)
    logger.info("ASR: 음성 변환 완료 (%d segments)", len(segments))
    result: List[Dict[str, float]] = []
    for segment in segments:
        words = []
        for word in getattr(segment, "words", []):
            word_text = _extract_word_text(word)
            if not word_text:
                continue
            words.append(
                {
                    "text": word_text,
                    "start": float(word.start),
                    "end": float(word.end),
                }
            )
        if not words:
            continue
        text = " ".join(word["text"] for word in words).strip()
        if not text:
            continue
        result.append(
            {
                "start": float(words[0]["start"]),
                "end": float(words[-1]["end"]),
                "text": text,
                "words": words,
            }
        )
    return result
