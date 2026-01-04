from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parent / "subtitle_config.json"


@dataclass
class SubtitleConfig:
    subtitle_char_length: int = 42
    last_media_dir: str | None = None
    start_offset: float = 0.0
    window_width: int | None = None
    window_height: int | None = None

    @property
    def as_dict(self) -> dict[str, Any]:
        return {
            "subtitle_char_length": self.subtitle_char_length,
            "last_media_dir": self.last_media_dir,
            "start_offset": self.start_offset,
            "window_width": self.window_width,
            "window_height": self.window_height,
        }


def load_config() -> SubtitleConfig:
    if not CONFIG_PATH.exists():
        return SubtitleConfig()
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        def maybe_int(value: Any) -> int | None:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        return SubtitleConfig(
            subtitle_char_length=int(data.get("subtitle_char_length", 42)),
            last_media_dir=data.get("last_media_dir"),
            start_offset=float(data.get("start_offset", 0.0)),
            window_width=maybe_int(data.get("window_width")),
            window_height=maybe_int(data.get("window_height")),
        )
    except Exception:
        return SubtitleConfig()


def save_config(config: SubtitleConfig) -> None:
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(config.as_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass
