from __future__ import annotations

import json
import math
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
    overlay_font_family: str | None = None
    overlay_font_size: int | None = None
    overlay_font_color: str | None = None
    overlay_outline_enabled: bool | None = None
    overlay_outline_thickness: int | None = None
    overlay_outline_color: str | None = None
    overlay_rect_norm: dict[str, float] | None = None

    @property
    def as_dict(self) -> dict[str, Any]:
        return {
            "subtitle_char_length": self.subtitle_char_length,
            "last_media_dir": self.last_media_dir,
            "start_offset": self.start_offset,
            "window_width": self.window_width,
            "window_height": self.window_height,
            "overlay_font_family": self.overlay_font_family,
            "overlay_font_size": self.overlay_font_size,
            "overlay_font_color": self.overlay_font_color,
            "overlay_outline_enabled": self.overlay_outline_enabled,
            "overlay_outline_thickness": self.overlay_outline_thickness,
            "overlay_outline_color": self.overlay_outline_color,
            "overlay_rect_norm": self.overlay_rect_norm,
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
        def maybe_bool(value: Any) -> bool | None:
            if value is None:
                return None
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "y", "on"}:
                    return True
                if lowered in {"false", "0", "no", "n", "off"}:
                    return False
            return None

        def maybe_str(value: Any) -> str | None:
            if value is None:
                return None
            if isinstance(value, str):
                return value
            return str(value)

        def maybe_rect_norm(value: Any) -> dict[str, float] | None:
            if not isinstance(value, dict):
                return None
            result: dict[str, float] = {}
            for key in ("x", "y", "width", "height"):
                raw = value.get(key)
                if raw is None:
                    return None
                try:
                    result[key] = float(raw)
                except (TypeError, ValueError):
                    return None
                if not math.isfinite(result[key]):
                    return None
            if result["width"] <= 0 or result["height"] <= 0:
                return None
            if (
                result["x"] == 0.0
                and result["y"] == 0.0
                and result["width"] == 0.0
                and result["height"] == 0.0
            ):
                return None
            return result

        return SubtitleConfig(
            subtitle_char_length=int(data.get("subtitle_char_length", 42)),
            last_media_dir=data.get("last_media_dir"),
            start_offset=float(data.get("start_offset", 0.0)),
            window_width=maybe_int(data.get("window_width")),
            window_height=maybe_int(data.get("window_height")),
            overlay_font_family=maybe_str(data.get("overlay_font_family")),
            overlay_font_size=maybe_int(data.get("overlay_font_size")),
            overlay_font_color=maybe_str(data.get("overlay_font_color")),
            overlay_outline_enabled=maybe_bool(data.get("overlay_outline_enabled")),
            overlay_outline_thickness=maybe_int(data.get("overlay_outline_thickness")),
            overlay_outline_color=maybe_str(data.get("overlay_outline_color")),
            overlay_rect_norm=maybe_rect_norm(data.get("overlay_rect_norm")),
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
