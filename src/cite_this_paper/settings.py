"""User-wide, validated preferences for the interactive shell."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


SETTINGS_VERSION = 1
PROMPT_COLORS = frozenset({"black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"})
COLOR_MODES = frozenset({"auto", "always", "never"})


class SettingsError(ValueError):
    """Raised when a setting name or value is invalid."""


@dataclass(frozen=True)
class UserSettings:
    """The supported user preferences, with defaults for future-compatible files."""

    prompt_color: str = "cyan"
    prompt_bold: bool = True
    prompt_marker: str = "❯"
    color_mode: str = "auto"

    @classmethod
    def defaults(cls) -> "UserSettings":
        return cls()

    def values(self) -> dict[str, str]:
        return {
            "prompt.color": self.prompt_color,
            "prompt.bold": str(self.prompt_bold).lower(),
            "prompt.marker": self.prompt_marker,
            "color.mode": self.color_mode,
        }

    def with_value(self, key: str, value: str) -> "UserSettings":
        """Return settings with one validated, user-supplied value replaced."""
        if key == "prompt.color":
            normalized = value.lower()
            if normalized not in PROMPT_COLORS:
                raise SettingsError(f"prompt.color must be one of: {', '.join(sorted(PROMPT_COLORS))}.")
            return replace(self, prompt_color=normalized)
        if key == "prompt.bold":
            normalized = value.lower()
            if normalized not in {"true", "false"}:
                raise SettingsError("prompt.bold must be true or false.")
            return replace(self, prompt_bold=normalized == "true")
        if key == "prompt.marker":
            if not value or len(value) > 8 or not value.isprintable() or value.isspace():
                raise SettingsError("prompt.marker must be one to eight printable, non-whitespace characters.")
            return replace(self, prompt_marker=value)
        if key == "color.mode":
            normalized = value.lower()
            if normalized not in COLOR_MODES:
                raise SettingsError("color.mode must be auto, always, or never.")
            return replace(self, color_mode=normalized)
        raise SettingsError(f"Unknown setting: {key}")


def user_settings_path() -> Path:
    """Return the platform-standard user configuration path for this application."""
    try:
        from platformdirs import user_config_dir
    except ImportError:  # Allows running a source checkout before dependencies are installed.
        if sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support"
        elif os.name == "nt":
            base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        else:
            base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        return base / "cite-this-paper" / "settings.json"
    return Path(user_config_dir("cite-this-paper")) / "settings.json"


def _from_json(raw: Any) -> UserSettings:
    if not isinstance(raw, dict) or raw.get("version") != SETTINGS_VERSION:
        raise SettingsError(f"Settings must be a version {SETTINGS_VERSION} object.")
    prompt = raw.get("prompt", {})
    color = raw.get("color", {})
    if not isinstance(prompt, dict) or not isinstance(color, dict):
        raise SettingsError("Settings sections must be objects.")
    expected = {"version", "prompt", "color"}
    if set(raw) != expected or set(prompt) != {"color", "bold", "marker"} or set(color) != {"mode"}:
        raise SettingsError("Settings file contains unsupported keys.")
    if not isinstance(prompt["color"], str) or not isinstance(prompt["bold"], bool) or not isinstance(prompt["marker"], str) or not isinstance(color["mode"], str):
        raise SettingsError("Settings values have invalid types.")
    settings = UserSettings()
    settings = settings.with_value("prompt.color", prompt["color"])
    settings = replace(settings, prompt_bold=prompt["bold"])
    settings = settings.with_value("prompt.marker", prompt["marker"])
    return settings.with_value("color.mode", color["mode"])


def load_settings(path: Path | None = None) -> tuple[UserSettings, str | None]:
    """Load user preferences, falling back to defaults with a concise warning."""
    path = path or user_settings_path()
    if not path.exists():
        return UserSettings.defaults(), None
    try:
        return _from_json(json.loads(path.read_text(encoding="utf-8"))), None
    except (OSError, SettingsError, UnicodeDecodeError, json.JSONDecodeError):
        return UserSettings.defaults(), f"Ignoring invalid settings file: {path}. Using defaults."


def save_settings(settings: UserSettings, path: Path | None = None) -> Path:
    """Persist all supported settings in a readable, versioned JSON file."""
    path = path or user_settings_path()
    document = {
        "version": SETTINGS_VERSION,
        "prompt": {
            "color": settings.prompt_color,
            "bold": settings.prompt_bold,
            "marker": settings.prompt_marker,
        },
        "color": {"mode": settings.color_mode},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def prompt_uses_color(settings: UserSettings, stream: object) -> bool:
    """Determine whether prompt ANSI styling is appropriate for this stream."""
    if os.environ.get("NO_COLOR") is not None or settings.color_mode == "never":
        return False
    if settings.color_mode == "always":
        return True
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())
