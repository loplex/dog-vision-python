"""What the window remembers between runs: where snapshots, recordings and conversions go.

The settings are a JSON file in the user's configuration directory. A missing or unreadable
file means the defaults, so a first run and a damaged file behave alike.
"""

import dataclasses
import json
import os
import sys
from pathlib import Path


@dataclasses.dataclass
class Settings:
    output_dir: Path | None = None  # for snapshots and recordings; None for the working directory
    convert_to_output_dir: bool = False  # put a converted file there too, instead of next to its original


def config_path() -> Path:
    """settings.json in the configuration directory each system has for it."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "dog-vision" / "settings.json"


def load(path: Path | None = None) -> Settings:
    """The settings saved at path (config_path() by default), or the defaults if there are none to read."""
    try:
        saved = json.loads((path or config_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Settings()
    if not isinstance(saved, dict):
        return Settings()
    output_dir = saved.get("output_dir")
    return Settings(
        output_dir=Path(output_dir) if isinstance(output_dir, str) and output_dir else None,
        convert_to_output_dir=saved.get("convert_to_output_dir") is True,
    )


def save(settings: Settings, path: Path | None = None) -> None:
    """Write settings to path (config_path() by default); raises OSError if it cannot."""
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {
        "output_dir": str(settings.output_dir) if settings.output_dir is not None else None,
        "convert_to_output_dir": settings.convert_to_output_dir,
    }
    path.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
