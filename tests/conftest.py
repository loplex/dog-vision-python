import shutil
import subprocess
import threading
from typing import ClassVar

import cv2
import numpy as np
import pytest

from dog_vision.core import settings as settings_store
from dog_vision.core import video


class Clock:
    """A time.monotonic that moves only when a test sets now."""

    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now


class FakeWriter:
    """A VideoWriter that keeps the frames written to it instead of encoding them."""

    made: ClassVar[list["FakeWriter"]] = []
    fail_on_write: ClassVar[Exception | None] = None
    block_writes: ClassVar[threading.Event | None] = None  # while unset, write() waits for it
    writing: ClassVar[threading.Event] = threading.Event()  # set on the first write()

    def __init__(self, path, sound_from, width, height, fps) -> None:
        self.path, self.size, self.fps = path, (width, height), fps
        self.frames: list[np.ndarray] = []
        self.closed = self.aborted = False
        FakeWriter.made.append(self)

    def write(self, frame: np.ndarray) -> None:
        FakeWriter.writing.set()
        if FakeWriter.block_writes is not None:
            FakeWriter.block_writes.wait(10)
        if FakeWriter.fail_on_write is not None:
            raise FakeWriter.fail_on_write
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True

    def abort(self) -> None:
        self.aborted = True


@pytest.fixture(autouse=True)
def english(monkeypatch):
    """Every test in English, whatever language the machine running it asks for."""
    monkeypatch.setenv("LANGUAGE", "en")


@pytest.fixture(autouse=True)
def settings_file(tmp_path, monkeypatch):
    """Where the settings are saved during a test, never the user's own file."""
    path = tmp_path / "config" / "settings.json"
    monkeypatch.setattr(settings_store, "config_path", lambda: path)
    return path


@pytest.fixture
def clock(monkeypatch) -> Clock:
    """Stands in for the clock the recorder reads, and for no other module's."""
    fake = Clock()
    monkeypatch.setattr(video, "time", fake)
    return fake


@pytest.fixture
def fake_writer(monkeypatch) -> type[FakeWriter]:
    monkeypatch.setattr(FakeWriter, "made", [])
    monkeypatch.setattr(FakeWriter, "fail_on_write", None)
    monkeypatch.setattr(FakeWriter, "block_writes", None)
    monkeypatch.setattr(FakeWriter, "writing", threading.Event())
    monkeypatch.setattr(video, "VideoWriter", FakeWriter)
    return FakeWriter


def solid(value: int, width: int = 128, height: int = 96) -> np.ndarray:
    """A BGR frame of one grey, which tells frames apart after encoding too."""
    return np.full((height, width, 3), value, np.uint8)


def read_frames(path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def write_video(path, values: list[int], fps: float = 10, width: int = 64, height: int = 48):
    """A silent video of one grey frame per value, written with OpenCV alone."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for value in values:
        writer.write(solid(value, width, height))
    writer.release()
    return path


def write_video_with_sound(path, seconds: float = 0.5):
    """A test pattern with a tone, written with ffmpeg."""
    command = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=s=64x48:d={seconds}:r=10"]
    command += ["-f", "lavfi", "-i", f"sine=d={seconds}", "-shortest", "-c:v", "libx264", "-c:a", "aac", str(path)]
    subprocess.run(command, check=True)
    return path
