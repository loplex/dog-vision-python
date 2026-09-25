from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from conftest import (
    needs_ffmpeg,
    read_frames,
    solid,
    write_video,
    write_video_with_sound,
)

from dog_vision.core import video
from dog_vision.core.video import (
    VideoWriter,
    audio_codec,
    convert_video,
    converted_path,
    writer_description,
)


@pytest.fixture
def without_ffmpeg(monkeypatch):
    monkeypatch.setattr(video, "best_ffmpeg_encoder", lambda: None)


def invert(frame: np.ndarray) -> np.ndarray:
    return 255 - frame


def test_a_converted_file_goes_next_to_the_original():
    assert converted_path(Path("a/photo.jpg"), is_video=False) == Path("a/photo.dog.png")
    assert converted_path(Path("a/clip.mov"), is_video=True) == Path("a/clip.dog.mp4")


def test_every_frame_is_rendered_at_the_videos_rate(tmp_path):
    source = write_video(tmp_path / "in.mp4", [20, 60, 100, 140, 180], fps=12)
    out = tmp_path / "out.mp4"
    shares = []
    writer = convert_video(source, out, invert, progress=shares.append)

    frames = read_frames(out)
    assert [round(255 - frame.mean(), -1) for frame in frames] == [20, 60, 100, 140, 180]
    assert cv2.VideoCapture(str(out)).get(cv2.CAP_PROP_FPS) == pytest.approx(12)
    assert shares == sorted(shares) and shares[-1] == 1.0 and len(shares) == 5
    assert writer.path == out


def test_the_video_takes_the_size_render_gives(tmp_path):
    source = write_video(tmp_path / "in.mp4", [100, 100])
    out = tmp_path / "out.mp4"
    convert_video(source, out, lambda frame: np.hstack([frame, frame]))
    assert read_frames(out)[0].shape == (48, 128, 3)


def test_cancelling_before_the_first_frame_writes_nothing(tmp_path):
    source = write_video(tmp_path / "in.mp4", [100] * 5)
    out = tmp_path / "out.mp4"
    assert convert_video(source, out, invert, cancelled=lambda: True) is None
    assert not out.exists()


def test_cancelling_midway_aborts_the_writer(tmp_path, fake_writer):
    source = write_video(tmp_path / "in.mp4", [100] * 5)
    shares = []
    assert convert_video(source, tmp_path / "out.mp4", invert, shares.append, cancelled=lambda: len(shares) == 2) is None
    (writer,) = fake_writer.made
    assert writer.aborted and not writer.closed and len(writer.frames) == 2


def test_a_failing_render_aborts_the_writer(tmp_path, fake_writer):
    source = write_video(tmp_path / "in.mp4", [100] * 5)
    calls = []

    def render(frame):
        calls.append(frame)
        if len(calls) == 3:
            raise ValueError("render failed")
        return frame

    with pytest.raises(ValueError, match="render failed"):
        convert_video(source, tmp_path / "out.mp4", render)
    assert fake_writer.made[0].aborted


def test_a_failing_render_removes_the_file(tmp_path):
    source = write_video(tmp_path / "in.mp4", [100] * 5)
    out = tmp_path / "out.mp4"
    calls = []

    def render(frame):
        calls.append(frame)
        if len(calls) == 3:
            raise ValueError("render failed")
        return frame

    with pytest.raises(ValueError, match="render failed"):
        convert_video(source, out, render)
    assert not out.exists()


def test_a_file_that_is_no_video_is_an_error(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("no video")
    with pytest.raises(RuntimeError, match="Cannot read video"):
        convert_video(path, tmp_path / "out.mp4", invert)


@needs_ffmpeg
def test_the_sound_of_the_original_is_kept(tmp_path):
    source = write_video_with_sound(tmp_path / "in.mp4")
    out = tmp_path / "out.mp4"
    writer = convert_video(source, out, invert)
    assert writer.sound == "kept"
    assert audio_codec(out) == "aac"


@needs_ffmpeg
def test_a_video_without_sound_says_so(tmp_path):
    source = write_video(tmp_path / "in.mp4", [100] * 3)
    writer = convert_video(source, tmp_path / "out.mp4", invert)
    assert writer.sound == "none"
    assert audio_codec(tmp_path / "out.mp4") == ""


@needs_ffmpeg
def test_without_ffprobe_the_sound_is_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "audio_codec", lambda path: None)
    source = write_video(tmp_path / "in.mp4", [100] * 3)
    assert convert_video(source, tmp_path / "out.mp4", invert).sound == "unknown"


@needs_ffmpeg
def test_colours_survive_encoding(tmp_path):
    colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (40, 120, 200), (128, 128, 128)]
    path = tmp_path / "out.mp4"
    writer = VideoWriter(path, None, 64, 48, 30)
    for colour in colours:
        for _ in range(3):
            writer.write(np.full((48, 64, 3), colour, np.uint8))
    writer.close()

    frames = read_frames(path)
    for index, colour in enumerate(colours):
        centre = frames[3 * index + 1][24, 32].astype(int)
        assert np.abs(centre - colour).max() <= 2, (colour, centre)


@needs_ffmpeg
def test_odd_dimensions_are_padded_to_even(tmp_path):
    path = tmp_path / "out.mp4"
    writer = VideoWriter(path, None, 65, 49, 30)
    writer.write(solid(100, 65, 49))
    writer.close()
    assert read_frames(path)[0].shape == (50, 66, 3)


@needs_ffmpeg
def test_a_failing_ffmpeg_is_an_error(tmp_path):
    writer = VideoWriter(tmp_path / "missing" / "out.mp4", None, 64, 48, 30)
    with pytest.raises(RuntimeError, match=r"ffmpeg \(.+\) failed"):
        for _ in range(100):
            writer.write(solid(100, 64, 48))
        writer.close()


@needs_ffmpeg
def test_abort_removes_the_unfinished_file(tmp_path):
    path = tmp_path / "out.mp4"
    writer = VideoWriter(path, None, 64, 48, 30)
    writer.write(solid(100, 64, 48))
    writer.abort()
    assert not path.exists()


def test_without_ffmpeg_opencv_writes_the_video(tmp_path, without_ffmpeg):
    path = tmp_path / "out.mp4"
    writer = VideoWriter(path, None, 64, 48, 30)
    for _ in range(3):
        writer.write(solid(100, 64, 48))
    writer.close()
    assert writer.encoder.startswith("OpenCV ") and writer.sound == "silent"
    assert len(read_frames(path)) == 3


def test_without_ffmpeg_the_sound_is_lost(tmp_path, without_ffmpeg):
    source = write_video(tmp_path / "in.mp4", [100] * 3)
    assert convert_video(source, tmp_path / "out.mp4", invert).sound == "lost"


def test_without_ffmpeg_abort_removes_the_file(tmp_path, without_ffmpeg):
    path = tmp_path / "out.mp4"
    writer = VideoWriter(path, None, 64, 48, 30)
    writer.write(solid(100, 64, 48))
    writer.abort()
    assert not path.exists()


def test_no_ffmpeg_on_the_path_means_no_encoder(monkeypatch):
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    video.best_ffmpeg_encoder.cache_clear()
    try:
        assert video.best_ffmpeg_encoder() is None
    finally:
        video.best_ffmpeg_encoder.cache_clear()


@needs_ffmpeg
def test_the_encoder_found_is_one_of_the_list():
    found = video.best_ffmpeg_encoder()
    assert found is not None
    assert found[1] in video.FFMPEG_ENCODERS


@pytest.mark.parametrize(
    ("sound", "language", "expected"),
    [
        ("kept", "en", "H.265 (libx265), with the original sound"),
        ("none", "en", "H.265 (libx265); the original has no sound"),
        ("lost", "en", "H.265 (libx265), without sound: ffmpeg is not installed"),
        ("unknown", "en", "H.265 (libx265)"),
        ("silent", "en", "H.265 (libx265)"),
        ("kept", "cs", None),
    ],
)
def test_the_writer_is_described(sound, language, expected):
    writer = SimpleNamespace(sound=sound, format_name="H.265", encoder="libx265")
    description = writer_description(writer, language)
    if expected is None:
        assert "H.265 (libx265)" in description and description != writer_description(writer)
    else:
        assert description == expected
