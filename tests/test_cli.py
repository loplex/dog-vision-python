import sys

import cv2
import numpy as np
import pytest
from conftest import read_frames, write_video

from dog_vision import cli
from dog_vision.gui import tk as tk_gui


def run(monkeypatch, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["dog-vision", *map(str, arguments)])
    cli.main()


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "photo.png"
    cv2.imwrite(str(path), np.random.default_rng(0).integers(0, 256, (48, 64, 3), np.uint8))
    return path


def test_info_prints_the_model_and_its_checks(monkeypatch, capsys):
    run(monkeypatch, "--info", "--species", "dog")
    out = capsys.readouterr().out
    assert "Species dog, cone peaks (429.0, 555.0) nm" in out
    assert "grey preserved   T @ (1,1,1) = [1. 1. 1.]" in out
    assert "neutral point    479 nm" in out
    assert "Neutral point:  479 nm (model)" in out


def test_info_checks_a_trichromats_rnl_scale(monkeypatch, capsys):
    run(monkeypatch, "--info", "--species", "macaque")
    out = capsys.readouterr().out
    assert "rnl keeps blue" in out
    assert "  neutral point  " not in out


def test_a_photo_is_converted_next_to_it(monkeypatch, capsys, photo):
    run(monkeypatch, photo)
    out_path = photo.with_suffix(".dog.png")
    assert cv2.imread(str(out_path)).shape == (48, 64, 3)
    assert capsys.readouterr().out == f"Wrote {out_path}\n"


def test_a_photo_is_converted_beside_another_species(monkeypatch, photo):
    run(monkeypatch, photo, "--compare", "cat")
    assert cv2.imread(str(photo.with_suffix(".dog.png"))).shape == (48, 128, 3)


def test_a_photo_is_converted_with_its_map_of_differences(monkeypatch, capsys, photo):
    run(monkeypatch, photo, "--difference")
    assert cv2.imread(str(photo.with_suffix(".dog.png"))).shape == (48, 192, 3)
    assert "% of pixels differ noticeably" in capsys.readouterr().out


def test_a_video_is_converted_next_to_it(monkeypatch, capsys, tmp_path):
    source = write_video(tmp_path / "clip.mp4", [50, 100, 150])
    run(monkeypatch, source, "--species", "cat")
    out_path = tmp_path / "clip.dog.mp4"
    assert len(read_frames(out_path)) == 3
    assert capsys.readouterr().out.startswith(f"Wrote {out_path}: ")


def test_a_file_that_is_neither_photo_nor_video_is_an_error(monkeypatch, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("nothing to see")
    with pytest.raises(SystemExit, match="Cannot read image or video"):
        run(monkeypatch, path)


def test_an_unknown_species_is_refused(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        run(monkeypatch, "--species", "unicorn")
    assert exit_info.value.code == 2
    assert "invalid choice: 'unicorn'" in capsys.readouterr().err


def test_the_options_reach_the_window(monkeypatch, photo):
    shown = []
    monkeypatch.setattr(tk_gui, "run", shown.append)
    run(monkeypatch, "--window", photo, "--species", "cat", "--compare", "dog", "--difference", "--acuity",
        "--fov", "40", "--adaptation", "0.5", "--strength", "0.8", "--chroma-scale", "rnl")
    (session,) = shown
    assert session.source == photo
    assert (session.compare, session.difference) == ("dog", True)
    params = session.params
    assert (params.species, params.acuity, params.field_of_view) == ("cat", True, 40)
    assert (params.adaptation, params.strength, params.chroma_scale) == (0.5, 0.8, "rnl")


def test_the_session_is_closed_when_the_window_is(monkeypatch, tmp_path, photo):
    shown = []

    def record(session):
        shown.append(session)
        session.render()
        session.start_recording()  # and the window closed while recording

    monkeypatch.setattr(tk_gui, "run", record)
    monkeypatch.chdir(tmp_path)
    run(monkeypatch, "--window", photo)
    (session,) = shown
    assert not session.recording and session._recorded.done
    assert session.recording_status().startswith("Wrote dog-dog-")


def test_a_camera_that_stops_ends_with_its_error(monkeypatch, photo):
    def fail(session):
        session.error = "Camera stopped delivering frames"

    monkeypatch.setattr(tk_gui, "run", fail)
    with pytest.raises(SystemExit, match="Camera stopped delivering frames"):
        run(monkeypatch, "--window", photo)
