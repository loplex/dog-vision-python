import re

import cv2
import numpy as np
import pytest
from conftest import read_frames

from dog_vision.core.model import Params
from dog_vision.core.session import LiveSession
from dog_vision.core.video import RECORDING_FPS


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A session showing a colourful photo, recording into tmp_path."""
    photo = tmp_path / "photo.png"
    cv2.imwrite(str(photo), np.random.default_rng(0).integers(0, 256, (48, 64, 3), np.uint8))
    monkeypatch.chdir(tmp_path)
    live = LiveSession(0, Params(), path=photo)
    live.language = "en"
    yield live
    live.close()


def test_records_a_still_photo_to_the_working_directory(session, tmp_path):
    shown = session.render()
    assert session.recording_status() is None

    assert session.start_recording()
    assert session.recording
    assert re.fullmatch(r"Recording dog-dog-\d{8}-\d{6}\.mp4: 0:00", session.recording_status())
    session.stop_recording()
    assert not session.recording
    session.close()  # waits for the video to be finished

    status = session.recording_status()
    assert re.fullmatch(r"Wrote dog-dog-\d{8}-\d{6}\.mp4: .+", status), status
    (path,) = tmp_path.glob("dog-dog-*.mp4")
    frames = read_frames(path)
    assert frames and frames[0].shape == shown.shape


def test_starting_twice_keeps_the_first_recording(session):
    session.render()
    assert session.start_recording()
    first = session._recorder
    assert not session.start_recording()
    assert session._recorder is first


def test_the_view_is_recorded_as_it_is_rendered(session, clock, fake_writer):
    dog = session.render().copy()
    session.start_recording()
    clock.now += 0.5
    session.params.species = "cat"
    cat = session.render().copy()
    assert not np.array_equal(dog, cat)
    clock.now += 0.5
    session.stop_recording()
    session.close()

    frames = fake_writer.made[0].frames
    assert len(frames) == RECORDING_FPS
    assert all(np.array_equal(frame[..., ::-1], dog) for frame in frames[:15])
    assert all(np.array_equal(frame[..., ::-1], cat) for frame in frames[15:])


def test_a_failed_recording_says_why(session):
    session.start_recording()  # before the first frame is rendered
    session.stop_recording()
    session.close()

    assert session.recording_status() == "Recording failed: No frame was shown while recording"
