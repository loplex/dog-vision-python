import re
import threading
import time

import cv2
import numpy as np
import pytest
from conftest import read_frames, write_video

from dog_vision.core import session as session_module
from dog_vision.core.imaging import compose
from dog_vision.core.model import Params
from dog_vision.core.session import PREVIEW_LONGEST_SIDE, LiveSession
from dog_vision.core.video import RECORDING_FPS


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A session showing a colourful photo, recording into tmp_path."""
    photo = tmp_path / "photo.png"
    cv2.imwrite(str(photo), np.random.default_rng(0).integers(0, 256, (48, 64, 3), np.uint8))
    monkeypatch.chdir(tmp_path)
    live = LiveSession(0, Params(), path=photo)
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


def wait_for(condition, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_a_photo_is_named_by_its_file(session):
    assert session.source_name() == "photo.png"
    assert not session.source_is_video


def test_captions_say_what_each_image_shows(session):
    session.render()
    assert session.captions() == ["original", "dog (dichromat)"]
    session.compare = "cat"
    assert session.captions() == ["cat (dichromat)", "dog (dichromat)"]
    session.side_by_side = False
    assert session.captions() == ["dog (dichromat)"]


def test_the_map_of_differences_is_captioned_with_its_share(session):
    session.difference = True
    session.render()
    *_images, caption = session.captions()
    assert re.fullmatch(r"red: noticeably different \(\d+% of pixels\)", caption)


def test_captions_follow_the_language(session):
    session.language = "cs"
    assert session.captions() == ["originál", "pes (dichromat)"]
    assert session.species_labels[0] == "pes (dichromat)"


def test_a_still_view_is_not_rendered_again(session):
    first = session.render()
    assert session.render() is first
    session.params.species = "cat"
    assert session.render() is not first


def test_reset_restores_the_command_line_values(tmp_path):
    photo = tmp_path / "photo.png"
    cv2.imwrite(str(photo), np.zeros((8, 8, 3), np.uint8))
    live = LiveSession(0, Params("cat", strength=0.5), compare="dog", path=photo)
    live.params.species, live.params.strength, live.compare = "cow", 1.0, None
    live.reset()
    assert (live.params, live.compare) == (Params("cat", strength=0.5), "dog")
    live.params.species = "pig"
    assert live.initial.species == "cat"  # reset() copies, and changes made after it stay there
    live.close()


def test_a_snapshot_is_the_view_as_shown(session, tmp_path):
    assert session.save_snapshot() is None  # nothing rendered yet
    shown = session.render()
    name = session.save_snapshot()
    assert re.fullmatch(r"dog-dog-\d{8}-\d{6}\.png", name)
    np.testing.assert_array_equal(cv2.imread(str(tmp_path / name))[..., ::-1], shown)


def test_a_snapshot_names_both_species_compared(session):
    session.compare = "cat"
    session.render()
    assert session.save_snapshot().startswith("dog-cat-vs-dog-")


def test_a_large_photo_is_shown_smaller_and_converted_at_full_size(tmp_path):
    photo = tmp_path / "large.png"
    cv2.imwrite(str(photo), np.full((1500, 3000, 3), 90, np.uint8))
    live = LiveSession(0, Params(), path=photo)
    live.side_by_side = False
    assert live.render().shape == (PREVIEW_LONGEST_SIDE // 2, PREVIEW_LONGEST_SIDE, 3)
    assert live.convert_source()
    wait_for(lambda: not live.converting)
    assert live.conversion_status() == "Wrote large.dog.png"
    assert cv2.imread(str(tmp_path / "large.dog.png")).shape == (1500, 3000, 3)
    live.close()


def test_a_file_that_cannot_be_read_leaves_the_source_as_it_was(session, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("no picture")
    assert not session.open_file(notes)
    assert session.source_name() == "photo.png"


def test_a_video_is_played_in_the_window(session, tmp_path):
    clip = write_video(tmp_path / "clip.mp4", [60, 120, 180])
    assert session.open_file(clip)
    assert session.source_is_video and session.source_name() == "clip.mp4"
    wait_for(lambda: session.render() is not None)
    assert session.render().shape == (48, 128, 3)


def test_a_video_is_converted_with_the_settings_of_the_moment(session, tmp_path):
    clip = write_video(tmp_path / "clip.mp4", [60, 120, 180])
    session.open_file(clip)
    session.side_by_side = False
    assert session.conversion_status() is None
    assert session.convert_source()
    wait_for(lambda: not session.converting)
    assert session.conversion_status().startswith("Wrote clip.dog.mp4: ")
    frames = read_frames(tmp_path / "clip.dog.mp4")
    assert len(frames) == 3 and frames[0].shape == (48, 64, 3)


def test_a_conversion_reports_its_progress(session, monkeypatch):
    started, finish = threading.Event(), threading.Event()

    def convert_video(path, out_path, render, progress, cancelled):
        progress(0.25)
        started.set()
        finish.wait(10)

    monkeypatch.setattr(session_module, "convert_video", convert_video)
    session.source_is_video = True
    assert session.convert_source()
    started.wait(10)
    assert session.converting and not session.convert_source()  # one at a time
    assert session.conversion_status() == "Converting: 25%"
    finish.set()
    wait_for(lambda: not session.converting)
    assert session.conversion_status() is None  # returned None: cancelled


def test_a_failed_conversion_says_why(session, tmp_path):
    (tmp_path / "photo.png").unlink()  # after it was opened
    session.convert_source()
    wait_for(lambda: not session.converting)
    assert session.conversion_status().startswith("Conversion failed: Cannot read image: ")


def test_close_cancels_a_conversion(session, monkeypatch):
    seen = []

    def convert_video(path, out_path, render, progress, cancelled):
        wait_for(cancelled)
        seen.append("cancelled")

    monkeypatch.setattr(session_module, "convert_video", convert_video)
    session.source_is_video = True
    session.convert_source()
    session.close()
    assert seen == ["cancelled"]


def test_the_camera_cannot_be_converted(session):
    session.source = None
    assert not session.convert_source()
    assert session.source_name() == "Camera 0"


def test_a_missing_camera_is_refused():
    with pytest.raises(SystemExit, match="Cannot open camera 99"):
        LiveSession(99, Params())


def test_a_missing_camera_leaves_the_file_shown(session):
    session.camera_index = 99
    assert not session.open_camera()
    assert session.source_name() == "photo.png"


def test_settings_changed_after_a_conversion_started_are_not_in_it(session, monkeypatch):
    changed, rendered = threading.Event(), []
    frame = np.random.default_rng(1).integers(0, 256, (48, 64, 3), np.uint8)

    def convert_video(path, out_path, render, progress, cancelled):
        changed.wait(10)
        rendered.append(render(frame))

    monkeypatch.setattr(session_module, "convert_video", convert_video)
    session.source_is_video = True
    session.convert_source()
    session.params.species, session.side_by_side = "cat", False
    changed.set()
    wait_for(lambda: not session.converting)
    np.testing.assert_array_equal(rendered[0], compose(frame, Params("dog"), True, None, False)[0])
