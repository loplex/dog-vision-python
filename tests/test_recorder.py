import threading

import pytest
from conftest import read_frames, solid

from dog_vision.core.video import RECORDING_FPS, RECORDING_QUEUE, Recorder


def markers(writer) -> list[int]:
    return [int(frame[0, 0, 0]) for frame in writer.frames]


def record(recorder: Recorder) -> None:
    recorder.join(10)
    assert recorder.done


def test_each_frame_is_repeated_for_as_long_as_it_was_shown(tmp_path, clock, fake_writer):
    recorder = Recorder(tmp_path / "out.mp4")
    recorder.add(solid(1))
    clock.now += 0.1
    recorder.add(solid(2))
    clock.now += 0.4
    recorder.add(solid(3))
    clock.now += 0.5
    recorder.stop()
    record(recorder)

    assert recorder.error is None
    (writer,) = fake_writer.made
    assert markers(writer) == [1] * 3 + [2] * 12 + [3] * 15
    assert writer.closed and not writer.aborted
    assert writer.size == (128, 96) and writer.fps == RECORDING_FPS


def test_video_starts_at_the_first_frame_not_at_start(tmp_path, clock, fake_writer):
    recorder = Recorder(tmp_path / "out.mp4")
    clock.now += 5
    recorder.add(solid(1))
    clock.now += 1
    recorder.stop()
    record(recorder)

    assert markers(fake_writer.made[0]) == [1] * RECORDING_FPS


def test_stopped_within_half_a_frame_still_writes_the_frame(tmp_path, clock, fake_writer):
    recorder = Recorder(tmp_path / "out.mp4")
    recorder.add(solid(1))
    clock.now += 0.4 / RECORDING_FPS
    recorder.stop()
    record(recorder)

    assert recorder.error is None
    assert markers(fake_writer.made[0]) == [1]


def test_a_frame_of_another_size_is_dropped(tmp_path, clock, fake_writer):
    recorder = Recorder(tmp_path / "out.mp4")
    recorder.add(solid(1))
    clock.now += 0.5
    recorder.add(solid(2, width=64, height=48))
    clock.now += 0.5
    recorder.stop()
    record(recorder)

    assert recorder.error is None
    assert markers(fake_writer.made[0]) == [1] * RECORDING_FPS


def test_a_frame_that_comes_while_the_encoder_is_behind_is_dropped(tmp_path, clock, fake_writer):
    fake_writer.block_writes = threading.Event()
    recorder = Recorder(tmp_path / "out.mp4")
    recorder.add(solid(1))
    clock.now += 0.1
    recorder.add(solid(2))  # taken off the queue, then the encoder stalls on frame 1
    assert fake_writer.writing.wait(10)
    for marker in range(3, 3 + RECORDING_QUEUE):  # fill the queue
        clock.now += 0.1
        recorder.add(solid(marker))
    clock.now += 0.1
    recorder.add(solid(99))  # the queue is full
    clock.now += 0.1
    recorder.stop()
    fake_writer.block_writes.set()
    record(recorder)

    shown = markers(fake_writer.made[0])
    assert 99 not in shown
    last = 2 + RECORDING_QUEUE
    assert shown[-6:] == [last] * 6  # shown for its own 0.1 s and the dropped frame's
    assert sorted(set(shown)) == list(range(1, last + 1))


def test_stopping_before_any_frame_is_an_error_not_a_crash(tmp_path, clock, fake_writer):
    recorder = Recorder(tmp_path / "out.mp4")
    recorder.stop()
    record(recorder)

    assert recorder.error == "No frame was shown while recording"
    assert recorder.writer is None and fake_writer.made == []


@pytest.mark.parametrize("error", [RuntimeError("ffmpeg (libx265) failed: boom"), OSError("disk full")])
def test_a_failing_encoder_is_reported_and_its_file_removed(tmp_path, clock, fake_writer, error):
    fake_writer.fail_on_write = error
    recorder = Recorder(tmp_path / "out.mp4")
    recorder.add(solid(1))
    clock.now += 1
    recorder.stop()
    record(recorder)

    assert recorder.error == str(error)
    assert fake_writer.made[0].aborted


def test_records_a_video_that_plays_back_as_shown(tmp_path, clock):
    path = tmp_path / "out.mp4"
    recorder = Recorder(path)
    recorder.add(solid(30))
    clock.now += 0.5
    recorder.add(solid(230))
    clock.now += 0.5
    recorder.stop()
    record(recorder)

    assert recorder.error is None
    frames = read_frames(path)
    assert len(frames) == RECORDING_FPS
    assert frames[0].shape == (96, 128, 3)
    greys = [round(float(frame.mean())) for frame in frames]
    assert all(abs(grey - 30) <= 3 for grey in greys[:15]), greys
    assert all(abs(grey - 230) <= 3 for grey in greys[15:]), greys
