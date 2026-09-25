"""LiveSession: what the live window shows and does, independent of the GUI toolkit."""

import dataclasses
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from dog_vision.core import i18n
from dog_vision.core import settings as settings_store
from dog_vision.core.facts import percent, species_facts, species_label
from dog_vision.core.i18n import N_
from dog_vision.core.imaging import compose
from dog_vision.core.model import CHROMA_SCALES, Params
from dog_vision.core.settings import Settings
from dog_vision.core.species import SPECIES
from dog_vision.core.video import (
    Recorder,
    convert_video,
    converted_path,
    writer_description,
)

PREVIEW_LONGEST_SIDE = 1280  # pixels a file is shown at in the window; converting it keeps its size
RECORDING_FINISH_S = 30  # seconds close() waits for a stopped recording to be written


def preview(image: np.ndarray) -> np.ndarray:
    """A file's frame scaled down to PREVIEW_LONGEST_SIDE, which keeps the window responsive."""
    scale = PREVIEW_LONGEST_SIDE / max(image.shape[:2])
    if scale >= 1:
        return image
    size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


class LiveSession:
    """Everything the live window shows and does, independent of the GUI toolkit.

    The source is a camera or a file. A camera or a video is read on a background
    thread, so a GUI can poll render() from its own timer without waiting for the next
    frame; a video plays at its own rate and starts over at its end. A GUI changes
    params, side_by_side, compare and difference directly, and calls reset(),
    save_snapshot(), start_recording(), stop_recording(), open_file(), open_camera(),
    convert_source(), set_output_dir(), set_convert_to_output_dir() and close(). While
    recording, a video's size cannot change, so a GUI leaves side_by_side, difference
    and the source as they are.
    """

    def __init__(
        self,
        camera_index: int,
        initial: Params,
        compare: str | None = None,
        path: Path | None = None,
        saved: Settings | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """saved is what the window remembers, and output_dir overrides its folder until another is chosen."""
        self.species_names = list(SPECIES)
        self.chroma_scales = CHROMA_SCALES
        self.languages = {code: language.name for code, language in i18n.LANGUAGES.items()}  # in itself
        self.language = i18n.system_language()
        self.initial = initial
        self.params = dataclasses.replace(initial)
        self.side_by_side = True
        self.initial_compare = compare
        self.compare = compare  # the species on the left instead of the original, if any
        self.difference = False  # add a map of where left and right differ noticeably
        self.camera_index = camera_index
        self.settings = saved or Settings()
        self._output_dir_override = output_dir
        self.source: Path | None = None  # the open file, or None for the camera
        self.source_is_video = False
        self.error: str | None = None  # set when the camera stops delivering frames
        self._difference_share: float | None = None
        self._frame: np.ndarray | None = None
        self._last_images: np.ndarray | None = None
        self._last_rgb: np.ndarray | None = None
        self._last_input: tuple | None = None  # the frame and settings _last_rgb was rendered from
        self._lock = threading.Lock()
        self._reader: tuple[threading.Thread, threading.Event, cv2.VideoCapture] | None = None
        self._conversion: threading.Thread | None = None
        self._conversion_progress: float | None = None
        self._conversion_result: tuple[str, dict[str, str]] | None = None  # English text and its fields
        self._cancel_conversion = threading.Event()
        self._recorder: Recorder | None = None  # while recording
        self._recorded: Recorder | None = None  # the last recording stopped, finished or not
        if path is not None and not self.open_file(path):
            sys.exit(f"Cannot read image or video: {path}")
        if path is None and not self.open_camera():
            sys.exit(f"Cannot open camera {camera_index}")

    def open_camera(self) -> bool:
        """Show the camera again; False if it cannot be opened."""
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            return False
        self._stop_reader()
        self.source, self.source_is_video = None, False
        self._start_reader(capture, is_file=False)
        return True

    def open_file(self, path: Path) -> bool:
        """Show a photo or a video instead of the camera; False if it is neither."""
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        capture = None
        if image is None:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                return False
        self._stop_reader()
        self.source, self.source_is_video = Path(path), capture is not None
        if capture is None:
            with self._lock:
                self._frame = preview(image)
        else:
            self._start_reader(capture, is_file=True)
        return True

    def _start_reader(self, capture: cv2.VideoCapture, is_file: bool) -> None:
        with self._lock:
            self._frame = None
        stop = threading.Event()
        thread = threading.Thread(target=self._read_frames, args=(capture, stop, is_file), daemon=True)
        self._reader = (thread, stop, capture)
        thread.start()

    def _stop_reader(self) -> None:
        if self._reader is not None:
            thread, stop, capture = self._reader
            stop.set()
            thread.join(timeout=1)
            capture.release()
            self._reader = None

    def _read_frames(self, capture: cv2.VideoCapture, stop: threading.Event, is_file: bool) -> None:
        interval = 1 / (capture.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 0.0
        due = time.monotonic()
        while not stop.is_set():
            ok, frame = capture.read()
            if not ok and is_file:  # the end of a video: start over
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
            if not ok:
                if not is_file:
                    self.error = "Camera stopped delivering frames"
                return
            with self._lock:
                self._frame = preview(frame) if is_file else frame
            if is_file:  # play at the video's own rate, without catching up after a stall
                due = max(due + interval, time.monotonic())
                stop.wait(due - time.monotonic())

    def render(self) -> np.ndarray | None:
        """The current view as an RGB image, or None before the first frame.

        A camera is shown at its resolution, a file at PREVIEW_LONGEST_SIDE at most. The
        same frame with the same settings is not rendered twice, so a photo costs nothing
        while it stands still.
        """
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        settings = (dataclasses.astuple(self.params), self.side_by_side, self.compare, self.difference)
        if self._last_input is not None and self._last_input[0] is frame and self._last_input[1] == settings:
            return self._last_rgb
        self._last_images, self._difference_share = compose(
            frame, self.params, self.side_by_side, self.compare, self.difference
        )
        self._last_input = (frame, settings)
        self._last_rgb = self._last_images[..., ::-1]
        if self._recorder is not None:
            self._recorder.add(self._last_images)
        return self._last_rgb

    def translate(self, text: str) -> str:
        """The GUI's own English text in the current language."""
        return i18n.translate(text, self.language)

    @property
    def species_labels(self) -> list[str]:
        """species_names in the current language, each with its kind of colour vision."""
        return [species_label(name, self.language) for name in self.species_names]

    def source_name(self) -> str:
        """What is being shown: the camera, or the file's name."""
        if self.source is None:
            return self.translate("Camera {index}").format(index=self.camera_index)
        return self.source.name

    def captions(self) -> list[str]:
        """What each image of the rendered view shows, left to right."""
        right = species_label(self.params.species, self.language)
        if not self.side_by_side:
            return [right]
        left = self.translate("original") if self.compare is None else species_label(self.compare, self.language)
        if self._difference_share is None:
            return [left, right]
        share = percent(self._difference_share, self._difference_share, self.language)
        return [left, right, self.translate("red: noticeably different ({share} of pixels)").format(share=share)]

    def reset(self) -> None:
        self.params = dataclasses.replace(self.initial)
        self.compare = self.initial_compare

    def species_facts(self, species: str | None = None) -> list[tuple[str, str, str]]:
        """The simulation's facts about a species, the current one by default, as (label, value, description) rows."""
        return species_facts(species or self.params.species, self.language)

    @property
    def output_dir(self) -> Path:
        """Where snapshots and recordings go, and conversions if convert_to_output_dir says so."""
        return (self._output_dir_override or self.settings.output_dir or Path.cwd()).expanduser().absolute()

    def set_output_dir(self, path: Path) -> str | None:
        """Save snapshots and recordings in path from now on, and remember it; the error, if it cannot be."""
        self._output_dir_override = None
        self.settings.output_dir = Path(path)
        return self._save_settings()

    def set_convert_to_output_dir(self, convert_to_output_dir: bool) -> str | None:
        """Put converted files in output_dir or next to their originals, and remember it; the error, if it cannot be."""
        self.settings.convert_to_output_dir = convert_to_output_dir
        return self._save_settings()

    def _save_settings(self) -> str | None:
        try:
            settings_store.save(self.settings)
        except OSError as error:
            return self.translate("Cannot save the settings: {error}").format(error=error)
        return None

    def _output_path(self, extension: str) -> Path:
        """A path in output_dir for the view as shown now, e.g. dog-cat-20260925-105600.png.

        The folder is made if it is missing; if it cannot be, writing the file says so.
        """
        shown = (
            self.params.species
            if self.compare is None or not self.side_by_side
            else f"{self.compare}-vs-{self.params.species}"
        )
        folder = self.output_dir
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return folder / f"dog-{shown}-{time.strftime('%Y%m%d-%H%M%S')}.{extension}"

    def save_snapshot(self) -> str | None:
        """Write the last rendered view to output_dir and return the file name; None before the first frame.

        Raises OSError if the file cannot be written.
        """
        if self._last_images is None:
            return None
        path = self._output_path("png")
        if not cv2.imwrite(str(path), self._last_images):
            raise OSError(f"Cannot write {path}")
        return path.name

    @property
    def recording(self) -> bool:
        return self._recorder is not None

    def start_recording(self) -> bool:
        """Record the view as shown to dog-<species>-<time>.mp4 in output_dir; False if recording."""
        if self._recorder is not None:
            return False
        self._recorder = Recorder(self._output_path("mp4"))
        if self._last_images is not None:  # a still photo renders no new frame to start with
            self._recorder.add(self._last_images)
        return True

    def stop_recording(self) -> None:
        """Stop recording; the video is finished in the background, as recording_status() says."""
        if self._recorder is not None:
            self._recorder.stop()
            self._recorded, self._recorder = self._recorder, None

    def recording_status(self) -> str | None:
        """How the running or the last recording stands, or None if there was none."""
        if self._recorder is not None:
            minutes, seconds = divmod(int(time.monotonic() - self._recorder.started), 60)
            return self.translate("Recording {name}: {time}").format(
                name=self._recorder.path.name, time=f"{minutes}:{seconds:02d}"
            )
        recorded = self._recorded
        if recorded is None:
            return None
        if not recorded.done:
            return self.translate("Finishing {name}").format(name=recorded.path.name)
        if recorded.error is not None:
            return self.translate("Recording failed: {error}").format(error=recorded.error)
        description = writer_description(recorded.writer, self.language)
        return self.translate("Wrote {name}: {description}").format(name=recorded.path.name, description=description)

    @property
    def converting(self) -> bool:
        return self._conversion is not None and self._conversion.is_alive()

    def convert_source(self) -> bool:
        """Convert the open file at full size with the current settings, in the background.

        The result is <name>.dog.png or <name>.dog.mp4, next to the file or in output_dir as
        settings.convert_to_output_dir says, and shows the view as the window does. False if
        the camera is shown or a conversion is running.
        """
        if self.source is None or self.converting:
            return False
        source, is_video = self.source, self.source_is_video
        params = dataclasses.replace(self.params)
        side_by_side, compare, difference = self.side_by_side, self.compare, self.difference
        out_path = converted_path(source, is_video)
        if self.settings.convert_to_output_dir:
            out_path = self.output_dir / out_path.name

        def render(frame: np.ndarray) -> np.ndarray:
            return compose(frame, params, side_by_side, compare, difference)[0]

        def convert() -> None:
            result = None  # stays None when cancelled
            try:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if is_video:
                    writer = convert_video(source, out_path, render, self._set_progress, self._cancel_conversion.is_set)
                    if writer is not None:
                        result = (N_("Wrote {name}: {description}"), {"name": out_path.name, "writer": writer})
                else:
                    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
                    if image is None:
                        raise RuntimeError(f"Cannot read image: {source}")
                    if not cv2.imwrite(str(out_path), render(image)):
                        raise OSError(f"Cannot write {out_path}")
                    result = (N_("Wrote {name}"), {"name": out_path.name})
            except (RuntimeError, OSError, cv2.error) as error:  # shown in the window, not lost with the thread
                result = (N_("Conversion failed: {error}"), {"error": str(error)})
            finally:
                # The result first, so that conversion_status() never falls silent in between.
                self._conversion_result = result
                self._conversion_progress = None

        self._cancel_conversion.clear()
        self._conversion_result = None
        self._conversion_progress = 0.0
        self._conversion = threading.Thread(target=convert, daemon=True)
        self._conversion.start()
        return True

    def _set_progress(self, share: float) -> None:
        self._conversion_progress = share

    def conversion_status(self) -> str | None:
        """How the running or the last conversion stands, or None if there was none."""
        if self._conversion_progress is not None:
            share = percent(self._conversion_progress, self._conversion_progress, self.language)
            return self.translate("Converting: {share}").format(share=share)
        if self._conversion_result is None:
            return None
        text, fields = self._conversion_result
        if "writer" in fields:
            fields = {"name": fields["name"], "description": writer_description(fields["writer"], self.language)}
        return self.translate(text).format(**fields)

    def close(self) -> None:
        self.stop_recording()
        if self._recorded is not None:
            self._recorded.join(RECORDING_FINISH_S)
        self._cancel_conversion.set()
        if self._conversion is not None:
            self._conversion.join(timeout=5)
        self._stop_reader()
