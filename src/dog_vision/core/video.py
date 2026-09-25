"""Converting a photo or a video, recording the window's view, and writing video with the best method the system has.

With ffmpeg installed, the first encoder in FFMPEG_ENCODERS that actually encodes a
frame on this machine is used, and the sound of the original is carried over. A
listed encoder can still fail: a hardware one needs its hardware. Without ffmpeg,
OpenCV writes the video with the best codec its build has, and without sound, since
OpenCV does not handle audio.
"""

import functools
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from dog_vision.core import i18n
from dog_vision.core.i18n import N_

# ffmpeg encoders, best first: H.265 before H.264, software before hardware, as the
# software encoders give the better picture for the size. None stands for a hardware
# encoder, which gets a bitrate (BITS_PER_PIXEL) instead of a quality target.
FFMPEG_ENCODERS = [
    ("libx265", "H.265", ["-crf", "24", "-preset", "medium", "-x265-params", "log-level=error"]),
    ("hevc_videotoolbox", "H.265", None),  # macOS
    ("hevc_nvenc", "H.265", None),  # NVIDIA
    ("hevc_qsv", "H.265", None),  # Intel
    ("hevc_amf", "H.265", None),  # AMD, on Windows
    ("libx264", "H.264", ["-crf", "20", "-preset", "medium"]),
    ("h264_videotoolbox", "H.264", None),
    ("h264_nvenc", "H.264", None),
    ("h264_qsv", "H.264", None),
    ("h264_amf", "H.264", None),
    ("mpeg4", "MPEG-4", ["-q:v", "3"]),
]
BITS_PER_PIXEL = 0.1  # of every frame, for the encoders without a quality target
RECORDING_FPS = 30  # frames per second a recording is written at, whatever rate its frames come at
RECORDING_QUEUE = 8  # frames a recording holds for its encoder before it drops the newest

# OpenCV's own codecs, best first, for a system without ffmpeg.
OPENCV_CODECS = [("hvc1", "H.265"), ("avc1", "H.264"), ("mp4v", "MPEG-4")]

# Audio codecs an .mp4 holds as they are; any other sound is re-encoded to AAC.
MP4_AUDIO = {"aac", "mp3", "ac3", "eac3", "opus", "flac", "alac"}

# The frames are sRGB, whose primaries are BT.709's. Encoding them with the BT.709 matrix,
# and saying so, keeps players from reading them with BT.601's and shifting every colour.
# swscale's default fast rounding darkens every colour by up to 5 of 255; accurate_rnd and
# full_chroma_int bring that to the 1 or 2 that 8-bit 4:2:0 costs anyway. H.265 and H.264
# need even dimensions, which the padding provides.
VIDEO_FILTER = (
    "pad=ceil(iw/2)*2:ceil(ih/2)*2,"
    "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p"
)
BT709_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]


@functools.cache
def best_ffmpeg_encoder() -> tuple[str, tuple[str, str, list[str] | None]] | None:
    """ffmpeg's path and the first of FFMPEG_ENCODERS that works here, or None without ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    listed = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False).stdout
    for encoder, name, options in FFMPEG_ENCODERS:
        if f" {encoder} " not in listed:
            continue
        test = [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=256x256:d=0.1", "-frames:v", "1"]
        test += ["-vf", VIDEO_FILTER, "-c:v", encoder, *(options or ["-b:v", "1M"]), "-f", "null", "-"]
        try:
            if subprocess.run(test, capture_output=True, timeout=30, check=False).returncode == 0:
                return ffmpeg, (encoder, name, options)
        except subprocess.TimeoutExpired:
            pass
    return None


def audio_codec(path: Path) -> str | None:
    """The codec of the file's first sound track, "" if it has none, None if that cannot be told."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    command = [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0"]
    result = subprocess.run([*command, str(path)], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


class VideoWriter:
    """An .mp4 fed BGR frames, carrying over the sound of the video they come from where it can.

    After it is made, format_name and encoder say how it is written, and sound is "kept",
    "none" (the original has none), "lost" (no ffmpeg), "unknown" (no ffprobe to tell) or
    "silent" (no video to take it from, as for a recording).
    """

    def __init__(self, path: Path, sound_from: Path | None, width: int, height: int, fps: float) -> None:
        self.path = path
        found = best_ffmpeg_encoder()
        self._process: subprocess.Popen | None = None
        self._writer: cv2.VideoWriter | None = None
        if found is None:
            self._open_opencv(width, height, fps)
            self.sound = "lost" if sound_from is not None else "silent"
            return
        ffmpeg, (self.encoder, self.format_name, options) = found
        options = list(options or ["-b:v", str(round(width * height * fps * BITS_PER_PIXEL))])
        if self.format_name == "H.265":
            options += ["-tag:v", "hvc1"]  # the tag Apple's players need to play it
        command = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}"]
        command += ["-r", f"{fps:.6g}", "-i", "-"]
        if sound_from is None:
            self.sound = "silent"
            sound = []
        else:
            codec = audio_codec(sound_from)
            self.sound = {None: "unknown", "": "none"}.get(codec, "kept")
            command += ["-i", str(sound_from), "-map", "0:v", "-map", "1:a:0?"]
            sound = ["-c:a", "copy" if codec in MP4_AUDIO else "aac"]
        command += ["-vf", VIDEO_FILTER, "-c:v", self.encoder, *options, *BT709_TAGS]
        command += [*sound, "-movflags", "+faststart", str(path)]
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        # Drained on a thread of its own, so that a chatty ffmpeg cannot fill the pipe and stall.
        self._errors: list[bytes] = []
        self._drain = threading.Thread(target=lambda: self._errors.append(self._process.stderr.read()), daemon=True)
        self._drain.start()

    def _open_opencv(self, width: int, height: int, fps: float) -> None:
        level = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)  # a codec it lacks is no news
        try:
            for fourcc, name in OPENCV_CODECS:
                writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
                if writer.isOpened():
                    self._writer, self.encoder, self.format_name = writer, f"OpenCV {fourcc}", name
                    return
                writer.release()
        finally:
            cv2.utils.logging.setLogLevel(level)
        raise RuntimeError("OpenCV cannot write video on this system, and ffmpeg is not installed")

    def write(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)
            return
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            self._fail()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            return
        self._process.stdin.close()
        if self._process.wait() != 0:
            self._fail()
        self._drain.join()

    def abort(self) -> None:
        """Stop writing and remove the unfinished file."""
        if self._writer is not None:
            self._writer.release()
        else:
            self._process.kill()
            self._process.wait()
        self.path.unlink(missing_ok=True)

    def _fail(self) -> None:
        self._process.wait()
        self._drain.join()
        message = b"".join(self._errors).decode(errors="replace").strip()
        raise RuntimeError(f"ffmpeg ({self.encoder}) failed: {message}")


class Recorder:
    """An .mp4 of frames handed over as they are shown, written at RECORDING_FPS on a thread of its own.

    Each frame is repeated for as long as it was shown, so the video keeps the clock's pace
    whether frames come faster or slower than that. A frame that comes while the encoder is
    behind is dropped, and the one before it shown longer; so is one of another size than the
    first, which a video cannot change. After stop() the thread finishes the file by itself:
    done says when, and writer or error how it went.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.monotonic()
        self.writer: VideoWriter | None = None
        self.error: str | None = None
        self._frames: queue.Queue[tuple[float, np.ndarray]] = queue.Queue(maxsize=RECORDING_QUEUE)
        self._stopped_at: float | None = None
        self._first_at = 0.0  # when the first frame came, the video's start
        self._shown: np.ndarray | None = None
        self._written = 0
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._thread.start()

    def add(self, frame: np.ndarray) -> None:
        """Show frame from now on; it must not be changed afterwards."""
        try:
            self._frames.put_nowait((time.monotonic(), frame))
        except queue.Full:
            pass

    def stop(self) -> None:
        self._stopped_at = time.monotonic()

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()

    def join(self, timeout: float) -> None:
        self._thread.join(timeout)

    def _record(self) -> None:
        try:
            while True:
                try:
                    at, frame = self._frames.get(timeout=0.05)
                except queue.Empty:
                    if self._stopped_at is not None:
                        break
                    continue
                self._show(at, frame)
            if self._shown is None:
                raise RuntimeError("No frame was shown while recording")
            self._write_until(self._stopped_at)
            if self._written == 0:  # stopped within half a frame of the first
                self.writer.write(self._shown)
            self.writer.close()
        except (RuntimeError, OSError) as error:  # kept for the window, not lost with the thread
            self.error = str(error)
            if self.writer is not None:
                self.writer.abort()

    def _show(self, at: float, frame: np.ndarray) -> None:
        if self._shown is None:
            self._first_at = at
            self.writer = VideoWriter(self.path, None, frame.shape[1], frame.shape[0], RECORDING_FPS)
        else:
            self._write_until(at)
            if frame.shape != self._shown.shape:
                return
        self._shown = frame

    def _write_until(self, at: float) -> None:
        """Repeat the frame shown for every frame of the video that falls before the time at."""
        due = round((at - self._first_at) * RECORDING_FPS)
        while self._written < due:
            self.writer.write(self._shown)
            self._written += 1


def converted_path(path: Path, is_video: bool) -> Path:
    return path.with_suffix(".dog.mp4" if is_video else ".dog.png")


def convert_video(path: Path, out_path: Path, render, progress=None, cancelled=None) -> VideoWriter | None:
    """Write render(frame) for every frame of the video at path, at its frame rate.

    progress, if given, is called with the share done; cancelled, if given, is asked
    before each frame and makes the conversion stop and remove its file. Returns the
    writer, which says how the video was written, or None if cancelled.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot read video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    writer = None
    try:
        done = 0
        while True:
            if cancelled is not None and cancelled():
                if writer is not None:
                    writer.abort()
                return None
            ok, frame = capture.read()
            if not ok:
                break
            image = render(frame)
            if writer is None:
                writer = VideoWriter(out_path, path, image.shape[1], image.shape[0], fps)
            writer.write(image)
            done += 1
            if progress is not None:
                progress(min(1.0, done / total))
        if writer is None:
            raise RuntimeError(f"No frames in video: {path}")
        writer.close()
        return writer
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    finally:
        capture.release()


def writer_description(writer: VideoWriter, language: str = "en") -> str:
    """How a video was written, e.g. "H.265 (libx265), with the original sound"."""
    template = {
        "kept": N_("{format} ({encoder}), with the original sound"),
        "none": N_("{format} ({encoder}); the original has no sound"),
        "lost": N_("{format} ({encoder}), without sound: ffmpeg is not installed"),
        "unknown": N_("{format} ({encoder})"),
        "silent": N_("{format} ({encoder})"),
    }[writer.sound]
    return i18n.translate(template, language).format(format=writer.format_name, encoder=writer.encoder)
