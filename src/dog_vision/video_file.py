"""Writing a converted video with the best method the system has.

With ffmpeg installed, the first encoder in FFMPEG_ENCODERS that actually encodes a
frame on this machine is used, and the sound of the original is carried over. A
listed encoder can still fail: a hardware one needs its hardware. Without ffmpeg,
OpenCV writes the video with the best codec its build has, and without sound, since
OpenCV does not handle audio.
"""

import functools
import shutil
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np

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
    "none" (the original has none), "lost" (no ffmpeg) or "unknown" (no ffprobe to tell).
    """

    def __init__(self, path: Path, sound_from: Path, width: int, height: int, fps: float) -> None:
        self.path = path
        found = best_ffmpeg_encoder()
        self._process: subprocess.Popen | None = None
        self._writer: cv2.VideoWriter | None = None
        if found is None:
            self._open_opencv(width, height, fps)
            self.sound = "lost"
            return
        ffmpeg, (self.encoder, self.format_name, options) = found
        options = list(options or ["-b:v", str(round(width * height * fps * BITS_PER_PIXEL))])
        if self.format_name == "H.265":
            options += ["-tag:v", "hvc1"]  # the tag Apple's players need to play it
        codec = audio_codec(sound_from)
        self.sound = {None: "unknown", "": "none"}.get(codec, "kept")
        command = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}"]
        command += ["-r", f"{fps:.6g}", "-i", "-", "-i", str(sound_from), "-map", "0:v", "-map", "1:a:0?"]
        command += ["-vf", VIDEO_FILTER, "-c:v", self.encoder, *options, *BT709_TAGS]
        command += ["-c:a", "copy" if codec in MP4_AUDIO else "aac", "-movflags", "+faststart", str(path)]
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
