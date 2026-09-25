"""The model applied to 8-bit BGR images: sRGB decoding, the acuity blur and the composed view."""

import dataclasses

import cv2
import numpy as np

from dog_vision.core.model import Params, simulation_matrix
from dog_vision.core.species import ACUITY

# sRGB transfer function via lookup tables; the forward LUT is exact for 8-bit input.
_DECODE = np.array(
    [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in np.arange(256) / 255.0],
    dtype=np.float32,
)
_ENCODE_STEPS = 4096
_ENCODE = np.array(
    [
        255.0 * (12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055)
        for v in np.linspace(0.0, 1.0, _ENCODE_STEPS)
    ]
).round().astype(np.uint8)


def acuity_blur(params: Params, width: int) -> tuple[float, float] | None:
    """Gaussian sigmas in pixels (x, y) matching the species' acuity, or None for no blur.

    AcuityView (Caves & Johnsen 2018) multiplies the spectrum by the modulation transfer
    function exp(-3.56 (MRA f)^2), with f in cycles per degree and MRA = 1 / acuity the
    minimum resolvable angle. That is a Gaussian of sigma = sqrt(3.56 / (2 pi^2)) MRA
    degrees, which is cheaper to apply to video than a Fourier transform.
    """
    if not params.acuity or params.species not in ACUITY:
        return None
    pixels_per_degree = width / params.field_of_view
    return tuple(np.sqrt(3.56 / (2 * np.pi**2)) / acuity * pixels_per_degree for acuity in ACUITY[params.species][0])


def apply_bgr(frame_bgr: np.ndarray, t_rgb: np.ndarray, blur: tuple[float, float] | None = None) -> np.ndarray:
    """Apply a linear-RGB 3x3 matrix, and optionally a Gaussian blur, to an 8-bit BGR image."""
    flip = np.eye(3)[::-1]  # RGB <-> BGR permutation
    t_bgr = (flip @ t_rgb @ flip).astype(np.float32)
    linear = _DECODE[frame_bgr]
    if blur is not None:  # in linear light, like the optics it stands for
        linear = cv2.GaussianBlur(linear, (0, 0), sigmaX=blur[0], sigmaY=blur[1])
    out = linear @ t_bgr.T
    np.clip(out, 0.0, 1.0, out=out)
    # Rounded, not truncated: truncating darkens values near black, where the curve is steepest,
    # so that even the identity would not give the image back.
    return _ENCODE[(out * (_ENCODE_STEPS - 1) + 0.5).astype(np.int32)]


HUMAN_JND_DELTA_E = 2.3  # CIELAB Delta E*ab of one just-noticeable difference (Mahy et al. 1994)
DIFFERENCE_FULL_RED = 10.0  # Delta E*ab at which the difference map is fully red


def lab_from_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """CIELAB (D65) of an 8-bit sRGB image in BGR order."""
    linear = _DECODE[image_bgr][..., ::-1]
    xyz = (
        linear @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32).T
    )
    t = xyz / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def difference_map(left_bgr: np.ndarray, right_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Where two images look different to a human, and the share of pixels that do.

    The map is the right image in dimmed grey, reddened where the CIELAB difference
    exceeds one just-noticeable difference, fully so at DIFFERENCE_FULL_RED.
    """
    delta_e = np.linalg.norm(lab_from_bgr(left_bgr) - lab_from_bgr(right_bgr), axis=-1)
    noticeable = delta_e > HUMAN_JND_DELTA_E
    grey = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) * 0.6
    weight = np.where(noticeable, np.clip(delta_e / DIFFERENCE_FULL_RED, 0.3, 1.0), 0.0)[..., None]
    red = np.array([40, 40, 255], np.float32)  # BGR
    image = np.repeat(grey[..., None], 3, axis=-1) * (1 - weight) + red * weight
    return image.astype(np.uint8), float(noticeable.mean())


def mean_linear_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """Mean linear RGB of an 8-bit BGR image, estimated from every 8th pixel."""
    return _DECODE[frame_bgr[::8, ::8]].reshape(-1, 3).mean(axis=0)[::-1]


def simulate(frame_bgr: np.ndarray, params: Params) -> np.ndarray:
    mean_rgb = mean_linear_rgb(frame_bgr) if params.adaptation > 0 else None
    return apply_bgr(frame_bgr, simulation_matrix(params, mean_rgb), acuity_blur(params, frame_bgr.shape[1]))


def compose(
    frame_bgr: np.ndarray, params: Params, side_by_side: bool, compare: str | None, difference: bool
) -> tuple[np.ndarray, float | None]:
    """The view as shown: the simulation, after the original or another species when side by side,
    then the map of differences if asked for; and the share of pixels that differ, if mapped."""
    simulated = simulate(frame_bgr, params)
    if not side_by_side:
        return simulated, None
    left = frame_bgr if compare is None else simulate(frame_bgr, dataclasses.replace(params, species=compare))
    if not difference:
        return np.hstack([left, simulated]), None
    diff, share = difference_map(left, simulated)
    return np.hstack([left, simulated, diff]), share
