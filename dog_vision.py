# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python"]
# ///
"""Simulate dichromatic dog colour vision on a live camera feed or a photo.

Model (after Brettel, Viénot & Mollon 1997):

1. Every cone is modelled with the Govardovskii et al. (2000) A1 visual-pigment
   template, parametrised only by its peak wavelength.
2. The display is modelled as three Gaussian primaries (typical sRGB LCD),
   scaled so that RGB (1,1,1) excites the human cones like D65 daylight.
3. M_dog (2x3) maps linear RGB to the dog's S and L cone excitations. Its
   one-dimensional null space n is the direction along which colours differ
   only in ways the dog cannot see.
4. Each pixel is moved along n into the plane R = G. That plane contains the
   grey axis (so neutral colours stay neutral) and the blue-yellow axis (the
   conventional rendering of a dichromat's single chromatic axis).

The whole transform collapses into one rank-2 3x3 matrix on linear RGB, and by
construction the dog excitation of every output pixel equals that of the input.

Usage:
    uv run dog_vision.py                 # live camera 0
    uv run dog_vision.py --camera 1      # another camera
    uv run dog_vision.py photo.jpg       # convert a photo, writes photo.dog.png
    uv run dog_vision.py --info          # print the derived model and checks

Keys in the live window: m = toggle side-by-side / dog only, s = save
snapshot, q or Esc = quit.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Photopigment peak wavelengths in nm.
HUMAN_CONES = {"S": 420.7, "M": 530.3, "L": 558.9}  # Stockman & Sharpe (2000)
DOG_CONES = {"S": 429.0, "L": 555.0}  # Neitz, Geist & Jacobs (1989)

# Gaussian approximation of a typical sRGB LCD: (peak nm, sigma nm).
DISPLAY_PRIMARIES = {"R": (610.0, 20.0), "G": (540.0, 18.0), "B": (450.0, 10.0)}

WAVELENGTHS = np.arange(380.0, 781.0, 1.0)


def govardovskii_a1(lambda_max: float, wl: np.ndarray = WAVELENGTHS) -> np.ndarray:
    """Relative spectral sensitivity of an A1 (retinal) visual pigment."""
    x = lambda_max / wl
    a = 0.8795 + 0.0459 * np.exp(-((lambda_max - 300.0) ** 2) / 11940.0)
    alpha = 1.0 / (
        np.exp(69.7 * (a - x))
        + np.exp(28.0 * (0.922 - x))
        + np.exp(-14.9 * (1.104 - x))
        + 0.674
    )
    beta_peak = 189.0 + 0.315 * lambda_max
    beta_width = -40.5 + 0.195 * lambda_max
    beta = 0.26 * np.exp(-(((wl - beta_peak) / beta_width) ** 2))
    return alpha + beta


def planck(temperature: float, wl: np.ndarray = WAVELENGTHS) -> np.ndarray:
    """Black-body spectrum; at 6504 K a stand-in for D65 daylight."""
    wl_m = wl * 1e-9
    h, c, k = 6.626e-34, 2.998e8, 1.381e-23
    return 1.0 / (wl_m**5 * (np.exp(h * c / (wl_m * k * temperature)) - 1.0))


def cone_matrix(cone_peaks: dict[str, float], primaries: np.ndarray) -> np.ndarray:
    """Cone excitations (rows) produced by each display primary (columns)."""
    sens = np.stack([govardovskii_a1(p) for p in cone_peaks.values()])
    return sens @ primaries.T


def display_primaries() -> np.ndarray:
    """Primary spectra (3 x wavelengths), white-balanced to D65 for a human."""
    raw = np.stack(
        [np.exp(-0.5 * ((WAVELENGTHS - peak) / sigma) ** 2) for peak, sigma in DISPLAY_PRIMARIES.values()]
    )
    human = cone_matrix(HUMAN_CONES, raw)
    human_d65 = np.stack([govardovskii_a1(p) for p in HUMAN_CONES.values()]) @ planck(6504.0)
    scales = np.linalg.solve(human, human_d65)
    return raw * scales[:, None]


def dog_simulation_matrix() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (T, M_dog, n): the 3x3 linear-RGB transform, dog cone matrix and its null vector."""
    m_dog = cone_matrix(DOG_CONES, display_primaries())
    m_dog /= m_dog.sum(axis=1, keepdims=True)  # von Kries: white excites each cone to 1
    n = np.cross(m_dog[0], m_dog[1])  # spans the null space of a 2x3 matrix
    c = np.array([1.0, -1.0, 0.0])  # normal of the target plane R = G
    t = np.eye(3) - np.outer(n, c) / (c @ n)
    return t, m_dog, n


def neutral_point() -> float:
    """Wavelength of monochromatic light the dog sees with the same S/L ratio as white."""
    dog_sens = np.stack([govardovskii_a1(p) for p in DOG_CONES.values()])
    white = cone_matrix(DOG_CONES, display_primaries()) @ np.ones(3)
    ratio = dog_sens[0] / dog_sens[1] - white[0] / white[1]
    visible = (WAVELENGTHS > 420) & (WAVELENGTHS < 600)
    idx = np.where(visible[:-1] & (np.sign(ratio[:-1]) != np.sign(ratio[1:])))[0][0]
    return float(WAVELENGTHS[idx])


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


def apply_bgr(frame_bgr: np.ndarray, t_rgb: np.ndarray) -> np.ndarray:
    """Apply a linear-RGB 3x3 matrix to an 8-bit BGR image."""
    flip = np.eye(3)[::-1]  # RGB <-> BGR permutation
    t_bgr = (flip @ t_rgb @ flip).astype(np.float32)
    linear = _DECODE[frame_bgr]
    out = linear @ t_bgr.T
    np.clip(out, 0.0, 1.0, out=out)
    return _ENCODE[(out * (_ENCODE_STEPS - 1)).astype(np.int32)]


def print_info() -> None:
    t, m_dog, n = dog_simulation_matrix()
    np.set_printoptions(precision=4, suppress=True)
    print("Dog cone matrix M_dog (rows S, L; columns linear R, G, B):")
    print(m_dog)
    print("\nConfusion direction n (normalised):", n / np.abs(n).max())
    print("\nSimulation matrix T (linear RGB):")
    print(t)
    print("\nChecks:")
    print("  grey preserved   T @ (1,1,1) =", t @ np.ones(3))
    print("  dog-invariant    max |M_dog @ T - M_dog| =", np.abs(m_dog @ t - m_dog).max())
    print(f"  neutral point    {neutral_point():.0f} nm (measured in dogs: ~480 nm)")
    print("\nFor comparison, human deuteranope (Viénot et al. 1999): R' = G' = 0.293 R + 0.707 G")


def convert_file(path: Path, t: np.ndarray) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        sys.exit(f"Cannot read image: {path}")
    out_path = path.with_suffix(".dog.png")
    cv2.imwrite(str(out_path), apply_bgr(image, t))
    print(f"Wrote {out_path}")


def run_camera(index: int, t: np.ndarray) -> None:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {index}")
    window = "Dog vision (m = mode, s = snapshot, q = quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    side_by_side = True
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                sys.exit("Camera stopped delivering frames")
            dog = apply_bgr(frame, t)
            view = np.hstack([frame, dog]) if side_by_side else dog
            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("m"):
                side_by_side = not side_by_side
            elif key == ord("s"):
                name = f"dog-{time.strftime('%Y%m%d-%H%M%S')}.png"
                cv2.imwrite(name, view)
                print(f"Saved {name}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", type=Path, help="photo to convert instead of using the camera")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--info", action="store_true", help="print the derived model and exit")
    args = parser.parse_args()

    if args.info:
        print_info()
        return
    t, _, _ = dog_simulation_matrix()
    if args.image:
        convert_file(args.image, t)
    else:
        run_camera(args.camera, t)


if __name__ == "__main__":
    main()
