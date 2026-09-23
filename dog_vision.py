# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python"]
# ///
"""Simulate dichromatic dog colour vision on a live camera feed or a photo.

Other dichromats and two cone monochromats are available as presets (SPECIES).

Model (after Brettel, Viénot & Mollon 1997):

1. Every cone is modelled with the Govardovskii et al. (2000) A1 visual-pigment
   template, parametrised only by its peak wavelength.
2. The display is modelled as three Gaussian primaries (typical sRGB LCD),
   scaled so that RGB (1,1,1) excites the human cones like D65 daylight.
3. M_animal (2x3 for a dichromat) maps linear RGB to the animal's S and L cone
   excitations. Its one-dimensional null space n is the direction along which
   colours differ only in ways the animal cannot see.
4. Each pixel is moved along n into the plane R = G. That plane contains the
   grey axis (so neutral colours stay neutral) and the blue-yellow axis (the
   conventional rendering of a dichromat's single chromatic axis). A cone
   monochromat is mapped onto the grey axis instead.
5. Optionally the cones adapt to the scene (von Kries gains taken from the
   image mean, "grey world"), and the result is blended with the input.

The whole transform collapses into one 3x3 matrix on linear RGB. Without
adaptation its rank equals the number of cone types, and the animal's cone
excitation of every output pixel equals that of the input.

Usage:
    uv run dog_vision.py                 # live camera 0
    uv run dog_vision.py --camera 1      # another camera
    uv run dog_vision.py photo.jpg       # convert a photo, writes photo.dog.png
    uv run dog_vision.py --info          # print the derived model and checks
    uv run dog_vision.py --species cat   # another dichromat

Keys in the live window: m = toggle side-by-side / simulation only, s = save
snapshot, q or Esc = quit.
"""

import argparse
import dataclasses
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Photopigment peak wavelengths in nm.
HUMAN_CONES = {"S": 420.7, "M": 530.3, "L": 558.9}  # Stockman & Sharpe (2000)

# Cone peaks in nm: (S, L) for a dichromat, (L,) for a cone monochromat.
# Species with an ultraviolet cone (mice, rats, birds) are left out: an RGB
# camera records nothing of what that cone sees.
SPECIES = {
    "dog": (429.0, 555.0),  # Neitz, Geist & Jacobs (1989)
    "cat": (450.0, 550.0),  # Guenther & Zrenner (1993)
    "horse": (428.0, 539.0),  # Carroll et al. (2001)
    "cow": (451.3, 555.3),  # this and the next five: Jacobs, Deegan & Neitz (1998), table 1
    "sheep": (445.3, 552.2),
    "goat": (443.3, 552.5),
    "pig": (440.7, 556.7),
    "fallow-deer": (453.6, 542.2),
    "white-tailed-deer": (456.0, 536.8),
    "guinea-pig": (429.0, 529.0),  # Jacobs & Deegan (1994)
    "tree-squirrel": (444.0, 543.0),  # Blakeslee, Jacobs & Neitz (1988)
    "ground-squirrel": (436.7, 518.9),  # Jacobs, Neitz & Crognale (1985), California species
    "ferret": (430.0, 558.0),  # Calderone & Jacobs (2003)
    "protanope": (HUMAN_CONES["S"], HUMAN_CONES["M"]),  # human lacking L cones
    "deuteranope": (HUMAN_CONES["S"], HUMAN_CONES["L"]),  # human lacking M cones
    "harbour-seal": (510.0,),  # Crognale et al. (1998)
    "bottlenose-dolphin": (524.0,),  # Fasick et al. (1998), L opsin
}


@dataclasses.dataclass
class Params:
    species: str = "dog"
    adaptation: float = 0.0  # 0 = adapted to daylight, 1 = fully to the scene mean
    strength: float = 1.0  # 0 = original image, 1 = full simulation

    def cones(self) -> tuple[float, ...]:
        return SPECIES[self.species]


# Output subspace per number of cone types. For two: the plane R = G, whose
# columns are the grey-yellow and blue directions. For one: the grey axis.
OUTPUT_BASIS = {
    2: np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    1: np.ones((3, 1)),
}

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


def cone_matrix(cone_peaks: tuple[float, ...], primaries: np.ndarray) -> np.ndarray:
    """Cone excitations (rows) produced by each display primary (columns)."""
    sens = np.stack([govardovskii_a1(p) for p in cone_peaks])
    return sens @ primaries.T


def display_primaries() -> np.ndarray:
    """Primary spectra (3 x wavelengths), white-balanced to D65 for a human."""
    raw = np.stack(
        [np.exp(-0.5 * ((WAVELENGTHS - peak) / sigma) ** 2) for peak, sigma in DISPLAY_PRIMARIES.values()]
    )
    human = cone_matrix(tuple(HUMAN_CONES.values()), raw)
    human_d65 = np.stack([govardovskii_a1(p) for p in HUMAN_CONES.values()]) @ planck(6504.0)
    scales = np.linalg.solve(human, human_d65)
    return raw * scales[:, None]


def animal_cone_matrix(params: Params) -> np.ndarray:
    """M_animal (cones x 3): cone excitations from linear RGB, white mapping to all ones."""
    m_animal = cone_matrix(params.cones(), display_primaries())
    return m_animal / m_animal.sum(axis=1, keepdims=True)  # von Kries adaptation to daylight


def grey_world_gains(m_animal: np.ndarray, mean_rgb: np.ndarray, adaptation: float) -> np.ndarray:
    """Von Kries gains that move the scene mean towards neutral as adaptation goes 0 -> 1."""
    mean_cones = np.maximum(m_animal @ mean_rgb, 1e-6)
    return (mean_cones.mean() / mean_cones) ** adaptation


def simulation_matrix(params: Params, mean_rgb: np.ndarray | None = None) -> np.ndarray:
    """The 3x3 linear-RGB transform for the given parameters.

    Output colours lie in the subspace spanned by B = OUTPUT_BASIS and produce the
    (adapted) cone excitation of the input: x' = B (M_animal B)^-1 diag(gains) M_animal x.
    With unit gains this is the projection of x along the animal's confusion
    directions onto that subspace.
    """
    m_animal = animal_cone_matrix(params)
    basis = OUTPUT_BASIS[len(m_animal)]
    gains = np.ones(len(m_animal)) if mean_rgb is None else grey_world_gains(m_animal, mean_rgb, params.adaptation)
    simulated = basis @ np.linalg.solve(m_animal @ basis, np.diag(gains) @ m_animal)
    return (1.0 - params.strength) * np.eye(3) + params.strength * simulated


def neutral_point(params: Params) -> float:
    """Wavelength of monochromatic light a dichromat sees with the same S/L ratio as white."""
    sens = np.stack([govardovskii_a1(p) for p in params.cones()])
    white = cone_matrix(params.cones(), display_primaries()) @ np.ones(3)
    ratio = sens[0] / sens[1] - white[0] / white[1]
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


def print_info(params: Params) -> None:
    m_animal = animal_cone_matrix(params)
    t = simulation_matrix(dataclasses.replace(params, strength=1.0))
    np.set_printoptions(precision=4, suppress=True)
    print(f"Species {params.species}, cone peaks {params.cones()} nm")
    print("\nCone matrix M_animal (rows: cones, short to long; columns: linear R, G, B):")
    print(m_animal)
    if len(m_animal) == 2:
        n = np.cross(m_animal[0], m_animal[1])  # spans the null space of a 2x3 matrix
        print("\nConfusion direction n (normalised):", n / np.abs(n).max())
    print("\nSimulation matrix T (linear RGB):")
    print(t)
    print("\nChecks:")
    print("  grey preserved   T @ (1,1,1) =", t @ np.ones(3))
    print("  cone-invariant   max |M_animal @ T - M_animal| =", np.abs(m_animal @ t - m_animal).max())
    if len(m_animal) == 2:
        print(f"  neutral point    {neutral_point(params):.0f} nm")
    print("\nReference values: the dog's neutral point was measured at about 480 nm")
    print("(Neitz, Geist & Jacobs 1989); a human deuteranope is R' = G' = 0.293 R + 0.707 G")
    print("(Viénot, Brettel & Mollon 1999).")


def mean_linear_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """Mean linear RGB of an 8-bit BGR image, estimated from every 8th pixel."""
    return _DECODE[frame_bgr[::8, ::8]].reshape(-1, 3).mean(axis=0)[::-1]


def simulate(frame_bgr: np.ndarray, params: Params) -> np.ndarray:
    mean_rgb = mean_linear_rgb(frame_bgr) if params.adaptation > 0 else None
    return apply_bgr(frame_bgr, simulation_matrix(params, mean_rgb))


def convert_file(path: Path, params: Params) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        sys.exit(f"Cannot read image: {path}")
    out_path = path.with_suffix(".dog.png")
    cv2.imwrite(str(out_path), simulate(image, params))
    print(f"Wrote {out_path}")


def use_system_fonts() -> None:
    """Point Qt at a system font directory.

    The opencv-python wheel bundles Qt without any fonts, and importing cv2 sets
    QT_QPA_FONTDIR to the missing cv2/qt/fonts directory, so toolbar tooltips,
    trackbar labels and the status bar render blank. An existing directory is kept.
    """
    if Path(os.environ.get("QT_QPA_FONTDIR", "")).is_dir():
        return
    try:
        font = subprocess.run(["fc-match", "-f", "%{file}", "sans"], capture_output=True, text=True).stdout
    except FileNotFoundError:  # no fontconfig, e.g. on Windows
        return
    if font and Path(font).is_file():
        os.environ["QT_QPA_FONTDIR"] = str(Path(font).parent)


def run_camera(index: int, params: Params) -> None:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {index}")
    window = "Dog vision (m = mode, s = snapshot, q = quit)"
    use_system_fonts()
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    side_by_side = True
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                sys.exit("Camera stopped delivering frames")
            simulated = simulate(frame, params)
            view = np.hstack([frame, simulated]) if side_by_side else simulated
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
    defaults = Params()
    parser.add_argument("--species", choices=SPECIES, default=defaults.species, help="animal to simulate (default %(default)s)")
    parser.add_argument(
        "--adaptation", type=float, default=defaults.adaptation, help="adaptation to the scene mean, 0-1 (default %(default)s)"
    )
    parser.add_argument(
        "--strength", type=float, default=defaults.strength, help="0 = original, 1 = full simulation (default %(default)s)"
    )
    args = parser.parse_args()
    params = Params(args.species, args.adaptation, args.strength)

    if args.info:
        print_info(params)
    elif args.image:
        convert_file(args.image, params)
    else:
        run_camera(args.camera, params)


if __name__ == "__main__":
    main()
