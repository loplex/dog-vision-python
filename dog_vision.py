# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless"]
#
# [tool.uv]
# # uv's own Python builds ship a Tk without Xft, which draws text unantialiased.
# python-preference = "system"
# ///
"""Simulate dichromatic dog colour vision on a live camera feed or a photo.

Other dichromats, some trichromats and two cone monochromats are available as
presets (SPECIES).

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
   monochromat is mapped onto the grey axis instead, and a trichromat maps
   onto all of RGB: nothing merges, so without step 6 its image is unchanged.
5. Optionally the cones adapt to the scene (von Kries gains taken from the
   image mean, "grey world"), and the result is blended with the input.
6. The saturation of the animal's colour axes is either left as step 4 gives
   it ("fixed"), or scaled so that one step the animal can just discriminate is
   one step a human can ("rnl"), both judged by the receptor noise limited model
   of Vorobyev & Osorio (1998), with the same cone noise for animal and human.

The whole transform collapses into one 3x3 matrix on linear RGB. Without
adaptation its rank equals the number of cone types, and the animal's cone
excitation of every output pixel equals that of the input.

Usage:
    uv run dog_vision.py                 # live camera 0
    uv run dog_vision.py --camera 1      # another camera
    uv run dog_vision.py photo.jpg       # convert a photo, writes photo.dog.png
    uv run dog_vision.py --info          # print the derived model and checks
    uv run dog_vision.py --species cat   # another dichromat

The live window (tk_window.py) lists the species on the right; keys: m = toggle
side-by-side / simulation only, r = reset to the command-line values, s = save
snapshot, q or Esc = quit. The window only drives LiveSession, so another GUI
toolkit needs nothing but a module with the same run(session) function.
"""

import argparse
import dataclasses
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# Photopigment peak wavelengths in nm.
HUMAN_CONES = {"S": 420.7, "M": 530.3, "L": 558.9}  # Stockman & Sharpe (2000)

# Cone peaks in nm: (S, M, L) for a trichromat, (S, L) for a dichromat, (L,) for a
# cone monochromat. Species with an ultraviolet cone (mice, rats, birds) are left
# out: an RGB camera records nothing of what that cone sees.
SPECIES = {
    "dog": (429.0, 555.0),
    "cat": (450.0, 550.0),
    "horse": (428.0, 539.0),
    "cow": (451.3, 555.3),
    "sheep": (445.3, 552.2),
    "goat": (443.3, 552.5),
    "pig": (440.7, 556.7),
    "fallow-deer": (453.6, 542.2),
    "white-tailed-deer": (456.0, 536.8),
    "guinea-pig": (429.0, 529.0),
    "tree-squirrel": (444.0, 543.0),
    "ground-squirrel": (436.7, 518.9),  # the California species
    "ferret": (430.0, 558.0),
    "protanope": (HUMAN_CONES["S"], HUMAN_CONES["M"]),  # human lacking L cones
    "deuteranope": (HUMAN_CONES["S"], HUMAN_CONES["L"]),  # human lacking M cones
    "human": tuple(HUMAN_CONES.values()),
    # Anomalous trichromats of moderate severity: one cone shifted 10 nm towards the
    # other, where Machado, Oliveira & Fernandes (2009) take 20 nm as dichromacy.
    "protanomalous": (HUMAN_CONES["S"], HUMAN_CONES["M"], HUMAN_CONES["L"] - 10),
    "deuteranomalous": (HUMAN_CONES["S"], HUMAN_CONES["M"] + 10, HUMAN_CONES["L"]),
    "macaque": (431.0, 536.0, 565.0),
    "howler-monkey": (430.0, 530.0, 562.0),
    "marmoset-female": (423.0, 543.0, 563.0),  # with the 543 and 563 nm M/L alleles
    "harbour-seal": (510.0,),
    "bottlenose-dolphin": (524.0,),  # the L opsin
}

# Where each species' cone peaks come from.
PEAKS_FROM = {
    "dog": "Neitz, Geist & Jacobs 1989",
    "cat": "Guenther & Zrenner 1993",
    "horse": "Carroll et al. 2001",
    **dict.fromkeys(["cow", "sheep", "goat", "pig", "fallow-deer", "white-tailed-deer"], "Jacobs, Deegan & Neitz 1998"),
    "guinea-pig": "Jacobs & Deegan 1994",
    "tree-squirrel": "Blakeslee, Jacobs & Neitz 1988",
    "ground-squirrel": "Jacobs, Neitz & Crognale 1985",
    "ferret": "Calderone & Jacobs 2003",
    **dict.fromkeys(["protanope", "deuteranope", "human"], "Stockman & Sharpe 2000"),
    "protanomalous": "L shifted 10 nm, see text",
    "deuteranomalous": "M shifted 10 nm, see text",
    "macaque": "Bowmaker et al. 1978, 1991",
    "howler-monkey": "Jacobs et al. 1996, S assumed",
    "marmoset-female": "Travis 1988, Williams 1992",
    "harbour-seal": "Crognale et al. 1998",
    "bottlenose-dolphin": "Fasick et al. 1998",
}

# Share of S cones among all cones, as the (lowest, highest) reported across the
# retina, and where it comes from; the RNL scale uses the middle of the range.
# A species missing here gets ASSUMED_S_CONE_FRACTION.
S_CONE_FRACTION = {
    "dog": (0.10, 0.18, "Mowat et al. 2008"),  # area centralis, periphery
    "cat": (0.10, 0.20, "Linberg et al. 2001"),
    "horse": (0.10, 0.25, "Sandmann et al. 1996"),
    **dict.fromkeys(["cow", "sheep", "pig"], (0.05, 0.10, "Schiviz et al. 2008")),
    "ground-squirrel": (1 / 15, 1 / 15, "Kryger et al. 1998"),  # 14 M cones per S cone
    "ferret": (1 / 15, 1 / 15, "Calderone & Jacobs 2003"),  # 14 L cones per S cone
    **dict.fromkeys(
        ["protanope", "deuteranope", "human", "protanomalous", "deuteranomalous"], (0.08, 0.12, "Curcio et al. 1991")
    ),
}
ASSUMED_S_CONE_FRACTION = (0.10, 0.10)  # the middle of what the measured species span

# L cones per M cone for a trichromat. Humans with normal colour vision range from
# about twice as many L as M cones to the reverse (Roorda & Williams 1999), so the
# RNL scale takes 1:1 for every trichromat and calls it assumed.
ASSUMED_L_TO_M = 1.0

COLOUR_VISION = {1: "monochromat", 2: "dichromat", 3: "trichromat"}

CHROMA_SCALES = ("fixed", "rnl")


@dataclasses.dataclass
class Params:
    species: str = "dog"
    adaptation: float = 0.0  # 0 = adapted to daylight, 1 = fully to the scene mean
    strength: float = 1.0  # 0 = original image, 1 = full simulation
    chroma_scale: str = "fixed"  # one of CHROMA_SCALES

    def cones(self) -> tuple[float, ...]:
        return SPECIES[self.species]


# Output subspace per number of cone types. For three: all of RGB. For two: the
# plane R = G, whose columns are the grey-yellow and blue directions. For one:
# the grey axis.
OUTPUT_BASIS = {
    3: np.eye(3),
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


def chroma_directions(m_animal: np.ndarray) -> np.ndarray:
    """Columns: output colours that raise one non-L cone's excitation by one, leaving the rest.

    One column for a dichromat (S), two for a trichromat (S, M), none for a monochromat.
    """
    basis = OUTPUT_BASIS[len(m_animal)]
    units = np.eye(len(m_animal))[:, :-1]
    return basis @ np.linalg.solve(m_animal @ basis, units)


def s_cone_fraction(species: str) -> tuple[float, bool]:
    """The S-cone share the RNL scale uses, and whether it was measured."""
    low, high = S_CONE_FRACTION[species][:2] if species in S_CONE_FRACTION else ASSUMED_S_CONE_FRACTION
    return (low + high) / 2, species in S_CONE_FRACTION


def cone_shares(species: str) -> np.ndarray:
    """Relative abundance of each cone class, short to long, summing to one."""
    fraction, _ = s_cone_fraction(species)
    if len(SPECIES[species]) == 2:
        return np.array([fraction, 1 - fraction])
    return np.array([fraction, (1 - fraction) / (1 + ASSUMED_L_TO_M), (1 - fraction) * ASSUMED_L_TO_M / (1 + ASSUMED_L_TO_M)])


def symmetric_power(matrix: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * values**power) @ vectors.T


def rnl_metric(species: str, cone_matrix_rgb: np.ndarray) -> np.ndarray:
    """RNL discrimination of small cone contrasts around grey, as a quadratic form.

    For cone contrasts f the squared distance in JNDs is f' P' (P E P')^-1 P f
    (Vorobyev & Osorio 1998), with P taking f to the differences f_i - f_L and E the
    squared noise of each cone, e_i = w / sqrt(n_i / n_max). The Weber fraction w is
    left at 1: it scales animal and human alike, so it cancels in rnl_chroma_matrix.
    Given the cone matrix, the form is returned in the coordinates it takes.
    """
    n = len(cone_matrix_rgb)
    shares = cone_shares(species)
    noise = 1 / np.sqrt(shares / shares.max())
    to_chroma = np.hstack([np.eye(n - 1), -np.ones((n - 1, 1))])
    return cone_matrix_rgb.T @ to_chroma.T @ np.linalg.inv(to_chroma @ np.diag(noise**2) @ to_chroma.T) @ to_chroma @ cone_matrix_rgb


def rnl_chroma_matrix(params: Params) -> np.ndarray:
    """K: how the "rnl" scale remaps the chromatic coordinates c = (q_i - q_L) of the fixed output.

    The fixed output is x' = q_L w + U c, with w the white and U = chroma_directions.
    Both observers judge a small c by the RNL model around grey: the animal in its own
    cone coordinates, c' A c, and a human looking at the output, (U c)' H (U c).
    Replacing U c by U K c with K' B K = A (B = U' H U) makes the two agree. Of all such
    K this is the one that is self-adjoint in the B metric, so it rescales the axes
    without rotating hues; for a dichromat it is the single factor sqrt(A / B), and for
    a human it is the identity.
    """
    m_animal = animal_cone_matrix(params)
    n = len(m_animal)
    if n == 1:
        return np.zeros((0, 0))  # a monochromat has no chromatic axis to scale
    # U c changes the cones by (c, 0): L stays, so c' A c is the form's leading block.
    animal = rnl_metric(params.species, np.eye(n))[:-1, :-1]
    u = chroma_directions(m_animal)
    human = u.T @ rnl_metric("human", animal_cone_matrix(Params("human"))) @ u
    half, inverse_half = symmetric_power(human, 0.5), symmetric_power(human, -0.5)
    return inverse_half @ symmetric_power(half @ animal @ half, 0.5) @ inverse_half


def rnl_gains(params: Params) -> np.ndarray:
    """The factors the "rnl" scale applies along its principal axes, largest first."""
    return np.sort(np.linalg.eigvals(rnl_chroma_matrix(params)).real)[::-1]


def simulation_matrix(params: Params, mean_rgb: np.ndarray | None = None) -> np.ndarray:
    """The 3x3 linear-RGB transform for the given parameters.

    Output colours lie in the subspace spanned by B = OUTPUT_BASIS and produce the
    (adapted) cone excitation of the input: x' = B (M_animal B)^-1 diag(gains) M_animal x.
    With unit gains this is the projection of x along the animal's confusion
    directions onto that subspace (the identity for a trichromat). It equals
    x' = q_L w + U c with w the white, U = chroma_directions and c = (q_i - q_L), and
    the "rnl" chroma scale replaces U c by U K c with K = rnl_chroma_matrix.
    """
    m_animal = animal_cone_matrix(params)
    n = len(m_animal)
    basis = OUTPUT_BASIS[n]
    gains = np.ones(n) if mean_rgb is None else grey_world_gains(m_animal, mean_rgb, params.adaptation)
    adapted = np.diag(gains) @ m_animal
    if n > 1 and params.chroma_scale == "rnl":
        to_chroma = np.hstack([np.eye(n - 1), -np.ones((n - 1, 1))])
        chroma = chroma_directions(m_animal) @ rnl_chroma_matrix(params) @ to_chroma @ adapted
        simulated = np.outer(np.ones(3), adapted[-1]) + chroma
    else:
        simulated = basis @ np.linalg.solve(m_animal @ basis, adapted)
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
    print()
    for label, value in species_facts(params.species):
        print(f"  {label + ':':15s} {value}")
    print("\nReference values: the dog's neutral point was measured at about 480 nm")
    print("(Neitz, Geist & Jacobs 1989); a human deuteranope is R' = G' = 0.293 R + 0.707 G")
    print("(Viénot, Brettel & Mollon 1999).")


def species_label(species: str) -> str:
    """The species' name with its kind of colour vision, for lists."""
    return f"{species} ({COLOUR_VISION[len(SPECIES[species])]})"


def species_facts(species: str) -> list[tuple[str, str]]:
    """What the simulation knows about a species, as (label, value) rows for a table."""
    peaks = SPECIES[species]
    n = len(peaks)
    names = {3: "SML", 2: "SL", 1: "L"}[n]
    facts = [
        ("Colour vision", f"{COLOUR_VISION[n]}, {n} cone type{'s' if n > 1 else ''}"),
        ("Cone peaks", ", ".join(f"{name} {peak:g} nm" for name, peak in zip(names, peaks))),
        ("Peaks from", PEAKS_FROM[species]),
    ]
    if n == 1:
        return facts + [("RNL scale", "nothing to scale")]
    if species in S_CONE_FRACTION:
        low, high, source = S_CONE_FRACTION[species]
        share = f"{low:.0%}" if round(low * 100) == round(high * 100) else f"{low:.0%}–{high:.0%}"
        facts.append(("S cones", f"{share} of cones ({source})"))
    else:
        facts.append(("S cones", f"{ASSUMED_S_CONE_FRACTION[0]:.0%} of cones (assumed)"))
    if n == 3:
        facts.append(("L : M cones", f"{ASSUMED_L_TO_M:g} : 1 (assumed)"))
    else:
        facts.append(("Neutral point", f"{neutral_point(Params(species)):.0f} nm (model)"))
    gains = " and ".join(f"x{gain:.2f}" for gain in rnl_gains(Params(species)))
    return facts + [("RNL scale", f"{gains} of fixed")]


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


class LiveSession:
    """Everything the live window shows and does, independent of the GUI toolkit.

    The camera is read on a background thread, so a GUI can poll render() from
    its own timer without waiting for the next frame. A GUI changes params and
    side_by_side directly, and calls reset(), save_snapshot() and close().
    """

    def __init__(self, camera_index: int, initial: Params) -> None:
        self.species_names = list(SPECIES)
        self.species_labels = [species_label(name) for name in SPECIES]
        self.chroma_scales = CHROMA_SCALES
        self.initial = initial
        self.params = dataclasses.replace(initial)
        self.side_by_side = True
        self.error: str | None = None  # set when the camera stops delivering frames
        self._capture = cv2.VideoCapture(camera_index)
        if not self._capture.isOpened():
            sys.exit(f"Cannot open camera {camera_index}")
        self._frame: np.ndarray | None = None
        self._last_images: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._read_frames, daemon=True)
        self._thread.start()

    def _read_frames(self) -> None:
        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                self.error = "Camera stopped delivering frames"
                return
            with self._lock:
                self._frame = frame

    def render(self) -> np.ndarray | None:
        """The current view as an RGB image at camera resolution, or None before the first frame."""
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        simulated = simulate(frame, self.params)
        self._last_images = np.hstack([frame, simulated]) if self.side_by_side else simulated
        return self._last_images[..., ::-1]

    def reset(self) -> None:
        self.params = dataclasses.replace(self.initial)

    def species_facts(self) -> list[tuple[str, str]]:
        """What the simulation knows about the current species, as (label, value) rows."""
        return species_facts(self.params.species)

    def save_snapshot(self) -> str | None:
        """Write the last rendered view to the working directory and return the file name."""
        if self._last_images is None:
            return None
        name = f"dog-{self.params.species}-{time.strftime('%Y%m%d-%H%M%S')}.png"
        cv2.imwrite(name, self._last_images)
        return name

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1)
        self._capture.release()


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
    parser.add_argument(
        "--chroma-scale",
        choices=CHROMA_SCALES,
        default=defaults.chroma_scale,
        help="saturation of a dichromat's colour axis: fixed by the projection, or matched"
        " to the animal's discrimination with the RNL model (default %(default)s)",
    )
    args = parser.parse_args()
    params = Params(args.species, args.adaptation, args.strength, args.chroma_scale)

    if args.info:
        print_info(params)
    elif args.image:
        convert_file(args.image, params)
    else:
        import tk_window  # the only GUI-specific line in this module

        session = LiveSession(args.camera, params)
        try:
            tk_window.run(session)
        finally:
            session.close()
        if session.error:
            sys.exit(session.error)


if __name__ == "__main__":
    main()
