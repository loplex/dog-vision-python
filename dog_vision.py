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
   Optionally the image is also blurred to the animal's visual acuity, as
   AcuityView does (Caves & Johnsen 2018), for a given field of view.
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
    uv run dog_vision.py --species cat --compare dog   # two species side by side

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

# Visual acuity in cycles per degree, as (resolving detail side by side, resolving
# detail one above another) — the two differ only for cattle, whose pupil is a
# horizontal oval — and where it comes from. A species missing here was not found
# measured, and is left sharp.
HUMAN_ACUITY = ((72.0, 72.0), "Land & Nilsson 2012")
ACUITY = {
    "dog": ((11.6, 11.6), "Odom et al. 1983"),
    "cat": ((10.0, 10.0), "Wässle 1971"),
    "horse": ((23.3, 23.3), "Timney & Keil 1992"),
    "cow": ((2.6, 1.6), "Rehkämper et al. 2000"),
    "sheep": ((12.85, 12.85), "Sumita et al. 2013, 11.7 to 14"),
    "tree-squirrel": ((2.8, 2.8), "Jacobs, Birch & Blakeslee 1982, 1.8 to 3.8"),
    "ground-squirrel": ((4.0, 4.0), "Jacobs et al. 1980"),
    **dict.fromkeys(["protanope", "deuteranope", "human", "protanomalous", "deuteranomalous"], HUMAN_ACUITY),
    "macaque": ((60.0, 60.0), "Nature Neuroscience 2024, about 60"),
    "marmoset-female": ((30.0, 30.0), "Troilo, Howland & Judge 1993"),
    "harbour-seal": ((5.5, 5.5), "Hanke & Dehnhardt 2009, in air"),
    "bottlenose-dolphin": ((60 / (2 * 8.2), 60 / (2 * 8.2)), "Herman et al. 1975, 8.2 arcmin stripes"),
}
DEFAULT_FIELD_OF_VIEW = 60.0  # degrees across the image; a common camera, an assumption for photos

CHROMA_SCALES = ("fixed", "rnl")


@dataclasses.dataclass
class Params:
    species: str = "dog"
    adaptation: float = 0.0  # 0 = adapted to daylight, 1 = fully to the scene mean
    strength: float = 1.0  # 0 = original image, 1 = full simulation
    chroma_scale: str = "fixed"  # one of CHROMA_SCALES
    acuity: bool = False  # blur to the species' visual acuity
    field_of_view: float = DEFAULT_FIELD_OF_VIEW  # degrees the image spans horizontally

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
    Replacing U c by U K c with K' B K = A (B = U' H U) makes the two agree. For a
    dichromat that fixes K as the single factor sqrt(A / B); for a trichromat it leaves
    a rotation free, spent on keeping the hue of the blue-yellow direction. For a
    human K is the identity.
    """
    m_animal = animal_cone_matrix(params)
    n = len(m_animal)
    if n == 1:
        return np.zeros((0, 0))  # a monochromat has no chromatic axis to scale
    # U c changes the cones by (c, 0): L stays, so c' A c is the form's leading block.
    animal = rnl_metric(params.species, np.eye(n))[:-1, :-1]
    u = chroma_directions(m_animal)
    human = u.T @ rnl_metric("human", animal_cone_matrix(Params("human"))) @ u
    if n == 2:
        return np.sqrt(animal / human)
    # Any K with K'BK = A is B^-1/2 Q A^1/2 for a rotation Q. Q is chosen so that K maps
    # the blue-yellow direction onto itself: blue and yellow keep their hue, as in the
    # fixed scale, and only their saturation changes.
    blue_yellow = np.hstack([np.eye(n - 1), -np.ones((n - 1, 1))]) @ m_animal @ np.array([-0.5, -0.5, 1.0])
    a, b = symmetric_power(animal, 0.5) @ blue_yellow, symmetric_power(human, 0.5) @ blue_yellow
    angle = np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return symmetric_power(human, -0.5) @ rotation @ symmetric_power(animal, 0.5)


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
    if len(m_animal) > 1:
        k, u = rnl_chroma_matrix(params), chroma_directions(m_animal)
        animal = rnl_metric(params.species, np.eye(len(m_animal)))[:-1, :-1]
        human = u.T @ rnl_metric("human", animal_cone_matrix(Params("human"))) @ u
        print("  rnl matches JNDs  max |K'BK - A| / |A| =", np.abs(k.T @ human @ k - animal).max() / np.abs(animal).max())
        if len(m_animal) == 3:
            blue_yellow = np.hstack([np.eye(2), -np.ones((2, 1))]) @ m_animal @ np.array([-0.5, -0.5, 1.0])
            mapped = k @ blue_yellow
            print("  rnl keeps blue    angle of K d to d =", abs(np.arctan2(*mapped[::-1]) - np.arctan2(*blue_yellow[::-1])))
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
        return facts + [("RNL scale", "nothing to scale"), acuity_fact(species)]
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
    return facts + [("RNL scale", f"{gains} of fixed"), acuity_fact(species)]


def acuity_fact(species: str) -> tuple[str, str]:
    if species not in ACUITY:
        return ("Acuity", "not found measured; left sharp")
    (across, up), source = ACUITY[species]
    value = f"{across:.3g} c/deg" if across == up else f"{across:.3g} c/deg side by side, {up:.3g} one above another"
    return ("Acuity", f"{value} ({source})")


def mean_linear_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """Mean linear RGB of an 8-bit BGR image, estimated from every 8th pixel."""
    return _DECODE[frame_bgr[::8, ::8]].reshape(-1, 3).mean(axis=0)[::-1]


def simulate(frame_bgr: np.ndarray, params: Params) -> np.ndarray:
    mean_rgb = mean_linear_rgb(frame_bgr) if params.adaptation > 0 else None
    return apply_bgr(frame_bgr, simulation_matrix(params, mean_rgb), acuity_blur(params, frame_bgr.shape[1]))


def convert_file(path: Path, params: Params, compare: str | None = None) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        sys.exit(f"Cannot read image: {path}")
    out_path = path.with_suffix(".dog.png")
    simulated = simulate(image, params)
    if compare is not None:
        simulated = np.hstack([simulate(image, dataclasses.replace(params, species=compare)), simulated])
    cv2.imwrite(str(out_path), simulated)
    print(f"Wrote {out_path}")


class LiveSession:
    """Everything the live window shows and does, independent of the GUI toolkit.

    The camera is read on a background thread, so a GUI can poll render() from
    its own timer without waiting for the next frame. A GUI changes params,
    side_by_side and compare directly, and calls reset(), save_snapshot() and close().
    """

    def __init__(self, camera_index: int, initial: Params, compare: str | None = None) -> None:
        self.species_names = list(SPECIES)
        self.species_labels = [species_label(name) for name in SPECIES]
        self.chroma_scales = CHROMA_SCALES
        self.initial = initial
        self.params = dataclasses.replace(initial)
        self.side_by_side = True
        self.initial_compare = compare
        self.compare = compare  # the species on the left instead of the original, if any
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
        if self.side_by_side:
            left = frame if self.compare is None else simulate(frame, dataclasses.replace(self.params, species=self.compare))
            self._last_images = np.hstack([left, simulated])
        else:
            self._last_images = simulated
        return self._last_images[..., ::-1]

    def caption(self) -> str:
        """What the rendered view shows, left to right."""
        right = species_label(self.params.species)
        if not self.side_by_side:
            return right
        left = "original" if self.compare is None else species_label(self.compare)
        return f"left: {left}    right: {right}"

    def reset(self) -> None:
        self.params = dataclasses.replace(self.initial)
        self.compare = self.initial_compare

    def species_facts(self) -> list[tuple[str, str]]:
        """What the simulation knows about the current species, as (label, value) rows."""
        return species_facts(self.params.species)

    def save_snapshot(self) -> str | None:
        """Write the last rendered view to the working directory and return the file name."""
        if self._last_images is None:
            return None
        shown = self.params.species if self.compare is None or not self.side_by_side else f"{self.compare}-vs-{self.params.species}"
        name = f"dog-{shown}-{time.strftime('%Y%m%d-%H%M%S')}.png"
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
    parser.add_argument(
        "--compare", choices=SPECIES, help="show this species beside --species instead of the original"
    )
    parser.add_argument("--acuity", action="store_true", help="blur to the species' visual acuity")
    parser.add_argument(
        "--fov",
        type=float,
        default=defaults.field_of_view,
        help="degrees the image spans horizontally, for --acuity (default %(default)s)",
    )
    args = parser.parse_args()
    params = Params(args.species, args.adaptation, args.strength, args.chroma_scale, args.acuity, args.fov)

    if args.info:
        print_info(params)
    elif args.image:
        convert_file(args.image, params, args.compare)
    else:
        import tk_window  # the only GUI-specific line in this module

        session = LiveSession(args.camera, params, args.compare)
        try:
            tk_window.run(session)
        finally:
            session.close()
        if session.error:
            sys.exit(session.error)


if __name__ == "__main__":
    main()
