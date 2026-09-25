"""The colour-vision model, for one species at a time.

After Brettel, Viénot & Mollon 1997:

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
"""

import dataclasses

import numpy as np

from dog_vision.core.species import (
    ASSUMED_L_TO_M,
    ASSUMED_S_CONE_FRACTION,
    HUMAN_CONES,
    S_CONE_FRACTION,
    SPECIES,
)

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
    return np.array(
        [fraction, (1 - fraction) / (1 + ASSUMED_L_TO_M), (1 - fraction) * ASSUMED_L_TO_M / (1 + ASSUMED_L_TO_M)]
    )


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
    return (
        cone_matrix_rgb.T
        @ to_chroma.T
        @ np.linalg.inv(to_chroma @ np.diag(noise**2) @ to_chroma.T)
        @ to_chroma
        @ cone_matrix_rgb
    )


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
