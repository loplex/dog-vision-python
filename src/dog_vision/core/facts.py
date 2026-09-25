"""What the window says about a species: its label and its facts, in any language of i18n."""

import re

from dog_vision.core import i18n
from dog_vision.core.i18n import N_
from dog_vision.core.model import Params, neutral_point, rnl_gains
from dog_vision.core.species import (
    ACUITY,
    ASSUMED_L_TO_M,
    ASSUMED_S_CONE_FRACTION,
    PEAKS_FROM,
    S_CONE_FRACTION,
    SPECIES,
)

COLOUR_VISION = {1: N_("monochromat"), 2: N_("dichromat"), 3: N_("trichromat")}


def percent(low: float, high: float, language: str = "en") -> str:
    """A share, or a range of shares when the two differ in whole percent."""
    low_text, high_text = f"{low * 100:.0f}", f"{high * 100:.0f}"
    if low_text == high_text:
        return i18n.translate("{0}%", language).format(low_text)
    return i18n.translate("{0}%–{1}%", language).format(low_text, high_text)


def species_label(species: str, language: str = "en") -> str:
    """The species' name with its kind of colour vision, for lists."""
    kind = i18n.translate(COLOUR_VISION[len(SPECIES[species])], language)
    return f"{i18n.species_name(species, language)} ({kind})"


# What each row of species_facts means, for a reader who does not know the model.
FACT_DESCRIPTIONS = {
    "Colour vision": (
        "How many kinds of cone the species has, and so how many independent colour signals its eye sends."
        "\n\nA trichromat, like us, has three. A camera records nothing that would merge for it, so its image"
        " stays as it is unless Colour saturation is matched to discrimination."
        "\n\nA dichromat has two. Colours that differ only along one direction look the same to it, and the"
        " simulation shows them on the blue–yellow axis."
        "\n\nA cone monochromat has one, and sees shades of grey."
    ),
    "Cone peaks": (
        "The wavelength each kind of cone is most sensitive to: S for short, M for middle and L for long"
        " wavelengths. A dichromat's longer cone is listed as L whatever its source calls it."
        "\n\nThe study the peaks are taken from is in brackets; the README lists every source in full under"
        " References. A note after the citation marks a value that is not simply measured, such as a cone"
        " shifted on purpose or one assumed."
        "\n\nEach cone is modelled from its peak alone, with the pigment template of Govardovskii et al. (2000)."
        " Its sensitivity is about 100 nm wide at half height, which is why a shift of a few nanometres"
        " between species changes little."
        "\n\nA human's cones peak at 420.7, 530.3 and 558.9 nm."
    ),
    "S cones": (
        "The share of all cones that are S cones, with its source. Where the source reports a range across"
        " the retina, the middle of it is used."
        "\n\nOnly the RNL colour saturation uses it: the fewer cones of a kind, the noisier their signal, and"
        " the larger a colour difference along that cone's axis must be to be noticed."
        "\n\n\"Assumed\" means no measurement was found, and 10 % is used, the middle of what the measured"
        " species span."
    ),
    "L : M cones": (
        "How many L cones a trichromat has for every M cone. Only the RNL colour saturation uses it."
        "\n\nHumans with normal colour vision range from about twice as many L as M cones to the reverse"
        " (Roorda & Williams 1999), so every trichromat is given 1 : 1, and it is marked as assumed."
    ),
    "Neutral point": (
        "The wavelength of pure spectral light that a dichromat cannot tell from white. Shorter wavelengths"
        " look bluish to it, longer ones yellowish."
        "\n\nThe value is the model's, computed from the cone peaks. For the dog it matches the measured"
        " neutral point of about 480 nm (Neitz, Geist & Jacobs 1989). For a human deuteranope it falls well"
        " short of the measured 505 nm, probably because the model ignores the lens and the macular pigment."
    ),
    "RNL scale": (
        "How much the RNL colour saturation changes saturation compared with the fixed one. Below 1 the"
        " species tells colours apart worse than we do, above 1 better."
        "\n\nIt comes from the receptor noise limited model of Vorobyev & Osorio (1998), which counts"
        " just-noticeable differences from each cone's noise, set by the share of cones of its kind. The"
        " output is rescaled until a human looking at it counts as many differences as the animal would."
        " A trichromat has two colour axes and so two factors; a monochromat has none."
        "\n\nIt assumes the animal's cones are as noisy as ours, which is measured for almost no mammal."
    ),
    "Acuity": (
        "The finest stripes the species resolves, in cycles per degree: pairs of one dark and one light"
        " stripe within one degree of view. A human resolves about 72."
        "\n\nWith Blur to the species' acuity checked, detail finer than that is removed, as AcuityView does"
        " (Caves & Johnsen 2018)."
        "\n\nCattle resolve detail side by side better than one above another, since their pupil is a"
        " horizontal oval. A species without a measurement is left sharp."
    ),
}


def pieces(text: str) -> list[str]:
    """text split after each ", " that follows a number, the comma kept with what it ends.

    In a citation the commas before the year separate authors, and the ones after it a note or
    another citation, as in "Jacobs, Birch & Blakeslee 1982, 1.8 to 3.8".
    """
    parts, start = [], 0
    for comma in re.finditer(", ", text):
        if re.search(r"\d", text[start : comma.start()]):
            parts.append(text[start : comma.start() + 1])
            start = comma.end()
    return [*parts, text[start:]]


def species_facts(species: str, language: str = "en") -> list[tuple[str, tuple[str, ...], str]]:
    """What the simulation knows about a species, as (label, value, description) rows for a table.

    A value is the pieces of information it holds, such as a share and its source: joined by
    spaces they read as one line, and a table that has to break it breaks it only between them.
    """
    def _(text: str) -> str:
        return i18n.translate(text, language)

    peaks = SPECIES[species]
    n = len(peaks)
    names = {3: "SML", 2: "SL", 1: "L"}[n]
    cone_types = _("{kind}, {n} cone types" if n > 1 else "{kind}, 1 cone type")
    cones = [f"{name} {i18n.number(peak, 'g', language)} nm" for name, peak in zip(names, peaks)]
    facts = [
        ("Colour vision", [cone_types.format(kind=_(COLOUR_VISION[n]), n=n)]),
        ("Cone peaks", [f"{cone}," for cone in cones[:-1]] + cones[-1:] + pieces(f"({_(PEAKS_FROM[species])})")),
    ]
    if n == 1:
        facts += [("RNL scale", [_("no colour axis"), _("(sees only grey)")])]
    else:
        low, high, source = S_CONE_FRACTION.get(species, (*ASSUMED_S_CONE_FRACTION, _("assumed")))
        facts.append(("S cones", [_("{share} of cones").format(share=percent(low, high, language)), *pieces(f"({source})")]))
        if n == 3:
            facts.append(("L : M cones", [f"{i18n.number(ASSUMED_L_TO_M, 'g', language)} : 1", f"({_('assumed')})"]))
        else:
            facts.append(("Neutral point", [f"{neutral_point(Params(species)):.0f} nm", "(model)"]))
        gains = _(" and ").join(f"x{i18n.number(gain, '.2f', language)}" for gain in rnl_gains(Params(species)))
        facts.append(("RNL scale", [_("{gains} of fixed").format(gains=gains)]))
    facts.append(("Acuity", acuity_value(species, language)))
    return [(_(label), tuple(value), _(FACT_DESCRIPTIONS[label])) for label, value in facts]


def acuity_value(species: str, language: str = "en") -> list[str]:
    """The species' acuity and its source, as species_facts gives a value."""
    def _(text: str) -> str:
        return i18n.translate(text, language)

    if species not in ACUITY:
        return [_("not found measured;"), _("left sharp")]
    (across, up), source = ACUITY[species]
    across, up = (i18n.number(value, ".3g", language) for value in (across, up))
    if across == up:
        value = [_("{value} c/deg").format(value=across)]
    else:
        value = [_("{across} c/deg side by side,").format(across=across), _("{up} one above another").format(up=up)]
    return [*value, *pieces(f"({_(source)})")]
