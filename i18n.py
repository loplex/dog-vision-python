"""The languages dog_vision's window speaks.

As in gettext, a text is looked up by its English wording, and a text missing from
a language's catalogue stays English; that is how source citations pass through
unchanged. Adding a language is adding an entry to LANGUAGES.
"""

import dataclasses
import locale
import os
import sys

CZECH = {
    # Window
    "Dog vision": "Psí vidění",
    "Species": "Druh",
    "Selected species": "Vybraný druh",
    "Adaptation to scene [%]": "Adaptace na scénu [%]",
    "Simulation strength [%]": "Síla simulace [%]",
    "Colour saturation": "Sytost barev",
    "Fixed by the projection": "Daná projekcí",
    "Matched to discrimination (RNL)": "Podle rozlišování (RNL)",
    "Acuity": "Ostrost",
    "Blur to the species' acuity": "Rozostřit na ostrost druhu",
    "Image spans [degrees]": "Šířka záběru [°]",
    "View": "Zobrazení",
    "Side by side (m)": "Vedle sebe (m)",
    "Left image": "Levý obraz",
    "Map of differences (d)": "Mapa rozdílů (d)",
    "Reset (r)": "Obnovit (r)",
    "Save snapshot (s)": "Uložit snímek (s)",
    "Saved {name}": "Uloženo: {name}",
    "No frame yet": "Zatím žádný snímek",
    "Language": "Jazyk",
    # Caption
    "original": "originál",
    "left: {left}    right: {right}": "vlevo: {left}    vpravo: {right}",
    "    red: noticeably different ({share} of pixels)": "    červeně: znatelně odlišné ({share} pixelů)",
    # Species facts
    "{0}%": "{0} %",
    "{0}%–{1}%": "{0}–{1} %",
    "Colour vision": "Barevné vidění",
    "{kind}, 1 cone type": "{kind}, 1 typ čípků",
    "{kind}, {n} cone types": "{kind}, {n} typy čípků",
    "Cone peaks": "Maxima čípků",
    "Peaks from": "Zdroj maxim",
    "S cones": "Čípky S",
    "{share} of cones ({source})": "{share} čípků ({source})",
    "L : M cones": "Poměr L : M",
    "{ratio} : 1 (assumed)": "{ratio} : 1 (předpoklad)",
    "Neutral point": "Neutrální bod",
    "RNL scale": "Škála RNL",
    "nothing to scale": "není co škálovat",
    "{gains} of fixed": "{gains} vůči projekci",
    " and ": " a ",
    "assumed": "předpoklad",
    "not found measured; left sharp": "měření nenalezeno; ponecháno ostré",
    "{value} c/deg": "{value} c/°",
    "{across} c/deg side by side, {up} one above another": "{across} c/° vodorovně, {up} svisle",
    # Notes attached to a source in dog_vision's tables
    "L shifted 10 nm, see text": "L posunutý o 10 nm, viz text",
    "M shifted 10 nm, see text": "M posunutý o 10 nm, viz text",
    "Jacobs et al. 1996, S assumed": "Jacobs et al. 1996, S předpokládán",
    "Sumita et al. 2013, 11.7 to 14": "Sumita et al. 2013, 11,7 až 14",
    "Jacobs, Birch & Blakeslee 1982, 1.8 to 3.8": "Jacobs, Birch & Blakeslee 1982, 1,8 až 3,8",
    "Nature Neuroscience 2024, about 60": "Nature Neuroscience 2024, asi 60",
    "Hanke & Dehnhardt 2009, in air": "Hanke & Dehnhardt 2009, na vzduchu",
    "Herman et al. 1975, 8.2 arcmin stripes": "Herman et al. 1975, pruhy 8,2′",
}

CZECH_SPECIES = {
    "dog": "pes",
    "cat": "kočka",
    "horse": "kůň",
    "cow": "kráva",
    "sheep": "ovce",
    "goat": "koza",
    "pig": "prase",
    "fallow-deer": "daněk",
    "white-tailed-deer": "jelenec běloocasý",
    "guinea-pig": "morče",
    "tree-squirrel": "veverka",
    "ground-squirrel": "sysel kalifornský",
    "ferret": "fretka",
    "protanope": "protanop",
    "deuteranope": "deuteranop",
    "human": "člověk",
    "protanomalous": "protanomál",
    "deuteranomalous": "deuteranomál",
    "macaque": "makak",
    "howler-monkey": "vřešťan",
    "marmoset-female": "kosman, samice",
    "harbour-seal": "tuleň obecný",
    "bottlenose-dolphin": "delfín skákavý",
}


@dataclasses.dataclass(frozen=True)
class Language:
    name: str  # in the language itself
    decimal_point: str = "."
    texts: dict[str, str] = dataclasses.field(default_factory=dict)  # English text: its translation
    species: dict[str, str] = dataclasses.field(default_factory=dict)  # SPECIES key: its name


# Keyed by ISO 639-1 code, the way locale names start. English needs no catalogue:
# its texts are the keys, and the SPECIES keys its species names.
LANGUAGES = {
    "cs": Language("Čeština", ",", CZECH, CZECH_SPECIES),
    "en": Language("English"),
}


def translate(text: str, language: str) -> str:
    return LANGUAGES[language].texts.get(text, text)


def species_name(species: str, language: str) -> str:
    return LANGUAGES[language].species.get(species, species)


def number(value: float, spec: str, language: str) -> str:
    """format(value, spec), with the language's decimal point."""
    return format(value, spec).replace(".", LANGUAGES[language].decimal_point)


def _language_of(locale_name: str) -> str | None:
    """"en" from "en_GB.UTF-8", if it is one of LANGUAGES."""
    code = locale_name.lower().split("_")[0].split(".")[0].split("@")[0]
    return code if code in LANGUAGES else None


def system_language() -> str:
    """The first of LANGUAGES the user's system asks for, and English if none.

    Like gettext, only the first of LANGUAGE, LC_ALL, LC_MESSAGES and LANG that is set
    counts, and LANGUAGE may list several languages. Windows usually sets none of them,
    so there the user's interface language is asked for instead.
    """
    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(variable)
        if value:
            for name in value.split(":"):
                language = _language_of(name)
                if language:
                    return language
            return "en"
    if sys.platform == "win32":
        import ctypes

        name = locale.windows_locale.get(ctypes.windll.kernel32.GetUserDefaultUILanguage(), "")
    else:
        name = locale.getlocale()[0] or ""
    return _language_of(name) or "en"
