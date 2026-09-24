"""The languages dog_vision's window speaks.

As in gettext, a text is looked up by its English wording, and a text missing from
a language's catalogue stays English; that is how source citations pass through
unchanged. Adding a language is adding an entry to LANGUAGES.
"""

import dataclasses
import locale
import os
import sys


@dataclasses.dataclass(frozen=True)
class Language:
    name: str  # in the language itself
    decimal_point: str = "."
    texts: dict[str, str] = dataclasses.field(default_factory=dict)  # English text: its translation
    species: dict[str, str] = dataclasses.field(default_factory=dict)  # SPECIES key: its name


# Keyed by ISO 639-1 code, the way locale names start. English needs no catalogue:
# its texts are the keys, and the SPECIES keys its species names.
LANGUAGES = {
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
