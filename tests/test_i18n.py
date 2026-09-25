import ctypes
from types import SimpleNamespace

import pytest

from dog_vision.core import i18n

LOCALE_VARIABLES = ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG")


@pytest.fixture
def environment(monkeypatch):
    """No locale variable set, and no locale either, until a test sets one."""
    for variable in LOCALE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(i18n.sys, "platform", "linux")
    monkeypatch.setattr(i18n.locale, "getlocale", lambda: (None, None))
    return monkeypatch


def test_a_text_is_translated():
    assert i18n.translate("Species", "cs") == "Druh"


def test_a_text_missing_from_a_catalogue_stays_english():
    assert i18n.translate("Odom et al. 1983", "cs") == "Odom et al. 1983"
    assert i18n.translate("Species", "en") == "Species"


def test_a_species_is_named_in_the_language():
    assert i18n.species_name("dog", "cs") == "pes"
    assert i18n.species_name("dog", "en") == "dog"


def test_a_number_takes_the_languages_decimal_point():
    assert i18n.number(11.6, "g", "cs") == "11,6"
    assert i18n.number(11.6, "g", "en") == "11.6"


def test_n_marks_a_text_without_changing_it():
    assert i18n.N_("Species") == "Species"


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ({"LANG": "cs_CZ.UTF-8"}, "cs"),
        ({"LANG": "en_GB.UTF-8"}, "en"),
        ({"LANG": "de_DE.UTF-8"}, "en"),  # not among LANGUAGES
        ({"LC_ALL": "cs_CZ@euro"}, "cs"),
        ({"LANGUAGE": "de:cs:en"}, "cs"),  # the first of the list that is among LANGUAGES
        ({"LANGUAGE": "de", "LANG": "cs_CZ.UTF-8"}, "en"),  # only the first variable set counts
        ({"LANGUAGE": "", "LANG": "cs_CZ.UTF-8"}, "cs"),  # an empty one is not set
        ({"LC_MESSAGES": "cs_CZ", "LANG": "en_US"}, "cs"),
        ({"LANG": "C"}, "en"),
    ],
)
def test_the_system_language_comes_from_the_first_locale_variable_set(environment, variables, expected):
    for variable, value in variables.items():
        environment.setenv(variable, value)
    assert i18n.system_language() == expected


def test_without_locale_variables_the_locale_decides(environment):
    environment.setattr(i18n.locale, "getlocale", lambda: ("cs_CZ", "UTF-8"))
    assert i18n.system_language() == "cs"


def test_without_any_locale_it_is_english(environment):
    assert i18n.system_language() == "en"



@pytest.mark.parametrize(("language_id", "expected"), [(0x0405, "cs"), (0x0409, "en"), (0x0407, "en")])
def test_on_windows_the_interface_language_decides(environment, language_id, expected):
    kernel32 = SimpleNamespace(GetUserDefaultUILanguage=lambda: language_id)
    environment.setattr(i18n.sys, "platform", "win32")
    environment.setattr(ctypes, "windll", SimpleNamespace(kernel32=kernel32), raising=False)
    assert i18n.system_language() == expected
