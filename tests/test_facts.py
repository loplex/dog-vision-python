import re

import pytest

from dog_vision.core import i18n
from dog_vision.core.facts import (
    FACT_DESCRIPTIONS,
    acuity_value,
    percent,
    pieces,
    species_facts,
    species_label,
)
from dog_vision.core.species import SPECIES


def test_a_share_is_given_in_whole_percent():
    assert percent(0.1, 0.1) == "10%"
    assert percent(0.101, 0.104) == "10%"


def test_a_range_is_given_when_its_ends_differ_in_whole_percent():
    assert percent(0.08, 0.12) == "8%–12%"


def test_a_share_in_czech_has_a_space_before_the_percent_sign():
    assert percent(0.1, 0.1, "cs") == "10 %"
    assert percent(0.08, 0.12, "cs") == "8–12 %"


def test_a_label_names_the_kind_of_colour_vision():
    assert species_label("dog") == "dog (dichromat)"
    assert species_label("human") == "human (trichromat)"
    assert species_label("harbour-seal") == "harbour-seal (monochromat)"
    assert species_label("dog", "cs") == f"{i18n.species_name('dog', 'cs')} (dichromat)"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Odom et al. 1983", ["Odom et al. 1983"]),
        ("Neitz, Geist & Jacobs 1989", ["Neitz, Geist & Jacobs 1989"]),  # authors stay together
        ("Jacobs, Birch & Blakeslee 1982, 1.8 to 3.8", ["Jacobs, Birch & Blakeslee 1982,", "1.8 to 3.8"]),
        ("Travis 1988, Williams 1992", ["Travis 1988,", "Williams 1992"]),
        ("Bowmaker et al. 1978, 1991", ["Bowmaker et al. 1978,", "1991"]),
    ],
)
def test_pieces_are_cut_after_a_comma_that_follows_a_number(text, expected):
    assert pieces(text) == expected
    assert " ".join(pieces(text)) == text


def test_the_dogs_facts():
    facts = {label: value for label, value, _description in species_facts("dog")}
    (rnl_scale,) = facts.pop("RNL scale")
    assert facts == {
        "Colour vision": ("dichromat, 2 cone types",),
        "Cone peaks": ("S 429 nm,", "L 555 nm", "(Neitz, Geist & Jacobs 1989)"),
        "S cones": ("10%–18% of cones", "(Mowat et al. 2008)"),
        "Neutral point": ("479 nm", "(model)"),
        "Acuity": ("11.6 c/deg", "(Odom et al. 1983)"),
    }
    assert re.fullmatch(r"x\d\.\d\d of fixed", rnl_scale)


def test_a_trichromat_has_an_assumed_l_to_m_ratio_and_two_rnl_factors():
    facts = {label: value for label, value, _description in species_facts("macaque")}
    assert facts["Colour vision"] == ("trichromat, 3 cone types",)
    assert facts["Cone peaks"][:3] == ("S 431 nm,", "M 536 nm,", "L 565 nm")
    assert facts["L : M cones"] == ("1 : 1", "(assumed)")
    assert " and " in facts["RNL scale"][0]
    assert "Neutral point" not in facts


def test_a_monochromat_has_no_colour_axis():
    facts = {label: value for label, value, _description in species_facts("harbour-seal")}
    assert facts["Colour vision"] == ("monochromat, 1 cone type",)
    assert facts["RNL scale"] == ("no colour axis", "(sees only grey)")
    assert "S cones" not in facts


def test_an_unmeasured_s_cone_share_is_marked_assumed():
    facts = {label: value for label, value, _description in species_facts("goat")}
    assert facts["S cones"] == ("10% of cones", "(assumed)")


@pytest.mark.parametrize("language", list(i18n.LANGUAGES))
@pytest.mark.parametrize("species", list(SPECIES))
def test_every_species_has_its_facts_described_in_every_language(species, language):
    rows = species_facts(species, language)
    english = species_facts(species)
    assert [label for label, _v, _d in rows] == [i18n.translate(label, language) for label, _v, _d in english]
    for (_label, value, description), (english_label, _ev, _ed) in zip(rows, english):
        assert value and all(value)
        assert description == i18n.translate(FACT_DESCRIPTIONS[english_label], language)


def test_numbers_in_czech_facts_have_a_decimal_comma():
    facts = {label: value for label, value, _d in species_facts("dog", "cs")}
    assert "11,6 c/°" in facts[i18n.translate("Acuity", "cs")]


def test_acuity_that_differs_by_direction_is_given_both_ways():
    assert acuity_value("cow") == ["2.6 c/deg side by side,", "1.6 one above another", "(Rehkämper et al. 2000)"]


def test_an_unmeasured_acuity_is_left_sharp():
    assert acuity_value("goat") == ["not found measured;", "left sharp"]
