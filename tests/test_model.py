import numpy as np
import pytest

from dog_vision.core.model import (
    CHROMA_SCALES,
    WAVELENGTHS,
    Params,
    animal_cone_matrix,
    chroma_directions,
    cone_matrix,
    cone_shares,
    display_primaries,
    govardovskii_a1,
    grey_world_gains,
    neutral_point,
    planck,
    rnl_chroma_matrix,
    rnl_gains,
    rnl_metric,
    s_cone_fraction,
    simulation_matrix,
)
from dog_vision.core.species import (
    ASSUMED_S_CONE_FRACTION,
    HUMAN_CONES,
    S_CONE_FRACTION,
    SPECIES,
)

ALL = list(SPECIES)
DICHROMATS = [name for name, peaks in SPECIES.items() if len(peaks) == 2]
TRICHROMATS = [name for name, peaks in SPECIES.items() if len(peaks) == 3]
MONOCHROMATS = [name for name, peaks in SPECIES.items() if len(peaks) == 1]


def test_every_kind_of_colour_vision_is_among_the_species():
    assert DICHROMATS and TRICHROMATS and MONOCHROMATS


@pytest.mark.parametrize("peak", [420.0, 500.0, 560.0])
def test_a_pigment_is_most_sensitive_at_its_peak(peak):
    sensitivity = govardovskii_a1(peak)
    assert WAVELENGTHS[sensitivity.argmax()] == pytest.approx(peak, abs=1)
    assert sensitivity.max() == pytest.approx(1, abs=0.01)


def test_the_display_white_excites_human_cones_like_d65():
    human = cone_matrix(tuple(HUMAN_CONES.values()), display_primaries())
    d65 = np.stack([govardovskii_a1(p) for p in HUMAN_CONES.values()]) @ planck(6504.0)
    np.testing.assert_allclose(human @ np.ones(3), d65, rtol=1e-9)


@pytest.mark.parametrize("species", ALL)
def test_the_white_excites_each_cone_by_one(species):
    m_animal = animal_cone_matrix(Params(species))
    assert m_animal.shape == (len(SPECIES[species]), 3)
    np.testing.assert_allclose(m_animal @ np.ones(3), 1)


@pytest.mark.parametrize("species", ALL)
@pytest.mark.parametrize("chroma_scale", CHROMA_SCALES)
def test_grey_stays_grey(species, chroma_scale):
    t = simulation_matrix(Params(species, chroma_scale=chroma_scale))
    np.testing.assert_allclose(t @ np.ones(3), 1, atol=1e-12)


@pytest.mark.parametrize("species", ALL)
def test_the_fixed_scale_excites_the_cones_as_the_input_does(species):
    params = Params(species)
    m_animal = animal_cone_matrix(params)
    np.testing.assert_allclose(m_animal @ simulation_matrix(params), m_animal, atol=1e-12)


@pytest.mark.parametrize("species", ALL)
@pytest.mark.parametrize("chroma_scale", CHROMA_SCALES)
def test_the_rank_is_the_number_of_cone_types(species, chroma_scale):
    t = simulation_matrix(Params(species, chroma_scale=chroma_scale))
    assert np.linalg.matrix_rank(t, tol=1e-9) == len(SPECIES[species])


@pytest.mark.parametrize("species", DICHROMATS)
@pytest.mark.parametrize("chroma_scale", CHROMA_SCALES)
def test_a_dichromat_is_shown_in_the_plane_red_equals_green(species, chroma_scale):
    t = simulation_matrix(Params(species, chroma_scale=chroma_scale))
    np.testing.assert_allclose(t[0], t[1], atol=1e-12)


@pytest.mark.parametrize("species", MONOCHROMATS)
def test_a_monochromat_sees_grey(species):
    t = simulation_matrix(Params(species))
    np.testing.assert_allclose(t, np.outer(np.ones(3), t[0]), atol=1e-12)


@pytest.mark.parametrize("chroma_scale", CHROMA_SCALES)
def test_a_human_sees_the_image_unchanged(chroma_scale):
    np.testing.assert_allclose(simulation_matrix(Params("human", chroma_scale=chroma_scale)), np.eye(3), atol=1e-12)


@pytest.mark.parametrize("species", ALL)
def test_strength_blends_with_the_original(species):
    full = simulation_matrix(Params(species))
    np.testing.assert_allclose(simulation_matrix(Params(species, strength=0.0)), np.eye(3))
    np.testing.assert_allclose(simulation_matrix(Params(species, strength=0.25)), 0.75 * np.eye(3) + 0.25 * full)


def test_no_adaptation_leaves_the_gains_at_one():
    m_animal = animal_cone_matrix(Params("dog"))
    np.testing.assert_allclose(grey_world_gains(m_animal, np.array([0.8, 0.3, 0.1]), 0.0), 1)


def test_full_adaptation_makes_the_scene_mean_neutral():
    m_animal = animal_cone_matrix(Params("dog"))
    mean_rgb = np.array([0.8, 0.3, 0.1])
    adapted = grey_world_gains(m_animal, mean_rgb, 1.0) * (m_animal @ mean_rgb)
    np.testing.assert_allclose(adapted, adapted.mean())


def test_adaptation_to_a_grey_scene_changes_nothing():
    params = Params("dog", adaptation=1.0)
    np.testing.assert_allclose(simulation_matrix(params, np.full(3, 0.2)), simulation_matrix(params), atol=1e-12)


@pytest.mark.parametrize("species", DICHROMATS + TRICHROMATS)
def test_a_chroma_direction_raises_one_cone_and_leaves_the_rest(species):
    m_animal = animal_cone_matrix(Params(species))
    n = len(m_animal)
    np.testing.assert_allclose(m_animal @ chroma_directions(m_animal), np.eye(n)[:, :-1], atol=1e-12)


@pytest.mark.parametrize("species", DICHROMATS + TRICHROMATS)
def test_cone_shares_sum_to_one(species):
    shares = cone_shares(species)
    assert len(shares) == len(SPECIES[species])
    assert shares.sum() == pytest.approx(1)


def test_a_measured_s_cone_share_is_the_middle_of_its_range():
    species = next(iter(S_CONE_FRACTION))
    low, high = S_CONE_FRACTION[species][:2]
    assert s_cone_fraction(species) == ((low + high) / 2, True)


def test_an_unmeasured_s_cone_share_is_assumed():
    species = next(name for name in DICHROMATS if name not in S_CONE_FRACTION)
    assert s_cone_fraction(species) == (sum(ASSUMED_S_CONE_FRACTION) / 2, False)


def rnl_forms(species: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K, the animal's form A and the human's B, as rnl_chroma_matrix relates them."""
    m_animal = animal_cone_matrix(Params(species))
    n = len(m_animal)
    animal = rnl_metric(species, np.eye(n))[:-1, :-1]
    u = chroma_directions(m_animal)
    human = u.T @ rnl_metric("human", animal_cone_matrix(Params("human"))) @ u
    return rnl_chroma_matrix(Params(species)), animal, human


@pytest.mark.parametrize("species", DICHROMATS + TRICHROMATS)
def test_the_rnl_scale_makes_a_human_count_the_animals_differences(species):
    k, animal, human = rnl_forms(species)
    np.testing.assert_allclose(k.T @ human @ k, animal, rtol=1e-9, atol=1e-12 * np.abs(animal).max())


@pytest.mark.parametrize("species", TRICHROMATS)
def test_the_rnl_scale_keeps_the_hue_of_blue_and_yellow(species):
    k, _animal, _human = rnl_forms(species)
    m_animal = animal_cone_matrix(Params(species))
    blue_yellow = np.hstack([np.eye(2), -np.ones((2, 1))]) @ m_animal @ np.array([-0.5, -0.5, 1.0])
    mapped = k @ blue_yellow
    assert mapped[0] * blue_yellow[1] - mapped[1] * blue_yellow[0] == pytest.approx(0, abs=1e-12)
    assert mapped @ blue_yellow > 0


def test_the_rnl_scale_is_the_identity_for_a_human():
    np.testing.assert_allclose(rnl_chroma_matrix(Params("human")), np.eye(2), atol=1e-12)


def test_a_monochromat_has_no_rnl_scale():
    assert rnl_chroma_matrix(Params(MONOCHROMATS[0])).shape == (0, 0)
    assert rnl_gains(Params(MONOCHROMATS[0])).size == 0


@pytest.mark.parametrize("species", DICHROMATS + TRICHROMATS)
def test_rnl_gains_are_largest_first(species):
    gains = rnl_gains(Params(species))
    assert len(gains) == len(SPECIES[species]) - 1
    assert list(gains) == sorted(gains, reverse=True)
    assert all(gains > 0)


def test_the_dogs_neutral_point_is_near_the_measured_480_nm():
    assert neutral_point(Params("dog")) == pytest.approx(480, abs=5)


@pytest.mark.parametrize("species", DICHROMATS)
def test_a_dichromats_neutral_point_lies_between_its_cone_peaks(species):
    short, long = SPECIES[species]
    assert short < neutral_point(Params(species)) < long


def test_params_give_the_species_cone_peaks():
    assert Params("cat").cones() == SPECIES["cat"]


@pytest.mark.parametrize("species", DICHROMATS)
def test_the_rnl_noise_of_a_cone_grows_as_its_share_falls(species):
    # For two cones the form is 1 / (e_S^2 + e_L^2), with e_i^2 = n_max / n_i.
    shares = cone_shares(species)
    noise = shares.max() / shares
    form = rnl_metric(species, np.eye(2))
    assert form[0, 0] == pytest.approx(1 / noise.sum())
    np.testing.assert_allclose(form, form[0, 0] * np.array([[1, -1], [-1, 1]]))
