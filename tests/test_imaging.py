import numpy as np
import pytest
from conftest import solid

from dog_vision.core.imaging import (
    acuity_blur,
    apply_bgr,
    compose,
    difference_map,
    lab_from_bgr,
    mean_linear_rgb,
    simulate,
)
from dog_vision.core.model import Params
from dog_vision.core.species import ACUITY, SPECIES


@pytest.fixture
def photo() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (48, 64, 3), np.uint8)


def test_apply_bgr_reads_the_matrix_in_rgb_order():
    red_to_blue = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], float)  # linear RGB: B' = R
    out = apply_bgr(np.array([[[0, 0, 255]]], np.uint8), red_to_blue)  # BGR red
    assert out[0, 0].tolist() == [255, 0, 0]  # BGR blue


def test_the_identity_gives_every_value_back():
    values = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
    np.testing.assert_array_equal(apply_bgr(values, np.eye(3)), values)


def test_strength_zero_shows_the_original(photo):
    np.testing.assert_array_equal(simulate(photo, Params("dog", strength=0.0)), photo)


def test_apply_bgr_works_in_linear_light():
    half = np.eye(3) * 0.5
    out = apply_bgr(solid(255, 1, 1), half)
    assert out[0, 0].tolist() == [188] * 3  # sRGB of linear 0.5 is 187.5, not 128


def test_apply_bgr_clips_to_the_displayable_range():
    out = apply_bgr(solid(200, 1, 1), np.eye(3) * 4)
    assert out[0, 0].tolist() == [255] * 3


def test_no_acuity_blur_unless_asked_for():
    assert acuity_blur(Params("dog"), 640) is None


def test_no_acuity_blur_for_a_species_not_measured():
    species = next(name for name in SPECIES if name not in ACUITY)
    assert acuity_blur(Params(species, acuity=True), 640) is None


def test_the_acuity_blur_grows_with_the_pixels_per_degree():
    (across, up), _source = ACUITY["cow"]
    sigma_x, sigma_y = acuity_blur(Params("cow", acuity=True, field_of_view=60), 600)
    assert sigma_x == pytest.approx(np.sqrt(3.56 / (2 * np.pi**2)) / across * 10)
    assert sigma_y / sigma_x == pytest.approx(across / up)  # cattle resolve less one above another
    wider = acuity_blur(Params("cow", acuity=True, field_of_view=30), 600)
    assert wider[0] == pytest.approx(2 * sigma_x)


def test_the_acuity_blur_removes_detail(photo):
    sharp = simulate(photo, Params("dog"))
    blurred = simulate(photo, Params("dog", acuity=True, field_of_view=1))  # sigma of about 2 pixels
    assert blurred.std() < sharp.std()


def test_cielab_of_white_black_and_red():
    lab = lab_from_bgr(np.array([[[255, 255, 255], [0, 0, 0], [0, 0, 255]]], np.uint8))[0]
    np.testing.assert_allclose(lab[0], [100, 0, 0], atol=0.05)
    np.testing.assert_allclose(lab[1], [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(lab[2], [53.24, 80.09, 67.20], atol=0.05)


def test_identical_images_do_not_differ(photo):
    image, share = difference_map(photo, photo)
    assert share == 0
    assert (image[..., 0] == image[..., 2]).all()  # grey, no red anywhere


def test_a_large_difference_is_fully_red():
    image, share = difference_map(solid(0), solid(255))
    assert share == 1
    assert image[0, 0].tolist() == [40, 40, 255]


def test_a_difference_below_one_jnd_is_not_marked():
    _image, share = difference_map(solid(100), solid(101))
    assert share == 0


def test_the_difference_map_reddens_with_the_difference():
    grey = solid(128)
    image_slight, slight = difference_map(grey, solid(135))
    image_more, _ = difference_map(grey, solid(150))
    assert slight == 1
    assert image_slight[0, 0, 2] < image_more[0, 0, 2]


def test_mean_linear_rgb_is_in_rgb_order():
    np.testing.assert_allclose(mean_linear_rgb(np.array([[[0, 0, 255]]], np.uint8)), [1, 0, 0])


def test_a_dog_sees_grey_as_grey():
    out = simulate(solid(128), Params("dog", adaptation=1.0, chroma_scale="rnl"))
    assert out.min() == out.max() == 128


def test_compose_shows_the_simulation_alone(photo):
    image, share = compose(photo, Params("dog"), side_by_side=False, compare=None, difference=False)
    assert image.shape == photo.shape and share is None
    np.testing.assert_array_equal(image, simulate(photo, Params("dog")))


def test_compose_puts_the_original_on_the_left(photo):
    image, share = compose(photo, Params("dog"), side_by_side=True, compare=None, difference=False)
    assert image.shape == (48, 128, 3) and share is None
    np.testing.assert_array_equal(image[:, :64], photo)
    np.testing.assert_array_equal(image[:, 64:], simulate(photo, Params("dog")))


def test_compose_puts_another_species_on_the_left(photo):
    image, _ = compose(photo, Params("dog"), side_by_side=True, compare="cat", difference=False)
    np.testing.assert_array_equal(image[:, :64], simulate(photo, Params("cat")))


def test_compose_adds_the_map_of_differences(photo):
    image, share = compose(photo, Params("dog"), side_by_side=True, compare=None, difference=True)
    assert image.shape == (48, 192, 3)
    expected, expected_share = difference_map(photo, simulate(photo, Params("dog")))
    np.testing.assert_array_equal(image[:, 128:], expected)
    assert share == expected_share > 0


def test_cattle_blur_detail_one_above_another_more_than_side_by_side():
    stripes = np.zeros((64, 64, 3), np.uint8)
    stripes[::2] = 255  # rows: detail one above another
    params = Params("cow", acuity=True, field_of_view=30)
    across_rows = simulate(stripes, params)
    across_columns = simulate(stripes.transpose(1, 0, 2).copy(), params)
    assert across_rows.std() < across_columns.std()


def test_full_adaptation_makes_a_coloured_scene_grey():
    scene = np.full((16, 16, 3), (40, 120, 200), np.uint8)
    unadapted = simulate(scene, Params("dog"))
    adapted = simulate(scene, Params("dog", adaptation=1.0))
    assert np.ptp(unadapted[0, 0].astype(int)) > 20
    assert np.ptp(adapted[0, 0].astype(int)) <= 1
