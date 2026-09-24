# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless"]
#
# [tool.uv]
# python-preference = "system"
# ///
"""Render the README's figures made from docs/shiny-red-apples.jpg.

- docs/apples-species.png: the photo as it is and as five species see it.
- docs/apples-acuity.png: a sticker on an apple, as sharp as a human and as a dog resolve it.
- docs/apples-difference.png: a deuteranope's view, a dog's, and the map of where they differ.

    uv run render_photo_figures.py            # write the images
    uv run render_photo_figures.py --check    # exit 1 if an image is not what the code renders

The photo is "Shiny red apples" by Leon Brooks, released into the public domain:
https://commons.wikimedia.org/wiki/File:Shiny_red_apples.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import dog_vision as dv

DOCS = Path(__file__).parent / "docs"
PHOTO = DOCS / "shiny-red-apples.jpg"
SHRINK = 8  # the photo is averaged in 8 x 8 blocks, which gives tiles of 320 x 240
SPECIES_SHOWN = ["dog", "cat", "protanope", "deuteranope", "harbour-seal"]
FIELD_OF_VIEW = 30.0  # degrees the whole photo is taken to span, for the acuity figure
STICKER = (slice(440, 680), slice(0, 320))  # rows and columns of the full photo around a sticker
# JPEG decoding and the blur may differ in the last bit between OpenCV builds; a change in the
# model moves pixels much further than this.
TOLERANCE = 2


def shrink(image: np.ndarray) -> np.ndarray:
    """Average SHRINK x SHRINK blocks, which, unlike cv2.resize, gives the same bytes everywhere."""
    height, width = image.shape[0] // SHRINK, image.shape[1] // SHRINK
    blocks = image[: height * SHRINK, : width * SHRINK].reshape(height, SHRINK, width, SHRINK, 3)
    return np.round(blocks.mean(axis=(1, 3))).astype(np.uint8)


def captioned(image: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((28, image.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([image, bar])


def row(cells: list[np.ndarray]) -> np.ndarray:
    gap = np.full((cells[0].shape[0], 8, 3), 255, np.uint8)
    parts = [cells[0]]
    for cell in cells[1:]:
        parts += [gap, cell]
    return np.hstack(parts)


def species_figure(photo: np.ndarray) -> np.ndarray:
    small = shrink(photo)
    cells = [captioned(small, "original")]
    cells += [captioned(dv.simulate(small, dv.Params(species)), dv.species_label(species)) for species in SPECIES_SHOWN]
    top, bottom = row(cells[:3]), row(cells[3:])
    return np.vstack([top, np.full((8, top.shape[1], 3), 255, np.uint8), bottom])


def acuity_figure(photo: np.ndarray) -> np.ndarray:
    """The blur is set by the whole photo's width, so it is applied to the whole photo and then cropped."""
    cells = [captioned(photo[STICKER], "original")]
    for species in ("human", "dog"):
        blurred = dv.simulate(photo, dv.Params(species, acuity=True, field_of_view=FIELD_OF_VIEW))
        cells.append(captioned(blurred[STICKER], f"{dv.species_label(species)}, {FIELD_OF_VIEW:g} deg across"))
    return row(cells)


def difference_figure(photo: np.ndarray) -> np.ndarray:
    small = shrink(photo)
    left, right = dv.simulate(small, dv.Params("deuteranope")), dv.simulate(small, dv.Params("dog"))
    difference, share = dv.difference_map(left, right)
    return row([
        captioned(left, dv.species_label("deuteranope")),
        captioned(right, dv.species_label("dog")),
        captioned(difference, f"red: noticeably different ({share:.0%} of pixels)"),
    ])


FIGURES = {
    "apples-species.png": species_figure,
    "apples-acuity.png": acuity_figure,
    "apples-difference.png": difference_figure,
}


def render() -> dict[str, np.ndarray]:
    photo = cv2.imread(str(PHOTO), cv2.IMREAD_COLOR)
    return {name: figure(photo) for name, figure in FIGURES.items()}


def stale() -> list[str]:
    """The figures whose file is missing or differs from what the code renders."""
    names = []
    for name, rendered in render().items():
        stored = cv2.imread(str(DOCS / name), cv2.IMREAD_COLOR)
        if stored is None or stored.shape != rendered.shape or np.abs(stored.astype(int) - rendered).max() > TOLERANCE:
            names.append(name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    if parser.parse_args().check:
        names = stale()
        for name in names:
            print(f"{name} is out of date; run: uv run render_photo_figures.py", file=sys.stderr)
        return 1 if names else 0
    for name, image in render().items():
        cv2.imwrite(str(DOCS / name), image)
        print(f"Wrote {DOCS / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
