"""Render docs/species-grid.png: a hue sweep and eight colour patches, as every species sees them.

    uv run tools/render_species_grid.py            # write the image
    uv run tools/render_species_grid.py --check    # exit 1 if the image is not what the code renders
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from dog_vision.core import facts, imaging, model, species

OUTPUT = Path(__file__).parent.parent / "docs" / "species-grid.png"
STRIP_WIDTH = 400
LABEL_WIDTH = 240
PATCHES = [(230, 40, 40), (240, 140, 20), (240, 220, 40), (60, 180, 60), (40, 190, 200), (40, 80, 220), (140, 60, 200), (220, 80, 170)]


def test_image() -> np.ndarray:
    """A saturated hue sweep from red to magenta above eight patches, as BGR."""
    hue = np.linspace(0, 150, STRIP_WIDTH).astype(np.uint8)  # OpenCV hue runs 0-180
    hsv = np.stack([np.tile(hue, (40, 1)), np.full((40, STRIP_WIDTH), 255, np.uint8), np.full((40, STRIP_WIDTH), 255, np.uint8)], -1)
    patches = np.zeros((28, STRIP_WIDTH, 3), np.uint8)
    for i, rgb in enumerate(PATCHES):
        patches[:, i * STRIP_WIDTH // len(PATCHES) : (i + 1) * STRIP_WIDTH // len(PATCHES)] = rgb[::-1]
    return np.vstack([cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), patches])


def label(text: str, height: int, width: int = LABEL_WIDTH) -> np.ndarray:
    cell = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(cell, text, (6, height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return cell


def render() -> np.ndarray:
    source = test_image()

    def gap() -> np.ndarray:
        return np.full((source.shape[0], 8, 3), 255, np.uint8)

    rows = [np.hstack([label("", 24), label("fixed", 24, STRIP_WIDTH), gap()[:24], label("rnl", 24, STRIP_WIDTH)])]
    rows.append(np.hstack([label("original", source.shape[0]), source, gap(), source]))
    for name in species.SPECIES:
        fixed = imaging.simulate(source, model.Params(name))
        rnl = imaging.simulate(source, model.Params(name, chroma_scale="rnl"))
        rows.append(np.full((4, rows[0].shape[1], 3), 255, np.uint8))
        rows.append(np.hstack([label(facts.species_label(name), source.shape[0]), fixed, gap(), rnl]))
    return np.vstack(rows)


def matches_file() -> bool:
    stored = cv2.imread(str(OUTPUT), cv2.IMREAD_COLOR)
    return stored is not None and np.array_equal(stored, render())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    if parser.parse_args().check:
        if matches_file():
            return 0
        print(f"{OUTPUT.name} is out of date; run: uv run tools/render_species_grid.py", file=sys.stderr)
        return 1
    OUTPUT.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT), render())
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
