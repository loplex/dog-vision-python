"""Simulate dichromatic dog colour vision on a live camera feed or a photo.

Other dichromats, some trichromats and two cone monochromats are available as
presets (SPECIES).

Usage:
    uv run dog-vision                 # live camera 0
    uv run dog-vision --camera 1      # another camera
    uv run dog-vision photo.jpg       # convert a photo, writes photo.dog.png
    uv run dog-vision clip.mp4        # convert a video, writes clip.dog.mp4
    uv run dog-vision --window clip.mp4             # show a file in the window instead
    uv run dog-vision --info          # print the derived model and checks
    uv run dog-vision --species cat   # another dichromat
    uv run dog-vision --species cat --compare dog   # two species side by side

The live window (dog_vision.gui.tk) lists the species on the right; keys: m = toggle
side-by-side / simulation only, d = map of differences, o = open a photo or a video,
r = reset to the command-line values, s = save snapshot, F9 = hide or show the
controls, q or Esc = quit. The window only drives LiveSession, so another GUI
toolkit needs nothing but a module with the same run(session) function.
"""

import argparse
import dataclasses
import sys
from pathlib import Path

import cv2
import numpy as np

from dog_vision.core import model
from dog_vision.core.facts import species_facts
from dog_vision.core.imaging import compose
from dog_vision.core.model import (
    CHROMA_SCALES,
    Params,
    animal_cone_matrix,
    chroma_directions,
    neutral_point,
    rnl_chroma_matrix,
    rnl_metric,
    simulation_matrix,
)
from dog_vision.core.session import LiveSession
from dog_vision.core.species import SPECIES
from dog_vision.core.video import convert_video, converted_path, writer_description


def print_info(params: Params) -> None:
    m_animal = animal_cone_matrix(params)
    t = simulation_matrix(dataclasses.replace(params, strength=1.0))
    np.set_printoptions(precision=4, suppress=True)
    print(f"Species {params.species}, cone peaks {params.cones()} nm")
    print("\nCone matrix M_animal (rows: cones, short to long; columns: linear R, G, B):")
    print(m_animal)
    if len(m_animal) == 2:
        n = np.cross(m_animal[0], m_animal[1])  # spans the null space of a 2x3 matrix
        print("\nConfusion direction n (normalised):", n / np.abs(n).max())
    print("\nSimulation matrix T (linear RGB):")
    print(t)
    print("\nChecks:")
    print("  grey preserved   T @ (1,1,1) =", t @ np.ones(3))
    print("  cone-invariant   max |M_animal @ T - M_animal| =", np.abs(m_animal @ t - m_animal).max())
    if len(m_animal) == 2:
        print(f"  neutral point    {neutral_point(params):.0f} nm")
    if len(m_animal) > 1:
        k, u = rnl_chroma_matrix(params), chroma_directions(m_animal)
        animal = rnl_metric(params.species, np.eye(len(m_animal)))[:-1, :-1]
        human = u.T @ rnl_metric("human", animal_cone_matrix(Params("human"))) @ u
        print("  rnl matches JNDs  max |K'BK - A| / |A| =", np.abs(k.T @ human @ k - animal).max() / np.abs(animal).max())
        if len(m_animal) == 3:
            blue_yellow = np.hstack([np.eye(2), -np.ones((2, 1))]) @ m_animal @ np.array([-0.5, -0.5, 1.0])
            mapped = k @ blue_yellow
            print("  rnl keeps blue    angle of K d to d =", abs(np.arctan2(*mapped[::-1]) - np.arctan2(*blue_yellow[::-1])))
    print()
    for label, value, _description in species_facts(params.species):
        print(f"  {label + ':':15s} {value}")
    print("\nReference values: the dog's neutral point was measured at about 480 nm")
    print("(Neitz, Geist & Jacobs 1989); a human deuteranope is R' = G' = 0.293 R + 0.707 G")
    print("(Viénot, Brettel & Mollon 1999).")


def convert_file(path: Path, params: Params, compare: str | None = None, difference: bool = False) -> None:
    """Convert a photo to <name>.dog.png, or a video to <name>.dog.mp4, next to it."""
    side_by_side = compare is not None or difference
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        out_path = converted_path(path, is_video=False)
        composed, share = compose(image, params, side_by_side, compare, difference)
        if share is not None:
            print(f"{share:.0%} of pixels differ noticeably")
        cv2.imwrite(str(out_path), composed)
        print(f"Wrote {out_path}")
        return
    if not cv2.VideoCapture(str(path)).isOpened():
        sys.exit(f"Cannot read image or video: {path}")
    out_path = converted_path(path, is_video=True)

    def show_progress(share: float) -> None:
        if sys.stderr.isatty():
            print(f"\rConverting: {share:.0%}", end="", file=sys.stderr, flush=True)

    try:
        writer = convert_video(path, out_path, lambda frame: compose(frame, params, side_by_side, compare, difference)[0], show_progress)
    except RuntimeError as error:
        sys.exit(f"\n{error}")
    finally:
        if sys.stderr.isatty():
            print(file=sys.stderr)
    print(f"Wrote {out_path}: {writer_description(writer)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dog-vision", description=f"{__doc__}\n{model.__doc__}", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", nargs="?", type=Path, help="photo or video to convert instead of using the camera")
    parser.add_argument("--window", action="store_true", help="show the photo or video in the window instead of converting it")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--info", action="store_true", help="print the derived model and exit")
    defaults = Params()
    parser.add_argument("--species", choices=SPECIES, default=defaults.species, help="animal to simulate (default %(default)s)")
    parser.add_argument(
        "--adaptation", type=float, default=defaults.adaptation, help="adaptation to the scene mean, 0-1 (default %(default)s)"
    )
    parser.add_argument(
        "--strength", type=float, default=defaults.strength, help="0 = original, 1 = full simulation (default %(default)s)"
    )
    parser.add_argument(
        "--chroma-scale",
        choices=CHROMA_SCALES,
        default=defaults.chroma_scale,
        help="saturation of a dichromat's colour axis: fixed by the projection, or matched"
        " to the animal's discrimination with the RNL model (default %(default)s)",
    )
    parser.add_argument(
        "--compare", choices=SPECIES, help="show this species beside --species instead of the original"
    )
    parser.add_argument(
        "--difference", action="store_true", help="add a map of where the two images differ noticeably"
    )
    parser.add_argument("--acuity", action="store_true", help="blur to the species' visual acuity")
    parser.add_argument(
        "--fov",
        type=float,
        default=defaults.field_of_view,
        help="degrees the image spans horizontally, for --acuity (default %(default)s)",
    )
    args = parser.parse_args()
    params = Params(args.species, args.adaptation, args.strength, args.chroma_scale, args.acuity, args.fov)

    if args.info:
        print_info(params)
    elif args.image and not args.window:
        convert_file(args.image, params, args.compare, args.difference)
    else:
        # The only GUI-specific line in this module.
        from dog_vision.gui import tk as tk_gui

        session = LiveSession(args.camera, params, args.compare, args.image)
        session.difference = args.difference
        try:
            tk_gui.run(session)
        finally:
            session.close()
        if session.error:
            sys.exit(session.error)


if __name__ == "__main__":
    main()
