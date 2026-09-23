# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless"]
# ///
"""Check README.md against the code it describes.

- The species table lists exactly SPECIES, in order, with the same peaks.
- Every relative link points at an existing file, and every #anchor at a heading.
- The neutral points quoted under "What it cannot show" still hold for the model.

External URLs are not fetched. Exits non-zero and names each mismatch.
"""

import re
import sys
from pathlib import Path

import dog_vision as dv

ROOT = Path(__file__).parent


def slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    text = re.sub(r"[`*_]", "", heading.strip().lower())
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def without_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def check_species_table(readme: str) -> list[str]:
    rows = re.findall(r"^\| `([\w-]+)` +\| ([\d.–]+) +\| ([\d.]+) +\|", readme, flags=re.M)
    documented = {name: (() if s == "–" else (float(s),)) + (float(l),) for name, s, l in rows}
    errors = []
    if list(documented) != list(dv.SPECIES):
        errors.append(f"species table lists {list(documented)}, SPECIES has {list(dv.SPECIES)}")
    for name, peaks in dv.SPECIES.items():
        if name in documented and documented[name] != tuple(peaks):
            errors.append(f"species table gives {name} {documented[name]}, SPECIES has {peaks}")
    return errors


def check_links(path: Path) -> list[str]:
    text = without_code_blocks(path.read_text())
    anchors = {slug(h) for h in re.findall(r"^#+ (.+)$", text, flags=re.M)}
    errors = []
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if re.match(r"[a-z]+://", target):
            continue
        file, _, anchor = target.partition("#")
        if file and not (path.parent / file).exists():
            errors.append(f"{path.name}: link to missing file {file}")
        if anchor and not file and anchor not in anchors:
            errors.append(f"{path.name}: link to missing anchor #{anchor}")
    return errors


def check_neutral_points() -> list[str]:
    errors = []
    dog = dv.neutral_point(dv.Params("dog"))
    if abs(dog - 480) > 5:
        errors.append(f"README says the model matches the dog's 480 nm neutral point; it gives {dog:.0f} nm")
    deuteranope = dv.neutral_point(dv.Params("deuteranope"))
    if deuteranope > 505 - 15:
        errors.append(f"README says the deuteranope's is well short of 505 nm; the model gives {deuteranope:.0f} nm")
    return errors


def main() -> int:
    readme = ROOT / "README.md"
    errors = check_species_table(readme.read_text()) + check_links(readme) + check_neutral_points()
    for error in errors:
        print(error, file=sys.stderr)
    print(f"{len(errors)} problem(s)" if errors else "README.md matches the code")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
