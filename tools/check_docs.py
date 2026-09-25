"""Check README.md against the code it describes.

- The species table lists exactly SPECIES, in order, with the same peaks, S-cone shares and sources.
- The acuity table does the same for ACUITY.
- Every relative link points at an existing file, and every #anchor at a heading.
- The neutral points quoted under "What it cannot show" still hold for the model.
- docs/species-grid.png is what render_species_grid.py renders from the current code, and the
  apple figures are what render_photo_figures.py does.
- Every language in i18n names every species and describes every label, and every English
  text it translates still occurs in the code, so none silently stays English.

External URLs are not fetched. Exits non-zero and names each mismatch.
"""

import ast
import re
import sys
from pathlib import Path

import render_photo_figures
import render_species_grid

from dog_vision.core import facts, i18n, model, species
from dog_vision.gui import tk as tk_gui

ROOT = Path(__file__).parent.parent


def slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    text = re.sub(r"[`*_]", "", heading.strip().lower())
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def without_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def expected_row(name: str) -> list[str]:
    """The cells after the name that the species table should hold for a species."""
    peaks = species.SPECIES[name]
    slots = peaks if len(peaks) == 3 else (peaks[0], None, peaks[1]) if len(peaks) == 2 else (None, None, peaks[0])
    cells = ["–" if peak is None else f"{peak:g}" for peak in slots] + [species.PEAKS_FROM[name]]
    if len(peaks) == 1:
        return cells + ["–", ""]
    if name not in species.S_CONE_FRACTION:
        return cells + [f"{species.ASSUMED_S_CONE_FRACTION[0] * 100:.0f} *", "assumed"]
    low, high, source = species.S_CONE_FRACTION[name]
    share = f"{low * 100:.0f}" if round(low * 100) == round(high * 100) else f"{low * 100:.0f}–{high * 100:.0f}"
    return cells + [share, source]


def check_species_table(readme: str) -> list[str]:
    table = readme[readme.index("| `--species`") :].split("\n\n")[0].splitlines()[2:]  # skip header, rule
    rows = {}
    for line in table:
        match = re.match(r"^\| `([\w-]+)` +\|(.*)\|$", line)
        if match:
            rows[match[1]] = [cell.strip() for cell in match[2].split("|")]
    errors = []
    if list(rows) != list(species.SPECIES):
        errors.append(f"species table lists {list(rows)}, SPECIES has {list(species.SPECIES)}")
    for name in species.SPECIES:
        if name in rows and rows[name] != expected_row(name):
            errors.append(f"species table row {name}: {rows[name]}, the code gives {expected_row(name)}")
    return errors


def acuity_cells(name: str) -> list[str]:
    """The cells the acuity table should hold for a species: across, up, source."""
    if name not in species.ACUITY:
        return ["–", "–", "not found measured"]
    (across, up), source = species.ACUITY[name]
    return [f"{across:.3g}", f"{up:.3g}", source]


def check_acuity_table(readme: str) -> list[str]:
    start = re.search(r"^\| `--species` +\| Side by side", readme, flags=re.MULTILINE).start()
    table = readme[start:].split("\n\n")[0].splitlines()[2:]
    rows = {}
    for line in table:
        match = re.match(r"^\| `([\w-]+)` +\|(.*)\|$", line)
        if match:
            rows[match[1]] = [cell.strip() for cell in match[2].split("|")]
    errors = []
    if list(rows) != list(species.SPECIES):
        errors.append(f"acuity table lists {list(rows)}, SPECIES has {list(species.SPECIES)}")
    for name in species.SPECIES:
        if name in rows and rows[name] != acuity_cells(name):
            errors.append(f"acuity table row {name}: {rows[name]}, the code gives {acuity_cells(name)}")
    return errors


def check_links(path: Path) -> list[str]:
    text = without_code_blocks(path.read_text())
    anchors = {slug(h) for h in re.findall(r"^#+ (.+)$", text, flags=re.MULTILINE)}
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
    dog = model.neutral_point(model.Params("dog"))
    if abs(dog - 480) > 5:
        errors.append(f"README says the model matches the dog's 480 nm neutral point; it gives {dog:.0f} nm")
    deuteranope = model.neutral_point(model.Params("deuteranope"))
    if deuteranope > 505 - 15:
        errors.append(f"README says the deuteranope's is well short of 505 nm; the model gives {deuteranope:.0f} nm")
    return errors


def string_constants(path: Path) -> set[str]:
    """Every string literal in a Python file, with implicitly concatenated parts joined as Python joins them."""
    tree = ast.parse(path.read_text())
    return {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}


def check_translations() -> list[str]:
    errors = []
    # Every module but the catalogue, which holds each English text it translates.
    modules = [path for path in (ROOT / "src" / "dog_vision").rglob("*.py") if path.name != "i18n.py"]
    code = set().union(*(string_constants(path) for path in modules))
    for name, language in i18n.LANGUAGES.items():
        if language.species and list(language.species) != list(species.SPECIES):
            errors.append(f"i18n language {name} names {list(language.species)}, SPECIES has {list(species.SPECIES)}")
        for english in language.texts:
            if english not in code:
                errors.append(f"i18n language {name} translates {english!r}, which the code no longer contains")
        if language.texts:
            for label, english in {**facts.FACT_DESCRIPTIONS, **tk_gui.DESCRIPTIONS}.items():
                if english not in language.texts:
                    errors.append(f"i18n language {name} has no translation of the description of {label!r}")
    return errors


def main() -> int:
    readme = ROOT / "README.md"
    text = readme.read_text()
    errors = check_species_table(text) + check_acuity_table(text) + check_links(readme) + check_neutral_points() + check_translations()
    if not render_species_grid.matches_file():
        errors.append("docs/species-grid.png is out of date; run: uv run tools/render_species_grid.py")
    for name in render_photo_figures.stale():
        errors.append(f"docs/{name} is out of date; run: uv run tools/render_photo_figures.py")
    for error in errors:
        print(error, file=sys.stderr)
    print(f"{len(errors)} problem(s)" if errors else "README.md matches the code")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
