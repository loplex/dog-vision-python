# dog-vision

*A camera feed or a photo, shown with the colours a dog — or one of 16 other animals — can tell apart.*

- [Running it](#running-it) — camera, photo, `--species`, `--adaptation`, `--strength`.
- [The camera window](#the-camera-window) — the species list, sliders and keys.
- [Another GUI toolkit](#another-gui-toolkit) — `LiveSession` and `run(session)`.
- [Species](#species) — every preset, with its cone peaks and source.
- [How it works](#how-it-works) — the model in five steps.
- [What it cannot show](#what-it-cannot-show) — limits, grouped by whether they can be lifted.
- [Checking it](#checking-it) — `--info` and `check_docs.py`.
- [References](#references)

## Running it

The script declares its dependencies inline ([PEP 723](https://peps.python.org/pep-0723/)), so
[uv](https://docs.astral.sh/uv/) is the only thing to install.
The window uses Tkinter, which the Python builds uv installs include; a distribution's own Python
may need its Tk package (`python3-tk` on Debian and Ubuntu).

```sh
uv run dog_vision.py                     # live camera 0, as a dog
uv run dog_vision.py --camera 1          # another camera
uv run dog_vision.py --species horse     # another animal
uv run dog_vision.py photo.jpg           # writes photo.dog.png next to it
uv run dog_vision.py --adaptation 1 sunset.jpg
```

- `--adaptation` (0–1) adapts the cones to the scene instead of to daylight, so a sunset-lit
  scene loses its warm cast the way it would for an eye that has been in it for a while.
- `--strength` (0–1) blends the simulation with the original; 0 is the original.
- `uv run dog_vision.py --help` lists every option.

## The camera window

The camera image is on the left, the controls on the right.
The image follows the window's size; the controls keep theirs.

| Control                         | Does                                                |
|---------------------------------|-----------------------------------------------------|
| *Species* list                  | a click or the arrow keys pick the animal           |
| mouse wheel over the list       | scrolls it                                          |
| *Adaptation to scene* slider    | same as `--adaptation`, in percent                  |
| *Simulation strength* slider    | same as `--strength`, in percent                    |
| *Side by side* box, `m`         | the original beside the simulation, or not          |
| *Reset* button, `r`             | back to the values given on the command line        |
| *Save snapshot* button, `s`     | saves `dog-<species>-<time>.png` in the current dir |
| `q`, Esc                        | quits                                               |

## Another GUI toolkit

The window lives in [`tk_window.py`](tk_window.py) and does nothing but lay out widgets and forward
events.
Everything else sits in `LiveSession` in [`dog_vision.py`](dog_vision.py):

- `render()` returns the current view as an RGB array, or `None` before the first frame.
  The camera is read on a thread of its own, so a GUI can call it from any timer.
- `params` (species, adaptation, strength) and `side_by_side` are plain attributes to set.
- `species_names` lists the species in display order.
- `reset()`, `save_snapshot()` and `error` cover the buttons and a camera that stops.

A window in another toolkit is a module with the same `run(session)` function; `main()` in
`dog_vision.py` imports the window in one line.

## Species

Peaks are the wavelengths of maximum cone sensitivity as measured in each source.
A dash marks a cone monochromat, which has no S cone and sees only shades of grey.

| `--species`          | S cone [nm] | M/L cone [nm] | Source                         |
|----------------------|-------------|---------------|--------------------------------|
| `dog`                | 429         | 555           | Neitz, Geist & Jacobs 1989     |
| `cat`                | 450         | 550           | Guenther & Zrenner 1993        |
| `horse`              | 428         | 539           | Carroll et al. 2001            |
| `cow`                | 451.3       | 555.3         | Jacobs, Deegan & Neitz 1998    |
| `sheep`              | 445.3       | 552.2         | Jacobs, Deegan & Neitz 1998    |
| `goat`               | 443.3       | 552.5         | Jacobs, Deegan & Neitz 1998    |
| `pig`                | 440.7       | 556.7         | Jacobs, Deegan & Neitz 1998    |
| `fallow-deer`        | 453.6       | 542.2         | Jacobs, Deegan & Neitz 1998    |
| `white-tailed-deer`  | 456         | 536.8         | Jacobs, Deegan & Neitz 1998    |
| `guinea-pig`         | 429         | 529           | Jacobs & Deegan 1994           |
| `tree-squirrel`      | 444         | 543           | Blakeslee, Jacobs & Neitz 1988 |
| `ground-squirrel`    | 436.7       | 518.9         | Jacobs, Neitz & Crognale 1985  |
| `ferret`             | 430         | 558           | Calderone & Jacobs 2003        |
| `protanope`          | 420.7       | 530.3         | Stockman & Sharpe 2000         |
| `deuteranope`        | 420.7       | 558.9         | Stockman & Sharpe 2000         |
| `harbour-seal`       | –           | 510           | Crognale et al. 1998           |
| `bottlenose-dolphin` | –           | 524           | Fasick et al. 1998             |

`protanope` and `deuteranope` are humans lacking the L or the M cone; their values are the human
pigment peaks.

## How it works

The method is the one Brettel, Viénot & Mollon (1997) use for human dichromats:

1. Each cone is modelled with the Govardovskii et al. (2000) pigment template, from its peak alone.
2. The display is three Gaussian primaries, balanced so that white excites human cones like daylight.
3. That gives a matrix from linear RGB to the animal's cone excitations.
   For a dichromat it has one direction it cannot see: colours differing only along it look the same.
4. Each pixel is moved along that direction onto the plane R = G, which holds the grey axis and
   the blue–yellow axis.
   Neutral colours therefore stay neutral, and the animal's cones respond to the output exactly as
   to the input.
   A monochromat is moved onto the grey axis instead.
5. Adaptation and strength scale and blend the result.

The whole transform is one 3×3 matrix on linear RGB.
The docstring of [`dog_vision.py`](dog_vision.py) states it as formulas.

## What it cannot show

### Permanent: the information is not in the picture

- **Animals with an ultraviolet cone** — mice, rats, birds, bees — are left out.
  An RGB camera records nothing of what that cone sees.
- **Trichromats** are left out too. The camera is itself trichromatic, so two colours it records
  differently also excite a trichromat's cones differently: nothing merges.
  What changes is how far apart the colours look, and that needs a model of colour discrimination,
  not of cone excitation.
- **Colours a camera merges** stay merged. Two surfaces that look alike to a human, and so to the
  camera, can differ for the animal.
- **What the colours feel like** to the animal is unknowable. The simulation shows which colours
  it cannot tell apart, and nothing more.

### By design: the model is simpler than the eye

- **The lens and the macular pigment are ignored.**
  This is the likely reason the model puts the human deuteranope's neutral point well short of the
  measured value of about 505 nm, while it matches the dog's 480 nm (both from Neitz, Geist & Jacobs 1989).
- **The display primaries are generic**, not those of the screen in front of you.
- **Blue–yellow is a convention.** Which human hues stand for a dichromat's single chromatic axis
  is not fixed by physics.
- **Only colour is simulated** — not the dog's lower acuity, its motion sensitivity or its dim-light vision.

## Checking it

- `uv run dog_vision.py --info --species <name>` prints the derived matrices and checks that grey
  is preserved, that the output excites the cones exactly as the input does, and the neutral point.
- `uv run check_docs.py` checks this README against the code: the species table, the relative links
  and anchors, and the neutral points quoted above.

## References

- Blakeslee, B., Jacobs, G. H. & Neitz, J. (1988). Spectral mechanisms in the tree squirrel retina.
  *J. Comp. Physiol. A* 162, 773–780.
- Brettel, H., Viénot, F. & Mollon, J. D. (1997). Computerized simulation of color appearance for
  dichromats. *J. Opt. Soc. Am. A* 14, 2647–2655.
- Calderone, J. B. & Jacobs, G. H. (2003). Spectral properties and retinal distribution of ferret
  cones. *Visual Neuroscience* 20.
- Carroll, J., Murphy, C. J., Neitz, M. et al. (2001). Photopigment basis for dichromatic color
  vision in the horse. *Journal of Vision* 1.
- Crognale, M. A., Levenson, D. H., Ponganis, P. J., Deegan, J. F. & Jacobs, G. H. (1998). Cone
  spectral sensitivity in the harbor seal (*Phoca vitulina*) and implications for color vision.
  *Can. J. Zool.* 76, 2114–2118.
- Fasick, J. I., Cronin, T. W. et al. (1998). The visual pigments of the bottlenose dolphin
  (*Tursiops truncatus*). *Visual Neuroscience* 15.
- Govardovskii, V. I., Fyhrquist, N., Reuter, T., Kuzmin, D. G. & Donner, K. (2000). In search of
  the visual pigment template. *Visual Neuroscience* 17, 509–528.
- Guenther, E. & Zrenner, E. (1993). The spectral sensitivity of dark- and light-adapted cat
  retinal ganglion cells. *J. Neurosci.* 13, 1543–1550.
- Jacobs, G. H. & Deegan, J. F. (1994). Spectral sensitivity, photopigments, and color vision in the
  guinea pig (*Cavia porcellus*). *Behav. Neurosci.* 108, 993–1004.
- Jacobs, G. H., Deegan, J. F. & Neitz, J. (1998). Photopigment basis for dichromatic color vision
  in cows, goats, and sheep. *Visual Neuroscience* 15, 581–584.
- Jacobs, G. H., Neitz, J. & Crognale, M. (1985). Spectral sensitivity of ground squirrel cones
  measured with ERG flicker photometry. *J. Comp. Physiol. A* 156, 503–509.
- Neitz, J., Geist, T. & Jacobs, G. H. (1989). Color vision in the dog. *Visual Neuroscience* 3,
  119–125.
- Stockman, A. & Sharpe, L. T. (2000). The spectral sensitivities of the middle- and
  long-wavelength-sensitive cones derived from measurements in observers of known genotype.
  *Vision Research* 40, 1711–1737.
- Viénot, F., Brettel, H. & Mollon, J. D. (1999). Digital video colourmaps for checking the
  legibility of displays by dichromats. *Color Research & Application* 24, 243–252.
