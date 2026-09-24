# dog-vision

*A camera feed or a photo, shown with the colours a dog — or another animal — can tell apart.*

- [Running it](#running-it) — camera, photo, video, `--window`, `--species`, `--compare`, `--difference`,
  `--adaptation`, `--strength`, `--chroma-scale`, `--acuity`, `--fov`.
- [The camera window](#the-camera-window) — the species list, sliders, keys, language and tooltips.
- [Another GUI toolkit](#another-gui-toolkit) — `LiveSession` and `run(session)`.
- [Species](#species) — every preset, with its cone peaks and source, and how each sees a test chart.
- [How it works](#how-it-works) — the model in five steps.
- [Colour saturation](#colour-saturation) — `fixed` or `rnl`, and what the second rests on.
- [Acuity](#acuity) — blurring to what each species resolves, and when that shows.
- [Map of differences](#map-of-differences) — where two views differ noticeably, and by what measure.
- [What it cannot show](#what-it-cannot-show) — limits, grouped by whether they can be lifted.
- [Checking it](#checking-it) — `--info` and `check_docs.py`.
- [References](#references)

## Running it

The script declares its dependencies inline ([PEP 723](https://peps.python.org/pep-0723/)), so
[uv](https://docs.astral.sh/uv/) is the only thing to install.
The window uses Tkinter.
The script asks uv for a Python already on the system, because the Tk in uv's own Python builds
lacks Xft and draws text without antialiasing; the window warns when that is the Tk it got.
A distribution's Python may need its Tk package (`python3-tk` on Debian and Ubuntu).

```sh
uv run dog_vision.py                     # live camera 0, as a dog
uv run dog_vision.py --camera 1          # another camera
uv run dog_vision.py --species horse     # another animal
uv run dog_vision.py --species cow --compare horse   # two animals side by side
uv run dog_vision.py --species cow --compare horse --difference   # and where they differ
uv run dog_vision.py photo.jpg           # writes photo.dog.png next to it
uv run dog_vision.py clip.mp4            # writes clip.dog.mp4 next to it
uv run dog_vision.py --window clip.mp4   # shows it in the window instead
uv run dog_vision.py --adaptation 1 sunset.jpg
```

- `--compare` puts a second species where the original would be, with the same settings, so two
  animals can be told apart directly; with a photo, both go into the output side by side.
- `--difference` adds a third image marking where the two differ noticeably; see
  [Map of differences](#map-of-differences).
- `--adaptation` (0–1) adapts the cones to the scene instead of to daylight, so a sunset-lit
  scene loses its warm cast the way it would for an eye that has been in it for a while.
- `--strength` (0–1) blends the simulation with the original; 0 is the original.
- `--chroma-scale fixed|rnl` picks how saturated a dichromat's colours come out; see
  [Colour saturation](#colour-saturation).
- `--acuity` blurs to the species' visual acuity, and `--fov` says how many degrees the image spans;
  see [Acuity](#acuity).
- A video is converted frame by frame with the same options as a photo, at its own frame rate; see
  [Video is written with the best method the system has](#video-is-written-with-the-best-method-the-system-has).
- `--window` shows the photo or video in the window instead of converting it, and the window can
  convert it from there.
- `uv run dog_vision.py --help` lists every option.

### Video is written with the best method the system has

With [ffmpeg](https://ffmpeg.org/) installed, the sound of the original is carried over, and the
video is encoded with the first of these that works on the machine:

1. H.265 in software (`libx265`)
2. H.265 in hardware: Apple VideoToolbox, NVIDIA, Intel Quick Sync, AMD
3. H.264 in software (`libx264`), then in the same hardware
4. MPEG-4 Part 2, which every ffmpeg has

- **Each is tried on one frame first**, since ffmpeg lists a hardware encoder whether or not its
  hardware is there.
- **Without ffmpeg, OpenCV writes the video**, with H.265, H.264 or MPEG-4, whichever its build has,
  and without sound: OpenCV does not handle audio.
- **The colours are encoded and tagged as BT.709**, the HD standard whose primaries sRGB shares, so
  that players do not read them as the older BT.601 and shift every colour.
- The command line and the window say which method was used, and whether the sound was kept.

## The camera window

The images are on the left, with a line under them saying what each shows, and the controls on the
right.
The images follow the window's size; the controls keep theirs.

| Control                         | Does                                                |
|---------------------------------|-----------------------------------------------------|
| *Species* list                  | a click or the arrow keys pick the animal           |
| *Selected species* panel        | its cones, their sources and its RNL factors        |
| mouse wheel over the list       | scrolls it                                          |
| *Adaptation to scene* slider    | same as `--adaptation`, in percent                  |
| *Simulation strength* slider    | same as `--strength`, in percent                    |
| *Colour saturation* choice      | same as `--chroma-scale`                            |
| *Acuity* box and slider         | same as `--acuity` and `--fov`                      |
| *Side by side* box, `m`         | a second image beside the simulation, or not        |
| *Left image* choice             | the original, or another species, as `--compare`    |
| *Map of differences* box, `d`   | a third image marking where left and right differ   |
| *Reset* button, `r`             | back to the values given on the command line        |
| *Save snapshot* button, `s`     | saves `dog-<species>-<time>.png` in the current dir |
| *Open file…* button, `o`        | a photo or a video instead of the camera            |
| *Camera* button                 | back to the camera                                  |
| *Convert file* button           | writes the open file as shown, at full size         |
| *Language* choice               | the window's language, starting as the system's     |
| `q`, Esc                        | quits                                               |

Resting the pointer on a label, a box or a button shows what it means, in a few paragraphs.

### The window starts in the system's language, looked up as gettext does

The window speaks English and Czech.
A language is one entry in `LANGUAGES` in [`i18n.py`](i18n.py): its name, its decimal point, and
its translations, each keyed by the English text as in gettext.

The first of these that is set decides, and a language the window does not speak gives English:

1. `LANGUAGE`, which may list several languages, as in `cs:en_GB:en`
2. `LC_ALL`
3. `LC_MESSAGES`
4. `LANG`
5. on Windows, the user's interface language; elsewhere, the locale Python started in

So `LANGUAGE=en uv run dog_vision.py` opens it in English whatever the system says.
Only the window is translated: `--help`, `--info` and what a photo conversion prints stay English.

## Another GUI toolkit

The window lives in [`tk_window.py`](tk_window.py) and does nothing but lay out widgets and forward
events.
Everything else sits in `LiveSession` in [`dog_vision.py`](dog_vision.py):

- `render()` returns the current view as an RGB array, or `None` before the first frame.
  The camera or a video is read on a thread of its own, so a GUI can call it from any timer.
- `open_file()` and `open_camera()` change the source; `source` is the open file, or `None` for the
  camera, and `source_name()` says what is shown.
- `convert_source()` converts the open file in the background with the current settings;
  `converting` and `conversion_status()` say how it stands.
- `params` (species, adaptation, strength, chroma scale, acuity, field of view), `side_by_side`,
  `compare` (the species on the left, or `None` for the original) and `difference` are plain
  attributes to set.
- `caption()` says what the rendered view shows, left to right.
- `species_names` and `chroma_scales` list the choices in display order; `species_labels` adds
  each species' kind of colour vision to its name.
- `species_facts()` returns what is known about the current species as (label, value, description)
  rows.
- `language` is a code from `languages`, which maps each to the language's own name, and sets
  the language of `species_labels`, `species_facts()` and `caption()`; `translate()` gives the
  window's own texts in it.
- `reset()`, `save_snapshot()` and `error` cover the buttons and a camera that stops.

A window in another toolkit is a module with the same `run(session)` function; `main()` in
`dog_vision.py` imports the window in one line.

## Species

Peaks are the wavelengths of maximum cone sensitivity as measured in each source.
A dichromat's longer cone is listed under L whatever it is called elsewhere, and a cone monochromat
has only that one, so it sees shades of grey.
The S-cone share is used only by the [`rnl` colour saturation](#colour-saturation); where a
source reports a range across the retina, the middle of it is used, and a star marks a species
with no measurement found, which gets the assumed 10 %.

| `--species`          | S [nm] | M [nm] | L [nm] | Peaks from                     | S cones [%] | Share from              |
|----------------------|--------|--------|--------|--------------------------------|-------------|-------------------------|
| `dog`                | 429    | –      | 555    | Neitz, Geist & Jacobs 1989     | 10–18       | Mowat et al. 2008       |
| `cat`                | 450    | –      | 550    | Guenther & Zrenner 1993        | 10–20       | Linberg et al. 2001     |
| `horse`              | 428    | –      | 539    | Carroll et al. 2001            | 10–25       | Sandmann et al. 1996    |
| `cow`                | 451.3  | –      | 555.3  | Jacobs, Deegan & Neitz 1998    | 5–10        | Schiviz et al. 2008     |
| `sheep`              | 445.3  | –      | 552.2  | Jacobs, Deegan & Neitz 1998    | 5–10        | Schiviz et al. 2008     |
| `goat`               | 443.3  | –      | 552.5  | Jacobs, Deegan & Neitz 1998    | 10 *        | assumed                 |
| `pig`                | 440.7  | –      | 556.7  | Jacobs, Deegan & Neitz 1998    | 5–10        | Schiviz et al. 2008     |
| `fallow-deer`        | 453.6  | –      | 542.2  | Jacobs, Deegan & Neitz 1998    | 10 *        | assumed                 |
| `white-tailed-deer`  | 456    | –      | 536.8  | Jacobs, Deegan & Neitz 1998    | 10 *        | assumed                 |
| `guinea-pig`         | 429    | –      | 529    | Jacobs & Deegan 1994           | 10 *        | assumed                 |
| `tree-squirrel`      | 444    | –      | 543    | Blakeslee, Jacobs & Neitz 1988 | 10 *        | assumed                 |
| `ground-squirrel`    | 436.7  | –      | 518.9  | Jacobs, Neitz & Crognale 1985  | 7           | Kryger et al. 1998      |
| `ferret`             | 430    | –      | 558    | Calderone & Jacobs 2003        | 7           | Calderone & Jacobs 2003 |
| `protanope`          | 420.7  | –      | 530.3  | Stockman & Sharpe 2000         | 8–12        | Curcio et al. 1991      |
| `deuteranope`        | 420.7  | –      | 558.9  | Stockman & Sharpe 2000         | 8–12        | Curcio et al. 1991      |
| `human`              | 420.7  | 530.3  | 558.9  | Stockman & Sharpe 2000         | 8–12        | Curcio et al. 1991      |
| `protanomalous`      | 420.7  | 530.3  | 548.9  | L shifted 10 nm, see text      | 8–12        | Curcio et al. 1991      |
| `deuteranomalous`    | 420.7  | 540.3  | 558.9  | M shifted 10 nm, see text      | 8–12        | Curcio et al. 1991      |
| `macaque`            | 431    | 536    | 565    | Bowmaker et al. 1978, 1991     | 10 *        | assumed                 |
| `howler-monkey`      | 430    | 530    | 562    | Jacobs et al. 1996, S assumed  | 10 *        | assumed                 |
| `marmoset-female`    | 423    | 543    | 563    | Travis 1988, Williams 1992     | 10 *        | assumed                 |
| `harbour-seal`       | –      | –      | 510    | Crognale et al. 1998           | –           |                         |
| `bottlenose-dolphin` | –      | –      | 524    | Fasick et al. 1998             | –           |                         |

Notes on the human and primate rows:

- `protanope` and `deuteranope` are humans lacking the L or the M cone; their values are the human
  pigment peaks.
- `protanomalous` and `deuteranomalous` are anomalous trichromats of moderate severity: the L or the
  M cone is shifted 10 nm towards the other.
  Machado, Oliveira & Fernandes (2009) model severity the same way, with a 20 nm shift standing for
  dichromacy.
- `human` is normal colour vision, and the reference the [`rnl` scale](#colour-saturation) measures
  every other species against, so both its images are the original.
- `marmoset-female` is a female with the 543 nm and 563 nm variants of the M/L pigment; males, and
  females with two copies of one variant, are dichromats.
- Every trichromat is given one L cone per M cone. Humans with normal colour vision range from about
  twice as many L as M cones to the reverse (Roorda & Williams 1999).

### How each species sees a test chart

![A hue sweep and eight colour patches as every species sees them, with both colour
saturation scales](docs/species-grid.png)

[`render_species_grid.py`](render_species_grid.py) draws this image from the code, and
`check_docs.py` fails when it is out of date.

### Why most mammals look alike

- **They share the same two cone genes.** Every dichromatic mammal here inherited one S and one L
  pigment from a common ancestor; species differ only in how those are tuned, by a few tens of
  nanometres at most.
- **Cone sensitivities are broad**, about 100 nm wide at half height, so a shift of 10 nm changes
  little.
- **The visible differences are small but real:** reds are darker for an L cone at shorter
  wavelengths (horse, ground squirrel), and the hue a species sees as white moves along the sweep.
- **The large steps are between kinds of colour vision** — trichromat, dichromat, monochromat — not
  between species of one kind.

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

6. The saturation of the result is set by one of two [scales](#colour-saturation).

The whole transform is one 3×3 matrix on linear RGB.
The docstring of [`dog_vision.py`](dog_vision.py) states it as formulas.

## Colour saturation

Step 4 fixes which colours merge, but not how saturated the rest look: the plane has no natural
scale. `--chroma-scale` picks one of two:

| Scale   | Saturation of the animal's colour axes                                         |
|---------|--------------------------------------------------------------------------------|
| `fixed` | as the projection leaves it; the default, and what earlier versions always did |
| `rnl`   | scaled so that a step the animal can just tell apart is one a human can        |

A dichromat has one colour axis, blue–yellow. A trichromat has two, and for it `fixed` leaves the
image as it is, since nothing merges: `rnl` is the only way it looks different.

`rnl` counts just-noticeable differences (JNDs) the same way for the animal and for a human:

- **Both are judged by the receptor noise limited (RNL) model** of Vorobyev & Osorio (1998), around
  grey.
  It turns each cone's noise, set by the share of cones of that class, into a JND.
- **The human is a normal trichromat** looking at the output, with the `human` row's cones and
  shares.
- **The output is rescaled until the two counts agree.** The lightness is untouched.
  A trichromat has two axes: blue–yellow keeps its hue, as in the `fixed` scale, and changes only
  in saturation, while the other axis takes up the rest of the difference.
  `--info` and the window show the factors for the species chosen.
- **`human` therefore comes out exactly as the original**, and an animal shows how its
  discrimination compares with ours: below 1 it tells colours apart worse, above 1 better.

### Caveat: the animal's cones are assumed to be as noisy as ours

- **RNL needs each cone's noise as a Weber fraction**, and that is measured for almost no mammal.
  Taking the same value for animal and human makes it cancel, so none has to be chosen; the
  assumption is that it is the same.
- **Some S-cone shares are assumed**, marked with a star in the [species table](#species), and every
  trichromat's L:M ratio is.
- **RNL is a threshold model.** It is matched near grey; strongly saturated colours are
  extrapolated.
- **Monochromats** have no chromatic axis, so both scales give the same grey image.

## Acuity

`--acuity` removes the detail finer than the species resolves, the way AcuityView does
(Caves & Johnsen 2018): the image is filtered with the modulation transfer function
exp(−3.56 (f / acuity)²), f in cycles per degree, which is a Gaussian blur in linear light.

Acuity in cycles per degree, where a measurement was found:

| `--species`          | Side by side [c/°] | One above another [c/°] | Source                                     |
|----------------------|--------------------|-------------------------|--------------------------------------------|
| `dog`                | 11.6               | 11.6                    | Odom et al. 1983                           |
| `cat`                | 10                 | 10                      | Wässle 1971                                |
| `horse`              | 23.3               | 23.3                    | Timney & Keil 1992                         |
| `cow`                | 2.6                | 1.6                     | Rehkämper et al. 2000                      |
| `sheep`              | 12.8               | 12.8                    | Sumita et al. 2013, 11.7 to 14             |
| `goat`               | –                  | –                       | not found measured                         |
| `pig`                | –                  | –                       | not found measured                         |
| `fallow-deer`        | –                  | –                       | not found measured                         |
| `white-tailed-deer`  | –                  | –                       | not found measured                         |
| `guinea-pig`         | –                  | –                       | not found measured                         |
| `tree-squirrel`      | 2.8                | 2.8                     | Jacobs, Birch & Blakeslee 1982, 1.8 to 3.8 |
| `ground-squirrel`    | 4                  | 4                       | Jacobs et al. 1980                         |
| `ferret`             | –                  | –                       | not found measured                         |
| `protanope`          | 72                 | 72                      | Land & Nilsson 2012                        |
| `deuteranope`        | 72                 | 72                      | Land & Nilsson 2012                        |
| `human`              | 72                 | 72                      | Land & Nilsson 2012                        |
| `protanomalous`      | 72                 | 72                      | Land & Nilsson 2012                        |
| `deuteranomalous`    | 72                 | 72                      | Land & Nilsson 2012                        |
| `macaque`            | 60                 | 60                      | Nature Neuroscience 2024, about 60         |
| `howler-monkey`      | –                  | –                       | not found measured                         |
| `marmoset-female`    | 30                 | 30                      | Troilo, Howland & Judge 1993               |
| `harbour-seal`       | 5.5                | 5.5                     | Hanke & Dehnhardt 2009, in air             |
| `bottlenose-dolphin` | 3.66               | 3.66                    | Herman et al. 1975, 8.2 arcmin stripes     |

- **Cattle resolve detail side by side better than one above another**; their pupil is a horizontal
  oval, and the blur is stretched accordingly.
- **A range in the source** (sheep, tree squirrel) is replaced by its middle, and the source column
  says so.
- **Species without a measurement** are left sharp, and the window says so.

### Caveat: the blur needs to know how wide the image is in degrees

- **`--fov` is that angle**, 60° by default, which is typical of a camera but only a guess for a
  photo. Halving it halves the blur.
- **An image shows the blur only if it has more pixels per degree than the species resolves.**
  An image 640 pixels wide spanning 60° has about 11 pixels per degree, so a dog's blur is under half
  a pixel and invisible, while a 4000-pixel photo of the same scene blurs it by over two pixels.
- **The result is right when it is seen at the angle it spans.** Viewed smaller, the viewer's own
  acuity blurs it further; viewed larger, blur a human would not notice becomes visible.

## Map of differences

`--difference`, or the window's *Map of differences*, adds a third image beside the two:

- **Grey** where the left and right images look the same.
- **Red** where they differ by more than one just-noticeable difference, deeper the larger the
  difference; the caption gives the share of pixels that do.

It compares the two images as a human sees them, in CIELAB, where one just-noticeable difference is
about ΔE\*ab 2.3 (Mahy et al. 1994).
That catches differences in colour and in sharpness alike.

### Caveat: it measures the two images, not the two animals

- **A human looks at both images**, so the map says what differs between two renderings, each
  already reduced to what its animal can tell apart.
- **ΔE\*ab 2.3 is an average.** The true threshold varies across colour space, so the edge of the
  red region is approximate.
- **Camera noise differs between a sharp and a blurred image**, so with `--acuity` noise shows up as
  scattered red specks.

## What it cannot show

### Permanent: the information is not in the picture

- **Animals with an ultraviolet cone** — mice, rats, birds, bees — are left out.
  An RGB camera records nothing of what that cone sees.
- **For trichromats only the spacing of colours can change.** The camera is itself trichromatic, so
  two colours it records differently also excite a trichromat's cones differently: nothing merges.
  The `rnl` scale shows how far apart they look instead.
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
- **Colour and acuity are simulated** — not motion sensitivity or dim-light vision.

## Checking it

- `uv run dog_vision.py --info --species <name>` prints the derived matrices and checks that grey
  is preserved, that the output excites the cones exactly as the input does, and the neutral point.
- `uv run check_docs.py` checks this README against the code: the species table with its S-cone
  shares and sources, the relative links and anchors, the neutral points quoted above, and that the
  test-chart image is what the code renders.
  It also checks that each translation in [`i18n.py`](i18n.py) names every species and describes
  every label, and that every English text it translates still occurs in the code.

## References

- Blakeslee, B., Jacobs, G. H. & Neitz, J. (1988). Spectral mechanisms in the tree squirrel retina.
  *J. Comp. Physiol. A* 162, 773–780.
- Bowmaker, J. K. et al. (1978). The visual pigments of rods and cones in the rhesus monkey,
  *Macaca mulatta*. *J. Physiol.*
- Bowmaker, J. K., Astell, S., Hunt, D. M. & Mollon, J. D. (1991). Photosensitive and photostable
  pigments in the retinae of Old World monkeys. *J. Exp. Biol.*
- Brettel, H., Viénot, F. & Mollon, J. D. (1997). Computerized simulation of color appearance for
  dichromats. *J. Opt. Soc. Am. A* 14, 2647–2655.
- Calderone, J. B. & Jacobs, G. H. (2003). Spectral properties and retinal distribution of ferret
  cones. *Visual Neuroscience* 20.
- Carroll, J., Murphy, C. J., Neitz, M. et al. (2001). Photopigment basis for dichromatic color
  vision in the horse. *Journal of Vision* 1.
- Caves, E. M. & Johnsen, S. (2018). AcuityView: an R package for portraying the effects of visual
  acuity on scenes observed by an animal. *Methods Ecol. Evol.* 9, 793–797.
- Crognale, M. A., Levenson, D. H., Ponganis, P. J., Deegan, J. F. & Jacobs, G. H. (1998). Cone
  spectral sensitivity in the harbor seal (*Phoca vitulina*) and implications for color vision.
  *Can. J. Zool.* 76, 2114–2118.
- Curcio, C. A. et al. (1991). Distribution and morphology of human cone photoreceptors stained with
  anti-blue opsin. *J. Comp. Neurol.* 312.
- Fasick, J. I., Cronin, T. W. et al. (1998). The visual pigments of the bottlenose dolphin
  (*Tursiops truncatus*). *Visual Neuroscience* 15.
- Govardovskii, V. I., Fyhrquist, N., Reuter, T., Kuzmin, D. G. & Donner, K. (2000). In search of
  the visual pigment template. *Visual Neuroscience* 17, 509–528.
- Guenther, E. & Zrenner, E. (1993). The spectral sensitivity of dark- and light-adapted cat
  retinal ganglion cells. *J. Neurosci.* 13, 1543–1550.
- Hanke, F. D. & Dehnhardt, G. (2009). Aerial visual acuity in harbor seals (*Phoca vitulina*) as a
  function of luminance. *J. Comp. Physiol. A* 195.
- Herman, L. M., Peacock, M. F., Yunker, M. P. & Madsen, C. J. (1975). Bottle-nosed dolphin:
  double-slit pupil yields equivalent aerial and underwater diurnal acuity. *Science* 189.
- Jacobs, G. H. & Deegan, J. F. (1994). Spectral sensitivity, photopigments, and color vision in the
  guinea pig (*Cavia porcellus*). *Behav. Neurosci.* 108, 993–1004.
- Jacobs, G. H., Birch, D. G. & Blakeslee, B. (1982). Visual acuity and spatial contrast sensitivity
  in tree squirrels. *Behavioural Processes* 7.
- Jacobs, G. H., Blakeslee, B., McCourt, M. E. & Tootell, R. B. H. (1980). Visual sensitivity of
  ground squirrels to spatial and temporal luminance variations. *J. Comp. Physiol. A* 136.
- Jacobs, G. H., Deegan, J. F. & Neitz, J. (1998). Photopigment basis for dichromatic color vision
  in cows, goats, and sheep. *Visual Neuroscience* 15, 581–584.
- Jacobs, G. H., Neitz, J. & Crognale, M. (1985). Spectral sensitivity of ground squirrel cones
  measured with ERG flicker photometry. *J. Comp. Physiol. A* 156, 503–509.
- Jacobs, G. H., Neitz, M., Deegan, J. F. & Neitz, J. (1996). Trichromatic colour vision in New
  World monkeys. *Nature* 382.
- Kryger, Z. et al. (1998). The topography of rod and cone photoreceptors in the retina of the
  ground squirrel. *Visual Neuroscience* 15.
- Land, M. F. & Nilsson, D.-E. (2012). *Animal Eyes*, 2nd edition. Oxford University Press.
- Linberg, K. A., Lewis, G. P. et al. (2001). Distribution of S- and M-cones in normal and
  experimentally detached cat retina. *J. Comp. Neurol.* 430.
- Machado, G. M., Oliveira, M. M. & Fernandes, L. A. F. (2009). A physiologically-based model for
  simulation of color vision deficiency. *IEEE Trans. Vis. Comput. Graph.* 15.
- Mahy, M., Van Eycken, L. & Oosterlinck, A. (1994). Evaluation of uniform color spaces developed
  after the adoption of CIELAB and CIELUV. *Color Research & Application* 19, 105–121.
- Mowat, F. M. et al. (2008). Topographical characterization of cone photoreceptors and the area
  centralis of the canine retina. *Molecular Vision* 14.
- *Multiple loci for foveolar vision in macaque monkey visual cortex* (2024). *Nature Neuroscience*.
- Neitz, J., Geist, T. & Jacobs, G. H. (1989). Color vision in the dog. *Visual Neuroscience* 3,
  119–125.
- Odom, J. V., et al. (1983). Canine visual acuity: retinal and cortical field potentials evoked by
  pattern stimulation. *Am. J. Physiol.* 245.
- Rehkämper, G., Perrey, A., Werner, C. W., Opfermann-Rüngeler, C. & Görlach, A. (2000). Visual
  perception and stimulus orientation in cattle. *Vision Research* 40, 2489–2497.
- Roorda, A. & Williams, D. R. (1999). The arrangement of the three cone classes in the living
  human eye. *Nature* 397, 520–522.
- Sandmann, D., Boycott, B. B. & Peichl, L. (1996). Blue-cone horizontal cells in the retinae of
  horses and other Equidae. *J. Neurosci.* 16.
- Schiviz, A. N., Ruf, T., Kuebber-Heiss, A., Schubert, C. & Ahnelt, P. K. (2008). Retinal cone
  topography of artiodactyl mammals: influence of body height and habitat. *J. Comp. Neurol.* 507,
  1336–1350.
- Stockman, A. & Sharpe, L. T. (2000). The spectral sensitivities of the middle- and
  long-wavelength-sensitive cones derived from measurements in observers of known genotype.
  *Vision Research* 40, 1711–1737.
- Sumita, S., Prescott, N. B., Broom, D. M., Wathes, C. M. & Phillips, C. J. C. (2013). Visual
  discrimination learning and spatial acuity in sheep. *Appl. Anim. Behav. Sci.* 147.
- Timney, B. & Keil, K. (1992). Visual acuity in the horse. *Vision Research* 32.
- Travis, D. S., Bowmaker, J. K. & Mollon, J. D. (1988). Polymorphism of visual pigments in a
  callitrichid monkey. *Vision Research* 28, 481–490.
- Troilo, D., Howland, H. C. & Judge, S. J. (1993). Visual optics and retinal cone topography in the
  common marmoset (*Callithrix jacchus*). *Vision Research* 33.
- Viénot, F., Brettel, H. & Mollon, J. D. (1999). Digital video colourmaps for checking the
  legibility of displays by dichromats. *Color Research & Application* 24, 243–252.
- Vorobyev, M. & Osorio, D. (1998). Receptor noise as a determinant of colour thresholds.
  *Proc. R. Soc. B* 265, 351–358.
- Williams, A. J. et al. (1992). The polymorphic photopigments of the marmoset: spectral tuning and
  genetic basis. *EMBO J.* 11.
- Wässle, H. (1971), as listed in the
  [micaToolbox acuity list](http://www.empiricalimaging.com/knowledge-base/list-of-animal-spatial-acuities/).
