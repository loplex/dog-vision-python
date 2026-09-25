"""Tkinter window for dog_vision's live camera view.

This module only lays out widgets and forwards events to a LiveSession; the
session does the capturing, simulating and saving. A window in another toolkit
is a module with the same run(session) function.
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font, ttk
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dog_vision.core.session import LiveSession

FRAME_INTERVAL_MS = 15
TOOLTIP_DELAY_MS = 500
TOOLTIP_WIDTH = 380  # pixels a tooltip's text wraps at
# What the open dialog lists; Tk matches patterns case-sensitively on some systems, so both cases.
FILE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp", "mp4", "mov", "m4v", "avi", "mkv", "webm"]
MIN_LIST_ROWS = 4  # rows of the species list the panel keeps before it scrolls instead
WHEEL_LINES = 3  # lines of text the panel scrolls by per notch of the mouse wheel
CAPTION_GAP = 12  # pixels kept free between the captions of two images
CAPTION_PAD = 4  # pixels above and below the captions
IMAGE_BACKGROUND = "#282828"
CAPTION_COLOUR = "#d8d8d8"
FACT_GAP = 8  # pixels between a species fact's name and its value
PERCENT_SLIDERS = {"adaptation": "Adaptation to scene [%]", "strength": "Simulation strength [%]"}

# What a control does, shown while the pointer rests on it; keyed by the control's own text.
DESCRIPTIONS = {
    "Species": (
        "The animal to simulate, with its kind of colour vision in brackets."
        "\n\nMost dichromatic mammals look much alike: they share the same two cone genes, tuned a few tens"
        " of nanometres apart at most. The large steps are between kinds of colour vision, not between"
        " species of one kind."
    ),
    "Adaptation to scene [%]": (
        "How far the cones adapt to the scene instead of to daylight."
        "\n\nAt 0 the eye is adapted to daylight, so a scene lit by a warm sunset looks warm. At 100 each"
        " cone's signal is scaled so that the scene's average colour becomes neutral, the way an eye that has"
        " been in that light for a while stops noticing its cast (von Kries adaptation to a \"grey world\")."
    ),
    "Simulation strength [%]": (
        "Blends the simulation with the original image: 0 shows the original, 100 the full simulation."
        "\n\nIn between, the colours the animal cannot tell apart are only partly merged."
    ),
    "Fixed by the projection": (
        "Colours the animal cannot tell apart are merged, and the rest keep the saturation the projection"
        " happens to give them."
        "\n\nThe merging says which colours look alike, but not how vivid the others look: that has no"
        " natural scale. For a trichromat nothing merges, so this leaves the image as it is."
    ),
    "Matched to discrimination (RNL)": (
        "Saturation is rescaled so that a human looking at the image can tell as many colour steps apart as"
        " the animal can in the scene. An animal that tells colours apart worse than we do gets paler colours."
        "\n\nThe steps are counted by the receptor noise limited model, from the share of each kind of cone;"
        " the factors are under RNL scale in Selected species."
        "\n\nIt assumes the animal's cones are as noisy as ours, and it is matched near grey, so strongly"
        " saturated colours are extrapolated."
    ),
    "Blur to the species' acuity": (
        "Removes the detail finer than the species resolves, with a blur matching its measured acuity"
        " (Acuity in Selected species)."
        "\n\nThe blur shows only if the image has more pixels per degree than the species resolves. A camera"
        " image 640 pixels wide spanning 60° has about 11 pixels per degree, so a dog's blur is under half a"
        " pixel and invisible, while a large photo shows it clearly."
    ),
    "Image spans [degrees]": (
        "How many degrees of view the image spans from left to right. The blur is set per degree, so this"
        " decides how many pixels wide it is: halving the angle halves the blur."
        "\n\n60° is typical of a camera. The result is right when the image is seen at that same angle; seen"
        " smaller, your own acuity blurs it further."
    ),
    "Side by side (m)": (
        "Shows a second image to the left of the simulation: the original, or another species chosen under"
        " Left image. Unchecked, only the simulation is shown."
    ),
    "Left image": (
        "What the left image shows: the original, or another species rendered with the same settings, so"
        " that two animals can be compared directly."
    ),
    "Map of differences (d)": (
        "Adds a third image: grey where the left and right images look the same, red where they differ by more"
        " than one just-noticeable difference, deeper the larger the difference."
        "\n\nIt compares the two images as a human sees them, in CIELAB, where one just-noticeable difference"
        " is about ΔE 2.3 (Mahy et al. 1994), so it catches differences in colour and in sharpness alike."
        "\n\nIt measures the two renderings, each already reduced to what its animal can tell apart. The"
        " threshold is an average, so the edge of the red region is approximate."
    ),
    "Reset (r)": "Returns every control to the values given on the command line.",
}


def to_photo(rgb: np.ndarray, width: int, height: int) -> tk.PhotoImage:
    """Fit an RGB image into width x height, keeping its aspect, as a Tk image; 0 keeps its size."""
    if width > 0 and height > 0:
        scale = min(width / rgb.shape[1], height / rgb.shape[0])
        size = (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale)))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    header = f"P6 {rgb.shape[1]} {rgb.shape[0]} 255 ".encode()
    return tk.PhotoImage(data=header + np.ascontiguousarray(rgb).tobytes(), format="ppm")


def warn_without_xft(root: tk.Tk) -> None:
    """Tk on X11 built without Xft draws text with unantialiased bitmap fonts."""
    try:
        font_system = root.tk.call("::tk::pkgconfig", "get", "fontsystem")
    except tk.TclError:
        return
    if root.tk.call("tk", "windowingsystem") == "x11" and font_system != "xft":
        print(
            "This Python's Tk has no Xft, so text is drawn without antialiasing;"
            " run it with a Python whose Tk has Xft, e.g. uv run --python /usr/bin/python3 dog-vision",
            file=sys.stderr,
        )


class Tooltip:
    """Text that appears beside the pointer while it rests on a widget, or on a tag's text in a Text widget."""

    def __init__(self, widget: tk.Misc, text, tag: str | None = None) -> None:
        self.widget = widget
        self.text = text  # called on showing, so it gives the current language
        self.window: tk.Toplevel | None = None
        self.pending: str | None = None
        if tag is None:
            widget.bind("<Enter>", self._schedule, add="+")
            for event in ("<Leave>", "<ButtonPress>", "<Destroy>"):
                widget.bind(event, self.hide, add="+")
        else:
            # A tag outlives its text, so whoever deletes the text hides the tooltip.
            widget.tag_bind(tag, "<Enter>", self._schedule)
            for event in ("<Leave>", "<ButtonPress>"):
                widget.tag_bind(tag, event, self.hide, add="+")

    def _schedule(self, _event: tk.Event) -> None:
        self.hide()
        self.pending = self.widget.after(TOOLTIP_DELAY_MS, self._show)

    def _show(self) -> None:
        self.pending = None
        self.window = tk.Toplevel(self.widget)
        # Hidden until placed: measuring its size maps it, at the screen's top left corner.
        self.window.withdraw()
        self.window.wm_overrideredirect(True)
        tk.Label(
            self.window,
            text=self.text(),
            justify="left",
            wraplength=TOOLTIP_WIDTH,
            background="#ffffe8",
            foreground="#202020",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=6,
        ).pack()
        self.window.update_idletasks()
        # Beside the pointer, on whichever side has room: the controls sit at the screen's right edge.
        width, height = self.window.winfo_reqwidth(), self.window.winfo_reqheight()
        x, y = self.widget.winfo_pointerx() + 16, self.widget.winfo_pointery() + 16
        if x + width > self.widget.winfo_screenwidth():
            x -= width + 32
        if y + height > self.widget.winfo_screenheight():
            y -= height + 32
        self.window.wm_geometry(f"+{max(0, x)}+{max(0, y)}")
        self.window.deiconify()

    def hide(self, _event: tk.Event | None = None) -> None:
        if self.pending is not None:
            self.widget.after_cancel(self.pending)
            self.pending = None
        if self.window is not None:
            self.window.destroy()
            self.window = None


class Section:
    """A titled part of the controls that a click on its title opens or closes, to keep the panel short."""

    def __init__(self, parent: tk.Misc, title_font: font.Font, is_open: bool, on_toggle=None) -> None:
        self.on_toggle = on_toggle  # called after a click opens or closes the section
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(0, weight=1)
        header = ttk.Frame(self.frame, cursor="hand2")
        header.grid(row=0, column=0, sticky="ew")
        # The arrow is drawn, not a character: fonts draw the triangles at unrelated sizes, some as emoji.
        self.size = round(title_font.metrics("linespace") * 0.6)
        style = ttk.Style(parent)
        self.colour = style.lookup("TLabel", "foreground") or "black"
        self.arrow = tk.Canvas(
            header,
            width=self.size,
            height=self.size,
            highlightthickness=0,
            background=style.lookup("TFrame", "background"),
        )
        self.arrow.pack(side="left", padx=(0, 6))
        self.title = ttk.Label(header, font=title_font)
        self.title.pack(side="left")
        for widget in (header, self.arrow, self.title):
            widget.bind("<Button-1>", self.toggle, add="+")
        self.body = ttk.Frame(self.frame, padding=(self.size + 6, 4, 0, 0))
        self.body.grid(row=1, column=0, sticky="ew")
        self.body.columnconfigure(0, weight=1)
        self.is_open = is_open
        self._show()

    def toggle(self, _event: tk.Event | None = None) -> None:
        self.is_open = not self.is_open
        self._show()
        if self.on_toggle is not None:
            self.on_toggle()

    def _show(self) -> None:
        s = self.size
        corners = (
            (0, s * 0.2, s, s * 0.2, s / 2, s * 0.85) if self.is_open else (s * 0.2, 0, s * 0.85, s / 2, s * 0.2, s)
        )
        self.arrow.delete("all")
        self.arrow.create_polygon(corners, fill=self.colour, outline=self.colour)
        if self.is_open:
            self.body.grid()
        else:
            self.body.grid_remove()


class LabelledSlider:
    """A slider with its label and current value on the line above it, so the three read as one."""

    def __init__(self, parent: tk.Misc, label: str, low: int, high: int, on_change) -> None:
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(0, weight=1)
        self.label = ttk.Label(self.frame, text=label)
        self.label.grid(row=0, column=0, sticky="w")
        self.value = ttk.Label(self.frame, width=4, anchor="e")
        self.value.grid(row=0, column=1, sticky="e")
        self.on_change = on_change
        self.scale = ttk.Scale(self.frame, from_=low, to=high, orient="horizontal", command=self._slid)
        self.scale.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))

    def _slid(self, position: str) -> None:
        number = round(float(position))
        self.value.configure(text=str(number))
        self.on_change(number)

    def set(self, number: int) -> None:
        self.scale.set(number)
        self.value.configure(text=str(number))


def run(session: LiveSession) -> None:
    root = tk.Tk()
    _ = session.translate
    # Every widget whose text is fixed, with that text in English, so a change of
    # language can translate them all again.
    translated: list[tuple[tk.Misc, str]] = []

    def text(widget: tk.Misc, english: str) -> tk.Misc:
        widget.configure(text=_(english))
        translated.append((widget, english))
        if english in DESCRIPTIONS:
            Tooltip(widget, lambda: _(DESCRIPTIONS[english]))
        return widget

    root.title(_("Dog vision"))
    warn_without_xft(root)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # The images with each one's caption under it, on one canvas: the dark area reaches down to the
    # status bar, level with the panel, and the captions stay under the images wherever they are drawn.
    image_area = tk.Canvas(root, background=IMAGE_BACKGROUND, borderwidth=0, highlightthickness=0)
    image_area.grid(row=0, column=0, sticky="nsew")
    photo_item = image_area.create_image(0, 0, anchor="n")
    caption_items: list[int] = []
    caption_line = font.nametofont("TkDefaultFont").metrics("linespace")

    def show_images(rgb: np.ndarray, texts: list[str], width: int, height: int) -> tk.PhotoImage:
        """Draw rgb as large as fits in width x height with texts under its images, left to right.

        0 draws rgb at its own size and asks for the room that takes. Returns the photo drawn.
        """
        while len(caption_items) < len(texts):
            caption_items.append(
                image_area.create_text(0, 0, anchor="n", justify="center", fill=CAPTION_COLOUR, font="TkDefaultFont")
            )
        for unused in caption_items[len(texts) :]:
            image_area.itemconfigure(unused, state="hidden")
        used = caption_items[: len(texts)]
        # The captions wrap at the images' width, which is fitted to the room the captions leave:
        # captions that wrap to more lines than there is room for take that room and fit the images again.
        captions_height = caption_line
        while True:
            photo = to_photo(rgb, width, max(1, height - captions_height - 2 * CAPTION_PAD) if height else 0)
            part = photo.width() / len(texts)
            for item, text in zip(used, texts):
                image_area.itemconfigure(item, text=text, width=max(1, round(part) - CAPTION_GAP), state="normal")
            needed = max(image_area.bbox(item)[3] - image_area.bbox(item)[1] for item in used)
            if needed <= captions_height:
                break
            captions_height = needed
        block = photo.height() + 2 * CAPTION_PAD + captions_height
        if not width:
            image_area.configure(width=photo.width(), height=block)
        left = ((width or photo.width()) - photo.width()) / 2
        top = round(((height or block) - block) / 2)
        image_area.itemconfigure(photo_item, image=photo)
        image_area.coords(photo_item, round(left + photo.width() / 2), top)
        for index, item in enumerate(used):
            image_area.coords(item, round(left + (index + 0.5) * part), top + photo.height() + CAPTION_PAD)
        return photo

    # A status bar across the bottom: what is shown, how a conversion stands, what the last action did.
    status_rule = ttk.Separator(root)
    status_rule.grid(row=1, column=0, columnspan=2, sticky="ew")
    status_bar = ttk.Frame(root, padding=(8, 3))
    status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")
    source_name = ttk.Label(status_bar)
    source_name.pack(side="left")
    conversion = ttk.Label(status_bar)
    conversion.pack(side="left", padx=(24, 0))
    recording_status = ttk.Label(status_bar)
    recording_status.pack(side="left", padx=(24, 0))
    status = ttk.Label(status_bar)
    status.pack(side="left", padx=(24, 0))

    # The controls lie on a canvas, the Tk widget that scrolls anything, for when the open sections do not fit.
    panel = ttk.Frame(root)
    panel.grid(row=0, column=1, sticky="nsew")
    panel.rowconfigure(0, weight=1)
    panel_canvas = tk.Canvas(
        panel, borderwidth=0, highlightthickness=0, background=ttk.Style(root).lookup("TFrame", "background")
    )
    panel_canvas.grid(row=0, column=0, sticky="nsew")
    panel_scrollbar = ttk.Scrollbar(panel, orient="vertical", command=panel_canvas.yview)
    panel_canvas.configure(yscrollcommand=panel_scrollbar.set)
    # The scrollbar's room stays while it is hidden, so that showing it does not narrow the images.
    panel.columnconfigure(1, minsize=panel_scrollbar.winfo_reqwidth())
    side = ttk.Frame(panel_canvas, padding=10)
    side_item = panel_canvas.create_window(0, 0, window=side, anchor="nw")
    side.columnconfigure(0, weight=1)
    side.rowconfigure(1, weight=1)
    section_font = font.nametofont("TkDefaultFont").copy()
    section_font.configure(weight="bold")

    text(ttk.Label(side), "Species").grid(row=0, column=0, sticky="w")
    list_frame = ttk.Frame(side)
    list_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 10))
    list_frame.rowconfigure(0, weight=1)
    species = tk.Listbox(list_frame, exportselection=False, activestyle="none", width=30)
    species.grid(row=0, column=0, sticky="ns")
    scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=species.yview)

    def on_list_scroll(first: str, last: str) -> None:
        """Show the scrollbar only while the list does not fit."""
        if float(first) <= 0 and float(last) >= 1:
            scrollbar.grid_remove()
        else:
            scrollbar.grid(row=0, column=1, sticky="ns")
        scrollbar.set(first, last)

    species.configure(yscrollcommand=on_list_scroll)

    def keep_selection_in_view(_event: tk.Event) -> None:
        """The list's height changes as sections open and close; keep the chosen species in it."""
        selection = species.curselection()
        if selection:
            species.see(selection[0])

    species.bind("<Configure>", keep_selection_in_view, add="+")

    def fit_panel() -> None:
        """Stretch the controls to the panel's height, or scroll them when the open sections need more.

        The species list is the only part that stretches, so without a minimum an opened
        section would take its room until one row is left; it keeps MIN_LIST_ROWS instead.
        """
        root.update_idletasks()
        # The list asks for its "height" in rows plus its border; a row's pitch is more than the font's line.
        rows = int(species.cget("height"))
        border = 2 * (int(species.cget("borderwidth")) + int(species.cget("highlightthickness")))
        row = (species.winfo_reqheight() - border) / rows
        needed = side.winfo_reqheight() - round((rows - MIN_LIST_ROWS) * row)
        shown = panel_canvas.winfo_height()
        height = max(needed, shown)
        # Asking for the whole height sizes the window the first frame opens; later the window keeps its size.
        panel_canvas.configure(height=side.winfo_reqheight(), scrollregion=(0, 0, 0, height))
        panel_canvas.itemconfigure(side_item, width=panel_canvas.winfo_width(), height=height)
        if height > shown:
            panel_scrollbar.grid(row=0, column=1, sticky="ns")
        else:
            panel_scrollbar.grid_remove()

    panel_canvas.bind("<Configure>", lambda _event: fit_panel(), add="+")

    def scroll_panel(event: tk.Event) -> None:
        """Scroll the controls with the wheel, except over the widgets the wheel already works on."""
        path = str(event.widget)
        if not panel_scrollbar.winfo_ismapped() or not path.startswith(str(panel) + "."):
            return
        # By path, not widget: a combobox's drop-down list is a Tk widget Python has no object for.
        if root.tk.call("winfo", "class", path) in ("Listbox", "TCombobox"):
            return
        panel_canvas.yview_scroll(-WHEEL_LINES if event.num == 4 or event.delta > 0 else WHEEL_LINES, "units")

    for wheel in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        root.bind_all(wheel, scroll_panel, add="+")
    species.insert("end", *session.species_labels)
    panel_canvas.configure(yscrollincrement=font.Font(font=species.cget("font")).metrics("linespace"))

    facts_section = Section(side, section_font, is_open=True, on_toggle=fit_panel)
    text(facts_section.title, "Selected species")
    facts_section.frame.grid(row=2, column=0, sticky="ew")
    facts_frame = facts_section.body
    fact_font = font.nametofont("TkDefaultFont")
    # Text, not labels, so that the facts can be selected and copied; a tab stop makes the values a column.
    facts = tk.Text(
        facts_frame,
        width=1,  # the column's minsize gives the width
        wrap="none",
        font=fact_font,
        borderwidth=0,
        highlightthickness=0,
        padx=0,
        pady=0,
        background=ttk.Style(root).lookup("TFrame", "background"),
        foreground=ttk.Style(root).lookup("TLabel", "foreground") or "black",
    )
    facts.grid(row=0, column=0, sticky="ew")
    facts.tag_configure("name", foreground="#555555")
    fact_tooltips: list[Tooltip] = []
    # A disabled Text is focused by a click only on Windows, and without focus Ctrl+C does not reach it.
    facts.bind("<Button-1>", lambda _event: facts.focus_set(), add="+")

    def select_all_facts() -> str:
        facts.tag_add("sel", "1.0", "end-1c")
        return "break"  # Tk's own Ctrl+A goes to the line's start

    facts.bind("<Control-a>", lambda _event: select_all_facts())
    fact_widths: dict[str, int] = {}  # by language

    def fact_width() -> int:
        """The widest piece of any species' facts: lines that wide never have to break a piece."""
        if session.language not in fact_widths:
            fact_widths[session.language] = max(
                fact_font.measure(piece)
                for name in session.species_names
                for _label, value, _description in session.species_facts(name)
                for piece in value
            )
        return fact_widths[session.language]

    def fact_text(value: tuple[str, ...]) -> str:
        """A fact's pieces of information, as many to a line as fit in fact_width().

        The pieces of a source in brackets are kept together: a source that does not fit after
        the value starts a line of its own, and breaks between its pieces only where it is wider
        than a line.
        """
        groups: list[list[str]] = []
        inside = False
        for piece in value:
            if inside:
                groups[-1].append(piece)
            else:
                groups.append([piece])
            inside = (inside or piece.startswith("(")) and not piece.endswith(")")
        lines: list[str] = []
        for group in groups:
            whole = " ".join(group)
            if lines and fact_font.measure(f"{lines[-1]} {whole}") <= fact_width():
                lines[-1] = f"{lines[-1]} {whole}"
            elif fact_font.measure(whole) <= fact_width():
                lines.append(whole)
            else:
                lines.append(group[0])
                for piece in group[1:]:
                    joined = f"{lines[-1]} {piece}"
                    if fact_font.measure(joined) <= fact_width():
                        lines[-1] = joined
                    else:
                        lines.append(piece)
        return "\n".join(lines)

    def show_facts() -> None:
        for tooltip in fact_tooltips:
            tooltip.hide()
        fact_tooltips.clear()
        facts.configure(state="normal")
        facts.delete("1.0", "end")
        for row, (label, value, description) in enumerate(session.species_facts()):
            tag = f"name{row}"
            facts.insert("end", "\n" if row else "")
            facts.insert("end", label, ("name", tag))
            # A fact's further lines start at the tab stop too, under its first.
            facts.insert("end", "\t" + fact_text(value).replace("\n", "\n\t"))
            fact_tooltips.append(Tooltip(facts, lambda description=description: description, tag))
        lines = int(facts.index("end-1c").split(".")[0])
        facts.configure(state="disabled", height=lines)
        fit_panel()  # species have more or fewer facts

    simulation = Section(side, section_font, is_open=True, on_toggle=fit_panel)
    text(simulation.title, "Simulation")
    simulation.frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
    sliders: dict[str, LabelledSlider] = {}
    for row, (field, label) in enumerate(PERCENT_SLIDERS.items()):
        slider = LabelledSlider(
            simulation.body, label, 0, 100, lambda percent, field=field: setattr(session.params, field, percent / 100)
        )
        text(slider.label, label)
        slider.frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        sliders[field] = slider

    text(ttk.Label(simulation.body), "Colour saturation").grid(row=2, column=0, sticky="w")
    chroma = ttk.Frame(simulation.body, padding=(8, 2, 0, 0))
    chroma.grid(row=3, column=0, sticky="ew")
    chroma_scale = tk.StringVar(value=session.params.chroma_scale)
    chroma_labels = {"fixed": "Fixed by the projection", "rnl": "Matched to discrimination (RNL)"}

    def on_chroma_scale() -> None:
        session.params.chroma_scale = chroma_scale.get()

    for value in session.chroma_scales:
        text(
            ttk.Radiobutton(chroma, value=value, variable=chroma_scale, command=on_chroma_scale), chroma_labels[value]
        ).pack(anchor="w")

    # A section the command line left at its defaults starts closed.
    acuity_section = Section(side, section_font, is_open=session.params.acuity, on_toggle=fit_panel)
    text(acuity_section.title, "Acuity")
    acuity_section.frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))
    acuity_frame = acuity_section.body
    acuity = tk.BooleanVar(value=session.params.acuity)
    text(
        ttk.Checkbutton(acuity_frame, variable=acuity, command=lambda: setattr(session.params, "acuity", acuity.get())),
        "Blur to the species' acuity",
    ).grid(row=0, column=0, sticky="w")
    field_of_view = LabelledSlider(
        acuity_frame, "", 10, 120, lambda degrees: setattr(session.params, "field_of_view", float(degrees))
    )
    text(field_of_view.label, "Image spans [degrees]")
    field_of_view.frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))

    view_section = Section(
        side, section_font, is_open=session.compare is not None or session.difference, on_toggle=fit_panel
    )
    text(view_section.title, "View")
    view_section.frame.grid(row=5, column=0, sticky="ew", pady=(10, 0))
    view = view_section.body
    view.columnconfigure(0, weight=0)
    view.columnconfigure(1, weight=1)
    side_by_side = tk.BooleanVar(value=session.side_by_side)
    side_by_side_box = text(
        ttk.Checkbutton(
            view, variable=side_by_side, command=lambda: setattr(session, "side_by_side", side_by_side.get())
        ),
        "Side by side (m)",
    )
    side_by_side_box.grid(row=0, column=0, columnspan=2, sticky="w")
    text(ttk.Label(view), "Left image").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
    left = ttk.Combobox(view, values=[_("original"), *session.species_labels], state="readonly", width=24)
    left.grid(row=1, column=1, sticky="ew", pady=(4, 0))

    left_index = tk.IntVar()  # the View menu's choice of left image, as the combobox counts

    def show_compare() -> None:
        index = 0 if session.compare is None else 1 + session.species_names.index(session.compare)
        left.current(index)
        left_index.set(index)

    def set_compare(index: int) -> None:
        session.compare = None if index == 0 else session.species_names[index - 1]
        show_compare()

    left.bind("<<ComboboxSelected>>", lambda _event: set_compare(left.current()))
    difference = tk.BooleanVar(value=session.difference)
    difference_box = text(
        ttk.Checkbutton(view, variable=difference, command=lambda: setattr(session, "difference", difference.get())),
        "Map of differences (d)",
    )
    difference_box.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def toggle_difference() -> None:
        if session.recording:  # a recording keeps its size
            return
        difference.set(not difference.get())
        session.difference = difference.get()

    chosen_species = tk.StringVar()  # the Species menu's choice

    def show_species() -> None:
        """Make the list, the Species menu and the facts reflect the chosen species."""
        index = session.species_names.index(session.params.species)
        species.selection_clear(0, "end")
        species.selection_set(index)
        species.activate(index)
        species.see(index)
        chosen_species.set(session.params.species)
        show_facts()

    def choose_species(name: str) -> None:
        session.params.species = name
        show_species()

    def show_params() -> None:
        """Make the widgets reflect session.params."""
        for field, slider in sliders.items():
            slider.set(round(getattr(session.params, field) * 100))
        acuity.set(session.params.acuity)
        show_compare()
        field_of_view.set(round(session.params.field_of_view))
        chroma_scale.set(session.params.chroma_scale)
        show_species()

    def reset() -> None:
        session.reset()
        show_params()

    def save() -> None:
        try:
            name = session.save_snapshot()
        except OSError as error:
            status.configure(text=_("Cannot save snapshot: {error}").format(error=error))
            return
        status.configure(text=_("Saved {name}").format(name=name) if name else _("No frame yet"))

    def choose_output_dir() -> None:
        path = filedialog.askdirectory(
            parent=root, title=_("Folder for snapshots and recordings"), initialdir=str(session.output_dir)
        )
        if path:
            error = session.set_output_dir(Path(path))
            status.configure(
                text=error or _("Snapshots and recordings go to {folder}").format(folder=session.output_dir)
            )

    convert_to_output_dir = tk.BooleanVar(value=session.settings.convert_to_output_dir)  # the File menu's check

    def set_convert_to_output_dir() -> None:
        status.configure(text=session.set_convert_to_output_dir(convert_to_output_dir.get()) or "")

    recording = tk.BooleanVar(value=False)  # the File menu's check

    def toggle_recording() -> None:
        if session.recording:
            session.stop_recording()
        else:
            session.start_recording()
        recording.set(session.recording)

    def toggle_side_by_side() -> None:
        if session.recording:
            return
        side_by_side.set(not side_by_side.get())
        session.side_by_side = side_by_side.get()

    def open_file() -> None:
        if session.recording:
            return
        patterns = " ".join(f"*.{extension} *.{extension.upper()}" for extension in FILE_EXTENSIONS)
        path = filedialog.askopenfilename(
            parent=root,
            title=_("Open a photo or a video"),
            filetypes=[(_("Photos and videos"), patterns), (_("All files"), "*")],
        )
        if path and not session.open_file(Path(path)):
            status.configure(text=_("Cannot open {name} as a photo or a video").format(name=Path(path).name))

    def open_camera() -> None:
        if not session.open_camera():
            status.configure(text=_("Cannot open camera {index}").format(index=session.camera_index))

    text(ttk.Button(side, command=reset), "Reset (r)").grid(row=6, column=0, sticky="w", pady=(12, 0))
    sections = (facts_section, simulation, acuity_section, view_section)

    def fix_panel_width() -> None:
        """Keep the panel as wide as it is with every section open, for any species.

        Grid gives the panel the width its open sections ask for and the images the rest, so
        without this the images would narrow and widen as sections open and close.
        """
        # The names and the values of the facts, each as wide as any species needs them.
        rows = [row for name in session.species_names for row in session.species_facts(name)]
        names = max(fact_font.measure(label) for label, _v, _d in rows) + FACT_GAP
        values = max(fact_font.measure(line) for _l, value, _d in rows for line in fact_text(value).split("\n"))
        facts.configure(tabs=(names,))
        facts_frame.columnconfigure(0, minsize=names + values)
        closed = [section for section in sections if not section.is_open]
        for section in closed:
            section.body.grid()
        root.update_idletasks()
        panel_canvas.configure(width=side.winfo_reqwidth())
        for section in closed:
            section.body.grid_remove()
        fit_panel()

    panel_shown = tk.BooleanVar(value=True)

    def show_panel() -> None:
        """Hide or show the whole panel; the images take its room."""
        if panel_shown.get():
            panel.grid()
        else:
            panel.grid_remove()

    def toggle_panel() -> None:
        panel_shown.set(not panel_shown.get())
        show_panel()

    # Menu entries are not widgets, so they are translated again by their place in the menu.
    translated_entries: list[tuple[tk.Menu, int, str]] = []

    def entry(parent: tk.Menu, kind: str, english: str, **options) -> int:
        parent.add(kind, label=_(english), **options)
        index = parent.index("end")
        translated_entries.append((parent, index, english))
        return index

    menu_bar = tk.Menu(root)
    root.configure(menu=menu_bar)
    file_menu = tk.Menu(menu_bar, tearoff=False)
    entry(menu_bar, "cascade", "File", menu=file_menu)
    open_entry = entry(file_menu, "command", "Open file…", command=open_file, accelerator="o")
    camera = entry(file_menu, "command", "Camera", command=open_camera)
    convert = entry(file_menu, "command", "Convert file", command=session.convert_source)
    file_menu.add_separator()
    entry(file_menu, "command", "Save snapshot", command=save, accelerator="s")
    entry(file_menu, "checkbutton", "Record video", variable=recording, command=toggle_recording, accelerator="v")
    file_menu.add_separator()
    entry(file_menu, "command", "Output folder…", command=choose_output_dir)
    entry(
        file_menu,
        "checkbutton",
        "Converted files into the output folder",
        variable=convert_to_output_dir,
        command=set_convert_to_output_dir,
    )
    file_menu.add_separator()
    entry(file_menu, "command", "Quit", command=root.destroy, accelerator="q")

    def enable(menu: tk.Menu, index: int, enabled: bool) -> None:
        """Set an entry's state only when it changes: this runs on every frame, and an open menu redraws."""
        state = "normal" if enabled else "disabled"
        if menu.entrycget(index, "state") != state:
            menu.entryconfigure(index, state=state)

    def show_source() -> None:
        """Make the menus, the controls and the status bar reflect the session.

        Called on every frame, as a conversion or a recording runs on. While recording, the
        controls that would change the video's size are disabled.
        """
        source_name.configure(text=session.source_name())
        free = not session.recording
        enable(file_menu, open_entry, free)
        enable(file_menu, camera, session.source is not None and free)
        enable(file_menu, convert, session.source is not None and not session.converting)
        enable(view_menu, side_by_side_entry, free)
        enable(view_menu, difference_entry, free)
        for box in (side_by_side_box, difference_box):
            if box.instate(["disabled"]) == free:
                box.state(["!disabled" if free else "disabled"])
        conversion.configure(text=session.conversion_status() or "")
        recording_status.configure(text=session.recording_status() or "")

    # The species and the View section's controls are in the menus too, for when the side panel is hidden;
    # they share the panel's variables, so either shows what the other set.
    species_menu = tk.Menu(menu_bar, tearoff=False)
    entry(menu_bar, "cascade", "Species", menu=species_menu)
    view_menu = tk.Menu(menu_bar, tearoff=False)
    entry(menu_bar, "cascade", "View", menu=view_menu)
    entry(view_menu, "checkbutton", "Side panel", variable=panel_shown, command=show_panel, accelerator="F9")
    view_menu.add_separator()
    side_by_side_entry = entry(
        view_menu,
        "checkbutton",
        "Side by side",
        variable=side_by_side,
        command=lambda: setattr(session, "side_by_side", side_by_side.get()),
        accelerator="m",
    )
    left_menu = tk.Menu(view_menu, tearoff=False)
    entry(view_menu, "cascade", "Left image", menu=left_menu)
    difference_entry = entry(
        view_menu,
        "checkbutton",
        "Map of differences",
        variable=difference,
        command=lambda: setattr(session, "difference", difference.get()),
        accelerator="d",
    )
    view_menu.add_separator()
    chroma_menu = tk.Menu(view_menu, tearoff=False)
    entry(view_menu, "cascade", "Colour saturation", menu=chroma_menu)
    for value in session.chroma_scales:
        entry(
            chroma_menu,
            "radiobutton",
            chroma_labels[value],
            value=value,
            variable=chroma_scale,
            command=on_chroma_scale,
        )
    entry(
        view_menu,
        "checkbutton",
        "Blur to the species' acuity",
        variable=acuity,
        command=lambda: setattr(session.params, "acuity", acuity.get()),
    )

    def fill_species_menus() -> None:
        """List the species in the menus, in the current language."""
        species_menu.delete(0, "end")
        for name, label in zip(session.species_names, session.species_labels):
            species_menu.add_radiobutton(
                label=label, value=name, variable=chosen_species, command=lambda: choose_species(chosen_species.get())
            )
        left_menu.delete(0, "end")
        for index, label in enumerate([_("original"), *session.species_labels]):
            left_menu.add_radiobutton(
                label=label, value=index, variable=left_index, command=lambda: set_compare(left_index.get())
            )

    fill_species_menus()

    # What a right click on the species' facts offers.
    facts_menu = tk.Menu(facts, tearoff=False)
    copy_facts = entry(
        facts_menu, "command", "Copy", command=lambda: facts.event_generate("<<Copy>>"), accelerator="Ctrl+C"
    )
    entry(facts_menu, "command", "Select all", command=select_all_facts, accelerator="Ctrl+A")

    def post_facts_menu(event: tk.Event) -> None:
        facts.focus_set()  # the copy goes to the focused widget
        enable(facts_menu, copy_facts, bool(facts.tag_ranges("sel")))
        facts_menu.tk_popup(event.x_root, event.y_root)

    facts.bind("<Button-2>" if root.tk.call("tk", "windowingsystem") == "aqua" else "<Button-3>", post_facts_menu)

    language = tk.StringVar(value=session.language)

    def on_language() -> None:
        session.language = language.get()
        root.title(_("Dog vision"))
        for widget, english in translated:
            widget.configure(text=_(english))
        for menu, index, english in translated_entries:
            menu.entryconfigure(index, label=_(english))
        species.delete(0, "end")
        species.insert("end", *session.species_labels)
        left.configure(values=[_("original"), *session.species_labels])
        fill_species_menus()
        status.configure(text="")
        show_params()  # puts back the selections the new items dropped, and the facts
        fix_panel_width()  # the texts have new widths

    language_menu = tk.Menu(menu_bar, tearoff=False)
    entry(menu_bar, "cascade", "Language", menu=language_menu)
    for code, name in session.languages.items():
        language_menu.add_radiobutton(label=name, value=code, variable=language, command=on_language)

    def on_select(_event: tk.Event) -> None:
        selection = species.curselection()
        if selection:
            choose_species(session.species_names[selection[0]])

    species.bind("<<ListboxSelect>>", on_select)
    root.bind("<KeyPress-m>", lambda _: toggle_side_by_side())
    root.bind("<KeyPress-d>", lambda _: toggle_difference())
    root.bind("<KeyPress-r>", lambda _: reset())
    root.bind("<KeyPress-s>", lambda _: save())
    root.bind("<KeyPress-v>", lambda _: toggle_recording())
    root.bind("<KeyPress-o>", lambda _: open_file())
    root.bind("<F9>", lambda _: toggle_panel())
    root.bind("<KeyPress-q>", lambda _: root.destroy())
    root.bind("<Escape>", lambda _: root.destroy())

    sized = False
    shown: tuple | None = None  # the image, canvas size and captions last drawn, not to draw a still photo again

    def tick() -> None:
        nonlocal sized, shown
        if session.error:
            root.destroy()
            return
        show_source()
        rgb = session.render()
        size = (image_area.winfo_width(), image_area.winfo_height())
        # The captions count too: a change of language changes them, not the image.
        texts = session.captions()
        if rgb is not None and (shown is None or shown[0] is not rgb or shown[1:] != (size, texts)):
            shown = (rgb, size, texts)
            # The first frame is shown at its own size: the empty canvas is not laid out yet.
            photo = show_images(rgb, texts, *size) if sized else show_images(rgb, texts, 0, 0)
            image_area.photo = photo  # Tk drops images that Python no longer references
            if not sized:
                # The first frame sets the window to its natural size, but no larger than the
                # screen holds; fixing that geometry stops each new image from resizing the window
                # again, and the next frame is fitted to the images' share of it.
                root.update_idletasks()
                largest_width, largest_height = root.wm_maxsize()
                width = min(root.winfo_reqwidth(), largest_width)
                height = min(root.winfo_reqheight(), largest_height)
                root.geometry(f"{width}x{height}")
                sized = True
        root.after(FRAME_INTERVAL_MS, tick)

    show_params()
    fix_panel_width()
    species.focus_set()
    tick()
    root.mainloop()
