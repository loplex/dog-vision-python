"""Tkinter window for dog_vision's live camera view.

This module only lays out widgets and forwards events to a LiveSession; the
session does the capturing, simulating and saving. A window in another toolkit
is a module with the same run(session) function.
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dog_vision import LiveSession

FRAME_INTERVAL_MS = 15
TOOLTIP_DELAY_MS = 500
TOOLTIP_WIDTH = 380  # pixels a tooltip's text wraps at
# What the open dialog lists; Tk matches patterns case-sensitively on some systems, so both cases.
FILE_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp", "mp4", "mov", "m4v", "avi", "mkv", "webm"]
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
            " run it with a Python whose Tk has Xft, e.g. uv run --python /usr/bin/python3 dog_vision.py",
            file=sys.stderr,
        )


class Tooltip:
    """Text that appears beside the pointer while it rests on a widget."""

    def __init__(self, widget: tk.Misc, text) -> None:
        self.widget = widget
        self.text = text  # called on showing, so it gives the current language
        self.window: tk.Toplevel | None = None
        self.pending: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        for event in ("<Leave>", "<ButtonPress>", "<Destroy>"):
            widget.bind(event, self._hide, add="+")

    def _schedule(self, _event: tk.Event) -> None:
        self._hide()
        self.pending = self.widget.after(TOOLTIP_DELAY_MS, self._show)

    def _show(self) -> None:
        self.pending = None
        self.window = tk.Toplevel(self.widget)
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

    def _hide(self, _event: tk.Event | None = None) -> None:
        if self.pending is not None:
            self.widget.after_cancel(self.pending)
            self.pending = None
        if self.window is not None:
            self.window.destroy()
            self.window = None


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

    image = tk.Label(root, background="#282828", borderwidth=0, highlightthickness=0)
    image.grid(row=0, column=0, sticky="nsew")
    caption = ttk.Label(root, anchor="center", padding=(0, 4))
    caption.grid(row=1, column=0, sticky="ew")
    # A status bar across the bottom: what is shown, how a conversion stands, what the last action did.
    ttk.Separator(root).grid(row=2, column=0, columnspan=2, sticky="ew")
    status_bar = ttk.Frame(root, padding=(8, 3))
    status_bar.grid(row=3, column=0, columnspan=2, sticky="ew")
    source_name = ttk.Label(status_bar)
    source_name.pack(side="left")
    conversion = ttk.Label(status_bar)
    conversion.pack(side="left", padx=(24, 0))
    status = ttk.Label(status_bar)
    status.pack(side="left", padx=(24, 0))

    side = ttk.Frame(root, padding=10)
    side.grid(row=0, column=1, rowspan=2, sticky="ns")
    side.rowconfigure(1, weight=1)

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
    species.insert("end", *session.species_labels)

    facts_frame = text(ttk.LabelFrame(side, padding=(8, 4)), "Selected species")
    facts_frame.grid(row=2, column=0, sticky="ew")
    facts_frame.columnconfigure(1, weight=1)

    def show_facts() -> None:
        for child in facts_frame.winfo_children():
            child.destroy()
        for row, (label, value, description) in enumerate(session.species_facts()):
            name = ttk.Label(facts_frame, text=label, foreground="#555555")
            name.grid(row=row, column=0, sticky="nw", padx=(0, 8))
            Tooltip(name, lambda description=description: description)
            ttk.Label(facts_frame, text=value, wraplength=190).grid(row=row, column=1, sticky="w")

    sliders: dict[str, LabelledSlider] = {}
    for row, (field, label) in enumerate(PERCENT_SLIDERS.items(), start=3):
        slider = LabelledSlider(side, label, 0, 100, lambda percent, field=field: setattr(session.params, field, percent / 100))
        text(slider.label, label)
        slider.frame.grid(row=row, column=0, sticky="ew", pady=(8, 0))
        sliders[field] = slider

    chroma = text(ttk.LabelFrame(side, padding=(8, 4)), "Colour saturation")
    chroma.grid(row=5, column=0, sticky="ew", pady=(10, 0))
    chroma_scale = tk.StringVar(value=session.params.chroma_scale)
    chroma_labels = {"fixed": "Fixed by the projection", "rnl": "Matched to discrimination (RNL)"}

    def on_chroma_scale() -> None:
        session.params.chroma_scale = chroma_scale.get()

    for value in session.chroma_scales:
        text(
            ttk.Radiobutton(chroma, value=value, variable=chroma_scale, command=on_chroma_scale), chroma_labels[value]
        ).pack(anchor="w")

    acuity_frame = text(ttk.LabelFrame(side, padding=(8, 4)), "Acuity")
    acuity_frame.grid(row=6, column=0, sticky="ew", pady=(10, 0))
    acuity_frame.columnconfigure(0, weight=1)
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

    view = text(ttk.LabelFrame(side, padding=(8, 4)), "View")
    view.grid(row=7, column=0, sticky="ew", pady=(10, 0))
    view.columnconfigure(1, weight=1)
    side_by_side = tk.BooleanVar(value=session.side_by_side)
    text(
        ttk.Checkbutton(view, variable=side_by_side, command=lambda: setattr(session, "side_by_side", side_by_side.get())),
        "Side by side (m)",
    ).grid(row=0, column=0, columnspan=2, sticky="w")
    text(ttk.Label(view), "Left image").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
    left = ttk.Combobox(view, values=[_("original"), *session.species_labels], state="readonly", width=24)
    left.grid(row=1, column=1, sticky="ew", pady=(4, 0))

    def on_left(_event: tk.Event) -> None:
        index = left.current()
        session.compare = None if index == 0 else session.species_names[index - 1]

    left.bind("<<ComboboxSelected>>", on_left)
    difference = tk.BooleanVar(value=session.difference)
    text(
        ttk.Checkbutton(view, variable=difference, command=lambda: setattr(session, "difference", difference.get())),
        "Map of differences (d)",
    ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def toggle_difference() -> None:
        difference.set(not difference.get())
        session.difference = difference.get()

    def show_params() -> None:
        """Make the widgets reflect session.params."""
        index = session.species_names.index(session.params.species)
        species.selection_clear(0, "end")
        species.selection_set(index)
        species.activate(index)
        species.see(index)
        for field, slider in sliders.items():
            slider.set(round(getattr(session.params, field) * 100))
        acuity.set(session.params.acuity)
        left.current(0 if session.compare is None else 1 + session.species_names.index(session.compare))
        field_of_view.set(round(session.params.field_of_view))
        chroma_scale.set(session.params.chroma_scale)
        show_facts()

    def reset() -> None:
        session.reset()
        show_params()

    def save() -> None:
        name = session.save_snapshot()
        status.configure(text=_("Saved {name}").format(name=name) if name else _("No frame yet"))

    def toggle_side_by_side() -> None:
        side_by_side.set(not side_by_side.get())
        session.side_by_side = side_by_side.get()

    def open_file() -> None:
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

    text(ttk.Button(side, command=reset), "Reset (r)").grid(row=8, column=0, sticky="w", pady=(10, 0))

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
    entry(file_menu, "command", "Open file…", command=open_file, accelerator="o")
    camera = entry(file_menu, "command", "Camera", command=open_camera)
    convert = entry(file_menu, "command", "Convert file", command=session.convert_source)
    file_menu.add_separator()
    entry(file_menu, "command", "Save snapshot", command=save, accelerator="s")
    file_menu.add_separator()
    entry(file_menu, "command", "Quit", command=root.destroy, accelerator="q")

    def enable(menu: tk.Menu, index: int, enabled: bool) -> None:
        """Set an entry's state only when it changes: this runs on every frame, and an open menu redraws."""
        state = "normal" if enabled else "disabled"
        if menu.entrycget(index, "state") != state:
            menu.entryconfigure(index, state=state)

    def show_source() -> None:
        """Make the menu and the status bar reflect the session; called on every frame, as a conversion runs on."""
        source_name.configure(text=session.source_name())
        enable(file_menu, camera, session.source is not None)
        enable(file_menu, convert, session.source is not None and not session.converting)
        conversion.configure(text=session.conversion_status() or "")

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
        status.configure(text="")
        show_params()  # puts back the selections the new items dropped, and the facts

    language_menu = tk.Menu(menu_bar, tearoff=False)
    entry(menu_bar, "cascade", "Language", menu=language_menu)
    for code, name in session.languages.items():
        language_menu.add_radiobutton(label=name, value=code, variable=language, command=on_language)

    def on_select(_event: tk.Event) -> None:
        selection = species.curselection()
        if selection:
            session.params.species = session.species_names[selection[0]]
            show_facts()

    species.bind("<<ListboxSelect>>", on_select)
    root.bind("<KeyPress-m>", lambda _: toggle_side_by_side())
    root.bind("<KeyPress-d>", lambda _: toggle_difference())
    root.bind("<KeyPress-r>", lambda _: reset())
    root.bind("<KeyPress-s>", lambda _: save())
    root.bind("<KeyPress-o>", lambda _: open_file())
    root.bind("<KeyPress-q>", lambda _: root.destroy())
    root.bind("<Escape>", lambda _: root.destroy())

    sized = False
    shown: tuple | None = None  # the image and label size last drawn, not to draw a still photo again

    def tick() -> None:
        nonlocal sized, shown
        if session.error:
            root.destroy()
            return
        show_source()
        rgb = session.render()
        size = (image.winfo_width(), image.winfo_height())
        if rgb is not None and (shown is None or shown[0] is not rgb or shown[1] != size):
            shown = (rgb, size)
            # The first frame is shown at its own size: the empty label is not laid out yet.
            photo = to_photo(rgb, image.winfo_width(), image.winfo_height()) if sized else to_photo(rgb, 0, 0)
            image.configure(image=photo)
            caption.configure(text=session.caption())
            image.photo = photo  # Tk drops images that Python no longer references
            if not sized:
                # The first frame sets the window to its natural size; fixing that geometry
                # stops each new image from resizing the window again.
                root.update_idletasks()
                root.geometry(root.geometry())
                sized = True
        root.after(FRAME_INTERVAL_MS, tick)

    show_params()
    species.focus_set()
    tick()
    root.mainloop()
