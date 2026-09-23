"""Tkinter window for dog_vision's live camera view.

This module only lays out widgets and forwards events to a LiveSession; the
session does the capturing, simulating and saving. A window in another toolkit
is a module with the same run(session) function.
"""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dog_vision import LiveSession

FRAME_INTERVAL_MS = 15
PERCENT_SLIDERS = {"adaptation": "Adaptation to scene [%]", "strength": "Simulation strength [%]"}


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


class LabelledSlider:
    """A slider with its label and current value on the line above it, so the three read as one."""

    def __init__(self, parent: tk.Misc, label: str, low: int, high: int, on_change) -> None:
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(0, weight=1)
        ttk.Label(self.frame, text=label).grid(row=0, column=0, sticky="w")
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
    root.title("Dog vision")
    warn_without_xft(root)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    image = tk.Label(root, background="#282828", borderwidth=0, highlightthickness=0)
    image.grid(row=0, column=0, sticky="nsew")
    caption = ttk.Label(root, anchor="center", padding=(0, 4))
    caption.grid(row=1, column=0, sticky="ew")

    side = ttk.Frame(root, padding=10)
    side.grid(row=0, column=1, rowspan=2, sticky="ns")
    side.rowconfigure(1, weight=1)

    ttk.Label(side, text="Species").grid(row=0, column=0, sticky="w")
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

    facts_frame = ttk.LabelFrame(side, text="Selected species", padding=(8, 4))
    facts_frame.grid(row=2, column=0, sticky="ew")
    facts_frame.columnconfigure(1, weight=1)

    def show_facts() -> None:
        for child in facts_frame.winfo_children():
            child.destroy()
        for row, (label, value) in enumerate(session.species_facts()):
            ttk.Label(facts_frame, text=label, foreground="#555555").grid(row=row, column=0, sticky="nw", padx=(0, 8))
            ttk.Label(facts_frame, text=value, wraplength=190).grid(row=row, column=1, sticky="w")

    sliders: dict[str, LabelledSlider] = {}
    for row, (field, label) in enumerate(PERCENT_SLIDERS.items(), start=3):
        slider = LabelledSlider(side, label, 0, 100, lambda percent, field=field: setattr(session.params, field, percent / 100))
        slider.frame.grid(row=row, column=0, sticky="ew", pady=(8, 0))
        sliders[field] = slider

    chroma = ttk.LabelFrame(side, text="Colour saturation", padding=(8, 4))
    chroma.grid(row=5, column=0, sticky="ew", pady=(10, 0))
    chroma_scale = tk.StringVar(value=session.params.chroma_scale)
    chroma_labels = {"fixed": "Fixed by the projection", "rnl": "Matched to discrimination (RNL)"}

    def on_chroma_scale() -> None:
        session.params.chroma_scale = chroma_scale.get()

    for value in session.chroma_scales:
        ttk.Radiobutton(
            chroma, text=chroma_labels[value], value=value, variable=chroma_scale, command=on_chroma_scale
        ).pack(anchor="w")

    acuity_frame = ttk.LabelFrame(side, text="Acuity", padding=(8, 4))
    acuity_frame.grid(row=6, column=0, sticky="ew", pady=(10, 0))
    acuity_frame.columnconfigure(0, weight=1)
    acuity = tk.BooleanVar(value=session.params.acuity)
    ttk.Checkbutton(
        acuity_frame,
        text="Blur to the species' acuity",
        variable=acuity,
        command=lambda: setattr(session.params, "acuity", acuity.get()),
    ).grid(row=0, column=0, sticky="w")
    field_of_view = LabelledSlider(
        acuity_frame, "Image spans [degrees]", 10, 120, lambda degrees: setattr(session.params, "field_of_view", float(degrees))
    )
    field_of_view.frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))

    view = ttk.LabelFrame(side, text="View", padding=(8, 4))
    view.grid(row=7, column=0, sticky="ew", pady=(10, 0))
    view.columnconfigure(1, weight=1)
    side_by_side = tk.BooleanVar(value=session.side_by_side)
    ttk.Checkbutton(
        view,
        text="Side by side (m)",
        variable=side_by_side,
        command=lambda: setattr(session, "side_by_side", side_by_side.get()),
    ).grid(row=0, column=0, columnspan=2, sticky="w")
    ttk.Label(view, text="Left image").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
    left_choices = ["original", *session.species_labels]
    left = ttk.Combobox(view, values=left_choices, state="readonly", width=24)
    left.grid(row=1, column=1, sticky="ew", pady=(4, 0))

    def on_left(_event: tk.Event) -> None:
        index = left.current()
        session.compare = None if index == 0 else session.species_names[index - 1]

    left.bind("<<ComboboxSelected>>", on_left)
    difference = tk.BooleanVar(value=session.difference)
    ttk.Checkbutton(
        view,
        text="Map of differences (d)",
        variable=difference,
        command=lambda: setattr(session, "difference", difference.get()),
    ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def toggle_difference() -> None:
        difference.set(not difference.get())
        session.difference = difference.get()

    status = ttk.Label(side, text="", width=28)

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
        status.configure(text=f"Saved {name}" if name else "No frame yet")

    def toggle_side_by_side() -> None:
        side_by_side.set(not side_by_side.get())
        session.side_by_side = side_by_side.get()

    buttons = ttk.Frame(side)
    buttons.grid(row=8, column=0, sticky="ew", pady=(10, 0))
    ttk.Button(buttons, text="Reset (r)", command=reset).pack(side="left")
    ttk.Button(buttons, text="Save snapshot (s)", command=save).pack(side="left", padx=(6, 0))
    status.grid(row=9, column=0, sticky="w", pady=(6, 0))

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
    root.bind("<KeyPress-q>", lambda _: root.destroy())
    root.bind("<Escape>", lambda _: root.destroy())

    sized = False

    def tick() -> None:
        nonlocal sized
        if session.error:
            root.destroy()
            return
        rgb = session.render()
        if rgb is not None:
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
