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


def run(session: LiveSession) -> None:
    root = tk.Tk()
    root.title("Dog vision")
    warn_without_xft(root)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    image = tk.Label(root, background="#282828", borderwidth=0, highlightthickness=0)
    image.grid(row=0, column=0, sticky="nsew")

    side = ttk.Frame(root, padding=10)
    side.grid(row=0, column=1, sticky="ns")
    side.rowconfigure(1, weight=1)

    ttk.Label(side, text="Species").grid(row=0, column=0, sticky="w")
    list_frame = ttk.Frame(side)
    list_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 10))
    list_frame.rowconfigure(0, weight=1)
    species = tk.Listbox(list_frame, exportselection=False, activestyle="none", width=20)
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
    species.insert("end", *session.species_names)

    sliders: dict[str, tk.Scale] = {}
    for row, (field, label) in enumerate(PERCENT_SLIDERS.items(), start=2):
        slider = tk.Scale(side, label=label, from_=0, to=100, orient="horizontal", length=200)
        slider.configure(command=lambda value, field=field: setattr(session.params, field, int(value) / 100))
        slider.grid(row=row, column=0, sticky="ew")
        sliders[field] = slider

    side_by_side = tk.BooleanVar(value=session.side_by_side)
    ttk.Checkbutton(
        side,
        text="Side by side (m)",
        variable=side_by_side,
        command=lambda: setattr(session, "side_by_side", side_by_side.get()),
    ).grid(row=4, column=0, sticky="w", pady=(10, 0))

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
    buttons.grid(row=5, column=0, sticky="ew", pady=(10, 0))
    ttk.Button(buttons, text="Reset (r)", command=reset).pack(side="left")
    ttk.Button(buttons, text="Save snapshot (s)", command=save).pack(side="left", padx=(6, 0))
    status.grid(row=6, column=0, sticky="w", pady=(6, 0))

    def on_select(_event: tk.Event) -> None:
        selection = species.curselection()
        if selection:
            session.params.species = session.species_names[selection[0]]

    species.bind("<<ListboxSelect>>", on_select)
    root.bind("<KeyPress-m>", lambda _: toggle_side_by_side())
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
