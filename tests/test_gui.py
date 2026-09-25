"""The Tk window, driven through its keys and menus on a virtual display.

run() builds the window and enters mainloop(); here mainloop() is replaced by a scenario
that pumps the window's events, acts on it and checks the session. Xvfb keeps the
windows off the screen of whoever runs the tests; without it the tests use $DISPLAY.
"""

import os
import shutil
import subprocess
import time
import tkinter as tk

import cv2
import numpy as np
import pytest

from dog_vision.core.model import Params
from dog_vision.core.session import LiveSession
from dog_vision.gui import tk as tk_gui


@pytest.fixture(scope="module")
def display():
    if shutil.which("Xvfb") is None:
        if not os.environ.get("DISPLAY"):
            pytest.skip("needs Xvfb or a display")
        yield
        return
    server = subprocess.Popen(
        ["Xvfb", "-displayfd", "1", "-screen", "0", "1600x1000x24", "-nolisten", "tcp"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    number = server.stdout.readline().strip()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DISPLAY", f":{number}")
        yield
    server.terminate()
    server.wait()


@pytest.fixture
def session(tmp_path, monkeypatch, display):
    photo = tmp_path / "photo.png"
    cv2.imwrite(str(photo), np.random.default_rng(0).integers(0, 256, (120, 160, 3), np.uint8))
    monkeypatch.chdir(tmp_path)
    live = LiveSession(0, Params(), path=photo)
    yield live
    live.close()


def show(session: LiveSession, monkeypatch, scenario) -> None:
    """Run the window with scenario(root) in place of its main loop; the window is closed after it.

    An error in one of the window's own callbacks, which Tk would only print, fails the test too.
    """
    errors = []

    def mainloop(root, _n=0):
        try:
            pump(root)
            scenario(root)
        finally:
            if root_exists(root):
                root.destroy()

    monkeypatch.setattr(tk.Tk, "mainloop", mainloop)
    monkeypatch.setattr(tk.Tk, "report_callback_exception", lambda root, kind, error, trace: errors.append(error))
    tk_gui.run(session)
    assert errors == []


def root_exists(root: tk.Tk) -> bool:
    try:
        return bool(root.winfo_exists())
    except tk.TclError:
        return False


def pump(root: tk.Tk, seconds: float = 0.3) -> None:
    """Let the window handle its events and a few frames."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and root_exists(root):
        root.update()
        time.sleep(0.01)


def press(root: tk.Tk, key: str) -> None:
    """Press key in the window; Tk delivers a key only to the widget with the focus."""
    root.focus_force()
    root.update()
    (root.focus_get() or root).event_generate(f"<KeyPress-{key}>")
    pump(root, 0.1)


def entry(root: tk.Tk, *labels: str) -> tuple[tk.Menu, int]:
    """The menu holding the entry that labels lead to, from the menu bar, and its index."""
    menu = root.nametowidget(root["menu"])
    for depth, label in enumerate(labels):
        index = next(
            i
            for i in range(menu.index("end") + 1)
            if menu.type(i) not in ("separator", "tearoff") and menu.entrycget(i, "label") == label
        )
        if depth == len(labels) - 1:
            return menu, index
        menu = root.nametowidget(menu.entrycget(index, "menu"))
    raise AssertionError(labels)


def choose(root: tk.Tk, *labels: str) -> None:
    menu, index = entry(root, *labels)
    menu.invoke(index)
    pump(root, 0.1)


def canvas_items(root: tk.Tk, kind: str) -> tuple[list[int], tk.Canvas]:
    """The image canvas's visible items of a kind, and the canvas."""
    canvas = next(child for child in root.winfo_children() if isinstance(child, tk.Canvas))
    return [
        item for item in canvas.find_all() if canvas.type(item) == kind and canvas.itemcget(item, "state") != "hidden"
    ], canvas


def test_the_window_shows_the_view_with_its_captions(session, monkeypatch):
    def scenario(root):
        assert root.title() == "Dog vision"
        (image,), canvas = canvas_items(root, "image")
        photo = canvas.itemcget(image, "image")
        assert int(root.tk.call("image", "width", photo)) > 0
        texts, canvas = canvas_items(root, "text")
        assert [canvas.itemcget(item, "text") for item in texts] == ["original", "dog (dichromat)"]

    show(session, monkeypatch, scenario)


def test_the_keys_change_the_view(session, monkeypatch):
    def scenario(root):
        press(root, "m")
        assert not session.side_by_side
        press(root, "m")
        press(root, "d")
        assert session.side_by_side and session.difference
        pump(root)
        texts, _canvas = canvas_items(root, "text")
        assert len(texts) == 3

    show(session, monkeypatch, scenario)


def test_a_recording_keeps_the_size_of_the_view(session, monkeypatch):
    def scenario(root):
        press(root, "v")
        assert session.recording
        press(root, "m")
        press(root, "d")
        assert session.side_by_side and not session.difference
        menu, index = entry(root, "View", "Side by side")
        assert menu.entrycget(index, "state") == "disabled"
        press(root, "v")
        assert not session.recording
        pump(root)
        assert menu.entrycget(index, "state") == "normal"

    show(session, monkeypatch, scenario)
    assert session.recording_status().startswith("Wrote dog-dog-")


def test_record_video_in_the_file_menu(session, monkeypatch):
    def scenario(root):
        choose(root, "File", "Record video")
        assert session.recording
        choose(root, "File", "Record video")
        assert not session.recording

    show(session, monkeypatch, scenario)


def test_s_saves_a_snapshot(session, monkeypatch, tmp_path):
    show(session, monkeypatch, lambda root: press(root, "s"))
    assert len(list(tmp_path.glob("dog-dog-*.png"))) == 1


def test_r_resets_to_the_command_line_values(session, monkeypatch):
    def scenario(root):
        choose(root, "Species", "cat (dichromat)")
        assert session.params.species == "cat"
        press(root, "r")
        assert session.params.species == "dog"

    show(session, monkeypatch, scenario)


def test_the_left_image_can_be_another_species(session, monkeypatch):
    def scenario(root):
        choose(root, "View", "Left image", "cat (dichromat)")
        assert session.compare == "cat"
        choose(root, "View", "Left image", "original")
        assert session.compare is None

    show(session, monkeypatch, scenario)


def test_the_view_menu_sets_saturation_and_acuity(session, monkeypatch):
    def scenario(root):
        choose(root, "View", "Colour saturation", "Matched to discrimination (RNL)")
        assert session.params.chroma_scale == "rnl"
        choose(root, "View", "Blur to the species' acuity")
        assert session.params.acuity

    show(session, monkeypatch, scenario)


def test_f9_hides_and_shows_the_panel(session, monkeypatch):
    def scenario(root):
        panel = next(child for child in root.winfo_children() if child.grid_info().get("column") == 1)
        press(root, "F9")
        assert not panel.winfo_ismapped()
        press(root, "F9")
        assert panel.winfo_ismapped()

    show(session, monkeypatch, scenario)


def test_the_window_speaks_the_language_chosen(session, monkeypatch):
    def scenario(root):
        choose(root, "Language", "Čeština")
        assert session.language == "cs"
        assert root.title() == "Psí vidění"
        entry(root, "Soubor", "Nahrávat video")
        entry(root, "Druh", "pes (dichromat)")
        pump(root)
        texts, canvas = canvas_items(root, "text")
        assert [canvas.itemcget(item, "text") for item in texts] == ["originál", "pes (dichromat)"]

    show(session, monkeypatch, scenario)


def test_convert_file_in_the_file_menu(session, monkeypatch, tmp_path):
    def scenario(root):
        choose(root, "File", "Convert file")
        deadline = time.monotonic() + 10
        while session.converting and time.monotonic() < deadline:
            pump(root, 0.05)
        assert session.conversion_status() == "Wrote photo.dog.png"

    show(session, monkeypatch, scenario)
    assert (tmp_path / "photo.dog.png").exists()


def test_q_closes_the_window(session, monkeypatch):
    closed = []

    def scenario(root):
        press(root, "q")
        closed.append(not root_exists(root))

    show(session, monkeypatch, scenario)
    assert closed == [True]


def test_the_window_closes_when_the_camera_stops(session, monkeypatch):
    closed = []

    def scenario(root):
        session.error = "Camera stopped delivering frames"
        pump(root, 1)
        closed.append(not root_exists(root))

    show(session, monkeypatch, scenario)
    assert closed == [True]
