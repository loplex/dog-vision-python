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
from tkinter import ttk
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from dog_vision.core import settings as settings_store
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


def status_text(root: tk.Tk) -> str:
    """The status bar's last label: what the last action did."""
    bar = next(child for child in root.winfo_children() if child.grid_info().get("row") == 2)
    return bar.pack_slaves()[-1].cget("text")


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


def test_the_output_folder_is_chosen_in_the_file_menu(session, monkeypatch, tmp_path, settings_file):
    monkeypatch.setattr(tk_gui.filedialog, "askdirectory", lambda **options: str(tmp_path / "chosen"))

    def scenario(root):
        choose(root, "File", "Output folder…")
        assert session.output_dir == tmp_path / "chosen"
        assert status_text(root) == f"Snapshots and recordings go to {tmp_path / 'chosen'}"
        press(root, "s")

    show(session, monkeypatch, scenario)
    assert settings_store.load(settings_file).output_dir == tmp_path / "chosen"
    assert len(list((tmp_path / "chosen").glob("dog-dog-*.png"))) == 1


def test_cancelling_the_folder_dialog_keeps_the_folder(session, monkeypatch, tmp_path):
    monkeypatch.setattr(tk_gui.filedialog, "askdirectory", lambda **options: "")
    show(session, monkeypatch, lambda root: choose(root, "File", "Output folder…"))
    assert session.output_dir == tmp_path


def test_converted_files_into_the_output_folder_is_a_check_in_the_file_menu(session, monkeypatch, settings_file):
    def scenario(root):
        choose(root, "File", "Converted files into the output folder")
        assert session.settings.convert_to_output_dir

    show(session, monkeypatch, scenario)
    assert settings_store.load(settings_file).convert_to_output_dir


def test_a_snapshot_that_cannot_be_saved_says_why(session, monkeypatch, tmp_path):
    (tmp_path / "a-file").write_text("")
    session.set_output_dir(tmp_path / "a-file")

    def scenario(root):
        press(root, "s")
        assert status_text(root).startswith("Cannot save snapshot: Cannot write ")

    show(session, monkeypatch, scenario)


def descendants(widget: tk.Misc):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def find(root: tk.Tk, kind: type, text: str | None = None) -> tk.Misc:
    """The first widget of a kind, with the text given if any."""
    return next(w for w in descendants(root) if isinstance(w, kind) and (text is None or w.cget("text") == text))


def tooltip_windows(root: tk.Tk) -> list[tk.Toplevel]:
    return [widget for widget in descendants(root) if isinstance(widget, tk.Toplevel)]


def tooltips(root: tk.Tk) -> list[str]:
    return [find(window, tk.Label).cget("text") for window in tooltip_windows(root)]


def test_a_tooltip_shows_what_a_control_means(session, monkeypatch):
    def scenario(root):
        button = find(root, ttk.Button, "Reset (r)")
        button.event_generate("<Enter>")
        pump(root, (tk_gui.TOOLTIP_DELAY_MS + 200) / 1000)
        assert tooltips(root) == [tk_gui.DESCRIPTIONS["Reset (r)"]]
        button.event_generate("<Leave>")
        pump(root, 0.05)
        assert tooltips(root) == []

    show(session, monkeypatch, scenario)


def test_a_tooltip_left_before_its_delay_never_shows(session, monkeypatch):
    def scenario(root):
        button = find(root, ttk.Button, "Reset (r)")
        button.event_generate("<Enter>")
        pump(root, 0.05)
        button.event_generate("<Leave>")
        pump(root, (tk_gui.TOOLTIP_DELAY_MS + 200) / 1000)
        assert tooltips(root) == []

    show(session, monkeypatch, scenario)


def test_a_tooltip_at_the_screens_corner_opens_towards_the_middle(session, monkeypatch):
    def scenario(root):
        button = find(root, ttk.Button, "Reset (r)")
        corner = (root.winfo_screenwidth() - 2, root.winfo_screenheight() - 2)
        root.event_generate("<Motion>", warp=True, x=corner[0] - root.winfo_rootx(), y=corner[1] - root.winfo_rooty())
        pump(root, 0.05)
        button.event_generate("<Enter>")
        pump(root, (tk_gui.TOOLTIP_DELAY_MS + 200) / 1000)
        (window,) = tooltip_windows(root)
        assert window.winfo_rootx() + window.winfo_width() <= corner[0]
        assert window.winfo_rooty() + window.winfo_height() <= corner[1]

    show(session, monkeypatch, scenario)


def test_a_click_on_a_sections_title_closes_and_opens_it(session, monkeypatch):
    def scenario(root):
        title = find(root, ttk.Label, "Simulation")
        slider_label = find(root, ttk.Label, "Simulation strength [%]")
        assert slider_label.winfo_ismapped()
        title.event_generate("<Button-1>")
        pump(root, 0.1)
        assert not slider_label.winfo_ismapped()
        title.event_generate("<Button-1>")
        pump(root, 0.1)
        assert slider_label.winfo_ismapped()

    show(session, monkeypatch, scenario)


def test_a_small_window_scrolls_the_list_and_the_controls(session, monkeypatch):
    def scenario(root):
        root.geometry("640x320")
        pump(root, 0.5)
        species = find(root, tk.Listbox)
        list_bar = next(w for w in species.master.winfo_children() if isinstance(w, ttk.Scrollbar))
        assert list_bar.winfo_ismapped()
        panel = next(child for child in root.winfo_children() if child.grid_info().get("column") == 1)
        canvas = next(w for w in panel.winfo_children() if isinstance(w, tk.Canvas))
        assert canvas.yview()[0] == 0
        species.event_generate("<Button-5>")  # the list scrolls itself, not the panel
        pump(root, 0.1)
        assert canvas.yview()[0] == 0
        find(root, ttk.Label, "Simulation").event_generate("<Button-5>")
        pump(root, 0.1)
        assert canvas.yview()[0] > 0
        find(root, ttk.Label, "Simulation").event_generate("<Button-4>")
        pump(root, 0.1)
        assert canvas.yview()[0] == 0
        canvas_items(root, "image")[1].event_generate("<Button-5>")  # over the images, not the panel
        pump(root, 0.1)
        assert canvas.yview()[0] == 0
        for title in ("Selected species", "Simulation"):  # closed, they leave the list room
            find(root, ttk.Label, title).event_generate("<Button-1>")
        root.geometry("1500x950")
        pump(root, 0.5)
        assert not list_bar.winfo_ismapped()  # the list fits again

    show(session, monkeypatch, scenario)


def test_the_facts_can_be_selected_and_copied(session, monkeypatch):
    posted = []
    monkeypatch.setattr(tk.Menu, "tk_popup", lambda menu, x, y, entry="": posted.append(menu))

    def scenario(root):
        facts = find(root, tk.Text)
        facts.event_generate("<Button-3>", x=5, y=5)
        (menu,) = posted
        copy = next(i for i in range(menu.index("end") + 1) if menu.entrycget(i, "label") == "Copy")
        assert menu.entrycget(copy, "state") == "disabled"  # nothing selected yet
        facts.focus_force()
        facts.event_generate("<Control-a>")
        assert facts.get("sel.first", "sel.last") == facts.get("1.0", "end-1c")
        facts.tag_remove("sel", "1.0", "end")
        menu.invoke(next(i for i in range(menu.index("end") + 1) if menu.entrycget(i, "label") == "Select all"))
        facts.event_generate("<Button-3>", x=5, y=5)
        assert menu.entrycget(copy, "state") == "normal"
        root.clipboard_clear()
        menu.invoke(copy)
        assert root.clipboard_get().startswith("Colour vision\tdichromat, 2 cone types")

    show(session, monkeypatch, scenario)


def test_a_click_in_the_species_list_chooses_the_species(session, monkeypatch):
    def scenario(root):
        species = find(root, tk.Listbox)
        species.selection_clear(0, "end")
        species.selection_set(session.species_names.index("horse"))
        species.event_generate("<<ListboxSelect>>")
        pump(root, 0.1)
        assert session.params.species == "horse"

    show(session, monkeypatch, scenario)


def test_o_opens_a_file(session, monkeypatch, tmp_path):
    clip = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    monkeypatch.setattr(tk_gui.filedialog, "askopenfilename", lambda **options: str(clip))

    def scenario(root):
        press(root, "o")
        assert session.source == clip and session.source_is_video

    show(session, monkeypatch, scenario)


def test_a_file_that_is_no_photo_or_video_is_reported(session, monkeypatch, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("nothing")
    monkeypatch.setattr(tk_gui.filedialog, "askopenfilename", lambda **options: str(notes))

    def scenario(root):
        choose(root, "File", "Open file…")
        assert status_text(root) == "Cannot open notes.txt as a photo or a video"
        assert session.source_name() == "photo.png"

    show(session, monkeypatch, scenario)


def test_no_file_is_opened_while_recording(session, monkeypatch):
    asked = []
    monkeypatch.setattr(tk_gui.filedialog, "askopenfilename", lambda **options: asked.append(options) or "")

    def scenario(root):
        press(root, "v")
        press(root, "o")
        press(root, "v")

    show(session, monkeypatch, scenario)
    assert asked == []


def test_a_camera_that_cannot_be_opened_is_reported(session, monkeypatch):
    monkeypatch.setattr(session, "open_camera", lambda: False)

    def scenario(root):
        choose(root, "File", "Camera")
        assert status_text(root) == "Cannot open camera 0"

    show(session, monkeypatch, scenario)


def fake_root(windowing_system: str, font_system: str | None):
    def call(*arguments):
        if arguments[0] == "::tk::pkgconfig":
            if font_system is None:
                raise tk.TclError("no pkgconfig")
            return font_system
        return windowing_system

    return SimpleNamespace(tk=SimpleNamespace(call=call))


@pytest.mark.parametrize(
    ("windowing_system", "font_system", "warns"),
    [("x11", "x11", True), ("x11", "xft", False), ("win32", "win32", False), ("x11", None, False)],
)
def test_a_tk_without_xft_is_warned_about(capsys, windowing_system, font_system, warns):
    tk_gui.warn_without_xft(fake_root(windowing_system, font_system))
    assert ("without antialiasing" in capsys.readouterr().err) == warns
