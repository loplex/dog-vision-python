import json
from pathlib import Path

import pytest

from dog_vision.core import settings as settings_store
from dog_vision.core.settings import Settings, load, save

# The function itself, as the autouse settings_file fixture replaces the module's.
config_path = settings_store.config_path


def test_without_a_file_the_settings_are_the_defaults(settings_file):
    assert not settings_file.exists()
    assert load() == Settings(output_dir=None, convert_to_output_dir=False)


def test_saved_settings_are_loaded_again(settings_file):
    save(Settings(Path("/data/dogs"), convert_to_output_dir=True))
    assert json.loads(settings_file.read_text()) == {"output_dir": "/data/dogs", "convert_to_output_dir": True}
    assert load() == Settings(Path("/data/dogs"), convert_to_output_dir=True)


def test_no_folder_is_saved_as_null(settings_file):
    save(Settings())
    assert json.loads(settings_file.read_text())["output_dir"] is None
    assert load() == Settings()


@pytest.mark.parametrize(
    "content",
    ["not json", "[1, 2]", '"a string"', '{"output_dir": 3, "convert_to_output_dir": "yes"}', '{"output_dir": ""}'],
)
def test_a_damaged_file_gives_the_defaults(settings_file, content):
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text(content)
    assert load() == Settings()


def test_an_unknown_setting_is_ignored(settings_file):
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text('{"output_dir": "/x", "colour": "blue"}')
    assert load() == Settings(Path("/x"))


def test_a_path_can_be_given(tmp_path):
    path = tmp_path / "elsewhere" / "s.json"
    save(Settings(convert_to_output_dir=True), path)
    assert load(path) == Settings(convert_to_output_dir=True)


@pytest.mark.parametrize(
    ("platform", "variables", "expected"),
    [
        ("linux", {"XDG_CONFIG_HOME": "/xdg"}, "/xdg/dog-vision/settings.json"),
        ("linux", {"HOME": "/home/u"}, "/home/u/.config/dog-vision/settings.json"),
        ("win32", {"APPDATA": "C:/Users/u/AppData/Roaming"}, "C:/Users/u/AppData/Roaming/dog-vision/settings.json"),
        ("darwin", {"HOME": "/Users/u"}, "/Users/u/Library/Application Support/dog-vision/settings.json"),
    ],
)
def test_the_file_is_where_each_system_keeps_configuration(monkeypatch, platform, variables, expected):
    for variable in ("XDG_CONFIG_HOME", "APPDATA"):
        monkeypatch.delenv(variable, raising=False)
    for variable, value in variables.items():
        monkeypatch.setenv(variable, value)
    monkeypatch.setattr(settings_store.sys, "platform", platform)
    assert config_path() == Path(expected)
