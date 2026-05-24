"""Unit tests for the data_version cache-key helper."""
import os

from src.data import data_version


def test_missing_directory_returns_empty(tmp_path):
    assert data_version(tmp_path / "does_not_exist") == ()


def test_stable_when_unchanged(tmp_path):
    (tmp_path / "a.csv").write_text("x")
    assert data_version(tmp_path) == data_version(tmp_path)


def test_changes_when_file_added(tmp_path):
    (tmp_path / "a.csv").write_text("x")
    before = data_version(tmp_path)
    (tmp_path / "b.csv").write_text("y")
    assert data_version(tmp_path) != before


def test_changes_when_file_removed(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text("x")
    (tmp_path / "b.csv").write_text("y")
    before = data_version(tmp_path)
    a.unlink()
    assert data_version(tmp_path) != before


def test_changes_when_mtime_changes(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text("x")
    before = data_version(tmp_path)
    os.utime(a, (1_000_000_000, 1_000_000_000))  # fixed, deterministically different mtime
    assert data_version(tmp_path) != before


def test_recurses_into_subdirectories(tmp_path):
    sub = tmp_path / "Brazil"
    sub.mkdir()
    bond = sub / "bond.csv"
    bond.write_text("x")
    before = data_version(tmp_path)
    os.utime(bond, (1_000_000_000, 1_000_000_000))
    assert data_version(tmp_path) != before

def test_accepts_string_path(tmp_path):
    (tmp_path / "a.csv").write_text("x")
    assert data_version(str(tmp_path)) == data_version(tmp_path)


def test_existing_empty_directory_returns_empty(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert data_version(empty) == ()
