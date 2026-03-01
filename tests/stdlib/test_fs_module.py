"""Tests for stdlib FileSystemModule."""

import os
import pytest
from src.zexus.stdlib.fs import FileSystemModule, PathTraversalError


@pytest.fixture
def fs():
    mod = FileSystemModule()
    # Disable strict mode for tests so tmp_path is allowed
    FileSystemModule.configure_security(strict=False)
    yield mod
    # Re-enable strict mode
    FileSystemModule.configure_security(strict=True)


# ── read / write ─────────────────────────────────────────────────────────
class TestReadWrite:
    def test_write_and_read(self, fs, tmp_path):
        fp = str(tmp_path / "hello.txt")
        fs.write_file(fp, "hello world")
        assert fs.read_file(fp) == "hello world"

    def test_append(self, fs, tmp_path):
        fp = str(tmp_path / "append.txt")
        fs.write_file(fp, "a")
        fs.append_file(fp, "b")
        assert fs.read_file(fp) == "ab"

    def test_read_nonexistent(self, fs):
        with pytest.raises(Exception):
            fs.read_file("/nonexistent/file.txt")

    def test_write_creates_parents(self, fs, tmp_path):
        fp = str(tmp_path / "deep" / "nested" / "file.txt")
        fs.write_file(fp, "ok")
        assert fs.read_file(fp) == "ok"


# ── binary ───────────────────────────────────────────────────────────────
class TestBinary:
    def test_write_and_read_binary(self, fs, tmp_path):
        fp = str(tmp_path / "bin.dat")
        fs.write_binary(fp, b"\x00\x01\x02")
        assert fs.read_binary(fp) == b"\x00\x01\x02"


# ── existence checks ────────────────────────────────────────────────────
class TestExistence:
    def test_exists(self, fs, tmp_path):
        fp = str(tmp_path / "e.txt")
        assert fs.exists(fp) is False
        fs.write_file(fp, "")
        assert fs.exists(fp) is True

    def test_is_file(self, fs, tmp_path):
        fp = str(tmp_path / "f.txt")
        fs.write_file(fp, "")
        assert fs.is_file(fp) is True
        assert fs.is_file(str(tmp_path)) is False

    def test_is_dir(self, fs, tmp_path):
        assert fs.is_dir(str(tmp_path)) is True


# ── directory ops ────────────────────────────────────────────────────────
class TestDirectoryOps:
    def test_mkdir(self, fs, tmp_path):
        d = str(tmp_path / "newdir")
        fs.mkdir(d)
        assert os.path.isdir(d)

    def test_rmdir(self, fs, tmp_path):
        d = str(tmp_path / "toremove")
        os.makedirs(d)
        fs.rmdir(d)
        assert not os.path.exists(d)

    def test_list_dir(self, fs, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        result = fs.list_dir(str(tmp_path))
        assert sorted(result) == ["a.txt", "b.txt"]


# ── file ops ─────────────────────────────────────────────────────────────
class TestFileOps:
    def test_remove(self, fs, tmp_path):
        fp = str(tmp_path / "del.txt")
        fs.write_file(fp, "bye")
        fs.remove(fp)
        assert not os.path.exists(fp)

    def test_rename(self, fs, tmp_path):
        old = str(tmp_path / "old.txt")
        new = str(tmp_path / "new.txt")
        fs.write_file(old, "data")
        fs.rename(old, new)
        assert not os.path.exists(old)
        assert fs.read_file(new) == "data"

    def test_copy_file(self, fs, tmp_path):
        src = str(tmp_path / "src.txt")
        dst = str(tmp_path / "dst.txt")
        fs.write_file(src, "copy me")
        fs.copy_file(src, dst)
        assert fs.read_file(dst) == "copy me"

    def test_get_size(self, fs, tmp_path):
        fp = str(tmp_path / "size.txt")
        fs.write_file(fp, "12345")
        assert fs.get_size(fp) == 5


# ── path utilities ───────────────────────────────────────────────────────
class TestPathUtils:
    def test_abs_path(self, fs):
        result = fs.abs_path(".")
        assert os.path.isabs(result)

    def test_join(self, fs):
        result = fs.join("a", "b", "c.txt")
        assert result == os.path.join("a", "b", "c.txt")

    def test_basename(self, fs):
        assert fs.basename("/foo/bar.txt") == "bar.txt"

    def test_dirname(self, fs):
        assert fs.dirname("/foo/bar.txt") == "/foo"

    def test_splitext(self, fs):
        result = fs.splitext("file.tar.gz")
        assert result[1] == ".gz"

    def test_get_stat(self, fs, tmp_path):
        fp = str(tmp_path / "stat.txt")
        fs.write_file(fp, "hi")
        stat = fs.get_stat(fp)
        assert "size" in stat
        assert stat["size"] == 2


# ── glob ─────────────────────────────────────────────────────────────────
class TestGlob:
    def test_glob(self, fs, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("y")
        (tmp_path / "c.txt").write_text("z")
        result = fs.glob(str(tmp_path / "*.py"))
        assert len(result) == 2


# ── walk ─────────────────────────────────────────────────────────────────
class TestWalk:
    def test_walk(self, fs, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "f.txt").write_text("ok")
        result = fs.walk(str(tmp_path))
        assert isinstance(result, list)
        assert len(result) >= 1


# ── security ─────────────────────────────────────────────────────────────
class TestSecurity:
    def test_path_traversal_error_class(self):
        err = PathTraversalError("bad path")
        assert str(err) == "bad path"
