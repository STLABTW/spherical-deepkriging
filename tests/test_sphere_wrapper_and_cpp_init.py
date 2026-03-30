import builtins
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import setuptools
from setuptools import Distribution


def test_cpp_extensions_init_with_and_without_binary(monkeypatch):
    module_name = "spherical_deepkriging.basis_functions.mrts_sphere.cpp_extensions"
    sys.modules.pop(module_name, None)
    fake_binary = types.ModuleType(f"{module_name}.spherical_basis")
    for name in [
        "cpp_exp",
        "cpp_fk",
        "cpp_K",
        "cpp_Kf",
        "cpp_Kmatrix",
        "getEigen",
        "getEigenTopK",
    ]:
        setattr(fake_binary, name, object())
    monkeypatch.setitem(sys.modules, f"{module_name}.spherical_basis", fake_binary)

    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    assert "cpp_K" in module.__all__

    sys.modules.pop(module_name, None)
    sys.modules.pop(f"{module_name}.spherical_basis", None)
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("spherical_basis"):
            raise ImportError("forced missing binary")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    assert module.__all__ == []
    assert not hasattr(module, "cpp_K")


def test_cpp_setup_executes_and_build_class_issues_cmake_commands(
    monkeypatch, tmp_path
):
    setup_path = (
        Path(__file__).resolve().parents[1]
        / "spherical_deepkriging"
        / "basis_functions"
        / "mrts_sphere"
        / "cpp_extensions"
        / "setup.py"
    )
    captured = {}
    monkeypatch.setattr(
        setuptools, "setup", lambda **kwargs: captured.setdefault("kwargs", kwargs)
    )
    monkeypatch.setattr(setuptools, "find_packages", lambda: ["pkg"])

    spec = importlib.util.spec_from_file_location("tmp_cpp_setup", setup_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert captured["kwargs"]["name"] == "spherical_basis"
    assert captured["kwargs"]["include_package_data"] is True

    commands = []
    monkeypatch.setattr(
        module.os,
        "makedirs",
        lambda path, exist_ok: commands.append(("mkdir", path, exist_ok)),
    )
    monkeypatch.setattr(module.os, "system", lambda cmd: commands.append(cmd) or 0)

    dist = Distribution()
    cmd = module.CMakeBuild(dist)
    cmd.build_temp = str(tmp_path / "build")
    cmd.debug = False
    cmd.get_ext_fullpath = lambda name: str(tmp_path / f"{name}.so")
    ext = types.SimpleNamespace(name="demo")
    cmd.build_extension(ext)

    assert any("cmake -S . -B" in entry for entry in commands if isinstance(entry, str))
    assert any("cmake --build" in entry for entry in commands if isinstance(entry, str))
