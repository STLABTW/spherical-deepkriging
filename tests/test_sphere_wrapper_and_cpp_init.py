import importlib
import importlib.util
import sys
import types
import warnings  # pragma: no cover
from pathlib import Path

import numpy as np
import pytest
import setuptools
from setuptools import Distribution


class FakeConverter:
    def __add__(self, other):
        return ("combined", other)


def test_named_values_and_cpp_path(monkeypatch):
    import spherical_deepkriging.basis_functions.mrts_sphere.sphere as sphere

    captured = {}

    def fake_cpp(knot, k, X=None):
        captured["knot_dtype"] = knot.dtype
        captured["X_dtype"] = None if X is None else X.dtype
        return {"mrts": np.ones((2, k))}

    monkeypatch.setattr(sphere, "CPP_EXTENSIONS_AVAILABLE", True)
    monkeypatch.setattr(sphere, "_mrts_sphere_cpp", fake_cpp)

    result = sphere.mrts_sphere(
        np.array([[1, 2], [3, 4]], dtype=np.float32),
        k=2,
        X=np.array([[5, 6], [7, 8]], dtype=np.float32),
    )

    assert result.names() == ["mrts"]
    assert "mrts" in result
    assert list(result.keys()) == ["mrts"]
    assert list(result)[0].shape == (2, 2)
    assert captured["knot_dtype"] == np.float64
    assert captured["X_dtype"] == np.float64


def test_sphere_r_fallback_and_import_error(monkeypatch):
    import spherical_deepkriging.basis_functions.mrts_sphere.sphere as sphere

    monkeypatch.setattr(sphere, "CPP_EXTENSIONS_AVAILABLE", False)
    monkeypatch.setattr(sphere, "_RPY2_AVAILABLE", True)
    calls = {}

    def fake_r(knot, k, x):
        calls["args"] = (knot, k, x)
        return {"mrts": knot}

    monkeypatch.setattr(sphere, "_mrts_sphere_r", fake_r)

    knot = np.array([[1.0, 2.0], [3.0, 4.0]])
    sphere.mrts_sphere(knot, k=2)
    assert np.array_equal(calls["args"][2], knot)

    monkeypatch.setattr(sphere, "_RPY2_AVAILABLE", False)
    monkeypatch.setattr(sphere, "_mrts_sphere_r", None)
    with pytest.raises(
        ImportError,
        match=r"Neither C\+\+ extensions nor rpy2/R implementation are available",
    ):
        sphere.mrts_sphere(knot, k=2)


def test_sphere_import_time_rpy2_initialization(monkeypatch):
    fake_ro = types.ModuleType("rpy2.robjects")
    commands = []
    fake_ro.globalenv = {"mrts_sphere": lambda knot, k, x: {"mrts": x}}
    fake_ro.r = lambda cmd: commands.append(cmd)
    fake_ro.default_converter = FakeConverter()
    fake_numpy2ri = types.SimpleNamespace(converter="numpy-converter")
    fake_ro.numpy2ri = fake_numpy2ri

    fake_rpy2 = types.ModuleType("rpy2")
    monkeypatch.setitem(sys.modules, "rpy2", fake_rpy2)
    monkeypatch.setitem(sys.modules, "rpy2.robjects", fake_ro)
    monkeypatch.setattr("subprocess.check_output", lambda args: b"/fake/R/home\n")

    fake_cpp = types.ModuleType(
        "spherical_deepkriging.basis_functions.mrts_sphere.sphere_cpp"
    )
    fake_cpp.CPP_EXTENSIONS_AVAILABLE = False
    fake_cpp.mrts_sphere = lambda **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "spherical_deepkriging.basis_functions.mrts_sphere.sphere_cpp",
        fake_cpp,
    )
    sys.modules.pop("spherical_deepkriging.basis_functions.mrts_sphere.sphere", None)

    module = importlib.import_module(
        "spherical_deepkriging.basis_functions.mrts_sphere.sphere"
    )
    module = importlib.reload(module)

    assert module._RPY2_AVAILABLE is True
    assert module.numpy2ri_converter == ("combined", "numpy-converter")
    assert module._mrts_sphere_r is fake_ro.globalenv["mrts_sphere"]
    assert any("source(" in command for command in commands)


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
