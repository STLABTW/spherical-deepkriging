import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parent


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str):
        super().__init__(name=name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: Extension) -> None:
        if not isinstance(ext, CMakeExtension):
            return super().build_extension(ext)

        extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        build_temp = Path(self.build_temp) / ext.name.split(".")[-1]
        build_temp.mkdir(parents=True, exist_ok=True)

        cfg = "Release"
        python_exe = sys.executable
        jobs = max(1, (os.cpu_count() or 2) - 1)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPython3_EXECUTABLE={python_exe}",
        ]

        subprocess.check_call(
            ["cmake", "-S", ext.sourcedir, "-B", str(build_temp), *cmake_args]
        )

        # cmake --build passes `--config` only on multi-config generators (Windows)
        if os.name == "nt":
            subprocess.check_call(
                ["cmake", "--build", str(build_temp), "--config", cfg]
            )
        else:
            subprocess.check_call(
                [
                    "cmake",
                    "--build",
                    str(build_temp),
                    "--config",
                    cfg,
                    "--",
                    f"-j{jobs}",
                ]
            )


def _load_project_metadata() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return data.get("project", {})  # type: ignore[no-any-return]


project = _load_project_metadata()

name = project.get("name", "spherical-deepkriging")
version = project.get("version", "0.0.0")
dependencies = project.get("dependencies", [])
python_requires = project.get("requires-python", ">=3.10")

cpp_ext_spherical_dir = (
    ROOT
    / "spherical_deepkriging"
    / "basis_functions"
    / "mrts_sphere"
    / "cpp_extensions"
)

setup(
    name=name,
    version=version,
    python_requires=python_requires,
    install_requires=dependencies,
    packages=find_packages(include=["spherical_deepkriging*"]),
    include_package_data=True,
    ext_modules=[
        CMakeExtension(
            "spherical_deepkriging.basis_functions.mrts_sphere.cpp_extensions.spherical_basis",
            sourcedir=str(cpp_ext_spherical_dir),
        )
    ],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)

