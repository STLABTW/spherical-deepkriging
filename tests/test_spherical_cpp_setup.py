"""Unit tests for setup.py files."""

import os

import pytest


class TestCppExtensionsSetup:
    """Test suite for spherical_deepkriging/basis_functions/mrts_sphere/cpp_extensions/setup.py."""

    def test_setup_can_be_imported(self):
        setup_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "spherical_deepkriging",
                "basis_functions",
                "mrts_sphere",
                "cpp_extensions",
                "setup.py",
            )
        )
        assert os.path.exists(setup_path)

        try:
            with open(setup_path, "r") as f:
                code = f.read()
            compile(code, setup_path, "exec")
        except SyntaxError as e:
            pytest.fail(f"setup.py has syntax errors: {e}")

    def test_setup_module_structure(self):
        setup_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "spherical_deepkriging",
                "basis_functions",
                "mrts_sphere",
                "cpp_extensions",
                "setup.py",
            )
        )

        with open(setup_path, "r") as f:
            content = f.read()

        assert "from setuptools import" in content
        assert "class CMakeBuild" in content
        assert "setup(" in content
        assert "name=" in content
        assert "spherical_basis" in content

