"""Smoke tests: the package and its subpackages import, and torch is available."""

import importlib

import pytest


def test_package_imports():
    pkg = importlib.import_module("cvttcn")
    assert hasattr(pkg, "__version__")
    assert pkg.__version__ == "0.1.0"


@pytest.mark.parametrize(
    "module",
    [
        "cvttcn.config",
        "cvttcn.data",
        "cvttcn.models",
        "cvttcn.training",
    ],
)
def test_subpackages_import(module):
    assert importlib.import_module(module) is not None


def test_torch_available():
    torch = importlib.import_module("torch")
    # A tiny tensor op confirms the build is functional, not just importable.
    x = torch.ones(2, 3)
    assert x.sum().item() == 6.0
