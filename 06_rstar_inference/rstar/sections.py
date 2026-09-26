"""Import section 01's and section 03's `src` packages without letting them collide.

Both sections own a top-level package called `src`, so putting both on sys.path silently resolves to one of them.
Each is loaded here under a unique alias with importlib and registered in sys.modules under that alias only;
neither directory touches sys.path, so `import src` elsewhere in the process is unaffected. Both packages use
relative imports (`from .model import ...`), which resolve against the alias.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

from .config import REPO_ROOT

_SECTIONS = {
    "01": ("rstar_sec01_src", "01_model_federated"),
    "03": ("rstar_sec03_src", "03_augmentation_eval"),
}


def alias_name(section: str) -> str:
    return _SECTIONS[section][0]


def load_section_package(section: str, repo_root: str | Path | None = None):
    """Return the alias package for section '01' or '03' (cached in sys.modules after the first call)."""
    if section not in _SECTIONS:
        raise ValueError(f"section must be one of {sorted(_SECTIONS)}, got {section!r}")
    alias, folder = _SECTIONS[section]
    if alias in sys.modules:
        return sys.modules[alias]
    pkg_dir = Path(repo_root or REPO_ROOT) / folder / "src"
    init = pkg_dir / "__init__.py"
    if not init.exists():
        raise FileNotFoundError(f"section {section} package not found: {init}")
    spec = importlib.util.spec_from_file_location(alias, init, submodule_search_locations=[str(pkg_dir)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for name in [n for n in sys.modules if n == alias or n.startswith(alias + ".")]:
            sys.modules.pop(name, None)
        raise
    return module


def import_submodule(section: str, name: str, repo_root: str | Path | None = None):
    """e.g. import_submodule('03', 'model') -> the module rstar_sec03_src.model."""
    load_section_package(section, repo_root)
    return importlib.import_module(f"{alias_name(section)}.{name}")
