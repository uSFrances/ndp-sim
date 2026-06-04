from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_src_package_dir = Path(__file__).resolve().parent.parent / "src" / __name__
if _src_package_dir.is_dir():
    __path__.append(str(_src_package_dir))