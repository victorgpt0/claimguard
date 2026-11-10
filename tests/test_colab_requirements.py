"""The Colab GPU worker installs the web packages from requirements/colab-worker.txt, not from
uv.lock. A package added to the web group but not to that file breaks the worker at start-up
(lucide did: "No module named 'lucide'"), so every web dependency must be pinned there, at the
version uv.lock holds."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _name(spec: str) -> str:
    return re.split(r"[<>=!~\[; ]", spec, maxsplit=1)[0].lower().replace("_", "-")


def test_colab_worker_pins_every_web_dependency_at_the_locked_version():
    web = [_name(s) for s in tomllib.loads((ROOT / "pyproject.toml").read_text())["dependency-groups"]["web"]]
    pinned = {}
    for line in (ROOT / "requirements" / "colab-worker.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, _, rest = line.partition("==")
            pinned[name.strip().lower()] = rest.split(";")[0].strip()
    locked = {p["name"]: p["version"] for p in tomllib.loads((ROOT / "uv.lock").read_text())["package"]
              if "version" in p}
    assert not [n for n in web if n not in pinned], "add these to requirements/colab-worker.txt"
    assert {n: v for n, v in pinned.items() if locked.get(n) != v} == {}
