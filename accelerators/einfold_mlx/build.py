"""Build the pinned EinFold dense-executor hook into this Python environment.

EinFold (https://github.com/xqlsystems/einfold) is a DataFusion tensor
optimizer. Its released Python API has no external kernel interface. This
script builds and installs a wheel from the pinned feature commit in an
isolated checkout. Requires git, a Rust toolchain, maturin,
and uv. Nothing is installed into the system Python.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REVISION = "9feb8e5a7efaecf6322afc436d0cac60814ef842"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="reuse a clean pinned source checkout"
    )
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        parser.error(
            "run this builder with the checkout's virtual-environment Python"
        )
    for command in ("git", "cargo", "uv"):
        if shutil.which(command) is None:
            parser.error(f"{command} must be on PATH")
    with tempfile.TemporaryDirectory(prefix="xql-einfold-") as temporary:
        work = Path(temporary)
        source = args.source or work / "source"
        if args.source is None:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/cromwellian/einfold.git",
                    str(source),
                ],
                check=True,
            )
            subprocess.run(
                ["git", "checkout", _REVISION], cwd=source, check=True
            )
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        if revision != _REVISION:
            parser.error(
                f"expected EinFold revision {_REVISION}, got {revision}"
            )
        if subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=source
        ):
            parser.error(
                "the source checkout must be clean"
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "maturin",
                "build",
                "--release",
                "--locked",
                "--out",
                str(work / "wheels"),
            ],
            cwd=source / "python",
            env={**os.environ, "PYO3_PYTHON": sys.executable},
            check=True,
        )
        (wheel,) = (work / "wheels").glob("*.whl")
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--force-reinstall",
                "--no-deps",
                str(wheel),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
