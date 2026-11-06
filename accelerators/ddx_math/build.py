"""Build ddx's exact error-function derivative in an isolated checkout.

ddx (https://github.com/xqlsystems/ddx) differentiates SQL. The pinned
feature commit adds erf's analytic derivative, including higher derivatives, and
exposes prepared primal output metadata for loss reuse.
Requires git, cargo, CMake, maturin and uv on PATH.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REVISION = "953b78ecfc8bce2b1c1a71f0d4c7300444d00cc7"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="reuse a clean pinned checkout"
    )
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        parser.error("use the repository virtual environment's Python")
    for command in ("git", "cargo", "cmake", "uv"):
        if shutil.which(command) is None:
            parser.error(f"{command} must be on PATH")
    with tempfile.TemporaryDirectory(prefix="xql-ddx-math-") as temporary:
        source = args.source or Path(temporary) / "source"
        if args.source is None:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/cromwellian/ddx.git",
                    str(source),
                ],
                check=True,
            )
            subprocess.run(
                ["git", "checkout", _REVISION], cwd=source, check=True
            )
        if (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source, text=True
            ).strip()
            != _REVISION
        ):
            parser.error(f"expected ddx revision {_REVISION}")
        if subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=source
        ):
            parser.error("source checkout must be clean")
        wheels = Path(temporary) / "wheels"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "maturin",
                "build",
                "--release",
                "--locked",
                "--out",
                str(wheels),
            ],
            cwd=source / "python/ddxdb",
            env=os.environ | {"PYO3_PYTHON": sys.executable},
            check=True,
        )
        (wheel,) = wheels.glob("*.whl")
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-deps",
                "--reinstall",
                str(wheel),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
