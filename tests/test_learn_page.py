"""The "Learn xarray-sql in 15 minutes" docs page runs, and says what it does.

The page is one Python program. This runs it and checks every ``# =>``
comment against the value of the line it annotates, so the page fails here
before it drifts from the library.
"""

import ast
import io
import pathlib
import re
import tokenize

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("polars")
pytest.importorskip("adbc_driver_sqlite")
pytest.importorskip("adbc_driver_flightsql")

PAGE = pathlib.Path(__file__).parent.parent / "docs" / "learn.md"


def _tour() -> str:
    """The page's program: the Python block under "The tour"."""
    text = PAGE.read_text()
    tour = text[text.index("## The tour") :]
    block = re.search(r"```python\n(.*?)\n```", tour, re.DOTALL)
    assert block, "the tour has no Python block"
    return block.group(1)


def _expected(lines: list[str], end: int) -> str | None:
    """The ``# =>`` text after line ``end`` (1-based), with any ``#``
    continuation lines that follow it, or None."""
    line = lines[end - 1]
    if "# =>" in line:
        parts = [line.split("# =>", 1)[1]]
    elif end < len(lines) and lines[end].lstrip().startswith("# =>"):
        parts = [lines[end].split("# =>", 1)[1]]
        end += 1
    else:
        return None
    while end < len(lines) and re.match(r"\s*#\s{2,}", lines[end]):
        parts.append(lines[end].lstrip()[1:])
        end += 1
    return "".join(parts)


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


class _Annotate(ast.NodeTransformer):
    """Wraps each annotated expression statement in a call to ``_check``."""

    def __init__(self, lines):
        self.lines = lines
        self.count = 0

    def visit_Expr(self, node):
        expected = _expected(self.lines, node.end_lineno)
        if expected is None:
            return node
        self.count += 1
        call = ast.Call(
            func=ast.Name("_check", ast.Load()),
            args=[
                node.value,
                ast.Constant(expected),
                ast.Constant(node.lineno),
            ],
            keywords=[],
        )
        return ast.copy_location(ast.Expr(call), node)


def _check(value, expected, line):
    assert _squash(expected) in (_squash(repr(value)), _squash(str(value))), (
        f"line {line} of the tour: the page says {expected.strip()}, "
        f"but it is {value!r}"
    )


def test_the_tour_runs_and_every_result_is_what_the_page_says():
    source = _tour()
    annotate = _Annotate(source.splitlines())
    tree = ast.fix_missing_locations(annotate.visit(ast.parse(source)))
    comments = [
        token
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT and token.string.startswith("# =>")
    ]
    assert annotate.count == len(comments), "a # => comment annotates nothing"
    exec(compile(tree, str(PAGE), "exec"), {"_check": _check})
