# AGENTS.md

Guidance for contributors (including AI assistants) working on `xarray-sql`. It
summarizes recurring maintainer review feedback so changes land clean.

## Documentation and comments

- Keep docstrings and comments self-contained. Do **not** put GitHub issue or PR
  numbers in docstrings or code comments; a reader should not need the issue
  tracker to understand the code. Issue references belong in the commit message
  and PR description (e.g. `Closes #189`), not in the source.
- Do not reference the review conversation, chat, or "the reporter" in comments.
  Describe the behavior, not how it came up.

## API surface

- Mark internal helpers private with a leading underscore when they are not part
  of the public API.

## Tests

- Test the public contract (values, dims, coords, attrs), not internal call
  counts or private classes, so the suite survives refactors.
- Avoid redundant tests: if a public-path test already covers a behavior, do not
  add a second lower-level test for the same thing.
- Make query results deterministic with `ORDER BY` so assertions do not have to
  re-sort the output.
- Do not pass `dims=` to `to_dataset()` when inference already resolves them.
  Reserve explicit `dims=` / `template=` for genuinely ambiguous cases (multiple
  registered Datasets, or a test that is specifically exercising those
  arguments).

## Imports

- Keep imports at the top of the file. Assume transitive dependencies are safe
  to import non-locally, rather than deferring imports into functions to avoid
  a dependency.
