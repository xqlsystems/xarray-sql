window.BENCHMARK_DATA = {
  "lastUpdate": 1795110695874,
  "repoUrl": "https://github.com/xqlsystems/xarray-sql",
  "entries": {
    "Python benchmarks (pytest-benchmark)": [
      {
        "commit": {
          "author": {
            "email": "ekgto445@gmail.com",
            "name": "Dave Bunten",
            "username": "d33bs"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b7748127e79d307a148227445a2885a62e2479fe",
          "message": "Track benchmarks over time in CI, starting with one (#306)\n\nTimes a group-by mean over an in-memory Dataset with pytest-benchmark on every push to main, and stores the history on gh-pages with github-action-benchmark. Every PR runs the benchmark once, untimed, as a test.\n\nCloses #298.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01NyN7idACeDMh94qrnFMu6h",
          "timestamp": "2026-11-19T08:55:50-08:00",
          "tree_id": "adba5a62d73ec7c7662a2c0d960f02da42fc754a",
          "url": "https://github.com/xqlsystems/xarray-sql/commit/b7748127e79d307a148227445a2885a62e2479fe"
        },
        "date": 1795110688994,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_groupby_mean_over_time",
            "value": 333.40981609943645,
            "unit": "iter/sec",
            "range": "stddev: 0.0001509981587069901",
            "extra": "mean: 2.9993118130084064 msec\nrounds: 123"
          }
        ]
      }
    ]
  }
}