# Model-free merge checks

From the repository root, run the same suite as hosted CI:

```sh
uv run --frozen --extra dev pytest -q
```

`pyproject.toml` owns discovery: `harness/tests`, the concurrency benchmark
unit suites, and `server/tests/test_server_parallel_unit.py`. New tests under
the two directories are collected automatically. Tests that need model weights,
a live inference server, or GPU execution must remain explicitly invoked.

The C++ runner contract tests compile tiny binaries with `c++` (C++17) and use
`cmake`/`ctest` to exercise discovery and exit statuses. Those tools are required;
missing tools fail the suite instead of skipping the gate. No CUDA/HIP toolkit
is needed. Subprocesses and binaries are confined to pytest temporary directories.

The runner returns 1 for assertion failures, exceptions, and empty selections;
77 is reserved for selections whose tests explicitly skip. `--exact` matches
complete test names only; ordinary selectors still match substrings and tags.
CTest discovery rejects binaries that register no tests.

Run one suite directly when iterating, for example:

```sh
uv run --frozen --extra dev pytest -q harness/tests/test_cppunit_runner.py
```

This suite complements the CMake server unit tests and hardware qualification;
it does not establish kernel correctness or model quality by itself.
