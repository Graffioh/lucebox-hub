# Host C/C++ Coverage

The `coverage` target produces an LLVM source-based coverage report for host
C and C++ code. CUDA and HIP source files are intentionally excluded.

## Prerequisites

- Clang and matching `llvm-cov` and `llvm-profdata` executables
- The usual backend build prerequisites (CUDA or HIP)

On Debian or Ubuntu, install the LLVM toolchain package matching the Clang
version used for the build.

## Generate the report

Configure a separate Debug build so coverage instrumentation does not affect
the normal build:

```bash
cd server
cmake -S . -B build-coverage \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DDFLASH27B_COVERAGE=ON \
  -DDFLASH27B_TESTS=ON
cmake --build build-coverage --target coverage -j
```

For CUDA builds where `nvcc` is not on `PATH`, set `CUDACXX` before
configuring:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc cmake -S . -B build-coverage ...
```

The target runs CTest with `LLVM_PROFILE_FILE` configured, merges the emitted
profiles, prints a text summary, and writes the HTML report to:

```text
server/build-coverage/coverage/html/index.html
```

The merged profile is `server/build-coverage/coverage/coverage.profdata`.

## Test failures

The HTML report is generated even when CTest fails, then the `coverage` target
returns CTest's failure status. Review the test failures before treating the
report as a passing coverage run.
