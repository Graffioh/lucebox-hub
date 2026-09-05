"""Exercise the real C++ runner and CTest adapter without a GPU toolkit."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cppunit")
    source = directory / "cases.cpp"
    source.write_text('''#include "CppUnitTestFramework.hpp"
#include <stdexcept>
struct Runner {};
TEST_CASE(Runner, pass) { CHECK(true); }
TEST_CASE(Runner, pass_suffix) { CHECK(false); }
TEST_CASE_WITH_TAGS(Runner, tagged, "Runner::pass", "tag") { CHECK(false); }
TEST_CASE(Runner, check_failure) { CHECK(false); }
TEST_CASE(Runner, require_failure) { REQUIRE(false); }
TEST_CASE(Runner, exception) { throw std::runtime_error("intentional"); }
TEST_CASE(Runner, skip) { SKIP("unavailable fixture"); }
''')
    result = {}
    for name, sources in {"cases": [source], "empty": []}.items():
        binary = directory / name
        subprocess.run(
            ["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "server/test"),
             str(ROOT / "server/test/test_unit_main.cpp"), *map(str, sources), "-o", str(binary)],
            check=True, capture_output=True, text=True, timeout=60,
        )
        result[name] = binary
    return result


@pytest.mark.parametrize(("arguments", "expected"), [
    (["--exact", "Runner::pass"], 0),
    (["--exact", "Runner::check_failure"], 1),
    (["--exact", "Runner::require_failure"], 1),
    (["--exact", "Runner::exception"], 1),
    (["--exact", "Runner::skip"], 77),
    (["--exact", "Runner::missing"], 1),
    (["missing"], 1),
    (["--exact", "tag"], 1),
    (["tag"], 1),
    (["Runner::pass"], 1),
    (["--exact", "Runner::pass", "Runner::skip"], 0),
    (["--exact", "Runner::check_failure", "Runner::skip"], 1),
    (["--unknown-option"], 2),
])
def test_runner_exit_status(binaries, arguments, expected):
    result = subprocess.run(
        [str(binaries["cases"]), *arguments], capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    if arguments == ["tag"]:
        assert "Failed:  1" in result.stdout
    if arguments == ["--exact", "tag"]:
        assert "No test cases matched" in result.stderr


def test_empty_binary_fails(binaries):
    result = subprocess.run([str(binaries["empty"])], capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "No test cases matched" in result.stderr


def discover(binary, directory):
    return subprocess.run(
        ["cmake", f"-DTEST_EXECUTABLE={binary}", f"-DTEST_WORKING_DIR={directory}",
         f"-DCTEST_FILE={directory / 'discovered.cmake'}", "-DTEST_PREFIX=probe.",
         "-P", str(ROOT / "server/cmake/DiscoverCppUnitTests.cmake")],
        capture_output=True, text=True, timeout=15,
    )


def test_discovery_registers_exact_cases_and_preserves_skip(binaries, tmp_path):
    result = discover(binaries["cases"], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    (tmp_path / "CTestTestfile.cmake").write_text('include("discovered.cmake")\n')
    result = subprocess.run(
        ["ctest", "--test-dir", str(tmp_path), "--output-on-failure", "--no-tests=error",
         "-R", r"^probe\.Runner\.(pass|skip)$"],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "100% tests passed" in result.stdout
    assert "(Skipped)" in result.stdout


def test_empty_discovery_fails(binaries, tmp_path):
    result = discover(binaries["empty"], tmp_path)
    assert result.returncode != 0
    assert "No tests discovered" in result.stderr
