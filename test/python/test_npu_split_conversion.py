# test_npu_split_conversion.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Guard the compile/execute split's lit-configuration invariants — no NPU required.

A test converted to the split writes its artifacts in one lit invocation and
reads them back in another. If the two ends disagree about which artifacts
belong to which test, the result is not an error: the execute phase runs, and
passes, against another test's xclbin. That is the failure this file exists to
make impossible to introduce quietly, so it runs on every leg rather than only
on the ones with a device attached.

The second half covers which parallelism group each test lands in, where the
mistake to guard against is conflating the device (serialized by run_on_npu.py's
lock, or by the lit group where fcntl is missing) with chess compile memory,
which the device lock says nothing about.
"""

import builtins
import os
import sys
import textwrap

import pytest

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
NPU_XRT = os.path.normpath(os.path.join(_TEST_DIR, "..", "npu-xrt"))

# aie_lit_utils is lit's own configuration support: it is imported by the config
# files, so it is on the source tree rather than in the installed package that
# lit puts on a test's PYTHONPATH.
sys.path.insert(0, os.path.normpath(os.path.join(_TEST_DIR, "..", "..", "python")))

from aie_lit_utils.lit_config_helpers import (  # noqa: E402
    npu_split_conversion_violations,
    npu_split_parallelism_group,
)

CONVERTED = """\
// RUN: %npu_build% %aiecc --get-xclbin --xclbin-name=aie.xclbin %S/aie.mlir
// RUN: %npu_build% %host_clang %S/test.cpp -o test.exe
// RUN: %npu_run% %run_on_npu1% ./test.exe -x aie.xclbin
"""


def write(tmp_path, **tests):
    for name, body in tests.items():
        (tmp_path / f"{name}.lit").write_text(textwrap.dedent(body))
    return str(tmp_path)


def test_in_tree_conversions_are_sound():
    assert npu_split_conversion_violations(NPU_XRT) == []


def test_unconverted_siblings_are_not_the_split_s_business(tmp_path):
    plain = CONVERTED.replace("%npu_build% ", "").replace("%npu_run% ", "")
    assert npu_split_conversion_violations(write(tmp_path, a=plain, b=plain)) == []


def test_a_lone_converted_test_owns_its_directory(tmp_path):
    assert npu_split_conversion_violations(write(tmp_path, a=CONVERTED)) == []


def test_siblings_without_a_private_directory_are_rejected(tmp_path):
    violations = npu_split_conversion_violations(
        write(tmp_path, a=CONVERTED, b=CONVERTED)
    )
    assert len(violations) == 2
    assert all("shares its directory" in v for v in violations)


def test_siblings_with_a_private_directory_are_accepted(tmp_path):
    scoped = "// RUN: cd %t.d\n" + CONVERTED
    assert npu_split_conversion_violations(write(tmp_path, a=scoped, b=scoped)) == []


@pytest.mark.parametrize(
    "suffix",
    [" > out.log", " | FileCheck %s", " && echo ok", "; echo ok", " < in.txt"],
)
def test_a_marked_line_must_be_a_simple_command(tmp_path, suffix):
    body = CONVERTED.replace("-o test.exe", "-o test.exe" + suffix)
    (violation,) = npu_split_conversion_violations(write(tmp_path, a=body))
    assert "not a simple command" in violation


def test_both_phases_must_be_present(tmp_path):
    build_only = CONVERTED.replace("%npu_run%", "%npu_build%")
    (violation,) = npu_split_conversion_violations(write(tmp_path, a=build_only))
    assert "no %npu_run% line" in violation

    run_only = CONVERTED.replace("%npu_build%", "%npu_run%")
    (violation,) = npu_split_conversion_violations(write(tmp_path, a=run_only))
    assert "no %npu_build% line" in violation


def test_a_device_line_must_state_its_phase(tmp_path):
    body = CONVERTED + "// RUN: %run_on_npu2% ./test.exe -x aie.xclbin\n"
    (violation,) = npu_split_conversion_violations(write(tmp_path, a=body))
    assert "no phase marker" in violation


def test_a_continued_line_is_checked_as_one_command(tmp_path):
    body = (
        "// RUN: %npu_build% %aiecc --get-xclbin \\\n"
        "// RUN:   --xclbin-name=aie.xclbin %S/aie.mlir > log.txt\n"
        "// RUN: %npu_run% %run_on_npu1% ./test.exe -x aie.xclbin\n"
    )
    (violation,) = npu_split_conversion_violations(write(tmp_path, a=body))
    assert "not a simple command" in violation


# --- parallelism groups ----------------------------------------------------

CHESS = "// REQUIRES: ryzen_ai_npu1, chess\n" + CONVERTED
UNCONVERTED = CONVERTED.replace("%npu_build% ", "").replace("%npu_run% ", "")


class FakeTest:
    def __init__(self, path):
        self._path = path

    def getSourcePath(self):
        return self._path


@pytest.fixture
def group(tmp_path, monkeypatch):
    def ask(body, split, fcntl_available=True):
        path = tmp_path / "run.lit"
        path.write_text(body)
        monkeypatch.setenv("AIE_NPU_SPLIT", split)
        real_import = builtins.__import__

        def maybe_import(name, *args, **kwargs):
            if name == "fcntl" and not fcntl_available:
                raise ImportError("no fcntl on this platform")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", maybe_import)
        return npu_split_parallelism_group(FakeTest(str(path)))

    return ask


@pytest.mark.parametrize("split", ["", "execute"])
def test_the_device_lock_replaces_the_group(group, split):
    assert group(CONVERTED, split) is None
    assert group(UNCONVERTED, split) is None


@pytest.mark.parametrize("split", ["", "execute"])
def test_without_fcntl_the_group_comes_back(group, split):
    assert group(CONVERTED, split, fcntl_available=False) == "npu-xrt"
    assert group(UNCONVERTED, split, fcntl_available=False) == "npu-xrt"


def test_a_converted_test_compiles_in_parallel(group):
    assert group(CONVERTED, "compile") is None


def test_an_unconverted_test_still_dispatches_in_the_compile_phase(group):
    # No markers means it runs whole in both passes, so it reaches the device
    # and has to stay serialized whether or not the file lock exists.
    assert group(UNCONVERTED, "compile") == "npu-xrt"
    assert group(UNCONVERTED, "compile", fcntl_available=False) == "npu-xrt"


@pytest.mark.parametrize("fcntl_available", [True, False])
def test_chess_compiles_are_bounded_by_memory_not_by_the_device(group, fcntl_available):
    # The regression this pins: gating the chess bound on the fcntl import would
    # leave parallel chess compiles unbounded on the platform that runs them.
    assert (
        group(CHESS, "compile", fcntl_available=fcntl_available)
        == "npu-split-chess-compile"
    )


def test_an_unreadable_test_stays_serialized(monkeypatch, tmp_path):
    monkeypatch.setenv("AIE_NPU_SPLIT", "compile")
    missing = str(tmp_path / "does_not_exist.lit")
    assert npu_split_parallelism_group(FakeTest(missing)) == "npu-xrt"
