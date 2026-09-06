# test_aiecc_cache_key.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""What the aiecc artifact cache's key must and must not notice -- no NPU required.

A compiled-artifact cache has one failure mode worth testing: serving a previous
run's output after an input changed, so a broken change passes on last-good
binaries. Every input class therefore gets a mutation here.

The two halves are deliberately opposed. `test_key_notices` fails for a key that
is constant, that ignores an input, or that returns None; `test_key_ignores`
fails for a key that is random or over-keyed. Neither half proves anything alone,
and `test_a_degenerate_key_cannot_pass_this_suite` is what asserts that -- it
runs three broken key functions through both halves and requires each to be
rejected. Without it this file could go green against a key that reads nothing.
"""

import pytest
from aie.utils.compile import utils

BASE_MLIR = 'module { aie.device(npu2) { } } // link_with = "k.o"\n'
BASE_OBJ = b"object-v1"
BASE_ARGS = (
    "--peano=/nonexistent/peano",
    "-j8",
    "--get-npu-insts",
    "--npu-insts-name=/tmp/build-a/insts.bin",
    "--get-xclbin",
    "--xclbin-name=/tmp/build-a/final.xclbin",
    "--xclbin-kernel-name=MLIR_AIE",
    "--tmpdir=/tmp/build-a",
    "--output-dir=/tmp/build-a",
)

# An input state, as the five things the key is given.  Mutations are whole
# states rather than side effects, so one cannot leak into the next test.
BASE = dict(mlir=BASE_MLIR, obj=BASE_OBJ, args=BASE_ARGS, chess=False, fold=True)


def _evaluate(keyfn, work, **overrides):
    state = {**BASE, **overrides}
    (
        (work / "k.o").write_bytes(state["obj"])
        if state["obj"] is not None
        else (work / "k.o").unlink(missing_ok=True)
    )
    return keyfn(
        state["mlir"], list(state["args"]), str(work), state["chess"], state["fold"]
    )


# --- what a change to the run must move the key --------------------------

MUTATIONS = {
    "mlir text": dict(mlir=BASE_MLIR.replace("npu2", "npu1")),
    "linked object content": dict(obj=b"object-v2"),
    "a semantic flag": dict(
        args=tuple(a.replace("MLIR_AIE", "OTHER") for a in BASE_ARGS)
    ),
    "which outputs are requested": dict(
        args=tuple(a for a in BASE_ARGS if a != "--get-xclbin")
    ),
    "an unrecognised flag": dict(args=BASE_ARGS + ("--some-new-aiecc-flag",)),
    "the chess/peano front end": dict(chess=True),
    "the DDR-patch ABI": dict(fold=False),
}

# --- and what must NOT, or the cache never hits across two build dirs -----

INVARIANTS = {
    "the destination paths": dict(
        args=tuple(a.replace("build-a", "build-b") for a in BASE_ARGS)
    ),
    "the thread count": dict(
        args=tuple(("-j1" if a == "-j8" else a) for a in BASE_ARGS)
    ),
}


def _notices_everything(keyfn, work):
    base = _evaluate(keyfn, work)
    return all(_evaluate(keyfn, work, **m) != base for m in MUTATIONS.values())


def _ignores_everything(keyfn, work):
    base = _evaluate(keyfn, work)
    return all(_evaluate(keyfn, work, **i) == base for i in INVARIANTS.values())


@pytest.fixture
def work(tmp_path):
    return tmp_path


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_key_notices(work, name):
    base = _evaluate(utils._aiecc_cache_key, work)
    assert base is not None
    assert (
        _evaluate(utils._aiecc_cache_key, work, **MUTATIONS[name]) != base
    ), f"the key ignored a change to {name}"


@pytest.mark.parametrize("name", sorted(INVARIANTS))
def test_key_ignores(work, name):
    base = _evaluate(utils._aiecc_cache_key, work)
    assert (
        _evaluate(utils._aiecc_cache_key, work, **INVARIANTS[name]) == base
    ), f"the key moved with {name}, which does not change what aiecc produces"


def test_unreadable_linked_object_disables_the_cache(work):
    """Fail closed: a missing input yields no key, never a key computed without it."""
    assert _evaluate(utils._aiecc_cache_key, work, obj=None) is None


_DEGENERATE = {
    "constant": lambda *a, **k: "c" * 32,
    "random": lambda *a, **k: __import__("os").urandom(16).hex(),
    "always none": lambda *a, **k: None,
    "keyed on the mlir alone": lambda mlir, *a, **k: __import__("hashlib")
    .sha256(mlir.encode())
    .hexdigest(),
}


@pytest.mark.parametrize("name", sorted(_DEGENERATE))
def test_a_degenerate_key_cannot_pass_this_suite(work, name):
    keyfn = _DEGENERATE[name]
    assert not (
        _notices_everything(keyfn, work) and _ignores_everything(keyfn, work)
    ), f"a key that is '{name}' passes both halves -- they prove nothing"


def test_the_real_key_passes_both_halves(work):
    """The dual of the test above: the assertions have teeth AND the key clears them."""
    assert _notices_everything(utils._aiecc_cache_key, work)
    assert _ignores_everything(utils._aiecc_cache_key, work)
