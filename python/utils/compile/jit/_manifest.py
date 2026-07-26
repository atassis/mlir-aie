"""Recorded dependency manifest for the JIT cache.

The cache key cannot know a design's kernel sources: they are registered while
MLIR is generated, which happens after the key is computed and not at all on a
hit. So instead of predicting inputs, record the ones a build actually consumed
and check them on the next lookup.

**Peano only.** A manifest is written when every kernel in the design was
compiled by Peano, because Peano is asked for a depfile and so reports exactly
which files it opened. The Chess front-end is driven through
``xchesscc_wrapper`` and emits nothing equivalent, so for a Chess design the
consumed set is unknowable from here. Rather than record a set that is silently
short -- which would read as verified and is worse than recording nothing -- no
manifest is written, and such designs keep exactly the cache behaviour they have
today.

Everything here fails closed: an absent, unparsable, partial or unverifiable
manifest is a miss. The cache never trusts what it cannot check.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_NAME = "deps.json"
_VERSION = 1


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _entry(path: Path) -> dict:
    st = path.stat()
    return {
        "path": str(path),
        "size": st.st_size,
        "mtime": st.st_mtime,
        "sha256": _digest(path),
    }


def _parse_depfile(depfile: Path) -> list[Path]:
    """Read a make-style depfile: 'target: a b \\\n  c d'."""
    text = depfile.read_text()
    _, _, rhs = text.partition(":")
    return [Path(tok) for tok in rhs.replace("\\\n", " ").split() if tok.strip()]


def record(kernel_dir, external_kernels, source_files, used_chess=False) -> None:
    """Record the inputs this build consumed, if they can be known exactly.

    Every kernel compiled through Peano leaves a depfile naming what the
    compiler actually opened.  A kernel compiled any other way -- today, the
    Chess path, which takes no -MD -- leaves none, and its includes are
    therefore unknown.  Rather than record a set that is silently short, no
    manifest is written at all: a missing manifest reads as a miss, so such a
    design keeps exactly the behaviour it has now instead of gaining a cache
    entry nobody can validate.
    """
    kernel_dir = Path(kernel_dir)
    compiled = [
        f
        for f in external_kernels
        if getattr(f, "_source_file", None) or getattr(f, "_source_string", None)
    ]
    depfiles = list(kernel_dir.glob("*.d"))
    if used_chess or (compiled and not depfiles):
        logger.debug(
            "cache manifest: %d kernel(s) compiled without a depfile; not "
            "recording an unverifiable input set",
            len(compiled),
        )
        return

    found: set[Path] = set()
    for dep in depfiles:
        try:
            found.update(_parse_depfile(dep))
        except OSError:
            return  # cannot know the set -> do not claim one
    for f in compiled:
        if getattr(f, "_source_file", None):
            found.add(Path(f._source_file))
    found.update(Path(sf) for sf in source_files)
    _write(kernel_dir, sorted({p for p in found if p.is_file()}, key=str))


def _write(kernel_dir: Path, inputs: list[Path]) -> None:
    payload = {"version": _VERSION, "inputs": []}
    for path in inputs:
        try:
            payload["inputs"].append(_entry(path))
        except OSError:
            # An input we cannot record is an input we cannot verify.  Refuse to
            # write a manifest that would later read as complete.
            logger.warning("cache manifest: %s unreadable; not recording", path)
            return
    tmp = Path(kernel_dir) / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, Path(kernel_dir) / MANIFEST_NAME)


def write_for_test(kernel_dir, inputs) -> None:
    """Write a manifest directly. For tests that fabricate a cache entry."""
    _write(Path(kernel_dir), [Path(i) for i in inputs])


def is_valid(kernel_dir: Path) -> bool:
    """True only if every recorded input is still exactly as recorded.

    Fails closed: a missing, unparsable or partial manifest is a miss, never a
    hit.  Anything this cannot verify, it does not trust.
    """
    path = Path(kernel_dir) / MANIFEST_NAME
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    if payload.get("version") != _VERSION:
        return False
    for rec in payload.get("inputs", ()):
        f = Path(rec["path"])
        try:
            st = f.stat()
        except OSError:
            return False
        # tsbuildinfo pattern: size+mtime agree -> trust it, skip the hash.
        if st.st_size == rec["size"] and st.st_mtime == rec["mtime"]:
            continue
        try:
            if _digest(f) != rec["sha256"]:
                return False
        except OSError:
            return False
    return True
