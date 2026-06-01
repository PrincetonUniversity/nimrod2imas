#!/usr/bin/env python3
"""Lightweight validation for IMAS mhd_linear HDF5 files.

This script is intended for the reduced CPC regression example. It checks that
one or more filesystem-backed IMAS mhd_linear occurrence files exist and contain
basic mhd_linear structure: time slices, toroidal modes, grid coordinates, and
finite perturbation arrays. It deliberately avoids detailed physics comparisons,
because the reduced example is a compact structural/regression test rather than
a full production validation case.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import h5py
import numpy as np


def report(name: str, message: str, status: str) -> None:
    print(f"{name:<32s} {message}; status={status}")


def dataset_exists(h5: h5py.File, path: str) -> bool:
    return path in h5 and isinstance(h5[path], h5py.Dataset)


def finite_dataset_summary(h5: h5py.File, path: str) -> tuple[bool, tuple[int, ...], float]:
    arr = h5[path][()]
    shape = tuple(arr.shape)
    if arr.size == 0:
        return False, shape, 0.0
    finite = np.all(np.isfinite(arr))
    max_abs = float(np.nanmax(np.abs(arr))) if arr.size else 0.0
    return bool(finite), shape, max_abs


def validate_file(filename: Path) -> bool:
    ok = True
    report(filename.name, "checking file", "INFO")

    with h5py.File(filename, "r") as h5:
        roots = list(h5.keys())
        expected_root = filename.stem
        if expected_root in h5:
            root = expected_root
            report("root group", f"found {root}", "PASS")
        elif len(roots) == 1:
            root = roots[0]
            report("root group", f"using sole root {root}", "PASS")
        else:
            report("root group", f"could not identify root; roots={roots}", "FAIL")
            return False

        required = [
            "time",
            "time_slice[]&AOS_SHAPE",
            "time_slice[]&time",
            "time_slice[]&toroidal_mode[]&AOS_SHAPE",
            "time_slice[]&toroidal_mode[]&n_phi",
            "time_slice[]&toroidal_mode[]&plasma&grid&dim1",
            "time_slice[]&toroidal_mode[]&plasma&grid&dim2",
        ]
        for rel in required:
            path = f"{root}/{rel}"
            if dataset_exists(h5, path):
                report(rel, f"shape={h5[path].shape}", "PASS")
            else:
                report(rel, "missing", "FAIL")
                ok = False

        aos_path = f"{root}/time_slice[]&AOS_SHAPE"
        if dataset_exists(h5, aos_path):
            nts = int(np.ravel(h5[aos_path][()])[0])
            report("time_slice count", f"n_time_slice={nts}", "PASS" if nts > 0 else "FAIL")
            ok = ok and nts > 0

        tm_path = f"{root}/time_slice[]&toroidal_mode[]&AOS_SHAPE"
        if dataset_exists(h5, tm_path):
            tm = h5[tm_path][()]
            has_modes = bool(np.any(tm > 0))
            report("toroidal_mode count", f"shape={tm.shape}, max={int(np.max(tm)) if tm.size else 0}", "PASS" if has_modes else "FAIL")
            ok = ok and has_modes

        nphi_path = f"{root}/time_slice[]&toroidal_mode[]&n_phi"
        if dataset_exists(h5, nphi_path):
            nphi = h5[nphi_path][()]
            finite = bool(np.all(np.isfinite(nphi)))
            report("n_phi", f"shape={nphi.shape}, values={np.unique(nphi).tolist()}", "PASS" if finite and nphi.size else "FAIL")
            ok = ok and finite and nphi.size > 0

        field_names = []
        prefix = f"{root}/time_slice[]&toroidal_mode[]&plasma&"
        for path in h5:
            pass
        def collect(name: str, obj) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            if not name.startswith(prefix):
                return
            if "perturbed" not in name:
                return
            if name.endswith("_SHAPE"):
                return
            if name.endswith("&real") or name.endswith("&imaginary"):
                field_names.append(name)

        h5.visititems(collect)
        if not field_names:
            report("perturbation fields", "no perturbation datasets found", "FAIL")
            ok = False
        else:
            report("perturbation fields", f"n_datasets={len(field_names)}", "PASS")

        any_nonzero = False
        for name in sorted(field_names):
            finite, shape, max_abs = finite_dataset_summary(h5, name)
            short = name.replace(prefix, "")
            status = "PASS" if finite and len(shape) >= 2 else "FAIL"
            report(short, f"shape={shape}, max|value|={max_abs:.3e}", status)
            ok = ok and status == "PASS"
            any_nonzero = any_nonzero or max_abs > 0.0

        report("nonzero perturbation check", "at least one nonzero perturbation dataset" if any_nonzero else "all perturbation datasets are zero", "PASS" if any_nonzero else "INFO")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate reduced-example mhd_linear HDF5 output.")
    parser.add_argument("--entry", required=True, help="Filesystem IMAS entry directory, e.g. tests/d3d/4/163518/1")
    parser.add_argument("--occ", type=int, default=None, help="Specific mhd_linear occurrence to check, e.g. 1 or 2. Default: all occurrences.")
    args = parser.parse_args()

    entry = Path(args.entry)
    if args.occ is None:
        files = sorted(Path(p) for p in glob.glob(str(entry / "mhd_linear_*.h5")))
    else:
        files = [entry / f"mhd_linear_{args.occ}.h5"]

    if not files:
        report("mhd_linear files", f"none found under {entry}", "FAIL")
        return 1

    overall = True
    for fn in files:
        if not fn.exists():
            report(fn.name, "missing", "FAIL")
            overall = False
            continue
        overall = validate_file(fn) and overall

    if overall:
        print("PASS: mhd_linear structural validation completed successfully.")
        return 0
    print("FAIL: mhd_linear structural validation found problems.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
