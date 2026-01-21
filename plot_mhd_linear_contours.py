#!/usr/bin/env python3
"""
plot_mhd_linear_contours_fixed.py

Contour plotting utility for IMAS mhd_linear IDS produced by dump2imas.py.

Supports:
  - scalar perturbations: p, t, n
  - vector perturbations: b, v (components r|z|phi)
  - parts: real, imag, amp  (amp = sqrt(real^2 + imag^2))

Time indexing:
  --time-index refers to the *non-empty* time_slices by default (i.e. those with toroidal_mode entries).
  Use --raw-time-index to index the raw time_slice array.

Compatibility:
  Some imas-python versions require db.get(<ids_name_str>, occ) and return the IDS.
  Others accept db.get(<IDSToplevel>, occ) and fill in place. This script handles both.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, List, Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt


# IMAS integer placeholder commonly used for "not set" in many DDs
PLACEHOLDER_INT = -999999999


# ----------------------------- IMAS helpers -----------------------------

def _open_dbentry(backend: str, entry_dir: str, mode: str = "r", dd_version: str | None = None):
    # IMAS uses the Data Dictionary (DD) version both for DBEntry I/O and for IDS
    # object materialization. If DBEntry is created with a different DD (even a
    # different *minor* version), some fields may read back as placeholders.
    #
    # We therefore try, in order:
    #   1) Pass dd_version directly to DBEntry (newer imas-python)
    #   2) Temporarily set IMAS_VERSION (older environments)
    import imas  # type: ignore

    old_ver = os.environ.get("IMAS_VERSION")
    if dd_version:
        os.environ["IMAS_VERSION"] = str(dd_version)

    uri = f"imas:{backend}?path={entry_dir}"

    db = None
    if dd_version:
        # Signature differs across versions; try a few.
        for ctor_args, ctor_kwargs in (
            ((uri, mode), {"dd_version": str(dd_version)}),
            ((uri, mode, str(dd_version)), {}),
        ):
            try:
                db = imas.DBEntry(*ctor_args, **ctor_kwargs)
                break
            except TypeError:
                db = None
    if db is None:
        db = imas.DBEntry(uri, mode)
    try:
        db.open()
    except Exception as e:
        # Some backends open in constructor; tolerate "already open".
        if "already open" not in str(e).lower():
            # Some versions don't have open() at all; tolerate that as well.
            if "has no attribute" not in str(e).lower():
                raise
    # Restore environment (best effort)
    try:
        if dd_version:
            if old_ver is None:
                os.environ.pop("IMAS_VERSION", None)
            else:
                os.environ["IMAS_VERSION"] = old_ver
    except Exception:
        pass
    return db, uri, imas


def _get_ids(db: Any, factory: Any, ids_name: str, occ: int):
    """Robust getter across IMAS python variants."""
    # Prefer the "fill in-place" variant when available. This ensures the IDS is
    # instantiated from the factory (and thus the requested DD version), rather
    # than whatever DD DBEntry happens to default to.
    ids = factory.new(ids_name)
    try:
        db.get(ids, occ)
        return ids
    except Exception:
        pass

    # Fallback: db.get(name, occ) returns populated IDS (DD controlled by DBEntry)
    obj = db.get(ids_name, occ)
    if obj is not None:
        return obj
    return ids


def _aos_size(x: Any) -> int:
    for attr in ("size", "n"):
        try:
            v = getattr(x, attr)
            if isinstance(v, int):
                return v
        except Exception:
            pass
    try:
        return len(x)
    except Exception:
        pass
    # last resort
    n = 0
    while True:
        try:
            _ = x[n]
            n += 1
        except Exception:
            break
    return n


# ---------------------------- IDS navigation ----------------------------

def _mode_count(ts: Any) -> int:
    try:
        return _aos_size(ts.toroidal_mode)
    except Exception:
        return 0


def _ts_time(ts: Any) -> float:
    try:
        return float(ts.time)
    except Exception:
        return float("nan")


def _pick_time_slice(mhd: Any, time_index: int, raw: bool) -> Tuple[int, Any, List[int]]:
    n_ts = _aos_size(mhd.time_slice)
    nonempty = [i for i in range(n_ts) if _mode_count(mhd.time_slice[i]) > 0]

    if raw:
        if time_index < 0 or time_index >= n_ts:
            raise RuntimeError(f"time-index {time_index} out of range [0,{n_ts-1}] for raw time_slices")
        return time_index, mhd.time_slice[time_index], nonempty

    if not nonempty:
        raise RuntimeError("mhd_linear has no non-empty time_slice entries (no toroidal_mode data).")

    if time_index < 0 or time_index >= len(nonempty):
        raise RuntimeError(f"time-index {time_index} out of range [0,{len(nonempty)-1}] for non-empty time_slices")
    raw_i = nonempty[time_index]
    return raw_i, mhd.time_slice[raw_i], nonempty


def _select_mode_index(ts: Any, n_tor: Optional[int], mode_index: Optional[int]) -> int:
    nm = _mode_count(ts)
    if nm == 0:
        raise RuntimeError("Selected time_slice has no toroidal_mode entries.")

    if mode_index is not None:
        if mode_index < 0 or mode_index >= nm:
            raise RuntimeError(f"mode-index {mode_index} out of range [0,{nm-1}]")
        return mode_index

    if n_tor is None:
        return 0

    # Normal selection: match by stored n_tor.
    nvals: List[int] = []
    for i in range(nm):
        try:
            nvals.append(int(ts.toroidal_mode[i].n_tor))
        except Exception:
            nvals.append(PLACEHOLDER_INT)
    for i, nv in enumerate(nvals):
        if nv == int(n_tor):
            return i

    # If n_tor is not stored (all placeholders), provide a safe fallback:
    # interpret --n-tor/--keff as a *mode index* if it is in range.
    if nm > 0 and all((nv == PLACEHOLDER_INT) for nv in nvals):
        if 0 <= int(n_tor) < nm:
            return int(n_tor)
        if 1 <= int(n_tor) <= nm:
            # Allow 1-based mode indexing as a convenience
            return int(n_tor) - 1
        raise RuntimeError(
            "This time_slice has toroidal_mode entries, but n_tor is not set (all values are placeholders). "
            "Use --mode-index (0-based) or fix dump2imas.py to populate toroidal_mode[].n_tor from the dump file."
        )

    raise RuntimeError(
        f"Requested n_tor={n_tor} not found at this time_slice. Use --info to list available modes."
    )


def _pick_part(node: Any, part: str) -> np.ndarray:
    part = part.lower()
    if part == "real":
        return np.asarray(node.real)
    if part == "imag":
        return np.asarray(node.imaginary)
    if part == "amp":
        re = np.asarray(node.real)
        im = np.asarray(node.imaginary)
        return np.sqrt(re * re + im * im)
    raise RuntimeError(f"Unknown part {part!r}")


def _extract_rz_and_field(ts: Any, mi: int, field: str, part: str, component: Optional[str]):
    tm = ts.toroidal_mode[mi]
    pl = tm.plasma

    # Prefer coordinate_system r/z (stitched 2D)
    R2d = Z2d = None
    try:
        cs = pl.coordinate_system
        r = np.asarray(cs.r)
        z = np.asarray(cs.z)
        if r.size > 0 and z.size > 0:
            R2d, Z2d = r, z
    except Exception:
        pass

    field = field.lower()
    title = ""

    if field in ("p", "pressure"):
        F = _pick_part(pl.pressure_perturbed, part)
        title = "pressure_perturbed"
    elif field in ("t", "temperature"):
        F = _pick_part(pl.temperature_perturbed, part)
        title = "temperature_perturbed"
    elif field in ("n", "mass_density"):
        F = _pick_part(pl.mass_density_perturbed, part)
        title = "mass_density_perturbed"
    elif field in ("b", "bfield"):
        if component is None:
            raise RuntimeError("Vector field b requires --component r|z|phi")
        comp = component.lower()
        if comp == "r":
            F = _pick_part(pl.b_field_perturbed.coordinate1, part)
            title = "b_field_perturbed.r"
        elif comp == "z":
            F = _pick_part(pl.b_field_perturbed.coordinate2, part)
            title = "b_field_perturbed.z"
        elif comp == "phi":
            F = _pick_part(pl.b_field_perturbed.coordinate3, part)
            title = "b_field_perturbed.phi"
        else:
            raise RuntimeError("Invalid --component (use r|z|phi)")
    elif field in ("v", "vel", "velocity"):
        if component is None:
            raise RuntimeError("Vector field v requires --component r|z|phi")
        comp = component.lower()
        if comp == "r":
            F = _pick_part(pl.velocity_perturbed.coordinate1, part)
            title = "velocity_perturbed.r"
        elif comp == "z":
            F = _pick_part(pl.velocity_perturbed.coordinate2, part)
            title = "velocity_perturbed.z"
        elif comp == "phi":
            F = _pick_part(pl.velocity_perturbed.coordinate3, part)
            title = "velocity_perturbed.phi"
        else:
            raise RuntimeError("Invalid --component (use r|z|phi)")
    else:
        raise RuntimeError(f"Unknown field {field!r}")

    F2d = np.asarray(F)
    if F2d.ndim != 2:
        raise RuntimeError(f"Field array must be 2D for contour plot; got {F2d.shape}")

    if R2d is None or Z2d is None:
        # Fallback: index-space plot
        ny, nx = F2d.shape
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
        R2d, Z2d = X, Y

    # Auto transpose grid if needed
    if R2d.shape != F2d.shape:
        if R2d.T.shape == F2d.shape:
            R2d = R2d.T
            Z2d = Z2d.T
        else:
            raise RuntimeError(f"Grid shape {R2d.shape} incompatible with field shape {F2d.shape}")

    return R2d, Z2d, F2d, title


def _print_info(mhd: Any, occ: int):
    n_ts = _aos_size(mhd.time_slice)
    times = []
    counts = []
    for i in range(n_ts):
        ts = mhd.time_slice[i]
        times.append(_ts_time(ts))
        counts.append(_mode_count(ts))

    finite = [t for t in times if math.isfinite(t) and t > -1e30]
    print(f"mhd_linear occ={occ}: time_slices={n_ts}")
    if finite:
        print(f"  time range (valid): [{min(finite):g}, {max(finite):g}]  (n_time_with_values={len(finite)})")
    else:
        print("  time range (valid): [none]")

    print("  time_slices:")
    for i,(t,c) in enumerate(zip(times, counts)):
        flags = []
        if (not math.isfinite(t)) or (t <= -1e30):
            flags.append("placeholder_time")
        if c == 0:
            flags.append("no_modes")
        else:
            flags.append("ok")
        print(f"    [{i:3d}] time={t:g}  n_modes={c:3d}  ({', '.join(flags)})")

    n_tor_union = set()
    for i in range(n_ts):
        ts = mhd.time_slice[i]
        nm = _mode_count(ts)
        for j in range(nm):
            try:
                v = int(ts.toroidal_mode[j].n_tor)
                if v != PLACEHOLDER_INT:
                    n_tor_union.add(v)
            except Exception:
                pass
    if n_tor_union:
        nt_sorted = sorted(n_tor_union)
        print("  n_tor union:", nt_sorted)
        if len(nt_sorted) == 1 and nt_sorted[0] == PLACEHOLDER_INT:
            print(
                "  NOTE: n_tor appears unset (placeholder). Use --mode-index to select a mode, "
                "or update dump2imas.py to populate toroidal_mode[].n_tor from the dump file."
            )


# --------------------------------- Main --------------------------------

def main():
    ap = argparse.ArgumentParser(description="Contour plot fields from IMAS mhd_linear IDS (RZ plane).")

    ap.add_argument("--backend", default="hdf5")
    ap.add_argument("--dbpath", default=".")
    ap.add_argument("--dd", required=True)
    ap.add_argument("--dd-version", dest="dd_version", required=True)
    ap.add_argument("--pulse", type=int, required=True)
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--occ", type=int, default=0)

    ap.add_argument("--time-index", type=int, default=0)
    ap.add_argument("--raw-time-index", action="store_true")
    ap.add_argument("--mode-index", type=int, default=None)
    ap.add_argument("--keff", type=int, default=None, help="alias for --n-tor")
    ap.add_argument("--n-tor", dest="n_tor", type=int, default=None)

    ap.add_argument("--info", action="store_true")
    ap.add_argument("--field", default=None, choices=["p", "t", "n", "b", "v"])
    ap.add_argument("--part", default="real", choices=["real", "imag", "amp"])
    ap.add_argument("--component", default=None, choices=["r", "z", "phi"])
    ap.add_argument("--levels", type=int, default=80)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default="X11", help="X11/show to display, or filename to save")

    args = ap.parse_args()

    if args.n_tor is None and args.keff is not None:
        args.n_tor = int(args.keff)

    root = os.path.abspath(args.dbpath)
    entry_dir = os.path.join(root, str(args.dd), str(args.dd_version), str(args.pulse), str(args.run))
    print(f"IMAS entry directory: {entry_dir}")

    db, uri, imas = _open_dbentry(args.backend, entry_dir, mode="r", dd_version=str(args.dd_version))
    print(f"IMAS URI: {uri}")

    factory = imas.IDSFactory(str(args.dd_version))
    mhd = _get_ids(db, factory, "mhd_linear", int(args.occ))

    if args.info:
        _print_info(mhd, int(args.occ))
        if args.field is None:
            return 0

    if args.field is None:
        raise RuntimeError("Provide --field (or use --info).")

    raw_i, ts, nonempty = _pick_time_slice(mhd, int(args.time_index), bool(args.raw_time_index))
    mi = _select_mode_index(ts, args.n_tor, args.mode_index)

    R2d, Z2d, F2d, title = _extract_rz_and_field(ts, mi, args.field, args.part, args.component)

    try:
        n_tor_val = int(ts.toroidal_mode[mi].n_tor)
    except Exception:
        n_tor_val = None

    if args.title is None:
        args.title = f"occ={args.occ} raw_ts={raw_i} time={_ts_time(ts):.6g} n_tor={n_tor_val} {title} ({args.part})"

    fig, ax = plt.subplots()
    cf = ax.contourf(R2d, Z2d, F2d, levels=int(args.levels))
    fig.colorbar(cf, ax=ax)
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(args.title)
    ax.set_aspect("equal", adjustable="box")

    out = str(args.out)
    if out.upper() == "X11" or out.lower() in ("show", "-"):
        plt.show()
    else:
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
