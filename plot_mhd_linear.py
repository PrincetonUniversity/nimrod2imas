#!/usr/bin/env python3
"""
plot_mhd_linear.py

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

from matplotlib.colors import LogNorm, SymLogNorm

from pathlib import Path

from nimrod2imas import (
    add_entry_args as _add_entry_args_common,
    resolve_entry_path as _resolve_entry_common,
    open_ids_h5 as _open_ids_h5_common,
    h5_list_keys as _h5_list_keys_common,
    open_dbentry as _open_db_common,
    ids_factory as _ids_factory_common,
    get_ids as _get_ids_common,
)



# Optional: cmasher colormaps (https://cmasher.readthedocs.io/)
# If installed, importing cmasher registers its colormaps with Matplotlib (names like 'cmr.gothic').
_HAS_CMASher = False
try:
    import cmasher as cmr  # type: ignore  # noqa: F401
    _HAS_CMASher = True
except Exception:
    cmr = None  # type: ignore
# IMAS integer placeholder commonly used for "not set" in many DDs
PLACEHOLDER_INT = -999999999


# Field aliases (user -> canonical) for --field
_Q_ALIASES = {
    'p': ['p', 'pressure'],
    't': ['t', 'temperature'],
    'n': ['n', 'mass_density'],
    'b': ['b', 'bfield', 'b_field'],
    'v': ['v', 'vel', 'velocity'],
}


def _canonical_field(field: str) -> str:
    f = (field or '').strip().lower()
    for canon, names in _Q_ALIASES.items():
        if f == canon or f in names:
            return canon
    return f


def _normalize_field_name(field: str) -> Optional[str]:
    """Return canonical short name (p,t,n,b,v) or None if unknown."""
    f = _canonical_field(field)
    return f if f in _Q_ALIASES else None


def _node_info(node: Any):
    # Best-effort presence/shape for complex nodes (real/imaginary).
    try:
        re_a = np.asarray(node.real)
        im_a = np.asarray(node.imaginary)
        present = (re_a.size > 0) or (im_a.size > 0)
        return present, f"real{tuple(re_a.shape)} imag{tuple(im_a.shape)}"
    except Exception:
        return False, '(unavailable)'


def _print_field_help(ts: Any, mi: int):
    tm = ts.toroidal_mode[mi]
    pl = tm.plasma

    print('')
    print('Available fields and aliases for mhd_linear:')
    print('  Scalars:')
    ok, s = _node_info(getattr(pl, 'pressure_perturbed', None))
    print(f"    - p / pressure           -> pressure_perturbed      : {'OK' if ok else 'MISSING'} ({s})")
    ok, s = _node_info(getattr(pl, 'temperature_perturbed', None))
    print(f"    - t / temperature        -> temperature_perturbed   : {'OK' if ok else 'MISSING'} ({s})")
    ok, s = _node_info(getattr(pl, 'mass_density_perturbed', None))
    print(f"    - n / mass_density       -> mass_density_perturbed  : {'OK' if ok else 'MISSING'} ({s})")

    print('  Vectors (require --component r|z|phi):')
    try:
        b = pl.b_field_perturbed
        ok1, s1 = _node_info(b.coordinate1)
        ok2, s2 = _node_info(b.coordinate2)
        ok3, s3 = _node_info(b.coordinate3)
        print(f"    - b / bfield             -> b_field_perturbed       : r={'OK' if ok1 else 'MISSING'} ({s1}), z={'OK' if ok2 else 'MISSING'} ({s2}), phi={'OK' if ok3 else 'MISSING'} ({s3})")
    except Exception:
        print('    - b / bfield             -> b_field_perturbed       : (unavailable)')

    try:
        v = pl.velocity_perturbed
        ok1, s1 = _node_info(v.coordinate1)
        ok2, s2 = _node_info(v.coordinate2)
        ok3, s3 = _node_info(v.coordinate3)
        print(f"    - v / vel / velocity     -> velocity_perturbed      : r={'OK' if ok1 else 'MISSING'} ({s1}), z={'OK' if ok2 else 'MISSING'} ({s2}), phi={'OK' if ok3 else 'MISSING'} ({s3})")
    except Exception:
        print('    - v / vel / velocity     -> velocity_perturbed      : (unavailable)')

    print('')
    print('Parts (--part): real | imag | amp')
    print('Normalization (--norm): linear | log | symlog')
    print('  - log requires strictly positive data (typical with --part amp).')
    print('  - symlog supports signed data; adjust --linthresh if needed.')
    print('')


def _leaf_value(x: Any) -> Any:
    """Return primitive value for IMAS leaf wrappers (IDS*0D), or x itself."""
    try:
        return x.value  # type: ignore[attr-defined]
    except Exception:
        return x


def _leaf_as_int(x: Any) -> Optional[int]:
    """Best-effort conversion of an IMAS leaf (possibly wrapped) to int."""
    try:
        x = _leaf_value(x)
        if x is None:
            return None
        # Some IMAS leaves come back as numpy scalars
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return int(round(float(x)))
        return int(x)
    except Exception:
        return None


def _leaf_as_float(x: Any) -> Optional[float]:
    """Best-effort conversion of an IMAS leaf (possibly wrapped) to float."""
    try:
        x = _leaf_value(x)
        if x is None:
            return None
        return float(x)
    except Exception:
        return None



# --------------------------- Plotting helpers ---------------------------


def _mode_number(tm: Any) -> Optional[int]:
    """Return toroidal mode number from a toroidal_mode entry.

    DD 4.x may use n_phi instead of n_tor. Fall back to identifier.index if needed.
    """
    for name in ("n_phi", "n_tor"):
        if hasattr(tm, name):
            v = _leaf_as_int(getattr(tm, name))
            if v is not None:
                return v
    # Fallback: identifier.index
    if hasattr(tm, "identifier"):
        try:
            v = _leaf_as_int(tm.identifier.index)
            if v is not None:
                return v
        except Exception:
            pass
    return None


def _resolve_time_index(mhd: Any, time_index: int) -> int:
    """Map a logical --time-index to a raw mhd.time_slice index.

    By default, plot_mhd_linear treats --time-index as indexing only the
    *non-empty* time_slices (those with at least one toroidal_mode entry).
    This avoids confusing gaps when the IDS contains placeholder time slices.

    If no non-empty slices exist, falls back to raw indexing.
    """
    try:
        nraw = len(mhd.time_slice)
    except Exception:
        nraw = 0

    if nraw <= 0:
        raise SystemExit("mhd_linear: no time_slice entries found")

    nonempty: List[int] = []
    for i in range(nraw):
        try:
            ts = mhd.time_slice[i]
            if hasattr(ts, "toroidal_mode") and len(ts.toroidal_mode) > 0:
                nonempty.append(i)
        except Exception:
            continue

    # If filtering produced nothing, use raw indexing.
    if not nonempty:
        it = int(time_index)
        if it < 0:
            it = nraw + it
        if it < 0 or it >= nraw:
            raise SystemExit(f"--time-index {time_index} out of range for raw time_slice (n={nraw})")
        return it

    it = int(time_index)
    if it < 0:
        it = len(nonempty) + it
    if it < 0 or it >= len(nonempty):
        raise SystemExit(
            f"--time-index {time_index} out of range for non-empty time_slices (n_nonempty={len(nonempty)}, n_raw={nraw}). "
            "Use --raw-time-index to index raw time_slice array."
        )
    return nonempty[it]

def _available_cmaps() -> List[str]:
    """Return available Matplotlib colormap names (sorted).

    If cmasher is installed, its colormaps are included automatically once imported.
    """
    try:
        return sorted(list(plt.colormaps()))
    except Exception:
        # Older Matplotlib (<3.6) fallback.
        try:
            return sorted(list(plt.cm.cmap_d.keys()))  # type: ignore[attr-defined]
        except Exception:
            return []


def _default_cmap_for_part(part: str) -> str:
    """Reasonable defaults: diverging for signed fields, sequential for amplitudes."""
    part = (part or "").lower()
    if part in ("real", "imag"):
        return "RdBu_r"
    return "viridis"


def _resolve_cmap(cmap: Optional[str], part: str) -> str:
    """Validate/resolve the colormap name, with helpful diagnostics."""
    if cmap is None or str(cmap).strip() == "":
        return _default_cmap_for_part(part)

    cmap = str(cmap).strip()

    # If user selected a cmasher map explicitly, ensure cmasher is available.
    if cmap.lower().startswith("cmr.") and not _HAS_CMASher:
        raise RuntimeError(
            f"Requested colormap {cmap!r}, but cmasher is not available in this Python environment. "
            "Install it (e.g. 'pip install cmasher') or choose a Matplotlib colormap."
        )

    avail = _available_cmaps()
    if avail and cmap not in avail:
        # Try a case-insensitive match as a convenience.
        lower_map = {c.lower(): c for c in avail}
        if cmap.lower() in lower_map:
            return lower_map[cmap.lower()]
        raise RuntimeError(
            f"Unknown colormap {cmap!r}. Use --list-cmaps to see available names."
        )
    return cmap



def _build_norm_and_levels(F2d: np.ndarray, nlevels: int, norm_kind: str = 'linear', linthresh: float = 1e-6):
    """Build (norm, levels) for contourf based on requested normalization.

    - linear: norm=None, levels=int(nlevels)
    - log: LogNorm with log-spaced levels (requires positive finite data)
    - symlog: SymLogNorm with integer levels (works with signed data)
    """
    norm_kind = (norm_kind or 'linear').lower()
    vv = np.asarray(F2d, dtype=float)
    vv = vv[np.isfinite(vv)]
    if vv.size == 0:
        return None, int(nlevels)

    if norm_kind in ('linear', 'none'):
        return None, int(nlevels)

    if norm_kind == 'log':
        pos = vv[vv > 0]
        if pos.size == 0:
            raise RuntimeError('Log normalization requires positive data. Use --part amp or --norm symlog for signed data.')
        vmin = float(pos.min())
        vmax = float(pos.max())
        if vmax <= vmin:
            vmax = vmin * 10.0
        levels = np.logspace(np.log10(vmin), np.log10(vmax), int(nlevels))
        return LogNorm(vmin=vmin, vmax=vmax), levels

    if norm_kind == 'symlog':
        absmax = float(np.max(np.abs(vv)))
        if absmax == 0.0:
            absmax = 1.0
        lt = float(linthresh) if linthresh and linthresh > 0 else absmax * 1e-3
        return SymLogNorm(linthresh=lt, vmin=-absmax, vmax=absmax), int(nlevels)

    raise RuntimeError(f'Unknown --norm {norm_kind!r} (use linear|log|symlog).')

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
            nvals.append((_mode_number(ts.toroidal_mode[i])))
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

    field = _canonical_field(field)
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
                v = _mode_number(ts.toroidal_mode[j])
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



# ----------------- plotting helpers -----------------

def _list_cmaps() -> None:
    import matplotlib.pyplot as plt

    cmaps = sorted(list(plt.colormaps()))
    for c in cmaps:
        print(c)


def _plot_contour(
    R,
    Z,
    F,
    *,
    title: str,
    cmap: str,
    levels: int,
    norm: str | None,
    linthresh: float,
    out: str,
    show: bool,
    dpi: int,
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, SymLogNorm

    fig, ax = plt.subplots(figsize=(7, 6))

    F_plot = F
    mnorm = None

    if norm == "log":
        # Mask non-positive values for log scaling
        F_plot = np.ma.masked_where(np.asarray(F) <= 0, F)
        vmin = float(np.nanmin(F_plot))
        vmax = float(np.nanmax(F_plot))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0 or vmax <= 0:
            raise SystemExit("--norm log requires positive data")
        mnorm = LogNorm(vmin=vmin, vmax=vmax)

    elif norm == "symlog":
        vmax = float(np.nanmax(np.abs(F_plot)))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        mnorm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax)

    cf = ax.contourf(R, Z, F_plot, levels=levels, cmap=cmap, norm=mnorm)
    fig.colorbar(cf, ax=ax)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title)

    if show or out == "X11":
        plt.show()
    else:
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(
        description="Contour plot fields from IMAS mhd_linear IDS (RZ plane)."
    )

    _add_entry_args_common(
        ap,
        include_backend=True,
        backend_default="hdf5",
        include_ids=True,
        ids_default="mhd_linear",
        ids_choices=["mhd_linear"],
        include_occ=True,
        occ_default=0,
    )

    ap.add_argument("--time-index", type=int, default=0, help="Time slice index")
    ap.add_argument(
        "--raw-time-index",
        action="store_true",
        help="Use --time-index without filtering empty time_slices",
    )

    ap.add_argument(
        "--mode-index",
        type=int,
        default=None,
        help="Toroidal mode index within time_slice (default: 0)",
    )
    ap.add_argument(
        "--n-tor",
        type=int,
        default=None,
        help="Select toroidal mode by n_tor (overrides --mode-index if provided)",
    )

    ap.add_argument(
        "--quantity",
        default=None,
        help=(
            "Field to plot. Canonical: p,t,n,b,v. Aliases include: "
            "p->pressure_perturbed, t->temperature_perturbed, n->density_perturbed, "
            "b->magnetic_field_perturbed, v->velocity_perturbed"
        ),
    )
    ap.add_argument(
        "--part",
        default="real",
        choices=["real", "imag", "amp"],
        help="Use real/imag component or amplitude",
    )
    ap.add_argument("--component", default=None, choices=["r", "z", "phi"], help="Vector component")

    ap.add_argument("--norm", default=None, choices=["log", "symlog"], help="Optional color normalization")
    ap.add_argument(
        "--linthresh",
        type=float,
        default=1e-6,
        help="Symlog linear threshold (only for --norm symlog)",
    )
    ap.add_argument("--levels", type=int, default=50, help="Number of contour levels")

    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap name")
    ap.add_argument("--list-cmaps", action="store_true", help="List available Matplotlib colormaps and exit")

    ap.add_argument("--title", default=None, help="Override plot title")
    ap.add_argument("--out", default="X11", help="Output image path or 'X11' for interactive")
    ap.add_argument("--show", action="store_true", help="Show plot interactively (even if --out is set)")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI when saving")

    ap.add_argument("--info", action="store_true", help="Print a short summary of the entry contents")
    ap.add_argument(
        "--help-quantities",
        action="store_true",
        help="Print available datasets/aliases for this entry and exit",
    )

    args = ap.parse_args()

    if args.list_cmaps:
        _list_cmaps()
        return 0

    if args.dd_version is None:
        raise SystemExit("--dd-version is required (e.g. 4.1.1)")

    entry_dir = str(_resolve_entry_common(args))
    print(f"IMAS entry directory: {entry_dir}")

    db, uri, imas = _open_db_common(args.backend, entry_dir, mode="r", dd_version=str(args.dd_version))
    try:
        mhd = _get_ids_common(db, _ids_factory_common(imas, str(args.dd_version)), "mhd_linear", occ=args.occ)
        if args.help_quantities:
            # Use a representative (time_slice, toroidal_mode) to report presence/shape.
            try:
                it0 = _resolve_time_index(mhd, 0)
            except Exception:
                it0 = 0
            ts0 = mhd.time_slice[it0]
            mi0 = _select_mode_index(ts0, mode_index=0, n_tor=None)
            _print_field_help(ts0, mi0)
            try:
                f_h5, g_h5, _h5p, grp = _open_ids_h5_common(entry_dir, "mhd_linear", args.occ)
                try:
                    keys = _h5_list_keys_common(g_h5, exclude_shape=True)
                finally:
                    f_h5.close()

                if keys:
                    print("")
                    print(f"HDF5 keys under /{grp}:")
                    ts_keys = [k for k in keys if k.startswith("time_slice[]&")]
                    other = [k for k in keys if not k.startswith("time_slice[]&")]
                    for k in ts_keys[:200]:
                        print("  - " + k)
                    if len(ts_keys) > 200:
                        print(f"  ... ({len(ts_keys)-200} more time_slice keys omitted)")
                    for k in other[:50]:
                        print("  - " + k)
                    if len(other) > 50:
                        print(f"  ... ({len(other)-50} more non-time_slice keys omitted)")
            except Exception as e:
                print(f"Could not list HDF5 keys: {e}")
            return 0

        if args.info:
            _print_info(mhd)

        if not args.quantity:
            raise SystemExit("No --quantity provided. Use --help-quantities to see options.")

        # Resolve time index (optionally skipping empty time_slices)
        if args.raw_time_index:
            it = int(args.time_index)
        else:
            it = _resolve_time_index(mhd, int(args.time_index))

        ts = mhd.time_slice[it]
        im = _select_mode_index(ts, mode_index=args.mode_index, n_tor=args.n_tor)
        mode = ts.toroidal_mode[im]

        field = _normalize_field_name(str(args.quantity))
        if field is None:
            raise SystemExit(f"Unknown quantity '{args.quantity}'. Use --help-quantities.")

        if field in ("b", "v") and args.component is None:
            raise SystemExit(f"Quantity '{field}' requires --component (r|z|phi).")

        R, Z, F = _extract_rz_and_field(mode, field, part=args.part, component=args.component)

        title = args.title
        if title is None:
            n = getattr(mode, "n_tor", None)
            title = f"mhd_linear: {field} ({args.part})"
            if args.component:
                title += f" {args.component}"
            if n is not None:
                title += f", n_tor={n}"
            if hasattr(ts, "time"):
                try:
                    title += f", t={float(ts.time):.6g}"
                except Exception:
                    pass

        _plot_contour(
            R,
            Z,
            F,
            title=title,
            cmap=args.cmap,
            levels=int(args.levels),
            norm=args.norm,
            linthresh=float(args.linthresh),
            out=args.out,
            show=args.show,
            dpi=int(args.dpi),
        )

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
