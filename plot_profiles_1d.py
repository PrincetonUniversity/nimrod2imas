#!/usr/bin/env python3
"""
plot_profiles_1d.py

Plot 1D profiles from IMAS core_profiles and edge_profiles IDSs written by the NIMROD->IMAS tools.

Examples
  # Core profiles (electron temperature and density)
  python plot_profiles_1d.py --entry /path/to/entry/ --ids core_profiles --occ 0 --time-index 0 --quantities te,ne

  # Edge profiles (total pressure + toroidal current density)
  python plot_profiles_1d.py --entry /path/to/entry/ --ids edge_profiles --occ 0 --time-index 0 --quantities ptot,jtor

  # Force HDF5 mode (bypass IMAS)
  python plot_profiles_1d.py --entry /path/to/entry/ --ids edge_profiles --occ 0 --time-index 0 --quantities te --hdf5-only

Notes
- The script is intentionally defensive w.r.t. DD leaf names. For each requested quantity it tries a small
  set of plausible HDF5 dataset paths and falls back to IMAS access if available.
- X-axis rules:
  • core_profiles: plot vs normalized toroidal flux (rho_tor_norm; if only psi_tor_norm present, uses sqrt(psi_tor_norm)).
  • edge_profiles: plot vs normalized poloidal flux (psi_pol_norm aka psi_norm), including SOL/PF (psi_norm > 1 if present).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nimrod2imas import entry_dir as _entry_dir_common

try:
    import imas
except Exception:
    imas = None

try:
    import h5py
except Exception as e:
    raise SystemExit(f"h5py is required for this script: {e}")

__version__ = "0.3.0"

# ----------------------------
# HDF5 dataset candidates
# ----------------------------

# Coordinate (x-axis) candidates under the IDS group.
# NOTE: core_profiles and edge_profiles use different coordinates by design.
_H5_X_CORE_CANDIDATES = [
    # Preferred: normalized toroidal flux radius
    "profiles_1d[]&grid&rho_tor_norm",
    "profiles_1d[]&grid&rho_tor_norm[]",
    "profiles_1d[]&grid&rho_tor",
    # Accept: normalized toroidal flux (squared)
    "profiles_1d[]&grid&psi_tor_norm",
    "profiles_1d[]&grid&psi_tor_norm[]",
    # Fallbacks
    "profiles_1d[]&grid&psi_norm",
    "profiles_1d[]&grid&psi_pol_norm",
    "profiles_1d[]&grid&psi_pol_norm[]",
    "profiles_1d[]&grid&psi",
    "profiles_1d[]&grid&psi_pol",
]

_H5_X_EDGE_CANDIDATES = [
    # Preferred: normalized poloidal flux
    "profiles_1d[]&grid&psi_norm",
    "profiles_1d[]&grid&psi_pol_norm",
    "profiles_1d[]&grid&psi_pol_norm[]",
    # Absolute psi fallback
    "profiles_1d[]&grid&psi",
    "profiles_1d[]&grid&psi_pol",
    # Last resort (should not normally be needed for edge_profiles)
    "profiles_1d[]&grid&rho_tor_norm",
    "profiles_1d[]&grid&psi_tor_norm",
]

# Quantity mapping: user keyword -> list of candidate datasets
_H5_Q_DS: Dict[str, List[str]] = {
    # electrons
    "te": ["profiles_1d[]&electrons&temperature", "profiles_1d[]&electrons&t_e", "profiles_1d[]&electrons&temp"],
    "ne": ["profiles_1d[]&electrons&density", "profiles_1d[]&electrons&density_thermal"],
    "pe": ["profiles_1d[]&electrons&pressure", "profiles_1d[]&electrons&p"],
    # ions (main-ion / averaged)
    "ti": ["profiles_1d[]&ion[]&temperature", "profiles_1d[]&ion[]&t_i", "profiles_1d[]&ion[]&temp"],
    "ni": ["profiles_1d[]&ion[]&density"],
    "pi": ["profiles_1d[]&ion[]&pressure", "profiles_1d[]&ion[]&p"],
    "vtor": ["profiles_1d[]&ion[]&velocity&toroidal"],
    "vpol": ["profiles_1d[]&ion[]&velocity&poloidal"],
    # totals
    "ptot": ["profiles_1d[]&pressure_thermal", "profiles_1d[]&pressure_perpendicular", "profiles_1d[]&pressure_parallel"],
    "pressure": ["profiles_1d[]&pressure", "profiles_1d[]&pressure_thermal"],
    "jtor": ["profiles_1d[]&j_tor", "profiles_1d[]&jtor", "profiles_1d[]&j_phi"],
    "omega": ["profiles_1d[]&ion[]&rotation_frequency_tor_s"],
}

# Quantity pretty labels (aliases) and units for plotting
_Q_META: Dict[str, Dict[str, str]] = {
    "te": {"alias": r"$T_e$", "unit": "eV"},
    "ne": {"alias": r"$n_e$", "unit": r"m$^{-3}$"},
    "pe": {"alias": r"$p_e$", "unit": "Pa"},
    "ti": {"alias": r"$T_i$", "unit": "eV"},
    "ni": {"alias": r"$n_i$", "unit": r"m$^{-3}$"},
    "pi": {"alias": r"$p_i$", "unit": "Pa"},
    "ptot": {"alias": r"$p$", "unit": "Pa"},
    "vtor": {"alias": r"$v_{tor}$", "unit": "m/s"},
    "vpol": {"alias": r"$v_{pol}$", "unit": "m/s"},
    "pressure": {"alias": r"$p$", "unit": "Pa"},
    "jtor": {"alias": r"$j_\phi$", "unit": r"A m$^{-2}$"},
    "omega": {"alias": r"$\omega_{\mathrm{tor}}$", "unit": r"s$^{-1}$"},
}

def _q_pretty(q: str, with_unit: bool = True) -> str:
    qk = q.strip().lower()
    if qk in _Q_META:
        a = _Q_META[qk]["alias"]
        u = _Q_META[qk]["unit"]
        return f"{a} [{u}]" if with_unit and u else a
    return qk



def _infer_ids_h5_path(entry: str, ids_name: str, occ: int) -> str:
    """Infer IDS HDF5 filename for an entry.

    Note: Many IMAS HDF5 backends store occurrence 0 without an "_0" suffix:
      - file: <ids>.h5  group: /<ids>
    while occurrences >0 typically use:
      - file: <ids>_<occ>.h5 group: /<ids>_<occ>

    We therefore try a small ordered set of candidates and return the first that exists.
    """
    entry = entry.rstrip("/")
    ids_name = str(ids_name).strip()

    cands = []
    if int(occ) == 0:
        cands += [
            os.path.join(entry, f"{ids_name}.h5"),
            os.path.join(entry, f"{ids_name}_0.h5"),
            os.path.join(entry, f"{ids_name}_{occ}.h5"),
        ]
    else:
        cands += [
            os.path.join(entry, f"{ids_name}_{occ}.h5"),
            os.path.join(entry, f"{ids_name}.h5"),
        ]

    for fp in cands:
        if os.path.exists(fp):
            return fp

    # Fall back to the conventional name to produce a clear error message upstream.
    return os.path.join(entry, f"{ids_name}_{occ}.h5")


def _h5_open_group(h5_path: str, ids_name: str, occ: int) -> Tuple[h5py.File, h5py.Group]:
    f = h5py.File(h5_path, "r")

    ids_name = str(ids_name).strip()
    occ = int(occ)

    # Occurrence 0 is often stored without a suffix.
    if occ == 0:
        group_candidates = [ids_name, f"{ids_name}_0", f"{ids_name}_{occ}"]
    else:
        group_candidates = [f"{ids_name}_{occ}", ids_name]

    for grp_name in group_candidates:
        if grp_name in f:
            return f, f[grp_name]

    f.close()
    raise RuntimeError(
        f"Group not found in {h5_path}. Tried: " + ", ".join([f"/{g}" for g in group_candidates])
    )


def _h5_get_first_existing(g: h5py.Group, names: Sequence[str]) -> Optional[str]:
    for nm in names:
        if nm in g:
            return nm
    return None


def _h5_read_1d(g: h5py.Group, ds_name: str, time_index: int, ion_index: int = 0) -> np.ndarray:
    ds = g[ds_name]
    arr = np.asarray(ds)
    # Common layouts:
    #   (ntime, npts)                            -> scalar
    #   (ntime, 1, npts)                         -> scalar with singleton axis
    #   (ntime, nion, npts) or (ntime, nion, 1, npts) -> ion AoS
    if arr.ndim == 1:
        if time_index != 0:
            raise RuntimeError(f"{ds_name} has only one time slice; use --time-index 0")
        return arr
    if arr.ndim == 2:
        if time_index >= arr.shape[0]:
            raise RuntimeError(f"time_index {time_index} out of range (ntime={arr.shape[0]}) for {ds_name}")
        return arr[time_index, :]
    if arr.ndim == 3:
        if time_index >= arr.shape[0]:
            raise RuntimeError(f"time_index {time_index} out of range (ntime={arr.shape[0]}) for {ds_name}")
        # Heuristic: if second axis is small, treat as ion index; else treat as singleton axis.
        if arr.shape[1] <= 16 and arr.shape[2] > 16:
            if ion_index >= arr.shape[1]:
                raise RuntimeError(f"ion_index {ion_index} out of range (nion={arr.shape[1]}) for {ds_name}")
            return arr[time_index, ion_index, :]
        # singleton axis
        if arr.shape[1] == 1:
            return arr[time_index, 0, :]
        # fallback
        return arr[time_index, 0, :]
    if arr.ndim == 4:
        if time_index >= arr.shape[0]:
            raise RuntimeError(f"time_index {time_index} out of range (ntime={arr.shape[0]}) for {ds_name}")
        # assume (ntime,nion,1,npts) or (ntime,nion,npts,1)
        if ion_index >= arr.shape[1]:
            raise RuntimeError(f"ion_index {ion_index} out of range (nion={arr.shape[1]}) for {ds_name}")
        a = arr[time_index, ion_index, ...]
        a = np.asarray(a)
        return a.reshape(-1)
    raise RuntimeError(f"Unsupported dataset rank for {ds_name}: {arr.shape}")




def _h5_list_available_profiles(g: h5py.Group) -> List[str]:
    """List available profile datasets under the current IDS group.

    We return dataset names (relative to the group) that start with 'profiles_1d[]&'
    and exclude bookkeeping datasets such as '*_SHAPE' and 'AOS_SHAPE'.
    """
    out: List[str] = []
    for k in g.keys():
        if not isinstance(k, str):
            continue
        if not k.startswith("profiles_1d[]&"):
            continue
        if k.endswith("_SHAPE") or k.endswith("AOS_SHAPE"):
            continue
        out.append(k)
    return sorted(out)


def _print_quantity_help(ids_name: str, occ: int, time_index: int, entry: str, g: h5py.Group) -> None:
    print("")
    print(f"Available 1D profile datasets for ids='{ids_name}', occ={occ}, time_index={time_index}:")
    avail = _h5_list_available_profiles(g)
    if not avail:
        print("  (none found under profiles_1d[]&... in this IDS group)")
    else:
        for k in avail:
            print(f"  - {k}")
    print("")
    print("How to plot:")
    print("  - Use --quantities with one or more aliases (comma-separated), e.g.:")
    print("      --quantities ne,te,ni,ti,ptot,jtor")
    print("  - Aliases supported by this script:")
    print(f"      {', '.join(sorted(_H5_Q_DS.keys()))}")
    print("")
    print("Notes:")
    print("  - For total pressure, the preferred storage is profiles_1d[]&pressure_thermal (or ...&pressure).")
    print("  - If only ...&pressure_perpendicular exists (legacy), 'ptot' will be plotted as 3*p_perp.")
    print("")
def _choose_x_dataset(ids_name: str, g: h5py.Group) -> Optional[str]:
    ids_name = str(ids_name).strip().lower()
    cand = _H5_X_CORE_CANDIDATES if ids_name == "core_profiles" else _H5_X_EDGE_CANDIDATES
    return _h5_get_first_existing(g, cand)


def _read_x_and_label(
    g: h5py.Group, ids_name: str, ds_name: str, time_index: int, ion_index: int = 0
) -> Tuple[np.ndarray, str, str]:
    '''
    Returns (x, x_label, x_kind).
    For core_profiles, if only psi_tor_norm is available, plots rho_tor_norm = sqrt(psi_tor_norm).
    '''
    x_raw = _h5_read_1d(g, ds_name, time_index=time_index, ion_index=ion_index)
    ds = ds_name.lower()

    if "rho_tor_norm" in ds:
        return np.asarray(x_raw, dtype=float), r"$\rho_{\mathrm{tor,norm}}$", "rho_tor_norm"

    if "psi_tor_norm" in ds:
        xr = np.asarray(x_raw, dtype=float)
        x = np.sqrt(np.clip(xr, 0.0, None))
        return x, r"$\rho_{\mathrm{tor,norm}}$ (from $\psi_{\mathrm{tor,norm}}$)", "psi_tor_norm->rho_tor_norm"

    if "psi_pol_norm" in ds or "grid&psi_norm" in ds:
        return np.asarray(x_raw, dtype=float), r"$\psi_{\mathrm{pol,norm}}$", "psi_pol_norm"

    if "grid&psi" in ds or "psi_pol" in ds:
        return np.asarray(x_raw, dtype=float), r"$\psi$ (Wb/rad)", "psi_abs"

    return np.asarray(x_raw, dtype=float), "x", "unknown"



def _find_equilibrium_file_and_group(entry: str) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort locate equilibrium HDF5 file/group in an IMAS entry."""
    entry = str(Path(entry).expanduser().resolve())
    candidates = [
        os.path.join(entry, "equilibrium_0.h5"),
        os.path.join(entry, "equilibrium.h5"),
        os.path.join(entry, "equilibrium_1.h5"),
    ]
    for fp in candidates:
        if not os.path.exists(fp):
            continue
        try:
            with h5py.File(fp, "r") as f:
                # Common group names
                for gn in ("equilibrium_0", "equilibrium", "equilibrium_1"):
                    if gn in f:
                        return fp, gn
        except Exception:
            continue
    return None, None


def _read_ids_time_from_h5(entry: str, ids_name: str, occ: int) -> Tuple[Optional[np.ndarray], str]:
    """Best-effort read IDS time array from <ids>_<occ>.h5."""
    fp = _infer_ids_h5_path(entry, ids_name, occ)
    if not os.path.exists(fp):
        return None, "time:h5_not_found"
    try:
        with h5py.File(fp, "r") as f:
            gn = None
            # occurrence 0 may use group "/<ids>" (no suffix)
            if int(occ) == 0:
                gnames = [str(ids_name), f"{ids_name}_0", f"{ids_name}_{occ}"]
            else:
                gnames = [f"{ids_name}_{occ}", str(ids_name)]
            for cand in gnames:
                if cand in f:
                    gn = cand
                    break
            if gn is None:
                return None, "time:group_not_found"
            g = f[gn]
            for nm in ("time", "profiles_1d[]&time", "time_slice[]&time"):
                if nm in g:
                    t = np.asarray(g[nm]).reshape(-1)
                    return t, f"time:{Path(fp).name}:{nm}"
    except Exception:
        return None, "time:read_failed"
    return None, "time:not_found"


def _read_core_profiles_psi_axis_lcfs(entry: str, time_index: int, target_time: Optional[float] = None, occ_hint: Optional[int] = None) -> Tuple[Optional[float], Optional[float], str]:
    """Infer (psi_axis, psi_lcfs) from core_profiles grid psi at rho_tor_norm~0 and ~1."""
    # Prefer matching occurrence if provided (edge occ), else try 0 then 1.
    occs = []
    if occ_hint is not None:
        occs.append(int(occ_hint))
    occs += [0, 1]
    tried = []
    for occ in occs:
        fp = _infer_ids_h5_path(entry, "core_profiles", occ)
        if not os.path.exists(fp):
            tried.append(f"core_profiles_{occ}.h5:missing")
            continue
        try:
            with h5py.File(fp, "r") as f:
                gn = f"core_profiles_{occ}"
                if gn not in f:
                    tried.append(f"{Path(fp).name}:group_missing")
                    continue
                g = f[gn]
                # read time
                ti = int(time_index)
                tarr = None
                for tnm in ("time", "profiles_1d[]&time"):
                    if tnm in g:
                        tarr = np.asarray(g[tnm]).reshape(-1)
                        break
                if target_time is not None and tarr is not None and tarr.size > 0:
                    ti = int(np.nanargmin(np.abs(tarr - float(target_time))))

                # grid psi abs
                psi_ds = _h5_get_first_existing(g, ["profiles_1d[]&grid&psi", "profiles_1d[]&grid&psi_pol"])
                if psi_ds is None:
                    tried.append(f"{Path(fp).name}:psi_missing")
                    continue
                psi = _h5_read_1d(g, psi_ds, ti)
                psi = np.asarray(psi, dtype=float).reshape(-1)

                # grid rho_tor_norm (preferred) or psi_tor_norm -> rho
                rho_ds = _h5_get_first_existing(g, ["profiles_1d[]&grid&rho_tor_norm", "profiles_1d[]&grid&psi_tor_norm"])
                if rho_ds is None:
                    # fall back: use index ordering
                    m = np.isfinite(psi)
                    if np.count_nonzero(m) < 2:
                        tried.append(f"{Path(fp).name}:insufficient_finite")
                        continue
                    psi_axis = float(psi[m][0])
                    psi_lcfs = float(psi[m][-1])
                    return psi_axis, psi_lcfs, f"core_profiles:{Path(fp).name}:{gn}:ti={ti}:rho_missing"
                rho_raw = _h5_read_1d(g, rho_ds, ti)
                rho_raw = np.asarray(rho_raw, dtype=float).reshape(-1)
                if "psi_tor_norm" in rho_ds.lower():
                    rho = np.sqrt(np.clip(rho_raw, 0.0, None))
                else:
                    rho = rho_raw

                m = np.isfinite(psi) & np.isfinite(rho)
                if np.count_nonzero(m) < 5:
                    tried.append(f"{Path(fp).name}:insufficient_finite")
                    continue
                psi_m = psi[m]
                rho_m = rho[m]
                ia = int(np.nanargmin(rho_m))
                il = int(np.nanargmax(rho_m))
                psi_axis = float(psi_m[ia])
                psi_lcfs = float(psi_m[il])
                return psi_axis, psi_lcfs, f"core_profiles:{Path(fp).name}:{gn}:ti={ti}:psi={psi_ds}:rho={rho_ds}"
        except Exception as exc:
            tried.append(f"{Path(fp).name}:read_failed:{exc}")
            continue
    return None, None, "core_profiles:not_found_or_unusable:" + ";".join(tried)

def _read_equilibrium_psi_axis_lcfs(entry: str, time_index: int, target_time: Optional[float] = None) -> Tuple[Optional[float], Optional[float], str]:
    """Return (psi_axis, psi_lcfs, source_tag) from equilibrium global_quantities."""
    fp, gn = _find_equilibrium_file_and_group(entry)
    if fp is None or gn is None:
        return None, None, "equilibrium:not_found"
    try:
        with h5py.File(fp, "r") as f:
            g = f[gn]
            ds_a = "time_slice[]&global_quantities&psi_axis"
            # Prefer explicit LCFS/separatrix leaves if present; fall back to psi_boundary.
            ds_lcfs_candidates = [
                "time_slice[]&global_quantities&psi_lcfs",
                "time_slice[]&global_quantities&psi_sep",
                "time_slice[]&global_quantities&psi_boundary",
            ]
            ds_l = None
            for nm in ds_lcfs_candidates:
                if nm in g:
                    ds_l = nm
                    break
            if ds_a not in g or ds_l is None:
                return None, None, "equilibrium:missing_datasets"

            a = np.asarray(g[ds_a]).reshape(-1)
            l = np.asarray(g[ds_l]).reshape(-1)

            if a.size == 0 or l.size == 0:
                return None, None, "equilibrium:empty"

            # Choose equilibrium time slice index
            ti = int(time_index)
            # Try match by time if target_time is provided
            if target_time is not None:
                tarr = None
                for tnm in ("time", "time_slice[]&time"):
                    if tnm in g:
                        tarr = np.asarray(g[tnm]).reshape(-1)
                        break
                if tarr is not None and tarr.size > 0:
                    ti = int(np.nanargmin(np.abs(tarr - float(target_time))))
            if ti >= a.size:
                ti = a.size - 1
            if ti >= l.size:
                ti = l.size - 1
            return float(a[ti]), float(l[ti]), f"equilibrium:{Path(fp).name}:{gn}:ti={ti}:lcfs={Path(ds_l).name if isinstance(ds_l,str) else ds_l}"
    except Exception:
        return None, None, "equilibrium:read_failed"
def _edge_psi_abs_to_psi_pol_norm(entry: str, psi_abs: np.ndarray, time_index: int, target_time: Optional[float] = None) -> Tuple[np.ndarray, str, str, float, float, str]:
    """Normalize absolute psi to psi_pol_norm using equilibrium psi_axis/psi_lcfs if possible."""
    log = None
    try:
        import logging as _logging
        log = _logging.getLogger(__name__)
    except Exception:
        log = None

    psi_abs = np.asarray(psi_abs, dtype=float)
    # Prefer normalization that guarantees LCFS at 1 using core_profiles grid if available
    psi_axis, psi_lcfs, src = _read_core_profiles_psi_axis_lcfs(entry, time_index, target_time=target_time, occ_hint=None)
    if psi_axis is None or psi_lcfs is None or not (np.isfinite(psi_axis) and np.isfinite(psi_lcfs)):
        psi_axis, psi_lcfs, src = _read_equilibrium_psi_axis_lcfs(entry, time_index, target_time=target_time)
    if psi_axis is None or psi_lcfs is None or not (np.isfinite(psi_axis) and np.isfinite(psi_lcfs)):
        # Robust fallback from psi itself
        vv = psi_abs[np.isfinite(psi_abs)]
        if vv.size >= 10:
            lo = float(np.nanquantile(vv, 0.01))
            hi = float(np.nanquantile(vv, 0.99))
        elif vv.size > 0:
            lo = float(np.nanmin(vv))
            hi = float(np.nanmax(vv))
        else:
            lo, hi = 0.0, 1.0

        # Heuristic: assume axis corresponds to the more "core-like" extremum.
        # Use the extremum with smaller |d(psi)| span on the provided array as axis; still imperfect but stable.
        psi_axis = lo
        psi_lcfs = hi if abs(hi - lo) > 1e-30 else (lo + 1.0)
        src = f"psi_quantile_fallback:axis={psi_axis:+.3e}:lcfs={psi_lcfs:+.3e}"
        if log:
            log.warning("edge_profiles: psi_norm not found; normalizing absolute psi using %s", src)

    den = float(psi_lcfs - psi_axis)
    if not np.isfinite(den) or abs(den) < 1e-30:
        den = 1.0

    psi_norm = (psi_abs - float(psi_axis)) / den
    # Ensure axis->0 and lcfs->1 by construction; allow psi_norm>1 in SOL/PF.
    return psi_norm, r"$\psi_{\mathrm{pol,norm}}$", f"psi_abs->psi_pol_norm:{src}", float(psi_axis), float(psi_lcfs), str(src)
def _try_imas_read(entry: str, dd_version: str, ids_name: str, occ: int):
    if imas is None:
        raise RuntimeError("IMAS-Python is not available in this environment.")
    factory = imas.IDSFactory(version=dd_version)
    db = imas.DBEntry(backend="hdf5", path=entry, mode="r", ids_factory=factory)
    try:
        ids = db.get(ids_name, occ)
    finally:
        db.close()
    return ids



def _plot(
    x: np.ndarray,
    ys: Dict[str, np.ndarray],
    title: str,
    x_label: str,
    y_label: Optional[str] = None,
    legend_labels: Optional[Dict[str, str]] = None,
    out: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    vline1: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    for k, y in ys.items():
        lab = legend_labels.get(k, k) if legend_labels else k
        ax.plot(x, y, label=lab)

    ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel("Value")
    ax.set_title(title)

    if xlim is not None:
        try:
            ax.set_xlim(xlim[0], xlim[1])
        except Exception:
            pass

    if vline1:
        try:
            ax.axvline(1.0, linestyle="--", linewidth=1.0, alpha=0.5)
        except Exception:
            pass

    if legend_labels or len(ys) > 1:
        ax.legend()

    ax.grid(True, which="both", alpha=0.3)

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    else:
        plt.show()
def main() -> int:
    p = argparse.ArgumentParser(description="Plot profiles_1d from core_profiles / edge_profiles")
    p.add_argument("--entry", default=None, help="IMAS entry directory (contains master.h5, core_profiles_*.h5, etc)")
    p.add_argument("--dbpath", default=".", help="DB root path")
    p.add_argument("--dd", default=None, help="DB name (directory name), e.g. nstx")
    p.add_argument("--pulse", type=int, default=None)
    p.add_argument("--run", type=int, default=None)
    p.add_argument("--dd-version", default=None, help="IMAS DD version for IMAS read (required unless --hdf5-only)")
    p.add_argument("--ids", choices=["core_profiles", "edge_profiles"], required=True)
    p.add_argument("--occ", type=int, required=True, help="Occurrence number")
    p.add_argument("--time-index", type=int, default=0, help="Time slice index")
    p.add_argument("--ion-index", type=int, default=0, help="Ion species index for ni/ti/pi/omega when applicable")
    p.add_argument("--quantities", required=True, help="Comma-separated list (e.g. te,ne,ptot,jtor)")
    p.add_argument("--hdf5-only", action="store_true", help="Force direct HDF5 mode (bypass IMAS)")
    p.add_argument("--output", default=None, help="Output image path (if omitted, show interactively)")
    p.add_argument("--info", action="store_true", help="Print resolved dataset paths/shapes and exit")

    args = p.parse_args()

    if args.entry:
        entry = str(Path(args.entry).expanduser().resolve()).rstrip("/") + "/"
    else:
        if args.dd is None or args.pulse is None or args.run is None or not args.dd_version:
            raise SystemExit("Provide either --entry, or (--dbpath --dd --dd-version --pulse --run).")
        entry = str(
            _entry_dir_common(
                args.dbpath,
                str(args.dd),
                str(args.dd_version),
                int(args.pulse),
                int(args.run),
                dd_version_dir=str(args.dd_version[0]),
            )
        ).rstrip("/") + "/"

    qlist = [q.strip().lower() for q in str(args.quantities).split(",") if q.strip()]
    if not qlist:
        raise SystemExit("No quantities requested.")

    # Try IMAS first (optional)
    ids_obj = None
    if not args.hdf5_only:
        if not args.dd_version:
            raise SystemExit("--dd-version is required unless --hdf5-only is used")
        try:
            ids_obj = _try_imas_read(entry, args.dd_version, args.ids, args.occ)
        except Exception:
            ids_obj = None

    # Always support direct HDF5 read
    h5_path = _infer_ids_h5_path(entry, args.ids, args.occ)
    if not os.path.exists(h5_path):
        raise SystemExit(f"Cannot find {h5_path}")

    f, g = _h5_open_group(h5_path, args.ids, args.occ)
    # Read IDS time array (if available) to align equilibrium time slices for normalization
    t_arr, t_src = _read_ids_time_from_h5(entry, args.ids, args.occ)
    target_time = None
    if t_arr is not None and t_arr.size > 0:
        ti = int(args.time_index)
        if ti >= t_arr.size:
            ti = t_arr.size - 1
        target_time = float(t_arr[ti])
    try:
        x_ds = _choose_x_dataset(args.ids, g)
        if x_ds is None:
            raise RuntimeError(
                "Could not locate x-axis dataset under profiles_1d[]&grid. "
                "For core_profiles expected rho_tor_norm/psi_tor_norm; for edge_profiles expected psi_norm/psi_pol_norm."
            )
        x, x_label, x_kind = _read_x_and_label(g, args.ids, x_ds, args.time_index, ion_index=args.ion_index)

        psi_axis_abs = None
        psi_lcfs_abs = None
        psi_src = None


        # Enforce edge_profiles x-axis: normalized poloidal flux.
        if str(args.ids).strip().lower() == "edge_profiles" and x_kind == "psi_abs":
            x, x_label, x_kind, psi_axis_abs, psi_lcfs_abs, psi_src = _edge_psi_abs_to_psi_pol_norm(entry, x, args.time_index, target_time=target_time)

        # Resolve absolute psi at magnetic axis and LCFS for reporting (and for sanity checks).
        ids_lc = str(args.ids).strip().lower()
        if ids_lc in ("edge_profiles", "core_profiles"):
            if psi_axis_abs is None or psi_lcfs_abs is None:
                # Prefer core_profiles-derived LCFS (guarantees psi_norm=1 at LCFS), else equilibrium.
                pa, pl, src = _read_core_profiles_psi_axis_lcfs(entry, args.time_index, target_time=target_time, occ_hint=args.occ if ids_lc=="edge_profiles" else args.occ)
                if pa is None or pl is None or not (np.isfinite(pa) and np.isfinite(pl)):
                    pa, pl, src = _read_equilibrium_psi_axis_lcfs(entry, args.time_index, target_time=target_time)
                psi_axis_abs = pa
                psi_lcfs_abs = pl
                psi_src = src

            # Print once per run (useful for debugging normalization and SOL extent).
            try:
                xa = x[np.isfinite(x)]
                xmin = float(np.nanmin(xa)) if xa.size else float("nan")
                xmax = float(np.nanmax(xa)) if xa.size else float("nan")
                nsol = int(np.count_nonzero(np.isfinite(x) & (x > 1.0)))
            except Exception:
                xmin, xmax, nsol = float("nan"), float("nan"), -1

            if psi_axis_abs is not None and psi_lcfs_abs is not None and np.isfinite(psi_axis_abs) and np.isfinite(psi_lcfs_abs):
                print(
                    f"{ids_lc}: psi_axis={float(psi_axis_abs):+.6e}  psi_lcfs={float(psi_lcfs_abs):+.6e}  (source: {psi_src}); "
                    f"x_range=[{xmin:.3f},{xmax:.3f}]  n(x>1)={nsol}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"{ids_lc}: psi_axis/psi_lcfs not available (source: {psi_src}); x_range=[{xmin:.3f},{xmax:.3f}] n(x>1)={nsol}",
                    file=sys.stderr,
                )

        ys: Dict[str, np.ndarray] = {}
        resolved: Dict[str, str] = {}
        legend_labels: Dict[str, str] = {}

        for q in qlist:
            qk = q.strip().lower()
            cand = _H5_Q_DS.get(qk, _H5_Q_DS.get(qk.replace(" ", ""), []))
            if not cand:
                print(f"ERROR: Unsupported quantity '{qk}'.")
                _print_quantity_help(args.ids, args.occ, args.time_index, args.entry, g)
                return 2
            ds_name = _h5_get_first_existing(g, cand)
            if ds_name is None:
                print(f"ERROR: Could not find datasets for '{qk}'. Tried: {cand}")
                _print_quantity_help(args.ids, args.occ, args.time_index, args.entry, g)
                return 2

            y = _h5_read_1d(g, ds_name, args.time_index, ion_index=args.ion_index)
            y = np.asarray(y, dtype=float)

            # Legacy compatibility: older input2imas stored ptot as p_perp = ptot/3
            if qk.strip().lower() == "ptot" and ds_name.endswith("&pressure_perpendicular"):
                y = 3.0 * y

            ys[qk] = y
            resolved[qk] = ds_name


        # Choose labels
        if len(qlist) == 1:
            q0 = qlist[0].strip().lower()
            y_label = _q_pretty(q0, with_unit=True)
            legend_labels = {}
        else:
            y_label = None
            for q in qlist:
                qk = q.strip().lower()
                legend_labels[qk] = _q_pretty(qk, with_unit=True)
        # Align lengths: keep the full x grid; pad shorter y arrays with NaN (do not truncate x).
        nx = int(x.size)
        for k in list(ys.keys()):
            y = np.asarray(ys[k], dtype=float).ravel()
            ny = int(y.size)
            if ny == nx:
                ys[k] = y
                continue
            if ny < nx:
                yy = np.full((nx,), np.nan, dtype=float)
                yy[:ny] = y
                ys[k] = yy
                print(f"WARNING: quantity '{k}' has length {ny} < x length {nx}; padding with NaN (likely missing SOL/PF in profiles_1d).", file=sys.stderr)
            else:  # ny > nx
                ys[k] = y[:nx]
                print(f"WARNING: quantity '{k}' has length {ny} > x length {nx}; truncating to {nx}.", file=sys.stderr)

        if args.info:
            print(f"HDF5: {h5_path}")
            print(f"Group: /{args.ids}_{args.occ}")
            print(f"x: {x_ds} kind={x_kind} label={x_label} shape={x.shape}")
            for q in qlist:
                qk = q.strip().lower()
                print(f"{qk}: {resolved[qk]} label={_q_pretty(qk, with_unit=True)} shape={ys[qk].shape}")
            return 0

        q_title = ", ".join([_q_pretty(q.strip().lower(), with_unit=False) for q in qlist])
        title = f"{args.ids}_{args.occ}: {q_title}   time_index={args.time_index}"
        # Plot cosmetics: enforce [0, ...] x-limits for normalized flux coordinates
        xlim = None
        vline1 = False
        ids_lc = str(args.ids).strip().lower()
        if ids_lc == "edge_profiles":
            # Always show magnetic axis at 0 and LCFS at 1, even if core bins are NaN.
            xf = x[np.isfinite(x)]
            xmax = float(np.nanmax(xf)) if xf.size else 1.1
            xmax = max(1.05, xmax)
            xlim = (0.0, xmax)
            vline1 = True
        elif ids_lc == "core_profiles":
            xf = x[np.isfinite(x)]
            xmax = float(np.nanmax(xf)) if xf.size else 1.0
            xmax = max(1.0, xmax)
            xlim = (0.0, xmax)
            vline1 = True

        _plot(x, ys, title=title, x_label=x_label, y_label=y_label, legend_labels=legend_labels, out=args.output, xlim=xlim, vline1=vline1)
        return 0

    finally:
        f.close()


if __name__ == "__main__":
    raise SystemExit(main())
