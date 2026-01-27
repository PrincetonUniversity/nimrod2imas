#!/usr/bin/env python3
"""dump2imas.py

NIMROD dumpgll (HDF5) -> IMAS conversion.

This version restores *stitched* RZ arrays in IMAS (global grid), rather than the packed
rblock layout.

What it writes:
  * equilibrium:
      - time_slice[...].profiles_2d[0] on stitched grid: R,Z, psi, B (and pressure if available)
  * core_profiles:
      - profiles_1d[...]: profiles as functions of absolute poloidal flux psi (grid.psi)
        (rho_tor_norm is not filled unless required by schema validation in your IMAS build)
  * mhd_linear (species_index):
      - time_slice[...].toroidal_mode[...] perturbations on stitched grid
      - occurrence for species 0 contains the common fields (B,V,J,p,T, density)
      - occurrences for species>0 contain only density perturbation + grid + n_tor

Notes:
  * Supports multiple dump files; appends new time slices when possible.
  * Supports rend/imnd packing ambiguity via --dens-pert-order {species_major,mode_major}.

Example:
  python dump2imas.py dumpgll.00000.h5 dumpgll.00010.h5 \
      --backend hdf5 --dbpath . --dd nstx --dd-version 3.42.0 --pulse 201991 --run 1 \
      --occ-base 1 \
      --dens-pert-order species_major
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import imas


try:
    import f90nml  # type: ignore
except Exception:
    f90nml = None  # type: ignore
import xml.etree.ElementTree as ET

from nimrod2imas import (
    entry_dir as _entry_dir_common,
    dd_version_dirname as _dd_version_dirname,
    open_dbentry as _open_db_common,
    ids_factory as _ids_factory_common,
    value_to_string as _value_to_string_common,
    namelist_file_to_xml as _namelist_file_to_xml_common,
)


# -----------------------------
# Small utilities
# -----------------------------

def _log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, flush=True)


def _die(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def _as_f64(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)

def _value_to_string(v: Any) -> str:
    """Serialize values to XML text consistently across tools."""
    return _value_to_string_common(v)



def _build_nimrod_xml(nimrod_path: Optional[str]) -> str:
    """Build an XML string for nimrod.in, matching input2imas semantics.

    Prefer f90nml; fall back to a lightweight parser instead of embedding raw text.
    """
    if nimrod_path is None:
        return ""
    nimrod_path = os.path.expanduser(nimrod_path)
    if not os.path.isfile(nimrod_path):
        return ""
    xml = _namelist_file_to_xml_common('nimrod_inputs', nimrod_path)
    return xml or ""


    try:
        nml = f90nml.read(nimrod_path)
    except Exception:
        return ""

    root = ET.Element("nimrod_inputs")
    nml_el = ET.SubElement(root, "nimrod_in", filename=os.path.basename(nimrod_path))
    for group_name, group in nml.items():
        g_el = ET.SubElement(nml_el, "group", name=str(group_name))
        for var_name, value in group.items():
            v_el = ET.SubElement(g_el, "var", name=str(var_name))
            v_el.text = _value_to_string(value)
    return ET.tostring(root, encoding="unicode")


def _nimrod_in_nonlinear(nimrod_path: Optional[str]) -> Optional[bool]:
    """Return nonlinear flag from nimrod.in if available, else None."""
    if nimrod_path is None:
        return None
    nimrod_path = os.path.expanduser(nimrod_path)
    if not os.path.isfile(nimrod_path):
        return None

    # Preferred: parse namelist
    if f90nml is not None:
        try:
            nml = f90nml.read(nimrod_path)
            for _, grp in nml.items():
                for k, v in grp.items():
                    if str(k).strip().lower() == "nonlinear":
                        if isinstance(v, (bool, np.bool_)):
                            return bool(v)
                        s = str(v).strip().lower()
                        if s in (".true.", "true", "t", "1"):
                            return True
                        if s in (".false.", "false", "f", "0"):
                            return False
        except Exception:
            pass

    # Fallback: regex scan
    try:
        raw = Path(nimrod_path).read_text(errors="ignore")
    except Exception:
        return None
    m = re.search(r"\bnonlinear\s*=\s*([\.]*true[\.]*)", raw, flags=re.IGNORECASE)
    if m:
        return True
    m = re.search(r"\bnonlinear\s*=\s*([\.]*false[\.]*)", raw, flags=re.IGNORECASE)
    if m:
        return False
    m = re.search(r"\bnonlinear\s*=\s*([tTfF])\b", raw)
    if m:
        return m.group(1).lower() == "t"
    return None





def _nimrod_species_info(nimrod_path: Optional[str]) -> Dict[str, Any]:
    """Extract species metadata from nimrod.in when available.

    Expected keys for impurity-capable runs:
      - zisp_input: list of ion charges (excluding electrons)
      - misp_input: list of ion masses in kg (excluding electrons)
      - me_input: electron mass in kg
      - chrg_input: elementary charge in Coulomb

    Returns a dictionary with optional keys: z_ions, m_ions_kg, me_kg, qe_c.
    """
    out: Dict[str, Any] = {}
    if nimrod_path is None:
        return out
    nimrod_path = os.path.expanduser(nimrod_path)
    if not os.path.isfile(nimrod_path):
        return out

    def _to_list(v) -> List[float]:
        if v is None:
            return []
        if isinstance(v, (list, tuple, np.ndarray)):
            return [float(x) for x in v]
        # f90nml may return scalar for singletons
        return [float(v)]

    # Preferred: parse namelist with f90nml
    if f90nml is not None:
        try:
            nml = f90nml.read(nimrod_path)
            for _, grp in nml.items():
                for k, v in grp.items():
                    kk = str(k).strip().lower()
                    if kk == "zisp_input":
                        out["z_ions"] = _to_list(v)
                    elif kk == "misp_input":
                        out["m_ions_kg"] = _to_list(v)
                    elif kk == "me_input":
                        out["me_kg"] = float(v)
                    elif kk == "chrg_input":
                        out["qe_c"] = float(v)
        except Exception:
            pass

    # Fallback: regex on raw text (handles simple 'key = ...' forms)
    if not out:
        try:
            raw = Path(nimrod_path).read_text(errors="ignore")
        except Exception:
            raw = ""
        def _grab_list(key: str) -> List[float]:
            m = re.search(rf"\b{re.escape(key)}\s*=\s*([^\n!#]+)", raw, flags=re.IGNORECASE)
            if not m:
                return []
            s = m.group(1).strip()
            # split on whitespace or commas
            parts = re.split(r"[\s,]+", s)
            vals: List[float] = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                try:
                    vals.append(float(p.replace("d", "e").replace("D", "e")))
                except Exception:
                    pass
            return vals

        def _grab_scalar(key: str) -> Optional[float]:
            vals = _grab_list(key)
            return vals[0] if vals else None

        z = _grab_list("zisp_input")
        m = _grab_list("misp_input")
        me = _grab_scalar("me_input")
        qe = _grab_scalar("chrg_input")
        if z:
            out["z_ions"] = z
        if m:
            out["m_ions_kg"] = m
        if me is not None:
            out["me_kg"] = me
        if qe is not None:
            out["qe_c"] = qe

    return out


def _squeeze1(a: np.ndarray) -> np.ndarray:
    """Squeeze a trailing singleton dimension (common in dumpgll)."""
    a = np.asarray(a)
    if a.ndim >= 1 and a.shape[-1] == 1:
        return a[..., 0]
    return a


def _aos_len(aos: Any) -> int:
    """Length of an IMAS AOS node without using len() (some builds disallow it)."""
    for attr in ("size", "n", "length"):
        try:
            v = getattr(aos, attr)
            if isinstance(v, (int, np.integer)):
                return int(v)
        except Exception:
            pass
    try:
        return int(aos.shape[0])  # type: ignore[attr-defined]
    except Exception:
        pass
    # last resort
    try:
        return int(len(list(aos)))
    except Exception:
        return 0


def _is_time_placeholder(x: Any) -> bool:
    """Detect IMAS placeholder / uninitialized time values.

    IMAS backends commonly use a very negative fill value (e.g. -9e40)
    for unset times.
    """
    try:
        xf = float(x)
    except Exception:
        return True
    return (not np.isfinite(xf)) or (xf < -1.0e20)


# -----------------------------
# IMAS helpers
# -----------------------------

#def _import_imas() -> Any:
#    try:
#        import imas  # type: ignore
#
#        return imas
#    except Exception as exc:
#        _die(f"Failed to import imas: {exc}")


def _normalize_mode(mode: str) -> str:
    """imas_core DBEntry accepts only r/a/w/x."""
    mode = mode.strip()
    if mode in ("r+", "rw"):
        return "a"
    if mode not in ("r", "a", "w", "x"):
        return "a"
    return mode


def _entry_dir(dbpath: Path, dd: str, dd_version: str, pulse: int, run: int, dd_version_dir: str = 'major') -> Path:
    return _entry_dir_common(dbpath, dd, dd_version, pulse, run, dd_version_dir=dd_version_dir)


def _open_db(imas: Any, backend: str, entry_dir: Path, mode: str, dd_version: str) -> Any:
    """Open an IMAS DBEntry using the shared helper.

    Note: we keep the function signature for backward compatibility inside this script.
    """
    db, _uri, _imas_mod = _open_db_common(
        backend,
        str(entry_dir),
        mode=mode,
        dd_version=str(dd_version) if dd_version else None,
    )
    return db


def _ids_factory(imas: Any, dd_version: str) -> Any:
    """Return IDSFactory with the best-compatible constructor."""
    return _ids_factory_common(imas, str(dd_version))

def estimate_psi_axis_and_lcfs_from_psi(psi2d: np.ndarray, qsep: float = 0.98) -> tuple[float, float]:
    """
    Te-free estimate:
      psi_axis = global extremum (min/max)
      psi_lcfs = robust boundary quantile (qsep) of psi on the rectangular boundary
    """
    ps = np.asarray(psi2d, dtype=float)
    psf = ps[np.isfinite(ps)]
    if psf.size == 0:
        return (np.nan, np.nan)

    psi_min = float(np.nanmin(psf))
    psi_max = float(np.nanmax(psf))

    # Take rectangular boundary samples
    b = np.concatenate([
        ps[0, :].ravel(), ps[-1, :].ravel(),
        ps[:, 0].ravel(), ps[:, -1].ravel()
    ])
    bf = b[np.isfinite(b)]
    if bf.size == 0:
        # fallback: use full-field quantile
        bf = psf

    # Decide which extremum is axis by choosing the one farthest from boundary median
    bmed = float(np.nanmedian(bf))
    if abs(psi_min - bmed) >= abs(psi_max - bmed):
        psi_axis = psi_min
        # boundary is "outside" -> higher psi typically
        psi_lcfs = float(np.nanquantile(bf, qsep))
    else:
        psi_axis = psi_max
        # boundary is "outside" -> lower psi typically
        psi_lcfs = float(np.nanquantile(bf, 1.0 - qsep))

    # guard against degeneracy
    if not np.isfinite(psi_lcfs) or abs(psi_lcfs - psi_axis) < 1e-12:
        psi_lcfs = bmed if abs(bmed - psi_axis) > 1e-12 else (psi_axis + 1.0)

    return (float(psi_axis), float(psi_lcfs))


def _db_get(db: Any, factory: Any, ids_name: str, occ: int) -> Optional[Any]:
    """Best-effort DBEntry.get across IMAS python variants."""
    try:
        return db.get(ids_name, occ)
    except Exception:
        pass
    try:
        ids = factory.new(ids_name)
    except Exception:
        try:
            ids = factory(ids_name)
        except Exception:
            return None
    try:
        db.get(ids, occ)
        return ids
    except Exception:
        return None


def _db_put(db: Any, ids: Any, occ: int) -> None:
    try:
        ids.ids_properties.homogeneous_time = 1
        db.put(ids, occ)
    except TypeError:
        db.put(ids)


def _db_put_slice(db: Any, ids: Any, occ: int) -> None:
    """Put a *single* time slice for a homogeneous_time IDS.

    This avoids a fragile pattern where we:
      1) db.get() the full IDS from disk,
      2) resize an Array-of-Structures (time_slice / profiles_1d),
      3) fill the new slice,
      4) db.put() the full IDS back.

    With the HDF5 backend (and depending on IMAS/Core + IMAS-Python versions),
    resizing AOS nodes on an IDS that contains already-loaded data can lead to
    corrupted earlier slices on disk (symptom: earlier slices read back as
    extremely large / nonsensical floating point values, while the last slice
    looks correct).

    Using put_slice() delegates appending to the backend and avoids in-memory
    reallocation issues.
    """

    try:
        ids.ids_properties.homogeneous_time = 1
    except Exception:
        pass

    # Preferred: dedicated API
    if hasattr(db, "put_slice"):
        try:
            db.put_slice(ids, occ)
            return
        except TypeError:
            # some IMAS-Python variants omit the occurrence argument
            db.put_slice(ids)
            return

    # Fallback: some backends expose the lower-level signature put(ids, occ, is_slice)
    try:
        db.put(ids, occ, True)
        return
    except Exception:
        pass

    raise RuntimeError("DBEntry does not support put_slice() in this IMAS-Python/IMAS-Core build")


def _append_time_equilibrium(eq: Any, t: float) -> int:
    """Ensure eq has a new time slice at time t; return index used."""
    # time array
    try:
        times = np.asarray(eq.time, dtype=float)
    except Exception:
        times = np.array([], dtype=float)

    if times.size == 0:
        try:
            eq.time = np.asarray([t], dtype=float)
        except Exception:
            pass
        try:
            eq.time_slice.resize(1)
            eq.time_slice[0].time = float(t)
            return 0
        except Exception:
            return 0

    # append
    idx = int(times.size)
    try:
        eq.time = np.asarray(list(times) + [t], dtype=float)
    except Exception:
        pass
    try:
        eq.time_slice.resize(idx + 1)
        eq.time_slice[idx].time = float(t)
    except Exception:
        pass
    return idx


def _append_time_mhd(mhd: Any, t: float) -> int:
    try:
        times = np.asarray(mhd.time, dtype=float)
    except Exception:
        times = np.array([], dtype=float)

    # IMAS may pre-initialize time_slice[0] (and sometimes time[0]) with a placeholder.
    # If the first time_slice exists, has placeholder time, and has no toroidal_mode entries,
    # we reuse index 0 instead of appending (prevents a leading "empty" time slice with
    # fill values like -9e40 / -999999999).
    try:
        n_ts = _aos_len(mhd.time_slice)
    except Exception:
        n_ts = 0
    if n_ts >= 1:
        try:
            t0 = getattr(mhd.time_slice[0], "time", None)
        except Exception:
            t0 = None
        try:
            n0 = _aos_len(mhd.time_slice[0].toroidal_mode)
        except Exception:
            n0 = 0
        if _is_time_placeholder(t0) and n0 == 0:
            # Ensure mhd.time exists and is consistent
            if times.size == 0:
                try:
                    mhd.time = np.asarray([t], dtype=float)
                except Exception:
                    pass
            else:
                try:
                    times2 = np.asarray(times, dtype=float).copy()
                    times2[0] = float(t)
                    mhd.time = times2
                except Exception:
                    pass
            try:
                mhd.time_slice.resize(max(1, n_ts))
                mhd.time_slice[0].time = float(t)
            except Exception:
                pass
            return 0

    if times.size == 0:
        try:
            mhd.time = np.asarray([t], dtype=float)
        except Exception:
            pass
        try:
            mhd.time_slice.resize(1)
            mhd.time_slice[0].time = float(t)
            return 0
        except Exception:
            return 0

    idx = int(times.size)
    try:
        mhd.time = np.asarray(list(times) + [t], dtype=float)
    except Exception:
        pass
    try:
        mhd.time_slice.resize(idx + 1)
        mhd.time_slice[idx].time = float(t)
    except Exception:
        pass

    # Best-effort consistency: if time array is valid but time_slice[0].time is placeholder, fix it.
    try:
        if times.size >= 1 and _aos_len(mhd.time_slice) >= 1:
            if _is_time_placeholder(getattr(mhd.time_slice[0], "time", None)) and not _is_time_placeholder(times[0]):
                mhd.time_slice[0].time = float(times[0])
    except Exception:
        pass
    return idx


def _append_time_core_profiles(cp: Any, t: float) -> int:
    # DD 3.42 core_profiles uses profiles_1d[] AOS with per-entry time.
    n = 0
    try:
        n = _aos_len(cp.profiles_1d)
    except Exception:
        n = 0

    if n == 0:
        try:
            cp.time = np.asarray([t], dtype=float)
        except Exception:
            pass
        try:
            cp.profiles_1d.resize(1)
            cp.profiles_1d[0].time = float(t)
        except Exception:
            pass
        return 0

    idx = n
    try:
        cp.time = np.asarray(list(np.asarray(cp.time, dtype=float)) + [t], dtype=float)
    except Exception:
        pass
    try:
        cp.profiles_1d.resize(idx + 1)
        cp.profiles_1d[idx].time = float(t)
    except Exception:
        pass
    return idx


# -----------------------------
# dumpgll block discovery
# -----------------------------

def _block_ids(f: h5py.File) -> List[str]:
    """Return block IDs as zero-padded 4-digit strings.

    Supports several NIMROD dumpgll HDF5 layouts:

      A) /rblocks/<bid>/<datasets...>    where <bid> is a group name, often numeric
      B) /rblocks/rz####, /rblocks/psi_eq####, ...  where datasets live directly under /rblocks
      C) /rz####, /psi_eq####, ...       datasets at root

    We only include block IDs for which an RZ geometry dataset is found.
    """
    out: set[str] = set()

    def _norm_bid(name: str) -> str:
        m = re.search(r"(\d+)$", name)
        if not m:
            return name
        return m.group(1).zfill(4)

    # Layout A/B: rblocks exists
    if "rblocks" in f and isinstance(f["rblocks"], h5py.Group):
        rg = f["rblocks"]

        # Case A: subgroups per block
        for name, obj in rg.items():
            if not isinstance(obj, h5py.Group):
                continue
            bid = _norm_bid(name)

            # accept if it contains rz in any common spelling
            if "rz" in obj and isinstance(obj["rz"], h5py.Dataset):
                out.add(bid)
                continue
            if f"rz{bid}" in obj and isinstance(obj[f"rz{bid}"], h5py.Dataset):
                out.add(bid)
                continue
            # looser check: any dataset key that matches rz#### inside the group
            for k in obj.keys():
                if re.fullmatch(r"rz\d{4}", k):
                    out.add(_norm_bid(k))
                    break

        # Case B: datasets directly under /rblocks
        for name, obj in rg.items():
            if isinstance(obj, h5py.Dataset):
                m = re.fullmatch(r"rz(\d{4})", name)
                if m:
                    out.add(m.group(1))

        if out:
            return sorted(out)

    # Layout C: root datasets named rz####
    for k in f.keys():
        m = re.fullmatch(r"rz(\d{4})", k)
        if m and isinstance(f[k], h5py.Dataset):
            out.add(m.group(1))

    return sorted(out)


def _read_block_ds(f: h5py.File, base: str, bid: str) -> np.ndarray:
    """Read block dataset for a given base and bid.

    Supports both:
      - /rblocks/<something>/base
      - root dataset base<bid>
    """
    # Try rblocks group layout
    if "rblocks" in f and isinstance(f["rblocks"], h5py.Group):
        # find group that ends with bid
        g: Optional[h5py.Group] = None
        rb = f["rblocks"]
        # Case: datasets live directly under /rblocks (e.g. /rblocks/rz0001)
        for nm in (
            f"{base}{bid}",
            f"{base}{int(bid)}",
            f"{base}{int(bid):04d}",
            f"{base}{bid.lstrip('0')}",
        ):
            if nm in rb and isinstance(rb[nm], h5py.Dataset):
                return np.asarray(rb[nm][...])
        # fast path: exact key
        for key in (bid, bid.lstrip("0"), f"{int(bid):04d}"):
            if key in rb:
                g = rb[key]
                break
        if g is None:
            # search by suffix
            for k in rb.keys():
                if k.endswith(bid):
                    g = rb[k]
                    break
        if g is not None:
            # Common NIMROD layouts observed in dumpgll files:
            #   1) /rblocks/<bid>/<base>              (e.g. .../rz)
            #   2) /rblocks/<bid>/<base><bid>         (e.g. .../rz0001)
            #   3) /rblocks/<bid>/<base>/<something>  (less common)
            cand_names = [
                base,
                f"{base}{bid}",
                f"{base}{int(bid)}",
                f"{base}{int(bid):04d}",
                f"{base}{bid.lstrip('0')}",
            ]
            for nm in cand_names:
                if nm in g:
                    obj = g[nm]
                    if isinstance(obj, h5py.Dataset):
                        return np.asarray(obj[...])
                    # sometimes nm is a group containing a dataset with the same name
                    if isinstance(obj, h5py.Group):
                        for sub in (base, f"{base}{bid}", bid, bid.lstrip('0')):
                            if sub in obj and isinstance(obj[sub], h5py.Dataset):
                                return np.asarray(obj[sub][...])

    # Root dataset layout
    name = f"{base}{bid}"
    if name in f:
        return np.asarray(f[name][...])

    # Some files may store as base/####
    if base in f and isinstance(f[base], h5py.Group):
        g2 = f[base]
        if bid in g2:
            return np.asarray(g2[bid][...])

    raise KeyError(f"Missing dataset for base='{base}', bid='{bid}'")


def _read_time(f: h5py.File, override: Optional[float]) -> float:
    if override is not None:
        return float(override)

    # common: dumpTime group with attribute vsTime
    if "dumpTime" in f and isinstance(f["dumpTime"], h5py.Group):
        g = f["dumpTime"]
        for k in ("vsTime", "time", "t"):
            try:
                if k in g.attrs:
                    v = g.attrs[k]
                    arr = np.ravel(np.asarray(v, dtype=float))
                    if arr.size > 0:
                        return float(arr[0])
            except Exception:
                pass

    # fallback: dataset "time"
    if "time" in f and isinstance(f["time"], h5py.Dataset):
        arr = np.ravel(np.asarray(f["time"][...], dtype=float))
        if arr.size > 0:
            return float(arr[0])

    return 0.0


def _read_keff(f: h5py.File) -> np.ndarray:
    if "keff" in f:
        return np.ravel(np.asarray(f["keff"][...], dtype=float))
    # fallback: attribute anywhere
    for k, v in f.attrs.items():
        if str(k).lower() == "keff":
            return np.ravel(np.asarray(v, dtype=float))
    return np.array([], dtype=float)


# -----------------------------
# Layout inference and stitching
# -----------------------------

def _lin_to_ij(i: int, nxbl: int, nybl: int, ordering: str) -> Tuple[int, int]:
    if ordering == "yfast":
        # iy varies fastest
        ix = i // nybl
        iy = i % nybl
    else:
        # xfast: ix varies fastest
        iy = i // nxbl
        ix = i % nxbl
    return ix, iy


def _layout_score(rz_blocks: List[np.ndarray], nxbl: int, nybl: int, ordering: str) -> float:
    # Compare overlapping edges; lower is better.
    nblk = len(rz_blocks)
    if nxbl * nybl != nblk:
        return float("inf")
    rz0 = rz_blocks[0]
    ny, nx, _ = rz0.shape
    score = 0.0
    ncomp = 0

    def blk(ix: int, iy: int) -> np.ndarray:
        lin = ix * nybl + iy if ordering == "yfast" else iy * nxbl + ix
        return rz_blocks[lin]

    # x neighbors
    for ix in range(nxbl - 1):
        for iy in range(nybl):
            a = blk(ix, iy)
            b = blk(ix + 1, iy)
            da = a[:, -1, :]
            db = b[:, 0, :]
            d = da - db
            score += float(np.nanmean(d * d))
            ncomp += 1

    # y neighbors
    for ix in range(nxbl):
        for iy in range(nybl - 1):
            a = blk(ix, iy)
            b = blk(ix, iy + 1)
            da = a[-1, :, :]
            db = b[0, :, :]
            d = da - db
            score += float(np.nanmean(d * d))
            ncomp += 1

    if ncomp == 0:
        return float("inf")
    return score / ncomp


def infer_block_layout(rz_blocks: List[np.ndarray]) -> Tuple[int, int, str]:
    """Infer (nxbl, nybl, ordering) from block coordinate continuity."""
    nblk = len(rz_blocks)
    if nblk == 0:
        _die("No rblocks found")

    # Candidates: all factor pairs
    factors: List[Tuple[int, int]] = []
    for nxbl in range(1, nblk + 1):
        if nblk % nxbl == 0:
            nybl = nblk // nxbl
            factors.append((nxbl, nybl))

    best = (1, nblk, "yfast")
    best_score = float("inf")
    for nxbl, nybl in factors:
        for ordering in ("yfast", "xfast"):
            s = _layout_score(rz_blocks, nxbl, nybl, ordering)
            if s < best_score:
                best_score = s
                best = (nxbl, nybl, ordering)

    return best


def stitch_blocks(blocks: List[np.ndarray], nxbl: int, nybl: int, ordering: str) -> np.ndarray:
    """Stitch block arrays into a global grid with 1-point overlap."""
    if len(blocks) == 0:
        _die("No blocks provided")

    b0 = np.asarray(blocks[0])
    ny_loc, nx_loc = b0.shape[0], b0.shape[1]
    Ny = nybl * (ny_loc - 1) + 1
    Nx = nxbl * (nx_loc - 1) + 1

    out_shape = (Ny, Nx) + b0.shape[2:]
    out = np.empty(out_shape, dtype=b0.dtype)
    out[...] = np.nan

    for i, blk in enumerate(blocks):
        ix, iy = _lin_to_ij(i, nxbl, nybl, ordering)
        y0 = iy * (ny_loc - 1)
        x0 = ix * (nx_loc - 1)
        out[y0 : y0 + ny_loc, x0 : x0 + nx_loc, ...] = blk

    return out


# -----------------------------
# Unpack perturbation datasets
# -----------------------------

def _unpack_vec3_modes(a: np.ndarray, nmodes: int) -> np.ndarray:
    """Return (ny,nx,nmodes,3). Accept common packings."""
    a = np.asarray(a)
    a = _squeeze1(a)

    if a.ndim == 4 and a.shape[-1] == 3 and a.shape[-2] == nmodes:
        return a
    if a.ndim == 4 and a.shape[-2] == 3 and a.shape[-1] == nmodes:
        return np.transpose(a, (0, 1, 3, 2))
    if a.ndim == 3 and a.shape[-1] == 3 * nmodes:
        ny, nx, _ = a.shape
        return a.reshape(ny, nx, nmodes, 3)

    raise ValueError(f"Unexpected vec3+modes shape {a.shape} (nmodes={nmodes})")


def _unpack_scalar_modes(a: np.ndarray, nmodes: int) -> np.ndarray:
    """Return (ny,nx,nmodes). Accept common packings."""
    a = np.asarray(a)
    a = _squeeze1(a)

    if a.ndim == 3 and a.shape[-1] == nmodes:
        return a
    if a.ndim == 2 and nmodes == 1:
        return a[:, :, None]

    raise ValueError(f"Unexpected scalar+modes shape {a.shape} (nmodes={nmodes})")


def _unpack_density_modes(
    a: np.ndarray,
    nmodes: int,
    nspec_hint: Optional[int],
    order: str,
) -> Tuple[np.ndarray, int]:
    """Return (ny,nx,nspec,nmodes). Supports ambiguity in packing."""
    a = np.asarray(a)
    a = _squeeze1(a)

    if a.ndim == 4:
        # guess which axis is nspec/nmodes
        if a.shape[2] == nmodes:
            # (ny,nx,nmodes,nspec)
            nspec = a.shape[3]
            return np.transpose(a, (0, 1, 3, 2)), int(nspec)
        if a.shape[3] == nmodes:
            nspec = a.shape[2]
            return a, int(nspec)

    if a.ndim == 3:
        ny, nx, k = a.shape
        # if nspec known, use it
        if nspec_hint is not None and nspec_hint > 0 and k == nspec_hint * nmodes:
            nspec = int(nspec_hint)
            if order == "species_major":
                # last dim = [spec0(m0..), spec1(m0..), ...]
                out = a.reshape(ny, nx, nspec, nmodes)
            else:
                # mode_major: [m0(spec0..), m1(spec0..), ...]
                out = a.reshape(ny, nx, nmodes, nspec).transpose(0, 1, 3, 2)
            return out, nspec

        # otherwise infer a factorization
        if k % nmodes != 0:
            raise ValueError(f"Cannot unpack density modes: last dim {k} not divisible by nmodes {nmodes}")
        nspec = k // nmodes
        if order == "species_major":
            out = a.reshape(ny, nx, nspec, nmodes)
        else:
            out = a.reshape(ny, nx, nmodes, nspec).transpose(0, 1, 3, 2)
        return out, int(nspec)

    if a.ndim == 2:
        # single species, single mode?
        if nmodes == 1:
            return a[:, :, None, None], 1

    raise ValueError(f"Unexpected density shape {a.shape} for nmodes={nmodes}")


# -----------------------------
# Profile binning on psi
# -----------------------------

def _make_equal_count_bins(x: np.ndarray, nbins: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    xv = x[m]
    if xv.size < 2:
        psi_grid = np.linspace(0.0, 1.0, nbins)
        bins = np.digitize(x, psi_grid, right=False)
        return psi_grid, bins

    # Use quantiles for equal-count bins
    qs = np.linspace(0.0, 1.0, nbins)
    edges = np.quantile(xv, qs)
    # Ensure strictly increasing
    edges = np.maximum.accumulate(edges)
    psi_grid = edges
    bins = np.digitize(x, psi_grid, right=False)
    return psi_grid, bins


def _bin_scalar_on_bins(y: np.ndarray, psi_grid: np.ndarray, bins: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    out = np.full_like(psi_grid, np.nan, dtype=float)
    for i in range(psi_grid.size):
        m = (bins == i) & np.isfinite(y)
        if np.any(m):
            out[i] = float(np.nanmean(y[m]))
    return out


# -----------------------------
# IMAS field setters (best effort)
# -----------------------------

def _set_complex_scalar(node: Any, base_names: Sequence[str], re2d: np.ndarray, im2d: np.ndarray) -> None:
    for base in base_names:
        if hasattr(node, base):
            obj = getattr(node, base)
            try:
                obj.real = _as_f64(re2d)
                obj.imaginary = _as_f64(im2d)
                return
            except Exception:
                try:
                    obj.real = _as_f64(re2d)
                    obj.imag = _as_f64(im2d)
                    return
                except Exception:
                    pass
    # Some DDs store scalars directly under name.real/imaginary
    for base in base_names:
        try:
            setattr(node, base + "_real", _as_f64(re2d))
            setattr(node, base + "_imag", _as_f64(im2d))
            return
        except Exception:
            continue



def _set_real_scalar(obj: Any, candidates: Sequence[str], arr: np.ndarray) -> bool:
    """Best-effort setter for a real scalar field (typically 2D on stitched grid)."""
    if arr is None:
        return False
    a = _as_f64(arr)
    for nm in candidates:
        if hasattr(obj, nm):
            try:
                setattr(obj, nm, a)
                return True
            except Exception:
                continue
    return False

def _set_complex_vector(node: Any, base_names: Sequence[str], re3: Tuple[np.ndarray, np.ndarray, np.ndarray], im3: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    for base in base_names:
        if hasattr(node, base):
            obj = getattr(node, base)
            # IMAS convention: coordinate1/2/3
            try:
                obj.coordinate1.real = _as_f64(re3[0])
                obj.coordinate2.real = _as_f64(re3[1])
                obj.coordinate3.real = _as_f64(re3[2])
                obj.coordinate1.imaginary = _as_f64(im3[0])
                obj.coordinate2.imaginary = _as_f64(im3[1])
                obj.coordinate3.imaginary = _as_f64(im3[2])
                return
            except Exception:
                pass


def _set_rz_grid_mhd(plasma: Any, R: np.ndarray, Z: np.ndarray, Nx: int, Ny: int) -> None:
    # grid indices (dim1, dim2)
    dim1 = np.arange(Nx, dtype=float)
    dim2 = np.arange(Ny, dtype=float)

    for grid_obj in (getattr(plasma, "grid", None), getattr(getattr(plasma, "coordinate_system", None), "grid", None)):
        if grid_obj is None:
            continue
        try:
            grid_obj.dim1 = _as_f64(dim1)
            grid_obj.dim2 = _as_f64(dim2)
        except Exception:
            pass

    # store R,Z as 2D arrays if possible
    for obj in (getattr(plasma, "coordinate_system", None), plasma):
        if obj is None:
            continue
        for rname, zname in (("r", "z"), ("R", "Z")):
            try:
                setattr(obj, rname, _as_f64(R))
                setattr(obj, zname, _as_f64(Z))
                return
            except Exception:
                continue


# -----------------------------
# Core conversion
# -----------------------------

def estimate_psi_axis_and_lcfs(psi2d, te2d=None, te_min=10.0, qsep=0.98):
    psi = np.asarray(psi2d, float)
    if te2d is not None:
        te = np.asarray(te2d, float)
        mask = np.isfinite(psi) & np.isfinite(te) & (te > te_min)
    else:
        mask = np.isfinite(psi)

    # Determine sign convention using robust extrema on the mask
    vals = psi[mask]
    if vals.size < 10:
        # fallback: use all finite values
        vals = psi[np.isfinite(psi)]
    if vals.size == 0:
        raise RuntimeError("Cannot estimate psi_axis/psi_lcfs: psi has no finite values")

    # Choose axis as the more “central” extremum (works for typical NIMROD psi)
    pmin = np.nanmin(vals)
    pmax = np.nanmax(vals)

    # Decide which is axis vs edge by assuming LCFS is closer to the opposite extreme than axis
    # In practice: axis is the “more extreme” value; LCFS is the opposite side.
    # If your psi is reversed, this still works with quantile below.
    psi_axis = pmin if abs(pmin) > abs(pmax) else pmax

    # LCFS estimate: high-quantile in plasma region, biased toward the “edge side”
    # If axis is min, edge is high; if axis is max, edge is low.
    if psi_axis == pmin:
        psi_lcfs = np.nanquantile(vals, qsep)
    else:
        psi_lcfs = np.nanquantile(vals, 1.0 - qsep)

    return float(psi_axis), float(psi_lcfs)

def psi_to_rho_norm(psi1d, psi_axis, psi_lcfs):
    psi1d = np.asarray(psi1d, float).ravel()
    den = psi_lcfs - psi_axis
    if not np.isfinite(den) or abs(den) < 1e-30:
        # fallback monotonic coordinate
        return np.linspace(0.0, 1.0, psi1d.size)

    x = (psi1d - psi_axis) / den
    # If den < 0, x flips; fix by using absolute mapping relative to axis->lcfs
    # Equivalent to x = (psi1d-psi_axis)/(psi_lcfs-psi_axis) and then clip
    x = np.clip(x, 0.0, 1.0)
    # sqrt is typical for rho_tor_norm; if you prefer linear, drop sqrt.
    rho = np.sqrt(x)
    # enforce exactly 1 at and beyond LCFS
    rho = np.minimum(rho, 1.0)
    return rho


def read_and_stitch_dump(fn: Path, args) -> Dict[str, Any]:
    with h5py.File(fn, "r") as f:
        bids = _block_ids(f)
        if not bids:
            _die("No rblocks found (no rblocks group, no rz#### datasets)")

        # read RZ for all blocks first (for layout)
        rz_blocks: List[np.ndarray] = []
        for bid in bids:
            rz = _read_block_ds(f, "rz", bid)
            rz = _squeeze1(rz)
            if rz.ndim != 3 or rz.shape[-1] != 2:
                _die(f"rz{bid} expected (ny,nx,2)[,1], got {rz.shape}")
            rz_blocks.append(rz.astype(float, copy=False))

        nxbl, nybl, ordering = infer_block_layout(rz_blocks)

        ny_loc, nx_loc, _ = rz_blocks[0].shape
        Ny = nybl * (ny_loc - 1) + 1
        Nx = nxbl * (nx_loc - 1) + 1

        # stitch R,Z
        R = stitch_blocks([rb[..., 0] for rb in rz_blocks], nxbl, nybl, ordering)
        Z = stitch_blocks([rb[..., 1] for rb in rz_blocks], nxbl, nybl, ordering)

        # Time and modes
        t0 = _read_time(f, args.time)
        keff = _read_keff(f)
        nmodes = int(np.size(keff))

        # Equilibrium fields
        def read_scalar(base: str) -> np.ndarray:
            blocks = []
            for bid in bids:
                a = _read_block_ds(f, base, bid)
                a = _squeeze1(a)
                if a.ndim != 2:
                    _die(f"{base}{bid} expected (ny,nx)[,1], got {a.shape}")
                blocks.append(a.astype(float, copy=False))
            return stitch_blocks(blocks, nxbl, nybl, ordering)

        def read_vec3(base: str) -> np.ndarray:
            blocks = []
            for bid in bids:
                a = _read_block_ds(f, base, bid)
                a = _squeeze1(a)
                if a.ndim != 3:
                    _die(f"{base}{bid} expected (ny,nx,3)[,1], got {a.shape}")
                if a.shape[-1] != 3:
                    _die(f"{base}{bid} expected last dim=3, got {a.shape}")
                blocks.append(a.astype(float, copy=False))
            return stitch_blocks(blocks, nxbl, nybl, ordering)

        def try_read_scalar(base: str) -> Optional[np.ndarray]:
            try:
                return read_scalar(base)
            except Exception:
                return None

        def try_read_vec3(base: str) -> Optional[np.ndarray]:
            try:
                return read_vec3(base)
            except Exception:
                return None

        psi_eq = try_read_scalar("psi_eq")
        bq = try_read_vec3("bq")
        prq = try_read_scalar("prq")
        peq = try_read_scalar("peq")
        teq = try_read_scalar("teq")
        tiq = try_read_scalar("tiq")
        vq = try_read_vec3("vq")
        jq = try_read_vec3("jq")

        # nq: can be (ny,nx) or (ny,nx,nspec)
# nq: can be (ny,nx) or (ny,nx,nspec)
        nq_blocks: List[np.ndarray] = []
        nspec_eq = 0
        nq: Optional[np.ndarray] = None
        try:
            for bid in bids:
                a = _read_block_ds(f, "nq", bid)
                a = _squeeze1(a)
                if a.ndim == 2:
                    a = a[:, :, None]
                if a.ndim != 3:
                    _die(f"nq{bid} expected (ny,nx,nspec)[,1] or (ny,nx), got {a.shape}")
                if nspec_eq == 0:
                    nspec_eq = int(a.shape[-1])
                nq_blocks.append(a.astype(float, copy=False))
            nq = stitch_blocks(nq_blocks, nxbl, nybl, ordering)
            nspec_eq = int(nq.shape[-1])
        except Exception:
            nq = None
            nspec_eq = 0

        # Perturbations: only if nmodes>0
        fields: Dict[str, np.ndarray] = {}
        if nmodes > 0:
            # vector fields with modes
            def read_vec3_modes(base: str) -> np.ndarray:
                blocks = []
                for bid in bids:
                    a = _read_block_ds(f, base, bid)
                    a = _squeeze1(a)
                    a = _unpack_vec3_modes(a, nmodes)
                    blocks.append(a.astype(float, copy=False))
                return stitch_blocks(blocks, nxbl, nybl, ordering)

            # scalar fields with modes
            def read_scalar_modes(base: str) -> np.ndarray:
                blocks = []
                for bid in bids:
                    a = _read_block_ds(f, base, bid)
                    a = _squeeze1(a)
                    a = _unpack_scalar_modes(a, nmodes)
                    blocks.append(a.astype(float, copy=False))
                return stitch_blocks(blocks, nxbl, nybl, ordering)

            fields["rebe"] = read_vec3_modes("rebe")
            fields["imbe"] = read_vec3_modes("imbe")

            # velocity (optional)
            for nm in ("reve", "imve"):
                try:
                    fields[nm] = read_vec3_modes(nm)
                except Exception:
                    pass

            # current density perturbation (requested)
            for nm in ("reja", "imja"):
                try:
                    fields[nm] = read_vec3_modes(nm)
                except Exception:
                    pass

            # pressure/temp
            for nm in ("repr", "impr", "repe", "impe", "rete", "imte", "reti", "imti"):
                try:
                    fields[nm] = read_scalar_modes(nm)
                except Exception:
                    pass

            # density perturbation: rend/imnd (multi-species)
            dens: Dict[str, Tuple[np.ndarray, int]] = {}
            for nm in ("rend", "imnd"):
                try:
                    blocks = []
                    nspec_here: Optional[int] = None
                    for bid in bids:
                        a = _read_block_ds(f, nm, bid)
                        a = _squeeze1(a)
                        out, ns = _unpack_density_modes(a, nmodes, nspec_eq if nspec_eq > 0 else None, args.dens_pert_order)
                        nspec_here = ns
                        blocks.append(out.astype(float, copy=False))
                    dens[nm] = (stitch_blocks(blocks, nxbl, nybl, ordering), int(nspec_here or 0))
                except Exception:
                    pass

            if "rend" in dens and "imnd" in dens:
                fields["rend"], nspec_dens = dens["rend"]
                fields["imnd"], _ = dens["imnd"]
                fields["nspec_dens"] = np.array([nspec_dens], dtype=int)

                # Store 2D fields in a stitched IMAS-friendly (dim1,dim2) layout.
        #
        # NIMROD rblock stitching produces arrays shaped (Ny, Nx) where the first axis is the
        # "y-like" direction (typically Z) and the second axis is the "x-like" direction (typically R).
        # For IMAS RZ grids (and for user-facing contour plots), we store arrays as (Nx, Ny),
        # i.e. dim1 corresponds to R-index and dim2 to Z-index.
        def _swap01(a: np.ndarray) -> np.ndarray:
            a = np.asarray(a)
            if a.ndim < 2:
                return a
            axes = list(range(a.ndim))
            axes[0], axes[1] = 1, 0
            return np.transpose(a, axes)

        R_imas = _swap01(R) * float(getattr(args, "L_scale", 1.0))
        Z_imas = _swap01(Z) * float(getattr(args, "L_scale", 1.0))

        psi_eq_imas = _swap01(psi_eq) if psi_eq is not None else None
        bq_imas = (_swap01(bq) * float(getattr(args, "B_scale", 1.0))) if bq is not None else None
        prq_imas = (_swap01(prq) * float(getattr(args, "p_scale", 1.0))) if prq is not None else None
        peq_imas = (_swap01(peq) * float(getattr(args, "p_scale", 1.0))) if ("peq" in locals() and peq is not None) else None
        teq_imas = (_swap01(teq) * float(getattr(args, "T_scale", 1.0))) if teq is not None else None
        tiq_imas = (_swap01(tiq) * float(getattr(args, "T_scale", 1.0))) if tiq is not None else None
        nq_imas  = (_swap01(nq)  * float(getattr(args, "n_scale", 1.0))) if nq is not None else None
        vq_imas  = (_swap01(vq)  * float(getattr(args, "v_scale", 1.0))) if vq is not None else None
        jq_imas  = (_swap01(jq)  * float(getattr(args, "j_scale", 1.0))) if jq is not None else None

        fields_imas: Dict[str, np.ndarray] = {}
        for k, v in fields.items():
            try:
                vv = _swap01(v)
                # Apply deterministic unit scaling by field family
                if k in ("repr","impr","repe","impe"):
                    vv = vv * float(getattr(args, "p_scale", 1.0))
                elif k in ("rete","imte","reti","imti"):
                    vv = vv * float(getattr(args, "T_scale", 1.0))
                elif k in ("rend","imnd"):
                    vv = vv * float(getattr(args, "n_scale", 1.0))
                elif k in ("rebe","imbe"):
                    vv = vv * float(getattr(args, "B_scale", 1.0))
                elif k in ("reve","imve"):
                    vv = vv * float(getattr(args, "v_scale", 1.0))
                elif k in ("reja","imja"):
                    vv = vv * float(getattr(args, "j_scale", 1.0))
                fields_imas[k] = vv
            except Exception:
                pass

        return dict(
            time=float(t0),
            keff=_as_f64(keff),
            nmodes=int(nmodes),
            R=_as_f64(R_imas),
            Z=_as_f64(Z_imas),
            psi_eq=_as_f64(psi_eq_imas) if psi_eq_imas is not None else None,
            bq=_as_f64(bq_imas) if bq_imas is not None else None,
            prq=_as_f64(prq_imas) if prq_imas is not None else None,
            peq=_as_f64(peq_imas) if peq_imas is not None else None,
            teq=_as_f64(teq_imas) if teq_imas is not None else None,
            tiq=_as_f64(tiq_imas) if tiq_imas is not None else None,
            nq=_as_f64(nq_imas) if nq_imas is not None else None,
            vq=_as_f64(vq_imas) if vq_imas is not None else None,
            jq=_as_f64(jq_imas) if jq_imas is not None else None,
            nspec_eq=int(nspec_eq),
            fields=fields_imas,
            layout=dict(
                nxbl=nxbl,
                nybl=nybl,
                ordering=ordering,
                # stitched global sizes in original orientation
                Ny_raw=Ny,
                Nx_raw=Nx,
                # stored IMAS orientation (dim1, dim2) = (Nx, Ny)
                Nx=int(Nx),
                Ny=int(Ny),
                ny_loc=ny_loc,
                nx_loc=nx_loc,
            ),
        )



def _psi_axis_and_sign(psi: np.ndarray) -> Tuple[float, float]:
    # Use median of finite psi as axis estimate and choose sign so that psi_out is mostly positive.
    p = np.asarray(psi, dtype=float)
    m = np.isfinite(p)
    if not np.any(m):
        return 0.0, 1.0
    psi_axis = float(np.nanmedian(p[m]))
    # sign: choose so that outside is positive
    po = p[m] - psi_axis
    sign = 1.0 if np.nanmean(po) >= 0 else -1.0
    return psi_axis, sign


def populate_equilibrium(eq: Any, data: Dict[str, Any], t_index: int, quiet: bool) -> None:
    t = float(data["time"])
    R = data["R"]
    Z = data["Z"]
    psi = data["psi_eq"]
    bq = data["bq"]
    prq = data["prq"]
    peq = data.get("peq", None)
    jq = data["jq"]

    if psi is None or bq is None or prq is None:
        return

    Nx, Ny = int(data["layout"]["Nx"]), int(data["layout"]["Ny"])

    # Ensure time slice
    idx = _append_time_equilibrium(eq, t)

    ts = eq.time_slice[idx]
    ts.time = float(t)

    # profiles_2d
    try:
        ts.profiles_2d.resize(1)
        p2d = ts.profiles_2d[0]
        # grid indices (avoid int64 warnings by storing float)
        p2d.grid.dim1 = _as_f64(np.arange(Nx))
        p2d.grid.dim2 = _as_f64(np.arange(Ny))
        p2d.r = _as_f64(R)
        p2d.z = _as_f64(Z)
        if hasattr(p2d, "psi"):
            p2d.psi = _as_f64(psi)
        # B components (best effort across DD variants)
        if hasattr(p2d, "b_field_r"):
            p2d.b_field_r = _as_f64(bq[..., 0])
        if hasattr(p2d, "b_field_z"):
            p2d.b_field_z = _as_f64(bq[..., 1])
        if hasattr(p2d, "b_field_tor"):
            p2d.b_field_tor = _as_f64(bq[..., 2])
        if hasattr(p2d, "pressure"):
            p2d.pressure = _as_f64(prq)
        if hasattr(p2d, "j_tor"):
            p2d.j_tor = _as_f64(jq[..., 2])
    except Exception as exc:
        _die(f"Failed to populate equilibrium.profiles_2d: {exc}")

    # metadata
    try:
        eq.code.name = "NIMROD"
    except Exception:
        pass

    if not quiet:
        _log("Populated equilibrium IDS", quiet=False)



def populate_core_profiles(cp: Any, data: Dict[str, Any], t_index: int, args) -> None:
    """Populate core_profiles as functions of (signed, axis-referenced) poloidal flux.

    This follows the original script's assumptions for the normalized toroidal flux:
      - Build a 1D grid from the stitched 2D psi_eq field.
      - Define rho_tor_norm = 0 at the magnetic axis and 1 at the separatrix (LCFS),
        and hold at 1 through the SOL extension.
    """
    t = float(data["time"])
    psi2d = data.get("psi_eq", None)
    pr2d = data.get("prq", None)   # total pressure (Pa)
    pe2d = data.get("peq", None)   # electron pressure (Pa), if present
    nq = data.get("nq", None)      # number densities (m^-3), species last dim (0=e)
    te2d = data.get("teq", None)   # electron temperature (eV), if present
    ti2d = data.get("tiq", None)   # ion temperature (eV), if present (typically single Ti)
    vq = data.get("vq", None)      # velocity (m/s), if present
    jq = data.get("jq", None)      # current density (A/m^2), if present
    R = data.get("R", None)

    if psi2d is None or pr2d is None or nq is None:
        return

    psi_axis, sign = _psi_axis_and_sign(psi2d)
    psi_out = sign * (psi2d - psi_axis)

    # equal-count psi grid
    psi_grid, bins = _make_equal_count_bins(psi_out, int(args.nbins))

    # ---- 1D binned quantities ----
    p_tot1d = _bin_scalar_on_bins(pr2d, psi_grid, bins)
    pe1d = _bin_scalar_on_bins(pe2d, psi_grid, bins) if pe2d is not None else None

    # densities (heuristic: 0 = electrons)
    ne1d = _bin_scalar_on_bins(nq[..., 0], psi_grid, bins) if nq.shape[-1] >= 1 else None
    ion_list: List[np.ndarray] = []
    for s in range(1, int(nq.shape[-1])):
        ion_list.append(_bin_scalar_on_bins(nq[..., s], psi_grid, bins))

    # elementary charge (C); allow override from nimrod.in if provided
    sp = getattr(args, "_nimrod_species", {}) or {}
    qe = float(sp.get("qe_c", 1.602176634e-19))

    # Electron temperature: prefer teq; otherwise derive from pe/ne when available
    if te2d is not None:
        te1d = _bin_scalar_on_bins(te2d, psi_grid, bins)
    elif pe1d is not None and ne1d is not None:
        te1d = _as_f64(pe1d) / (_as_f64(ne1d) * qe)
    else:
        te1d = None

    # Ion temperature: prefer tiq; otherwise derive from (ptot - pe)/ni_total when available
    if ti2d is not None:
        ti1d = _bin_scalar_on_bins(ti2d, psi_grid, bins)
    else:
        if pe1d is not None and ion_list:
            ni_tot = np.zeros_like(_as_f64(ion_list[0]))
            for ni in ion_list:
                ni_tot = ni_tot + _as_f64(ni)
            pi1d = _as_f64(p_tot1d) - _as_f64(pe1d)
            ti1d = pi1d / (ni_tot * qe)
        else:
            ti1d = None

    # Rotation frequency omega_tor = vphi/R (use toroidal component index 2)
    omega1d = None
    if vq is not None and R is not None:
        try:
            omega2d = np.full_like(pr2d, np.nan, dtype=float)
            msk = np.isfinite(R) & (np.abs(R) > 0) & np.isfinite(vq[..., 2])
            omega2d[msk] = vq[..., 2][msk] / R[msk]
            omega1d = _bin_scalar_on_bins(omega2d, psi_grid, bins)
        except Exception:
            omega1d = None

    # Toroidal current density
    jtor1d = None
    if jq is not None:
        try:
            jtor1d = _bin_scalar_on_bins(jq[..., 2], psi_grid, bins)
        except Exception:
            jtor1d = None

    # append profile entry
    idx = _append_time_core_profiles(cp, t)
    p = cp.profiles_1d[idx]

    # ---- grid coordinates ----
    if hasattr(p, "grid"):
        g = p.grid

        # psi_grid is on the positive (axis-referenced) psi_out coordinate.
        psi1d = np.asarray(psi_grid, dtype=np.float64).ravel()
        psi_abs_1d = np.asarray(psi_axis + sign * psi1d, dtype=np.float64)

        # Enforce monotonic ordering (axis -> SOL) and keep all 1D profiles aligned.
        order = np.argsort(psi1d)
        psi1d = psi1d[order]
        psi_abs_1d = psi_abs_1d[order]
        p_tot1d = _as_f64(p_tot1d)[order]
        if pe1d is not None:
            pe1d = _as_f64(pe1d)[order]
        if te1d is not None:
            te1d = _as_f64(te1d)[order]
        if ti1d is not None:
            ti1d = _as_f64(ti1d)[order]
        if jtor1d is not None:
            jtor1d = _as_f64(jtor1d)[order]
        if omega1d is not None:
            omega1d = _as_f64(omega1d)[order]
        if ne1d is not None:
            ne1d = _as_f64(ne1d)[order]
        for k in range(len(ion_list)):
            ion_list[k] = _as_f64(ion_list[k])[order]

        # Separatrix location (best-effort) in the same psi_out coordinate as psi1d.
        psi_axis_est, psi_lcfs_est = estimate_psi_axis_and_lcfs_from_psi(psi2d, qsep=0.98)
        if np.isfinite(psi_lcfs_est) and np.isfinite(psi_axis):
            psi_lcfs_out = sign * (psi_lcfs_est - psi_axis)
        else:
            psi_lcfs_out = np.nanmax(psi1d)

        # rho_tor_norm: 0 at axis, 1 at LCFS, held at 1 in SOL
        rho = np.zeros_like(psi1d)
        if np.isfinite(psi_lcfs_out) and psi_lcfs_out != 0:
            rho = psi1d / float(psi_lcfs_out)
        rho = np.clip(rho, 0.0, 1.0)

        # Assign grid fields (support multiple DD leaf names)
        for nm in ("psi", "psi_norm", "psi_tor_norm", "psi_pol"):
            if hasattr(g, nm):
                try:
                    setattr(g, nm, psi_abs_1d)
                    break
                except Exception:
                    pass
        if hasattr(g, "rho_tor_norm"):
            try:
                g.rho_tor_norm = rho
            except Exception:
                pass

    # ---- populate species ----
    # electrons
    try:
        if hasattr(p, "electrons"):
            e = p.electrons
            if ne1d is not None and hasattr(e, "density"):
                e.density = _as_f64(ne1d)
            if te1d is not None:
                for nm in ("temperature", "t_e", "temp"):
                    if hasattr(e, nm):
                        try:
                            setattr(e, nm, _as_f64(te1d))
                            break
                        except Exception:
                            pass
            # electron pressure (optional)
            if pe1d is not None:
                for nm in ("pressure", "p"):
                    if hasattr(e, nm):
                        try:
                            setattr(e, nm, _as_f64(pe1d))
                            break
                        except Exception:
                            pass
    except Exception:
        pass

    # ions (including impurities)
    try:
        nion = max(0, int(nq.shape[-1]) - 1)
        if hasattr(p, "ion") and nion > 0:
            if _aos_len(p.ion) < nion:
                p.ion.resize(nion)

            z_ions = sp.get("z_ions", []) or []
            m_ions = sp.get("m_ions_kg", []) or []
            AMU = 1.66053906660e-27
            # If nimrod.in does not define zisp_input/misp_input (common for non-impurity builds),
            # populate reasonable defaults for the main-ion species so that core_profiles has valid metadata.
            if not z_ions and nion > 0:
                z_ions = [1.0] * nion
            if not m_ions and nion > 0:
                # Default to deuterium mass (2 amu) for main ion; override by providing misp_input.
                m_ions = [2.0 * AMU] * nion

            for k in range(nion):
                ion_k = p.ion[k]
                # density
                if k < len(ion_list) and hasattr(ion_k, "density"):
                    ion_k.density = _as_f64(ion_list[k])
                # temperature (single Ti applied to all ion species if present/derived)
                if ti1d is not None:
                    for nm in ("temperature", "t_i", "temp"):
                        if hasattr(ion_k, nm):
                            try:
                                setattr(ion_k, nm, _as_f64(ti1d))
                                break
                            except Exception:
                                pass
                # per-ion pressure (optional): distribute via ideal gas p_i = n_i * Ti * qe
                # (Only if Ti is available and density exists)
                try:
                    if ti1d is not None and k < len(ion_list):
                        pi1d = _as_f64(ion_list[k]) * _as_f64(ti1d) * qe
                        for nm in ("pressure", "p"):
                            if hasattr(ion_k, nm):
                                setattr(ion_k, nm, pi1d)
                                break
                except Exception:
                    pass

                # species metadata from nimrod.in (best-effort)
                if k < len(z_ions):
                    for nm in ("z_ion", "z", "charge_state", "charge"):
                        if hasattr(ion_k, nm):
                            try:
                                setattr(ion_k, nm, float(z_ions[k]))
                                break
                            except Exception:
                                pass
                if k < len(m_ions):
                    mkg = float(m_ions[k])
                    for nm in ("mass", "mass_kg", "ion_mass", "m"):
                        if hasattr(ion_k, nm):
                            try:
                                setattr(ion_k, nm, mkg)
                                break
                            except Exception:
                                pass
                    aamu = mkg / AMU if AMU > 0 else np.nan
                    for nm in ("a", "a_ion", "atomic_mass"):
                        if hasattr(ion_k, nm):
                            try:
                                setattr(ion_k, nm, float(aamu))
                                break
                            except Exception:
                                pass

            # rotation frequency stored on main-ion entry when available
            if omega1d is not None:
                try:
                    if hasattr(p.ion[0], "rotation_frequency_tor_s"):
                        p.ion[0].rotation_frequency_tor_s = _as_f64(omega1d)
                except Exception:
                    pass
    except Exception:
        pass

    # total current density profile (optional)
    if jtor1d is not None:
        for name in ("j_tor", "jtor", "j_phi"):
            if hasattr(p, name):
                try:
                    setattr(p, name, _as_f64(jtor1d))
                    break
                except Exception:
                    pass

    # total pressure profile
    if hasattr(p, "pressure"):
        try:
            p.pressure = _as_f64(p_tot1d)
        except Exception:
            pass

    try:
        cp.code.name = "NIMROD"
    except Exception:
        pass


# ----------------------------
# Nonlinear mhd (GGD) support
# ----------------------------

# ----------------------------

def _bin2d_avg(R: np.ndarray, Z: np.ndarray, V: np.ndarray, nr: int, nz: int):
    """Average V(R,Z) onto a regular R-Z grid using simple binning.

    Returns (R_centers[nr], Z_centers[nz], V_binned[nr,nz]).
    """
    Rf = np.asarray(R, dtype=float).ravel()
    Zf = np.asarray(Z, dtype=float).ravel()
    Vf = np.asarray(V, dtype=float).ravel()

    m = np.isfinite(Rf) & np.isfinite(Zf) & np.isfinite(Vf)
    if not np.any(m):
        raise ValueError("No finite points to bin for mhd.ggd output")

    Rf = Rf[m]; Zf = Zf[m]; Vf = Vf[m]
    rmin, rmax = float(np.min(Rf)), float(np.max(Rf))
    zmin, zmax = float(np.min(Zf)), float(np.max(Zf))

    if rmax <= rmin:
        rmax = rmin + 1.0
    if zmax <= zmin:
        zmax = zmin + 1.0

    H, r_edges, z_edges = np.histogram2d(
        Rf, Zf,
        bins=[nr, nz],
        range=[[rmin, rmax], [zmin, zmax]],
        weights=Vf
    )
    C, _, _ = np.histogram2d(
        Rf, Zf,
        bins=[nr, nz],
        range=[[rmin, rmax], [zmin, zmax]]
    )
    Vb = H / np.maximum(C, 1.0)
    rc = 0.5 * (r_edges[:-1] + r_edges[1:])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    return rc.astype(float), zc.astype(float), Vb.astype(float)


def _reconstruct_full_from_modes(
    eq: np.ndarray | None,
    re_arr: np.ndarray | None,
    im_arr: np.ndarray | None,
    n_tor: np.ndarray,
    phi: float,
) -> np.ndarray | None:
    """Reconstruct a real-space field at toroidal angle phi from stored Fourier coefficients.

    NIMROD dumps store real/imag parts of Fourier coefficients. Depending on build and writer,
    the mode dimension may be the first or last axis (and some fields may include a species axis).
    This routine normalizes arrays to (nmodes, Ny, Nx) before reconstruction.

    f(phi) = f0 + 2*sum_{m>0} (re_m*cos(n_m*phi) - im_m*sin(n_m*phi))
    """

    def _reduce_eq(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        # common cases: (Ny,Nx), (Ny,Nx,1), (Ny,Nx,nspec)
        if a.ndim == 3:
            if a.shape[2] == 1:
                return a[:, :, 0]
            # for densities in multi-species dumps, export a total by summing species
            return np.nansum(a, axis=2)
        return a

    def _norm_modes(a: np.ndarray | None) -> np.ndarray | None:
        if a is None:
            return None
        A = np.asarray(a, dtype=float)

        # If a has a species axis, collapse it to a total before mode normalization
        # Expected common layouts:
        #   (Ny,Nx,nspec,nmodes) or (Ny,Nx,nmodes,nspec)
        if A.ndim == 4:
            # pick the axis that is likely species: the one not matching nmodes and not Ny/Nx
            nm = int(len(n_tor))
            if A.shape[-1] == nm:
                # (Ny,Nx,nspec,nmodes)
                A = np.nansum(A, axis=2)  # -> (Ny,Nx,nmodes)
            elif A.shape[2] == nm:
                # (Ny,Nx,nmodes,nspec)
                A = np.nansum(A, axis=3)  # -> (Ny,Nx,nmodes)
            elif A.shape[0] == nm:
                # (nmodes,Ny,Nx,nspec) etc.
                A = np.nansum(A, axis=3)
            else:
                # fallback: collapse the last axis as species
                A = np.nansum(A, axis=-1)

        # Promote 2D to 3D with explicit mode axis
        if A.ndim == 2:
            return A[None, :, :]

        if A.ndim != 3:
            # Unexpected; attempt squeeze
            A = np.squeeze(A)
            if A.ndim == 2:
                return A[None, :, :]
            if A.ndim != 3:
                return None

        nm = int(len(n_tor))
        # Mode axis could be 0, 1, or 2. Normalize to axis 0.
        if A.shape[0] == nm:
            return A
        if A.shape[2] == nm:
            return np.moveaxis(A, 2, 0)
        if A.shape[1] == nm:
            return np.moveaxis(A, 1, 0)

        # If none match exactly, fall back to assuming mode axis is last
        return np.moveaxis(A, -1, 0)

    base = None if eq is None else _reduce_eq(eq)

    reA = _norm_modes(re_arr)
    imA = _norm_modes(im_arr)
    if reA is None or imA is None:
        return base

    nm = min(int(len(n_tor)), int(reA.shape[0]), int(imA.shape[0]))
    if nm <= 0:
        return base

    out = np.zeros(reA.shape[1:], dtype=float)
    for m in range(1, nm):
        n = float(n_tor[m])
        if n == 0.0:
            continue
        c = np.cos(n * phi)
        s = np.sin(n * phi)
        out += 2.0 * (reA[m] * c - imA[m] * s)

    if base is None:
        return out

    # Ensure consistent orientation
    if base.shape != out.shape and base.T.shape == out.shape:
        base = base.T
    return base + out


def _append_time_ggd(mhd: Any, t: float) -> int:
    """Append a new time slice to both mhd.ggd and mhd.grid_ggd and return the new index.

    IMAS defines `grid_ggd(itime)` and `ggd(itime)` arrays with `time` as the coordinate. In some
    imas-python builds, the coordinate setter may raise if the node is absent/uninitialised; we
    therefore attempt multiple assignment paths and fail loudly only if all fail.
    """
    cur = _aos_len(getattr(mhd, "ggd"))
    mhd.ggd.resize(cur + 1)
    mhd.grid_ggd.resize(cur + 1)

    def _set_time(aos, idx: int, val: float, label: str) -> None:
        # Preferred: scalar leaf on the AoS element (DD: FLT_0D).
        try:
            aos[idx].time = float(val)
            return
        except Exception as e1:
            # Fallback: coordinate vector on the AoS container (rare, but seen in some wrappers).
            try:
                arr = getattr(aos, "time")
                # Handle numpy-like or list-like containers.
                if hasattr(arr, "size"):
                    arr = np.asarray(arr, dtype=float)
                    if arr.size < idx + 1:
                        arr2 = np.empty(idx + 1, dtype=float)
                        if arr.size:
                            arr2[:arr.size] = arr
                        arr2[idx] = float(val)
                        setattr(aos, "time", arr2)
                    else:
                        arr[idx] = float(val)
                        setattr(aos, "time", arr)
                else:
                    # list-like
                    while len(arr) < idx + 1:
                        arr.append(np.nan)
                    arr[idx] = float(val)
                    setattr(aos, "time", arr)
                return
            except Exception as e2:
                raise RuntimeError(f"Failed to set time coordinate for {label}[{idx}]") from e2

    _set_time(mhd.ggd, cur, t, "mhd.ggd")
    _set_time(mhd.grid_ggd, cur, t, "mhd.grid_ggd")

    # Optional: keep top-level `mhd.time` consistent when present (not required by DD,
    # but some backends validate it as the global timebase).
    try:
        times = getattr(mhd, "time", None)
        if times is None:
            pass
        else:
            times = np.asarray(times, dtype=float) if hasattr(times, "__len__") else np.asarray([], dtype=float)
            if times.size < cur + 1:
                times2 = np.empty(cur + 1, dtype=float)
                if times.size:
                    times2[:times.size] = times
                times2[cur] = float(t)
                mhd.time = times2
            else:
                times[cur] = float(t)
                mhd.time = times
    except Exception:
        pass

    return cur


def populate_mhd_ggd(mhd: Any, data: Dict[str, Any], args) -> None:
    """Populate the nonlinear mhd IDS (GGD-based) with reconstructed full fields.

    Complements mhd_linear mode-resolved output.
    """
    t = float(data["time"])
    fields: Dict[str, np.ndarray] = data["fields"]
    R = np.asarray(data["R"], dtype=float)
    Z = np.asarray(data["Z"], dtype=float)
    keff = np.asarray(data["keff"], dtype=float) if data.get("keff") is not None else np.arange(int(data.get("nmodes", 0)), dtype=float)

    nphi = max(1, int(getattr(args, "ggd_nphi", 8) or 1))
    nb = max(4, int(getattr(args, "ggd_nbins", 128) or 128))
    phi_list = np.linspace(0.0, 2.0*np.pi, num=nphi, endpoint=False)

    def _make_values(eq_key: str, re_key: str, im_key: str):
        eq = data.get(eq_key, None)
        reA = fields.get(re_key, None)
        imA = fields.get(im_key, None)
        if eq is None and (reA is None or imA is None):
            return None, None, None
        vals_phi = []
        rc = zc = None
        for phi in phi_list:
            full = _reconstruct_full_from_modes(eq, reA, imA, keff, float(phi))
            if full is None:
                continue

            # Ensure V, R, Z are aligned (some dumps store arrays transposed)
            Rloc = R
            Zloc = Z
            if full.shape != Rloc.shape:
                if full.T.shape == Rloc.shape:
                    full = full.T
                elif Rloc.T.shape == full.shape:
                    Rloc = Rloc.T
                    Zloc = Zloc.T
            rc, zc, vb = _bin2d_avg(Rloc, Zloc, full, nb, nb)
            vals_phi.append(vb)
        if not vals_phi:
            return None, None, None
        V3 = np.stack(vals_phi, axis=2)  # (nr, nz, nphi)
        return rc, zc, V3

    it = _append_time_ggd(mhd, t)
    g = mhd.grid_ggd[it]
    try:
        g.identifier.name = "nimrod_rzphi_regular"
        g.identifier.index = -1
        g.identifier.description = "Regular R-Z(-phi) grid for NIMROD full-field export (downsampled)"
    except Exception:
        pass

    # If requested, we represent the R-Z-phi grid as an *unstructured* GGD
    # (nodes + connectivity) to avoid the structured-GGD shape ambiguities
    # that can arise with the HDF5 backend. In this mode we only create a
    # tiny “skeleton” grid_ggd here (fast), and (when --ggd-h5py-direct is
    # used) a packed writer fills the large node/connectivity arrays directly
    # into the produced IDS HDF5.
    if getattr(args, "ggd_unstructured", False):
        try:
            g.space.resize(3)
            # Keep identifiers minimal; geometry will be provided via packed write.
            for ii, nm in enumerate(["R", "Z", "Phi"]):
                try:
                    g.space[ii].identifier.name = nm
                    g.space[ii].identifier.index = -1
                    g.space[ii].identifier.description = nm
                except Exception:
                    pass
        except Exception:
            pass

        # Minimal grid_subset skeleton (nodes + volumes) to force the backend to create
        # the relevant datasets. The packed writer will overwrite/resize them.
        try:
            g.grid_subset.resize(2)

            # Subset 0: nodes (dimension=0)
            s0 = g.grid_subset[0]
            try:
                s0.dimension = 0
                s0.identifier.name = "nodes"
                s0.identifier.index = 0
                s0.identifier.description = "Unstructured nodes"
            except Exception:
                pass
            try:
                s0.element.resize(1)
                s0.element[0].object.resize(3)
                for k in range(3):
                    try:
                        s0.element[0].object[k].real = 0.0
                    except Exception:
                        pass
            except Exception:
                pass

            # Subset 1: volumes (dimension=3)
            s1 = g.grid_subset[1]
            try:
                s1.dimension = 3
                s1.identifier.name = "volumes"
                s1.identifier.index = 1
                s1.identifier.description = "Unstructured connectivity"
            except Exception:
                pass
            try:
                # Base points to nodes subset
                s1.base.resize(1)
                s1.base[0].index = 0
                s1.base[0].grid_subset_index = 0
            except Exception:
                pass
            try:
                ncorner = 8 if getattr(args, "ggd_connectivity", "hex") == "hex" else 4
                s1.element.resize(1)
                s1.element[0].object.resize(ncorner)
                for k in range(ncorner):
                    try:
                        s1.element[0].object[k].index = 1
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

        # In unstructured mode we do not attempt to populate the structured axis geometry
        # via space.objects_per_dimension here.
        nspaces = 3
    else:
        nspaces = 3 if nphi > 1 else 2
    try:
        g.space.resize(nspaces)
    except Exception:
        pass

    def _fill_space(space_obj, coord_name: str, coord_vals: np.ndarray, geom_type_index: int = 0, geom_type_name: str = "standard"):
        """Populate grid_ggd.space for one coordinate axis.

        IMPORTANT: IMAS HDF5 backend expects a single `object[0].geometry` array holding
        the coordinate vector, not one object per point. Writing one object per point
        leads to ambiguous/padded shapes downstream.
        """
        try:
            space_obj.geometry_type.index = int(geom_type_index)
            space_obj.geometry_type.name = geom_type_name
            space_obj.geometry_type.description = geom_type_name
        except Exception:
            pass
        try:
            space_obj.coordinates_type.resize(1)
            space_obj.coordinates_type[0].name = coord_name
            space_obj.coordinates_type[0].index = -1
            space_obj.coordinates_type[0].description = coord_name
        except Exception:
            pass
        try:
            space_obj.objects_per_dimension.resize(1)
        except Exception:
            pass
        try:
            opd = space_obj.objects_per_dimension[0]
            try:
                opd.geometry_content.name = "coordinate"
                opd.geometry_content.index = -1
                opd.geometry_content.description = "Coordinate vector"
            except Exception:
                pass
            opd.object.resize(1)
            v = np.asarray(coord_vals, dtype=float).reshape(-1, 1)
            opd.object[0].geometry = v
        except Exception:
            pass

    def _set_values_and_shape(qleaf: Any, values_1d: np.ndarray, shape_hint: Sequence[int]) -> None:
        """Set qleaf.values and (if available) qleaf.values_shape for HDF5 backend.

        Best-effort behavior:
          - Prefer assigning a *shaped* ndarray to qleaf.values (nz,nr,nphi), so the backend
            can serialize `values_SHAPE` correctly.
          - Also populate qleaf.values_shape when the binding exposes it.
          - Fall back to 1D assignment if shaped assignment is not accepted.
        """
        vals = _as_f64(values_1d)
        shp = np.asarray(list(shape_hint), dtype=np.int32).ravel()
        try:
            if shp.size == 3:
                # NIMROD block data are assembled in (nr, nz, nphi) ordering and
                # flattened with Fortran-order in the writer. Preserve this here
                # so downstream tools can reliably restore dimensionality.
                nr_, nz_, nphi_ = (int(shp[0]), int(shp[1]), int(shp[2]))
                qleaf.values = vals.reshape((nr_, nz_, nphi_), order="F")
            else:
                qleaf.values = vals
        except Exception:
            try:
                qleaf.values = vals
            except Exception:
                return

        # Try common attribute spellings used by different bindings / DD versions.
        for attr in ("values_shape", "valuesShape", "values_SHAPE"):
            try:
                leaf = getattr(qleaf, attr)
            except Exception:
                leaf = None
            if leaf is None:
                continue
            try:
                try:
                    leaf.resize(int(shp.size))
                    leaf[:] = shp
                except Exception:
                    setattr(qleaf, attr, shp)
                break
            except Exception:
                continue
    if not getattr(args, "ggd_unstructured", False):
        # Structured-GGD mode: store explicit axes and a single “nodes” subset.
        # For the toroidal axis, the IMAS HDF5 backend can pad shorter vectors
        # up to max(axis_len) across R/Z/phi. To make the true nphi recoverable
        # without DD-dependent heuristics, we write a phi vector padded with
        # NaNs beyond the physical nphi.
        _fill_space(g.space[0], "R", np.asarray([0.0, 1.0]))
        _fill_space(g.space[1], "Z", np.asarray([0.0, 1.0]))
        if nphi > 1:
            _fill_space(g.space[2], "phi", np.asarray([0.0, 1.0]))

        try:
            g.grid_subset.resize(1)
            gs = g.grid_subset[0]
            gs.identifier.name = "nodes"
            gs.identifier.index = 1
            gs.identifier.description = "All nodes of the structured grid"
            gs.dimension = 1
        except Exception:
            pass

    q = mhd.ggd[it]

    # electrons.temperature
    rc, zc, Te3 = _make_values("teq", "rete", "imte")
    if Te3 is not None:
        if not getattr(args, "ggd_unstructured", False):
            _fill_space(g.space[0], "R", rc)
            _fill_space(g.space[1], "Z", zc)
            if nphi > 1:
                # Avoid ambiguity from implicit axis padding by explicitly padding
                # phi beyond the physical nphi with NaNs.
                phi_pad = np.full(len(rc), np.nan, dtype=float)
                phi_pad[: int(nphi)] = phi_list
                _fill_space(g.space[2], "phi", phi_pad)
        vals = np.asarray(Te3, dtype=float).ravel(order="F")
        try:
            q.electrons.temperature.resize(1)
            qt = q.electrons.temperature[0]
            qt.grid_index = 1
            qt.grid_subset_index = 1
            _set_values_and_shape(qt, vals, (len(rc), len(zc), int(Te3.shape[2])))
        except Exception:
            pass

    # t_i_average
    rc2, zc2, Ti3 = _make_values("tiq", "reti", "imti")
    if Ti3 is not None:
        if not getattr(args, "ggd_unstructured", False):
            _fill_space(g.space[0], "R", rc2)
            _fill_space(g.space[1], "Z", zc2)
            if nphi > 1:
                phi_pad = np.full(int(len(rc2)), np.nan, dtype=float)
                phi_pad[: int(nphi)] = phi_list
                _fill_space(g.space[2], "phi", phi_pad)
        vals = np.asarray(Ti3, dtype=float).ravel(order="F")
        try:
            q.t_i_average.resize(1)
            qt = q.t_i_average[0]
            qt.grid_index = 1
            qt.grid_subset_index = 1
            _set_values_and_shape(qt, vals, (len(rc2), len(zc2), int(Ti3.shape[2])))
        except Exception:
            pass

    # n_i_total
    rc3, zc3, n3 = _make_values("nq", "rend", "imnd")
    if n3 is not None:
        if not getattr(args, "ggd_unstructured", False):
            _fill_space(g.space[0], "R", rc3)
            _fill_space(g.space[1], "Z", zc3)
            if nphi > 1:
                phi_pad = np.full(int(len(rc3)), np.nan, dtype=float)
                phi_pad[: int(nphi)] = phi_list
                _fill_space(g.space[2], "phi", phi_pad)
        vals = np.asarray(n3, dtype=float).ravel(order="F")
        try:
            q.n_i_total.resize(1)
            qt = q.n_i_total[0]
            qt.grid_index = 1
            qt.grid_subset_index = 1
            _set_values_and_shape(qt, vals, (len(rc3), len(zc3), int(n3.shape[2])))
        except Exception:
            pass



def _write_unstructured_ggd_aux_h5(
    entry_dir: str,
    ids_name: str,
    occ: int,
    data: Dict[str, Any],
    args,
) -> None:
    """
    Write unstructured node coordinates (and optional connectivity) into the IDS HDF5 file
    using h5py, under a NIMROD-specific auxiliary group. This is intended as a *performance*
    escape hatch: we avoid slow per-element population of the official IDS unstructured grid
    leaves while still storing a documented, explicit node list + connectivity that downstream
    tools (e.g. plot_mhd.py) can consume.

    The auxiliary group is:
        /{ids_name}_{occ}/nimrod_unstructured
    with datasets:
        - nodes: float64, shape (Nnodes, 3) = (R, Z, phi)
        - connectivity (optional): int32, shape (Ncells, 8), 1-based node indices (hexahedra)
        - nr, nz, nphi: int32 scalars (reconstruction grid)
        - r_axis, z_axis, phi_axis_used: float64 vectors
    """
    if not getattr(args, "ggd_unstructured", False):
        return
    if not getattr(args, "ggd_h5py_direct", False):
        # The caller requested unstructured, but not direct HDF5 writing.
        # At the moment we only support the h5py fast-path (it is also what you want for performance).
        raise RuntimeError("Unstructured GGD writing is currently supported via --ggd-h5py-direct.")

    import numpy as _np
    import h5py as _h5py

    fn = os.path.join(entry_dir, f"{ids_name}_{occ}.h5")
    if not os.path.exists(fn):
        raise FileNotFoundError(fn)

    # Use the same reconstruction grid parameters as populate_mhd_ggd().
    nb = int(getattr(args, "ggd_nbins", 128))
    nphi = int(getattr(args, "ggd_nphi", 8))
    # R/Z axes (uniform) based on r/z extents of the stitched dump.
    # Prefer the already-computed extents in data, fall back to reconstruct from 'r2d','z2d' if present.
    rmin = float(data.get("rmin", _np.nan))
    rmax = float(data.get("rmax", _np.nan))
    zmin = float(data.get("zmin", _np.nan))
    zmax = float(data.get("zmax", _np.nan))
    if not _np.isfinite(rmin) or not _np.isfinite(rmax) or not _np.isfinite(zmin) or not _np.isfinite(zmax):
        # Fallback: infer from grid coordinates if available.
        rr = data.get("R")
        zz = data.get("Z")
        if rr is not None and zz is not None:
            rmin, rmax = float(_np.nanmin(rr)), float(_np.nanmax(rr))
            zmin, zmax = float(_np.nanmin(zz)), float(_np.nanmax(zz))
        else:
            raise RuntimeError("Cannot infer R/Z extents for unstructured node export (missing rmin/rmax/zmin/zmax and R/Z grids).")

    r_axis = _np.linspace(rmin, rmax, nb, dtype=_np.float64)
    z_axis = _np.linspace(zmin, zmax, nb, dtype=_np.float64)

    # "Used" phi axis is the reconstructed toroidal sampling used for values (nphi).
    phi_axis = _np.linspace(0.0, 2.0 * _np.pi, nphi, endpoint=False, dtype=_np.float64)

    # Build node coordinates for the Cartesian product grid (R,Z,phi); flatten with r-fast (C order).
    # Nodes are stored as (R, Z, phi).
    RR, ZZ = _np.meshgrid(r_axis, z_axis, indexing="ij")  # (nr, nz)
    rr_flat = RR.reshape(-1, order="C")
    zz_flat = ZZ.reshape(-1, order="C")
    n2 = rr_flat.size
    nodes = _np.empty((n2 * nphi, 3), dtype=_np.float64)
    for k, ph in enumerate(phi_axis):
        sl = slice(k * n2, (k + 1) * n2)
        nodes[sl, 0] = rr_flat
        nodes[sl, 1] = zz_flat
        nodes[sl, 2] = ph

    connectivity = None
    if getattr(args, "ggd_connectivity", "none") == "hex":
        if nb < 2 or nphi < 2:
            raise RuntimeError("hex connectivity requires nbins>=2 and nphi>=2.")
        # Hexahedra between adjacent (i,j,k) cells; periodic in phi.
        nr, nz = nb, nb
        ncell = (nr - 1) * (nz - 1) * nphi
        conn = _np.empty((ncell, 8), dtype=_np.int32)

        def node_index(i, j, k):
            # 1-based indexing for IMAS conventions.
            return 1 + k * (nr * nz) + j * nr + i

        c = 0
        for k in range(nphi):
            kp = (k + 1) % nphi
            for j in range(nz - 1):
                for i in range(nr - 1):
                    conn[c, :] = [
                        node_index(i, j, k),
                        node_index(i + 1, j, k),
                        node_index(i + 1, j + 1, k),
                        node_index(i, j + 1, k),
                        node_index(i, j, kp),
                        node_index(i + 1, j, kp),
                        node_index(i + 1, j + 1, kp),
                        node_index(i, j + 1, kp),
                    ]
                    c += 1
        connectivity = conn

    # Write/update datasets (overwriting if they already exist).
    group_path = f"/{ids_name}_{occ}/nimrod_unstructured"
    with _h5py.File(fn, "a") as h5:
        g = h5.require_group(group_path)
        for key, arr in [
            ("r_axis", r_axis),
            ("z_axis", z_axis),
            ("phi_axis_used", phi_axis),
        ]:
            if key in g:
                del g[key]
            g.create_dataset(key, data=arr)
        for key, val in [("nr", nb), ("nz", nb), ("nphi", nphi)]:
            if key in g:
                del g[key]
            g.create_dataset(key, data=_np.int32(val))
        if "nodes" in g:
            del g["nodes"]
        g.create_dataset("nodes", data=nodes, compression="gzip", compression_opts=1, shuffle=True)
        if connectivity is not None:
            if "connectivity" in g:
                del g["connectivity"]
            g.create_dataset("connectivity", data=connectivity, compression="gzip", compression_opts=1, shuffle=True)

        # Best-effort: patch IMAS-generated `...&values_SHAPE` datasets so downstream tools can
        # restore (nr, nz, nphi) without heuristics.
        #
        # Observed in practice: some backends leave values_SHAPE as a scalar (1,1,1) or unset;
        # it is resizable (UNLIMITED) so we can safely resize the last dim to 3.
        try:
            parent = h5.get(f"/{ids_name}_{occ}")
            if parent is not None:
                shp = _np.asarray([nb, nb, nphi], dtype=_np.int32)
                for name, ds in parent.items():
                    if not isinstance(ds, _h5py.Dataset):
                        continue
                    if ("ggd[]&" in name) and name.endswith("&values_SHAPE"):
                        # Expected rank=3: (time, channel, ndim)
                        if ds.ndim == 3:
                            new_shape = (ds.shape[0], ds.shape[1], 3)
                            if ds.shape != new_shape:
                                ds.resize(new_shape)
                            ds[0, 0, :] = shp
        except Exception:
            # Non-fatal; plotting can still infer shape from nr/nz/nphi.
            pass


def _write_unstructured_gridggd_packed_h5(
    entry_dir: str,
    ids_name: str,
    occ: int,
    nodes_xyz: "np.ndarray",
    connectivity: "np.ndarray",
    n_subsets: int = 2,
    nodes_subset_index: int = 0,
    vols_subset_index: int = 1,
    *,
    log: "logging.Logger | None" = None,
) -> None:
    """Populate IMAS grid_ggd (HDF5 backend) using a packed writer.

    Motivation: the python bindings implementation of GGD unstructured connectivity
    requires nested loops over elements/objects and becomes a bottleneck for large
    meshes. Here we write directly into the HDF5 datasets created by the IMAS
    HDF5 backend.

    Strategy:
      1) We assume the IDS writer has already created *some* grid_ggd / grid_subset
         scaffolding (even a tiny placeholder). That creates the correct dataset
         names/layout for the active DD/bindings.
      2) We locate the relevant datasets by pattern matching and overwrite them
         in bulk using numpy.

    If the expected datasets are not present, we log a warning and return.
    """

    import os
    import re
    import h5py
    import numpy as np

    if log is None:
        log = logging.getLogger(__name__)

    h5_path = os.path.join(entry_dir, f"{ids_name}_{occ}.h5")
    grp_name = f"{ids_name}_{occ}"

    if not os.path.exists(h5_path):
        log.warning("Packed grid_ggd writer: HDF5 file not found: %s", h5_path)
        return

    nodes_xyz = np.asarray(nodes_xyz)
    connectivity = np.asarray(connectivity)
    if nodes_xyz.ndim != 2 or nodes_xyz.shape[1] != 3:
        raise ValueError(f"nodes_xyz must have shape (N,3); got {nodes_xyz.shape}")
    if connectivity.ndim != 2:
        raise ValueError(f"connectivity must be 2D (Ncells,Nverts); got {connectivity.shape}")

    n_nodes = int(nodes_xyz.shape[0])
    n_cells = int(connectivity.shape[0])
    n_verts = int(connectivity.shape[1])

    # Expected backend dataset names (but allow minor DD/bindings variations).
    pat_subset_aos = re.compile(r"^grid_ggd\[\]\&grid_subset\[\]\&AOS_SHAPE$")
    pat_elem_aos = re.compile(r"^grid_ggd\[\]\&grid_subset\[\]\&element\[\]\&AOS_SHAPE$")
    pat_obj_aos = re.compile(r"^grid_ggd\[\]\&grid_subset\[\]\&element\[\]\&object\[\]\&AOS_SHAPE$")

    pat_real = re.compile(r"^grid_ggd\[\]\&grid_subset\[\]\&element\[\]\&object\[\]\&real$")
    pat_index = re.compile(r"^grid_ggd\[\]\&grid_subset\[\]\&element\[\]\&object\[\]\&index$")

    with h5py.File(h5_path, "r+") as h5:
        if grp_name not in h5:
            log.warning("Packed grid_ggd writer: group '/%s' not found in %s", grp_name, h5_path)
            return
        g = h5[grp_name]

        # Discover datasets.
        keys = list(g.keys())
        ds_subset_aos = next((k for k in keys if pat_subset_aos.match(k)), None)
        ds_elem_aos = next((k for k in keys if pat_elem_aos.match(k)), None)
        ds_obj_aos = next((k for k in keys if pat_obj_aos.match(k)), None)
        ds_real = next((k for k in keys if pat_real.match(k)), None)
        ds_index = next((k for k in keys if pat_index.match(k)), None)

        if ds_subset_aos is None or ds_elem_aos is None or ds_real is None or ds_index is None:
            # Don't hard-fail; some DD/bindings store these leaves differently.
            log.warning(
                "Packed grid_ggd writer: could not find required datasets in /%s. "
                "Found keys include: %s",
                grp_name,
                ", ".join(keys[:40]) + (" ..." if len(keys) > 40 else ""),
            )
            return

        # 1) grid_subset AOS_SHAPE: store number of subsets per ggd.
        dsa = g[ds_subset_aos]
        try:
            # Typical shape is (nggd, 1)
            dsa[...] = 0
            dsa[0, 0] = n_subsets
        except Exception as e:
            log.warning("Packed grid_ggd writer: failed to write %s: %s", ds_subset_aos, e)

        # 2) element AOS_SHAPE: store number of elements per subset.
        dea = g[ds_elem_aos]
        try:
            # Expected shape (nggd, nsubsets)
            if dea.shape[1] != n_subsets:
                # Resize if possible
                try:
                    dea.resize((dea.shape[0], n_subsets))
                except Exception:
                    pass
            dea[...] = 0
            dea[0, nodes_subset_index] = n_nodes
            dea[0, vols_subset_index] = n_cells
        except Exception as e:
            log.warning("Packed grid_ggd writer: failed to write %s: %s", ds_elem_aos, e)

        # 3) object AOS_SHAPE (optional): number of objects per element.
        if ds_obj_aos is not None:
            doa = g[ds_obj_aos]
            try:
                # Common patterns:
                #  - (nggd, nsubsets, nelem) : per-element object count
                #  - (nggd, nsubsets, 1)     : uniform count per subset
                if doa.ndim == 3:
                    # Resize element dimension if possible.
                    try:
                        max_e = max(n_nodes, n_cells)
                        if doa.shape[2] != max_e:
                            doa.resize((doa.shape[0], doa.shape[1], max_e))
                    except Exception:
                        pass
                    doa[...] = 0
                    doa[0, nodes_subset_index, :n_nodes] = 3
                    doa[0, vols_subset_index, :n_cells] = n_verts
                elif doa.ndim == 2:
                    # (nggd, nsubsets)
                    doa[...] = 0
                    doa[0, nodes_subset_index] = 3
                    doa[0, vols_subset_index] = n_verts
            except Exception as e:
                log.warning("Packed grid_ggd writer: failed to write %s: %s", ds_obj_aos, e)

        # 4) Bulk-write node coordinates (real) and cell connectivity (index).
        dreal = g[ds_real]
        dind = g[ds_index]

        # Prepare bulk arrays
        # IMAS indices are typically 1-based; ensure connectivity is 1-based.
        conn_1b = connectivity.astype(np.int32, copy=False)
        if conn_1b.min() == 0:
            conn_1b = conn_1b + 1

        # We support two common layouts:
        #   real:  (nggd, nsubsets, nelem, nobj)
        #   index: (nggd, nsubsets, nelem, nobj)
        if dreal.ndim == 4 and dind.ndim == 4:
            # Resize to hold maximum elements and objects.
            try:
                dreal.resize((1, n_subsets, n_nodes, 3))
            except Exception:
                pass
            try:
                dind.resize((1, n_subsets, n_cells, n_verts))
            except Exception:
                pass

            # Clear then write.
            try:
                dreal[...] = np.nan
            except Exception:
                pass
            try:
                dind[...] = 0
            except Exception:
                pass

            # Nodes subset
            dreal[0, nodes_subset_index, :n_nodes, :3] = nodes_xyz.astype(np.float64, copy=False)
            # Volumes subset: connectivity indices
            dind[0, vols_subset_index, :n_cells, :n_verts] = conn_1b

        else:
            # Fallback: try to write flattened payloads if the backend chose a packed 1D layout.
            # We do this best-effort and warn if it doesn't fit.
            flat_nodes = nodes_xyz.astype(np.float64, copy=False).reshape(-1)
            flat_conn = conn_1b.reshape(-1)

            wrote_any = False
            try:
                if dreal.ndim == 3:
                    # (nggd, nsubsets, nflat)
                    dreal.resize((1, n_subsets, flat_nodes.size))
                    dreal[...] = np.nan
                    dreal[0, nodes_subset_index, : flat_nodes.size] = flat_nodes
                    wrote_any = True
            except Exception as e:
                log.warning("Packed grid_ggd writer: could not write flattened real: %s", e)

            try:
                if dind.ndim == 3:
                    dind.resize((1, n_subsets, flat_conn.size))
                    dind[...] = 0
                    dind[0, vols_subset_index, : flat_conn.size] = flat_conn
                    wrote_any = True
            except Exception as e:
                log.warning("Packed grid_ggd writer: could not write flattened index: %s", e)

            if not wrote_any:
                log.warning(
                    "Packed grid_ggd writer: unsupported dataset ranks real=%s index=%s; leaving grid_ggd unchanged.",
                    getattr(dreal, "shape", None),
                    getattr(dind, "shape", None),
                )


def populate_mhd_linear(
    mhd: Any,
    data: Dict[str, Any],
    t_index: int,
    args,
    species_index: int,
    include_common_fields: bool,
) -> None:
    t = float(data["time"])
    fields: Dict[str, np.ndarray] = data["fields"]
    nmodes = int(data["nmodes"])
    keff = data["keff"]
    R = data["R"]
    Z = data["Z"]
    Nx, Ny = int(data["layout"]["Nx"]), int(data["layout"]["Ny"])

    # append time slice
    idx = _append_time_mhd(mhd, t)
    ts = mhd.time_slice[idx]
    ts.time = float(t)

    # toroidal modes (if present in schema)
    if nmodes > 0 and hasattr(ts, "toroidal_mode"):
        cur = _aos_len(ts.toroidal_mode)
        if cur != nmodes:
            try:
                ts.toroidal_mode.resize(nmodes)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to resize mhd_linear.time_slice[{idx}].toroidal_mode from {cur} to {nmodes}: {e}"
                )

    # density perturbations
    dens_re = None
    dens_im = None
    nspec_dens = 0
    if "rend" in fields and "imnd" in fields:
        dens_re = fields["rend"]  # (Ny,Nx,nspec,nmodes)
        dens_im = fields["imnd"]
        nspec_dens = int(np.ravel(fields.get("nspec_dens", np.array([dens_re.shape[2]], dtype=int)))[0])

    # ion mass for converting number density perturbation to mass density (optional)
    mpart = float(args.ion_mass_amu) * 1.66053906660e-27  # kg

    has_toroidal = hasattr(ts, "toroidal_mode")
    nmodes_eff = int(nmodes) if (has_toroidal and int(nmodes) > 0) else max(1, int(nmodes))

    for m in range(nmodes_eff):
        if has_toroidal and int(nmodes) > 0:
            tm = ts.toroidal_mode[m]
            # Store toroidal mode number. IMAS DD >=4.0 may rename n_tor -> n_phi.
            nval = None
            for _k in ("n_tor", "n_phi"):
                _arr = data.get(_k)
                if _arr is not None and int(np.size(_arr)) > m:
                    try:
                        nval = float(np.ravel(_arr)[m])
                        break
                    except Exception:
                        pass
            if nval is None:
                # Fallback: many NIMROD dumps store mode identifiers in `keff`.
                try:
                    nval = float(np.ravel(keff)[m])
                except Exception:
                    nval = float(m)
            nval_i = int(round(nval))
            for _attr in ("n_phi", "n_tor"):
                if hasattr(tm, _attr):
                    try:
                        setattr(tm, _attr, nval_i)
                    except Exception:
                        pass
            try:
                tm.omega = 0.0
            except Exception:
                pass
        else:
            tm = ts

        pl = tm.plasma if hasattr(tm, "plasma") else tm

        # grid


        # grid + RZ coordinates must always be present (for plotting)
        _set_rz_grid_mhd(pl, R, Z, Nx=Nx, Ny=Ny)

        # density pert for this species
        if dens_re is not None and dens_im is not None and nspec_dens > 0:
            s = int(species_index)
            if s >= dens_re.shape[2]:
                s = 0
            ndre = dens_re[:, :, s, m]
            ndim = dens_im[:, :, s, m]
            _set_complex_scalar(
                pl,
                ("mass_density_perturbed", "rho_perturbed", "density_perturbed"),
                ndre * mpart,
                ndim * mpart,
            )

        # temperature perturbation: pair Te with electron density occurrence and Ti with ion density occurrences
        # NIMROD provides rete/imte (electron) and reti/imti (ion). IMAS mhd_linear has a single temperature_perturbed,
        # so we store Te in the electron occurrence and Ti in ion occurrences.
        eidx = int(getattr(args, 'electrons_index', 0))
        if int(species_index) == int(eidx):
            if 'rete' in fields and 'imte' in fields:
                _set_complex_scalar(pl, ('temperature_perturbed','t_perturbed'), fields['rete'][:, :, m], fields['imte'][:, :, m])
        else:
            if 'reti' in fields and 'imti' in fields:
                _set_complex_scalar(pl, ('temperature_perturbed','t_perturbed'), fields['reti'][:, :, m], fields['imti'][:, :, m])

        
        # --- species-dependent equilibrium scalars (stored in this species occurrence) ---
        # Note: IMAS expects density [m^-3], temperature [eV], pressure [Pa].
        echarge = 1.602176634e-19  # J/eV
        nq2d = data.get("nq", None)
        teq2d = data.get("teq", None)
        tiq2d = data.get("tiq", None)
        prq2d = data.get("prq", None)
        peq2d = data.get("peq", None)

        # select equilibrium number density for this occurrence (fallback to the only available species)
        n0 = None
        if nq2d is not None:
            if getattr(nq2d, "ndim", 0) == 3:
                ssel = int(species_index)
                if ssel >= int(nq2d.shape[2]):
                    ssel = 0
                n0 = nq2d[:, :, ssel]
            else:
                n0 = nq2d

        is_electron = (int(species_index) == int(getattr(args, "electrons_index", 0)))

        # equilibrium temperature for this species (or derived if not present)
        T0 = None
        if is_electron:
            if teq2d is not None:
                T0 = teq2d[:, :, 0] if getattr(teq2d, "ndim", 0) == 3 else teq2d
            elif peq2d is not None and n0 is not None:
                # Te [eV] = pe [Pa] / (n [m^-3] * e [J/eV])
                T0 = (peq2d[:, :, 0] if getattr(peq2d, "ndim", 0) == 3 else peq2d) / (np.maximum(n0, 1e-60) * echarge)
        else:
            if tiq2d is not None:
                T0 = tiq2d[:, :, 0] if getattr(tiq2d, "ndim", 0) == 3 else tiq2d
            elif prq2d is not None and n0 is not None:
                # if electron pressure is available, use (p_total - p_e) for the single-ion case
                p_tot = prq2d[:, :, 0] if getattr(prq2d, "ndim", 0) == 3 else prq2d
                if peq2d is not None and int(getattr(data, "nspec_eq", data.get("nspec_eq", 0)) or 0) <= 1:
                    p_e = peq2d[:, :, 0] if getattr(peq2d, "ndim", 0) == 3 else peq2d
                    p_ion = p_tot - p_e
                    T0 = p_ion / (np.maximum(n0, 1e-60) * echarge)
                else:
                    T0 = p_tot / (np.maximum(n0, 1e-60) * echarge)

        # equilibrium pressure for this species (or derived)
        p0 = None
        if is_electron and peq2d is not None:
            p0 = peq2d[:, :, 0] if getattr(peq2d, "ndim", 0) == 3 else peq2d
        elif (not is_electron) and prq2d is not None and peq2d is not None and int(data.get("nspec_eq", 0) or 0) <= 1:
            # single-ion: p_i = p_total - p_e
            p_tot = prq2d[:, :, 0] if getattr(prq2d, "ndim", 0) == 3 else prq2d
            p_e = peq2d[:, :, 0] if getattr(peq2d, "ndim", 0) == 3 else peq2d
            p0 = p_tot - p_e
        elif n0 is not None and T0 is not None:
            p0 = n0 * T0 * echarge

        # populate equilibrium scalars in mhd_linear/mhd plasma if schema supports them
        _set_real_scalar(pl, ("density", "number_density", "n"), n0)
        _set_real_scalar(pl, ("temperature", "t"), T0)
        _set_real_scalar(pl, ("pressure", "p"), p0)

        # --- species-dependent pressure perturbation ---
        dp_re = None
        dp_im = None

        # prefer explicit electron pressure perturbation when available
        if is_electron and ("repe" in fields) and ("impe" in fields):
            dp_re = fields["repe"][:, :, m]
            dp_im = fields["impe"][:, :, m]
        # packed per-species pressure perturbation (rare but supported)
        elif ("repr" in fields) and ("impr" in fields) and getattr(fields["repr"], "ndim", 0) == 4:
            ssel = int(species_index)
            if ssel >= int(fields["repr"].shape[2]):
                ssel = 0
            dp_re = fields["repr"][:, :, ssel, m]
            dp_im = fields["impr"][:, :, ssel, m]
        # total pressure perturbation (repr/impr are (Nx,Ny,nmodes)) + electron split for single-ion case
        elif ("repr" in fields) and ("impr" in fields):
            dp_tot_re = fields["repr"][:, :, m]
            dp_tot_im = fields["impr"][:, :, m]
            if (not is_electron) and ("repe" in fields) and ("impe" in fields) and int(data.get("nspec_eq", 0) or 0) <= 1:
                dp_re = dp_tot_re - fields["repe"][:, :, m]
                dp_im = dp_tot_im - fields["impe"][:, :, m]
            elif is_electron:
                dp_re, dp_im = dp_tot_re, dp_tot_im

        # derive from (dn, dT) if still not available
        if dp_re is None or dp_im is None:
            dn_re = None
            dn_im = None
            if dens_re is not None and dens_im is not None and nspec_dens > 0:
                ssel = int(species_index)
                if ssel >= dens_re.shape[2]:
                    ssel = 0
                dn_re = dens_re[:, :, ssel, m]
                dn_im = dens_im[:, :, ssel, m]
            # temperature perturbation (electron or ion)
            dT_re = None
            dT_im = None
            if is_electron and ("rete" in fields) and ("imte" in fields):
                dT_re = fields["rete"][:, :, m]
                dT_im = fields["imte"][:, :, m]
            elif (not is_electron) and ("reti" in fields) and ("imti" in fields):
                dT_re = fields["reti"][:, :, m]
                dT_im = fields["imti"][:, :, m]

            if (dn_re is not None) and (dn_im is not None) and (n0 is not None) and (T0 is not None) and (dT_re is not None) and (dT_im is not None):
                # dp = e * (dn*T0 + n0*dT)  (computed separately for real/imag parts)
                dp_re = (dn_re * T0 + n0 * dT_re) * echarge
                dp_im = (dn_im * T0 + n0 * dT_im) * echarge

        if dp_re is not None and dp_im is not None:
            _set_complex_scalar(pl, ("pressure_perturbed", "p_perturbed"), dp_re, dp_im)

        # Velocity/rotation perturbations correspond to the bulk (ion) flow in NIMROD MHD output.
        # Store them in the main-ion occurrence only (species_index == 1).
        if (not is_electron) and int(species_index) == 1:
            if "reve" in fields and "imve" in fields:
                Vre = fields["reve"][:, :, m, 0]
                Vze = fields["reve"][:, :, m, 1]
                Vpe = fields["reve"][:, :, m, 2]
                Vim = fields["imve"][:, :, m, 0]
                Vzm = fields["imve"][:, :, m, 1]
                Vpm = fields["imve"][:, :, m, 2]
                _set_complex_vector(
                    pl,
                    ("velocity_perturbed", "v_perturbed"),
                    (Vre, Vze, Vpe),
                    (Vim, Vzm, Vpm),
                )

        if not include_common_fields:
            # Do not repeat B/V/J/p and other common fields
            continue

        # B perturbation
        if "rebe" in fields and "imbe" in fields:
            Bre = fields["rebe"][:, :, m, 0]
            Bze = fields["rebe"][:, :, m, 1]
            Bpe = fields["rebe"][:, :, m, 2]
            Bim = fields["imbe"][:, :, m, 0]
            Bzm = fields["imbe"][:, :, m, 1]
            Bpm = fields["imbe"][:, :, m, 2]
            _set_complex_vector(
                pl,
                ("b_field_perturbed", "b_field_perturbation", "magnetic_field_perturbed"),
                (Bre, Bze, Bpe),
                (Bim, Bzm, Bpm),
            )


        # Current density perturbation (requested)
        if "reja" in fields and "imja" in fields:
            Jre = fields["reja"][:, :, m, :]
            Jim = fields["imja"][:, :, m, :]
            _set_complex_vector(
                pl,
                ("current_density_perturbed", "j_perturbed", "current_perturbed"),
                (Jre[:, :, 0], Jre[:, :, 1], Jre[:, :, 2]),
                (Jim[:, :, 0], Jim[:, :, 1], Jim[:, :, 2]),
            )
# -----------------------------
# Main
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert NIMROD dumpgll HDF5 to IMAS equilibrium/core_profiles/mhd_linear (stitched RZ arrays).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("dumpgll", nargs="+", help="Input dumpgll HDF5 file(s)")

    p.add_argument("--dd", required=True, help="DB name (directory name), e.g. nstx")
    p.add_argument("--pulse", type=int, required=True)
    p.add_argument("--run", type=int, required=True)
    p.add_argument("--backend", default="hdf5", choices=("hdf5",), help="IMAS backend")
    p.add_argument("--dbpath", default=".", help="DB root path (output directory)")
    p.add_argument("--dd-version-dir", choices=["major", "full"], default="major",
                   help="Directory component for DD version (default: major, e.g. 3 for 3.42.0)")
    p.add_argument("--dd-version", default=None, help="IMAS data dictionary version, e.g. 3.42.0")

    p.add_argument("--mode", default="a", help="DBEntry open mode: r/a/w/x (r+/rw accepted and mapped to a)")

    p.add_argument(
        "--occ-base",
        type=int,
        default=1,
        help="Base occurrence for mhd/mhd_linear, equilibrium, and core_profiles IDSs (species occurrences use occ_base + species_index).",
    )

    p.add_argument("--nbins", type=int, default=256, help="Number of bins for 1D profiles")

    # MHD (GGD) output controls (nonlinear runs)
    p.add_argument(
        "--ggd-nbins",
        type=int,
        default=128,
        help="Downsample poloidal-plane fields onto a regular R-Z grid with this many bins in each direction when writing mhd.ggd (nonlinear runs).",
    )
    p.add_argument(
        "--ggd-nphi",
        type=int,
        default=8,
        help="Number of toroidal angle samples (uniform in [0,2pi)) used to reconstruct full fields from Fourier modes for mhd.ggd (nonlinear runs). Set to 1 to store a single toroidal cut at phi=0.",
    )



    # GGD output style: structured (default) vs unstructured-with-connectivity.
    # NOTE: The unstructured option is primarily meant to support robust downstream reconstruction
    # of 3D array shapes and connectivity in environments where the backend stores packed value arrays.
    p.add_argument(
        "--ggd-unstructured",
        action="store_true",
        help="In addition to the standard GGD fields, write unstructured node coordinates (and optional connectivity) for the mhd.ggd grid into the output HDF5 using h5py (under a NIMROD-specific auxiliary group).",
    )
    p.add_argument(
        "--ggd-connectivity",
        choices=["none", "hex"],
        default="none",
        help="Connectivity type to write when --ggd-unstructured is enabled. 'hex' writes hexahedral connectivity on the reconstructed (R,Z,phi) grid with periodicity in phi. Default: none.",
    )
    p.add_argument(
        "--ggd-h5py",
        dest="ggd_h5py_direct",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--ggd-h5py-direct",
        dest="ggd_h5py_direct",
        action="store_true",
        help="When --ggd-unstructured is enabled, write node coordinates/connectivity directly into the IDS HDF5 file using h5py after IMAS put(). This avoids slow per-element IDS population. Recommended for large grids.",
    )

    p.add_argument(
        "--ggd-gridggd-packed",
        dest="ggd_gridggd_packed",
        action="store_true",
        help=(
            "When --ggd-unstructured is enabled, also populate the *official* IDS grid_ggd tree (grid_subset/element/object) using a packed HDF5 writer. "
            "This keeps your fast --ggd-h5py-direct aux mesh AND provides grid_ggd content for tools that expect it. "
            "Implementation is best-effort across DD/bindings; validate with h5dump/IMAS readers."
        ),
    )

    p.add_argument(
        "--time",
        type=float,
        default=None,
        help="Override time (single value applied to all files). If not set, uses dumpTime.vsTime when available.",
    )

    p.add_argument(
        "--dens-pert-order",
        choices=("species_major", "mode_major"),
        default="species_major",
        help="How rend/imnd are packed if stored as (ny,nx,nspec*nmodes)",
    )

    p.add_argument(
        "--ion-mass-amu",
        type=float,
        default=2.0,
        help="Ion mass (amu) used to convert number density perturbation to mass density perturbation",
    )

    p.add_argument(
        "--electrons-index",
        type=int,
        default=0,
        help="Index in nq/rend/imnd corresponding to electrons (used to pair density with Te perturbation).",
    )

    
    p.add_argument("--p-scale", type=float, default=1.0,
                   help="Scale factor applied to pressure-like quantities from dump -> IMAS (e.g., kPa->Pa: 1e3).")
    p.add_argument("--T-scale", type=float, default=1.0,
                   help="Scale factor applied to temperature-like quantities from dump -> IMAS (e.g., keV->eV: 1e3).")
    p.add_argument("--n-scale", type=float, default=1.0,
                   help="Scale factor applied to number-density-like quantities from dump -> IMAS (e.g., cm^-3->m^-3: 1e6).")
    p.add_argument("--B-scale", type=float, default=1.0,
                   help="Scale factor applied to magnetic-field-like quantities from dump -> IMAS (e.g., Gauss->T: 1e-4).")
    p.add_argument("--L-scale", type=float, default=1.0,
                   help="Scale factor applied to length-like quantities from dump -> IMAS (e.g., cm->m: 1e-2).")
    p.add_argument("--v-scale", type=float, default=1.0,
                   help="Scale factor applied to velocity-like quantities from dump -> IMAS (e.g., cm/s->m/s: 1e-2).")
    p.add_argument("--j-scale", type=float, default=1.0,
                   help="Scale factor applied to current-density-like quantities from dump -> IMAS (e.g., A/cm^2->A/m^2: 1e4).")

    p.add_argument("--quiet", action="store_true")

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
 
    dump_files = [Path(x).expanduser().resolve() for x in args.dumpgll]
    for fn in dump_files:
        if not fn.exists():
            _die(f"Input file not found: {fn}")

    dd_version = args.dd_version
    if (dd_version == None):
         dd_version = os.environ.get("IMAS_VERSION", '3.42.0')

    #imas = _import_imas()
    mode = _normalize_mode(str(args.mode))

    dbpath = Path(args.dbpath).expanduser().resolve()
    entry_dir = _entry_dir(dbpath, str(args.dd), str(dd_version), int(args.pulse), int(args.run), dd_version_dir=str(args.dd_version_dir))

    _log(f"IMAS DB root: {dbpath}", args.quiet)
    _log(f"IMAS entry directory: {entry_dir}", args.quiet)

    db = _open_db(imas, args.backend, entry_dir, mode, str(dd_version))
    factory = _ids_factory(imas, str(dd_version))

    # Important implementation detail:
    # We append time slices using put_slice() to avoid corruption issues observed
    # when repeatedly db.get() -> resize(AOS) -> db.put() on HDF5 entries.
    # (Symptom: only the last time slice is valid; earlier slices read back as
    # huge nonsensical floating point values.)

    for ifile, fn in enumerate(dump_files, start=1):
        _log(f"Reading {fn.name} ({ifile}/{len(dump_files)})", args.quiet)
        data = read_and_stitch_dump(fn, args)

        lay = data["layout"]
        _log(
            f"  blocks={lay['nxbl']}x{lay['nybl']} ordering={lay['ordering']} local(ny,nx)=({lay['ny_loc']},{lay['nx_loc']}) global(Ny,Nx)=({lay['Ny']},{lay['Nx']})",
            args.quiet,
        )
        _log(f"  nmodes={data['nmodes']} time={data['time']}", args.quiet)

        # Determine whether this directory corresponds to a nonlinear run (nimrod.in) and choose IDS type.
        nimrod_in_path = None
        try:
            cand = fn.parent / "nimrod.in"
            if cand.is_file():
                nimrod_in_path = str(cand)
        except Exception:
            nimrod_in_path = None

        nonlinear_flag = _nimrod_in_nonlinear(nimrod_in_path)
        try:
            args._nimrod_species = _nimrod_species_info(nimrod_in_path)
        except Exception:
            args._nimrod_species = {}

        # For nonlinear runs (nimrod.in present and nonlinear=T/true), write BOTH:
        #   - mhd_linear: mode-resolved perturbations (multiple occurrences for multiple species)
        #   - mhd:        reconstructed full fields in GGD form (time-dependent)
        # For linear runs, write only mhd_linear.
        write_mhd = (nonlinear_flag is True)

        # Always instantiate mhd_linear for output
        mhd_linear = factory.new("mhd_linear") if hasattr(factory, "new") else factory("mhd_linear")

        # Instantiate mhd (GGD-based) only for nonlinear runs
        mhd = None
        if write_mhd:
            try:
                mhd = factory.new("mhd") if hasattr(factory, "new") else factory("mhd")
            except Exception:
                _log("[warn] Failed to instantiate mhd IDS (GGD-based). Continuing with mhd_linear only.", args.quiet)
                mhd = None

# By default, keep occurrences aligned: equilibrium/core_profiles occurrence == mhd(_linear) base occurrence.
        # Occurrence handling:
        #   - Use --occ-base as the single source of truth.
        #   - Accept legacy flags (suppressed from --help) for backward compatibility.
        occ_base = int(getattr(args, "occ_base", 0) or 0)

        # Keep all IDS occurrences aligned to occ_base.
        eq_occ = occ_base
        cp_occ = occ_base

        # Populate and store equilibrium/core_profiles (if available in the dump).
        eq = factory.new("equilibrium") if hasattr(factory, "new") else factory("equilibrium")
        populate_equilibrium(eq, data, t_index=0, quiet=args.quiet)
        # Only write equilibrium if we actually appended a time_slice (avoid time-only placeholders)
        try:
            if hasattr(eq, "time_slice") and len(eq.time_slice) > 0:
                _db_put_slice(db, eq, eq_occ)
        except Exception:
            pass

        cp = factory.new("core_profiles") if hasattr(factory, "new") else factory("core_profiles")
        populate_core_profiles(cp, data, t_index=0, args=args)
        # Only write core_profiles if we actually appended profiles_1d (avoid empty placeholders)
        try:
            if hasattr(cp, "profiles_1d") and len(cp.profiles_1d) > 0:
                _db_put_slice(db, cp, cp_occ)
        except Exception:
            pass

        # mhd_linear/mhd per-species occurrence:
        #   occurrence=occ_base+0 : electrons
        #   occurrence=occ_base+1 : main ions
        #   occurrence=occ_base+2.. : impurities (if present)
        fields = data.get("fields", {})
        nspec_eq = int(data.get("nspec_eq", 0) or 0)
        nspec_dens = 0
        if "nspec_dens" in fields:
            try:
                nspec_dens = int(np.ravel(fields["nspec_dens"])[0])
            except Exception:
                nspec_dens = 0

        nspec_out = max(nspec_eq, nspec_dens, 1)
        # If the dump does not carry an explicit multi-species dimension, still write electrons+ions.
        if nspec_out == 1:
            nspec_out = 2

        for s in range(nspec_out):
            occ = occ_base + int(s)
            mhd_ids = factory.new("mhd_linear") if hasattr(factory, "new") else factory("mhd_linear")

            include_common = (s == 0)  # only EM/common vector fields; species scalars are written for all
            populate_mhd_linear(
                mhd_ids,
                data,
                t_index=0,
                args=args,
                species_index=s,
                include_common_fields=include_common,
            )

            # annotate
            try:
                mhd_ids.code.name = "NIMROD"
            except Exception:
                pass
            try:
                # store nimrod.in XML if present
                if nimrod_in_path:
                    mhd_ids.code.parameters = _build_nimrod_xml(nimrod_in_path)
            except Exception:
                pass
            try:
                mhd_ids.ids_properties.comment = (
                    f"dump2imas: stitched grid; ids={ids_name}; occurrence={occ}; species_index={s}; "
                    f"dens_pert_order={args.dens_pert_order}; common_fields={include_common}; nonlinear={nonlinear_flag}"
                )
            except Exception:
                pass

            _db_put_slice(db, mhd_ids, occ)
        # For nonlinear runs, also write the GGD-based mhd IDS (full fields) alongside mhd_linear.
        if write_mhd and (mhd is not None):
            try:
                populate_mhd_ggd(mhd, data, args)
                # Store nimrod.in XML in mhd IDS as well (for nonlinear simulations)
                try:
                    if nimrod_in_path:
                        try:
                            mhd.code.name = "NIMROD"
                        except Exception:
                            pass
                        mhd.code.parameters = _build_nimrod_xml(nimrod_in_path)
                except Exception:
                    pass
                _db_put_slice(db, mhd, occ_base)
                # Optional: store unstructured node coordinates/connectivity for faster and unambiguous reconstruction.
                if getattr(args, 'ggd_unstructured', False) and getattr(args, 'ggd_h5py_direct', False):
                    try:
                        _write_unstructured_ggd_aux_h5(entry_dir, 'mhd', occ_base, data, args)
                        _log('Wrote nimrod_unstructured auxiliary datasets into mhd HDF5 (h5py)', args.quiet)
                    except Exception as _e:
                        _log(f"[warn] Could not write nimrod_unstructured auxiliary datasets: {_e}", args.quiet)

                # Optional (experimental): populate the official IDS grid_ggd with an unstructured representation
                # using a packed HDF5 writer. This avoids per-element IDS population overhead.
                if (
                    getattr(args, 'ggd_unstructured', False)
                    and getattr(args, 'ggd_h5py_direct', False)
                    and getattr(args, 'ggd_gridggd_packed', False)
                ):
                    try:
                        # Load the arrays we just wrote in the aux group to avoid recomputation.
                        import h5py
                        h5_path = os.path.join(entry_dir, f"mhd_{occ_base}.h5")
                        with h5py.File(h5_path, 'r') as _f:
                            grp = _f[f"mhd_{occ_base}"]
                            aux = grp.get('nimrod_unstructured', None)
                            if aux is None:
                                raise RuntimeError("nimrod_unstructured group missing; cannot pack-write grid_ggd")
                            nodes_xyz = aux['nodes'][...]
                            conn = aux['connectivity'][...]
                        _write_unstructured_gridggd_packed_h5(entry_dir, 'mhd', occ_base, nodes_xyz, conn)
                        _log('Packed-write populated mhd.grid_ggd (unstructured)', args.quiet)
                    except Exception as _e:
                        _log(f"[warn] Could not packed-write grid_ggd: {_e}", args.quiet)
                _log("Populated mhd IDS (GGD full-field snapshot)", args.quiet)
            except Exception as e:
                _log(f"[warn] Failed to populate/put mhd IDS (GGD): {e}", args.quiet)

        _log(f"Appended IDS slices for {fn.name}", args.quiet)

    try:
        db.close()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
