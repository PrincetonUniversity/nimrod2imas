#!/usr/bin/env python3
"""dump2imas.py

NIMROD dumpgll (HDF5) -> IMAS conversion.

This version restores *stitched* RZ arrays in IMAS (global grid), rather than the packed
rblock layout.

What it writes:
  * equilibrium (occurrence = --equilibrium-occ):
      - time_slice[...].profiles_2d[0] on stitched grid: R,Z, psi, B (and pressure if available)
  * core_profiles (occurrence = --core-profiles-occ):
      - profiles_1d[...]: profiles as functions of absolute poloidal flux psi (grid.psi)
        (rho_tor_norm is not filled unless required by schema validation in your IMAS build)
  * mhd_linear (occurrences = --mhd-linear-occ-base + species_index):
      - time_slice[...].toroidal_mode[...] perturbations on stitched grid
      - occurrence for species 0 contains the common fields (B,V,J,p,T, density)
      - occurrences for species>0 contain only density perturbation + grid + n_tor

Notes:
  * Supports multiple dump files; appends new time slices when possible.
  * Supports rend/imnd packing ambiguity via --dens-pert-order {species_major,mode_major}.

Example:
  python dump2imas.py dumpgll.00000.h5 dumpgll.00010.h5 \
      --backend hdf5 --dbpath . --dd nimrod --dd-version 3.42.0 --pulse 201991 --run 1 \
      --equilibrium-occ 1 --core-profiles-occ 1 --mhd-linear-occ-base 10 \
      --dens-pert-order species_major
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import imas


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


def _entry_dir(dbpath: Path, dd: str, dd_version: str, pulse: int, run: int) -> Path:
    return dbpath / dd / str(dd_version) / str(pulse) / str(run)


def _open_db(imas: Any, backend: str, entry_dir: Path, mode: str, dd_version: str) -> Any:
    if backend != "hdf5":
        _die("Only --backend hdf5 is supported in this script version.")
    entry_dir.mkdir(parents=True, exist_ok=True)
    uri = f"imas:hdf5?path={entry_dir.resolve()}"

    # Ensure that the DBEntry and the IDS objects use the SAME DD major version.
    # By default IMAS-Python follows IMAS_VERSION (or latest), which can mismatch
    # the --dd-version used to build IDSs.
    try:
        return imas.DBEntry(uri, mode, dd_version=dd_version)
    except TypeError:
        # Backwards compatibility: fall back to IMAS_VERSION environment variable.
        old = os.environ.get("IMAS_VERSION")
        os.environ["IMAS_VERSION"] = str(dd_version)
        try:
            return imas.DBEntry(uri, mode)
        finally:
            if old is None:
                os.environ.pop("IMAS_VERSION", None)
            else:
                os.environ["IMAS_VERSION"] = old


def _ids_factory(imas: Any, dd_version: str) -> Any:
    # In modern IMAS Python, IDSFactory takes DD version.
    try:
        return imas.IDSFactory(dd_version)
    except Exception:
        # Some builds allow IDSFactory() without args.
        return imas.IDSFactory()

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

        psi_eq = read_scalar("psi_eq")
        bq = read_vec3("bq")
        prq = read_scalar("prq")
        teq = read_scalar("teq")
        tiq = read_scalar("tiq")
        vq = read_vec3("vq")
        jq = read_vec3("jq")

        # nq: can be (ny,nx) or (ny,nx,nspec)
        nq_blocks: List[np.ndarray] = []
        nspec_eq = 0
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
            for nm in ("repr", "impr", "rete", "imte", "reti", "imti"):
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

        R_imas = _swap01(R)
        Z_imas = _swap01(Z)
        psi_eq_imas = _swap01(psi_eq)
        bq_imas = _swap01(bq)
        prq_imas = _swap01(prq)
        teq_imas = _swap01(teq)
        tiq_imas = _swap01(tiq)
        nq_imas = _swap01(nq)
        vq_imas = _swap01(vq)
        jq_imas = _swap01(jq)

        fields_imas: Dict[str, np.ndarray] = {}
        for k, v in fields.items():
            try:
                fields_imas[k] = _swap01(v)
            except Exception:
                pass

        return dict(
            time=float(t0),
            keff=_as_f64(keff),
            nmodes=int(nmodes),
            R=_as_f64(R_imas),
            Z=_as_f64(Z_imas),
            psi_eq=_as_f64(psi_eq_imas),
            bq=_as_f64(bq_imas),
            prq=_as_f64(prq_imas),
            teq=_as_f64(teq_imas),
            tiq=_as_f64(tiq_imas),
            nq=_as_f64(nq_imas),
            vq=_as_f64(vq_imas),
            jq=_as_f64(jq_imas),
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
    jq = data["jq"]
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
    t = float(data["time"])
    psi2d = data["psi_eq"]
    te2d = data["teq"]
    ti2d = data["tiq"]
    pr2d = data["prq"]
    nq = data["nq"]
    vq = data["vq"]
    jq = data["jq"]
    R = data["R"]

    psi_axis, sign = _psi_axis_and_sign(psi2d)
    psi_out = sign * (psi2d - psi_axis)

    # equal-count psi grid
    psi_grid, bins = _make_equal_count_bins(psi_out, int(args.nbins))
    
    # 1D quantities
    p1d = _bin_scalar_on_bins(pr2d, psi_grid, bins)
    te1d = _bin_scalar_on_bins(te2d, psi_grid, bins)
    ti1d = _bin_scalar_on_bins(ti2d, psi_grid, bins)

    # densities (heuristic: 0 = electrons)
    ne1d = _bin_scalar_on_bins(nq[..., 0], psi_grid, bins) if nq.shape[-1] >= 1 else None
    ion_list: List[np.ndarray] = []
    for s in range(1, int(nq.shape[-1])):
        ion_list.append(_bin_scalar_on_bins(nq[..., s], psi_grid, bins))

    # omega_tor = vphi/R (use toroidal component = index 2)
    omega2d = np.full_like(pr2d, np.nan, dtype=float)
    m = np.isfinite(R) & (np.abs(R) > 0) & np.isfinite(vq[..., 2])
    omega2d[m] = vq[..., 2][m] / R[m]
    omega1d = _bin_scalar_on_bins(omega2d, psi_grid, bins)

    # j_tor (toroidal component)
    jtor1d = _bin_scalar_on_bins(jq[..., 2], psi_grid, bins)

    # append profile entry
    idx = _append_time_core_profiles(cp, t)
    p = cp.profiles_1d[idx]

    # grid coordinates
    # Requested behavior: rho_tor_norm is a linspace from 0 at magnetic axis to 1 at separatrix,
    # and held at 1 through the SOL extension.
    if hasattr(p, "grid"):
        g = p.grid

        # psi_grid is on the positive (axis-referenced) psi_out coordinate.
        psi1d = np.asarray(psi_grid, dtype=np.float64).ravel()
        psi_abs_1d = np.asarray(psi_axis + sign * psi1d, dtype=np.float64)

        # Enforce monotonic ordering (axis -> SOL) and keep all 1D profiles aligned.
        order = np.argsort(psi1d)
        psi1d = psi1d[order]
        psi_abs_1d = psi_abs_1d[order]
        p1d = _as_f64(p1d)[order]
        te1d = _as_f64(te1d)[order]
        ti1d = _as_f64(ti1d)[order]
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
            psi_sep_out = float(sign * (psi_lcfs_est - psi_axis))
        else:
            psi_sep_out = np.nan

        if not np.isfinite(psi_sep_out):
            n_core = int(psi1d.size)
        else:
            core_mask = psi1d <= (psi_sep_out + 1e-12)
            n_core = int(np.count_nonzero(core_mask))
            if n_core < 1:
                n_core = 1

        rho1d = np.ones_like(psi1d, dtype=np.float64)
        if n_core == 1:
            rho1d[0] = 0.0
        else:
            rho1d[:n_core] = np.linspace(0.0, 1.0, n_core)

        if hasattr(g, "rho_tor_norm"):
            try:
                g.rho_tor_norm = _as_f64(rho1d)
            except Exception:
                pass

        # Optional: store absolute psi coordinate if the DD exposes a suitable leaf.
        for name in ("psi", "psi_abs", "psi_pol"):
            if hasattr(g, name):
                try:
                    setattr(g, name, _as_f64(psi_abs_1d))
                    break
                except Exception:
                    pass
# electrons
    if hasattr(p, "electrons"):
        if ne1d is not None and hasattr(p.electrons, "density"):
            try:
                p.electrons.density = _as_f64(ne1d)
            except Exception:
                pass
        if hasattr(p.electrons, "temperature"):
            try:
                p.electrons.temperature = _as_f64(te1d)
            except Exception:
                pass

    # ions
    if hasattr(p, "ion"):
        # ensure at least one ion
        try:
            if _aos_len(p.ion) == 0:
                p.ion.resize(max(1, len(ion_list)))
        except Exception:
            pass

        # Ti in ion[0]
        try:
            if hasattr(p.ion[0], "temperature"):
                p.ion[0].temperature = _as_f64(ti1d)
        except Exception:
            pass

        # densities
        for k, ni1d in enumerate(ion_list):
            try:
                if k >= _aos_len(p.ion):
                    p.ion.resize(k + 1)
                if hasattr(p.ion[k], "density"):
                    p.ion[k].density = _as_f64(ni1d)
            except Exception:
                pass

        # rotation frequency
        try:
            if hasattr(p.ion[0], "rotation_frequency_tor_s"):
                p.ion[0].rotation_frequency_tor_s = _as_f64(omega1d)
        except Exception:
            pass

    # current density profile
    for name in ("j_tor", "jtor", "j_phi"):
        if hasattr(p, name):
            try:
                setattr(p, name, _as_f64(jtor1d))
                break
            except Exception:
                pass

    # pressure profile
    if hasattr(p, "pressure"):
        try:
            p.pressure = _as_f64(p1d)
        except Exception:
            pass

    try:
        cp.code.name = "NIMROD"
    except Exception:
        pass


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

    # toroidal modes
    if nmodes > 0:
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

    for m in range(nmodes):
        tm = ts.toroidal_mode[m]
        try:
            tm.n_tor = int(round(float(keff[m])))
        except Exception:
            pass

        pl = tm.plasma

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

        # Velocity perturbation (optional)
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

        # pressure perturbation
        if "repr" in fields and "impr" in fields:
            _set_complex_scalar(pl, ("pressure_perturbed", "p_perturbed"), fields["repr"][:, :, m], fields["impr"][:, :, m])
# -----------------------------
# Main
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert NIMROD dumpgll HDF5 to IMAS equilibrium/core_profiles/mhd_linear (stitched RZ arrays).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("dumpgll", nargs="+", help="Input dumpgll HDF5 file(s)")

    p.add_argument("--backend", default="hdf5", choices=("hdf5",), help="IMAS backend")
    p.add_argument("--dbpath", default=".", help="DB root path (output directory)")
    p.add_argument("--dd", required=True, help="DB name (directory name), e.g. nimrod")
    p.add_argument("--dd-version", default=None, help="IMAS data dictionary version, e.g. 3.42.0")
    p.add_argument("--pulse", type=int, required=True)
    p.add_argument("--run", type=int, required=True)

    p.add_argument("--mode", default="a", help="DBEntry open mode: r/a/w/x (r+/rw accepted and mapped to a)")

    p.add_argument("--equilibrium-occ", type=int, default=1)
    p.add_argument("--core-profiles-occ", type=int, default=1)
    p.add_argument("--mhd-linear-occ-base", type=int, default=0, help="Base occurrence for mhd_linear; actual occ = base + species_index")

    p.add_argument("--nbins", type=int, default=256, help="Number of bins for 1D profiles")

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
    entry_dir = _entry_dir(dbpath, str(args.dd), str(dd_version)[0], int(args.pulse), int(args.run))

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

        # Populate and store equilibrium/core_profiles as SINGLE time-slices.
        eq = factory.new("equilibrium") if hasattr(factory, "new") else factory("equilibrium")
        populate_equilibrium(eq, data, t_index=0, quiet=args.quiet)
        _db_put_slice(db, eq, int(args.equilibrium_occ))

        cp = factory.new("core_profiles") if hasattr(factory, "new") else factory("core_profiles")
        populate_core_profiles(cp, data, t_index=0, args=args)
        _db_put_slice(db, cp, int(args.core_profiles_occ))

        # mhd_linear per species occurrence
        nspec_dens = 1
        fields = data["fields"]
        if "nspec_dens" in fields:
            try:
                nspec_dens = int(np.ravel(fields["nspec_dens"])[0])
            except Exception:
                nspec_dens = int(fields.get("rend", np.zeros((1, 1, 1, 1))).shape[2])

        for s in range(max(1, nspec_dens)):
            occ = int(args.mhd_linear_occ_base) + int(s)
            mhd = factory.new("mhd_linear") if hasattr(factory, "new") else factory("mhd_linear")

            include_common = (s == 0)
            populate_mhd_linear(mhd, data, t_index=0, args=args, species_index=s, include_common_fields=include_common)

            # annotate
            try:
                mhd.code.name = "NIMROD"
            except Exception:
                pass
            try:
                mhd.ids_properties.comment = (
                    f"dump2imas: stitched grid; occurrence={occ}; species_index={s}; "
                    f"dens_pert_order={args.dens_pert_order}; common_fields={include_common}"
                )
            except Exception:
                pass

            _db_put_slice(db, mhd, occ)

        _log(f"Appended IDS slices for {fn.name}", args.quiet)

    try:
        db.close()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
