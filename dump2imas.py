#!/usr/bin/env python3
"""dump2imas.py

NIMROD dumpgll (HDF5) -> IMAS conversion.


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
  * Stitches RZ arrays in IMAS (global grid), rather than the packed rblock layout.

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

__version__ = "0.3.0"

try:
    import f90nml  # type: ignore
except Exception:
    f90nml = None  # type: ignore
import xml.etree.ElementTree as ET

from nimrod2imas import (
    entry_dir as _entry_dir_common,
    open_dbentry as _open_db_common,
    ids_factory as _ids_factory_common,
    value_to_string as _value_to_string_common,
    namelist_file_to_xml as _namelist_file_to_xml_common,
    update_workflow_and_dataset_fair as _update_workflow_and_dataset_fair_common,
    sanitize_cli_command as _sanitize_cli_command_common,
)

def _import_imas():
    import imas
    return imas


# -----------------------------
# Small utilities
# -----------------------------

def _log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, flush=True)



def _use_h5py_patches(args) -> bool:
    """Return True if this run should apply direct HDF5 patching (h5py)."""
    w = str(getattr(args, "writer", "auto") or "auto").strip().lower()
    if w == "h5py":
        return True
    if w == "imas":
        return False
    # auto
    b = str(getattr(args, "backend", "hdf5") or "hdf5").strip().lower()
    return (b == "hdf5")


def _use_imas_connectivity_writer(args) -> bool:
    """Return True if unstructured grid_ggd connectivity should be written via IMAS objects."""
    w = str(getattr(args, "writer", "auto") or "auto").strip().lower()
    if w == "imas":
        return True
    if w == "h5py":
        return False
    b = str(getattr(args, "backend", "hdf5") or "hdf5").strip().lower()
    return (b != "hdf5")



def _die(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def _as_f64(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


# -----------------------------
# COCOS handling
# -----------------------------
#
# NIMROD uses an R–Z–phi cylindrical ordering ("RZPhi"), which corresponds to COCOS=12
# in the Sauter coordinate-conventions table: sigma_{R\phi Z} = -1 (toroidal angle
# increases clockwise when viewed from +Z).
#
# IMAS expects COCOS=11 (sigma_{R\phi Z} = +1), which is the standard right-handed
# (R,phi,Z) cylindrical system.
#
# The 12->11 conversion requires:
#   * flip sign of absolute poloidal flux psi (so that B_R/B_Z remain unchanged when
#     sigma_{R\phi Z} changes sign)
#   * flip sign of *toroidal* (phi) components of equilibrium vectors (B_phi, V_phi, J_phi)
#   * for Fourier-mode perturbations stored as (re, im): complex conjugation for scalars
#     (Im -> -Im) and -conjugation for toroidal vector components (Re_phi -> -Re_phi,
#     Im_phi unchanged)
#
# This script assumes NIMROD dump inputs are COCOS=12 and converts all dump2imas outputs
# to be COCOS=11-consistent.

COCOS_IN_DEFAULT = 12
COCOS_OUT_DEFAULT = 11


def _set_ids_cocos(ids_obj: Any, cocos: int) -> None:
    """Best-effort setter for IMAS COCOS metadata across DD versions."""
    try:
        ip = getattr(ids_obj, "ids_properties", None)
        if ip is None:
            return
        # Common DD4+ location
        cs = getattr(ip, "coordinate_system", None)
        if cs is not None and hasattr(cs, "cocos"):
            cs.cocos = int(cocos)
            return
        # Some DDs expose cocos directly under ids_properties
        if hasattr(ip, "cocos"):
            ip.cocos = int(cocos)
            return
    except Exception:
        return


def _apply_cocos_12_to_11_inplace(data: Dict[str, Any], log: logging.Logger) -> None:
    """In-place COCOS=12 -> COCOS=11 conversion for stitched NIMROD dump data."""
    if bool(data.get("_cocos_12_to_11_applied", False)):
        return

    # 1) Absolute poloidal flux (psi): flip sign
    if data.get("psi_eq", None) is not None:
        data["psi_eq"] = -np.asarray(data["psi_eq"], dtype=float)
        log.info("COCOS 12->11: flipped sign of psi_eq (absolute poloidal flux)")

    # 2) Equilibrium vectors: flip toroidal component
    for k in ("bq", "vq", "jq"):
        A = data.get(k, None)
        if A is None:
            continue
        AA = np.asarray(A)
        if AA.ndim >= 3 and AA.shape[-1] >= 3:
            AA = AA.copy()
            AA[..., 2] *= -1.0
            data[k] = AA
            log.info("COCOS 12->11: flipped sign of %s[...,phi]", k)

    # 3) Perturbations: conjugate mode coefficients for phi-reversal
    fields = data.get("fields", None)
    if isinstance(fields, dict) and fields:

        def _flip_vec_modes(re_key: str, im_key: str) -> None:
            # arrays are typically (Nx,Ny,nmodes,3)
            if re_key in fields:
                A = np.asarray(fields[re_key])
                if A.ndim >= 4 and A.shape[-1] == 3:
                    A = A.copy()
                    A[..., 2] *= -1.0  # Re_phi -> -Re_phi (toroidal basis flip)
                    fields[re_key] = A
            if im_key in fields:
                A = np.asarray(fields[im_key])
                if A.ndim >= 4 and A.shape[-1] == 3:
                    A = A.copy()
                    A[..., 0] *= -1.0  # Im_R  -> -Im_R  (conjugation)
                    A[..., 1] *= -1.0  # Im_Z  -> -Im_Z  (conjugation)
                    # Im_phi unchanged for -conjugation of toroidal component
                    fields[im_key] = A

        # Vector perturbations
        _flip_vec_modes("rebe", "imbe")
        _flip_vec_modes("reve", "imve")
        _flip_vec_modes("reja", "imja")

        # Scalar perturbations: Im -> -Im (complex conjugation)
        for k in list(fields.keys()):
            if not k.startswith("im"):
                continue
            if k in ("imbe", "imve", "imja"):
                continue
            try:
                fields[k] = -np.asarray(fields[k])
            except Exception:
                pass

        data["fields"] = fields
        log.info(
            "COCOS 12->11: applied conjugation to perturbations (scalar Im->-Im; vector components adjusted)"
        )

    data["_cocos_12_to_11_applied"] = True



def _sanity_print_cocos(tag: str, src_name: str, t: float, payload: Dict[str, Any], args: Any) -> None:
    """Print lightweight sanity statistics before/after COCOS conversion.

    Intended for quick verification that the 12->11 sign flips are being applied:
      - psi flips sign
      - Bphi/Jphi/Vphi flip sign (equilibrium)
      - scalar-mode imaginary parts flip sign; vector-mode rules for phi reversal hold
    """
    if not getattr(args, "sanity_print", False):
        return

    def _sample_finite(a: Optional[np.ndarray], max_n: int = 200000) -> np.ndarray:
        if a is None:
            return np.asarray([], dtype=float)
        x = np.asarray(a, dtype=float).ravel()
        if x.size == 0:
            return np.asarray([], dtype=float)
        if x.size > max_n:
            # deterministic strided subsample
            idx = np.linspace(0, x.size - 1, max_n, dtype=np.int64)
            x = x[idx]
        x = x[np.isfinite(x)]
        return x

    def _fmt_stats(name: str, a: Optional[np.ndarray]) -> str:
        x = _sample_finite(a)
        if x.size == 0:
            return f"{name}: (missing)"
        q = np.quantile(x, [0.0, 0.01, 0.5, 0.99, 1.0])
        mean = float(np.mean(x))
        return (
            f"{name}: min={q[0]:+.3e} q01={q[1]:+.3e} med={q[2]:+.3e} "
            f"q99={q[3]:+.3e} max={q[4]:+.3e} mean={mean:+.3e}"
        )

    def _phi_comp(v: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if v is None:
            return None
        a = np.asarray(v)
        if a.ndim >= 3 and a.shape[-1] >= 3:
            return a[..., 2]
        return None

    psi = payload.get("psi_eq", None)
    bq = payload.get("bq", None)
    vq = payload.get("vq", None)
    jq = payload.get("jq", None)
    fields = payload.get("fields", {}) if isinstance(payload.get("fields", None), dict) else {}

    lines = []
    lines.append(f"SANITY[{tag}] {src_name}  t={t:.6g}  (COCOS12->11 check)")
    lines.append("  " + _fmt_stats("psi_eq", psi))
    lines.append("  " + _fmt_stats("Bphi(eq)", _phi_comp(bq)))
    lines.append("  " + _fmt_stats("Jphi(eq)", _phi_comp(jq)))
    lines.append("  " + _fmt_stats("Vphi(eq)", _phi_comp(vq)))

    # Representative perturbation checks (if present)
    if isinstance(fields, dict) and fields:
        if "impr" in fields:
            lines.append("  " + _fmt_stats("impr(mode)", fields.get("impr")))
        if "rebe" in fields:
            lines.append("  " + _fmt_stats("rebe_phi(mode)", _phi_comp(fields.get("rebe"))))
        if "imbe" in fields:
            # For vector Im: R/Z should flip under conjugation, phi should not (per our 12->11 rule)
            imbe = np.asarray(fields.get("imbe"))
            try:
                lines.append("  " + _fmt_stats("imbe_R(mode)", imbe[..., 0]))
                lines.append("  " + _fmt_stats("imbe_Z(mode)", imbe[..., 1]))
                lines.append("  " + _fmt_stats("imbe_phi(mode)", imbe[..., 2]))
            except Exception:
                pass

    print("\n".join(lines), flush=True)


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
                    elif kk == "zeff_input":
                        out["zeff_input"] = float(v)
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
        zeff = _grab_scalar("zeff_input")
        if z:
            out["z_ions"] = z
        if m:
            out["m_ions_kg"] = m
        if me is not None:
            out["me_kg"] = me
        if qe is not None:
            out["qe_c"] = qe
        if zeff is not None:
            out["zeff_input"] = float(zeff)

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



def _aos_has_entries(aos: Any) -> bool:
    """True if an IMAS AoS node appears to have at least one entry.

    Some IMAS-Python builds implement __len__ incorrectly (returning 0 even when
    indexing works), so we check both metadata length and an index probe.
    """
    try:
        if _aos_len(aos) > 0:
            return True
    except Exception:
        pass
    try:
        _ = aos[0]  # type: ignore[index]
        return True
    except Exception:
        return False


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


def _entry_dir(dbpath: Path, dd: str, dd_version: str, pulse: int, run: int, dd_version_dir: str = '4') -> Path:
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
    # (IMAS-Python variants differ: is_slice may be boolean or integer).
    for is_slice in (True, 1):
        try:
            db.put(ids, occ, is_slice)
            return
        except TypeError:
            pass
        except Exception:
            pass

    # Final fallback: normal put() of the in-memory IDS.
    # This may overwrite rather than append on some backends, but it ensures the IDS is persisted
    # for IMAS builds that do not support put_slice()/is_slice.
    try:
        db.put(ids, occ)
        return
    except TypeError:
        db.put(ids)
        return


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



def _append_time_edge_profiles(ep: Any, t: float) -> int:
    """Append a new profiles_1d entry to edge_profiles and return its index.

    Mirrors _append_time_core_profiles() but targets edge_profiles.
    """
    n = 0
    try:
        n = _aos_len(ep.profiles_1d)
    except Exception:
        n = 0

    if n == 0:
        try:
            ep.time = np.asarray([t], dtype=float)
        except Exception:
            pass
        try:
            ep.profiles_1d.resize(1)
            ep.profiles_1d[0].time = float(t)
        except Exception:
            pass
        return 0

    idx = n
    try:
        ep.time = np.asarray(list(np.asarray(ep.time, dtype=float)) + [t], dtype=float)
    except Exception:
        pass
    try:
        ep.profiles_1d.resize(idx + 1)
        ep.profiles_1d[idx].time = float(t)
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


def _unpack_multispecies_scalar_modes(
    a: np.ndarray,
    nmodes: int,
    nspec: int,
    order: str,
) -> np.ndarray:
    """Return (ny,nx,nspec,nmodes). Supports ambiguity in packing."""
    a = np.asarray(a)
    a = _squeeze1(a)

    if a.ndim == 4:
        # Guess which axis is nspec/nmodes
        if a.shape[2] == nmodes and a.shape[3] == nspec:
            # (ny,nx,nmodes,nspec) -> (ny,nx,nspec,nmodes)
            return np.transpose(a, (0, 1, 3, 2))
        if a.shape[3] == nmodes and a.shape[2] == nspec:
            return a

    if a.ndim == 3:
        ny, nx, k = a.shape
        if k == nspec * nmodes:
            # This assumes species_major packing for simplicity
            return a.reshape(ny, nx, nspec, nmodes)

    raise ValueError(f"Unexpected multispecies scalar shape {a.shape} for nmodes={nmodes}, nspec={nspec}")



def _reconstruct_full_from_modes(
    eq: Optional[np.ndarray],
    re_modes: Optional[np.ndarray],
    im_modes: Optional[np.ndarray],
    keff: np.ndarray,
    phi: float,
    pert_scale: float = 1.0,
) -> Optional[np.ndarray]:
    """Reconstruct a scalar field at toroidal angle phi from equilibrium + Fourier modes.

    Expected shapes:
      - eq: (N1,N2) or None
      - re_modes/im_modes: (N1,N2,nmodes) or (N2,N1,nmodes) or None
      - keff: (nmodes,) toroidal mode numbers (typically n)

    Uses the convention:
        f(phi) = eq + pert_scale * sum_m [ re_m * cos(n_m*phi) - im_m * sin(n_m*phi) ].

    Returns None only if both eq and modes are unavailable.
    """
    if eq is None and re_modes is None and im_modes is None:
        return None

    # determine base shape
    base = None
    if eq is not None:
        base = np.asarray(eq, dtype=float)
        if base.ndim != 2:
            base = np.squeeze(base)
            if base.ndim != 2:
                raise ValueError(f"eq must be 2D; got shape {np.asarray(eq).shape}")
    else:
        # build a zero baseline from the available modes
        src = re_modes if re_modes is not None else im_modes
        src = np.asarray(src)
        if src.ndim != 3:
            src = np.squeeze(src)
        if src.ndim != 3:
            raise ValueError(f"modes must be 3D (N1,N2,nmodes); got shape {np.asarray(src).shape}")
        base = np.zeros(src.shape[:2], dtype=float)

    # normalize mode arrays to (N1,N2,nmodes) compatible with base
    def _norm_modes(a: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if a is None:
            return None
        aa = np.asarray(a, dtype=float)
        aa = np.squeeze(aa)
        if aa.ndim != 3:
            raise ValueError(f"mode array must be 3D; got shape {aa.shape}")
        if aa.shape[:2] == base.shape:
            return aa
        if aa.shape[:2] == base.T.shape:
            return np.transpose(aa, (1, 0, 2))
        # final fallback: try swapping first two axes if it matches
        if aa.shape[0] == base.shape[1] and aa.shape[1] == base.shape[0]:
            return np.transpose(aa, (1, 0, 2))
        return aa

    reA = _norm_modes(re_modes)
    imA = _norm_modes(im_modes)

    if reA is None and imA is None:
        return base

    if reA is None:
        reA = np.zeros_like(imA, dtype=float)
    if imA is None:
        imA = np.zeros_like(reA, dtype=float)

    nm = int(reA.shape[2])
    k = np.asarray(keff, dtype=float).ravel()
    if k.size < nm:
        # pad with sequential mode numbers if keff is short/missing
        k2 = np.arange(nm, dtype=float)
        k2[: k.size] = k
        k = k2
    k = k[:nm]

    phase = k * float(phi)
    c = np.cos(phase).reshape(1, 1, nm)
    s = np.sin(phase).reshape(1, 1, nm)

    pert = np.nansum(reA * c - imA * s, axis=2)
    try:
        ps = float(pert_scale)
    except Exception:
        ps = 1.0
    return base + ps * pert
def _expand_single_ion_to_e_plus_main(
    nq: "Optional[np.ndarray]",
    fields: "Dict[str, np.ndarray]",
    nspec_eq: int,
    nmodes: int,
    args: "Any",
    nimrod_in_path: "Optional[str]" = None,
) -> "tuple[Optional[np.ndarray], Dict[str, np.ndarray], int]":
    """Expand single-channel density dumps to IMAS-friendly [electrons, main ion].

    Some NIMROD builds (typically "single-ion" or "no-impurity" variants) store only one
    density channel in nq (and similarly only one channel in rend/imnd), even though downstream
    IMAS writers expect electrons plus at least one ion species.

    Policy:
      - If nspec_eq != 1 or nq is None: return unchanged.
      - Otherwise interpret nq[...,0] as electron density ne.
      - Construct a main-ion density ni via quasi-neutrality using Z_main when available:
            ni = ne / Z_main
        where Z_main is taken from nimrod.in zisp_input if possible, else falls back to zeff_input
        (as an effective divisor), else defaults to 1.
      - If rend/imnd exist with a single species channel, expand them similarly.

    This does *not* add impurity species; it only ensures ion[] leaves can be populated.
    """
    try:
        import numpy as _np
    except Exception:
        return nq, fields, nspec_eq

    if nq is None or int(nspec_eq) != 1:
        return nq, fields, nspec_eq

    # Resolve an effective main-ion charge.
    z_main: float = 1.0
    zeff: float | None = None

    sp = getattr(args, "_nimrod_species", None)
    if not sp and nimrod_in_path:
        try:
            sp = _nimrod_species_info(nimrod_in_path)
        except Exception:
            sp = None
    sp = sp or {}

    try:
        zlist = sp.get("z_ions", None) or []
        if zlist:
            z_main = float(zlist[0])
    except Exception:
        z_main = 1.0

    try:
        zeff = sp.get("zeff_input", None)
        zeff = float(zeff) if zeff not in (None, "") else None
    except Exception:
        zeff = None

    # Sanitize.
    if not (z_main and _np.isfinite(z_main) and z_main > 0):
        z_main = 1.0

    ne = _np.asarray(nq[..., 0], dtype=float)

    # In "e-only" dumps, infer a single effective ion density via quasi-neutrality.
    # Prefer zeff_input (constant in no-impurity runs) when available; otherwise fall back to Z_main.
    divisor = float(z_main) if (_np.isfinite(z_main) and z_main > 0.0) else 1.0
    if zeff is not None and _np.isfinite(zeff) and zeff > 0.0:
        divisor = float(zeff)

    with _np.errstate(divide='ignore', invalid='ignore'):
        ni = ne / divisor

    nq2 = _np.stack((ne, ni), axis=2)

    # Expand density perturbations if present with a single species channel.
    fields2 = dict(fields) if isinstance(fields, dict) else {}
    try:
        reN = fields2.get("rend", None)
        imN = fields2.get("imnd", None)
        if reN is not None and imN is not None:
            reN = _np.asarray(reN)
            imN = _np.asarray(imN)
            if reN.ndim == 4 and reN.shape[2] == 1:
                re_ne = reN[:, :, 0, :]
                im_ne = imN[:, :, 0, :]
                with _np.errstate(divide='ignore', invalid='ignore'):
                    re_ni = re_ne / divisor
                    im_ni = im_ne / divisor
                fields2["rend"] = _np.stack((re_ne, re_ni), axis=2)
                fields2["imnd"] = _np.stack((im_ne, im_ni), axis=2)
                fields2["nspec_dens"] = _np.asarray([2], dtype=int)
    except Exception:
        pass

    # Ensure consistent metadata for downstream slicing (even if rend/imnd are absent).
    try:
        fields2["nspec_dens"] = _np.asarray([2], dtype=int)
    except Exception:
        pass

    try:
        import logging
        logging.getLogger("dump2imas").info(
            "Single-ion/no-impurity density detected (nspec=1). Expanded nq (and rend/imnd when present) to [e, main ion] using divisor=%.6g (Z_main=%.6g, zeff=%s)",
            divisor, float(z_main), ("%.6g" % zeff) if zeff is not None else "None",
        )
    except Exception:
        pass

    return nq2, fields2, 2

# -----------------------------
# Profile binning on psi
# -----------------------------

def estimate_psi_axis_and_lcfs_robust(
    psi2d: np.ndarray,
    bq2d: Optional[np.ndarray] = None,
    te2d: Optional[np.ndarray] = None,
    pe2d: Optional[np.ndarray] = None,
    pr2d: Optional[np.ndarray] = None,
    nq: Optional[np.ndarray] = None,
    qe: float = 1.602176634e-19,
    te_min: float = 20.0,
    qedge: float = 0.995,
) -> tuple[float, float, str]:
    """Estimate (psi_axis, psi_lcfs) robustly and return a short method tag.

    Axis selection:
      - Prefer a *core indicator* to avoid confusing X-point/poloidal-field minima with the axis.
        Priority: Te (dump) -> Te from pe/ne -> total pressure -> ne.
      - Use a high-quantile "core mask" and take the median psi within it.

    LCFS proxy:
      - Use the farthest extreme (high or low) of psi relative to psi_axis over plasma-like points,
        estimated via dv quantiles.

    Returns:
      (psi_axis, psi_lcfs, tag)
    """
    ps = np.asarray(psi2d, dtype=float)
    mpsi = np.isfinite(ps)
    if not np.any(mpsi):
        return (np.nan, np.nan, "fail:nopsi")

    # --- build plasma-like mask + indicator ---
    tag = "psi_only"
    indicator = None
    m = mpsi.copy()

    # Electron density (robustly from nq)
    ne2d = None
    if nq is not None:
        try:
            A = np.asarray(nq, dtype=float)
            if A.ndim == 2:
                ne2d = A
            elif A.ndim >= 3 and A.shape[-1] >= 1:
                ne2d = A[..., 0]
        except Exception:
            ne2d = None

    if te2d is not None:
        # Best indicator when available
        te = np.asarray(te2d, dtype=float)
        indicator = te
        m = mpsi & np.isfinite(te) & (te > te_min)
        tag = "te"
    elif pe2d is not None and ne2d is not None:
        # Fallback: estimate Te from p_e/n_e, but suppress extreme values due to very small densities
        pe = np.asarray(pe2d, dtype=float)
        te_est = np.full_like(pe, np.nan, dtype=float)
        mm = mpsi & np.isfinite(pe) & np.isfinite(ne2d) & (ne2d > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            te_est[mm] = pe[mm] / (ne2d[mm] * float(qe))
        indicator = te_est
        m = mm & np.isfinite(te_est) & (te_est > te_min)
        tag = "te_from_pe_ne"
    elif pr2d is not None:
        pr = np.asarray(pr2d, dtype=float)
        indicator = pr
        m = mpsi & np.isfinite(pr) & (pr > 0.0)
        tag = "p_tot"
    elif ne2d is not None:
        indicator = ne2d
        m = mpsi & np.isfinite(ne2d) & (ne2d > 0.0)
        tag = "ne"
    else:
        m = mpsi.copy()
        tag = "psi_only"

    if int(np.count_nonzero(m)) < 50:
        m = mpsi.copy()
        tag = tag + "|fallback"

    vals = ps[m]
    if vals.size == 0:
        return (np.nan, np.nan, "fail:novals")

    # --- axis ---
    psi_axis = np.nan
    axis_tag = "axis:unset"
    if indicator is not None:
        ind = np.asarray(indicator, dtype=float)
        mm = m & np.isfinite(ind)
        if np.any(mm):
            indv = ind[mm]
            psiv = ps[mm]
            # Use top 0.5% of indicator values (robust against single-pixel spikes)
            qcore = 0.995 if indv.size > 5000 else 0.99
            thr = float(np.nanquantile(indv, qcore))
            core = mm & (ind >= thr)
            if int(np.count_nonzero(core)) >= 10:
                psi_axis = float(np.nanmedian(ps[core]))
                axis_tag = "axis:core_q"
            else:
                # fallback to argmax of indicator
                psi_axis = float(psiv[int(np.nanargmax(indv))])
                axis_tag = "axis:argmax"
    if not np.isfinite(psi_axis):
        # pure psi fallback: take median of central quantile of psi values
        try:
            lo = float(np.nanquantile(vals, 0.45))
            hi = float(np.nanquantile(vals, 0.55))
            mid = vals[(vals >= lo) & (vals <= hi)]
            psi_axis = float(np.nanmedian(mid)) if mid.size else float(np.nanmedian(vals))
            axis_tag = "axis:psi_mid"
        except Exception:
            psi_axis = float(np.nanmedian(vals))
            axis_tag = "axis:psi_med"

    # Optional refinement using bq2d: only within core region (avoids X-point)
    if bq2d is not None and indicator is not None:
        try:
            bq = np.asarray(bq2d, dtype=float)
            # accept either scalar |Bp| or vector (..,3)
            if bq.ndim == 3 and bq.shape[-1] == 3:
                # Interpret as (Br,Bz,Bphi); use poloidal magnitude
                bp = np.sqrt(bq[..., 0] ** 2 + bq[..., 1] ** 2)
            else:
                bp = np.asarray(bq, dtype=float)
            if bp.shape == ps.shape:
                ind = np.asarray(indicator, dtype=float)
                mm = m & np.isfinite(ind)
                if np.any(mm):
                    qcore = 0.995
                    thr = float(np.nanquantile(ind[mm], qcore))
                    core = mm & (ind >= thr) & np.isfinite(bp)
                    if int(np.count_nonzero(core)) >= 10:
                        ij = np.nanargmin(bp[core])
                        psi_axis = float(ps[core][int(ij)])
                        axis_tag = axis_tag + "+bpmin_core"
        except Exception:
            pass

    # --- lcfs proxy ---
    dv = vals - float(psi_axis)
    try:
        dv_hi = float(np.nanquantile(dv, qedge))
        dv_lo = float(np.nanquantile(dv, 1.0 - qedge))
    except Exception:
        dv_hi = float(np.nanmax(dv))
        dv_lo = float(np.nanmin(dv))

    if abs(dv_hi) >= abs(dv_lo):
        psi_lcfs = float(psi_axis + dv_hi)
        dir_tag = "dir:+"
    else:
        psi_lcfs = float(psi_axis + dv_lo)
        dir_tag = "dir:-"

    if not np.isfinite(psi_lcfs) or abs(psi_lcfs - psi_axis) < 1e-12:
        # fallback to farthest extreme
        try:
            dv_far = dv_hi if abs(dv_hi) >= abs(dv_lo) else dv_lo
            psi_lcfs = float(psi_axis + dv_far)
        except Exception:
            psi_lcfs = float(np.nanmax(vals))

    return (float(psi_axis), float(psi_lcfs), f"{tag}|{axis_tag}|{dir_tag}")

def _resolve_optional_file(args, attr: str, default_name: str) -> Optional[str]:
    """Resolve an optional input file path.

    Resolution order:
      1) explicit CLI flag value (absolute or relative to run_dir)
      2) <run_dir>/<default_name>
      3) ./<default_name>
    """
    run_dir = str(getattr(args, "_run_dir", "") or "")
    val = getattr(args, attr, None)
    cand = None

    def _isfile(p: Optional[str]) -> bool:
        try:
            return (p is not None) and os.path.isfile(p)
        except Exception:
            return False

    if val:
        # If relative, try run_dir first, then CWD
        if os.path.isabs(val):
            cand = val
        else:
            if run_dir:
                cand = os.path.join(run_dir, val)
                if not _isfile(cand):
                    cand = val
            else:
                cand = val
        if _isfile(cand):
            return cand

        # Common robustness: case-insensitive lookup for bare filenames (e.g. PEQDSK vs peqdsk)
        try:
            base = os.path.basename(str(val))
            if base == str(val) and default_name and base.lower() == str(default_name).lower():
                for d in (run_dir, os.getcwd()):
                    if not d or (not os.path.isdir(d)):
                        continue
                    for fn in os.listdir(d):
                        if fn.lower() == base.lower() or fn.lower().startswith(base.lower()):
                            p = os.path.join(d, fn)
                            if _isfile(p):
                                return p
        except Exception:
            pass

        return None

    # default name
    if run_dir:
        cand = os.path.join(run_dir, default_name)
        if _isfile(cand):
            return cand
    if _isfile(default_name):
        return default_name

    # Robustness: case-insensitive / prefix match for default_name (when user did not pass --{attr}).
    # This handles common variants like PEQDSK, peqdsk_*, peqdsk.dat, etc.
    try:
        base = str(default_name)
        for d in (run_dir, os.getcwd()):
            if not d or (not os.path.isdir(d)):
                continue
            for fn in os.listdir(d):
                fn_l = fn.lower()
                base_l = base.lower()
                if fn_l == base_l or fn_l.startswith(base_l):
                    pp = os.path.join(d, fn)
                    if _isfile(pp):
                        return pp
    except Exception:
        pass

    return None


def _read_peqdsk_block(path: str, key: str) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """Read a 1D profile block from a PEQDSK-like ASCII file.

    Many 'peqdsk' variants use different labels. We scan for a header line that contains
    *key* as a standalone token (e.g. 'TE', 'NE'), not merely as a substring, to avoid
    false matches (e.g. 'temperature').

    After the header, we expect whitespace-separated numeric rows with at least two columns:
        x  y  [optional...]

    Returns:
        (x, y, unit_str) where x is the first column (often psinorm) and y is the second.
    """
    if not path or not os.path.isfile(path):
        return None

    key_l = str(key).strip().lower()
    if not key_l:
        return None

    # Standalone-token match, so 'TE(keV)' matches, 'temperature' does not.
    pat = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(key_l)}(?![A-Za-z0-9_])", re.IGNORECASE)

    xs: List[float] = []
    ys: List[float] = []
    unit = ""
    in_block = False

    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                if in_block and xs:
                    break
                continue

            # Header trigger
            if (not in_block) and (not s[0].isdigit()) and pat.search(s):
                unit = _parse_peqdsk_units(s) if "_parse_peqdsk_units" in globals() else ""
                in_block = True
                continue

            if not in_block:
                continue

            # Numeric rows
            try:
                parts = s.replace(",", " ").split()
                if len(parts) < 2:
                    continue
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except Exception:
                if xs:
                    break
                continue

    if len(xs) < 4:
        return None

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    # Ensure monotone-increasing x for interpolation.
    if np.any(np.diff(x) < 0):
        o = np.argsort(x)
        x = x[o]
        y = y[o]

    return x, y, unit

def _peqdsk_te_sep_ev(peqdsk_path: str, *, log: Optional[logging.Logger] = None) -> Optional[float]:
    """Return Te at psinorm≈1 from peqdsk in eV."""
    blk = _read_peqdsk_block(peqdsk_path, "te")
    if blk is None:
        return None
    ps, te, unit = blk
    # nearest to psiN=1
    j = int(np.nanargmin(np.abs(ps - 1.0)))
    val = float(te[j])

    u = (unit or "").lower()
    if "kev" in u:
        val *= 1e3
    elif "ev" in u:
        val *= 1.0
    else:
        # assume keV if Te is O(1-10), else eV
        if val < 50.0:
            val *= 1e3

    if log is not None:
        log.info(f"peqdsk: Te_sep≈{val:.3g} eV (unit='{unit}', psin={ps[j]:.6g}) from {peqdsk_path}")
    return val


def _peqdsk_profile_si(peqdsk_path: str, key: str, *, log: Optional[logging.Logger] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read a peqdsk 1D profile and convert to SI-like units.

    Supported keys:
      - 'te': returns Te in eV
      - 'ne': returns ne in m^-3

    Returns (psinorm, values_si) with psinorm sorted ascending.
    """
    blk = _read_peqdsk_block(peqdsk_path, key)
    if blk is None:
        return None
    ps, vv, unit = blk
    ps = np.asarray(ps, dtype=float)
    vv = np.asarray(vv, dtype=float)

    # sort by psinorm
    o = np.argsort(ps)
    ps = ps[o]
    vv = vv[o]

    u = (unit or '').lower().replace(' ', '')

    if key.lower() == 'te':
        # Te -> eV
        if 'kev' in u:
            vv = vv * 1e3
        elif 'ev' in u:
            vv = vv * 1.0
        else:
            # heuristic: if values are O(1-50) assume keV
            med = float(np.nanmedian(vv[np.isfinite(vv)])) if np.any(np.isfinite(vv)) else float('nan')
            if np.isfinite(med) and med < 50.0:
                vv = vv * 1e3
        return (ps, vv)

    if key.lower() == 'ne':
        # ne -> m^-3
        scale = 1.0
        # explicit 10^N or 1eN
        m = re.search(r'10\^([+-]?\d+)', u)
        if m:
            scale = 10.0 ** int(m.group(1))
        else:
            m = re.search(r'1e([+-]?\d+)', u)
            if m:
                scale = 10.0 ** int(m.group(1))
        # cm^-3 -> m^-3
        if ('cm' in u) and ('-3' in u or '^-3' in u):
            scale *= 1e6
        # If unit string says per m^3 without explicit scale, keep scale=1.
        # Otherwise, apply a simple magnitude heuristic common for TRANSP peqdsk:
        # values are often in 10^19 or 10^20 m^-3.
        if scale == 1.0 and ('/m3' not in u) and ('m-3' not in u) and ('m^(-3' not in u) and ('m^-3' not in u):
            vmax = float(np.nanmax(vv[np.isfinite(vv)])) if np.any(np.isfinite(vv)) else float('nan')
            if np.isfinite(vmax) and vmax < 1e6:
                # typical: ne~(0.1-10) in units of 1e20 or 1e19
                if vmax < 5.0:
                    scale = 1e20
                else:
                    scale = 1e19
        vv = vv * scale
        return (ps, vv)

    return (ps, vv)


def _peqdsk_te_ne_si(peqdsk_path: str, *, log: Optional[logging.Logger] = None) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Convenience: read Te(eV) and ne(m^-3) on a common psinorm grid."""
    te_blk = _peqdsk_profile_si(peqdsk_path, 'te', log=log)
    ne_blk = _peqdsk_profile_si(peqdsk_path, 'ne', log=log)
    if te_blk is None or ne_blk is None:
        return None
    ps_te, te = te_blk
    ps_ne, ne = ne_blk
    # Interpolate ne onto ps_te if grids differ
    if ps_ne.shape != ps_te.shape or np.nanmax(np.abs(ps_ne - ps_te)) > 1e-6:
        ne = np.interp(ps_te, ps_ne, ne, left=ne[0], right=ne[-1])
    return (ps_te, te, ne)


def _imas_core_profiles_te_ne(
    cp_ids: Any,
    *,
    target_time: Optional[float] = None,
    log: Optional[logging.Logger] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract (psi_norm_like, Te[eV], ne[m^-3]) from an existing core_profiles IDS.

    This is intended for workflows where input2imas runs first and stores equilibrium
    kinetic profiles (from PEQDSK/P-file) into IMAS core_profiles. In that workflow,
    input2imas commonly stores psinorm into grid.rho_tor_norm (not true rho_tor_norm).

    We therefore accept the first available grid among:
        grid.psi_norm, grid.rho_pol_norm, grid.rho_tor_norm
    and treat it as a *normalized poloidal flux-like* coordinate for edge_profiles.
    """
    if cp_ids is None:
        return None

    # Resolve profiles_1d container
    try:
        aos = cp_ids.profiles_1d
    except Exception:
        return None

    # Determine number of slices
    try:
        n = len(aos)
    except Exception:
        try:
            aos = list(aos)
            n = len(aos)
        except Exception:
            return None

    if n <= 0:
        return None

    # Select slice by time if available
    idx_best = 0
    if target_time is not None:
        try:
            t0 = float(target_time)
            best = None
            for i in range(n):
                try:
                    p = aos[i]
                except Exception:
                    continue
                ti = getattr(p, 'time', None)
                if ti is None:
                    continue
                try:
                    dt = abs(float(ti) - t0)
                except Exception:
                    continue
                if best is None or dt < best:
                    best = dt
                    idx_best = i
        except Exception:
            pass

    try:
        p = aos[idx_best]
    except Exception:
        try:
            p = aos[0]
        except Exception:
            return None

    # Grid coordinate
    g = getattr(p, 'grid', None)
    psi = None
    if g is not None:
        for nm in ('psi_norm', 'rho_pol_norm', 'rho_tor_norm'):
            if hasattr(g, nm):
                try:
                    arr = getattr(g, nm)
                    if arr is None:
                        continue
                    arr = np.asarray(arr, dtype=float).ravel()
                    if arr.size > 1 and np.any(np.isfinite(arr)):
                        psi = arr
                        break
                except Exception:
                    continue
    if psi is None:
        return None

    # Electrons
    e = getattr(p, 'electrons', None)
    if e is None:
        return None

    # Te
    te = None
    if hasattr(e, 'temperature'):
        try:
            te = np.asarray(e.temperature, dtype=float).ravel()
        except Exception:
            te = None

    # ne (prefer density_thermal if available)
    ne = None
    for nm in ('density_thermal', 'density'):
        if hasattr(e, nm):
            try:
                arr = getattr(e, nm)
                if arr is None:
                    continue
                ne = np.asarray(arr, dtype=float).ravel()
                break
            except Exception:
                ne = None

    if te is None or ne is None:
        return None

    nmin = min(int(psi.size), int(te.size), int(ne.size))
    if nmin < 2:
        return None

    psi = psi[:nmin]
    te = te[:nmin]
    ne = ne[:nmin]

    m = np.isfinite(psi) & np.isfinite(te) & np.isfinite(ne)
    if int(np.count_nonzero(m)) < 2:
        return None

    psi_m = psi[m]
    te_m = te[m]
    ne_m = ne[m]

    # Sort by coordinate (ensure monotone increasing)
    o = np.argsort(psi_m)
    psi_m = psi_m[o]
    te_m = te_m[o]
    ne_m = ne_m[o]

    if log is not None:
        try:
            te_sep = float(np.interp(1.0, psi_m, te_m, left=te_m[0], right=te_m[-1]))
        except Exception:
            te_sep = float('nan')
        log.info(
            "edge_profiles(equilibrium): using existing IMAS core_profiles as equilibrium source "
            "(grid range %.3g..%.3g; Te(psiN=1)~%.3g eV)",
            float(psi_m[0]), float(psi_m[-1]), float(te_sep)
        )

    return (psi_m.astype(float), te_m.astype(float), ne_m.astype(float))

def _psi_lcfs_from_contours(
    contours_path: str,
    R: np.ndarray,
    Z: np.ndarray,
    psi2d: np.ndarray,
    *,
    log: Optional[logging.Logger] = None,
) -> Optional[float]:
    """Estimate psi_lcfs by sampling psi2d on LCFS polyline points from contours.h5."""
    try:
        import h5py as _h5py
        with _h5py.File(contours_path, "r") as h5:
            pts = None
            for gnm in ("LCFS", "lcfs", "gfile_lcfs"):
                if gnm in h5 and "points" in h5[gnm]:
                    pts = np.asarray(h5[gnm]["points"])
                    break
            if pts is None or pts.ndim != 2 or pts.shape[1] < 2:
                return None
            r0 = np.asarray(pts[:, 0], dtype=float)
            z0 = np.asarray(pts[:, 1], dtype=float)
    except Exception as exc:
        if log is not None:
            log.warning(f"contours.h5 read failed ({contours_path}): {exc}")
        return None

    # Heuristic: swap if clearly out of domain
    rmin, rmax = np.nanmin(R), np.nanmax(R)
    zmin, zmax = np.nanmin(Z), np.nanmax(Z)
    in0 = (np.nanmin(r0) >= rmin - 1e-6) and (np.nanmax(r0) <= rmax + 1e-6) and (np.nanmin(z0) >= zmin - 1e-6) and (np.nanmax(z0) <= zmax + 1e-6)
    in1 = (np.nanmin(z0) >= rmin - 1e-6) and (np.nanmax(z0) <= rmax + 1e-6) and (np.nanmin(r0) >= zmin - 1e-6) and (np.nanmax(r0) <= zmax + 1e-6)
    if (not in0) and in1:
        # swapped
        rq, zq = z0, r0
    else:
        rq, zq = r0, z0

    psi_pts = _interp_mesh_to_points(R, Z, psi2d, rq, zq)
    psi_pts = np.asarray(psi_pts, dtype=float)
    psi_pts = psi_pts[np.isfinite(psi_pts)]
    if psi_pts.size < 10:
        return None

    psi_lcfs = float(np.nanmedian(psi_pts))
    if log is not None:
        log.info(f"LCFS from contours: psi_lcfs≈{psi_lcfs:.6g} (median of {psi_pts.size} samples) from {contours_path}")
    return psi_lcfs


def _psi_lcfs_from_te_sep(
    psi2d: np.ndarray,
    te2d: np.ndarray,
    te_sep_ev: float,
    psi_axis: float,
    psi_lcfs_guess: float,
    *,
    log: Optional[logging.Logger] = None,
) -> Optional[float]:
    """Estimate psi_lcfs by matching Te≈Te_sep near the edge (best effort)."""
    ps = np.asarray(psi2d, dtype=float)
    te = np.asarray(te2d, dtype=float)
    m = np.isfinite(ps) & np.isfinite(te)
    if not np.any(m):
        return None

    den = float(psi_lcfs_guess - psi_axis)
    if (not np.isfinite(den)) or abs(den) < 1e-12:
        return None

    rho_g = (ps - psi_axis) / den
    # focus on near-edge band
    m &= (rho_g > 0.7) & (rho_g < 1.3)

    # pick Te close to Te_sep
    te_sep_ev = float(te_sep_ev)
    rel = np.abs(te - te_sep_ev) / max(te_sep_ev, 1e-12)
    m2 = m & (rel < 0.25)
    if int(np.count_nonzero(m2)) < 50:
        m2 = m & (rel < 0.40)
    if int(np.count_nonzero(m2)) < 20:
        return None

    psi_lcfs = float(np.nanmedian(ps[m2]))
    if log is not None:
        log.info(f"LCFS from peqdsk Te_sep: psi_lcfs≈{psi_lcfs:.6g} using Te_sep={te_sep_ev:.3g} eV (n={np.count_nonzero(m2)})")
    return psi_lcfs


def _choose_psi_axis_lcfs(data: Dict[str, Any], args, log: Optional[logging.Logger] = None) -> tuple[float, float, str]:
    """Select (psi_axis, psi_lcfs) consistently for all IDS writers.

    This routine is used by both core_profiles and edge_profiles.

    Priority:
      1) Stitch contours file (if provided) to identify LCFS value on the dump grid.
      2) Use PEQDSK Te at psiN=1 to identify LCFS on the dump grid (if Te exists in dump).
      3) Use equilibrium occurrence 0 (psi_axis, psi_boundary) if available.
      4) Use core_profiles occurrence 0 endpoints (fallback).
      5) If Te exists in dump, use user Te_sep_ev to estimate LCFS.
      6) Robust fallback (quantile method).

    Post-processing:
      - Apply a core-peaked indicator sanity check to swap axis/lcfs when the normalization
        would place the core near psiN≈1 (common sign convention mismatch).
    """
    psi2d = data.get("psi_eq", None)
    if psi2d is None:
        psi2d = data.get("psi", None)
    if psi2d is None:
        return (np.nan, np.nan, "fail:nopsi")

    bq2d = data.get("bq", None)
    te2d = data.get("teq", None)
    pe2d = data.get("peq", None)
    pr2d = data.get("prq", None)
    nq = data.get("nq", None)

    sp = getattr(args, "_nimrod_species", {}) or {}
    qe = float(sp.get("qe_c", 1.602176634e-19))
    te_min = float(getattr(args, "te_min_ev", 20.0) or 20.0)
    qedge = float(getattr(args, "psi_edge_quantile", 0.995) or 0.995)

    psi_axis, psi_lcfs_guess, tag = estimate_psi_axis_and_lcfs_robust(
        np.asarray(psi2d, dtype=float),
        bq2d=np.asarray(bq2d, dtype=float) if bq2d is not None else None,
        te2d=np.asarray(te2d, dtype=float) if te2d is not None else None,
        pe2d=np.asarray(pe2d, dtype=float) if pe2d is not None else None,
        pr2d=np.asarray(pr2d, dtype=float) if pr2d is not None else None,
        nq=np.asarray(nq, dtype=float) if nq is not None else None,
        qe=qe,
        te_min=te_min,
        qedge=qedge,
    )

    def _finalize(pa: float, pb: float, tag2: str) -> tuple[float, float, str]:
        pa = float(pa); pb = float(pb)
        if (not np.isfinite(pa)) or (not np.isfinite(pb)) or abs(pb - pa) < 1e-12:
            return (pa, pb, tag2)

        # sanity: a core-peaked indicator (Te) should live near psiN≈0
        try:
            if te2d is None:
                return (pa, pb, tag2)
            te = np.asarray(te2d, dtype=float)
            ps = np.asarray(psi2d, dtype=float)
            m = np.isfinite(te) & np.isfinite(ps) & (te > te_min)
            if int(np.count_nonzero(m)) < 50:
                return (pa, pb, tag2)

            psn = (ps - pa) / (pb - pa)
            ind = te[m]
            psnv = psn[m]
            qcore = 0.995 if ind.size > 5000 else 0.99
            thr = float(np.nanquantile(ind, qcore))
            core = m & (te >= thr)
            if int(np.count_nonzero(core)) < 10:
                return (pa, pb, tag2)

            ps_core = float(np.nanmedian(psn[core]))
            # If the core maps near 1, swap
            if ps_core > 0.8 and ps_core < 1.2:
                if log is not None:
                    log.warning(
                        f"Swapping psi_axis/psi_lcfs (core indicator at psiN≈{ps_core:.3f}); "
                        f"was axis≈{pa:.6g}, lcfs≈{pb:.6g} (tag={tag2})"
                    )
                return (pb, pa, tag2 + "|swap")
        except Exception:
            pass
        return (pa, pb, tag2)

    # 1) contours file (highest priority for LCFS value)
    contours_path = _resolve_optional_file(args, "contours", "contours.h5")
    if contours_path:
        try:
            psi_lcfs_c = _psi_lcfs_from_contours(contours_path, data.get("R"), data.get("Z"), psi2d, log=log)
        except Exception:
            psi_lcfs_c = None
        if psi_lcfs_c is not None and np.isfinite(psi_lcfs_c) and abs(float(psi_lcfs_c) - float(psi_axis)) > 1e-12:
            return _finalize(float(psi_axis), float(psi_lcfs_c), f"{tag}|contours")

    # 2) peqdsk: Te at psiN=1 used to infer LCFS from dump Te
    peqdsk_path = _resolve_optional_file(args, "peqdsk", "peqdsk")
    te_sep_ev = None
    if peqdsk_path:
        te_sep_ev = _peqdsk_te_sep_ev(peqdsk_path, log=log)
        if te_sep_ev is not None and te2d is not None:
            psi_lcfs_p = _psi_lcfs_from_te_sep(psi2d, te2d, te_sep_ev, psi_axis, psi_lcfs_guess, log=log)
            if psi_lcfs_p is not None and np.isfinite(psi_lcfs_p) and abs(float(psi_lcfs_p) - float(psi_axis)) > 1e-12:
                return _finalize(float(psi_axis), float(psi_lcfs_p), f"{tag}|peqdsk_te")

    # 3) equilibrium/core_profiles occurrence 0 (fallback when contours/peqdsk are not available)
    entry_dir = getattr(args, "_entry_dir", None)
    if entry_dir:
        occ0 = 0
        # 3a) equilibrium: global_quantities psi_axis / psi_boundary
        for fpath in (os.path.join(entry_dir, f"equilibrium_{occ0}.h5"), os.path.join(entry_dir, "equilibrium.h5")):
            if os.path.isfile(fpath):
                try:
                    with h5py.File(fpath, "r") as f:
                        pa = f["equilibrium/time_slice[]&global_quantities&psi_axis"][()]
                        pb = f["equilibrium/time_slice[]&global_quantities&psi_boundary"][()]
                    pa = float(np.asarray(pa).ravel()[0])
                    pb = float(np.asarray(pb).ravel()[0])
                    if np.isfinite(pa) and np.isfinite(pb) and abs(pb - pa) > 1e-12:
                        if log is not None:
                            log.info(f"LCFS from equilibrium occ0: psi_axis≈{pa:.6g}, psi_lcfs≈{pb:.6g} from {fpath}")
                        return _finalize(pa, pb, f"{tag}|equilibrium0")
                except Exception:
                    pass

        # 3b) core_profiles: infer from profiles_1d grid endpoints
        for fpath in (os.path.join(entry_dir, f"core_profiles_{occ0}.h5"), os.path.join(entry_dir, "core_profiles.h5")):
            if os.path.isfile(fpath):
                try:
                    grp = f"core_profiles_{occ0}"
                    with h5py.File(fpath, "r") as f:
                        psi = f[f"{grp}/profiles_1d[]&grid&psi"][()]
                        rho = f.get(f"{grp}/profiles_1d[]&grid&rho_tor_norm", None)
                        rho = rho[()] if rho is not None else None
                    psi = np.asarray(psi, dtype=float)
                    if psi.ndim >= 2:
                        psi = psi[0]
                    if rho is not None:
                        rr = np.asarray(rho, dtype=float)
                        if rr.ndim >= 2:
                            rr = rr[0]
                        i0 = int(np.nanargmin(rr))
                        i1 = int(np.nanargmax(rr))
                        pa = float(psi[i0])
                        pb = float(psi[i1])
                    else:
                        pa = float(psi[0])
                        pb = float(psi[-1])
                    if np.isfinite(pa) and np.isfinite(pb) and abs(pb - pa) > 1e-12:
                        if log is not None:
                            log.info(f"LCFS from core_profiles occ0: psi_axis≈{pa:.6g}, psi_lcfs≈{pb:.6g} from {fpath}")
                        return _finalize(pa, pb, f"{tag}|core_profiles0")
                except Exception:
                    pass

    # 4) user Te_sep (only used when peqdsk missing)
    if te_sep_ev is None:
        try:
            te_sep_ev = float(getattr(args, "te_sep_ev", 60.0) or 60.0)
        except Exception:
            te_sep_ev = 60.0
    if te2d is not None and np.isfinite(psi_axis) and np.isfinite(psi_lcfs_guess):
        psi_lcfs_u = _psi_lcfs_from_te_sep(psi2d, te2d, te_sep_ev, psi_axis, psi_lcfs_guess, log=log)
        if psi_lcfs_u is not None and np.isfinite(psi_lcfs_u) and abs(float(psi_lcfs_u) - float(psi_axis)) > 1e-12:
            return _finalize(float(psi_axis), float(psi_lcfs_u), f"{tag}|user_te")

    # 5) robust fallback
    if log is not None:
        log.info(f"LCFS fallback: psi_axis≈{psi_axis:.6g}, psi_lcfs≈{psi_lcfs_guess:.6g} (tag={tag})")
    return _finalize(float(psi_axis), float(psi_lcfs_guess), tag)

def _make_bin_edges_from_data(xv: np.ndarray, nbins: int, vmin: float, vmax: float, *, log: Optional[logging.Logger] = None, tag: str = "") -> np.ndarray:
    """Make monotone bin edges for x in [vmin,vmax] that avoid massive empty-bin NaNs.

    If x has only a limited number of unique values, automatically reduce the effective
    number of bins so each bin is populated.
    """
    x = np.asarray(xv, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return np.linspace(vmin, vmax, max(2, int(nbins) + 1), dtype=float)

    # Clip to requested interval for robust quantiles
    x = np.clip(x, vmin, vmax)

    nu = int(np.unique(x).size)
    nb_eff = int(min(int(nbins), max(8, nu - 1)))
    if nb_eff < int(nbins) and log is not None:
        log.info(f"{tag} binning: reduced nbins from {int(nbins)} to {nb_eff} (unique x={nu})")

    qs = np.linspace(0.0, 1.0, nb_eff + 1)
    edges = np.quantile(x, qs)
    edges[0] = float(vmin)
    edges[-1] = float(vmax)
    # enforce strict monotonicity by uniquing
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.asarray([vmin, vmax], dtype=float)
    # final safety: if still too few edges, fall back
    if edges.size < 3:
        edges = np.linspace(vmin, vmax, 3, dtype=float)
    return edges


def _fill_nan_1d(y: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Fill NaNs in a 1D array by linear interpolation; constant extrapolation at ends."""
    if y is None:
        return None
    a = np.asarray(y, dtype=float).copy()
    if a.ndim != 1:
        a = a.ravel()
    good = np.isfinite(a)
    if not np.any(good):
        return a
    x = np.arange(a.size, dtype=float)
    a[~good] = np.interp(x[~good], x[good], a[good])
    return a


def _bin_scalar_on_rho_bins(
    y: Optional[np.ndarray],
    rho2d: np.ndarray,
    edges: np.ndarray,
    mask: np.ndarray,
) -> Optional[np.ndarray]:
    """Bin a 2D scalar y onto 1D rho bins defined by edges.

    Returns bin means (length nbins) or None if y is None.
    """
    if y is None:
        return None
    yy = np.asarray(y, dtype=float)
    rr = np.asarray(rho2d, dtype=float)
    ed = np.asarray(edges, dtype=float).ravel()
    nb = ed.size - 1

    m = mask & np.isfinite(rr) & np.isfinite(yy)
    if not np.any(m):
        return np.full((nb,), np.nan, dtype=float)

    idx = np.digitize(rr[m], ed, right=False) - 1
    good = (idx >= 0) & (idx < nb)
    if not np.any(good):
        return np.full((nb,), np.nan, dtype=float)

    idx = idx[good]
    w = yy[m][good]
    s = np.bincount(idx, weights=w, minlength=nb).astype(float)
    c = np.bincount(idx, minlength=nb).astype(float)
    out = np.full((nb,), np.nan, dtype=float)
    nz = c > 0
    out[nz] = s[nz] / c[nz]
    return out


def _calculate_q_profile(
    R: np.ndarray,
    Z: np.ndarray,
    psi: np.ndarray,
    B_R: np.ndarray,
    B_Z: np.ndarray,
    B_phi: np.ndarray,
    psi_axis: float,
    psi_lcfs: float,
    n_levels: int = 32,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Calculate the safety factor q(psi) from 2D equilibrium fields.

    This function traces contours of the poloidal flux `psi` and performs the line integral
    q(psi) = (1 / 2*pi) *oint (B_phi / (R * B_p)) dl_p for each contour.

    Args:
        R: 2D array of major radius.
        Z: 2D array of vertical position.
        psi: 2D array of poloidal flux.
        B_R: 2D array of radial magnetic field.
        B_Z: 2D array of vertical magnetic field.
        B_phi: 2D array of toroidal magnetic field.
        psi_axis: Poloidal flux at the magnetic axis.
        psi_lcfs: Poloidal flux at the last closed flux surface.
        n_levels: Number of psi contours to calculate q on.

    Returns:
        A tuple of (psi_1d, q_1d) arrays, or None if the calculation fails.
    """
    from matplotlib.figure import Figure
    from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

    if not all(
        x is not None and np.any(np.isfinite(x))
        for x in [R, Z, psi, B_R, B_Z, B_phi]
    ):
        return None

    # Create a set of psi levels from axis to LCFS
    psi_levels = np.linspace(psi_axis, psi_lcfs, n_levels + 2)[1:-1]
    q_values = []
    psi_values = []

    with np.errstate(divide='ignore', invalid='ignore'):
        B_p = np.sqrt(B_R**2 + B_Z**2)
        integrand = B_phi / (R * B_p)

    interp_fn = None
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    integrand = np.asarray(integrand, dtype=float)
    if (
        R.ndim == 2 and Z.ndim == 2 and integrand.ndim == 2
        and R.shape == Z.shape == integrand.shape
        and R.shape[0] >= 2 and R.shape[1] >= 2
    ):
        r_axis = np.asarray(R[:, 0], dtype=float)
        z_axis = np.asarray(Z[0, :], dtype=float)
        is_structured = (
            np.allclose(R, r_axis[:, None], equal_nan=True)
            and np.allclose(Z, z_axis[None, :], equal_nan=True)
        )
        if is_structured:
            if r_axis[1] < r_axis[0]:
                r_axis = r_axis[::-1]
                integrand = integrand[::-1, :]
            if z_axis[1] < z_axis[0]:
                z_axis = z_axis[::-1]
                integrand = integrand[:, ::-1]
            interp_fn = RegularGridInterpolator(
                (r_axis, z_axis),
                integrand,
                method='linear',
                bounds_error=False,
                fill_value=np.nan,
            )

    if interp_fn is None:
        points = np.column_stack((R.ravel(), Z.ravel()))
        values = integrand.ravel()
        point_mask = np.all(np.isfinite(points), axis=1) & np.isfinite(values)
        if np.count_nonzero(point_mask) < 3:
            return None
        interp_fn = LinearNDInterpolator(points[point_mask], values[point_mask], fill_value=np.nan)

    fig = Figure()
    ax = fig.subplots()
    try:
        # Generate contours for each psi level without relying on pyplot/GUI state.
        cs = ax.contour(R, Z, psi, levels=psi_levels)

        for level, segs in zip(cs.levels, cs.allsegs):
            segments = [
                np.asarray(seg, dtype=float)
                for seg in segs
                if seg is not None and np.asarray(seg).ndim == 2 and np.asarray(seg).shape[0] >= 2
            ]
            if not segments:
                continue

            # Prefer the longest contour segment for this psi level.
            vertices = max(segments, key=lambda seg: seg.shape[0])
            r_path = vertices[:, 0]
            z_path = vertices[:, 1]

            # Interpolate the integrand onto the contour path.
            integrand_path = interp_fn(np.column_stack((r_path, z_path)))
            if integrand_path is None:
                continue
            integrand_path = np.asarray(integrand_path, dtype=float).reshape(-1)

            dl_p = np.sqrt(
                np.diff(r_path, prepend=r_path[0]) ** 2
                + np.diff(z_path, prepend=z_path[0]) ** 2
            )
            s_path = np.cumsum(dl_p)
            good = np.isfinite(integrand_path) & np.isfinite(s_path)
            if np.count_nonzero(good) < 2:
                continue

            integral = np.trapz(integrand_path[good], x=s_path[good])
            if not np.isfinite(integral):
                continue

            q_values.append(integral / (2 * np.pi))
            psi_values.append(level)
    finally:
        fig.clear()

    if not psi_values:
        return None

    return np.array(psi_values), np.array(q_values)


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

def read_and_stitch_dump(fn: Path, args) -> Dict[str, Any]:
    log = logging.getLogger(__name__)
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

        # Get zimp for renz/imnz
        nimrod_in_guess = None
        try:
            cand = fn.parent / "nimrod.in"
            if cand.is_file():
                nimrod_in_guess = str(cand)
        except Exception:
            nimrod_in_guess = None
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
        psi_src = None
        if psi_eq is not None:
            psi_src = "dump:psi_eq"
        if psi_eq is None:
            # Older/alternate NIMROD dumps may store poloidal flux under different names.
            # We fall back to common variants so 1D profile grids can still be produced.
            for _cand in ("psi", "psiq", "psi_pol", "psiRZ", "psip", "psif"):
                psi_eq = try_read_scalar(_cand)
                if psi_eq is not None:
                    psi_src = f"dump:{_cand}"
                    break
        bq = try_read_vec3("bq")
        if psi_eq is None and bq is not None and R is not None and Z is not None:
            # Non-impurity dumps sometimes omit psi_eq; reconstruct from (B_R,B_Z) if possible.
            psi_eq = _reconstruct_psi_from_bq(R, Z, bq)
            if psi_eq is not None:
                psi_src = "reconstruct:bq"
            else:
                psi_src = "reconstruct:bq_failed"
        if psi_src is None:
            psi_src = "missing"

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
            for nm in ("repr", "impr", "repe", "impe", "rete", "imte", "reti", "imti", "reqlosl", "imqlosl", "reqloso", "imqloso"):
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

            # Custom impurity charge state densities
            try:
                nimrod_species_info = _nimrod_species_info(nimrod_in_guess)
                zimp = int(nimrod_species_info.get('zimp', 0))
                nspec_imp = zimp + 1 if zimp > 0 else 0

                def _infer_nspec_from_block() -> int:
                    try:
                        a0 = _read_block_ds(f, "renz", bids[0])
                        a0 = _squeeze1(a0)
                        if a0.ndim == 4:
                            if a0.shape[2] == nmodes:
                                return int(a0.shape[3])
                            if a0.shape[3] == nmodes:
                                return int(a0.shape[2])
                        if a0.ndim == 3 and nmodes > 0:
                            k = int(a0.shape[2])
                            if k % nmodes == 0:
                                return int(k // nmodes)
                    except Exception:
                        return 0
                    return 0

                inferred_nspec = _infer_nspec_from_block()
                if inferred_nspec > 0 and inferred_nspec != nspec_imp:
                    if log is not None:
                        if log is not None:
                            log.info(f"Using inferred nspec_imp={inferred_nspec} (nimrod.in zimp implies {nspec_imp}).")
                    nspec_imp = inferred_nspec

                if nspec_imp > 0:
                    def read_impurity_modes(base: str) -> np.ndarray:
                        blocks = []
                        for bid in bids:
                            a = _read_block_ds(f, base, bid)
                            a = _squeeze1(a)
                            a = _unpack_multispecies_scalar_modes(a, nmodes, nspec_imp, 'species_major')
                            blocks.append(a.astype(float, copy=False))
                        return stitch_blocks(blocks, nxbl, nybl, ordering)

                    try:
                        fields["renz"] = read_impurity_modes("renz")
                        fields["imnz"] = read_impurity_modes("imnz")
                        fields["nspec_imp"] = np.array([nspec_imp], dtype=int)
                        if log is not None:
                            if log is not None:
                                log.info(f"Read renz/imnz for {nspec_imp} impurity charge states.")
                    except KeyError:
                        if log is not None:
                            if log is not None:
                                log.info("renz/imnz not found in dump file, skipping impurity mapping.")
                    except Exception as e:
                        if log is not None:
                            if log is not None:
                                log.warning(f"Could not process renz/imnz: {e}")
                else:
                    if log is not None:
                        if log is not None:
                            log.info("renz/imnz not found or nspec_imp=0; skipping impurity mapping.")
            except Exception as e:
                if log is not None:
                    if log is not None:
                        log.warning(f"Could not read impurity info for renz/imnz: {e}")
                # Store 2D fields in a stitched IMAS-friendly (dim1,dim2) layout.
        #
        # NIMROD rblock stitching produces arrays shaped (Ny, Nx) where the first axis is the
        # "y-like" direction (typically Z) and the second axis is the "x-like" direction (typically R).
        # For IMAS RZ grids (and for user-facing contour plots), we store arrays as (Nx, Ny),
        # i.e. dim1 corresponds to R-index and dim2 to Z-index.

        # --- Single-ion/no-impurity compatibility ---
        # Some NIMROD dumps provide only one density channel (nq[...,0]) even though IMAS
        try:
            nq, fields, nspec_eq = _expand_single_ion_to_e_plus_main(nq, fields, nspec_eq, nmodes, args, nimrod_in_guess)
        except Exception:
            pass

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
                elif k in ("reqlosl", "imqlosl", "reqloso", "imqloso"):
                    vv = vv * float(getattr(args, "power_density_scale", 1.0))
                fields_imas[k] = vv
            except Exception:
                pass

        # Convert from NIMROD's native COCOS=12 convention to IMAS COCOS=11.
        # This is applied only within dump2imas outputs (input2imas preserves original PEQDSK signs).
        try:
            _cocos_log = logging.getLogger("dump2imas")
            _cocos_payload = dict(
                psi_eq=psi_eq_imas,
                bq=bq_imas,
                vq=vq_imas,
                jq=jq_imas,
                fields=fields_imas,
            )
            
            _sanity_print_cocos("PRE", fn.name, float(t0), _cocos_payload, args)
            _apply_cocos_12_to_11_inplace(_cocos_payload, _cocos_log)

            _sanity_print_cocos("POST", fn.name, float(t0), _cocos_payload, args)
            psi_eq_imas = _cocos_payload.get("psi_eq", psi_eq_imas)
            bq_imas = _cocos_payload.get("bq", bq_imas)
            vq_imas = _cocos_payload.get("vq", vq_imas)
            jq_imas = _cocos_payload.get("jq", jq_imas)
            fields_imas = _cocos_payload.get("fields", fields_imas)
        except Exception:
            pass

        return dict(
            time=float(t0),
            keff=_as_f64(keff),
            nmodes=int(nmodes),
            R=_as_f64(R_imas),
            Z=_as_f64(Z_imas),
            psi_eq=_as_f64(psi_eq_imas) if psi_eq_imas is not None else None,
            psi_eq_source=str(psi_src),
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




def _reconstruct_psi_from_bq(R: np.ndarray, Z: np.ndarray, bq: np.ndarray) -> Optional[np.ndarray]:
    """Best-effort reconstruction of axisymmetric poloidal flux psi(R,Z) from B_R and B_Z.

    Assumes cylindrical coordinates where (B_R, B_Z) relate to poloidal flux as:
        B_R = -(1/R) * dpsi/dZ
        B_Z =  (1/R) * dpsi/dR

    This corresponds to the COCOS=12 sign convention (sigma_{R phi Z}=-1). The
    surrounding dump2imas workflow will subsequently convert the reconstructed
    psi to COCOS=11 by flipping its sign.

    NOTE: this relation corresponds to sigma_{R phi Z} = -1 (COCOS=12-type) in the Sauter
    coordinate-convention definitions. dump2imas enforces COCOS=11 output by flipping the
    sign of psi after stitching/reconstruction.

    We reconstruct psi on a logically-rectangular grid by:
      1) selecting a reference point near the magnetic axis (min |B_p|)
      2) integrating along Z to get psi at the reference R column
      3) integrating along R to fill each row

    The result is defined up to an additive constant; we set psi(axis)=0.

    Returns None if inputs are unusable.
    """
    try:
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        bq = np.asarray(bq, dtype=float)
        if R.ndim != 2 or Z.ndim != 2 or bq.ndim != 3 or bq.shape[2] < 2:
            return None

        BR = bq[..., 0]
        BZ = bq[..., 1]

        # Guard against R<=0
        Rpos = np.where(R > 0.0, R, np.nan)

        # Pick axis as min |B_p| (robust for equilibria; avoids needing psi)
        Bp2 = BR**2 + BZ**2
        # exclude NaNs
        idx_flat = np.nanargmin(Bp2)
        i0, j0 = np.unravel_index(idx_flat, Bp2.shape)

        ny, nx = R.shape
        psi = np.full((ny, nx), np.nan, dtype=float)

        # --- integrate along Z at fixed column j0 to get psi[:,j0] ---
        # Use local R at that column; assume Z varies primarily along axis 0.
        # Build a 1D Z coordinate from that column if possible.
        Zcol = Z[:, j0]
        Rcol = Rpos[:, j0]
        BRcol = BR[:, j0]

        psi[i0, j0] = 0.0

        # Upward (i0+1..)
        for i in range(i0 + 1, ny):
            dz = Zcol[i] - Zcol[i - 1]
            # trapezoid for BR
            brm = 0.5 * (BRcol[i] + BRcol[i - 1])
            rm = 0.5 * (Rcol[i] + Rcol[i - 1])
            if not np.isfinite(dz) or not np.isfinite(brm) or not np.isfinite(rm):
                psi[i, j0] = psi[i - 1, j0]
            else:
                psi[i, j0] = psi[i - 1, j0] + (-rm * brm) * dz

        # Downward (i0-1..0)
        for i in range(i0 - 1, -1, -1):
            dz = Zcol[i + 1] - Zcol[i]
            brm = 0.5 * (BRcol[i + 1] + BRcol[i])
            rm = 0.5 * (Rcol[i + 1] + Rcol[i])
            if not np.isfinite(dz) or not np.isfinite(brm) or not np.isfinite(rm):
                psi[i, j0] = psi[i + 1, j0]
            else:
                psi[i, j0] = psi[i + 1, j0] - (-rm * brm) * dz  # reverse step

        # --- integrate along R for each row i using BZ ---
        for i in range(ny):
            psi[i, j0] = 0.0 if not np.isfinite(psi[i, j0]) else psi[i, j0]
            Rrow = Rpos[i, :]
            BZrow = BZ[i, :]
            # right
            for j in range(j0 + 1, nx):
                dR = Rrow[j] - Rrow[j - 1]
                bzm = 0.5 * (BZrow[j] + BZrow[j - 1])
                rm = 0.5 * (Rrow[j] + Rrow[j - 1])
                if not np.isfinite(dR) or not np.isfinite(bzm) or not np.isfinite(rm):
                    psi[i, j] = psi[i, j - 1]
                else:
                    psi[i, j] = psi[i, j - 1] + (rm * bzm) * dR
            # left
            for j in range(j0 - 1, -1, -1):
                dR = Rrow[j + 1] - Rrow[j]
                bzm = 0.5 * (BZrow[j + 1] + BZrow[j])
                rm = 0.5 * (Rrow[j + 1] + Rrow[j])
                if not np.isfinite(dR) or not np.isfinite(bzm) or not np.isfinite(rm):
                    psi[i, j] = psi[i, j + 1]
                else:
                    psi[i, j] = psi[i, j + 1] - (rm * bzm) * dR  # reverse step

        # Normalize offset so axis point is zero
        psi = psi - float(psi[i0, j0])

        # If everything is NaN, give up
        if not np.isfinite(psi).any():
            return None

        return psi
    except Exception:
        return None


def _psi_from_mhd_fallback_h5(
    entry_dir: str,
    expected_shape: Tuple[int, int],
    occ_candidates: Sequence[int],
    *,
    log: Optional[logging.Logger] = None,
) -> Optional[np.ndarray]:
    """Try to load psi(R,Z) from an existing IMAS mhd IDS stored in the entry directory.

    This is intended as a last-resort fallback when the current dumpgll file does not contain psi
    and reconstruction from B-fields fails.

    We support both common HDF5-backend layouts:
      - file "mhd_<occ>.h5" with group "/mhd_<occ>" (IMAS HDF5 backend)
      - file "mhd.h5" with group "/mhd" (standalone IDS export)

    Strategy:
      - scan datasets whose name hints at poloidal flux (psi/psin/poloidal)
      - accept rank-2 datasets matching expected_shape
      - accept rank-3 datasets with a singleton time dimension, taking index 0
      - if shape matches the transpose, transpose and log a warning
    """
    import os
    import h5py
    import numpy as np

    if log is None:
        log = logging.getLogger(__name__)

    ex0, ex1 = int(expected_shape[0]), int(expected_shape[1])

    def _iter_datasets(g: "h5py.Group", prefix: str = ""):
        for k, v in g.items():
            p = f"{prefix}/{k}" if prefix else k
            if isinstance(v, h5py.Dataset):
                yield p, v
            elif isinstance(v, h5py.Group):
                yield from _iter_datasets(v, p)

    def _try_extract(ds: "h5py.Dataset") -> Optional[np.ndarray]:
        try:
            a = ds[()]
        except Exception:
            return None
        a = np.asarray(a)
        # squeeze singleton dims cautiously
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0, ...]
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[..., 0]
        if a.ndim != 2:
            return None
        if a.shape == (ex0, ex1):
            return a.astype(float, copy=False)
        if a.shape == (ex1, ex0):
            log.warning("psi fallback from mhd: dataset appears transposed (%s); transposing to match expected shape.", a.shape)
            return np.transpose(a).astype(float, copy=False)
        return None

    # Prefer mhd_<occ>.h5, then mhd.h5
    file_specs: List[Tuple[str, str]] = []
    for occ in occ_candidates:
        file_specs.append((os.path.join(entry_dir, f"mhd_{int(occ)}.h5"), f"mhd_{int(occ)}"))
    file_specs.append((os.path.join(entry_dir, "mhd.h5"), "mhd"))

    name_hints = ("psi", "psin", "poloidal", "polflux", "psip", "psif")

    for h5_path, grp_name in file_specs:
        if not os.path.exists(h5_path):
            continue
        try:
            with h5py.File(h5_path, "r") as h5:
                if grp_name not in h5:
                    continue
                g = h5[grp_name]
                # Pass 1: datasets with explicit name hints
                for p, ds in _iter_datasets(g):
                    pname = p.lower()
                    if not any(h in pname for h in name_hints):
                        continue
                    a = _try_extract(ds)
                    if a is not None:
                        log.info("psi fallback selected: mhd IDS dataset '%s' in %s", p, os.path.basename(h5_path))
                        return _as_f64(a)

                # Pass 2: if nothing matched, attempt any rank-2 dataset with the right shape
                for p, ds in _iter_datasets(g):
                    a = _try_extract(ds)
                    if a is None:
                        continue
                    log.info("psi fallback selected (shape match): mhd IDS dataset '%s' in %s", p, os.path.basename(h5_path))
                    return _as_f64(a)
        except Exception as e:
            log.warning("psi fallback: failed reading %s (%s)", h5_path, e)
            continue

    return None


def _ensure_psi_eq_available(
    data: Dict[str, Any],
    entry_dir: str,
    occ_base: int,
    *,
    log: Optional[logging.Logger] = None,
) -> None:
    """Ensure data['psi_eq'] exists, trying sequential fallbacks and logging the chosen path."""
    if log is None:
        log = logging.getLogger(__name__)

    psi = data.get("psi_eq", None)
    src = data.get("psi_eq_source", "unknown")

    if psi is not None:
        log.info("psi source selected: %s", src)
        return

    # At this point, dump did not provide psi and (if attempted) B-field reconstruction failed.
    # Try to fetch psi from an existing mhd IDS written during preprocessing.
    expected_shape = tuple(int(x) for x in np.asarray(data.get("R")).shape)  # (Nx,Ny) in stored IMAS orientation
    occ_candidates = [int(occ_base), 0, 1]
    psi2 = _psi_from_mhd_fallback_h5(entry_dir, expected_shape, occ_candidates, log=log)
    if psi2 is not None:
        data["psi_eq"] = _as_f64(psi2)
        data["psi_eq_source"] = "fallback:mhd"
        log.info("psi source selected: fallback:mhd")
        return

    log.warning("psi not available: dump psi missing; B-field reconstruction failed; mhd fallback not found. 1D profiles may be empty.")


def populate_equilibrium(eq: Any, data: Dict[str, Any], t_index: int, quiet: bool) -> None:
    t = float(data["time"])

    # IMAS output convention (dump2imas enforces COCOS=11)

    _set_ids_cocos(eq, COCOS_OUT_DEFAULT)
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

    # --- 1D profiles (q-profile) ---
    try:
        psi_axis, psi_lcfs, _ = _choose_psi_axis_lcfs(data, args, log)
        if np.isfinite(psi_axis) and np.isfinite(psi_lcfs):
            q_result = _calculate_q_profile(
                R, Z, psi, bq[..., 0], bq[..., 1], bq[..., 2],
                psi_axis, psi_lcfs, n_levels=64
            )
            if q_result:
                psi_1d, q_1d = q_result
                p1d = ts.profiles_1d
                p1d.psi = psi_1d
                p1d.q = q_1d
                log.info("Calculated and populated equilibrium.profiles_1d.q")
    except Exception as exc:
        # Non-fatal: q-profile is a derived quantity.
        log.warning(f"Failed to calculate q-profile: {exc}")
    
    # Pass q-profile result to other populators via the data dictionary
    # so it can be interpolated onto other grids (e.g. core_profiles).
    if 'q_result' in locals() and locals()['q_result']:
        data['q_result'] = locals()['q_result']
    else:
        data['q_result'] = None


    # metadata
    try:
        eq.code.name = "NIMROD"
    except Exception:
        pass

    if not quiet:
        _log("Populated equilibrium IDS", quiet=False)




def populate_core_profiles(cp: Any, data: Dict[str, Any], t_index: int, args) -> None:
    """Populate core_profiles.profiles_1d.

    Requirements (per your updated conventions):
      - Store 1D profiles as functions of normalized *toroidal* flux in core_profiles.
        Since toroidal flux is not available directly from the NIMROD dump, we use the
        common approximation rho_tor_norm ~ sqrt(psi_pol_norm), where
            psi_pol_norm = (psi - psi_axis)/(psi_lcfs - psi_axis),
        so psi_pol_norm=0 at the magnetic axis and =1 at the LCFS.
      - Always store absolute poloidal flux in grid.psi (Wb).
      - When the schema supports it, also store psi_norm / rho_pol_norm for downstream tooling.
    """
    log = logging.getLogger(__name__)
    t = float(data.get("time", 0.0))

    # IMAS output convention (dump2imas enforces COCOS=11)

    _set_ids_cocos(cp, COCOS_OUT_DEFAULT)

    psi2d = data.get("psi_eq", None)
    if psi2d is None:
        return

    pr2d = data.get("prq", None)   # total pressure (Pa)
    pe2d = data.get("peq", None)   # electron pressure (Pa), optional
    nq = data.get("nq", None)      # densities, last dim (0=e)
    te2d = data.get("teq", None)   # eV
    ti2d = data.get("tiq", None)   # eV
    vq = data.get("vq", None)
    jq = data.get("jq", None)
    R = data.get("R", None)

    # Elementary charge (C) used when deriving Te from p/n
    sp = getattr(args, "_nimrod_species", {}) or {}
    qe = float(sp.get("qe_c", 1.602176634e-19))
    n_scale = float(getattr(args, "n_scale", 1.0) or 1.0)

    psi_axis, psi_lcfs, tag = _choose_psi_axis_lcfs(data, args, log=log)
    if (not np.isfinite(psi_axis)) or (not np.isfinite(psi_lcfs)) or abs(psi_lcfs - psi_axis) < 1e-12:
        log.warning("core_profiles: cannot determine psi_axis/psi_lcfs; skipping (tag=%s)", str(tag))
        return
    den = float(psi_lcfs - psi_axis)

    ps = np.asarray(psi2d, dtype=float)
    psi_pol_norm2d = (ps - float(psi_axis)) / den

    # Optional: include perturbations in core_profiles when requested.
    # For --edge-ggd-values=full we reconstruct a "full field" snapshot at phi=0:
    #   full = equilibrium + pert_scale * sum_k [ re_k*cos(n_k*phi) - im_k*sin(n_k*phi) ].
    edge_mode = str(getattr(args, "edge_ggd_values", "equilibrium") or "equilibrium").strip().lower()
    _cp_vphi2d = None
    _cp_jphi2d = None
    _dens2d_cache = {}
    if edge_mode == "full" and bool(getattr(args, "_nonlinear_run", False)):
        fields = data.get("fields", {}) or {}
        keff = np.asarray(data.get("keff"), dtype=float) if data.get("keff") is not None else np.arange(int(data.get("nmodes", 0) or 0), dtype=float)
        phi0 = 0.0
        try:
            _pscl = float(getattr(args, "pert_scale", 1.0) or 1.0)
        except Exception:
            _pscl = 1.0

        def _slice_modes_cp(A, *, spec=None, comp=None):
            if A is None:
                return None
            AA = np.asarray(A)
            nm = int(len(keff))
            # vector component extraction
            if comp is not None:
                c = int(comp)
                if AA.ndim == 4:
                    if AA.shape[-1] == 3 and AA.shape[-2] == nm:
                        return AA[:, :, :, c]
                    if AA.shape[2] == 3 and AA.shape[3] == nm:
                        return AA[:, :, c, :]
                if AA.ndim == 3 and AA.shape[-1] == 3 * nm:
                    return AA[:, :, c * nm:(c + 1) * nm]
                return None
            # density/species extraction
            if spec is not None:
                s = int(spec)
                if AA.ndim == 4:
                    if AA.shape[-1] == nm:
                        return AA[:, :, s, :] if (0 <= s < AA.shape[2]) else None
                    if AA.shape[2] == nm:
                        return AA[:, :, :, s] if (0 <= s < AA.shape[3]) else None
                return None
            # scalar modes
            if AA.ndim == 3 and AA.shape[-1] == nm:
                return AA
            return None

        def _recon_cp(eq2d, reA, imA):
            if eq2d is None and (reA is None or imA is None):
                return None
            out = _reconstruct_full_from_modes(eq2d, reA, imA, keff, float(phi0), pert_scale=_pscl)
            return np.asarray(out, dtype=float) if out is not None else None

        def _recon_key(eq2d, rekey, imkey, *, spec=None, comp=None):
            return _recon_cp(eq2d, _slice_modes_cp(fields.get(rekey), spec=spec, comp=comp),
                             _slice_modes_cp(fields.get(imkey), spec=spec, comp=comp))

        # Override scalar equilibrium fields (when corresponding perturbations are present)
        if (_tmp := _recon_key(pr2d, "repr", "impr")) is not None:
            pr2d = _tmp
        if (_tmp := _recon_key(pe2d, "repe", "impe")) is not None:
            pe2d = _tmp
        if (_tmp := _recon_key(te2d, "rete", "imte")) is not None:
            te2d = _tmp
        if (_tmp := _recon_key(ti2d, "reti", "imti")) is not None:
            ti2d = _tmp

        # Density helper (spec=0 is electrons; spec>=1 are ion species)
        def dens2d(spec_index: int):
            key = int(spec_index)
            if key in _dens2d_cache:
                return _dens2d_cache[key]
            eq = None
            try:
                if nq is not None and getattr(nq, "ndim", 0) >= 3 and key < int(nq.shape[-1]):
                    eq = np.asarray(nq[..., key], dtype=float)
                elif nq is not None and getattr(nq, "ndim", 0) == 2 and key == 0:
                    eq = np.asarray(nq, dtype=float)
            except Exception:
                eq = None
            full = _recon_key(eq, "rend", "imnd", spec=key)
            out = full if full is not None else eq
            _dens2d_cache[key] = out
            return out

        # Toroidal velocity/current overrides (used later in binning)
        try:
            vphi_eq = np.asarray(vq[..., 2], dtype=float) if (vq is not None and getattr(vq, "ndim", 0) >= 3) else None
            _tmp = _recon_key(vphi_eq, "reve", "imve", comp=2)
            _cp_vphi2d = _tmp if _tmp is not None else vphi_eq
        except Exception:
            _cp_vphi2d = None
        try:
            jphi_eq = np.asarray(jq[..., 2], dtype=float) if (jq is not None and getattr(jq, "ndim", 0) >= 3) else None
            _tmp = _recon_key(jphi_eq, "reja", "imja", comp=2)
            _cp_jphi2d = _tmp if _tmp is not None else jphi_eq
        except Exception:
            _cp_jphi2d = None

    # --- plasma-like mask: stay inside LCFS and avoid vacuum artifacts ---
    m = np.isfinite(psi_pol_norm2d)
    m &= (psi_pol_norm2d >= 0.0) & (psi_pol_norm2d <= 1.0 + 1e-6)

    # Prefer Te > te_min, else use p/n, else use p>0 or n>0.
    te_min = float(getattr(args, "te_min_ev", 20.0) or 20.0)
    
    ne2d = None
    try:
        if nq is not None:
            _A = np.asarray(nq, dtype=float)
            if _A.ndim == 2:
                ne2d = _A
            elif _A.ndim >= 3 and _A.shape[-1] >= 1:
                ne2d = _A[..., 0]
    except Exception:
        ne2d = None

    # If "full field" was requested, override ne2d using density perturbations (spec=0).
    if edge_mode == "full" and bool(getattr(args, "_nonlinear_run", False)):
        try:
            ne2d_full = dens2d(0)
            if ne2d_full is not None:
                ne2d = ne2d_full
        except Exception:
            pass

    if te2d is not None:
        te = np.asarray(te2d, dtype=float)
        m &= np.isfinite(te) & (te > te_min)
    else:
        te_est = None
        if pe2d is not None and ne2d is not None:
            pe = np.asarray(pe2d, dtype=float)
            te_est = np.full_like(pe, np.nan, dtype=float)
            mm = np.isfinite(pe) & np.isfinite(ne2d) & (ne2d > 0.0)
            te_est[mm] = pe[mm] / (ne2d[mm] * qe)
            m &= np.isfinite(te_est) & (te_est > te_min)
        elif pr2d is not None:
            pr = np.asarray(pr2d, dtype=float)
            m &= np.isfinite(pr) & (pr > 0.0)
        elif ne2d is not None:
            m &= np.isfinite(ne2d) & (ne2d > 0.0)

    kept = int(np.count_nonzero(m))
    if kept < 50:
        log.warning("core_profiles: too few valid points for 1D averaging (%d); skipping", kept)
        return

    nbins = int(getattr(args, "nbins", 256) or 256)
    edges = _make_bin_edges_from_data(psi_pol_norm2d[m], nbins, 0.0, 1.0, log=log, tag="core_profiles")
    psi_pol_norm_1d = 0.5 * (edges[:-1] + edges[1:])
    psi_abs_1d = float(psi_axis) + psi_pol_norm_1d * den

    # Approximate toroidal normalized flux coordinate
    psi_tor_norm_1d = np.clip(psi_pol_norm_1d, 0.0, 1.0)
    rho_tor_norm_1d = np.sqrt(np.clip(psi_tor_norm_1d, 0.0, None))

    # ---- 1D binned quantities ----
    p_tot1d = _bin_scalar_on_rho_bins(pr2d, psi_pol_norm2d, edges, m)
    pe1d = _bin_scalar_on_rho_bins(pe2d, psi_pol_norm2d, edges, m)

    ne1d = _bin_scalar_on_rho_bins(ne2d, psi_pol_norm2d, edges, m) if (ne2d is not None) else None

    ion_dens: List[Optional[np.ndarray]] = []
    if nq is not None and getattr(nq, "ndim", 0) >= 3 and nq.shape[-1] >= 2:
        for s in range(1, int(nq.shape[-1])):
            ion_dens.append(_bin_scalar_on_rho_bins((dens2d(s) if (edge_mode == "full" and bool(getattr(args, "_nonlinear_run", False))) else nq[..., s]), psi_pol_norm2d, edges, m))

    # No explicit ion channels (single-ion/no-impurity dump): synthesize a single main-ion density
    # using constant zeff_input (preferred) or Z_main when available.
    if (not ion_dens) and (ne2d is not None):
        z_main = 1.0
        try:
            zlist = sp.get("z_ions", None) or []
            if zlist:
                z_main = float(zlist[0])
        except Exception:
            z_main = 1.0
        zeff_input = sp.get("zeff_input", None)
        try:
            zeff_v = float(zeff_input) if zeff_input not in (None, "") else None
        except Exception:
            zeff_v = None
        divisor = float(zeff_v) if (zeff_v is not None and np.isfinite(zeff_v) and zeff_v > 0.0) else float(z_main if (np.isfinite(z_main) and z_main > 0.0) else 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ni2d_eff = np.asarray(ne2d, dtype=float) / divisor
        ion_dens.append(_bin_scalar_on_rho_bins(ni2d_eff, psi_pol_norm2d, edges, m))

    # Electron temperature: prefer teq; else derive from pe/ne when possible
    te1d = _bin_scalar_on_rho_bins(te2d, psi_pol_norm2d, edges, m) if (te2d is not None) else None
    if te1d is None and pe1d is not None and ne1d is not None:
        te1d = _as_f64(pe1d) / (_as_f64(ne1d) * qe)

    # Electron pressure: prefer peq; else derive from ne*Te
    if pe1d is None and te1d is not None and ne1d is not None:
        pe1d = _as_f64(ne1d) * qe * _as_f64(te1d)

    # Ion temperature: prefer tiq; else derive from (ptot - pe)/ni_total
    ti1d = _bin_scalar_on_rho_bins(ti2d, psi_pol_norm2d, edges, m) if (ti2d is not None) else None
    if ti1d is None and (p_tot1d is not None) and (pe1d is not None) and ion_dens:
        ni_tot = np.zeros_like(_as_f64(ion_dens[0]))
        for ni in ion_dens:
            if ni is not None:
                ni_tot = ni_tot + _as_f64(ni)
        pi1d = _as_f64(p_tot1d) - _as_f64(pe1d)
        good = np.isfinite(ni_tot) & (ni_tot > 0.0) & np.isfinite(pi1d)
        ti_tmp = np.full_like(pi1d, np.nan, dtype=float)
        ti_tmp[good] = pi1d[good] / (ni_tot[good] * qe)
        ti1d = ti_tmp

    # Ion pressure (best-effort)
    pi1d = None
    if ti1d is not None and ion_dens:
        ni_tot = np.zeros_like(_as_f64(ti1d))
        for ni in ion_dens:
            if ni is not None:
                ni_tot = ni_tot + _as_f64(ni)
        pi1d = ni_tot * qe * _as_f64(ti1d)

    # Toroidal velocity v_phi and omega_tor = vphi/R
    vtor1d = _bin_scalar_on_rho_bins((_cp_vphi2d if _cp_vphi2d is not None else vq[..., 2]), psi_pol_norm2d, edges, m) if ((_cp_vphi2d is not None) or (vq is not None)) else None
    omega1d = None
    if ((_cp_vphi2d is not None) or (vq is not None)) and R is not None:
        try:
            RR = np.asarray(R, dtype=float)
            vv = np.asarray((_cp_vphi2d if _cp_vphi2d is not None else vq[..., 2]), dtype=float)
            omega2d = np.full_like(RR, np.nan, dtype=float)
            mm = m & np.isfinite(RR) & (np.abs(RR) > 0) & np.isfinite(vv)
            omega2d[mm] = vv[mm] / RR[mm]
            omega1d = _bin_scalar_on_rho_bins(omega2d, psi_pol_norm2d, edges, m)
        except Exception:
            omega1d = None

    # Toroidal current density
    jtor1d = _bin_scalar_on_rho_bins((_cp_jphi2d if _cp_jphi2d is not None else jq[..., 2]), psi_pol_norm2d, edges, m) if ((_cp_jphi2d is not None) or (jq is not None)) else None

    # ---- reduce empty-bin NaNs (common with quantized flux labels) ----
    te1d = _fill_nan_1d(te1d)
    ne1d = _fill_nan_1d(ne1d)
    pe1d = _fill_nan_1d(pe1d)
    ti1d = _fill_nan_1d(ti1d)
    pi1d = _fill_nan_1d(pi1d)
    p_tot1d = _fill_nan_1d(p_tot1d)
    vtor1d = _fill_nan_1d(vtor1d)
    omega1d = _fill_nan_1d(omega1d)
    jtor1d = _fill_nan_1d(jtor1d)
    ion_dens = [_fill_nan_1d(x) for x in ion_dens]

    # append profile entry (keeps ids.time consistent with AoS)
    idx = _append_time_core_profiles(cp, t)
    p = cp.profiles_1d[idx]
    try:
        p.time = float(t)
    except Exception:
        pass

    # ---- grid coordinates ----
    if hasattr(p, "grid"):
        g = p.grid
        if hasattr(g, "psi"):
            try:
                g.psi = _as_f64(psi_abs_1d)  # Wb
            except Exception:
                pass
        if hasattr(g, "rho_tor_norm"):
            try:
                g.rho_tor_norm = _as_f64(rho_tor_norm_1d)
            except Exception:
                pass
        if hasattr(g, "psi_tor_norm"):
            try:
                g.psi_tor_norm = _as_f64(psi_tor_norm_1d)
            except Exception:
                pass
        # Optional poloidal-normalized coordinate (helpful for plotting/debug)
        if hasattr(g, "psi_norm"):
            try:
                g.psi_norm = _as_f64(np.clip(psi_pol_norm_1d, 0.0, 1.0))
            except Exception:
                pass
        if hasattr(g, "rho_pol_norm"):
            try:
                g.rho_pol_norm = _as_f64(np.clip(psi_pol_norm_1d, 0.0, 1.0))
            except Exception:
                pass

    # ---- populate electrons ----
    try:
        e = p.electrons
        if ne1d is not None and hasattr(e, "density"):
            e.density = _as_f64(ne1d)
        if te1d is not None and hasattr(e, "temperature"):
            e.temperature = _as_f64(te1d)
        if pe1d is not None and hasattr(e, "pressure"):
            e.pressure = _as_f64(pe1d)
    except Exception:
        pass

    # ---- populate ions ----
    try:
        nion = len(ion_dens)
        if hasattr(p, "ion") and nion > 0:
            p.ion.resize(nion)
            # If a z list is known, use it; otherwise default Z=1 for all ions
            z_list = sp.get("z_ion", None)
            for i in range(nion):
                ion = p.ion[i]
                if ion_dens[i] is not None and hasattr(ion, "density"):
                    ion.density = _as_f64(ion_dens[i])
                if ti1d is not None and hasattr(ion, "temperature"):
                    ion.temperature = _as_f64(ti1d)
                # pressure per species is not uniquely defined; store total ion pressure when leaf exists
                if pi1d is not None and hasattr(ion, "pressure"):
                    ion.pressure = _as_f64(pi1d)
                if hasattr(ion, "z_ion"):
                    try:
                        if isinstance(z_list, (list, tuple)) and i < len(z_list):
                            ion.z_ion = float(z_list[i])
                        else:
                            ion.z_ion = 1.0
                    except Exception:
                        pass
    except Exception:
        pass

    # Current profile (best-effort)
    try:
        if jtor1d is not None and hasattr(p, "j_phi"):
            p.j_phi = _as_f64(jtor1d)
    except Exception:
        pass

    # ---- q-profile (interpolated from equilibrium calculation) ----
    q_result = data.get("q_result", None)
    if q_result:
        try:
            psi_q, q_vals = q_result
            # Interpolate q(psi) onto the core_profiles psi grid
            q_1d = np.interp(psi_abs_1d, psi_q, q_vals, left=np.nan, right=np.nan)
            q_1d = _fill_nan_1d(q_1d)
            if q_1d is not None and np.any(np.isfinite(q_1d)):
                p.q = _as_f64(q_1d)
                log.info("Populated core_profiles.profiles_1d.q by interpolating from equilibrium q-profile.")
        except Exception as exc:
            log.warning("Failed to interpolate q-profile for core_profiles: %s", exc)

    # metadata
    try:
        cp.code.name = "NIMROD"
    except Exception:
        pass

    log.info(
        "core_profiles: wrote profiles_1d with %d points (psi_axis=%.6g, psi_lcfs=%.6g, tag=%s)",
        int(psi_pol_norm_1d.size), float(psi_axis), float(psi_lcfs), str(tag),
    )


def populate_edge_profiles(ep: Any, data: Dict[str, Any], t_index: int, args) -> None:
    """Populate edge_profiles.profiles_1d.

    Requirements (per your updated conventions):
      - Store 1D profiles as functions of normalized *poloidal* flux in edge_profiles:
            psi_pol_norm = (psi - psi_axis)/(psi_lcfs - psi_axis)
        so psi_pol_norm=0 at the magnetic axis and =1 at the LCFS.
      - Include SOL / PF regions (psi_pol_norm > 1) when data are present.
      - Do NOT force values outside LCFS to zero.
      - When schema allows, write grid.psi_norm / grid.rho_pol_norm directly. Otherwise, fall back to
        grid.rho_tor_norm for downstream plotting compatibility.
    """
    log = logging.getLogger(__name__)
    t = float(data.get("time", 0.0))

    psi2d = data.get("psi_eq", None)
    if psi2d is None:
        return

    pr2d = data.get("prq", None)
    pe2d = data.get("peq", None)
    nq = data.get("nq", None)
    te2d = data.get("teq", None)
    ti2d = data.get("tiq", None)
    jq = data.get("jq", None)

    # --- ensure 2D electron equilibrium fields (NIMROD dumps may store some scalars with an extra trailing dim)
    # psi_eq is expected to be 2D (R,Z). For equilibrium kinetic fields:
    #   - nq is often [R,Z,nspecies]
    #   - teq/peq/prq/tiq may appear as [R,Z,1] or [R,Z,ncomp]; we consistently take [:,:,0].
    eidx = int(getattr(args, "electrons_index", 0))
    def _slice0(a):
        if a is None:
            return None
        a = np.asarray(a)
        return a[:, :, 0] if getattr(a, "ndim", 0) == 3 else a

    def _slice_species(a, idx):
        if a is None:
            return None
        a = np.asarray(a)
        if getattr(a, "ndim", 0) == 3:
            ii = int(idx)
            if ii < 0 or ii >= int(a.shape[2]):
                ii = 0
            return a[:, :, ii]
        return a

    # electron density / temperature / pressure used by edge_profiles
    nq = _slice_species(nq, eidx)
    te2d = _slice0(te2d)
    pe2d = _slice0(pe2d)
    pr2d = _slice0(pr2d)
    ti2d = _slice0(ti2d)


    sp = getattr(args, "_nimrod_species", {}) or {}
    qe = float(sp.get("qe_c", 1.602176634e-19))

    psi_axis, psi_lcfs, tag = _choose_psi_axis_lcfs(data, args, log=log)
    if (not np.isfinite(psi_axis)) or (not np.isfinite(psi_lcfs)) or abs(psi_lcfs - psi_axis) < 1e-12:
        log.warning("edge_profiles: cannot determine psi_axis/psi_lcfs; skipping (tag=%s)", str(tag))
        return
    den = float(psi_lcfs - psi_axis)

    ps = np.asarray(psi2d, dtype=float)
    psi_pol_norm2d = (ps - float(psi_axis)) / den

    # If requested, source edge_profiles 1D profiles from TRANSP-style peqdsk (equilibrium).
    # This is important for cases where NIMROD teq/nq are not on the same separatrix definition as peqdsk.
    edge_mode = str(getattr(args, 'edge_ggd_values', '') or '').strip().lower()
    # NOTE: edge_profiles.profiles_1d are intended to remain equilibrium-like.
    # The --edge-ggd-values switch controls edge_profiles.ggd export only.
    # Therefore we always try equilibrium sources (preprocessed core_profiles and/or PEQDSK) first,
    # regardless of --edge-ggd-values, and fall back to dump-derived binning only if those are unavailable.
    if True:

        # Prefer preprocessed IMAS core_profiles (typically written by input2imas) as the equilibrium source.
        # This avoids depending on external peqdsk discovery and avoids mixing dump COCOS with input2imas COCOS.
        cp_pre = getattr(args, '_preproc_cp', None)
        if cp_pre is not None:
            prof_imas = _imas_core_profiles_te_ne(cp_pre, target_time=t, log=log)
        else:
            prof_imas = None

        if prof_imas is not None:
            ps1d, te_pe_ev, ne_pe_m3 = prof_imas
            # Extend preprocessed core_profiles beyond LCFS (psi_pol_norm>1) so edge_profiles can
            # represent SOL / edge regions in addition to the core. Prefer PEQDSK tail when available,
            # because it is consistent with the input2imas equilibrium source.
            try:
                ps1d = _as_f64(ps1d)
                te_pe_ev = _as_f64(te_pe_ev)
                ne_pe_m3 = _as_f64(ne_pe_m3)
            except Exception:
                ps1d = np.asarray(ps1d, dtype=float)
                te_pe_ev = np.asarray(te_pe_ev, dtype=float)
                ne_pe_m3 = np.asarray(ne_pe_m3, dtype=float)

            # Ensure sorted/unique
            try:
                ok = np.isfinite(ps1d) & np.isfinite(te_pe_ev) & np.isfinite(ne_pe_m3)
                ps1d = ps1d[ok]; te_pe_ev = te_pe_ev[ok]; ne_pe_m3 = ne_pe_m3[ok]
                o = np.argsort(ps1d)
                ps1d = ps1d[o]; te_pe_ev = te_pe_ev[o]; ne_pe_m3 = ne_pe_m3[o]
                # drop duplicates in psi (keep first)
                if ps1d.size > 1:
                    keep = np.ones(ps1d.size, dtype=bool)
                    keep[1:] = (np.diff(ps1d) > 0.0)
                    ps1d = ps1d[keep]; te_pe_ev = te_pe_ev[keep]; ne_pe_m3 = ne_pe_m3[keep]
            except Exception:
                pass

            psi_core_max = float(np.nanmax(ps1d)) if ps1d.size else 1.0
            psi_cap = getattr(args, 'edge_psi_norm_max', None)
            try:
                psi_cap = float(psi_cap) if psi_cap not in (None, '') else None
            except Exception:
                psi_cap = None
            # If user did not specify --edge-psi-norm-max, do not impose an artificial cap.
            # The SOL extent is determined from the available equilibrium sources:
            #   * PEQDSK tail (preferred), otherwise
            #   * maximum psi_pol_norm present on the stitched (R,Z) grid.
            # --- Primary extension: PEQDSK tail beyond the last core point ---
            peqdsk_path = _resolve_optional_file(args, 'peqdsk', 'peqdsk')
            prof_ext = _peqdsk_te_ne_si(peqdsk_path, log=log) if peqdsk_path else None
            if prof_ext is not None:
                ps_ext, te_ext, ne_ext = prof_ext
                ps_ext = _as_f64(ps_ext); te_ext = _as_f64(te_ext); ne_ext = _as_f64(ne_ext)
                # Select only SOL tail and cap extent
                eps = 1e-6
                m = (ps_ext > (psi_core_max + eps)) & np.isfinite(ps_ext) & np.isfinite(te_ext) & np.isfinite(ne_ext)
                if psi_cap is not None:
                    m &= (ps_ext <= (psi_cap + 1e-12))
                if np.any(m):
                    ps1d = np.concatenate([ps1d, ps_ext[m]])
                    te_pe_ev = np.concatenate([te_pe_ev, te_ext[m]])
                    ne_pe_m3 = np.concatenate([ne_pe_m3, ne_ext[m]])
                    o = np.argsort(ps1d)
                    ps1d = ps1d[o]; te_pe_ev = te_pe_ev[o]; ne_pe_m3 = ne_pe_m3[o]
                    log.info("edge_profiles(equilibrium): extended core_profiles with PEQDSK tail to psi_pol_norm=%.6g", float(np.nanmax(ps1d)))

            # --- Secondary extension: dump-derived binning (only if still not extended) ---
            # NOTE: do NOT attempt to construct Te in the SOL from p/n if teq is absent.
            # In no-impurity cases, nq can become extremely small outside the LCFS and p/n will
            # produce unphysical multi-keV spikes. If teq is not available, we fall back to a
            # smooth extrapolation anchored at the separatrix values from the equilibrium source.
            if (ps1d.size > 0) and (float(np.nanmax(ps1d)) <= 1.0001):
                # separatrix anchors from the equilibrium 1D profile
                try:
                    i_sep = int(np.nanargmin(np.abs(ps1d - 1.0)))
                except Exception:
                    i_sep = int(ps1d.size - 1) if ps1d.size else 0
                te_sep = float(te_pe_ev[i_sep]) if (te_pe_ev.size and np.isfinite(te_pe_ev[i_sep])) else float('nan')
                ne_sep = float(ne_pe_m3[i_sep]) if (ne_pe_m3.size and np.isfinite(ne_pe_m3[i_sep])) else float('nan')
                te_core_max = float(np.nanmax(te_pe_ev)) if te_pe_ev.size else float('nan')
            
                # Determine SOL extent from the stitched dump grid, but use a high-quantile
                # rather than the absolute max to avoid vacuum/outlier regions dominating.
                x_all = _as_f64(psi_pol_norm2d).ravel()
                m_all = np.isfinite(x_all) & (x_all > 1.0)
                if nq is not None:
                    ne_all = _as_f64(nq).ravel()
                    m_all &= np.isfinite(ne_all) & (ne_all > 0.0)
                if int(np.count_nonzero(m_all)) >= 100:
                    try:
                        psi_max_2d = float(np.nanquantile(x_all[m_all], 0.999))
                    except Exception:
                        psi_max_2d = float(np.nanmax(x_all[m_all]))
                else:
                    psi_max_2d = float('nan')
            
                psi_target_max = float('nan')
                if np.isfinite(psi_max_2d) and (psi_max_2d > 1.0):
                    psi_target_max = float(psi_max_2d if (psi_cap is None) else min(psi_max_2d, psi_cap))
            
                if np.isfinite(psi_target_max) and (psi_target_max > 1.0 + 1e-3):
                    # Estimate spacing from last part of the core profile
                    try:
                        d = np.diff(ps1d)
                        dpsi = float(np.nanmedian(d[-min(20, d.size):])) if d.size else 0.01
                    except Exception:
                        dpsi = 0.01
                    if (not np.isfinite(dpsi)) or dpsi <= 0.0:
                        dpsi = 0.01
            
                    ps_sol = np.arange(1.0 + dpsi, psi_target_max + 0.5 * dpsi, dpsi, dtype=float)
                    if ps_sol.size > 0:
                        # Prefer teq from the dump; do NOT synthesize Te from p/n in the SOL.
                        te_src = te2d if (te2d is not None) else None
                        ne_src = nq
            
                        te_out = np.full(ps_sol.shape, np.nan, dtype=float)
                        ne_out = np.full(ps_sol.shape, np.nan, dtype=float)
            
                        x = x_all
                        if (te_src is not None) and (ne_src is not None):
                            # Dump-derived SOL extension can be noisy if we mix divertor/private-flux regions.
                            # Prefer an outboard-midplane sample in the SOL, with a fallback to the full-domain
                            # sample if too few points are available.
                            te_flat = _as_f64(te_src).ravel()
                            ne_flat = _as_f64(ne_src).ravel()

                            base_m = np.isfinite(x) & np.isfinite(te_flat) & np.isfinite(ne_flat)
            
                            # Density floor to prevent pathological regions dominating bins
                            if np.isfinite(ne_sep) and ne_sep > 0.0:
                                ne_floor = max(1e16, 1e-2 * float(ne_sep))
                            else:
                                ne_floor = 1e16
                            base_m &= (ne_flat >= ne_floor)

                            # Outboard-midplane preference (avoids divertor/private-flux mixing)
                            mid_mask = None
                            try:
                                R2d = data.get('R', None)
                                Z2d = data.get('Z', None)
                                if (R2d is not None) and (Z2d is not None):
                                    Rf = _as_f64(R2d).ravel()
                                    Zf = _as_f64(Z2d).ravel()
                                    if (Rf.size == x.size) and (Zf.size == x.size):
                                        # axis estimate from minimum psi
                                        ps_loc = _as_f64(psi2d)
                                        k = int(np.nanargmin(ps_loc))
                                        r_axis = float(Rf[k]); z_axis = float(Zf[k])
                                        z_rng = float(np.nanmax(Zf) - np.nanmin(Zf))
                                        dz = max(1e-3, 0.02 * z_rng)
                                        mid_mask = (np.abs(Zf - z_axis) <= dz) & (Rf >= r_axis)
                            except Exception:
                                mid_mask = None
            
                            half = 0.5 * dpsi
                            eps_lcfs = 2.0e-3
                            min_pts = 128
            
                            for i, c in enumerate(ps_sol):
                                lo = max(c - half, 1.0 + eps_lcfs)
                                hi = c + half
                                mm0 = base_m & (x >= lo) & (x < hi)
                                mm = (mm0 & mid_mask) if (mid_mask is not None) else mm0
                                if (mid_mask is not None) and (mm.sum() < min_pts):
                                    # fallback: too few points in the midplane band
                                    mm = mm0
                                if mm.any():
                                    # Use a low-quantile for Te to reject any residual hot-core leakage
                                    try:
                                        te_v = float(np.nanquantile(te_flat[mm], 0.2))
                                    except Exception:
                                        te_v = float(np.nanmedian(te_flat[mm]))
                                    ne_v = float(np.nanmedian(ne_flat[mm]))
                                    # Hard sanity: SOL Te must not exceed core max
                                    if np.isfinite(te_core_max) and np.isfinite(te_v) and (te_v > te_core_max * 1.05):
                                        te_v = float('nan')
                                    te_out[i] = te_v
                                    ne_out[i] = ne_v
            
                        # Fill missing SOL bins by smooth extrapolation anchored at the separatrix
                        def _exp_tail(y_sep, psi_arr, y_floor=0.0):
                            if (not np.isfinite(y_sep)) or y_sep <= 0.0:
                                return np.full_like(psi_arr, np.nan, dtype=float)
                            lam = 0.05
                            try:
                                j = np.where((ps1d > 0.85) & (ps1d <= 1.0) & np.isfinite(te_pe_ev) & (te_pe_ev > 0.0))[0]
                                if j.size >= 6:
                                    jj = j[-6:]
                                    yy = np.log(np.maximum(te_pe_ev[jj], 1e-12))
                                    xx = ps1d[jj]
                                    s = np.polyfit(xx, yy, 1)[0]
                                    if np.isfinite(s) and s < 0.0:
                                        lam = float(min(0.2, max(0.01, -1.0 / s)))
                            except Exception:
                                pass
                            out = y_sep * np.exp(-(psi_arr - 1.0) / lam)
                            if y_floor > 0.0:
                                out = np.maximum(out, float(y_floor))
                            return out
            
                        te_ex = _exp_tail(te_sep, ps_sol, y_floor=0.5)
                        ne_ex = _exp_tail(ne_sep, ps_sol, y_floor=0.0)
            
                        bad_te = ~np.isfinite(te_out)
                        bad_ne = ~np.isfinite(ne_out)
                        if np.any(bad_te) and np.any(np.isfinite(te_ex)):
                            te_out[bad_te] = te_ex[bad_te]
                        if np.any(bad_ne) and np.any(np.isfinite(ne_ex)):
                            ne_out[bad_ne] = ne_ex[bad_ne]
            
                        # Enforce non-increasing SOL tails outward from the LCFS
                        if np.isfinite(te_sep):
                            prev = float(te_sep)
                            for i in range(te_out.size):
                                if np.isfinite(te_out[i]):
                                    te_out[i] = min(float(te_out[i]), prev)
                                    prev = float(te_out[i])
                        if np.isfinite(ne_sep):
                            prev = float(ne_sep)
                            for i in range(ne_out.size):
                                if np.isfinite(ne_out[i]):
                                    ne_out[i] = min(float(ne_out[i]), prev)
                                    prev = float(ne_out[i])
            
                        ok2 = np.isfinite(te_out) & np.isfinite(ne_out)
                        if np.any(ok2):
                            ps1d = np.concatenate([ps1d, ps_sol[ok2]])
                            te_pe_ev = np.concatenate([te_pe_ev, te_out[ok2]])
                            ne_pe_m3 = np.concatenate([ne_pe_m3, ne_out[ok2]])
                            o = np.argsort(ps1d)
                            ps1d = ps1d[o]; te_pe_ev = te_pe_ev[o]; ne_pe_m3 = ne_pe_m3[o]
                            log.info("edge_profiles(equilibrium): extended with SOL tail to psi_pol_norm=%.6g", float(np.nanmax(ps1d)))

            # For absolute psi (optional), prefer preprocessed equilibrium global_quantities when available.
            pa_abs = float(psi_axis)
            pb_abs = float(psi_lcfs)
            eq_pre = getattr(args, '_preproc_eq', None)
            if eq_pre is not None:
                try:
                    ts = getattr(eq_pre, 'time_slice', None)
                    if ts is not None and len(ts) > 0:
                        gq = getattr(ts[0], 'global_quantities', None)
                        if gq is not None:
                            pa2 = float(getattr(gq, 'psi_axis')) if hasattr(gq, 'psi_axis') else pa_abs
                            pb2 = float(getattr(gq, 'psi_boundary')) if hasattr(gq, 'psi_boundary') else pb_abs
                            if (abs(pb2 - pa2) > 1e-12) and (pa2 == pa2) and (pb2 == pb2):
                                pa_abs, pb_abs = pa2, pb2
                except Exception:
                    pass

            den_abs = float(pb_abs - pa_abs)
            if (not np.isfinite(den_abs)) or abs(den_abs) < 1e-12:
                den_abs = den

            # pressures from profiles
            pe1d = _as_f64(ne_pe_m3) * qe * _as_f64(te_pe_ev)

            # synthesize single-ion density for no-impurity cases using zeff_input (preferred)
            z_main = 1.0
            try:
                zlist = sp.get('z_ions', None) or []
                if zlist:
                    z_main = float(zlist[0])
            except Exception:
                z_main = 1.0
            zeff_input = sp.get('zeff_input', None)
            try:
                zeff_v = float(zeff_input) if zeff_input not in (None, '') else None
            except Exception:
                zeff_v = None
            divisor = float(zeff_v) if (zeff_v is not None and np.isfinite(zeff_v) and zeff_v > 0.0) else float(z_main if (np.isfinite(z_main) and z_main > 0.0) else 1.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                ni1d = _as_f64(ne_pe_m3) / divisor

            # grid coordinates from psinorm-like coordinate
            psi_pol_norm_1d = _as_f64(ps1d)
            psi_abs_1d = float(pa_abs) + psi_pol_norm_1d * den_abs

            # append profile entry
            idx = _append_time_edge_profiles(ep, t)
            p = ep.profiles_1d[idx]
            try:
                p.time = float(t)
            except Exception:
                pass

            # ---- grid coordinates ----
            if hasattr(p, 'grid'):
                g = p.grid
                if hasattr(g, 'psi'):
                    try:
                        g.psi = _as_f64(psi_abs_1d)
                    except Exception:
                        pass
                wrote_norm = False
                for nm in ('psi_norm', 'rho_pol_norm'):
                    if hasattr(g, nm):
                        try:
                            setattr(g, nm, _as_f64(psi_pol_norm_1d))
                            wrote_norm = True
                        except Exception:
                            pass
                if (not wrote_norm) and hasattr(g, 'rho_tor_norm'):
                    try:
                        g.rho_tor_norm = _as_f64(psi_pol_norm_1d)
                    except Exception:
                        pass

                # Compatibility: also mirror psi_pol_norm into rho_tor_norm when available,
                # because some plotting utilities only look at rho_tor_norm for the x-axis.
                if hasattr(g, 'rho_tor_norm'):
                    try:
                        g.rho_tor_norm = _as_f64(psi_pol_norm_1d)
                    except Exception:
                        pass

            # ---- populate electrons ----
            try:
                e = p.electrons
                if hasattr(e, 'density'):
                    e.density = _as_f64(ne_pe_m3)
                if hasattr(e, 'temperature'):
                    e.temperature = _as_f64(te_pe_ev)
                if hasattr(e, 'pressure'):
                    e.pressure = _as_f64(pe1d)
            except Exception:
                pass

            # ---- populate ions (single representative main ion) ----
            try:
                if hasattr(p, 'ion'):
                    ions = p.ion
                elif hasattr(p, 'ions'):
                    ions = p.ions
                else:
                    ions = None
                if ions is not None:
                    try:
                        ions.resize(1)
                        it = ions[0]
                    except Exception:
                        it = ions
                    if hasattr(it, 'density'):
                        it.density = _as_f64(ni1d)
            except Exception:
                pass

            # Done: do not fall back to peqdsk or dump-derived binning.
            return
        peqdsk_path = _resolve_optional_file(args, 'peqdsk', 'peqdsk')
        if peqdsk_path:
            prof = _peqdsk_te_ne_si(peqdsk_path, log=log)
        else:
            prof = None
        if prof is not None:
            ps1d, te_pe_ev, ne_pe_m3 = prof
            # pressures from peqdsk profiles
            pe1d = _as_f64(ne_pe_m3) * qe * _as_f64(te_pe_ev)

            # synthesize single-ion density for no-impurity cases using zeff_input (preferred)
            z_main = 1.0
            try:
                zlist = sp.get('z_ions', None) or []
                if zlist:
                    z_main = float(zlist[0])
            except Exception:
                z_main = 1.0
            zeff_input = sp.get('zeff_input', None)
            try:
                zeff_v = float(zeff_input) if zeff_input not in (None, '') else None
            except Exception:
                zeff_v = None
            divisor = float(zeff_v) if (zeff_v is not None and np.isfinite(zeff_v) and zeff_v > 0.0) else float(z_main if (np.isfinite(z_main) and z_main > 0.0) else 1.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                ni1d = _as_f64(ne_pe_m3) / divisor

            # grid coordinates from psinorm
            psi_pol_norm_1d = _as_f64(ps1d)
            psi_abs_1d = float(psi_axis) + psi_pol_norm_1d * den

            # append profile entry
            idx = _append_time_edge_profiles(ep, t)
            p = ep.profiles_1d[idx]
            try:
                p.time = float(t)
            except Exception:
                pass

            # ---- grid coordinates ----
            if hasattr(p, 'grid'):
                g = p.grid
                if hasattr(g, 'psi'):
                    try:
                        g.psi = _as_f64(psi_abs_1d)
                    except Exception:
                        pass
                wrote_norm = False
                for nm in ('psi_norm', 'rho_pol_norm'):
                    if hasattr(g, nm):
                        try:
                            setattr(g, nm, _as_f64(psi_pol_norm_1d))
                            wrote_norm = True
                        except Exception:
                            pass
                if (not wrote_norm) and hasattr(g, 'rho_tor_norm'):
                    # fallback: store psi_pol_norm into rho_tor_norm
                    try:
                        g.rho_tor_norm = _as_f64(psi_pol_norm_1d)
                    except Exception:
                        pass

            # ---- populate electrons ----
            try:
                e = p.electrons
                if hasattr(e, 'density'):
                    e.density = _as_f64(ne_pe_m3)
                if hasattr(e, 'temperature'):
                    e.temperature = _as_f64(te_pe_ev)
                if hasattr(e, 'pressure'):
                    e.pressure = _as_f64(pe1d)
            except Exception:
                pass

            # ---- populate ions (single representative main ion) ----
            try:
                if hasattr(p, 'ion'):
                    ions = p.ion
                elif hasattr(p, 'ions'):
                    ions = p.ions
                else:
                    ions = None
                if ions is not None:
                    try:
                        ions.resize(1)
                        it = ions[0]
                    except Exception:
                        it = ions
                    if hasattr(it, 'density'):
                        it.density = _as_f64(ni1d)
            except Exception:
                pass

            # Done: do not fall back to dump-derived binning.
            return
        else:
            log.warning("edge_profiles: equilibrium sources (preprocessed core_profiles / PEQDSK) unavailable; falling back to dump-derived profiles")


    # Base finite mask
    m = np.isfinite(psi_pol_norm2d)

    # Use density/pressure positivity to avoid deep vacuum (but keep SOL/PF)
    
    ne2d = None
    try:
        if nq is not None:
            _A = np.asarray(nq, dtype=float)
            if _A.ndim == 2:
                ne2d = _A
            elif _A.ndim >= 3 and _A.shape[-1] >= 1:
                ne2d = _A[..., 0]
    except Exception:
        ne2d = None

    if ne2d is not None:
        m &= np.isfinite(ne2d) & (ne2d > 0.0)
    elif pr2d is not None:
        pr = np.asarray(pr2d, dtype=float)
        m &= np.isfinite(pr) & (pr > 0.0)
    elif te2d is not None:
        te = np.asarray(te2d, dtype=float)
        m &= np.isfinite(te) & (te > 0.0)

    # Determine psi_norm max (include SOL/PF)
    xmax_user = getattr(args, "edge_psi_norm_max", None)
    xmax = None
    try:
        if xmax_user is not None:
            xmax = float(xmax_user)
    except Exception:
        xmax = None
    if xmax is None or (not np.isfinite(xmax)) or xmax <= 1.0:
        # Robust estimate from data: high quantile, capped to avoid numerical outliers
        q = float(getattr(args, "edge_psi_norm_quantile", 0.9995) or 0.9995)
        vv = psi_pol_norm2d[m]
        if vv.size > 10:
            try:
                xmax = float(np.quantile(vv, q))
            except Exception:
                xmax = float(np.nanmax(vv))
        else:
            xmax = 1.2
        # Ensure at least a small SOL extension if present
        xmax = float(max(1.0, xmax))
        xmax = float(min(2.5, xmax))

    # Final coordinate window
    m &= (psi_pol_norm2d >= 0.0) & (psi_pol_norm2d <= xmax + 1e-6)

    kept = int(np.count_nonzero(m))
    if kept < 50:
        log.warning("edge_profiles: too few valid points for 1D averaging (%d); skipping", kept)
        return

    nbins = int(getattr(args, "nbins", 256) or 256)
    edges = _make_bin_edges_from_data(psi_pol_norm2d[m], nbins, 0.0, xmax, log=log, tag="edge_profiles")
    psi_pol_norm_1d = 0.5 * (edges[:-1] + edges[1:])
    psi_abs_1d = float(psi_axis) + psi_pol_norm_1d * den

    # ---- 1D binned quantities ----
    p_tot1d = _bin_scalar_on_rho_bins(pr2d, psi_pol_norm2d, edges, m)
    pe1d = _bin_scalar_on_rho_bins(pe2d, psi_pol_norm2d, edges, m)
    ne1d = _bin_scalar_on_rho_bins(ne2d, psi_pol_norm2d, edges, m) if (ne2d is not None) else None

    ion_dens: List[Optional[np.ndarray]] = []
    if nq is not None and getattr(nq, "ndim", 0) >= 3 and nq.shape[-1] >= 2:
        for s in range(1, int(nq.shape[-1])):
            ion_dens.append(_bin_scalar_on_rho_bins(nq[..., s], psi_pol_norm2d, edges, m))

    # No explicit ion channels (single-ion/no-impurity dump): synthesize a single main-ion density
    # using constant zeff_input (preferred) or Z_main when available.
    if (not ion_dens) and (ne2d is not None):
        z_main = 1.0
        try:
            zlist = sp.get("z_ions", None) or []
            if zlist:
                z_main = float(zlist[0])
        except Exception:
            z_main = 1.0
        zeff_input = sp.get("zeff_input", None)
        try:
            zeff_v = float(zeff_input) if zeff_input not in (None, "") else None
        except Exception:
            zeff_v = None
        divisor = float(zeff_v) if (zeff_v is not None and np.isfinite(zeff_v) and zeff_v > 0.0) else float(z_main if (np.isfinite(z_main) and z_main > 0.0) else 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ni2d_eff = np.asarray(ne2d, dtype=float) / divisor
        ion_dens.append(_bin_scalar_on_rho_bins(ni2d_eff, psi_pol_norm2d, edges, m))

    te1d = _bin_scalar_on_rho_bins(te2d, psi_pol_norm2d, edges, m) if (te2d is not None) else None
    if te1d is None and pe1d is not None and ne1d is not None:
        te1d = _as_f64(pe1d) / (_as_f64(ne1d) * qe)
    if pe1d is None and te1d is not None and ne1d is not None:
        pe1d = _as_f64(ne1d) * qe * _as_f64(te1d)

    ti1d = _bin_scalar_on_rho_bins(ti2d, psi_pol_norm2d, edges, m) if (ti2d is not None) else None
    if ti1d is None and (p_tot1d is not None) and (pe1d is not None) and ion_dens:
        ni_tot = np.zeros_like(_as_f64(ion_dens[0]))
        for ni in ion_dens:
            if ni is not None:
                ni_tot = ni_tot + _as_f64(ni)
        pi1d = _as_f64(p_tot1d) - _as_f64(pe1d)
        good = np.isfinite(ni_tot) & (ni_tot > 0.0) & np.isfinite(pi1d)
        ti_tmp = np.full_like(pi1d, np.nan, dtype=float)
        ti_tmp[good] = pi1d[good] / (ni_tot[good] * qe)
        ti1d = ti_tmp

    pi1d = None
    if ti1d is not None and ion_dens:
        ni_tot = np.zeros_like(_as_f64(ti1d))
        for ni in ion_dens:
            if ni is not None:
                ni_tot = ni_tot + _as_f64(ni)
        pi1d = ni_tot * qe * _as_f64(ti1d)

    jtor1d = _bin_scalar_on_rho_bins(jq[..., 2], psi_pol_norm2d, edges, m) if (jq is not None) else None

    # Reduce empty-bin NaNs (but keep real SOL extensions)
    te1d = _fill_nan_1d(te1d)
    ne1d = _fill_nan_1d(ne1d)
    pe1d = _fill_nan_1d(pe1d)
    ti1d = _fill_nan_1d(ti1d)
    pi1d = _fill_nan_1d(pi1d)
    p_tot1d = _fill_nan_1d(p_tot1d)
    jtor1d = _fill_nan_1d(jtor1d)
    ion_dens = [_fill_nan_1d(x) for x in ion_dens]

    # append profile entry
    idx = _append_time_edge_profiles(ep, t)
    p = ep.profiles_1d[idx]
    try:
        p.time = float(t)
    except Exception:
        pass

    # ---- grid coordinates ----
    if hasattr(p, "grid"):
        g = p.grid
        if hasattr(g, "psi"):
            try:
                g.psi = _as_f64(psi_abs_1d)  # Wb
            except Exception:
                pass
        # Preferred: poloidal normalized flux coordinate
        wrote_norm = False
        for nm in ("psi_norm", "rho_pol_norm"):
            if hasattr(g, nm):
                try:
                    setattr(g, nm, _as_f64(psi_pol_norm_1d))
                    wrote_norm = True
                except Exception:
                    pass
        # Fallback: store into rho_tor_norm for plotting tools that assume a normalized coordinate exists
        if (not wrote_norm) and hasattr(g, "rho_tor_norm"):
            try:
                g.rho_tor_norm = _as_f64(psi_pol_norm_1d)
            except Exception:
                pass

    # ---- populate electrons ----
    try:
        e = p.electrons
        if ne1d is not None and hasattr(e, "density"):
            e.density = _as_f64(ne1d)
        if te1d is not None and hasattr(e, "temperature"):
            e.temperature = _as_f64(te1d)
        if pe1d is not None and hasattr(e, "pressure"):
            e.pressure = _as_f64(pe1d)
    except Exception:
        pass

    # ---- populate ions ----
    try:
        nion = len(ion_dens)
        if hasattr(p, "ion") and nion > 0:
            p.ion.resize(nion)
            z_list = sp.get("z_ion", None)
            for i in range(nion):
                ion = p.ion[i]
                if ion_dens[i] is not None and hasattr(ion, "density"):
                    ion.density = _as_f64(ion_dens[i])
                if ti1d is not None and hasattr(ion, "temperature"):
                    ion.temperature = _as_f64(ti1d)
                if pi1d is not None and hasattr(ion, "pressure"):
                    ion.pressure = _as_f64(pi1d)
                if hasattr(ion, "z_ion"):
                    try:
                        if isinstance(z_list, (list, tuple)) and i < len(z_list):
                            ion.z_ion = float(z_list[i])
                        else:
                            ion.z_ion = 1.0
                    except Exception:
                        pass
    except Exception:
        pass

    # Current profile (best-effort)
    try:
        if jtor1d is not None and hasattr(p, "j_phi"):
            p.j_phi = _as_f64(jtor1d)
    except Exception:
        pass

    # metadata
    try:
        ep.code.name = "NIMROD"
    except Exception:
        pass

    log.info(
        "edge_profiles: wrote profiles_1d with %d points (psi_axis=%.6g, psi_lcfs=%.6g, xmax=%.3g, tag=%s)",
        int(psi_pol_norm_1d.size), float(psi_axis), float(psi_lcfs), float(xmax), str(tag),
    )


def _ggd_should_write_grid(args: Any) -> bool:
    """Return True if this call should (re)write grid_ggd for the current slice.

    When --ggd-reuse-grid is enabled, we write grid_ggd only for the first processed dump
    and reuse it for all subsequent time slices. The per-file decision is communicated
    via args._ggd_write_grid (set in main()).
    """
    if not bool(getattr(args, "ggd_reuse_grid", False)):
        return True
    return bool(getattr(args, "_ggd_write_grid", False))


def _append_time_ggd(mhd: Any, t: float, *, write_grid: bool = True, reuse_grid: bool = False) -> tuple[int, int]:
    """Append a new time slice to `mhd.ggd` and (optionally) `mhd.grid_ggd`.

    Returns:
        itime: 0-based index of the newly appended `ggd` slice
        igrid: 0-based index of the `grid_ggd` entry that should be referenced by this slice

    Reuse semantics:
        If reuse_grid=True and write_grid=False, `grid_ggd` is *not* extended; instead, the
        first grid entry (igrid=0) is reused for all subsequent slices. This is intended to
        prevent pathological output growth when the grid/connectivity are time-invariant.
    """
    # Always extend the values AoS
    cur = _aos_len(getattr(mhd, "ggd"))
    mhd.ggd.resize(cur + 1)

    # Decide grid policy
    if write_grid:
        mhd.grid_ggd.resize(cur + 1)
        igrid = cur
    else:
        # Ensure at least one grid exists if we intend to reuse it
        try:
            ng = _aos_len(getattr(mhd, "grid_ggd"))
        except Exception:
            ng = 0
        if ng < 1:
            try:
                mhd.grid_ggd.resize(1)
            except Exception:
                pass
        igrid = 0

    def _set_time(aos, idx: int, val: float, label: str) -> None:
        # Preferred: scalar leaf on the AoS element (DD: FLT_0D).
        try:
            aos[idx].time = float(val)
            return
        except Exception:
            # Fallback: coordinate vector on the AoS container (rare, but seen in some wrappers).
            try:
                arr = getattr(aos, "time")
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

    # Only set a grid time when we actually extended/wrote grid_ggd. For reuse mode,
    # keep the existing grid time (typically the first slice time).
    if write_grid:
        _set_time(mhd.grid_ggd, cur, t, "mhd.grid_ggd")
    elif reuse_grid:
        # Best-effort: ensure grid_ggd[0].time exists (set once if missing).
        try:
            _ = mhd.grid_ggd[0].time
        except Exception:
            try:
                _set_time(mhd.grid_ggd, 0, t, "mhd.grid_ggd")
            except Exception:
                pass

    # Optional: keep top-level `mhd.time` consistent when present.
    try:
        times = getattr(mhd, "time", None)
        if times is not None:
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

    return cur, igrid

def _fe_tri_connectivity_from_mask(mask: np.ndarray) -> np.ndarray:
    """Build 1-based triangle connectivity (ntri,3) from a rectangular (nr,nz) node lattice mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be 2D (nr,nz)")
    nr, nz = mask.shape
    if nr < 2 or nz < 2:
        return np.zeros((0, 3), dtype=np.int32)

    idx = (np.arange(nr * nz, dtype=np.int64).reshape((nr, nz), order="F") + 1)
    a = idx[:-1, :-1]
    b = idx[1:, :-1]
    c = idx[1:, 1:]
    d = idx[:-1, 1:]

    cell_ok = mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]
    if not np.any(cell_ok):
        return np.zeros((0, 3), dtype=np.int32)

    tri1 = np.stack([a[cell_ok], b[cell_ok], c[cell_ok]], axis=1)
    tri2 = np.stack([a[cell_ok], c[cell_ok], d[cell_ok]], axis=1)
    return np.vstack([tri1, tri2]).astype(np.int32, copy=False)


def _replicate_tri_connectivity_per_phi(conn2d: np.ndarray, nn2d: int, nphi: int) -> np.ndarray:
    """Replicate 2D triangle connectivity per toroidal plane by node offset."""
    conn2d = np.asarray(conn2d, dtype=np.int32)
    if nphi <= 1 or conn2d.size == 0:
        return conn2d
    offsets = (np.arange(int(nphi), dtype=np.int64) * int(nn2d)).reshape((-1, 1, 1))
    conn3 = (conn2d.reshape((1, -1, 3)).astype(np.int64) + offsets).reshape((-1, 3))
    return conn3.astype(np.int32, copy=False)


def _gridggd_write_node_vectors(g: Any, r_nodes: np.ndarray, z_nodes: np.ndarray, phi_nodes: np.ndarray) -> None:
    """Write per-node coordinate vectors into grid_ggd.space (standard IMAS location)."""
    r_nodes = np.asarray(r_nodes, dtype=float).reshape(-1, 1)
    z_nodes = np.asarray(z_nodes, dtype=float).reshape(-1, 1)
    phi_nodes = np.asarray(phi_nodes, dtype=float).reshape(-1, 1)

    try:
        g.space.resize(3)
    except Exception:
        pass

    def _fill(space_obj, coord_name: str, vec: np.ndarray) -> None:
        try:
            space_obj.geometry_type.index = 0
            space_obj.geometry_type.name = "standard"
            space_obj.geometry_type.description = "standard"
        except Exception:
            pass
        try:
            space_obj.coordinates_type.resize(1)
            _cn = str(coord_name).strip().lower()
            space_obj.coordinates_type[0].name = _cn
            space_obj.coordinates_type[0].index = {'r': 4, 'z': 3, 'phi': 5}.get(_cn, -1)
            space_obj.coordinates_type[0].description = _cn
        except Exception:
            pass
        try:
            space_obj.objects_per_dimension.resize(1)
            opd = space_obj.objects_per_dimension[0]
            try:
                opd.geometry_content.name = "coordinate"
                opd.geometry_content.index = -1
                opd.geometry_content.description = "Coordinate vector"
            except Exception:
                pass
            opd.object.resize(1)
            opd.object[0].geometry = vec
        except Exception:
            pass

    _fill(g.space[0], "R", r_nodes)
    _fill(g.space[1], "Z", z_nodes)
    _fill(g.space[2], "phi", phi_nodes)




def _gridggd_write_unstructured_grid_subset_imas(
    g: Any,
    nodes_xyz: np.ndarray,
    connectivity: Optional[np.ndarray],
    *,
    log: Optional[logging.Logger] = None,
) -> None:
    """Populate grid_ggd.grid_subset connectivity using IMAS-Python objects.

    Slow but DD-aware; use for --writer=imas or non-HDF5 backends.
    """
    if nodes_xyz is None:
        raise ValueError("nodes_xyz is None")
    nodes_xyz = np.asarray(nodes_xyz, dtype=float)
    if nodes_xyz.ndim != 2 or nodes_xyz.shape[1] != 3:
        raise ValueError(f"nodes_xyz must have shape (N,3); got {nodes_xyz.shape}")
    n_nodes = int(nodes_xyz.shape[0])

    if connectivity is None:
        n_subsets = 1
        n_cells = 0
        n_verts = 0
    else:
        connectivity = np.asarray(connectivity, dtype=np.int32)
        if connectivity.ndim != 2:
            raise ValueError(f"connectivity must be 2D; got {connectivity.shape}")
        n_cells = int(connectivity.shape[0])
        n_verts = int(connectivity.shape[1]) if n_cells > 0 else 0
        n_subsets = 2

    try:
        g.grid_subset.resize(n_subsets)
    except Exception:
        pass

    # nodes subset
    s0 = g.grid_subset[0]
    try:
        s0.dimension = 1
    except Exception:
        pass
    try:
        s0.identifier.name = "nodes"
        s0.identifier.index = 1
        s0.identifier.description = "Unstructured nodes"
    except Exception:
        pass

    try:
        s0.element.resize(n_nodes)
    except Exception as e:
        raise RuntimeError(f"Failed to resize nodes element to {n_nodes}: {e}") from e

    for i in range(n_nodes):
        el = s0.element[i]
        try:
            el.object.resize(3)
        except Exception:
            pass
        for j in range(3):
            obj = el.object[j]
            val = float(nodes_xyz[i, j])
            try:
                obj.real = val
                continue
            except Exception:
                pass
            try:
                obj.geometry = val
                continue
            except Exception:
                pass
            try:
                setattr(obj, "real", np.float64(val))
            except Exception:
                pass

    if log:
        log.info("IMAS grid_ggd: wrote nodes subset (N=%d)", n_nodes)

    if n_subsets < 2:
        return

    # connectivity subset
    s1 = g.grid_subset[1]
    try:
        s1.dimension = 4
    except Exception:
        pass
    try:
        s1.identifier.name = "volumes"
        s1.identifier.index = 43
        s1.identifier.description = "Unstructured connectivity"
    except Exception:
        pass

    try:
        s1.base.resize(1)
        s1.base[0].index = 0
        s1.base[0].grid_subset_index = 1
    except Exception:
        pass

    try:
        s1.element.resize(n_cells)
    except Exception as e:
        raise RuntimeError(f"Failed to resize connectivity element to {n_cells}: {e}") from e

    for i in range(n_cells):
        el = s1.element[i]
        try:
            el.object.resize(n_verts)
        except Exception:
            pass
        for j in range(n_verts):
            obj = el.object[j]
            idx = int(connectivity[i, j])
            try:
                obj.index = idx
                continue
            except Exception:
                pass
            try:
                setattr(obj, "index", np.int32(idx))
            except Exception:
                pass

    if log:
        log.info("IMAS grid_ggd: wrote connectivity subset (Nc=%d, Nv=%d)", n_cells, n_verts)



def _build_fe_tri_nodes_conn(R2d: "np.ndarray", Z2d: "np.ndarray", nphi: int, phi_list: "np.ndarray") -> tuple["np.ndarray","np.ndarray"]:
    """
    Build node coordinates and triangle connectivity for the stitched (R,Z) lattice.
    - Nodes: full rectangular lattice (nr*nz*nphi), including NaN nodes (kept to preserve indexing).
    - Connectivity: two triangles per valid quad cell, replicated per phi plane; indices are 0-based here.
    """
    import numpy as np
    R2d = np.asarray(R2d, dtype=float)
    Z2d = np.asarray(Z2d, dtype=float)
    if R2d.shape != Z2d.shape or R2d.ndim != 2:
        raise ValueError(f"R2d and Z2d must be same 2D shape; got {R2d.shape} vs {Z2d.shape}")
    ny, nx = R2d.shape  # NIMROD internal often (Ny,Nx)
    nr, nz = nx, ny     # we treat i=R index (fast) and j=Z index (slow) via transpose below

    # Node ordering: i fastest, then j, then k (phi), consistent with node(i,j,k)=k*(nr*nz)+j*nr+i.
    R = R2d.T.reshape(-1)  # (nr*nz)
    Z = Z2d.T.reshape(-1)  # (nr*nz)
    base = np.stack([R, Z], axis=1)  # (nr*nz,2)

    # Replicate in phi
    nphi = int(max(1, nphi))
    phi_list = np.asarray(phi_list, dtype=float)
    if phi_list.size != nphi:
        raise ValueError("phi_list size mismatch")
    nodes = np.empty((nr*nz*nphi, 3), dtype=np.float64)
    for k,phi in enumerate(phi_list):
        sl = slice(k*nr*nz, (k+1)*nr*nz)
        nodes[sl,0] = base[:,0]
        nodes[sl,1] = base[:,1]
        nodes[sl,2] = phi

    # Valid cell mask: all four corners finite.
    Rm = R2d.T  # (nr,nz)
    Zm = Z2d.T
    finite = np.isfinite(Rm) & np.isfinite(Zm)
    cell_ok = finite[:-1,:-1] & finite[1:,:-1] & finite[1:,1:] & finite[:-1,1:]

    ii, jj = np.nonzero(cell_ok)  # arrays of length ncell
    # Corner node ids (0-based) within one phi plane
    a = jj*nr + ii
    b = jj*nr + (ii+1)
    c = (jj+1)*nr + (ii+1)
    d = (jj+1)*nr + ii

    # Two tris per cell: (a,b,c) and (a,c,d)
    tri0 = np.stack([a,b,c], axis=1)
    tri1 = np.stack([a,c,d], axis=1)
    tri_plane = np.vstack([tri0, tri1]).astype(np.int32, copy=False)  # (2*ncell,3)

    # Replicate per phi plane with offset
    if nphi == 1:
        tri = tri_plane
    else:
        tri = np.empty((tri_plane.shape[0]*nphi, 3), dtype=np.int32)
        for k in range(nphi):
            off = k*(nr*nz)
            tri[k*tri_plane.shape[0]:(k+1)*tri_plane.shape[0], :] = tri_plane + off

    return nodes, tri


def _build_fe_wedge_nodes_conn(
    R2d: "np.ndarray",
    Z2d: "np.ndarray",
    nphi: int,
    phi_list: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray"]:
    """Build node coordinates and wedge (triangular-prism) connectivity.

    This follows the IMAS4NIMROD convention (Fig. 3c / Eq. (1)):
      - Node indexing is a tensor product of poloidal node index i and toroidal plane index k:
            node(i,k) = i + k*N2D
      - Each wedge cell extrudes one 2D triangle between planes (k, k+1) (periodic in phi),
        with local ordering:
            (a,b,c,a',b',c')
        where (a,b,c) are the triangle nodes on plane k and (a',b',c') are the corresponding
        nodes on plane k+1.

    Returns
    -------
    nodes : float64, shape (N2D*nphi, 3)
        (R,Z,phi) node coordinates.
    wedge : int32, shape (Ntri*nphi, 6)
        0-based node indices for wedge cells.
    """
    import numpy as np

    R2d = np.asarray(R2d, dtype=float)
    Z2d = np.asarray(Z2d, dtype=float)
    if R2d.shape != Z2d.shape or R2d.ndim != 2:
        raise ValueError(f"R2d and Z2d must be same 2D shape; got {R2d.shape} vs {Z2d.shape}")

    # Reuse the same node construction as fe_tri.
    nphi = int(max(1, nphi))
    phi_list = np.asarray(phi_list, dtype=float)
    if phi_list.size != nphi:
        raise ValueError("phi_list size mismatch")

    nodes, _tri_rep = _build_fe_tri_nodes_conn(R2d, Z2d, nphi, phi_list)

    # Build 2D triangle connectivity for one plane (0-based), consistent with _build_fe_tri_nodes_conn.
    ny, nx = R2d.shape
    nr, nz = nx, ny
    Rm = R2d.T  # (nr,nz)
    Zm = Z2d.T
    finite = np.isfinite(Rm) & np.isfinite(Zm)
    cell_ok = finite[:-1, :-1] & finite[1:, :-1] & finite[1:, 1:] & finite[:-1, 1:]
    ii, jj = np.nonzero(cell_ok)
    a = jj * nr + ii
    b = jj * nr + (ii + 1)
    c = (jj + 1) * nr + (ii + 1)
    d = (jj + 1) * nr + ii
    tri0 = np.stack([a, b, c], axis=1)
    tri1 = np.stack([a, c, d], axis=1)
    tri_plane = np.vstack([tri0, tri1]).astype(np.int32, copy=False)  # (ntri,3)

    nn2d = int(nr * nz)
    ntri = int(tri_plane.shape[0])
    if ntri == 0 or nphi < 2:
        # With nphi==1 there is no volumetric extrusion.
        return nodes, np.zeros((0, 6), dtype=np.int32)

    wedge = np.empty((ntri * nphi, 6), dtype=np.int32)
    for k in range(nphi):
        kp = (k + 1) % nphi
        off0 = k * nn2d
        off1 = kp * nn2d
        sl = slice(k * ntri, (k + 1) * ntri)
        wedge[sl, 0:3] = tri_plane + off0
        wedge[sl, 3:6] = tri_plane + off1

    return nodes, wedge


def _gridggd_write_tri_connectivity(g: Any, tri_conn: np.ndarray) -> None:
    """Write a minimal IMAS-compliant 2D unstructured connectivity (triangles).

    Defines two subsets:
      - nodes: identifier.index=1, dimension=1
      - cells: identifier.index=5, dimension=3 (2D faces / cells)

    Each cell element references three node indices (0-based).
    """
    import numpy as _np
    tri_conn = _np.asarray(tri_conn)
    if tri_conn.ndim != 2 or tri_conn.shape[1] != 3:
        raise ValueError(f"tri_conn must be (ntri,3); got {tri_conn.shape}")
    ntri = int(tri_conn.shape[0])

    # Subsets
    try:
        g.grid_subset.resize(2)
    except Exception:
        pass

    s_nodes = g.grid_subset[0]
    try:
        s_nodes.identifier.name = "nodes"
        s_nodes.identifier.index = 1
        s_nodes.identifier.description = "Unstructured nodes"
        s_nodes.dimension = 1
    except Exception:
        pass

    s_cells = g.grid_subset[1]
    try:
        s_cells.identifier.name = "cells"
        s_cells.identifier.index = 5
        s_cells.identifier.description = "Triangulated 2D cells"
        s_cells.dimension = 3
    except Exception:
        pass

    # Elements
    try:
        s_cells.element.resize(ntri)
    except Exception:
        return

    for i in range(ntri):
        e = s_cells.element[i]
        try:
            e.object.resize(3)
        except Exception:
            pass
        for k in range(3):
            o = e.object[k]
            # node index (0-based)
            try:
                o.index = int(tri_conn[i, k]) - 1
            except Exception:
                pass
            try:
                o.dimension = 1
            except Exception:
                pass
            try:
                o.space = 1
            except Exception:
                pass
def populate_mhd_ggd(mhd: Any, data: Dict[str, Any], args, species_index: int | None = None) -> None:
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

    conn_kind = str(getattr(args, "ggd_connectivity", "none") or "none").strip().lower()

    use_fe_nodes = (

        bool(getattr(args, "ggd_unstructured", False))

        and bool(getattr(args, "ggd_unstructured_fe_nodes", False))

        and conn_kind in ("fe_tri", "fe_wedge", "fe_pointcloud")

    )

    if use_fe_nodes:
        write_grid = _ggd_should_write_grid(args)
        it, ig = _append_time_ggd(mhd, t, write_grid=write_grid, reuse_grid=bool(getattr(args, "ggd_reuse_grid", False)))
        g = mhd.grid_ggd[ig]
        gidx = int(ig + 1)
        try:
            g.identifier.name = "nimrod_fe_rzphi_nodes_tri"
            g.identifier.index = int(ig + 1)
            g.identifier.description = "Native stitched NIMROD FE nodes (R,Z) replicated in phi; triangulated 2D connectivity per plane"
        except Exception:
            pass

        # Ensure IMAS HDF5 backend creates the nested packed datasets for unstructured grid_ggd.
        # Without at least one grid_subset/element placeholder, the backend may omit
        # grid_ggd[]&grid_subset[]&AOS_SHAPE (and friends), and the packed writer cannot proceed.
        try:
            # Coordinate axes (R, Z, Phi)
            g.space.resize(3)
            for ii, nm in enumerate(["R", "Z", "Phi"]):
                try:
                    g.space[ii].identifier.name = nm
                    g.space[ii].identifier.index = -1
                    g.space[ii].identifier.description = nm
                except Exception:
                    pass
        except Exception:
            pass

        try:
            # Two subsets: nodes and volumes/connectivity
            g.grid_subset.resize(2)

            s0 = g.grid_subset[0]
            try:
                s0.dimension = 1
                s0.identifier.name = "nodes"
                s0.identifier.index = 1
                s0.identifier.description = "Unstructured nodes"
            except Exception:
                pass
            try:
                s0.element.resize(1)
                try:
                    s0.element[0].object.resize(0)
                except Exception:
                    pass
            except Exception:
                pass

            s1 = g.grid_subset[1]
            try:
                s1.dimension = 4
                s1.identifier.name = "volumes"
                s1.identifier.index = 43
                s1.identifier.description = "Unstructured connectivity"
            except Exception:
                pass
            try:
                s1.base.resize(1)
                s1.base[0].index = 0
                s1.base[0].grid_subset_index = 1
            except Exception:
                pass
            try:
                s1.element.resize(1)
                try:
                    s1.element[0].object.resize(0)
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

        if conn_kind == "fe_pointcloud":
            try:
                g.grid_subset.resize(1)
            except Exception:
                pass

        Rloc = R
        Zloc = Z
        ref = None
        for _k in ("teq", "peq", "prq"):
            if data.get(_k) is not None:
                ref = np.asarray(data.get(_k))
                break
        if ref is not None and ref.shape != Rloc.shape:
            if ref.T.shape == Rloc.shape:
                ref = ref.T
            elif Rloc.T.shape == ref.shape:
                Rloc = Rloc.T
                Zloc = Zloc.T

        r2d = np.asarray(Rloc, dtype=float).ravel(order="F")
        z2d = np.asarray(Zloc, dtype=float).ravel(order="F")
        nn2d = int(r2d.size)
        r_nodes = np.tile(r2d, int(nphi))
        z_nodes = np.tile(z2d, int(nphi))
        phi_nodes = np.repeat(phi_list.astype(float), nn2d)

        if write_grid:
            _gridggd_write_node_vectors(g, r_nodes, z_nodes, phi_nodes)

        # Connectivity (optional).
        if conn_kind != "fe_pointcloud":
            if write_grid and _use_imas_connectivity_writer(args):
                try:
                    nodes_xyz, connectivity, _meta = _build_unstructured_nodes_connectivity(data, args)
                    _gridggd_write_unstructured_grid_subset_imas(g, nodes_xyz, connectivity, log=None)
                except Exception:
                    # Non-fatal; downstream tools may still use the space geometry vectors.
                    pass
            else:
                pass  # connectivity populated later via packed HDF5 writer (h5py)

        def _recon_native(eq_key: str, re_key: str, im_key: str):
            eq = data.get(eq_key, None)
            # If the equilibrium 2D field is not explicitly present in the dump (e.g. teq/tiq absent),
            # derive a consistent equilibrium baseline from the separate equilibrium entries (peq/prq and nq)
            # so that "full" reconstruction includes equilibrium + n=0..N Fourier content.
            if eq is None:
                try:
                    qe = 1.602176634e-19
                    if eq_key == "teq":
                        peq = data.get("peq", None)
                        nq = data.get("nq", None)
                        if peq is not None and nq is not None:
                            nqA = np.asarray(nq, dtype=float)
                            ne2d = nqA[..., 0] if nqA.ndim >= 3 else nqA
                            eq = np.asarray(peq, dtype=float) / (np.asarray(ne2d, dtype=float) * qe)
                    elif eq_key == "tiq":
                        prq = data.get("prq", None)
                        peq = data.get("peq", None)
                        nq = data.get("nq", None)
                        if prq is not None and peq is not None and nq is not None:
                            nqA = np.asarray(nq, dtype=float)
                            if nqA.ndim >= 3 and nqA.shape[-1] >= 2:
                                ni2d = np.nansum(np.asarray(nqA[..., 1:], dtype=float), axis=2)
                                pi2d = np.asarray(prq, dtype=float) - np.asarray(peq, dtype=float)
                                eq = np.asarray(pi2d, dtype=float) / (np.asarray(ni2d, dtype=float) * qe)
                except Exception:
                    eq = None

            reA = fields.get(re_key, None)
            imA = fields.get(im_key, None)
            if eq is None and (reA is None or imA is None):
                return None
            vals_phi = []
            # Optional scaling of the spectral contribution (debug/visualization):
            # use --edge-pert-scale for both mhd (always full) and edge_profiles(full via mirroring).
            try:
                _ps = float(getattr(args, "pert_scale", 1.0) or 1.0)
            except Exception:
                _ps = 1.0
            for phi in phi_list:
                full = _reconstruct_full_from_modes(eq, reA, imA, keff, float(phi), pert_scale=_ps)
                if full is None:
                    continue
                if full.shape != Rloc.shape and full.T.shape == Rloc.shape:
                    full = full.T
                vals_phi.append(np.asarray(full, dtype=float))
            if not vals_phi:
                return None
            return np.stack(vals_phi, axis=2)

        def _write_node_scalar(container: Any, V3: np.ndarray | None) -> None:
            if V3 is None:
                return
            try:
                container.resize(1)
                qt = container[0]
            except Exception:
                qt = container
            try:
                qt.grid_index = int(ig + 1)
                qt.grid_subset_index = 1
            except Exception:
                pass
            nr_, nz_, nphi_ = V3.shape
            shp = np.asarray([nr_, nz_, nphi_], dtype=np.int32)
            for attr in ("values_shape", "valuesShape", "values_SHAPE"):
                try:
                    leaf = getattr(qt, attr)
                except Exception:
                    leaf = None
                if leaf is None:
                    continue
                try:
                    try:
                        leaf.resize(3)
                        leaf[:] = shp
                    except Exception:
                        setattr(qt, attr, shp)
                    break
                except Exception:
                    continue
            try:
                qt.values = np.asarray(V3, dtype=float)
            except Exception:
                try:
                    qt.values = np.asarray(V3, dtype=float).ravel(order="F")
                except Exception:
                    pass

        q = mhd.ggd[it]

        try:
            Te3 = _recon_native("teq", "rete", "imte")
            _write_node_scalar(q.electrons.temperature, Te3)
        except Exception:
            pass

        # --- Additional GGD quantities (node-centered) ---
        # Helper to slice mode arrays that may carry species and/or vector component dimensions.

        def _slice_modes(A, spec=None, comp=None):
            """Extract a scalar (R,Z,nmodes) coefficient array from heterogeneous layouts.

            Supported layouts:
              - scalar modes: (R,Z,nmodes)
              - density modes: (R,Z,nspec,nmodes) or (R,Z,nmodes,nspec)
              - packed density modes: (R,Z,nspec*nmodes)
              - vector modes: (R,Z,nmodes,3) or (R,Z,3,nmodes)
              - packed vector modes: (R,Z,3*nmodes)

            Returns None if the requested slice cannot be interpreted.
            """
            if A is None:
                return None
            import numpy as _np
            nm = int(len(keff))
            AA = _np.asarray(A)

            # --- vector component extraction ---
            if comp is not None:
                # Unpacked 4D
                if AA.ndim == 4:
                    if AA.shape[-1] == 3 and AA.shape[-2] == nm:
                        # (R,Z,nmodes,3)
                        return AA[:, :, :, int(comp)]
                    if AA.shape[2] == 3 and AA.shape[3] == nm:
                        # (R,Z,3,nmodes)
                        return AA[:, :, int(comp), :]
                # Packed 3D
                if AA.ndim == 3 and AA.shape[-1] == 3 * nm:
                    c = int(comp)
                    if c < 0 or c > 2:
                        return None
                    return AA[:, :, c * nm : (c + 1) * nm]
                return None

            # --- density/species extraction ---
            if spec is not None:
                s = int(spec)
                if AA.ndim == 4:
                    if AA.shape[-1] == nm:
                        # (R,Z,nspec,nmodes)
                        if s < 0 or s >= AA.shape[2]:
                            return None
                        return AA[:, :, s, :]
                    if AA.shape[2] == nm:
                        # (R,Z,nmodes,nspec)
                        if s < 0 or s >= AA.shape[3]:
                            return None
                        return AA[:, :, :, s]
                if AA.ndim == 3 and AA.shape[-1] % nm == 0 and AA.shape[-1] != nm:
                    try:
                        order = getattr(args, 'dens_pert_order', 'species_major')
                    except Exception:
                        order = 'species_major'
                    try:
                        unpacked, ns = _unpack_density_modes(AA, nm, None, order)
                    except Exception:
                        return None
                    if s < 0 or s >= unpacked.shape[2]:
                        return None
                    return unpacked[:, :, s, :]
                return None

            # scalar modes
            if AA.ndim == 3 and AA.shape[-1] == nm:
                return AA
            return None

        def _recon_from(eq2d, reA, imA, spec=None, comp=None):
            if eq2d is None and (reA is None or imA is None):
                return None
            reS = _slice_modes(reA, spec=spec, comp=comp)
            imS = _slice_modes(imA, spec=spec, comp=comp)
            vals_phi = []
            try:
                _ps = float(getattr(args, 'pert_scale', 1.0) or 1.0)
            except Exception:
                _ps = 1.0
            for phi in phi_list:
                full = _reconstruct_full_from_modes(eq2d, reS, imS, keff, float(phi), pert_scale=_ps)
                if full is None:
                    continue
                if full.shape != Rloc.shape and getattr(full, 'T', None) is not None and full.T.shape == Rloc.shape:
                    full = full.T
                vals_phi.append(np.asarray(full, dtype=float))
            if not vals_phi:
                return None
            return np.stack(vals_phi, axis=2)

        # Species metadata (incl. qe and zeff_input)
        sp = getattr(args, '_nimrod_species', {}) or {}
        qe = float(sp.get('qe_c', 1.602176634e-19))
        n_scale = float(getattr(args, "n_scale", 1.0) or 1.0)
        zeff_input = sp.get('zeff_input', None)

        # Electron density (species 0)
        ne_eq = None
        try:
            nqA = np.asarray(data.get('nq'), dtype=float) if data.get('nq') is not None else None
            if nqA is not None and nqA.ndim >= 3:
                ne_eq = nqA[:, :, 0]
                if ne_eq.shape != Rloc.shape and ne_eq.T.shape == Rloc.shape:
                    ne_eq = ne_eq.T
        except Exception:
            ne_eq = None
        ne3 = _recon_from(ne_eq, fields.get('rend', None), fields.get('imnd', None), spec=0)
        try:
            _write_node_scalar(q.electrons.density, ne3)
        except Exception:
            pass

        # Electron pressure
        pe_eq = data.get('peq', None)
        if pe_eq is not None:
            pe_eq = np.asarray(pe_eq, dtype=float)
            if pe_eq.shape != Rloc.shape and pe_eq.T.shape == Rloc.shape:
                pe_eq = pe_eq.T
        pe3 = _recon_from(pe_eq, fields.get('repe', None), fields.get('impe', None))
        try:
            _write_node_scalar(q.electrons.pressure, pe3)
        except Exception:
            pass

        # Total pressure (if available)
        pr_eq = data.get('prq', None)
        if pr_eq is not None:
            pr_eq = np.asarray(pr_eq, dtype=float)
            if pr_eq.shape != Rloc.shape and pr_eq.T.shape == Rloc.shape:
                pr_eq = pr_eq.T
        pr3 = _recon_from(pr_eq, fields.get('repr', None), fields.get('impr', None))
        # Some DDs provide a scalar pressure leaf; try a few common names.
        for _leafname in ('pressure', 'p_total', 'pressure_total'):
            try:
                _write_node_scalar(getattr(q, _leafname), pr3)
                break
            except Exception:
                continue

        # Ion density: either total (sum over ions) or per-ion selection when species_index is set.
        ion_index = None
        try:
            if species_index is not None and int(species_index) >= 1:
                ion_index = int(species_index) - 1  # ions start after electrons
        except Exception:
            ion_index = None

        ni_eq = None
        try:
            if nqA is not None and nqA.ndim >= 3 and nqA.shape[2] >= 2:
                if ion_index is None:
                    ni_eq = np.nansum(nqA[:, :, 1:], axis=2)
                else:
                    _idx = 1 + int(ion_index)
                    if 1 <= _idx < int(nqA.shape[2]):
                        ni_eq = nqA[:, :, _idx]
                if ni_eq is not None and ni_eq.shape != Rloc.shape and ni_eq.T.shape == Rloc.shape:
                    ni_eq = ni_eq.T
            elif ne_eq is not None:
                z = float(zeff_input) if zeff_input not in (None, '') else None
                if z is not None and z > 0.0:
                    ni_eq = np.asarray(ne_eq, dtype=float) / z
                    _log(f"ion density fallback: ni = ne/zeff_input (zeff_input={z:g})")
                else:
                    ni_eq = np.asarray(ne_eq, dtype=float)
                    _log("ion density fallback: ni = ne (zeff_input unavailable)")
        except Exception:
            ni_eq = None

        # Ion density modes: if multi-species exist, either sum over ions or select a single ion species.
        reNi = imNi = None
        try:
            reN = fields.get('rend', None)
            imN = fields.get('imnd', None)
            if reN is not None and imN is not None:
                reN = np.asarray(reN)
                imN = np.asarray(imN)
                if reN.ndim == 4 and reN.shape[2] >= 2:
                    if ion_index is None:
                        reNi = np.nansum(reN[:, :, 1:, :], axis=2)
                        imNi = np.nansum(imN[:, :, 1:, :], axis=2)
                    else:
                        _idx = 1 + int(ion_index)
                        if 1 <= _idx < int(reN.shape[2]):
                            reNi = reN[:, :, _idx, :]
                            imNi = imN[:, :, _idx, :]
        except Exception:
            reNi = imNi = None
        ni3 = _recon_from(ni_eq, reNi, imNi)
        try:
            _write_node_scalar(q.n_i_total, ni3)
        except Exception:
            pass

        # --- Extra density-related leaves (when present in the DD) ---
        # IMAS 4.1.x `mhd` IDS provides n_i_total, zeff, and mass_density, but does **not**
        # define an electron density leaf. We still reconstruct n_e internally (ne3) for
        # computing zeff and for convenience, but it may not be storable in mhd.
        try:
            # ion charges/masses from nimrod.in (if available)
            z_ions = list(sp.get('z_ions', []) or sp.get('z_ion', []) or [])
            m_ions = list(sp.get('m_ions_kg', []) or [])
            amu = 1.66053906660e-27

            # Helper: default mass when metadata is missing (assume deuterium)
            def _mi_default(idx: int) -> float:
                try:
                    if idx < len(m_ions) and float(m_ions[idx]) > 0.0:
                        return float(m_ions[idx])
                except Exception:
                    pass
                return 2.0 * amu

            # --- mass density ---
            if hasattr(q, 'mass_density'):
                rho3 = None
                if ni3 is not None:
                    if ion_index is not None:
                        # Per-ion occurrence: rho = m_i * n_i
                        rho3 = _mi_default(int(ion_index)) * np.asarray(ni3, dtype=float)
                    else:
                        # Total occurrence: rho = sum_i m_i * n_i
                        if nqA is not None and nqA.ndim >= 3 and nqA.shape[2] >= 2:
                            nion2 = int(nqA.shape[2] - 1)
                            rho_acc = None
                            for ii in range(nion2):
                                ni_eq_i = nqA[:, :, 1 + ii]
                                if ni_eq_i.shape != Rloc.shape and ni_eq_i.T.shape == Rloc.shape:
                                    ni_eq_i = ni_eq_i.T
                                # pick ion-i modes
                                reN = fields.get('rend', None)
                                imN = fields.get('imnd', None)
                                re_i = im_i = None
                                try:
                                    if reN is not None and imN is not None:
                                        reN = np.asarray(reN); imN = np.asarray(imN)
                                        if reN.ndim == 4 and reN.shape[2] >= (2 + ii):
                                            re_i = reN[:, :, 1 + ii, :]
                                            im_i = imN[:, :, 1 + ii, :]
                                except Exception:
                                    re_i = im_i = None
                                ni3_i = _recon_from(ni_eq_i, re_i, im_i)
                                if ni3_i is None:
                                    continue
                                mi = _mi_default(ii)
                                term = mi * np.asarray(ni3_i, dtype=float)
                                rho_acc = term if rho_acc is None else (rho_acc + term)
                            rho3 = rho_acc
                        else:
                            # Fallback: treat ni3 as single-ion density
                            rho3 = _mi_default(0) * np.asarray(ni3, dtype=float)

                if rho3 is not None:
                    try:
                        _write_node_scalar(q.mass_density, rho3)
                    except Exception:
                        pass

            # --- zeff ---
            if hasattr(q, 'zeff'):
                zeff3 = None
                if ion_index is None:
                    # Prefer a density-based estimate: zeff = sum(Z_i^2 n_i) / sum(Z_i n_i)
                    if nqA is not None and nqA.ndim >= 3 and nqA.shape[2] >= 2:
                        nion2 = int(nqA.shape[2] - 1)
                        sumZ2n = None
                        sumZn  = None
                        for ii in range(nion2):
                            Zi = 1.0
                            try:
                                if ii < len(z_ions) and float(z_ions[ii]) > 0.0:
                                    Zi = float(z_ions[ii])
                            except Exception:
                                Zi = 1.0
                            ni_eq_i = nqA[:, :, 1 + ii]
                            if ni_eq_i.shape != Rloc.shape and ni_eq_i.T.shape == Rloc.shape:
                                ni_eq_i = ni_eq_i.T
                            reN = fields.get('rend', None)
                            imN = fields.get('imnd', None)
                            re_i = im_i = None
                            try:
                                if reN is not None and imN is not None:
                                    reN = np.asarray(reN); imN = np.asarray(imN)
                                    if reN.ndim == 4 and reN.shape[2] >= (2 + ii):
                                        re_i = reN[:, :, 1 + ii, :]
                                        im_i = imN[:, :, 1 + ii, :]
                            except Exception:
                                re_i = im_i = None
                            ni3_i = _recon_from(ni_eq_i, re_i, im_i)
                            if ni3_i is None:
                                continue
                            ni3_i = np.asarray(ni3_i, dtype=float)
                            termZn  = Zi * ni3_i
                            termZ2n = (Zi * Zi) * ni3_i
                            sumZn  = termZn  if sumZn  is None else (sumZn  + termZn)
                            sumZ2n = termZ2n if sumZ2n is None else (sumZ2n + termZ2n)

                        if sumZn is not None and sumZ2n is not None:
                            with np.errstate(divide='ignore', invalid='ignore'):
                                zeff3 = np.asarray(sumZ2n, dtype=float) / np.asarray(sumZn, dtype=float)
                    # Minimal fallback when no ion-resolved densities exist
                    if zeff3 is None and zeff_input not in (None, ''):
                        try:
                            if ni3 is not None:
                                _ones = np.ones_like(np.asarray(ni3, dtype=float))
                            else:
                                _ones = np.ones((int(Rloc.shape[0]), int(Rloc.shape[1]), int(nphi)), dtype=float)
                            zeff3 = float(zeff_input) * _ones
                        except Exception:
                            zeff3 = None
                # For per-ion occurrences, zeff is redundant; skip.
                if zeff3 is not None:
                    try:
                        _write_node_scalar(q.zeff, zeff3)
                    except Exception:
                        pass
        except Exception:
            pass

        # Ion pressure and ion temperature derived if needed
        pi3 = None
        if pr3 is not None and pe3 is not None:
            pi3 = np.asarray(pr3, dtype=float) - np.asarray(pe3, dtype=float)
            for _leafname in ('p_i_total', 'ions_pressure', 'ion_pressure', 'p_ions'):
                try:
                    _write_node_scalar(getattr(q, _leafname), pi3)
                    break
                except Exception:
                    continue

        # If Ti modes absent, derive Ti from pi/ni.
        if 't_i_average' in dir(q):
            try:
                Ti3 = _recon_native('tiq', 'reti', 'imti')
            except Exception:
                Ti3 = None
            if Ti3 is None and pi3 is not None and ni3 is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    Ti3 = np.asarray(pi3, dtype=float) / (np.asarray(ni3, dtype=float) * qe)
                _log('derived Ti = p_i / (n_i * qe) (tiq/reti/imti not available)')
            try:
                _write_node_scalar(q.t_i_average, Ti3)
            except Exception:
                pass

        # Current density components (J). NIMROD provides J in (R,Z,phi) components in jq[...,0:3].
        # Reconstruct full fields as eq + sum_m (Re*cos(m*phi)+Im*sin(m*phi)).
        jqA = None
        try:
            jqA = np.asarray(data.get('jq'), dtype=float) if data.get('jq') is not None else None
            if jqA is not None and jqA.ndim >= 3 and Rloc is not None:
                # Some dumps store transposed arrays; align to Rloc.
                if jqA.shape[0:2] != Rloc.shape and jqA[:, :, 0].T.shape == Rloc.shape:
                    jqA = np.transpose(jqA, (1, 0, 2))
        except Exception:
            jqA = None

        def _write_J_component(comp: int, leaf_candidates: tuple[str, ...]) -> None:
            j_eq = None
            try:
                if jqA is not None and jqA.ndim >= 3 and jqA.shape[2] > comp:
                    j_eq = jqA[:, :, comp]
            except Exception:
                j_eq = None
            j_full = _recon_from(j_eq, fields.get('reja', None), fields.get('imja', None), comp=comp)
            if j_full is None:
                return
            for _leaf in leaf_candidates:
                if hasattr(q, _leaf):
                    try:
                        _write_node_scalar(getattr(q, _leaf), j_full)
                        break
                    except Exception:
                        continue

        # Toroidal/phi (comp=2) + poloidal plane components (comp=0:R, comp=1:Z)
        _write_J_component(2, ("j_tor", "j_phi", "jtor", "current_density_tor", "current_density_phi", "current_density_tor_s"))
        _write_J_component(0, ("j_r", "j_R", "jr", "current_density_r", "current_density_R", "current_density_radial"))
        _write_J_component(1, ("j_z", "j_Z", "jz", "current_density_z", "current_density_Z", "current_density_axial"))

        # Toroidal rotation frequency omega = v_phi / R
        v_eq = None
        try:
            vqA = np.asarray(data.get('vq'), dtype=float) if data.get('vq') is not None else None
            if vqA is not None and vqA.ndim >= 3:
                v_eq = vqA[:, :, 2]
                if v_eq.shape != Rloc.shape and v_eq.T.shape == Rloc.shape:
                    v_eq = v_eq.T
        except Exception:
            v_eq = None
        v3 = _recon_from(v_eq, fields.get('reve', None), fields.get('imve', None), comp=2)
        omega3 = None
        if v3 is not None:
            R3 = np.repeat(np.asarray(Rloc, dtype=float)[:, :, None], int(nphi), axis=2)
            with np.errstate(divide='ignore', invalid='ignore'):
                omega3 = np.asarray(v3, dtype=float) / R3
        try:
            _write_node_scalar(q.rotation_frequency_tor_s, omega3)
        except Exception:
            pass

        # Velocity components (explicit leaves in mhd.ggd): velocity_r, velocity_z, velocity_phi
        try:
            vqA = np.asarray(data.get('vq'), dtype=float) if data.get('vq') is not None else None
            v0_eq = v1_eq = v2_eq = None
            if vqA is not None and vqA.ndim >= 3:
                v0_eq = vqA[:, :, 0]
                v1_eq = vqA[:, :, 1]
                v2_eq = vqA[:, :, 2]
                if v0_eq.shape != Rloc.shape and v0_eq.T.shape == Rloc.shape:
                    v0_eq = v0_eq.T
                    v1_eq = v1_eq.T
                    v2_eq = v2_eq.T
        except Exception:
            v0_eq = v1_eq = v2_eq = None

        v_r_3   = _recon_from(v0_eq, fields.get('reve', None), fields.get('imve', None), comp=0)
        v_z_3   = _recon_from(v1_eq, fields.get('reve', None), fields.get('imve', None), comp=1)
        v_phi_3 = _recon_from(v2_eq, fields.get('reve', None), fields.get('imve', None), comp=2)
        for _nm, _V3 in (('velocity_r', v_r_3), ('velocity_z', v_z_3), ('velocity_phi', v_phi_3)):
            if hasattr(q, _nm):
                try:
                    _write_node_scalar(getattr(q, _nm), _V3)
                except Exception:
                    pass
        # Finished FE-node GGD population; skip legacy duplicate reconstructions below.
        return

        # Electron density (species 0): use nq[...,0] for equilibrium and rend/imnd (species 0) for modes when present.
        try:
            ne_eq = None
            if data.get("nq") is not None:
                nqA = np.asarray(data.get("nq"), dtype=float)
                if nqA.ndim == 3 and nqA.shape[2] >= 1:
                    ne_eq = nqA[:, :, 0]
            reN = fields.get("rend", None)
            imN = fields.get("imnd", None)
            if reN is not None and getattr(reN, "ndim", 0) == 4:
                reN0 = np.asarray(reN)[:, :, 0, :]
                imN0 = np.asarray(imN)[:, :, 0, :]
            else:
                reN0 = reN
                imN0 = imN
            # reconstruct on phi planes
            ne3 = None
            if ne_eq is not None or (reN0 is not None and imN0 is not None):
                vals_phi = []
                try:
                    _ps = float(getattr(args, "pert_scale", 1.0) or 1.0)
                except Exception:
                    _ps = 1.0
                for phi in phi_list:
                    full = _reconstruct_full_from_modes(ne_eq, reN0, imN0, keff, float(phi), pert_scale=_ps)
                    if full is None:
                        continue
                    if full.shape != Rloc.shape and full.T.shape == Rloc.shape:
                        full = full.T
                    vals_phi.append(np.asarray(full, dtype=float))
                if vals_phi:
                    ne3 = np.stack(vals_phi, axis=2)
                _write_node_scalar(q.electrons.density, ne3)
        except Exception:
            pass

        # Electron pressure and total pressure (when available)
        try:
            Pe3 = _recon_native("peq", "repe", "impe")
            _write_node_scalar(q.electrons.pressure, Pe3)
        except Exception:
            pass
        try:
            Pr3 = _recon_native("prq", "repr", "impr")
            # DD leaf name varies; try common candidates
            for _leaf in ("pressure", "total_pressure", "pressure_total"):
                try:
                    _write_node_scalar(getattr(q, _leaf), Pr3)
                    break
                except Exception:
                    continue
        except Exception:
            pass

        # Toroidal current density and toroidal rotation frequency, component selection (phi comp=2)
        try:
            j_eq = data.get("jq", None)
            reJ = fields.get("reja", None)
            imJ = fields.get("imja", None)
            if j_eq is not None:
                j_eq = np.asarray(j_eq, dtype=float)
                if j_eq.ndim >= 3:
                    j_eq_phi = j_eq[:, :, 2]
                else:
                    j_eq_phi = None
            else:
                j_eq_phi = None
            if reJ is not None and getattr(reJ, "ndim", 0) == 4:
                reJphi = np.asarray(reJ)[:, :, 2, :]
                imJphi = np.asarray(imJ)[:, :, 2, :]
            else:
                reJphi = imJphi = None
            j3 = None
            if j_eq_phi is not None or (reJphi is not None and imJphi is not None):
                vals_phi=[]
                try:
                    _ps=float(getattr(args,"pert_scale",1.0) or 1.0)
                except Exception:
                    _ps=1.0
                for phi in phi_list:
                    full=_reconstruct_full_from_modes(j_eq_phi, reJphi, imJphi, keff, float(phi), pert_scale=_ps)
                    if full is None:
                        continue
                    if full.shape != Rloc.shape and full.T.shape == Rloc.shape:
                        full=full.T
                    vals_phi.append(full)
                if vals_phi:
                    j3=np.stack(vals_phi,axis=2)
                # DD leaf name varies
                for _leaf in ("current_density_tor", "j_phi", "current_density_tor_s"):
                    try:
                        _write_node_scalar(getattr(q,_leaf), j3)
                        break
                    except Exception:
                        continue
        except Exception:
            pass

        try:
            v_eq = data.get("vq", None)
            reV = fields.get("reve", None)
            imV = fields.get("imve", None)
            if v_eq is not None:
                v_eq = np.asarray(v_eq, dtype=float)
                vphi_eq = v_eq[:, :, 2] if v_eq.ndim >= 3 else None
            else:
                vphi_eq = None
            if reV is not None and getattr(reV,"ndim",0) == 4:
                reVphi = np.asarray(reV)[:, :, 2, :]
                imVphi = np.asarray(imV)[:, :, 2, :]
            else:
                reVphi = imVphi = None
            omega3 = None
            if vphi_eq is not None or (reVphi is not None and imVphi is not None):
                vals_phi=[]
                try:
                    _ps=float(getattr(args,"pert_scale",1.0) or 1.0)
                except Exception:
                    _ps=1.0
                for phi in phi_list:
                    full=_reconstruct_full_from_modes(vphi_eq, reVphi, imVphi, keff, float(phi), pert_scale=_ps)
                    if full is None:
                        continue
                    if full.shape != Rloc.shape and full.T.shape == Rloc.shape:
                        full=full.T
                    # omega = vphi/R
                    omega = full / np.where(np.asarray(Rloc,dtype=float)==0.0, 1.0, np.asarray(Rloc,dtype=float))
                    vals_phi.append(omega)
                if vals_phi:
                    omega3 = np.stack(vals_phi, axis=2)
                for _leaf in ("rotation_frequency_tor_s","omega_tor","omega_phi"):
                    try:
                        _write_node_scalar(getattr(q,_leaf), omega3)
                        break
                    except Exception:
                        continue
        except Exception:
            pass

        return

    def _make_values(eq_key: str, re_key: str, im_key: str):
        eq = data.get(eq_key, None)
        # If equilibrium temperature fields are not written by NIMROD (common for no-impurity dumps),
        # derive a consistent equilibrium baseline from (peq, prq, nq) so "full" fields include equilibrium
        # + all available Fourier content.
        if eq is None:
            try:
                qe = 1.602176634e-19
                if eq_key == "teq":
                    peq = data.get("peq", None)
                    nq  = data.get("nq", None)
                    if peq is not None and nq is not None:
                        nqA = np.asarray(nq, dtype=float)
                        ne2d = nqA[..., 0] if nqA.ndim >= 3 else nqA
                        with np.errstate(divide="ignore", invalid="ignore"):
                            eq = np.asarray(peq, dtype=float) / (np.asarray(ne2d, dtype=float) * qe)
                elif eq_key == "tiq":
                    prq = data.get("prq", None)
                    peq = data.get("peq", None)
                    nq  = data.get("nq", None)
                    if prq is not None and peq is not None and nq is not None:
                        nqA = np.asarray(nq, dtype=float)
                        if nqA.ndim >= 3 and nqA.shape[-1] >= 2:
                            ni2d = np.nansum(np.asarray(nqA[..., 1:], dtype=float), axis=2)
                            pi2d = np.asarray(prq, dtype=float) - np.asarray(peq, dtype=float)
                            with np.errstate(divide="ignore", invalid="ignore"):
                                eq = np.asarray(pi2d, dtype=float) / (np.asarray(ni2d, dtype=float) * qe)
            except Exception:
                eq = None
        reA = fields.get(re_key, None)
        imA = fields.get(im_key, None)
        if eq is None and (reA is None or imA is None):
            return None, None, None
        vals_phi = []
        rc = zc = None
        try:
            _ps = float(getattr(args, "pert_scale", 1.0) or 1.0)
        except Exception:
            _ps = 1.0
        for phi in phi_list:
            full = _reconstruct_full_from_modes(eq, reA, imA, keff, float(phi), pert_scale=_ps)
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
            # NOTE: histogram bin-averaging can leave empty bins between discrete
            # radial surfaces (appearing as "missing" rings and non-monotonic profiles).
            # Linear interpolation on a triangulation fills between surfaces.
            rc, zc, vb = _interp2d_linear(Rloc, Zloc, full, nb, nb)
            vals_phi.append(vb)
        if not vals_phi:
            return None, None, None
        V3 = np.stack(vals_phi, axis=2)  # (nr, nz, nphi)
        return rc, zc, V3        


    def _slice_component(A, comp: int):
        """Extract scalar component from vector equilibrium/modes with flexible axis ordering."""
        if A is None:
            return None
        import numpy as _np
        nm = int(len(keff))
        AA = _np.asarray(A)
        c = int(comp)

        # equilibrium vectors: (R,Z,3)
        if AA.ndim == 3 and AA.shape[-1] == 3:
            return AA[:, :, c]

        # unpacked vector modes: (R,Z,nmodes,3) or (R,Z,3,nmodes)
        if AA.ndim == 4:
            if AA.shape[-1] == 3 and AA.shape[-2] == nm:
                return AA[:, :, :, c]
            if AA.shape[2] == 3 and AA.shape[3] == nm:
                return AA[:, :, c, :]

        # packed vector modes: (R,Z,3*nmodes)
        if AA.ndim == 3 and AA.shape[-1] == 3 * nm:
            return AA[:, :, c * nm : (c + 1) * nm]

        return None

    def _make_values_comp(eq_key: str, re_key: str, im_key: str, comp: int):
        # Like _make_values, but for vector fields (select component first).
        eq0 = _slice_component(data.get(eq_key, None), comp)
        re0 = _slice_component(fields.get(re_key, None), comp)
        im0 = _slice_component(fields.get(im_key, None), comp)
        if eq0 is None and (re0 is None or im0 is None):
            return None, None, None
        vals_phi = []
        rc = zc = None
        try:
            _ps = float(getattr(args, 'pert_scale', 1.0) or 1.0)
        except Exception:
            _ps = 1.0
        for phi in phi_list:
            full = _reconstruct_full_from_modes(eq0, re0, im0, keff, float(phi), pert_scale=_ps)
            if full is None:
                continue
            Rloc = R
            Zloc = Z
            if hasattr(full, 'shape') and full.shape != Rloc.shape:
                if getattr(full, 'T', None) is not None and full.T.shape == Rloc.shape:
                    full = full.T
                elif Rloc.T.shape == full.shape:
                    Rloc = Rloc.T
                    Zloc = Zloc.T
            rc, zc, vb = _interp2d_linear(Rloc, Zloc, full, nb, nb)
            vals_phi.append(vb)
        if not vals_phi:
            return None, None, None
        import numpy as _np
        V3 = _np.stack(vals_phi, axis=2)
        return rc, zc, V3

    it, ig = _append_time_ggd(mhd, t, write_grid=_ggd_should_write_grid(args), reuse_grid=bool(getattr(args, "ggd_reuse_grid", False)))
    g = mhd.grid_ggd[ig]
    gidx = int(ig + 1)
    try:
        g.identifier.name = "nimrod_rzphi_regular"
        g.identifier.index = int(ig + 1)
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
                s0.dimension = 1
                s0.identifier.name = "nodes"
                s0.identifier.index = 1
                s0.identifier.description = "Unstructured nodes"
            except Exception:
                pass
            try:
                # Create minimal element AoS; packed writer will create/resize object leaves.
                # IMPORTANT: keep this non-empty so the HDF5 backend emits AOS_SHAPE datasets.
                s0.element.resize(1)
                try:
                    s0.element[0].object.resize(3)
                    for kk in range(3):
                        try:
                            s0.element[0].object[kk].real = 0.0
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                pass
            except Exception:
                pass

            # Subset 1: volumes (dimension=3)
            s1 = g.grid_subset[1]
            try:
                s1.dimension = 4
                s1.identifier.name = "volumes"
                s1.identifier.index = 43
                s1.identifier.description = "Unstructured connectivity"
            except Exception:
                pass
            try:
                # Base points to nodes subset
                s1.base.resize(1)
                s1.base[0].index = 0
                s1.base[0].grid_subset_index = 1
            except Exception:
                pass
            try:
                # Create minimal element AoS; packed writer will create/resize object leaves.
                # IMPORTANT: keep this non-empty so the HDF5 backend emits AOS_SHAPE datasets.
                s1.element.resize(1)
                try:
                    # vertex count placeholder depends on requested connectivity
                    _nobj = 8 if conn_kind == 'hex' else (6 if conn_kind == 'fe_wedge' else (3 if conn_kind == 'fe_tri' else 8))
                    s1.element[0].object.resize(int(_nobj))
                    for kk in range(int(_nobj)):
                        try:
                            s1.element[0].object[kk].index = 0
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                pass
            except Exception:
                pass
        except Exception:
            pass

        if conn_kind == "fe_pointcloud":
            try:
                g.grid_subset.resize(1)
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

    # In structured mode, define at least the nodes subset per IMAS GGD specification.
    if not getattr(args, 'ggd_unstructured', False):
        try:
            g.grid_subset.resize(1)
            gs = g.grid_subset[0]
            gs.identifier.name = 'nodes'
            gs.identifier.index = 1
            gs.identifier.description = 'All nodes of the structured grid'
            gs.dimension = 1
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
            _cn = str(coord_name).strip().lower()
            space_obj.coordinates_type[0].name = _cn
            space_obj.coordinates_type[0].index = {'r': 4, 'z': 3, 'phi': 5}.get(_cn, -1)
            space_obj.coordinates_type[0].description = _cn
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
        # Define explicit coordinate axes for the structured product grid (R, Z, phi).
        try:
            Rm = np.asarray(R, dtype=float).ravel(order='F')
            Zm = np.asarray(Z, dtype=float).ravel(order='F')
            msk = np.isfinite(Rm) & np.isfinite(Zm)
            Rm = Rm[msk]; Zm = Zm[msk]
            if Rm.size > 0 and Zm.size > 0:
                rmin, rmax = np.quantile(Rm, [0.001, 0.999])
                zmin, zmax = np.quantile(Zm, [0.001, 0.999])
            else:
                rmin, rmax = float(np.nanmin(R)), float(np.nanmax(R))
                zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
        except Exception:
            rmin, rmax = float(np.nanmin(R)), float(np.nanmax(R))
            zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
        rc0 = np.linspace(float(rmin), float(rmax), int(nb))
        zc0 = np.linspace(float(zmin), float(zmax), int(nb))
        _fill_space(g.space[0], "R", rc0)
        _fill_space(g.space[1], "Z", zc0)
        if nphi > 1:
            _fill_space(g.space[2], "phi", np.asarray(phi_list, dtype=float))

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

    # In unstructured mode, node-centered fields must reference the nodes subset (grid_subset_index=0).
    # In structured mode we keep the historical identifier.index=1 convention.
    gs_nodes = 1  # nodes subset identifier.index per IMAS ggd_subset_identifier


    def _density_reduce_eq(eqA, mode: str):
        """Reduce equilibrium density to a single 2D field: 'electron', 'ions', or 'total'."""
        if eqA is None:
            return None
        import numpy as _np
        A = _np.asarray(eqA, dtype=float)
        if A.ndim == 2:
            return A
        if A.ndim == 3:
            if A.shape[2] == 0:
                return None
            if mode == 'electron':
                return A[:, :, 0]
            if mode == 'ions':
                return _np.nansum(A[:, :, 1:], axis=2) if A.shape[2] > 1 else _np.zeros(A[:, :, 0].shape, dtype=float)
            return _np.nansum(A, axis=2)
        return None

    def _density_reduce_modes(modesA, mode: str):
        """Reduce density Fourier coeffs to (R,Z,nmodes): 'electron', 'ions', or 'total'."""
        if modesA is None:
            return None
        import numpy as _np
        nm = int(len(keff))
        A = _np.asarray(modesA, dtype=float)

        # Unpack packed (R,Z,nspec*nmodes)
        if A.ndim == 3 and A.shape[-1] % nm == 0 and A.shape[-1] != nm:
            try:
                order = getattr(args, 'dens_pert_order', 'species_major')
            except Exception:
                order = 'species_major'
            try:
                A, _ns = _unpack_density_modes(A, nm, None, order)
            except Exception:
                return None

        if A.ndim == 4:
            # (R,Z,nspec,nmodes)
            if A.shape[-1] == nm:
                if mode == 'electron':
                    return A[:, :, 0, :] if A.shape[2] >= 1 else None
                if mode == 'ions':
                    return _np.nansum(A[:, :, 1:, :], axis=2) if A.shape[2] > 1 else _np.zeros(A[:, :, 0, :].shape, dtype=float)
                return _np.nansum(A, axis=2)
            # (R,Z,nmodes,nspec)
            if A.shape[2] == nm:
                if mode == 'electron':
                    return A[:, :, :, 0] if A.shape[3] >= 1 else None
                if mode == 'ions':
                    return _np.nansum(A[:, :, :, 1:], axis=3) if A.shape[3] > 1 else _np.zeros(A[:, :, :, 0].shape, dtype=float)
                return _np.nansum(A, axis=3)

        # Already (R,Z,nmodes)
        if A.ndim == 3 and A.shape[-1] == nm:
            return A
        return None

    def _make_values_density(mode: str):
        """Reconstruct & downsample density to the product grid with explicit species handling."""
        eq0 = _density_reduce_eq(data.get('nq', None), mode)
        re0 = _density_reduce_modes(fields.get('rend', None), mode)
        im0 = _density_reduce_modes(fields.get('imnd', None), mode)
        if eq0 is None and (re0 is None or im0 is None):
            return None, None, None
        vals_phi = []
        rc = zc = None
        try:
            _ps = float(getattr(args, 'pert_scale', 1.0) or 1.0)
        except Exception:
            _ps = 1.0
        for phi in phi_list:
            full = _reconstruct_full_from_modes(eq0, re0, im0, keff, float(phi), pert_scale=_ps)
            if full is None:
                continue
            Rloc = R
            Zloc = Z
            if hasattr(full, 'shape') and full.shape != Rloc.shape:
                if getattr(full, 'T', None) is not None and full.T.shape == Rloc.shape:
                    full = full.T
                elif Rloc.T.shape == full.shape:
                    Rloc = Rloc.T
                    Zloc = Zloc.T
            rc, zc, vb = _interp2d_linear(Rloc, Zloc, full, nb, nb)
            vals_phi.append(vb)
        if not vals_phi:
            return None, None, None
        import numpy as _np
        V3 = _np.stack(vals_phi, axis=2)
        return rc, zc, V3

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
            qt.grid_index = gidx
            qt.grid_subset_index = 1
            _set_values_and_shape(qt, vals, (len(rc), len(zc), int(Te3.shape[2])))
        except Exception:
            pass


    # electrons.density (structured product grid)
    rcN, zcN, ne3 = _make_values_density('electron')
    if ne3 is not None:
        if not getattr(args, "ggd_unstructured", False):
            _fill_space(g.space[0], "R", rcN)
            _fill_space(g.space[1], "Z", zcN)
            if nphi > 1:
                phi_pad = np.full(int(len(rcN)), np.nan, dtype=float)
                phi_pad[: int(nphi)] = phi_list
                _fill_space(g.space[2], "phi", phi_pad)
        vals = np.asarray(ne3, dtype=float).ravel(order="F")
        try:
            q.electrons.density.resize(1)
            qn = q.electrons.density[0]
            qn.grid_index = gidx
            qn.grid_subset_index = 1
            _set_values_and_shape(qn, vals, (len(rcN), len(zcN), int(ne3.shape[2])))
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
            qt.grid_index = gidx
            qt.grid_subset_index = 1
            _set_values_and_shape(qt, vals, (len(rc2), len(zc2), int(Ti3.shape[2])))
        except Exception:
            pass

    # n_i_total (ions only; excludes electrons)
    rc3, zc3, n3 = _make_values_density('ions')
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
            qt.grid_index = gidx
            qt.grid_subset_index = 1
            _set_values_and_shape(qt, vals, (len(rc3), len(zc3), int(n3.shape[2])))
        except Exception:
            pass


    # --- Velocity components (equilibrium + perturbations): v_r, v_z, v_phi ---
    for _leaf, _comp in (("velocity_r", 0), ("velocity_z", 1), ("velocity_phi", 2)):
        try:
            rcv, zcv, Vv3 = _make_values_comp("vq", "reve", "imve", _comp)
        except Exception:
            rcv = zcv = Vv3 = None
        if Vv3 is None:
            continue
        vals = __import__('numpy').asarray(Vv3, dtype=float).ravel(order='F')
        if hasattr(q, _leaf):
            try:
                getattr(q, _leaf).resize(1)
                qt = getattr(q, _leaf)[0]
                qt.grid_index = gidx
                qt.grid_subset_index = 1
                _set_values_and_shape(qt, vals, (len(rcv), len(zcv), int(Vv3.shape[2])))
            except Exception:
                pass

    # Derived toroidal rotation frequency omega = v_phi / R
    try:
        rcv, zcv, Vphi3 = _make_values_comp("vq", "reve", "imve", 2)
    except Exception:
        rcv = zcv = Vphi3 = None
    if Vphi3 is not None and hasattr(q, 'rotation_frequency_tor_s'):
        try:
            import numpy as _np
            RR, ZZ = _np.meshgrid(_np.asarray(rcv, dtype=float), _np.asarray(zcv, dtype=float), indexing='ij')
            R3 = RR[:, :, None]
            with _np.errstate(divide='ignore', invalid='ignore'):
                omega3 = _np.asarray(Vphi3, dtype=float) / R3
            vals = _np.asarray(omega3, dtype=float).ravel(order='F')
            q.rotation_frequency_tor_s.resize(1)
            qt = q.rotation_frequency_tor_s[0]
            qt.grid_index = gidx
            qt.grid_subset_index = 1
            _set_values_and_shape(qt, vals, (len(rcv), len(zcv), int(Vphi3.shape[2])))
        except Exception:
            pass

    # Toroidal current density j_phi (equilibrium + perturbations)
    try:
        rcj, zcj, J3 = _make_values_comp("jq", "reja", "imja", 2)
    except Exception:
        rcj = zcj = J3 = None
    if J3 is not None:
        for _leaf in ("current_density_tor", "j_phi", "current_density_phi"):
            if hasattr(q, _leaf):
                try:
                    vals = __import__('numpy').asarray(J3, dtype=float).ravel(order='F')
                    getattr(q, _leaf).resize(1)
                    qt = getattr(q, _leaf)[0]
                    qt.grid_index = gidx
                    qt.grid_subset_index = 1
                    _set_values_and_shape(qt, vals, (len(rcj), len(zcj), int(J3.shape[2])))
                    break
                except Exception:
                    continue
    # Poloidal-plane current density components j_r and j_z (equilibrium + perturbations)
    for _leaf, _comp, _cands in (
        ("j_r", 0, ("j_r", "current_density_r")),
        ("j_z", 1, ("j_z", "current_density_z")),
    ):
        try:
            rcj, zcj, Jc3 = _make_values_comp("jq", "reja", "imja", _comp)
        except Exception:
            rcj = zcj = Jc3 = None
        if Jc3 is None:
            continue
        vals = __import__('numpy').asarray(Jc3, dtype=float).ravel(order='F')
        for _nm in _cands:
            if hasattr(q, _nm):
                try:
                    getattr(q, _nm).resize(1)
                    qt = getattr(q, _nm)[0]
                    qt.grid_index = gidx
                    qt.grid_subset_index = 1
                    _set_values_and_shape(qt, vals, (len(rcj), len(zcj), int(Jc3.shape[2])))
                    break
                except Exception:
                    continue







def _compute_product_grid_axes_nodes(data, args, *, nb=None, nphi=None):
    # Construct a simple product grid in (R,Z,phi) used by the unstructured GGD mode.
    # This mirrors the logic in _write_unstructured_ggd_aux_h5  # legacy (not used in DD-compliant FE mode), but returns arrays in-memory.
    nb = int(getattr(args, 'ggd_nbins', 128) if nb is None else nb)
    nphi = int(getattr(args, 'ggd_nphi', 8) if nphi is None else nphi)

    # Bounds from any available R/Z mesh information
    rmin = rmax = zmin = zmax = None
    for rk, zk in [('rc', 'zc'), ('R', 'Z')]:
        rr = data.get(rk, None)
        zz = data.get(zk, None)
        if rr is None or zz is None:
            continue
        try:
            rmin = float(np.nanmin(rr))
            rmax = float(np.nanmax(rr))
            zmin = float(np.nanmin(zz))
            zmax = float(np.nanmax(zz))
            break
        except Exception:
            pass

    if rmin is None:
        # Conservative fallback (should not happen for real dumpgll inputs)
        rmin, rmax, zmin, zmax = 0.0, 1.0, -1.0, 1.0

    # Slight padding helps keep boundary points inside interpolation domain
    pad_r = 0.01 * (rmax - rmin) if rmax > rmin else 1e-3
    pad_z = 0.01 * (zmax - zmin) if zmax > zmin else 1e-3
    rmin, rmax = rmin - pad_r, rmax + pad_r
    zmin, zmax = zmin - pad_z, zmax + pad_z

    r_axis = np.linspace(rmin, rmax, nb)
    z_axis = np.linspace(zmin, zmax, nb)
    phi_axis = np.linspace(0.0, 2.0*np.pi, nphi, endpoint=False)

    # Node ordering: flatten (r,z) grid, then tile across phi (same ordering as aux writer)
    rr, zz = np.meshgrid(r_axis, z_axis, indexing='ij')
    rr_flat = rr.reshape(-1, order='F')
    zz_flat = zz.reshape(-1, order='F')

    nodes = np.zeros((rr_flat.size * nphi, 3), dtype=np.float64)
    for k, ph in enumerate(phi_axis):
        s = k * rr_flat.size
        e = (k + 1) * rr_flat.size
        nodes[s:e, 0] = rr_flat
        nodes[s:e, 1] = zz_flat
        nodes[s:e, 2] = ph

    return r_axis, z_axis, phi_axis, nodes



def _interp2d_linear(R2: np.ndarray,
                     Z2: np.ndarray,
                     V2: np.ndarray,
                     nr: int = 128,
                     nz: int = 128,
                     *,
                     r_quantiles: tuple[float, float] = (0.001, 0.999),
                     z_quantiles: tuple[float, float] = (0.001, 0.999),
                     fill_nearest: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate values on an unstructured/stiched (R,Z) mesh to a regular (R,Z) product grid.

    Used for GGD 'product grid' exports to avoid empty-bin artifacts and large NaN regions.
    Steps:
      1) Linear interpolation on a Delaunay triangulation (matplotlib.tri).
      2) Optional nearest-neighbor fill for points outside the convex hull (scipy.spatial.cKDTree).

    Parameters
    ----------
    R2, Z2, V2 : array_like
        2D arrays with identical shapes (or transposes thereof).
    nr, nz : int
        Output grid sizes.
    r_quantiles, z_quantiles : (float, float)
        Robust bounds for (R,Z) limits to reduce outlier influence.
    fill_nearest : bool
        If True, fill NaNs (outside convex hull) using nearest neighbor.

    Returns
    -------
    rc : (nr,) ndarray
    zc : (nz,) ndarray
    Vb : (nr, nz) ndarray
    """
    R2 = np.asarray(R2, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)
    V2 = np.asarray(V2, dtype=float)

    if R2.shape != Z2.shape:
        if R2.T.shape == Z2.shape:
            R2 = R2.T
        else:
            raise ValueError(f"_interp2d_linear: R2.shape={R2.shape} Z2.shape={Z2.shape} mismatch")

    if V2.shape != R2.shape:
        if V2.T.shape == R2.shape:
            V2 = V2.T
        else:
            raise ValueError(f"_interp2d_linear: V2.shape={V2.shape} does not match R2.shape={R2.shape}")

    Rf = R2.ravel()
    Zf = Z2.ravel()
    Vf = V2.ravel()
    m = np.isfinite(Rf) & np.isfinite(Zf) & np.isfinite(Vf)

    if np.count_nonzero(m) < 3:
        # Degenerate input
        rc = np.linspace(float(np.nanmin(Rf[m])) if np.any(m) else 0.0,
                         float(np.nanmax(Rf[m])) if np.any(m) else 1.0, int(nr))
        zc = np.linspace(float(np.nanmin(Zf[m])) if np.any(m) else 0.0,
                         float(np.nanmax(Zf[m])) if np.any(m) else 1.0, int(nz))
        return rc, zc, np.full((int(nr), int(nz)), np.nan, dtype=float)

    Rm = Rf[m]
    Zm = Zf[m]
    Vm = Vf[m]

    rq0, rq1 = r_quantiles
    zq0, zq1 = z_quantiles
    rmin, rmax = np.quantile(Rm, [rq0, rq1])
    zmin, zmax = np.quantile(Zm, [zq0, zq1])

    if (not np.isfinite(rmin)) or (not np.isfinite(rmax)) or abs(rmax - rmin) < 1e-12:
        rmin, rmax = float(np.nanmin(Rm)), float(np.nanmax(Rm))
    if (not np.isfinite(zmin)) or (not np.isfinite(zmax)) or abs(zmax - zmin) < 1e-12:
        zmin, zmax = float(np.nanmin(Zm)), float(np.nanmax(Zm))

    rc = np.linspace(rmin, rmax, int(nr))
    zc = np.linspace(zmin, zmax, int(nz))
    RR, ZZ = np.meshgrid(rc, zc, indexing='ij')

    Vb = np.full((int(nr), int(nz)), np.nan, dtype=float)

    # 1) Linear interpolation on triangulation
    try:
        import matplotlib.tri as _mtri
        tri = _mtri.Triangulation(Rm, Zm)
        itp = _mtri.LinearTriInterpolator(tri, Vm)
        tmp = itp(RR, ZZ)
        if hasattr(tmp, "filled"):
            tmp = tmp.filled(np.nan)
        Vb = np.asarray(tmp, dtype=float)
    except Exception:
        # leave Vb as NaNs; nearest fill below (if enabled) will populate
        pass

    # 2) Nearest-neighbor fill for NaNs
    if fill_nearest:
        nan = ~np.isfinite(Vb)
        if np.any(nan):
            try:
                from scipy.spatial import cKDTree as _cKDTree
                tree = _cKDTree(np.c_[Rm, Zm])
                pts = np.c_[RR[nan], ZZ[nan]]
                _, idx = tree.query(pts, k=1)
                Vb[nan] = Vm[idx]
            except Exception:
                pass

    return rc, zc, Vb


def _interp_mesh_to_points(R, Z, V, rq, zq):
    # Interpolate V(R,Z) from an arbitrary 2D mesh onto query points (rq, zq).
    # Preferred: linear interpolation (scipy). To avoid large NaN regions outside the convex hull
    # (a common issue for regular product grids), we *fill* any remaining NaNs with nearest-neighbour
    # values (KDTree) when scipy is available. Otherwise use a nearest-neighbour fallback.
    pts = np.column_stack([np.asarray(R).ravel(order='C'), np.asarray(Z).ravel(order='C')])
    vals = np.asarray(V).ravel(order='C')

    # Drop NaNs in source to keep triangulation sane
    good = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & np.isfinite(vals)
    pts = pts[good]
    vals = vals[good]

    rq = np.asarray(rq, dtype=np.float64)
    zq = np.asarray(zq, dtype=np.float64)

    out = np.full(rq.shape, np.nan, dtype=np.float64)
    if pts.shape[0] == 0:
        return out

    # Helper: KDTree nearest neighbour fill for a boolean mask on the query grid
    def _fill_nearest(out_arr: np.ndarray, mask_nan: np.ndarray) -> np.ndarray:
        if not np.any(mask_nan):
            return out_arr
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            qpts = np.column_stack([rq[mask_nan].ravel(order='C'), zq[mask_nan].ravel(order='C')])
            _, idx = tree.query(qpts, k=1)
            out_flat = out_arr.ravel(order='C')
            out_flat[np.flatnonzero(mask_nan.ravel(order='C'))] = vals[idx]
            return out_flat.reshape(out_arr.shape, order='C')
        except Exception:
            # brute-force nearest for the masked points only (slow but safe)
            out_flat = out_arr.ravel(order='C')
            rqf = rq.ravel(order='C')
            zqf = zq.ravel(order='C')
            nan_flat = np.flatnonzero(mask_nan.ravel(order='C'))
            for k in nan_flat:
                d2 = (pts[:, 0] - rqf[k])**2 + (pts[:, 1] - zqf[k])**2
                out_flat[k] = vals[int(np.argmin(d2))]
            return out_flat.reshape(out_arr.shape, order='C')

    try:
        from scipy.interpolate import LinearNDInterpolator
        itp = LinearNDInterpolator(pts, vals, fill_value=np.nan)
        out = np.asarray(itp(rq, zq), dtype=np.float64)
        # Fill NaNs with nearest-neighbour values
        out = _fill_nearest(out, ~np.isfinite(out))
        return out
    except Exception:
        # Nearest neighbour fallback (KDTree if available, else brute force for all points)
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            qpts = np.column_stack([rq.ravel(order='C'), zq.ravel(order='C')])
            _, idx = tree.query(qpts, k=1)
            return vals[idx].reshape(rq.shape, order='C').astype(np.float64, copy=False)
        except Exception:
            out_flat = out.ravel(order='C')
            rqf = rq.ravel(order='C')
            zqf = zq.ravel(order='C')
            for k in range(out_flat.size):
                d2 = (pts[:, 0] - rqf[k])**2 + (pts[:, 1] - zqf[k])**2
                out_flat[k] = vals[int(np.argmin(d2))]
            return out_flat.reshape(rq.shape, order='C')

def _write_edge_profiles_electrons_temperature_ggd_h5(entry_dir, occ, *, values_1d, shape_rzp, grid_index=1, grid_subset_index=1, overwrite=True):
    # Write /edge_profiles_<occ>/ggd[]&electrons&temperature[] datasets directly via h5py,
    # so downstream tools (plot_mhd) can find them even if the DD omits electrons in edge_profiles.ggd.
    import h5py

    h5_path = os.path.join(entry_dir, f"edge_profiles_{occ}.h5")
    grp = f"/edge_profiles_{occ}"
    base = "ggd[]&electrons&temperature[]"

    values_1d = np.asarray(values_1d, dtype=np.float64).ravel(order='F')
    shp = np.asarray(shape_rzp, dtype=np.int32).ravel(order='C')
    if shp.size != 3:
        raise ValueError(f"shape_rzp must be (nr,nz,nphi); got {shape_rzp}")

    with h5py.File(h5_path, 'a') as h5:
        if grp not in h5:
            h5.create_group(grp)
        g = h5[grp]

        # If asked not to overwrite and the leaf already exists, leave it as-is
        if (not overwrite) and ((base + '&values') in g):
            return

        # AOS_SHAPE indicates one ggd element and one temperature element
        d = f"{base}&AOS_SHAPE"
        if d in g:
            del g[d]
        g.create_dataset(d, data=np.asarray([[1]], dtype=np.int32))

        # values and its per-element shape
        d = f"{base}&values"
        if d in g:
            del g[d]
        g.create_dataset(d, data=values_1d)

        d = f"{base}&values_SHAPE"
        if d in g:
            del g[d]
        g.create_dataset(d, data=shp.reshape(1, 1, 3))

        # grid references
        d = f"{base}&grid_index"
        if d in g:
            del g[d]
        g.create_dataset(d, data=np.asarray([[grid_index]], dtype=np.int32))

        d = f"{base}&grid_subset_index"
        if d in g:
            del g[d]
        g.create_dataset(d, data=np.asarray([[grid_subset_index]], dtype=np.int32))



def _write_edge_profiles_electrons_density_ggd_h5(entry_dir, occ, *, values_1d, shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True):
    # Write /edge_profiles_<occ>/ggd[]&electrons&density[] datasets directly via h5py.
    import h5py

    h5_path = os.path.join(entry_dir, f"edge_profiles_{occ}.h5")
    grp = f"/edge_profiles_{occ}"
    base = "ggd[]&electrons&density[]"

    values_1d = np.asarray(values_1d, dtype=np.float64).ravel(order='F')
    shp = np.asarray(shape_rzp, dtype=np.int32).ravel(order='C')
    if shp.size != 3:
        raise ValueError(f"shape_rzp must be (nr,nz,nphi); got {shape_rzp}")

    with h5py.File(h5_path, 'a') as h5:
        if grp not in h5:
            h5.create_group(grp)
        g = h5[grp]

        # If asked not to overwrite and the leaf already exists, leave it as-is
        if (not overwrite) and ((base + '&values') in g):
            return

        d = f"{base}&AOS_SHAPE"
        if d in g:
            del g[d]
        g.create_dataset(d, data=np.asarray([[1]], dtype=np.int32))

        d = f"{base}&values"
        if d in g:
            del g[d]
        g.create_dataset(d, data=values_1d)

        d = f"{base}&values_SHAPE"
        if d in g:
            del g[d]
        g.create_dataset(d, data=shp.reshape(1, 1, 3))

        d = f"{base}&grid_index"
        if d in g:
            del g[d]
        g.create_dataset(d, data=np.asarray([[grid_index]], dtype=np.int32))

        d = f"{base}&grid_subset_index"
        if d in g:
            del g[d]
        g.create_dataset(d, data=np.asarray([[grid_subset_index]], dtype=np.int32))


def _write_edge_profiles_ion_velocity_component_ggd_h5(entry_dir, occ, *, component, values_1d, shape_rzp,
                                                   nion=1, grid_index=1, grid_subset_index=0, overwrite=True):
    """Write edge_profiles.ggd ion velocity component datasets directly via h5py.

    We write DD4-compliant ion velocity tokenizations:
      - ggd[]&ion[]&velocity&<comp>[]&...
      - ggd[]&ion[]&velocity[]&<comp> (AoS component leaf)

    values_1d is a single-ion bulk-flow value list (flattened over nodes). We replicate it over nion ions,
    since per-ion flows are typically not available in dumpgll.
    """
    import h5py

    comp = str(component).lower().strip()
    if comp not in ("r", "z", "phi"):
        raise ValueError(f"component must be one of r,z,phi; got {component!r}")

    values_1d = np.asarray(values_1d, dtype=np.float64).ravel(order='F')
    shp = np.asarray(shape_rzp, dtype=np.int32).ravel(order='C')
    if shp.size != 3:
        raise ValueError(f"shape_rzp must be (nr,nz,nphi); got {shape_rzp}")

    if int(nion) < 1:
        nion = 1

    # replicate for all ions
    values_all = np.tile(values_1d, int(nion))

    bases = [f"ggd[]&ion[]&velocity&{comp}[]"]

    h5_path = os.path.join(entry_dir, f"edge_profiles_{occ}.h5")
    grp = f"/edge_profiles_{occ}"

    with h5py.File(h5_path, 'a') as h5:
        if grp not in h5:
            h5.create_group(grp)
        g = h5[grp]

        # AOS: one ggd element; nion ion elements
        aos = np.asarray([[1, int(nion)]], dtype=np.int32)
        gi = np.asarray([[grid_index] * int(nion)], dtype=np.int32)
        gsi = np.asarray([[grid_subset_index] * int(nion)], dtype=np.int32)
        vshape = np.tile(shp.reshape(1, 1, 3), (1, int(nion), 1))


        # DD-native tokenization (edge_profiles 4.1.x): ggd[]&ion[]&velocity[]&{r,phi,z}
        base_native = "ggd[]&ion[]&velocity[]"
        _h5_write_dataset(g, f"{base_native}&AOS_SHAPE", aos, dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f"{base_native}&grid_index", gi, dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f"{base_native}&grid_subset_index", gsi, dtype=np.int32, overwrite=overwrite)
        try:
            v_native = np.asarray(values_all, dtype=np.float64).reshape((1, int(nion), -1), order='C')
        except Exception:
            v_native = np.asarray(values_all, dtype=np.float64)
        _h5_write_dataset(g, f"{base_native}&{comp}", v_native, dtype=np.float64, overwrite=overwrite)
        _h5_write_dataset(g, f"{base_native}&{comp}_SHAPE", vshape, dtype=np.int32, overwrite=overwrite)

        for base in bases:
            _h5_write_dataset(g, f"{base}&AOS_SHAPE", aos, dtype=np.int32, overwrite=overwrite)
            _h5_write_dataset(g, f"{base}&grid_index", gi, dtype=np.int32, overwrite=overwrite)
            _h5_write_dataset(g, f"{base}&grid_subset_index", gsi, dtype=np.int32, overwrite=overwrite)
            _h5_write_dataset(g, f"{base}&values", values_all, dtype=np.float64, overwrite=overwrite)
            _h5_write_dataset(g, f"{base}&values_SHAPE", vshape, dtype=np.int32, overwrite=overwrite)


# -----------------------------------------------------------------------------
# Additional edge_profiles GGD writers (HDF5 backend; h5py direct)
# -----------------------------------------------------------------------------

def _h5_write_dataset(g, name, data, *, dtype=None, overwrite=True, **kwargs):
    """Create or overwrite a dataset under HDF5 group g."""
    if name in g:
        if overwrite:
            del g[name]
        else:
            return
    if dtype is not None:
        data = np.asarray(data, dtype=dtype)
    g.create_dataset(name, data=data, **kwargs)


def _ids_backend_h5_loc(entry_dir: str, ids_name: str, occ: int):
    """Return (h5_path, group_name) for an IDS occurrence in the HDF5 backend.

    IMAS convention: occurrence 0 typically uses <ids>.h5 and group /<ids> (no '_0' suffix).
    Some older tooling writes <ids>_0.h5 and /<ids>_0. We prefer the standard path but
    fall back to an existing alternative if present.
    """
    import os
    occ = int(occ)
    ids_name = str(ids_name)
    if occ == 0:
        cands = [
            (os.path.join(entry_dir, f"{ids_name}.h5"), f"/{ids_name}"),
            (os.path.join(entry_dir, f"{ids_name}_0.h5"), f"/{ids_name}_0"),
        ]
    else:
        cands = [
            (os.path.join(entry_dir, f"{ids_name}_{occ}.h5"), f"/{ids_name}_{occ}"),
            # Rare layout: all occurrences in a single file
            (os.path.join(entry_dir, f"{ids_name}.h5"), f"/{ids_name}_{occ}"),
        ]
    for p, g in cands:
        if os.path.exists(p):
            return p, g
    return cands[0]



def _write_edge_profiles_electrons_pressure_ggd_h5(entry_dir, occ, *, values_1d, shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True):
    base = 'ggd[]&electrons&pressure[]'
    import h5py
    fpath = os.path.join(entry_dir, f'edge_profiles_{occ}.h5')
    with h5py.File(fpath, 'a') as f:
        g = f[f'edge_profiles_{occ}']
        _h5_write_dataset(g, f'{base}&AOS_SHAPE', np.array([[1]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_index', np.array([[grid_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_subset_index', np.array([[grid_subset_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values', np.asarray(values_1d, dtype=np.float64).reshape((1, 1, -1)), dtype=np.float64, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values_SHAPE', np.array([[list(shape_rzp)]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)


def _write_edge_profiles_t_i_average_ggd_h5(entry_dir, occ, *, values_1d, shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True):
    base = 'ggd[]&t_i_average[]'
    import h5py
    fpath = os.path.join(entry_dir, f'edge_profiles_{occ}.h5')
    with h5py.File(fpath, 'a') as f:
        g = f[f'edge_profiles_{occ}']
        _h5_write_dataset(g, f'{base}&AOS_SHAPE', np.array([[1]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_index', np.array([[grid_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_subset_index', np.array([[grid_subset_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values', np.asarray(values_1d, dtype=np.float64).reshape((1, 1, -1)), dtype=np.float64, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values_SHAPE', np.array([[list(shape_rzp)]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)


def _write_edge_profiles_ion_scalar_ggd_h5(entry_dir, occ, *, field, values_all_1d, shape_rzp, nion=1, grid_index=1, grid_subset_index=0, overwrite=True):
    # field in {'density','pressure','temperature'}
    base = f'ggd[]&ion[]&{field}[]'
    import h5py
    fpath = os.path.join(entry_dir, f'edge_profiles_{occ}.h5')
    with h5py.File(fpath, 'a') as f:
        g = f[f'edge_profiles_{occ}']
        _h5_write_dataset(g, 'ggd[]&ion[]&AOS_SHAPE', np.array([[1, int(nion)]], dtype=np.int32), dtype=np.int32, overwrite=False)
        _h5_write_dataset(g, f'{base}&AOS_SHAPE', np.array([[1, int(nion)]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_index', np.full((1, int(nion)), int(grid_index), dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_subset_index', np.full((1, int(nion)), int(grid_subset_index), dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values', np.asarray(values_all_1d, dtype=np.float64).reshape((1, int(nion), -1)), dtype=np.float64, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values_SHAPE', np.tile(np.array([[list(shape_rzp)]], dtype=np.int32), (1, int(nion), 1)), dtype=np.int32, overwrite=overwrite)


def _write_edge_profiles_ion_z_ion_ggd_h5(entry_dir, occ, *, z_ions, overwrite=True):
    base = 'ggd[]&ion[]&z_ion'
    import h5py
    fpath = os.path.join(entry_dir, f'edge_profiles_{occ}.h5')
    z = np.asarray(z_ions)
    if z.ndim == 0:
        z = z.reshape((1,))
    z = z.astype(np.int32, copy=False)
    nion = int(z.size)
    with h5py.File(fpath, 'a') as f:
        g = f[f'edge_profiles_{occ}']
        _h5_write_dataset(g, 'ggd[]&ion[]&AOS_SHAPE', np.array([[1, int(nion)]], dtype=np.int32), dtype=np.int32, overwrite=False)
        _h5_write_dataset(g, base, z.reshape((1, int(nion))), dtype=np.int32, overwrite=overwrite)
def _write_edge_profiles_zeff_ggd_h5(entry_dir, occ, *, values_1d, shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True):
    """Write edge_profiles.ggd zeff values (DD: ggd(itime)/zeff(i1)/values)."""
    import h5py
    base = 'ggd[]&zeff[]'
    fpath = os.path.join(entry_dir, f'edge_profiles_{occ}.h5')
    with h5py.File(fpath, 'a') as f:
        g = f[f'edge_profiles_{occ}']
        _h5_write_dataset(g, f'{base}&AOS_SHAPE', np.array([[1]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_index', np.array([[grid_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_subset_index', np.array([[grid_subset_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values', np.asarray(values_1d, dtype=np.float64).reshape((1, 1, -1)), dtype=np.float64, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values_SHAPE', np.array([[list(shape_rzp)]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)


def _write_edge_profiles_n_i_total_over_n_e_ggd_h5(entry_dir, occ, *, values_1d, shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True):
    """Write edge_profiles.ggd n_i_total_over_n_e (DD: ggd(itime)/n_i_total_over_n_e(i1)/values)."""
    import h5py
    base = 'ggd[]&n_i_total_over_n_e[]'
    fpath = os.path.join(entry_dir, f'edge_profiles_{occ}.h5')
    with h5py.File(fpath, 'a') as f:
        g = f[f'edge_profiles_{occ}']
        _h5_write_dataset(g, f'{base}&AOS_SHAPE', np.array([[1]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_index', np.array([[grid_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_subset_index', np.array([[grid_subset_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values', np.asarray(values_1d, dtype=np.float64).reshape((1, 1, -1)), dtype=np.float64, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values_SHAPE', np.array([[list(shape_rzp)]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)

        # Common alias used in some downstream scripts
        base2 = 'ggd[]&ni_total_over_ne[]'
        _h5_write_dataset(g, f'{base2}&AOS_SHAPE', np.array([[1]], dtype=np.int32), dtype=np.int32, overwrite=False)
        _h5_write_dataset(g, f'{base2}&grid_index', np.array([[grid_index]], dtype=np.int32), dtype=np.int32, overwrite=False)
        _h5_write_dataset(g, f'{base2}&grid_subset_index', np.array([[grid_subset_index]], dtype=np.int32), dtype=np.int32, overwrite=False)
        _h5_write_dataset(g, f'{base2}&values', np.asarray(values_1d, dtype=np.float64).reshape((1, 1, -1)), dtype=np.float64, overwrite=False)
        _h5_write_dataset(g, f'{base2}&values_SHAPE', np.array([[list(shape_rzp)]], dtype=np.int32), dtype=np.int32, overwrite=False)


def _write_edge_profiles_j_total_component_ggd_h5(entry_dir, occ, *, component, values_1d, shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True):
    """Write edge_profiles.ggd j_total component.

    DD defines ggd(itime)/j_total(i1)/{r,phi,z}(:) among other components.
    Here we populate only cylindrical components available directly from NIMROD.
    """
    import h5py
    comp = str(component).lower().strip()
    if comp not in ('r', 'z', 'phi'):
        raise ValueError(f'component must be r|z|phi, got {component!r}')
    base = 'ggd[]&j_total[]'
    fpath = os.path.join(entry_dir, f'edge_profiles_{occ}.h5')
    with h5py.File(fpath, 'a') as f:
        g = f[f'edge_profiles_{occ}']
        _h5_write_dataset(g, f'{base}&AOS_SHAPE', np.array([[1]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_index', np.array([[grid_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_subset_index', np.array([[grid_subset_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&{comp}', np.asarray(values_1d, dtype=np.float64).reshape((1, 1, -1)), dtype=np.float64, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&{comp}_SHAPE', np.array([[list(shape_rzp)]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)

# -----------------------------------------------------------------------------
# Additional core_profiles GGD writers (HDF5 backend; h5py direct)
# -----------------------------------------------------------------------------

def _write_core_profiles_electrons_temperature_ggd_h5(entry_dir, occ, *, values_1d, shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True):
    import h5py
    base = 'ggd[]&electrons&temperature[]'
    fpath, grp = _ids_backend_h5_loc(entry_dir, 'core_profiles', occ)
    with h5py.File(fpath, 'a') as f:
        if grp not in f:
            f.create_group(grp)
        g = f[grp]
        _h5_write_dataset(g, f'{base}&AOS_SHAPE', np.array([[1]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_index', np.array([[grid_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_subset_index', np.array([[grid_subset_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values', np.asarray(values_1d, dtype=np.float64).reshape((1, 1, -1)), dtype=np.float64, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values_SHAPE', np.array([[list(shape_rzp)]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)

def _write_core_profiles_electrons_density_ggd_h5(entry_dir, occ, *, values_1d, shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True):
    import h5py
    base = 'ggd[]&electrons&density[]'
    fpath, grp = _ids_backend_h5_loc(entry_dir, 'core_profiles', occ)
    with h5py.File(fpath, 'a') as f:
        if grp not in f:
            f.create_group(grp)
        g = f[grp]
        _h5_write_dataset(g, f'{base}&AOS_SHAPE', np.array([[1]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_index', np.array([[grid_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_subset_index', np.array([[grid_subset_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values', np.asarray(values_1d, dtype=np.float64).reshape((1, 1, -1)), dtype=np.float64, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values_SHAPE', np.array([[list(shape_rzp)]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)

def _write_core_profiles_electrons_pressure_ggd_h5(entry_dir, occ, *, values_1d, shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True):
    import h5py
    base = 'ggd[]&electrons&pressure[]'
    fpath, grp = _ids_backend_h5_loc(entry_dir, 'core_profiles', occ)
    with h5py.File(fpath, 'a') as f:
        if grp not in f:
            f.create_group(grp)
        g = f[grp]
        _h5_write_dataset(g, f'{base}&AOS_SHAPE', np.array([[1]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_index', np.array([[grid_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_subset_index', np.array([[grid_subset_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values', np.asarray(values_1d, dtype=np.float64).reshape((1, 1, -1)), dtype=np.float64, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values_SHAPE', np.array([[list(shape_rzp)]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)

def _write_core_profiles_t_i_average_ggd_h5(entry_dir, occ, *, values_1d, shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True):
    import h5py
    base = 'ggd[]&t_i_average[]'
    fpath, grp = _ids_backend_h5_loc(entry_dir, 'core_profiles', occ)
    with h5py.File(fpath, 'a') as f:
        if grp not in f:
            f.create_group(grp)
        g = f[grp]
        _h5_write_dataset(g, f'{base}&AOS_SHAPE', np.array([[1]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_index', np.array([[grid_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_subset_index', np.array([[grid_subset_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values', np.asarray(values_1d, dtype=np.float64).reshape((1, 1, -1)), dtype=np.float64, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values_SHAPE', np.array([[list(shape_rzp)]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)

def _write_core_profiles_ion_scalar_ggd_h5(entry_dir, occ, *, field, values_all_1d, shape_rzp, nion=1, grid_index=1, grid_subset_index=0, overwrite=True):
    # field in {'density','pressure','temperature'}
    base = f'ggd[]&ion[]&{field}[]'
    import h5py
    fpath, grp = _ids_backend_h5_loc(entry_dir, 'core_profiles', occ)
    with h5py.File(fpath, 'a') as f:
        if grp not in f:
            f.create_group(grp)
        g = f[grp]
        _h5_write_dataset(g, 'ggd[]&ion[]&AOS_SHAPE', np.array([[1, int(nion)]], dtype=np.int32), dtype=np.int32, overwrite=False)
        _h5_write_dataset(g, f'{base}&AOS_SHAPE', np.array([[1, int(nion)]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_index', np.full((1, int(nion)), int(grid_index), dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_subset_index', np.full((1, int(nion)), int(grid_subset_index), dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values', np.asarray(values_all_1d, dtype=np.float64).reshape((1, int(nion), -1)), dtype=np.float64, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values_SHAPE', np.tile(np.array([[list(shape_rzp)]], dtype=np.int32), (1, int(nion), 1)), dtype=np.int32, overwrite=overwrite)

def _write_core_profiles_ion_z_ion_ggd_h5(entry_dir, occ, *, z_ions, overwrite=True):
    base = 'ggd[]&ion[]&z_ion'
    import h5py
    fpath, grp = _ids_backend_h5_loc(entry_dir, 'core_profiles', occ)
    z = np.asarray(z_ions)
    if z.ndim == 0:
        z = z.reshape((1,))
    z = z.astype(np.int32, copy=False)
    with h5py.File(fpath, 'a') as f:
        if grp not in f:
            f.create_group(grp)
        g = f[grp]
        _h5_write_dataset(g, 'ggd[]&ion[]&AOS_SHAPE', np.array([[1, int(z.size)]], dtype=np.int32), dtype=np.int32, overwrite=False)
        _h5_write_dataset(g, f'{base}&AOS_SHAPE', np.array([[1, int(z.size)]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, base, z.reshape((1, int(z.size))), dtype=np.int32, overwrite=overwrite)

def _write_core_profiles_zeff_ggd_h5(entry_dir, occ, *, values_1d, shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True):
    """Write core_profiles.ggd zeff values (DD: ggd(itime)/zeff(i1)/values)."""
    import h5py
    base = 'ggd[]&zeff[]'
    fpath, grp = _ids_backend_h5_loc(entry_dir, 'core_profiles', occ)
    with h5py.File(fpath, 'a') as f:
        if grp not in f:
            f.create_group(grp)
        g = f[grp]
        _h5_write_dataset(g, f'{base}&AOS_SHAPE', np.array([[1]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_index', np.array([[grid_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_subset_index', np.array([[grid_subset_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values', np.asarray(values_1d, dtype=np.float64).reshape((1, 1, -1)), dtype=np.float64, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values_SHAPE', np.array([[list(shape_rzp)]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)

def _write_core_profiles_n_i_total_over_n_e_ggd_h5(entry_dir, occ, *, values_1d, shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True):
    import h5py
    base = 'ggd[]&n_i_total_over_n_e[]'
    fpath, grp = _ids_backend_h5_loc(entry_dir, 'core_profiles', occ)
    with h5py.File(fpath, 'a') as f:
        if grp not in f:
            f.create_group(grp)
        g = f[grp]
        _h5_write_dataset(g, f'{base}&AOS_SHAPE', np.array([[1]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_index', np.array([[grid_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_subset_index', np.array([[grid_subset_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values', np.asarray(values_1d, dtype=np.float64).reshape((1, 1, -1)), dtype=np.float64, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&values_SHAPE', np.array([[list(shape_rzp)]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)

def _write_core_profiles_j_total_component_ggd_h5(entry_dir, occ, *, component, values_1d, shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True):
    import h5py
    comp = str(component).lower().strip()
    if comp not in ('r', 'z', 'phi'):
        raise ValueError(f'component must be r|z|phi, got {component!r}')
    base = 'ggd[]&j_total[]'
    fpath, grp = _ids_backend_h5_loc(entry_dir, 'core_profiles', occ)
    with h5py.File(fpath, 'a') as f:
        if grp not in f:
            f.create_group(grp)
        g = f[grp]
        _h5_write_dataset(g, f'{base}&AOS_SHAPE', np.array([[1]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_index', np.array([[grid_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&grid_subset_index', np.array([[grid_subset_index]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&{comp}', np.asarray(values_1d, dtype=np.float64).reshape((1, 1, -1)), dtype=np.float64, overwrite=overwrite)
        _h5_write_dataset(g, f'{base}&{comp}_SHAPE', np.array([[list(shape_rzp)]], dtype=np.int32), dtype=np.int32, overwrite=overwrite)

def _write_core_profiles_ion_velocity_component_ggd_h5(entry_dir, occ, *, component, values_1d, shape_rzp,
                                                   nion=1, grid_index=1, grid_subset_index=0, overwrite=True):
    """Write core_profiles.ggd ion velocity component datasets directly via h5py.

    Mirrors edge_profiles writer and writes DD-native tokenization:
      - ggd[]&ion[]&velocity[]&{r,phi,z}
    plus legacy aliases:
      - ggd[]&ion[]&velocity&<comp>[]&...
      - ggd[]&ion[]&velocity_<comp>[]&...
    """
    import h5py
    comp = str(component).lower().strip()
    if comp not in ("r", "z", "phi"):
        raise ValueError(f"component must be one of r,z,phi; got {component!r}")

    values_1d = np.asarray(values_1d, dtype=np.float64).ravel(order='F')
    shp = np.asarray(shape_rzp, dtype=np.int32).ravel(order='C')
    if shp.size != 3:
        raise ValueError(f"shape_rzp must be (nr,nz,nphi); got {shape_rzp}")

    if int(nion) < 1:
        nion = 1

    values_all = np.tile(values_1d, int(nion))

    bases = [f"ggd[]&ion[]&velocity&{comp}[]", f"ggd[]&ion[]&velocity_{comp}[]"]

    h5_path, grp = _ids_backend_h5_loc(entry_dir, "core_profiles", occ)

    with h5py.File(h5_path, 'a') as h5:
        if grp not in h5:
            h5.create_group(grp)
        g = h5[grp]

        aos = np.asarray([[1, int(nion)]], dtype=np.int32)
        gi = np.asarray([[grid_index] * int(nion)], dtype=np.int32)
        gsi = np.asarray([[grid_subset_index] * int(nion)], dtype=np.int32)
        vshape = np.tile(shp.reshape(1, 1, 3), (1, int(nion), 1))

        base_native = "ggd[]&ion[]&velocity[]"
        _h5_write_dataset(g, f"{base_native}&AOS_SHAPE", aos, dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f"{base_native}&grid_index", gi, dtype=np.int32, overwrite=overwrite)
        _h5_write_dataset(g, f"{base_native}&grid_subset_index", gsi, dtype=np.int32, overwrite=overwrite)
        v_native = np.asarray(values_all, dtype=np.float64).reshape((1, int(nion), -1), order='C')
        _h5_write_dataset(g, f"{base_native}&{comp}", v_native, dtype=np.float64, overwrite=overwrite)
        _h5_write_dataset(g, f"{base_native}&{comp}_SHAPE", vshape, dtype=np.int32, overwrite=overwrite)

        # Also write legacy aliases for compatibility with older plotters
        for base in bases:
            _h5_write_dataset(g, f"{base}&AOS_SHAPE", aos, dtype=np.int32, overwrite=overwrite)
            _h5_write_dataset(g, f"{base}&grid_index", gi, dtype=np.int32, overwrite=overwrite)
            _h5_write_dataset(g, f"{base}&grid_subset_index", gsi, dtype=np.int32, overwrite=overwrite)
            _h5_write_dataset(g, f"{base}&values", v_native, dtype=np.float64, overwrite=overwrite)
            _h5_write_dataset(g, f"{base}&values_SHAPE", vshape, dtype=np.int32, overwrite=overwrite)


def populate_core_profiles_ggd(cp: Any, data: Dict[str, Any], args) -> None:
    """Populate core_profiles GGD using the same equilibrium-only strategy as edge_profiles.

    This ensures core_profiles and edge_profiles export the same 2D equilibrium quantities in GGD form
    (when the DD provides ggd/grid_ggd containers).
    """
    try:
        # Reuse the edge_profiles implementation: it only assumes .ggd and .grid_ggd exist.
        populate_edge_profiles_ggd(cp, data, args)
    except Exception:
        # Never fail the conversion on optional GGD population
        return


def _patch_core_profiles_ggd_required_leaves_h5(entry_dir: str, occ: int, data: Dict[str, Any], args, log: "logging.Logger") -> None:
    """Ensure core_profiles GGD leaves needed for tooling are present and (optionally) write full equilibrium fields.

    - Always write grid_ggd.space geometry vectors (R,Z,phi) to the backend if we can infer node coordinates.
    - If --ggd-unstructured is enabled, write the full set of equilibrium fields (electrons/ions + j_total + flows)
      directly to the backend using the native FE-node ordering, preserving the original node structure.
    """
    import numpy as np

    nb = max(4, int(getattr(args, 'ggd_nbins', 128) or 128))
    conn_kind = str(getattr(args, 'ggd_connectivity', 'none')).lower()

    use_unstructured_nodes = (
        bool(getattr(args, 'ggd_unstructured', False))
        and bool(getattr(args, 'ggd_unstructured_fe_nodes', False))
        and (conn_kind in ('fe_tri', 'fe_wedge', 'fe_pointcloud'))
    )

    use_fe_hex = (
        bool(getattr(args, 'ggd_unstructured', False))
        and (conn_kind == 'hex')
    )

    # Node coordinate vectors for grid_ggd.space geometry
    try:
        if use_unstructured_nodes:
            Rbase = np.asarray(data.get('R'), dtype=float)
            Zbase = np.asarray(data.get('Z'), dtype=float)
            r_nodes = Rbase.ravel(order='F')
            z_nodes = Zbase.ravel(order='F')
            phi_nodes = np.zeros_like(r_nodes)
        else:
            Rm = np.asarray(data.get('R'), dtype=float)
            Zm = np.asarray(data.get('Z'), dtype=float)
            Vref = data.get('psi_eq', None)
            if Vref is None:
                Vref = data.get('teq', None)
            if Vref is None:
                Vref = Rm
            rc, zc, _ = _interp2d_linear(Rm, Zm, np.asarray(Vref, dtype=float), nbins=int(nb), log=log)
            r_nodes = rc.ravel(order='F')
            z_nodes = zc.ravel(order='F')
            phi_nodes = np.zeros_like(r_nodes)
        _write_gridggd_space_geometry_vectors_h5(str(entry_dir), 'core_profiles', int(occ), r_nodes, z_nodes, phi_nodes, log=log)
    except Exception as e:
        log.debug('core_profiles grid_ggd geometry vectors patch skipped: %s', e)

    # Only full-field patching in unstructured mode (preserve FE nodes ordering)
    if not bool(getattr(args, 'ggd_unstructured', False)):
        return

    # For unstructured hex connectivity, we currently still store values on the product (R,Z) grid.
    # Preserve the original FE ordering only for the FE-node cases above.
    if (not use_unstructured_nodes) and (not use_fe_hex):
        return

    try:
        nq = data.get('nq', None)
        peq = data.get('peq', None)
        teq = data.get('teq', None)
        prq = data.get('prq', None)
        tiq = data.get('tiq', None)
        vq = data.get('vq', None)
        jq = data.get('jq', None)

        sp = getattr(args, '_nimrod_species', {}) or {}
        qe = float(sp.get('qe_c', 1.602176634e-19))
        n_scale = float(getattr(args, 'n_scale', 1.0) or 1.0)
        z_ions = sp.get('z_ions', None)

        def _to_grid(a):
            if a is None:
                return None
            A = np.asarray(a, dtype=float)
            if use_unstructured_nodes:
                # Already node-centered in stitched FE order
                return A
            # Product grid downsample
            try:
                Rm = np.asarray(data.get('R'), dtype=float)
                Zm = np.asarray(data.get('Z'), dtype=float)
                rc, zc, val = _interp2d_linear(Rm, Zm, A, nbins=int(nb), log=log)
                return val
            except Exception:
                return None

        # Establish shape_rzp for values_SHAPE (nr,nz,nphi). Always nphi=1 here.
        if use_unstructured_nodes:
            nnode = int(np.asarray(data.get('R')).size)
            shape_rzp = (nnode, 1, 1)
        else:
            Rm = np.asarray(data.get('R'), dtype=float)
            Zm = np.asarray(data.get('Z'), dtype=float)
            Vref = data.get('psi_eq', None)
            if Vref is None:
                Vref = data.get('teq', None)
            if Vref is None:
                Vref = Rm
            rc, zc, _ = _interp2d_linear(Rm, Zm, np.asarray(Vref, dtype=float), nbins=int(nb), log=log)
            shape_rzp = (int(rc.shape[0]), int(rc.shape[1]), 1)

        # electrons
        nqA = np.asarray(nq, dtype=float) if nq is not None else None
        if nqA is not None:
            nqA = nqA * float(n_scale)

        ne2d = None
        if nqA is not None:
            if nqA.ndim == 2:
                ne2d = nqA
            elif nqA.ndim >= 3 and nqA.shape[-1] >= 1:
                ne2d = nqA[..., 0]
        ne2d = _to_grid(ne2d)

        if ne2d is not None:
            _write_core_profiles_electrons_density_ggd_h5(str(entry_dir), occ,
                                                         values_1d=np.asarray(ne2d, dtype=float).ravel(order='F'),
                                                         shape_rzp=shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True)

        pe2d = _to_grid(peq)
        if pe2d is not None:
            _write_core_profiles_electrons_pressure_ggd_h5(str(entry_dir), occ,
                                                          values_1d=np.asarray(pe2d, dtype=float).ravel(order='F'),
                                                          shape_rzp=shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True)

        te2d = _to_grid(teq)
        if te2d is None and (pe2d is not None) and (ne2d is not None):
            with np.errstate(divide='ignore', invalid='ignore'):
                te2d = np.asarray(pe2d, dtype=float) / (np.asarray(ne2d, dtype=float) * qe)
        if te2d is not None:
            _write_core_profiles_electrons_temperature_ggd_h5(str(entry_dir), occ,
                                                             values_1d=np.asarray(te2d, dtype=float).ravel(order='F'),
                                                             shape_rzp=shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True)

        # ions
        pi2d = None
        if (prq is not None) and (peq is not None):
            try:
                pi2d = _to_grid(np.asarray(prq, dtype=float) - np.asarray(peq, dtype=float))
            except Exception:
                pi2d = None

        nqA = np.asarray(nq, dtype=float) if nq is not None else None
        if nqA is not None:
            nqA = nqA * float(n_scale)

        nion = 1
        if nqA is not None and nqA.ndim >= 3 and nqA.shape[-1] >= 2:
            nion = int(nqA.shape[-1] - 1)
        elif z_ions is not None:
            try:
                nion = max(1, int(np.size(z_ions)))
            except Exception:
                nion = 1

        dens_list = []
        if nqA is not None and nqA.ndim >= 3 and nqA.shape[-1] >= 2:
            for k in range(nion):
                dens_list.append(_to_grid(np.asarray(nqA[..., 1 + k], dtype=float)))
        elif ne2d is not None:
            dens_list = [np.asarray(ne2d, dtype=float)]
        else:
            dens_list = [np.full(shape_rzp[:2], np.nan, dtype=float)]

        # t_i_average
        ti2d = _to_grid(tiq)
        if ti2d is None and (pi2d is not None):
            ni_total = np.zeros(shape_rzp[:2], dtype=float)
            for a in dens_list:
                ni_total += np.asarray(a, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                ti2d = np.asarray(pi2d, dtype=float) / (ni_total * qe)
        if ti2d is not None:
            _write_core_profiles_t_i_average_ggd_h5(str(entry_dir), occ,
                                                   values_1d=np.asarray(ti2d, dtype=float).ravel(order='F'),
                                                   shape_rzp=shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True)
        else:
            ti2d = np.full(shape_rzp[:2], np.nan, dtype=float)

        pres_list = []
        if pi2d is not None:
            ni_total = np.zeros(shape_rzp[:2], dtype=float)
            for a in dens_list:
                ni_total += np.asarray(a, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                for a in dens_list:
                    frac = np.asarray(a, dtype=float) / ni_total
                    pres_list.append(np.asarray(pi2d, dtype=float) * frac)
        else:
            pres_list = [np.full(shape_rzp[:2], np.nan, dtype=float) for _ in dens_list]

        def _cat(vals):
            return np.concatenate([np.asarray(v, dtype=float).ravel(order='F') for v in vals], axis=0)

        _write_core_profiles_ion_scalar_ggd_h5(str(entry_dir), occ, field='density',
                                              values_all_1d=_cat(dens_list), shape_rzp=shape_rzp, nion=len(dens_list),
                                              grid_index=1, grid_subset_index=0, overwrite=True)
        _write_core_profiles_ion_scalar_ggd_h5(str(entry_dir), occ, field='pressure',
                                              values_all_1d=_cat(pres_list), shape_rzp=shape_rzp, nion=len(pres_list),
                                              grid_index=1, grid_subset_index=0, overwrite=True)
        _write_core_profiles_ion_scalar_ggd_h5(str(entry_dir), occ, field='temperature',
                                              values_all_1d=np.tile(np.asarray(ti2d, dtype=float).ravel(order='F'), len(dens_list)),
                                              shape_rzp=shape_rzp, nion=len(dens_list), grid_index=1, grid_subset_index=0, overwrite=True)

        # ion velocity (replicate bulk v for all ions)
        if vq is not None:
            vqA = np.asarray(vq, dtype=float)
            if vqA.ndim >= 3 and vqA.shape[-1] >= 3:
                vR = _to_grid(vqA[..., 0])
                vZ = _to_grid(vqA[..., 1])
                vP = _to_grid(vqA[..., 2])
                _write_core_profiles_ion_velocity_component_ggd_h5(str(entry_dir), occ, component='r',
                                                                  values_1d=np.asarray(vR, dtype=float).ravel(order='F'),
                                                                  shape_rzp=shape_rzp, nion=len(dens_list),
                                                                  grid_index=1, grid_subset_index=0, overwrite=True)
                _write_core_profiles_ion_velocity_component_ggd_h5(str(entry_dir), occ, component='z',
                                                                  values_1d=np.asarray(vZ, dtype=float).ravel(order='F'),
                                                                  shape_rzp=shape_rzp, nion=len(dens_list),
                                                                  grid_index=1, grid_subset_index=0, overwrite=True)
                _write_core_profiles_ion_velocity_component_ggd_h5(str(entry_dir), occ, component='phi',
                                                                  values_1d=np.asarray(vP, dtype=float).ravel(order='F'),
                                                                  shape_rzp=shape_rzp, nion=len(dens_list),
                                                                  grid_index=1, grid_subset_index=0, overwrite=True)

        if z_ions is None:
            z_ions = np.arange(1, 1 + len(dens_list), dtype=np.int32)
        _write_core_profiles_ion_z_ion_ggd_h5(str(entry_dir), occ, z_ions=z_ions, overwrite=True)

        # zeff and ni/ne
        try:
            if ne2d is None:
                ne_use = np.zeros_like(ni_total, dtype=float)
                if z_ions is None or len(z_ions) != len(dens_list):
                    z_use = [1.0] * len(dens_list)
                else:
                    z_use = [float(z) for z in z_ions]
                for _n, _z in zip(dens_list, z_use):
                    ne_use += _z * np.asarray(_n, dtype=float)
            else:
                ne_use = np.asarray(ne2d, dtype=float)

            ni_total = np.zeros(shape_rzp[:2], dtype=float)
            for a in dens_list:
                ni_total += np.asarray(a, dtype=float)

            with np.errstate(divide='ignore', invalid='ignore'):
                ni_over_ne = np.where(ne_use != 0.0, np.asarray(ni_total, dtype=float) / ne_use, np.nan)
                if z_ions is None or len(z_ions) != len(dens_list):
                    zeff2d = np.ones_like(ni_over_ne, dtype=float)
                else:
                    num = np.zeros_like(ne_use, dtype=float)
                    for _n, _z in zip(dens_list, [float(z) for z in z_ions]):
                        num += (_z * _z) * np.asarray(_n, dtype=float)
                    zeff2d = np.where(ne_use != 0.0, num / ne_use, np.nan)
                if z_ions is not None and len(z_ions) == 1 and int(z_ions[0]) == 1:
                    zeff2d = np.ones_like(ni_over_ne, dtype=float)

            _write_core_profiles_zeff_ggd_h5(str(entry_dir), occ,
                                             values_1d=zeff2d.ravel(order='F'),
                                             shape_rzp=shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True)
            _write_core_profiles_n_i_total_over_n_e_ggd_h5(str(entry_dir), occ,
                                             values_1d=ni_over_ne.ravel(order='F'),
                                             shape_rzp=shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True)
        except Exception:
            pass

        # j_total components from jq
        try:
            if jq is not None and hasattr(jq, 'shape') and int(jq.shape[-1]) >= 3:
                jR2d = _to_grid(np.asarray(jq[..., 0], dtype=float))
                jZ2d = _to_grid(np.asarray(jq[..., 1], dtype=float))
                jP2d = _to_grid(np.asarray(jq[..., 2], dtype=float))
                _write_core_profiles_j_total_component_ggd_h5(str(entry_dir), occ, component='r',
                                                             values_1d=np.asarray(jR2d, dtype=float).ravel(order='F'),
                                                             shape_rzp=shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True)
                _write_core_profiles_j_total_component_ggd_h5(str(entry_dir), occ, component='z',
                                                             values_1d=np.asarray(jZ2d, dtype=float).ravel(order='F'),
                                                             shape_rzp=shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True)
                _write_core_profiles_j_total_component_ggd_h5(str(entry_dir), occ, component='phi',
                                                             values_1d=np.asarray(jP2d, dtype=float).ravel(order='F'),
                                                             shape_rzp=shape_rzp, grid_index=1, grid_subset_index=0, overwrite=True)
        except Exception:
            pass

    except Exception as e:
        log.warning('core_profiles GGD required leaves write failed: %s', e)


def populate_edge_profiles_ggd(ep: Any, data: Dict[str, Any], args) -> None:
    """Populate edge_profiles GGD (equilibrium-only).

    We downsample stitched RZ fields onto a regular grid using _bin2d_avg and store
    a minimal, portable set of equilibrium quantities in ep.ggd/ep.grid_ggd.

    Quantities written (when present in dumpgll):
      - electrons.temperature (from teq or from peq/nq)
      - t_i_average          (from tiq or from (prq-peq)/sum_i nq_i)
      - n_i_total            (sum of nq over ion species)
    """
    # Require GGD containers on this DD/build.
    if not hasattr(ep, "ggd") or not hasattr(ep, "grid_ggd"):
        return

    t = float(data["time"])
    R = np.asarray(data.get("R"), dtype=float) if data.get("R") is not None else None
    Z = np.asarray(data.get("Z"), dtype=float) if data.get("Z") is not None else None
    if R is None or Z is None:
        return

    pr2d = data.get("prq", None)
    pe2d = data.get("peq", None)
    nq = data.get("nq", None)
    te2d = data.get("teq", None)
    ti2d = data.get("tiq", None)
    vq = data.get("vq", None)  # velocity (m/s)
    jq = data.get("jq", None)  # current density (A/m^2)
    # Optional: override equilibrium GGD electron profiles from peqdsk when requested.
    # Optional: override *temperature only* from PEQDSK when requested.
    # Electron density is ALWAYS taken directly from dumpgll (nq), for both impurity and no-impurity cases.
    edge_mode = str(getattr(args, 'edge_ggd_values', '') or '').strip().lower()
    if edge_mode == 'equilibrium':
        psi2d = data.get('psi_eq', None)
        psi_axis = psi_lcfs = None
        if psi2d is not None:
            try:
                _log = logging.getLogger(__name__) if log is None else log
                psi_axis, psi_lcfs, _tag = _choose_psi_axis_lcfs(data, args, log=_log)
                if np.isfinite(psi_axis) and np.isfinite(psi_lcfs) and abs(float(psi_lcfs) - float(psi_axis)) > 1e-12:
                    den = float(psi_lcfs - psi_axis)
                    psn2d = (np.asarray(psi2d, dtype=float) - float(psi_axis)) / den
                    peqdsk_path = _resolve_optional_file(args, 'peqdsk', 'peqdsk')
                    prof = _peqdsk_te_ne_si(peqdsk_path, log=_log) if peqdsk_path else None
                    if prof is not None:
                        ps1d, te_pe_ev, _ne_pe_m3 = prof
                        psn_clip = np.clip(psn2d, float(ps1d[0]), float(ps1d[-1]))
                        te2d = np.interp(psn_clip.ravel(order='F'), ps1d, te_pe_ev).reshape(psn_clip.shape, order='F')
            except Exception:
                pass


    # Bin configuration (reuse mhd GGD knobs)
    nb = max(4, int(getattr(args, "ggd_nbins", 128) or 128))
    nphi = 1  # equilibrium-only export (no toroidal reconstruction)

    conn_kind = str(getattr(args, "ggd_connectivity", "none") or "none").strip().lower()
    # When using h5py-direct unstructured export, edge_profiles GGD is written directly to the backend file
    # after db.put_slice(). Keep the in-memory IDS minimal to avoid schema validation issues.
    if (
        bool(getattr(args, "ggd_unstructured", False))
        and bool(getattr(args, "ggd_unstructured_fe_nodes", False))
        and conn_kind in ("fe_tri", "fe_wedge", "fe_pointcloud")
    ):
        return


    use_fe_nodes = (

        bool(getattr(args, "ggd_unstructured", False))
        and bool(getattr(args, "ggd_unstructured_fe_nodes", False))

        and conn_kind in ("fe_tri", "fe_wedge", "fe_pointcloud")

    )

    if use_fe_nodes:
        it, ig = _append_time_ggd(ep, t, write_grid=_ggd_should_write_grid(args), reuse_grid=bool(getattr(args, "ggd_reuse_grid", False)))
        g = ep.grid_ggd[ig]
        try:
            g.identifier.name = "nimrod_fe_rz_nodes_tri"
            g.identifier.index = int(ig + 1)
            g.identifier.description = "Native stitched NIMROD FE nodes (R,Z); triangulated 2D connectivity; node-centered equilibrium fields"
        except Exception:
            pass

        Rloc = R
        Zloc = Z
        r_nodes = np.asarray(Rloc, dtype=float).ravel(order="F")
        z_nodes = np.asarray(Zloc, dtype=float).ravel(order="F")
        phi_nodes = np.zeros_like(r_nodes, dtype=float)
        if write_grid:
            _gridggd_write_node_vectors(g, r_nodes, z_nodes, phi_nodes)

        if conn_kind != "fe_pointcloud":
            mask2d = np.isfinite(np.asarray(Rloc, dtype=float)) & np.isfinite(np.asarray(Zloc, dtype=float))
            tri = _fe_tri_connectivity_from_mask(mask2d)

        if (conn_kind != "fe_pointcloud") and write_grid and _use_imas_connectivity_writer(args):
            # Slow, DD-aware path: populate grid_ggd.grid_subset connectivity using IMAS objects.
            try:
                nodes_xyz = np.stack((r_nodes, z_nodes, phi_nodes), axis=1).astype(np.float64, copy=False)
                _gridggd_write_unstructured_grid_subset_imas(g, nodes_xyz, tri, log=None)
            except Exception:
                # Non-fatal; downstream tools may still use the space geometry vectors.
                pass
        else:
            pass  # connectivity populated later via packed HDF5 writer (h5py)

        q = ep.ggd[it]

        # Compute equilibrium-only node-centered fields on FE nodes.
        sp = getattr(args, '_nimrod_species', {}) or {}
        qe = float(sp.get('qe_c', 1.602176634e-19))
        n_scale = float(getattr(args, "n_scale", 1.0) or 1.0)
        zeff_input = sp.get('zeff_input', None)

        nqA = np.asarray(nq, dtype=float) if nq is not None else None
        if nqA is not None:
            nqA = np.asarray(nqA, dtype=float) * float(n_scale)


        ne2d = None
        try:
            if nqA is not None:
                if nqA.ndim == 2:
                    ne2d = nqA
                elif nqA.ndim >= 3 and nqA.shape[2] >= 1:
                    ne2d = nqA[:, :, 0]
            if ne2d is not None and ne2d.shape != Rloc.shape and ne2d.T.shape == Rloc.shape:
                ne2d = ne2d.T
        except Exception:
            ne2d = None
        # electron temperature baseline
        te_eq = te2d

        # If PEQDSK equilibrium override was requested, keep pressures consistent with (ne,Te) if possible.
        try:
            edge_mode = str(getattr(args, 'edge_ggd_values', '') or '').strip().lower()
            if edge_mode == 'equilibrium' and te_eq is not None and ne2d is not None:
                pe2d = np.asarray(ne2d, dtype=float) * float(qe) * np.asarray(te_eq, dtype=float)
        except Exception:
            pass
        if te_eq is None and pe2d is not None and ne2d is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                te_eq = np.asarray(pe2d, dtype=float) / (np.asarray(ne2d, dtype=float) * qe)
            _log('edge_profiles(equilibrium): derived Te = peq/(ne*qe) (teq absent)')

        # ion density baseline
        ni2d_total = None
        if nqA is not None and nqA.ndim >= 3 and nqA.shape[2] >= 2:
            ni2d_total = np.nansum(nqA[:, :, 1:], axis=2)
        elif ne2d is not None:
            z = float(zeff_input) if zeff_input not in (None, '') else None
            if z is not None and z > 0.0:
                ni2d_total = np.asarray(ne2d, dtype=float) / z
                _log(f'edge_profiles(equilibrium): ion density fallback ni = ne/zeff_input (zeff_input={z:g})')
            else:
                ni2d_total = np.asarray(ne2d, dtype=float)
                _log('edge_profiles(equilibrium): ion density fallback ni = ne (zeff_input unavailable)')
        if ni2d_total is not None and ni2d_total.shape != Rloc.shape and ni2d_total.T.shape == Rloc.shape:
            ni2d_total = ni2d_total.T

        # pressures and derived Ti
        pi2d = None
        if pr2d is not None and pe2d is not None:
            pi2d = np.asarray(pr2d, dtype=float) - np.asarray(pe2d, dtype=float)
            _log('edge_profiles(equilibrium): computed p_i = p_total - p_e')
        ti_eq = ti2d
        if ti_eq is None and pi2d is not None and ni2d_total is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ti_eq = np.asarray(pi2d, dtype=float) / (np.asarray(ni2d_total, dtype=float) * qe)
            _log('edge_profiles(equilibrium): derived Ti = p_i/(n_i*qe) (tiq absent)')

        # Current density and flow (equilibrium; FE-node layout).
        jr2d = jz2d = jtor2d = None
        try:
            if jq is not None:
                jqA = np.asarray(jq, dtype=float)
                if jqA.ndim >= 3:
                    jr2d = jqA[:, :, 0]
                    jz2d = jqA[:, :, 1]
                    jtor2d = jqA[:, :, 2]
                    if jr2d is not None and Rloc is not None and jr2d.shape != Rloc.shape and jr2d.T.shape == Rloc.shape:
                        jr2d = jr2d.T
                        jz2d = jz2d.T
                        jtor2d = jtor2d.T
        except Exception:
            jr2d = jz2d = jtor2d = None

        vr2d = vz2d = vphi2d = omega2d = None
        try:
            if vq is not None:
                vqA = np.asarray(vq, dtype=float)
                if vqA.ndim >= 3:
                    vr2d = vqA[:, :, 0]
                    vz2d = vqA[:, :, 1]
                    vphi2d = vqA[:, :, 2]
                    if vr2d is not None and Rloc is not None and vr2d.shape != Rloc.shape and vr2d.T.shape == Rloc.shape:
                        vr2d = vr2d.T
                        vz2d = vz2d.T
                        vphi2d = vphi2d.T
                    with np.errstate(divide='ignore', invalid='ignore'):
                        omega2d = np.asarray(vphi2d, dtype=float) / np.asarray(Rloc, dtype=float)
        except Exception:
            vr2d = vz2d = vphi2d = omega2d = None

        # Write equilibrium scalars
        try:
            _write_node_scalar(q.electrons.density, ne2d)
        except Exception:
            pass
        try:
            _write_node_scalar(q.electrons.pressure, pe2d)
        except Exception:
            pass
        for _leafname in ('pressure', 'p_total', 'pressure_total'):
            try:
                _write_node_scalar(getattr(q, _leafname), pr2d)
                break
            except Exception:
                continue
        try:
            _write_node_scalar(q.electrons.temperature, te_eq)
        except Exception:
            pass
        try:
            _write_node_scalar(q.n_i_total, ni2d_total)
        except Exception:
            pass
        try:
            _write_node_scalar(q.t_i_average, ti_eq)
        except Exception:
            pass
        try:
            _write_node_scalar(q.current_density_tor, jtor2d)
        except Exception:
            pass
        # Also store J_R and J_Z if the DD provides leaves (common in some DD variants).
        for _leaf, _arr in (
            ("current_density_r", jr2d),
            ("current_density_R", jr2d),
            ("j_r", jr2d),
            ("j_R", jr2d),
            ("current_density_z", jz2d),
            ("current_density_Z", jz2d),
            ("j_z", jz2d),
            ("j_Z", jz2d),
        ):
            try:
                if hasattr(q, _leaf):
                    _write_node_scalar(getattr(q, _leaf), _arr)
            except Exception:
                pass
        # Also store J_R and J_Z if the DD provides leaves (common in some DD variants).
        for _leaf, _arr in (
            ("current_density_r", jr2d),
            ("current_density_R", jr2d),
            ("j_r", jr2d),
            ("j_R", jr2d),
            ("current_density_z", jz2d),
            ("current_density_Z", jz2d),
            ("j_z", jz2d),
            ("j_Z", jz2d),
        ):
            try:
                if hasattr(q, _leaf):
                    _write_node_scalar(getattr(q, _leaf), _arr)
            except Exception:
                pass
        try:
            _write_node_scalar(q.rotation_frequency_tor_s, omega2d)
        except Exception:
            pass


        def _write_node_scalar(container: Any, V2: np.ndarray | None) -> None:
            if V2 is None:
                return
            V2 = np.asarray(V2, dtype=float)
            if V2.shape != Rloc.shape and V2.T.shape == Rloc.shape:
                V2 = V2.T
            V3 = V2[:, :, None]
            try:
                container.resize(1)
                qt = container[0]
            except Exception:
                qt = container
            try:
                qt.grid_index = int(ig)
                qt.grid_subset_index = 0
            except Exception:
                pass
            shp = np.asarray([V3.shape[0], V3.shape[1], V3.shape[2]], dtype=np.int32)
            for attr in ("values_shape", "valuesShape", "values_SHAPE"):
                try:
                    leaf = getattr(qt, attr)
                except Exception:
                    leaf = None
                if leaf is None:
                    continue
                try:
                    try:
                        leaf.resize(3)
                        leaf[:] = shp
                    except Exception:
                        setattr(qt, attr, shp)
                    break
                except Exception:
                    continue
            try:
                qt.values = V3
            except Exception:
                try:
                    qt.values = V3.ravel(order="F")
                except Exception:
                    pass

        # --- Compute equilibrium node-centered quantities (no toroidal reconstruction) ---
        sp = getattr(args, '_nimrod_species', {}) or {}
        qe = float(sp.get('qe_c', 1.602176634e-19))
        zeff_input = sp.get('zeff_input', None)
        # Electron density (robust against nspec=1 and/or missing species axis)
        ne2d = None
        try:
            nqA = np.asarray(nq, dtype=float) if nq is not None else None
            if nqA is not None:
                if nqA.ndim == 2:
                    ne2d = nqA
                elif nqA.ndim >= 3 and nqA.shape[2] >= 1:
                    ne2d = nqA[:, :, 0]
                if ne2d is not None and ne2d.shape != Rloc.shape and getattr(ne2d, 'T', None) is not None and ne2d.T.shape == Rloc.shape:
                    ne2d = ne2d.T
        except Exception:
            ne2d = None


        # Electron pressure
        pe2d = np.asarray(pe2d, dtype=float) if pe2d is not None else None
        if pe2d is not None and pe2d.shape != Rloc.shape and pe2d.T.shape == Rloc.shape:
            pe2d = pe2d.T

        # Total pressure
        pr2d = np.asarray(pr2d, dtype=float) if pr2d is not None else None
        if pr2d is not None and pr2d.shape != Rloc.shape and pr2d.T.shape == Rloc.shape:
            pr2d = pr2d.T

        # Derive Te if not provided
        if te2d is None and pe2d is not None and ne2d is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                te2d = pe2d / (ne2d * qe)
            _log('edge_profiles(eq): derived Te = p_e / (n_e * qe) (teq not available)')

        # Ion density (sum over ion species if present; else fallback using zeff_input)
        ni2d = None
        try:
            if nqA is not None and nqA.ndim >= 3 and nqA.shape[2] >= 2:
                ni2d = np.nansum(nqA[:, :, 1:], axis=2)
                if ni2d.shape != Rloc.shape and ni2d.T.shape == Rloc.shape:
                    ni2d = ni2d.T
            elif ne2d is not None:
                z = float(zeff_input) if zeff_input not in (None, '') else None
                if z is not None and z > 0.0:
                    ni2d = np.asarray(ne2d, dtype=float) / z
                    _log(f'edge_profiles(eq): ion density fallback ni = ne/zeff_input (zeff_input={z:g})')
                else:
                    ni2d = np.asarray(ne2d, dtype=float)
                    _log('edge_profiles(eq): ion density fallback ni = ne (zeff_input unavailable)')
        except Exception:
            ni2d = None

        # Ion pressure and Ti
        pi2d = None
        if pr2d is not None and pe2d is not None:
            pi2d = pr2d - pe2d
            _log('edge_profiles(eq): computed p_i = p_total - p_e')
        if ti2d is None and pi2d is not None and ni2d is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ti2d = pi2d / (ni2d * qe)
            _log('edge_profiles(eq): derived Ti = p_i / (n_i * qe) (tiq not available)')

        # Toroidal current density and rotation frequency (if available)
        jtor2d = None
        try:
            if jq is not None:
                jqA = np.asarray(jq, dtype=float)
                if jqA.ndim >= 3:
                    jtor2d = jqA[:, :, 2]
                    if jtor2d.shape != Rloc.shape and jtor2d.T.shape == Rloc.shape:
                        jtor2d = jtor2d.T
        except Exception:
            jtor2d = None

        omega2d = None
        try:
            if vq is not None:
                vqA = np.asarray(vq, dtype=float)
                if vqA.ndim >= 3:
                    vphi = vqA[:, :, 2]
                    if vphi.shape != Rloc.shape and vphi.T.shape == Rloc.shape:
                        vphi = vphi.T
                    with np.errstate(divide='ignore', invalid='ignore'):
                        omega2d = vphi / np.asarray(Rloc, dtype=float)
        except Exception:
            omega2d = None

        # --- Write GGD leaves ---
        try:
            _write_node_scalar(q.electrons.temperature, te2d)
        except Exception:
            pass
        try:
            _write_node_scalar(q.electrons.density, ne2d)
        except Exception:
            pass
        try:
            _write_node_scalar(q.electrons.pressure, pe2d)
        except Exception:
            pass
        for _leafname in ('pressure', 'p_total', 'pressure_total'):
            try:
                _write_node_scalar(getattr(q, _leafname), pr2d)
                break
            except Exception:
                continue
        # n_i_total_over_n_e (IMAS DD leaf) instead of absolute n_i_total
        ni_over_ne = None
        try:
            if ni2d is not None and ne2d is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ni_over_ne = np.asarray(ni2d, dtype=float) / np.asarray(ne2d, dtype=float)
        except Exception:
            ni_over_ne = None
        for _leafname in ('n_i_total_over_n_e', 'n_i_total_over_ne'):
            if hasattr(q, _leafname):
                try:
                    _write_node_scalar(getattr(q, _leafname), ni_over_ne)
                    break
                except Exception:
                    pass

        # Per-ion equilibrium quantities in ggd(i)%ion(j) including velocity vector
        if hasattr(q, 'ion'):
            # Determine ion count: prefer nq species dim; else fall back to 1 ion species
            nion = 1
            try:
                if nqA is not None and nqA.ndim >= 3 and int(nqA.shape[2]) >= 2:
                    nion = int(nqA.shape[2] - 1)
            except Exception:
                nion = 1
            try:
                if _aos_len(q.ion) < nion:
                    q.ion.resize(nion)
            except Exception:
                pass
            # Velocity components (equilibrium only)
            vr2d = vz2d = vphi2d = None
            try:
                if vq is not None:
                    vqA = np.asarray(vq, dtype=float)
                    if vqA.ndim >= 3:
                        vr2d = vqA[:, :, 0]; vz2d = vqA[:, :, 1]; vphi2d = vqA[:, :, 2]
                        if vr2d.shape != Rloc.shape and vr2d.T.shape == Rloc.shape:
                            vr2d = vr2d.T; vz2d = vz2d.T; vphi2d = vphi2d.T
            except Exception:
                vr2d = vz2d = vphi2d = None
            for k in range(nion):
                try:
                    ion_k = q.ion[k]
                except Exception:
                    continue
                # density for this ion
                ni_k = None
                try:
                    if nqA is not None and nqA.ndim >= 3 and int(nqA.shape[2]) >= 2:
                        ni_k = np.asarray(nqA[:, :, 1 + k], dtype=float)
                        if ni_k.shape != Rloc.shape and ni_k.T.shape == Rloc.shape:
                            ni_k = ni_k.T
                    else:
                        ni_k = np.asarray(ni2d, dtype=float) if ni2d is not None else None
                except Exception:
                    ni_k = None
                for _nm in ('density', 'n', 'number_density'):
                    if hasattr(ion_k, _nm):
                        try:
                            _write_node_scalar(getattr(ion_k, _nm), ni_k)
                            break
                        except Exception:
                            pass
                for _nm in ('temperature', 't_i', 't'):
                    if hasattr(ion_k, _nm):
                        try:
                            _write_node_scalar(getattr(ion_k, _nm), ti2d)
                            break
                        except Exception:
                            pass
                # velocity vector lives under ion%velocity in edge_profiles DD
                vel_obj = getattr(ion_k, 'velocity', None)
                if vel_obj is None:
                    vel_obj = ion_k
                for _leaf, _V2 in (('r', vr2d), ('z', vz2d), ('phi', vphi2d)):
                    if hasattr(vel_obj, _leaf):
                        try:
                            _write_node_scalar(getattr(vel_obj, _leaf), _V2)
                        except Exception:
                            pass

                # pressure: p_i,k = n_i,k * T_i * e (best-effort)
                pi_k = None
                try:
                    if ni_k is not None and ti_eq is not None:
                        pi_k = np.asarray(ni_k, dtype=float) * np.asarray(ti_eq, dtype=float) * qe
                    elif pi2d is not None and ni_k is not None and ni2d_total is not None:
                        with np.errstate(divide='ignore', invalid='ignore'):
                            frac = np.asarray(ni_k, dtype=float) / np.asarray(ni2d_total, dtype=float)
                        pi_k = frac * np.asarray(pi2d, dtype=float)
                except Exception:
                    pi_k = None
                for _nm in ('pressure', 'p'):
                    if hasattr(ion_k, _nm):
                        try:
                            _write_node_scalar(getattr(ion_k, _nm), pi_k)
                            break
                        except Exception:
                            pass
        try:
            _write_node_scalar(q.t_i_average, ti2d)
        except Exception:
            pass
        for _leafname in ('p_i_total', 'ions_pressure', 'ion_pressure', 'p_ions'):
            try:
                _write_node_scalar(getattr(q, _leafname), pi2d)
                break
            except Exception:
                continue
        # rotation_frequency_tor_s not written: edge_profiles DD stores ion velocity vector directly
        try:
            _write_node_scalar(q.current_density_tor, jtor2d)
        except Exception:
            pass
        return

    sp = getattr(args, "_nimrod_species", {}) or {}
    qe = float(sp.get("qe_c", 1.602176634e-19))
    n_scale = float(getattr(args, "n_scale", 1.0) or 1.0)

    # Electron density: ALWAYS prefer nq (even when peq/teq are present) to avoid unit mismatches.
    ne2d = None
    nqA = None
    try:
        nqA = np.asarray(nq, dtype=float) if nq is not None else None
        if nqA is not None:
            nqA = np.asarray(nqA, dtype=float) * float(n_scale)
        if nqA is not None:
            if nqA.ndim >= 3 and nqA.shape[-1] >= 1:
                ne2d = nqA[..., 0]
            elif nqA.ndim == 2:
                ne2d = nqA
    except Exception:
        ne2d = None
        nqA = None

    # Derive missing equilibrium temperatures if needed (2D).  Te, Ti are stored in eV; pressure in Pa.
    if te2d is None and (pe2d is not None) and (ne2d is not None):
        try:
            te2d = _as_f64(pe2d) / (_as_f64(ne2d) * qe)
        except Exception:
            te2d = None

    if ti2d is None and (pr2d is not None) and (pe2d is not None) and (nqA is not None) and (getattr(nqA, "ndim", 0) >= 3) and (nqA.shape[-1] >= 2):
        try:
            ni2d = np.nansum(_as_f64(nqA[..., 1:]), axis=2)
            pi2d = _as_f64(pr2d) - _as_f64(pe2d)
            ti2d = pi2d / (ni2d * qe)
        except Exception:
            ti2d = None

    # Total ion density (needed by some DD variants that store only ratios).
    ni2d_total = None
    if (nqA is not None) and (getattr(nqA, "ndim", 0) >= 3) and (nqA.shape[-1] >= 2):
        try:
            ni2d_total = np.nansum(_as_f64(nqA[..., 1:]), axis=2)
        except Exception:
            ni2d_total = None
    elif (ne2d is not None):
        # No-impurity (single density channel) case: estimate main-ion density from constant Zeff.
        try:
            zeff = sp.get("zeff_input", None)
            zeff = float(zeff) if zeff is not None else None
            if zeff is not None and np.isfinite(zeff) and (zeff > 0.0):
                ni2d_total = _as_f64(ne2d) / float(zeff)
        except Exception:
            ni2d_total = None


    # Toroidal components (common analysis targets). We store equilibrium-only fields.

    vr2d = vz2d = vtor2d = None
    omega2d = None
    if vq is not None:
        try:
            vqA = np.asarray(vq, dtype=float)
            if vqA.ndim >= 3:
                vr2d = np.asarray(vqA[..., 0], dtype=float)
                vz2d = np.asarray(vqA[..., 1], dtype=float)
                vtor2d = np.asarray(vqA[..., 2], dtype=float)
                if R is not None and vr2d is not None and vr2d.shape != R.shape and vr2d.T.shape == R.shape:
                    vr2d = vr2d.T
                    vz2d = vz2d.T
                    vtor2d = vtor2d.T
        except Exception:
            vr2d = vz2d = vtor2d = None
        if vtor2d is not None:
            try:
                # omega = v_phi / R
                omega2d = np.full_like(vtor2d, np.nan, dtype=float)
                msk = np.isfinite(vtor2d) & np.isfinite(R) & (np.abs(R) > 0)
                omega2d[msk] = vtor2d[msk] / np.asarray(R, dtype=float)[msk]
            except Exception:
                omega2d = None

    jr2d = jz2d = jtor2d = None
    if jq is not None:
        try:
            jqA = np.asarray(jq, dtype=float)
            if jqA.ndim >= 3:
                jr2d = np.asarray(jqA[..., 0], dtype=float)
                jz2d = np.asarray(jqA[..., 1], dtype=float)
                jtor2d = np.asarray(jqA[..., 2], dtype=float)
                if R is not None and jr2d is not None and jr2d.shape != R.shape and jr2d.T.shape == R.shape:
                    jr2d = jr2d.T
                    jz2d = jz2d.T
                    jtor2d = jtor2d.T
        except Exception:
            jr2d = jz2d = jtor2d = None

    # Append time slice to GGD arrays.
    it, ig = _append_time_ggd(ep, t, write_grid=_ggd_should_write_grid(args), reuse_grid=bool(getattr(args, "ggd_reuse_grid", False)))
    g = ep.grid_ggd[ig]
    gidx = int(ig)

    # Minimal grid identifier
    try:
        g.identifier.name = "nimrod_rz_regular"
        g.identifier.index = int(ig + 1)
        g.identifier.description = "Regular R-Z grid for NIMROD equilibrium export (downsampled)"
    except Exception:
        pass


    # In structured mode, define at least the nodes subset per IMAS GGD specification.
    if not getattr(args, 'ggd_unstructured', False):
        try:
            g.grid_subset.resize(1)
            gs = g.grid_subset[0]
            gs.identifier.name = 'nodes'
            gs.identifier.index = 1
            gs.identifier.description = 'All nodes of the structured grid'
            gs.dimension = 1
        except Exception:
            pass

    # We reuse the structured/unstructured skeleton logic from mhd.ggd.
    if getattr(args, "ggd_unstructured", False):
        # IMPORTANT:
        #   plot_mhd.py expects per-node coordinate vectors stored in
        #     grid_ggd.space.objects_per_dimension.object.geometry
        #   with space[0]=R, space[1]=Z, space[2]=phi. For equilibrium-only exports
        #   we set phi=0 for all nodes.
        try:
            g.space.resize(3)
            for ii, nm in enumerate(["R", "Z", "phi"]):
                try:
                    g.space[ii].identifier.name = nm
                    g.space[ii].identifier.index = -1
                    g.space[ii].identifier.description = nm
                except Exception:
                    pass
        except Exception:
            pass

        try:
            g.grid_subset.resize(2)

            # Subset 0: nodes (dimension=0) — placeholder scaffolding
            s0 = g.grid_subset[0]
            try:
                s0.dimension = 1
                s0.identifier.name = "nodes"
                s0.identifier.index = 1
                s0.identifier.description = "Unstructured nodes"
            except Exception:
                pass
            try:
                s0.element.resize(1)
                # (R,Z,phi) components
                s0.element[0].object.resize(3)
                for k in range(3):
                    try:
                        s0.element[0].object[k].real = 0.0
                    except Exception:
                        pass
            except Exception:
                pass

            # Subset 1: cells (dimension=3) — placeholder connectivity (quads)
            s1 = g.grid_subset[1]
            try:
                s1.dimension = 3
                s1.identifier.name = "cells"
                s1.identifier.index = 5
                s1.identifier.description = "Unstructured connectivity (quad placeholder)"
            except Exception:
                pass
            try:
                s1.base.resize(1)
                s1.base[0].index = 0
                s1.base[0].grid_subset_index = 1
            except Exception:
                pass
            try:
                s1.element.resize(1)
                s1.element[0].object.resize(4)
                for k in range(4):
                    try:
                        s1.element[0].object[k].index = 1
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

        if conn_kind == "fe_pointcloud":
            try:
                g.grid_subset.resize(1)
            except Exception:
                pass

        nspaces = 3
    else:
        nspaces = 2


    try:
        g.space.resize(nspaces)
    except Exception:
        pass

    def _fill_space(space_obj, coord_name: str, coord_vals: np.ndarray):
        try:
            space_obj.geometry_type.index = 0
            space_obj.geometry_type.name = "standard"
            space_obj.geometry_type.description = "standard"
        except Exception:
            pass
        try:
            space_obj.coordinates_type.resize(1)
            _cn = str(coord_name).strip().lower()
            space_obj.coordinates_type[0].name = _cn
            space_obj.coordinates_type[0].index = {'r': 4, 'z': 3, 'phi': 5}.get(_cn, -1)
            space_obj.coordinates_type[0].description = _cn
        except Exception:
            pass
        try:
            space_obj.objects_per_dimension.resize(1)
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
        vals = _as_f64(values_1d)
        shp = np.asarray(list(shape_hint), dtype=np.int32).ravel()
        try:
            if shp.size == 3:
                nr_, nz_, nphi_ = (int(shp[0]), int(shp[1]), int(shp[2]))
                qleaf.values = vals.reshape((nr_, nz_, nphi_), order="F")
            else:
                qleaf.values = vals
        except Exception:
            try:
                qleaf.values = vals
            except Exception:
                return

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

    q = ep.ggd[it]

    def _write_scalar(container: Any, V2: np.ndarray | None) -> None:
        """Write scalar equilibrium field into a `leaf[]` container."""
        if V2 is None:
            return
        try:
            # Use triangulation-based interpolation to avoid empty-bin gaps.
            rc, zc, Vb = _interp2d_linear(R, Z, V2, nb, nb)

            if getattr(args, "ggd_unstructured", False):
                # Unstructured mode: store per-node (R,Z,phi) vectors in grid_ggd.space geometry.
                # Node ordering MUST match the Fortran-order flattening used for values below.
                RR, ZZ = np.meshgrid(rc, zc, indexing="ij")
                r_nodes = RR.ravel(order="F")
                z_nodes = ZZ.ravel(order="F")
                phi_nodes = np.zeros_like(r_nodes)

                _fill_space(g.space[0], "R", r_nodes)
                _fill_space(g.space[1], "Z", z_nodes)
                _fill_space(g.space[2], "phi", phi_nodes)
            else:
                # Structured mode: store coordinate axes.
                _fill_space(g.space[0], "R", rc)
                _fill_space(g.space[1], "Z", zc)

            V3 = np.asarray(Vb, dtype=float)[:, :, None]
            vals = V3.ravel(order="F")
            try:
                container.resize(1)
                qt = container[0]
            except Exception:
                qt = container
            try:
                qt.grid_index = int(ig)
                qt.grid_subset_index = 0
            except Exception:
                pass
            _set_values_and_shape(qt, vals, (len(rc), len(zc), int(nphi)))
        except Exception:
            pass

    def _try_write(obj: Any, names: Sequence[str], V2: np.ndarray | None) -> bool:
        for nm in names:
            if hasattr(obj, nm):
                try:
                    _write_scalar(getattr(obj, nm), V2)
                    return True
                except Exception:
                    return False
        return False

    # electrons.temperature
    try:
        _write_scalar(q.electrons.temperature, te2d)
    except Exception:
        pass

    # electrons.density and electrons.pressure
    try:
        _try_write(q.electrons, ("density", "n", "number_density"), ne2d)
    except Exception:
        pass
    try:
        _try_write(q.electrons, ("pressure", "p"), pe2d)
    except Exception:
        pass


    # t_i_average
    _try_write(q, ("t_i_average", "ti", "t_i"), ti2d)

    # n_i_total_over_n_e (IMAS DD leaf) instead of absolute n_i_total
    ni_over_ne = None
    try:
        if ni2d_total is not None and ne2d is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ni_over_ne = np.asarray(ni2d_total, dtype=float) / np.asarray(ne2d, dtype=float)
    except Exception:
        ni_over_ne = None
    _try_write(q, ('n_i_total_over_n_e', 'n_i_total_over_ne'), ni_over_ne)

    # current density (toroidal component)
    _try_write(q, ("j_r", "j_R", "jr", "current_density_r", "current_density_R"), jr2d)
    _try_write(q, ("j_z", "j_Z", "jz", "current_density_z", "current_density_Z"), jz2d)
    _try_write(q, ("j_tor", "j_phi", "jtor", "current_density_tor", "current_density_phi"), jtor2d)

    # velocity and rotation frequency (toroidal)
    _try_write(q, ("v_r", "v_R", "vr", "velocity_r", "velocity_R"), vr2d)
    _try_write(q, ("v_z", "vz", "velocity_z", "velocity_Z"), vz2d)
    _try_write(q, ("v_tor", "v_phi", "vtor", "velocity_tor", "velocity_phi"), vtor2d)
    # rotation_frequency_tor_s not written: edge_profiles DD stores ion velocity vector directly

    # Per-ion equilibrium quantities in ggd(i)%ion(j)
    if nq is not None and getattr(nq, "ndim", 0) >= 3 and nq.shape[-1] >= 2 and hasattr(q, "ion"):
        nion = int(nq.shape[-1] - 1)
        try:
            if _aos_len(q.ion) < nion:
                q.ion.resize(nion)
        except Exception:
            pass

        z_ions = (sp.get("z_ions", []) or [])
        m_ions = (sp.get("m_ions_kg", []) or [])
        AMU = 1.66053906660e-27

        for k in range(nion):
            try:
                ion_k = q.ion[k]
            except Exception:
                continue

            ni_k = None
            try:
                ni_k = np.asarray(nq[..., 1 + k], dtype=float)
            except Exception:
                ni_k = None

            _try_write(ion_k, ("density", "n", "number_density"), ni_k)

            # Many NIMROD cases have a single ion temperature; write it for each species.
            _try_write(ion_k, ("temperature", "t_i", "t"), ti2d)
            # Velocity vector (edge_profiles.ggd%ion%velocity). NIMROD provides a bulk flow; store for each ion species.
            vr2d = vz2d = vphi2d = None
            try:
                if vq is not None:
                    vqA = np.asarray(vq, dtype=float)
                    if vqA.ndim >= 3:
                        vr2d = vqA[:, :, 0]
                        vz2d = vqA[:, :, 1]
                        vphi2d = vqA[:, :, 2]
            except Exception:
                vr2d = vz2d = vphi2d = None
            vel_obj = getattr(ion_k, 'velocity', None)
            if vel_obj is None:
                vel_obj = ion_k
            _try_write(vel_obj, ('r', 'velocity_r'), vr2d)
            _try_write(vel_obj, ('z', 'velocity_z'), vz2d)
            _try_write(vel_obj, ('phi', 'velocity_phi'), vphi2d)


            # Species pressure: p_i,k = n_i,k * T_i * e
            if ni_k is not None and ti2d is not None:
                try:
                    pi_k = np.asarray(ni_k, dtype=float) * np.asarray(ti2d, dtype=float) * qe
                except Exception:
                    pi_k = None
                _try_write(ion_k, ("pressure", "p"), pi_k)

            # Metadata (best-effort)
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


    # Fallback for simulations without explicit ion species in nq:
    # populate a single main-ion entry from ni2d_total (computed above, typically ni = ne/Zeff).
    if (hasattr(q, "ion") and (ni2d_total is not None) and not (nq is not None and getattr(nq, "ndim", 0) >= 3 and nq.shape[-1] >= 2)):
        try:
            if _aos_len(q.ion) < 1:
                q.ion.resize(1)
        except Exception:
            pass
        try:
            ion0 = q.ion[0]
            _try_write(ion0, ("density", "n", "number_density"), ni2d_total)
            _try_write(ion0, ("temperature", "t_i", "t"), ti2d)
            vel0 = getattr(ion0, 'velocity', None)
            if vel0 is None:
                vel0 = ion0
            _try_write(vel0, ('r', 'velocity_r'), vr2d)
            _try_write(vel0, ('z', 'velocity_z'), vz2d)
            _try_write(vel0, ('phi', 'velocity_phi'), vtor2d)
            if ti2d is not None:
                try:
                    pi0 = np.asarray(ni2d_total, dtype=float) * np.asarray(ti2d, dtype=float) * qe
                except Exception:
                    pi0 = None
                _try_write(ion0, ("pressure", "p"), pi0)
            # Best-effort metadata
            for nm in ("z_ion", "z", "charge_state", "charge"):
                if hasattr(ion0, nm):
                    try:
                        setattr(ion0, nm, 1.0)
                        break
                    except Exception:
                        pass
        except Exception:
            pass

def _build_unstructured_nodes_connectivity(
    data: Dict[str, Any],
    args,
) -> tuple["np.ndarray", "np.ndarray | None", dict]:
    """Build an unstructured (R,Z,phi) node list and an optional connectivity array.

    This is used to populate IMAS-standard ``grid_ggd`` structures without relying on any
    non-standard / NIMROD-specific HDF5 groups.

    Returns
    -------
    nodes_xyz : float64, shape (Nnodes, 3)
        Per-node coordinates (R, Z, phi).
    connectivity : int32 or None, shape (Ncells, Nverts)
        1-based node indices for each cell. For triangles: Nverts=3, wedges: 6, hexes: 8.
        Returned as None when connectivity is disabled/unavailable.
    meta : dict
        Small metadata (nr, nz, nphi, connectivity_kind).
    """
    import numpy as np

    # Unstructured export configuration
    conn_kind = str(getattr(args, "ggd_connectivity", "none") or "none").strip().lower()
    nphi = max(1, int(getattr(args, "ggd_nphi", 8) or 1))
    phi_axis = np.linspace(0.0, 2.0 * np.pi, num=nphi, endpoint=False)

    Rloc = np.asarray(data.get("R"), dtype=float) if data.get("R") is not None else None
    Zloc = np.asarray(data.get("Z"), dtype=float) if data.get("Z") is not None else None
    if Rloc is None or Zloc is None:
        raise ValueError("Missing R/Z grids in data; cannot build unstructured nodes/connectivity.")

    # Prefer native stitched FE node lattice when requested.
    use_fe_nodes = (
        bool(getattr(args, "ggd_unstructured", False))
        and bool(getattr(args, "ggd_unstructured_fe_nodes", False))
        and conn_kind in ("fe_tri", "fe_wedge", "fe_pointcloud")
    )

    if use_fe_nodes:
        # Ensure R,Z are aligned (some dumps store transposed arrays).
        ref = None
        for _k in ("teq", "peq", "prq"):
            if data.get(_k) is not None:
                ref = np.asarray(data.get(_k))
                break
        if ref is not None and ref.shape != Rloc.shape:
            if ref.T.shape == Rloc.shape:
                ref = ref.T
            elif Rloc.T.shape == ref.shape:
                Rloc = Rloc.T
                Zloc = Zloc.T

        if conn_kind == "fe_pointcloud":
            # Nodes only: explicit per-node coordinates; no connectivity/cells.
            r2d = np.asarray(Rloc, dtype=float).ravel(order="F")
            z2d = np.asarray(Zloc, dtype=float).ravel(order="F")
            nn2d = int(r2d.size)
            r_nodes = np.tile(r2d, int(nphi))
            z_nodes = np.tile(z2d, int(nphi))
            phi_nodes = np.repeat(phi_axis.astype(float), nn2d)
            nodes_xyz = np.stack((r_nodes, z_nodes, phi_nodes), axis=1).astype(np.float64, copy=False)
            connectivity = None
        elif conn_kind == "fe_tri":
            nodes_xyz, conn0 = _build_fe_tri_nodes_conn(Rloc, Zloc, nphi, phi_axis)
            connectivity = conn0.astype(np.int32, copy=False)
            if connectivity.size:
                connectivity = connectivity + 1
        else:
            nodes_xyz, conn0 = _build_fe_wedge_nodes_conn(Rloc, Zloc, nphi, phi_axis)
            connectivity = conn0.astype(np.int32, copy=False)
            if connectivity.size:
                connectivity = connectivity + 1

        ny, nx = Rloc.shape
        nr, nz = int(nx), int(ny)
        return nodes_xyz.astype(np.float64, copy=False), (connectivity if connectivity is not None else None), {
            "nr": nr,
            "nz": nz,
            "nphi": int(nphi),
            "connectivity_kind": conn_kind,
        }

    # Regular product-grid mode (downsampled rectangular R-Z grid, extruded in phi)
    nb = max(4, int(getattr(args, "ggd_nbins", 128) or 128))
    nr = nz = int(nb)

    rmin = float(np.nanmin(Rloc))
    rmax = float(np.nanmax(Rloc))
    zmin = float(np.nanmin(Zloc))
    zmax = float(np.nanmax(Zloc))
    if not np.isfinite([rmin, rmax, zmin, zmax]).all():
        raise ValueError("Non-finite R/Z extents; cannot build unstructured product grid.")

    r_axis = np.linspace(rmin, rmax, num=nr)
    z_axis = np.linspace(zmin, zmax, num=nz)
    RR, ZZ, PP = np.meshgrid(r_axis, z_axis, phi_axis, indexing="ij")  # (nr,nz,nphi)

    nodes_xyz = np.stack(
        (
            RR.reshape(-1, order="F"),
            ZZ.reshape(-1, order="F"),
            PP.reshape(-1, order="F"),
        ),
        axis=1,
    ).astype(np.float64, copy=False)

    # Connectivity
    connectivity: "np.ndarray | None" = None

    if conn_kind in ("none", "fe_pointcloud"):
        connectivity = None

    elif conn_kind == "hex":
        # One hex cell per (i,j,k) with periodicity in phi.
        ncell = (nr - 1) * (nz - 1) * (nphi)
        conn = np.empty((int(ncell), 8), dtype=np.int32)

        def node_index(i: int, j: int, k: int) -> int:
            # 1-based index into nodes_xyz, consistent with the Fortran-order flattening.
            return 1 + int(k) * (nr * nz) + int(j) * nr + int(i)

        c = 0
        for k in range(int(nphi)):
            kp = (k + 1) % int(nphi)
            for j in range(nz - 1):
                for i in range(nr - 1):
                    conn[c, 0] = node_index(i, j, k)
                    conn[c, 1] = node_index(i + 1, j, k)
                    conn[c, 2] = node_index(i + 1, j + 1, k)
                    conn[c, 3] = node_index(i, j + 1, k)
                    conn[c, 4] = node_index(i, j, kp)
                    conn[c, 5] = node_index(i + 1, j, kp)
                    conn[c, 6] = node_index(i + 1, j + 1, kp)
                    conn[c, 7] = node_index(i, j + 1, kp)
                    c += 1
        connectivity = conn

    elif conn_kind in ("fe_tri", "fe_wedge", "fe_pointcloud"):
        # Triangulate each (i,j) quad on the regular R-Z grid and (optionally) extrude in phi.
        # Base 2D triangulation on one phi plane (0-based indices).
        ii, jj = np.meshgrid(np.arange(nr - 1, dtype=np.int32), np.arange(nz - 1, dtype=np.int32), indexing="ij")
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)
        a = jj * nr + ii
        b = jj * nr + (ii + 1)
        c = (jj + 1) * nr + (ii + 1)
        d = (jj + 1) * nr + ii
        tri0 = np.stack([a, b, c], axis=1)
        tri1 = np.stack([a, c, d], axis=1)
        tri_plane = np.vstack([tri0, tri1]).astype(np.int32, copy=False)  # (ntri,3)

        nn2d = int(nr * nz)
        ntri = int(tri_plane.shape[0])

        if conn_kind == "fe_tri":
            # Replicate triangles per phi plane (still 2D elements).
            if int(nphi) == 1:
                tri = tri_plane
            else:
                tri = np.empty((ntri * int(nphi), 3), dtype=np.int32)
                for k in range(int(nphi)):
                    tri[k * ntri : (k + 1) * ntri, :] = tri_plane + k * nn2d
            connectivity = tri + 1  # 1-based

        else:
            # Wedges: extrude triangles between adjacent phi planes (periodic).
            if int(nphi) < 2 or ntri == 0:
                connectivity = np.zeros((0, 6), dtype=np.int32)
            else:
                wedge = np.empty((ntri * int(nphi), 6), dtype=np.int32)
                for k in range(int(nphi)):
                    kp = (k + 1) % int(nphi)
                    off0 = k * nn2d
                    off1 = kp * nn2d
                    sl = slice(k * ntri, (k + 1) * ntri)
                    wedge[sl, 0:3] = tri_plane + off0
                    wedge[sl, 3:6] = tri_plane + off1
                connectivity = wedge + 1  # 1-based

    else:
        # Unknown/unsupported
        raise ValueError(f"Unsupported connectivity kind: {conn_kind}")

    return nodes_xyz, (connectivity.astype(np.int32, copy=False) if connectivity is not None else None), {
        "nr": int(nr),
        "nz": int(nz),
        "nphi": int(nphi),
        "connectivity_kind": conn_kind,
    }


def _write_unstructured_ggd_aux_h5(entry_dir: str, ids_name: str, occ: int, data: Dict[str, Any], args) -> None:
    """Populate IMAS-standard grid_ggd structures (h5py direct).

    NOTE: This function used to write a NIMROD-specific auxiliary HDF5 group
    ``nimrod_unstructured``. That group is *not* part of IMAS and is no longer
    written. The function name is retained for backwards compatibility with
    existing workflows, but it now writes:

      - ``grid_ggd.space`` per-node (R,Z,phi) coordinate vectors
      - ``grid_ggd.grid_subset`` nodes + connectivity via a packed writer

    This requires the IDS HDF5 file to already exist (i.e., after IMAS put()).
    """
    import numpy as np
    import os
    import h5py

    log = logging.getLogger(__name__)

    nodes_xyz, connectivity, meta = _build_unstructured_nodes_connectivity(data, args)

    # 1) Ensure node coordinate vectors are present in grid_ggd.space (portable standard location).
    try:
        _write_gridggd_space_geometry_vectors_h5(
            entry_dir,
            ids_name,
            occ,
            nodes_xyz[:, 0],
            nodes_xyz[:, 1],
            nodes_xyz[:, 2],
            log=log,
        )
    except Exception as e:
        log.warning("unstructured grid_ggd: failed to write space geometry vectors: %s", e)

    # 2) Populate official grid_ggd grid_subset node coordinates and connectivity.
    if connectivity is not None and int(np.prod(connectivity.shape)) > 0:
        try:
            _write_unstructured_gridggd_packed_h5(entry_dir, ids_name, occ, nodes_xyz, connectivity, log=log)
        except Exception as e:
            log.warning("unstructured grid_ggd: failed to packed-write grid_subset connectivity: %s", e)

    # 3) Best-effort: patch any missing values_SHAPE datasets for GGD quantity leaves, using
    #    the known (nr,nz,nphi) product-grid shape. (This does *not* affect connectivity.)
    try:
        nr = int(meta.get("nr", 0) or 0)
        nz = int(meta.get("nz", 0) or 0)
        nphi = int(meta.get("nphi", 0) or 0)
        if nr > 0 and nz > 0 and nphi > 0:
            shp = np.asarray([nr, nz, nphi], dtype=np.int32).reshape(1, 1, 3)
            h5_path = os.path.join(entry_dir, f"{ids_name}_{occ}.h5")
            grp_name = f"{ids_name}_{occ}"
            if os.path.exists(h5_path):
                with h5py.File(h5_path, "r+") as _h:
                    if grp_name in _h:
                        g = _h[grp_name]
                        for name, obj in list(g.items()):
                            if (
                                isinstance(obj, h5py.Dataset)
                                and ("ggd[]&" in name)
                                and name.endswith("&values_SHAPE")
                            ):
                                try:
                                    if obj.shape != shp.shape:
                                        try:
                                            obj.resize(shp.shape)
                                        except Exception:
                                            pass
                                    obj[...] = shp
                                except Exception:
                                    pass
    except Exception:
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

        # We must have subset/element AOS_SHAPE datasets to size the tree.
        # Normally these are created by the IMAS HDF5 backend when grid_ggd.grid_subset is non-empty.
        # However, some builds omit them unless explicitly populated. In that case we create the
        # packed datasets ourselves (best-effort) so bulk writing can proceed.
        if ds_subset_aos is None:
            ds_subset_aos = "grid_ggd[]&grid_subset[]&AOS_SHAPE"
            if ds_subset_aos not in g:
                g.create_dataset(
                    ds_subset_aos,
                    shape=(1, 1),
                    maxshape=(None, 1),
                    dtype=np.int32,
                )

        if ds_elem_aos is None:
            ds_elem_aos = "grid_ggd[]&grid_subset[]&element[]&AOS_SHAPE"
            if ds_elem_aos not in g:
                g.create_dataset(
                    ds_elem_aos,
                    shape=(1, n_subsets),
                    maxshape=(None, n_subsets),
                    dtype=np.int32,
                )

        # Optional, but helpful for backends/readers that expect an explicit object-count AoS.
        if ds_obj_aos is None:
            ds_obj_aos = "grid_ggd[]&grid_subset[]&element[]&object[]&AOS_SHAPE"
            if ds_obj_aos not in g:
                g.create_dataset(
                    ds_obj_aos,
                    shape=(1, n_subsets, 1),
                    maxshape=(None, n_subsets, None),
                    dtype=np.int32,
                )

        # Some backend/DD combinations do not create the leaf datasets (object[]&real / object[]&index)
        # unless they were populated via python bindings first. In packed mode, create them if missing
        # and then proceed with the bulk write.
        if ds_real is None:
            ds_real = "grid_ggd[]&grid_subset[]&element[]&object[]&real"
            if ds_real not in g:
                g.create_dataset(
                    ds_real,
                    shape=(1, n_subsets, 1, 1),
                    maxshape=(1, n_subsets, None, None),
                    dtype=np.float64,
                )
        if ds_index is None:
            ds_index = "grid_ggd[]&grid_subset[]&element[]&object[]&index"
            if ds_index not in g:
                g.create_dataset(
                    ds_index,
                    shape=(1, n_subsets, 1, 1),
                    maxshape=(1, n_subsets, None, None),
                    dtype=np.int32,
                )

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

        def _recreate_with_chunks(ds_name: str, shape: tuple[int, ...], maxshape: tuple[int | None, ...], dtype, *,
                                  chunks: tuple[int, ...], compression, compression_opts, shuffle: bool, fillvalue, log):
            """(Re)create dataset with reasonable chunking to avoid HDF5 allocating gigantic chunk buffers.

            Some IMAS backend versions choose pathological chunk shapes like (1,2,nelem,nverts) which can exceed RAM.
            We delete and recreate the dataset with smaller chunks, then the caller rewrites data fully.
            """
            try:
                if ds_name in g:
                    try:
                        del g[ds_name]
                    except Exception as e:
                        log.warning("Packed grid_ggd writer: could not delete %s for rechunking: %s", ds_name, e)
                        return
                g.create_dataset(
                    ds_name,
                    shape=shape,
                    maxshape=maxshape,
                    dtype=dtype,
                    chunks=chunks,
                    compression=compression,
                    compression_opts=compression_opts,
                    shuffle=shuffle,
                    fillvalue=fillvalue,
                )
            except Exception as e:
                log.warning("Packed grid_ggd writer: rechunk-create failed for %s: %s", ds_name, e)

        def _ensure_reasonable_chunks(ds, ds_name: str, want_shape: tuple[int, ...], want_maxshape: tuple[int | None, ...],
                                      elem_axis: int, elem_count: int, obj_axis: int, obj_count: int, bytes_per_item: int,
                                      log):
            """If ds uses a chunk that is too large along element/object axes, recreate with smaller chunks."""
            try:
                ch = ds.chunks
                if ch is None:
                    return
                # Estimate per-chunk memory footprint (rough).
                chunk_bytes = 1
                for c in ch:
                    chunk_bytes *= int(c)
                chunk_bytes *= int(bytes_per_item)
                # Trigger if chunk is enormous in element dimension or absolute bytes > 128 MiB.
                if ch[elem_axis] > 20000 or chunk_bytes > 128 * 1024 * 1024:
                    # Choose a conservative chunk along the element axis.
                    elem_chunk = min(8192, elem_count) if elem_count > 0 else 1
                    # Keep other axes minimal to avoid multiplying.
                    new_chunks = list(ch)
                    new_chunks[0] = 1
                    if len(new_chunks) > 1:
                        new_chunks[1] = 1
                    new_chunks[elem_axis] = elem_chunk
                    new_chunks[obj_axis] = obj_count
                    new_chunks = tuple(int(x) for x in new_chunks)
                    _recreate_with_chunks(
                        ds_name,
                        shape=want_shape,
                        maxshape=want_maxshape,
                        dtype=ds.dtype,
                        chunks=new_chunks,
                        compression=ds.compression,
                        compression_opts=ds.compression_opts,
                        shuffle=True,
                        fillvalue=ds.fillvalue,
                        log=log,
                    )
            except Exception:
                # Best-effort only.
                return

        # If the IMAS backend created pathological chunk sizes (common for large unstructured meshes),
        # recreate the coordinate/connectivity datasets with smaller chunks before writing.
        if hasattr(dreal, "chunks") and dreal.ndim == 4:
            _ensure_reasonable_chunks(
                dreal, ds_real,
                want_shape=(1, n_subsets, n_nodes, 3),
                want_maxshape=(1, n_subsets, None, 3),
                elem_axis=2, elem_count=n_nodes,
                obj_axis=3, obj_count=3,
                bytes_per_item=8,
                log=log,
            )
            dreal = g[ds_real]
        if hasattr(dind, "chunks") and dind.ndim == 4:
            _ensure_reasonable_chunks(
                dind, ds_index,
                want_shape=(1, n_subsets, n_cells, n_verts),
                want_maxshape=(1, n_subsets, None, n_verts),
                elem_axis=2, elem_count=n_cells,
                obj_axis=3, obj_count=n_verts,
                bytes_per_item=4,
                log=log,
            )
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
            # NOTE: do not clear the full packed datasets here (can trigger massive I/O and memory pressure).

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
                    # NOTE: avoid full-dataset clears; write only the required slice below.
                    dreal[0, nodes_subset_index, : flat_nodes.size] = flat_nodes
                    wrote_any = True
            except Exception as e:
                log.warning("Packed grid_ggd writer: could not write flattened real: %s", e)

            try:
                if dind.ndim == 3:
                    dind.resize((1, n_subsets, flat_conn.size))
                    # NOTE: avoid full-dataset clears; write only the required slice below.
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


def _write_gridggd_space_geometry_vectors_h5(
    entry_dir: str,
    ids_name: str,
    occ: int,
    r_nodes: "np.ndarray",
    z_nodes: "np.ndarray",
    phi_nodes: "np.ndarray",
    *,
    log: "logging.Logger | None" = None,
) -> None:
    """Write IMAS-standard grid_ggd.space node coordinate vectors.

    Some IMAS python bindings do not reliably materialize the leaf dataset
    ``grid_ggd[]&space[]&objects_per_dimension[]&object[]&geometry`` when
    populating GGD through the IDS object tree (especially for large vectors).
    Since downstream tooling (e.g. plot_mhd.py) expects this dataset, we write
    it directly via h5py using the standard backend path.

    The expected backend layout is:
      geometry shape = (1, 3, 1, 1, N, 1)
    where the second dimension indexes (R, Z, phi).
    """

    import os
    import h5py
    import numpy as np

    if log is None:
        log = logging.getLogger(__name__)

    h5_path = os.path.join(entry_dir, f"{ids_name}_{occ}.h5")
    grp_name = f"{ids_name}_{occ}"
    if not os.path.exists(h5_path):
        log.warning("space-geometry writer: HDF5 file not found: %s", h5_path)
        return

    r = np.asarray(r_nodes, dtype=np.float64).reshape(-1)
    z = np.asarray(z_nodes, dtype=np.float64).reshape(-1)
    p = np.asarray(phi_nodes, dtype=np.float64).reshape(-1)
    if not (r.size == z.size == p.size):
        raise ValueError("r_nodes, z_nodes, phi_nodes must have same length")
    n = int(r.size)

    geom = np.empty((1, 3, 1, 1, n, 1), dtype=np.float64)
    geom[0, 0, 0, 0, :, 0] = r
    geom[0, 1, 0, 0, :, 0] = z
    geom[0, 2, 0, 0, :, 0] = p

    ds_name = "grid_ggd[]&space[]&objects_per_dimension[]&object[]&geometry"
    with h5py.File(h5_path, "r+") as h5:
        if grp_name not in h5:
            log.warning("space-geometry writer: group '/%s' not found", grp_name)
            return
        g = h5[grp_name]
        if ds_name in g:
            ds = g[ds_name]
            try:
                ds.resize(geom.shape)
            except Exception:
                pass
            ds[...] = geom
        else:
            g.create_dataset(ds_name, data=geom, maxshape=(1, 3, 1, 1, None, 1))

        # Ensure minimal AoS metadata exists for downstream readers (e.g. GGD contour plotters).
        # Some tools rely on these helper datasets to interpret coordinate ordering.
        try:
            nggd, nspace, nopd, nobj, npts, _ = geom.shape

            def _ensure_int_dataset(name: str, payload):
                if name in g:
                    return
                g.create_dataset(name, data=np.asarray(payload, dtype=np.int32), dtype=np.int32)

            _ensure_int_dataset("grid_ggd[]&space[]&objects_per_dimension[]&AOS_SHAPE", [[nggd, nspace, nopd]])
            _ensure_int_dataset("grid_ggd[]&space[]&objects_per_dimension[]&object[]&AOS_SHAPE", [[nggd, nspace, nopd, nobj]])
            _ensure_int_dataset("grid_ggd[]&space[]&coordinates_type[]&AOS_SHAPE", [[nggd, nspace, 1]])

            # Coordinate names and indices (r,z,phi). Write only if missing.
            strdt = h5py.string_dtype("utf-8")
            shp = (nggd, nspace, 1)

            if "grid_ggd[]&space[]&coordinates_type[]&name" not in g:
                nm = np.empty(shp, dtype=object)
                desc = np.empty(shp, dtype=object)
                idx = np.empty(shp, dtype=np.int32)

                base_names = ["r", "z", "phi"]
                base_idx = [4, 3, 5]

                for ig in range(nggd):
                    for ispace in range(nspace):
                        n = base_names[ispace] if ispace < len(base_names) else f"coord{ispace}"
                        nm[ig, ispace, 0] = n
                        desc[ig, ispace, 0] = n
                        idx[ig, ispace, 0] = base_idx[ispace] if ispace < len(base_idx) else -1

                g.create_dataset("grid_ggd[]&space[]&coordinates_type[]&name", data=nm, dtype=strdt)
                g.create_dataset("grid_ggd[]&space[]&coordinates_type[]&description", data=desc, dtype=strdt)
                g.create_dataset("grid_ggd[]&space[]&coordinates_type[]&index", data=idx, dtype=np.int32)

        except Exception as e:
            try:
                log.debug("space-geometry writer: could not write metadata datasets: %s", e)
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
    p.add_argument(
        "--backend",
        default="hdf5",
        help="IMAS backend (e.g. hdf5, ascii, netcdf). Use --writer=imas for non-HDF5 backends.",
    )

    p.add_argument(
        "--writer",
        default="auto",
        choices=("auto", "h5py", "imas"),
        help=(
            "IDS write implementation: "
            "auto=use direct HDF5 patching (h5py) only for backend=hdf5; "
            "h5py=force direct HDF5 patching (fast, HDF5-only); "
            "imas=IMAS-Python only (DD-aware; supports non-HDF5 backends; slower for GGD connectivity)."
        ),
    )
    p.add_argument("--dbpath", default=".", help="DB root path (output directory)")
    p.add_argument("--dd-version", default=None, help="IMAS data dictionary version, e.g. 3.42.0")

    p.add_argument("--mode", default="a", help="DBEntry open mode: r/a/w/x (r+/rw accepted and mapped to a)")

    p.add_argument(
        "--occ-base",
        type=int,
        default=1,
        help="Base occurrence for mhd/mhd_linear, equilibrium, and core_profiles IDSs (species occurrences use occ_base + species_index).",
    )

    p.add_argument("--nbins", type=int, default=256, help="Number of bins for 1D profiles")

    # Optional inputs for LCFS identification / profile normalization
    p.add_argument("--contours", default="contours.h5",
                   help="Optional contours.h5 file (LCFS polyline). If present, LCFS is determined from this file.")
    p.add_argument("--peqdsk", default="peqdsk",
                   help="Optional peqdsk file (TRANSP-style profile table). Used as a fallback to infer Te at psi_N=1.")
    p.add_argument("--te-sep-ev", dest="te_sep_ev", type=float, default=60.0,
                   help="Fallback Te at separatrix [eV] used only if neither contours.h5 nor peqdsk can be used (default 60 eV).")

    # Optional edge extent controls (SOL/PF)
    p.add_argument("--edge-psi-norm-max", dest="edge_psi_norm_max", type=float, default=1.2,
                   help="Override max normalized poloidal flux for edge_profiles 1D grid (include SOL/PF). Default: auto from data.")
    p.add_argument("--edge-psi-norm-quantile", dest="edge_psi_norm_quantile", type=float, default=0.9995,
                   help="Quantile used to estimate max psi_pol_norm from 2D data for edge_profiles (robust against outliers).")

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


    # Edge_profiles GGD controls
    p.add_argument(
        "--edge-ggd-values",
        choices=["equilibrium", "full"],
        default="equilibrium",
        help="When writing edge_profiles.ggd, write only equilibrium-like fields (equilibrium) or include perturbations when available (full).",
    )
    p.add_argument(
        "--pert-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to edge perturbations  with ggd constructed from equilibrium + perturbations (primarily for visualization or debug).",
    )
    p.add_argument(
        "--edge-eq-add-pert",
        action="store_true",
        help="If set, edge_profiles.ggd will be constructed as equilibrium + scaled perturbation (when perturbations are available).",
    )
    p.add_argument(
        "--edge-pert-phi",
        type=float,
        default=0.0,
        help="Toroidal angle (radians) at which to sample perturbations when constructing edge_profiles.ggd in structured mode.",
    )
    # GGD output style: structured (default) vs unstructured-with-connectivity.
    # NOTE: The unstructured option is primarily meant to support robust downstream reconstruction
    # of 3D array shapes and connectivity in environments where the backend stores packed value arrays.

    p.add_argument(
        "--ggd-unstructured",
        action="store_true",
        help=(
            "Write GGD grids in unstructured form using standard IMAS locations: store explicit per-node "
            "(R,Z,phi) coordinates in grid_ggd.space and (optionally) explicit connectivity in "
            "grid_ggd.grid_subset. This mode avoids reliance on implicit structured axes and is intended "
            "for robust downstream use (e.g. ML training)."
        ),
    )

    p.add_argument(
        "--ggd-unstructured-fe-nodes",
        dest="ggd_unstructured_fe_nodes",
        action="store_true",
        help=(
            "When used with --ggd-unstructured, export the native stitched NIMROD finite-element node "
            "locations (R,Z) as the GGD node set (no poloidal resampling). Intended to preserve edge/SOL "
            "mesh packing. Use with --ggd-connectivity fe_tri for a triangulated 2D element representation."
        ),
    )

    p.add_argument(
        "--ggd-connectivity",
        choices=["none", "fe_pointcloud", "hex", "fe_tri", "fe_wedge"],
        default="fe_tri",
        help=(
            "Connectivity type to write when --ggd-unstructured is enabled. "
            "'fe_tri' writes a triangulated 2D connectivity on the native stitched (R,Z) node lattice "
            "(two triangles per valid quad cell), replicated per toroidal plane when Nphi>1. "
            "'fe_wedge' writes a triangular-prism (wedge) volumetric connectivity by extruding the 2D "
            "triangulation between adjacent toroidal planes (periodic in phi). "
            "'hex' writes hexahedral connectivity on the reconstructed (R,Z,phi) product grid with periodicity "
            "in phi (legacy/regular-grid mode). 'fe_pointcloud' writes nodes only and omits connectivity/cells (recommended for very large meshes). Default: fe_tri."
        ),
    )

    p.add_argument(
        "--ggd-reuse-grid",
        action="store_true",
        help=(
            "Assume grid and connectivity are invariant over time. Write grid_ggd geometry/connectivity only for the "
            "first input dump, and reuse it for subsequent time slices (ggd values will reference grid_index=1)."
        ),
    )

    p.add_argument(
        "--mem-limit-gb",
        type=float,
        default=64.0,
        help=(
            "Best-effort hard memory cap for this process in GB (Linux RLIMIT_AS). Default 32. "
            "Use to reduce risk of OS-level OOM/reboots on very large GGD exports."
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
    p.add_argument("--power-density-scale", type=float, default=1.0,
                   help="Scale factor applied to power-density-like quantities from dump -> IMAS (e.g., W/m^3).")


    p.add_argument(
        "--sanity-print",
        dest="sanity_print",
        action="store_true",
        help=(
            "Print sanity statistics before and after the internal COCOS=12 -> COCOS=11 conversion "
            "(psi_eq, Bphi/Jphi/Vphi and representative perturbation components). Useful for quick sign checks."
        ),
    )

    p.add_argument("--quiet", action="store_true")

    p.add_argument("--no-checksums", dest="record_checksums", action="store_false",
                   help="Disable provenance file checksums in workflow/dataset_fair IDSs.")
    p.set_defaults(record_checksums=True)
    p.add_argument("--checksum-algorithm", default="sha256",
                   help="Hash algorithm for provenance checksums (sha256, sha1, md5, ...).")


    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # Normalize writer/backend combination.
    args.writer = str(getattr(args, "writer", "auto") or "auto").strip().lower()
    if args.writer not in ("auto", "h5py", "imas"):
        args.writer = "auto"
    args.backend = str(getattr(args, "backend", "hdf5") or "hdf5").strip()
    if (str(args.backend).strip().lower() != "hdf5") and (args.writer == "h5py"):
        # h5py patching can only target HDF5 backend directories.
        args.writer = "imas"

    # Logging (explicitly show which psi reconstruction/fallback path is selected)
    logging.basicConfig(
        level=(logging.WARNING if getattr(args, "quiet", False) else logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("dump2imas")

    # Safety guard: best-effort process memory cap (Linux RLIMIT_AS).
    try:
        mem_gb = float(getattr(args, "mem_limit_gb", 0.0) or 0.0)
    except Exception:
        mem_gb = 0.0
    args._mem_limit_bytes = int(mem_gb * (1024**3)) if mem_gb and mem_gb > 0 else None
    if mem_gb and mem_gb > 0:
        try:
            import resource
            limit = int(mem_gb * (1024**3))
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            log.info("Applied RLIMIT_AS memory cap: %.1f GB", mem_gb)
        except Exception as e:
            log.warning("Could not apply RLIMIT_AS memory cap: %s", e)

    dump_files = [Path(x).expanduser().resolve() for x in args.dumpgll]
    for fn in dump_files:
        if not fn.exists():
            _die(f"Input file not found: {fn}")

    dd_version = args.dd_version
    if (dd_version == None):
         dd_version = os.environ.get("IMAS_VERSION", '4.1.1')

    imas = _import_imas()
    mode = _normalize_mode(str(args.mode))

    dbpath = Path(args.dbpath).expanduser().resolve()
    entry_dir = _entry_dir(dbpath, str(args.dd), str(dd_version), int(args.pulse), int(args.run), dd_version_dir=str(args.dd_version[0]))

    # Make entry_dir visible to helper routines (LCFS identification fallbacks)
    setattr(args, "_entry_dir", str(entry_dir))

    _log(f"IMAS DB root: {dbpath}", args.quiet)
    _log(f"IMAS entry directory: {entry_dir}", args.quiet)

    db = _open_db(imas, args.backend, entry_dir, mode, str(dd_version))
    factory = _ids_factory(imas, str(dd_version))
    
    # If input2imas ran before dump2imas, equilibrium/core_profiles may already exist in this entry.
    # For --edge-ggd-values=equilibrium, prefer those preprocessed 1D profiles as the equilibrium source
    # (avoids relying on external peqdsk discovery and avoids COCOS/sign mismatches).
    try:
        occ_try = int(getattr(args, 'occ_base', 1) or 1)
    except Exception:
        occ_try = 1

    args._preproc_occ = None
    args._preproc_eq = None
    args._preproc_cp = None

    try:
        cp0 = _db_get(db, factory, 'core_profiles', occ_try)
        eq0 = _db_get(db, factory, 'equilibrium', occ_try)
        occ_found = occ_try if (cp0 is not None or eq0 is not None) else None

        # Common workflow: input2imas writes occ=0, while dump2imas default occ_base=1.
        if occ_found is None and occ_try != 0:
            cp0 = cp0 or _db_get(db, factory, 'core_profiles', 0)
            eq0 = eq0 or _db_get(db, factory, 'equilibrium', 0)
            if cp0 is not None or eq0 is not None:
                occ_found = 0

        args._preproc_occ = occ_found
        args._preproc_cp = cp0
        args._preproc_eq = eq0

        if occ_found is not None:
            log.info('Preloaded existing equilibrium/core_profiles from IMAS entry (occ=%d) for edge_profiles equilibrium sourcing', int(occ_found))
    except Exception:
        pass


    # Important implementation detail:
    # We append time slices using put_slice() to avoid corruption issues observed
    # when repeatedly db.get() -> resize(AOS) -> db.put() on HDF5 entries.
    # (Symptom: only the last time slice is valid; earlier slices read back as
    # huge nonsensical floating point values.)

    for ifile, fn in enumerate(dump_files, start=1):

        # Directory of the current dump file (used to resolve optional inputs like contours.h5/peqdsk)
        setattr(args, "_run_dir", str(fn.parent))
        # Communicate per-file grid-write policy to GGD helpers.
        # When --ggd-reuse-grid is enabled, we only write grid_ggd on the first processed dump.
        setattr(args, "_ggd_write_grid", (ifile == 1))

        _log(f"Reading {fn.name} ({ifile}/{len(dump_files)})", args.quiet)
        data = read_and_stitch_dump(fn, args)

        lay = data["layout"]
        _log(
            f"  blocks={lay['nxbl']}x{lay['nybl']} ordering={lay['ordering']} local(ny,nx)=({lay['ny_loc']},{lay['nx_loc']}) global(Ny,Nx)=({lay['Ny']},{lay['Nx']})",
            args.quiet,
        )
        _log(f"  nmodes={data['nmodes']} time={data['time']}", args.quiet)
        # Ensure psi is available for 1D profile grids. Prefer dump psi, then B-field reconstruction,
        # then an existing preprocessing mhd IDS (contains psi for some workflows).
        _ensure_psi_eq_available(data, entry_dir, occ_base=int(getattr(args, "occ_base", 0) or 0), log=log)

        # Determine whether this directory corresponds to a nonlinear run (nimrod.in) and choose IDS type.
        nimrod_in_path = None
        try:
            cand = fn.parent / "nimrod.in"
            if cand.is_file():
                nimrod_in_path = str(cand)
        except Exception:
            nimrod_in_path = None

        nonlinear_flag = _nimrod_in_nonlinear(nimrod_in_path)
        # Record run type for downstream helpers (edge/core perturbations only for nonlinear runs).
        setattr(args, "_nonlinear_run", bool(nonlinear_flag is True))
        if (not getattr(args, "_nonlinear_run", False)) and str(getattr(args, "edge_ggd_values", "equilibrium") or "equilibrium").strip().lower() == "full":
            log.info("Linear run detected: forcing core_profiles/edge_profiles to equilibrium-only (ignoring --edge-ggd-values=full). Perturbations are written only to mhd_linear.")
        try:
            args._nimrod_species = _nimrod_species_info(nimrod_in_path)
        except Exception:
            args._nimrod_species = {}

        # For nonlinear runs (nimrod.in present and nonlinear=T/true), write BOTH:
        #   - mhd_linear: mode-resolved perturbations (multiple occurrences for multiple species)
        #   - mhd:        reconstructed full fields in GGD form (time-dependent)
        # For linear runs, write only mhd_linear.
        write_mhd = (nonlinear_flag is True)

        # Linear runs: mhd IDS is not written. For linear runs we ALWAYS keep core_profiles and edge_profiles
        # equilibrium-only, regardless of --edge-ggd-values, to avoid contaminating equilibrium profiles with
        # eigenfunction content. (Perturbations are stored only in mhd_linear.)

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
        if hasattr(eq, "time_slice") and _aos_has_entries(eq.time_slice):
            _db_put_slice(db, eq, eq_occ)
            log.info("Wrote equilibrium (occ=%d)", int(eq_occ))

        cp = factory.new("core_profiles") if hasattr(factory, "new") else factory("core_profiles")
        populate_core_profiles(cp, data, t_index=0, args=args)
        populate_core_profiles_ggd(cp, data, args=args)

        try:
            has_1d = hasattr(cp, "profiles_1d") and _aos_has_entries(cp.profiles_1d)
        except Exception:
            has_1d = False
        try:
            has_ggd = hasattr(cp, "ggd") and hasattr(cp, "grid_ggd")
            try:
                has_ggd = has_ggd and _aos_has_entries(cp.ggd)
            except Exception:
                has_ggd = has_ggd
        except Exception:
            has_ggd = False

        # Write core_profiles if we populated either profiles_1d or ggd (avoid empty placeholders)
        if has_1d or has_ggd:
            _db_put_slice(db, cp, cp_occ)
            log.info("Wrote core_profiles (occ=%d)", int(cp_occ))
            if has_ggd and _use_h5py_patches(args):
                _patch_core_profiles_ggd_required_leaves_h5(str(entry_dir), int(cp_occ), data, args, log)


        # edge_profiles: equilibrium-only (profiles_1d + optional GGD)
        # For linear simulations, write core_profiles/edge_profiles only once when converting dumpgll.00000.h5.
        do_profiles_once = True
        try:
            is_nonlinear = bool(nonlinear_flag)
        except Exception:
            is_nonlinear = False
        if (not is_nonlinear) and (ifile > 1):
            do_profiles_once = False

        if do_profiles_once:
            ep = factory.new("edge_profiles") if hasattr(factory, "new") else factory("edge_profiles")
            populate_edge_profiles(ep, data, t_index=0, args=args)
            populate_edge_profiles_ggd(ep, data, args=args)

            try:
                has_1d = hasattr(ep, "profiles_1d") and _aos_has_entries(ep.profiles_1d)
            except Exception:
                has_1d = False
            try:
                # Some IMAS python bindings do not implement __len__ reliably for AOS containers.
                # If ggd/grid_ggd exist but len() fails, assume GGD was populated and proceed with post-write geometry patching.
                has_ggd = hasattr(ep, "ggd") and hasattr(ep, "grid_ggd")
                try:
                    has_ggd = has_ggd and (_aos_has_entries(ep.ggd))
                except Exception:
                    has_ggd = has_ggd
            except Exception:
                has_ggd = False

            try:
                if has_1d or has_ggd:
                    
                    try:
                        _db_put_slice(db, ep, occ_base)
                        log.info("Wrote edge_profiles (occ=%d)", int(occ_base))
                    except Exception as e:
                        log.error("Failed to write edge_profiles (occ=%d): %s", int(occ_base), e, exc_info=True)
                        raise

                    # Always ensure grid_ggd node geometry vectors exist in the IMAS backend for edge_profiles.
                    # plot_mhd.py expects the dataset:
                    #   /edge_profiles_<occ>/grid_ggd[]&space[]&objects_per_dimension[]&object[]&geometry
                    # The in-memory IDS population in structured mode may not materialize this dataset reliably,
                    # so we write it explicitly via h5py.
                    if has_ggd and _use_h5py_patches(args):
                        try:
                            nb = max(4, int(getattr(args, 'ggd_nbins', 128) or 128))
                            conn_kind_ep = str(getattr(args, 'ggd_connectivity', 'none')).lower()
                            use_unstructured_nodes = (
                                bool(getattr(args, 'ggd_unstructured', False))
                                and bool(getattr(args, 'ggd_unstructured_fe_nodes', False))
                                and (conn_kind_ep in ('fe_tri', 'fe_wedge', 'fe_pointcloud'))
                            )

                            if use_unstructured_nodes:
                                Rbase = np.asarray(data.get('R'), dtype=float)
                                Zbase = np.asarray(data.get('Z'), dtype=float)
                                r_nodes = Rbase.ravel(order='F')
                                z_nodes = Zbase.ravel(order='F')
                                phi_nodes = np.zeros_like(r_nodes)
                            else:
                                # Match the regular (R,Z) grid used by populate_edge_profiles_ggd() via _interp2d_linear().
                                Rm = np.asarray(data.get('R'), dtype=float)
                                Zm = np.asarray(data.get('Z'), dtype=float)
                                Vref = data.get('psi_eq', None)
                                if Vref is None:
                                    Vref = data.get('teq', None)
                                if Vref is None:
                                    Vref = Rm
                                rc, zc, _ = _interp2d_linear(Rm, Zm, np.asarray(Vref, dtype=float), nb, nb)
                                RR, ZZ = np.meshgrid(rc, zc, indexing='ij')
                                r_nodes = RR.ravel(order='F')
                                z_nodes = ZZ.ravel(order='F')
                                phi_nodes = np.zeros_like(r_nodes)

                            _write_gridggd_space_geometry_vectors_h5(
                                str(entry_dir),
                                'edge_profiles',
                                occ_base,
                                r_nodes,
                                z_nodes,
                                phi_nodes,
                                log=log,
                            )
                            log.info("edge_profiles: wrote grid_ggd.space geometry vectors (N=%d)", int(np.size(r_nodes)))
                        except Exception as e:
                            log.warning("edge_profiles: failed to ensure grid_ggd geometry vectors: %s", e)

                    # Patch missing electrons.density in edge_profiles.ggd for structured (product-grid) outputs.
                    # Some DD/python bindings omit this leaf; downstream tools may recompute ne from (p,T) and get wrong units.
                    if has_ggd and _use_h5py_patches(args) and (not bool(getattr(args, 'ggd_unstructured', False))):
                        try:
                            nb = max(4, int(getattr(args, 'ggd_nbins', 128) or 128))
                            Rm = np.asarray(data.get('R'), dtype=float)
                            Zm = np.asarray(data.get('Z'), dtype=float)
                            nq = data.get('nq', None)
                            ne_raw = None
                            if nq is not None:
                                nqA = np.asarray(nq, dtype=float)
                                n_scale = float(getattr(args, "n_scale", 1.0) or 1.0)
                                nqA = np.asarray(nqA, dtype=float) * float(n_scale)
                                if nqA.ndim >= 3 and nqA.shape[-1] >= 1:
                                    ne_raw = nqA[..., 0]
                                elif nqA.ndim == 2:
                                    ne_raw = nqA
                            if ne_raw is not None:
                                rc, zc, ne_grid = _interp2d_linear(Rm, Zm, np.asarray(ne_raw, dtype=float), nb, nb)
                                shape_rzp = (int(ne_grid.shape[0]), int(ne_grid.shape[1]), 1)
                                _write_edge_profiles_electrons_density_ggd_h5(
                                    str(entry_dir), occ_base,
                                    values_1d=np.asarray(ne_grid, dtype=float).ravel(order='F'),
                                    shape_rzp=shape_rzp,
                                    grid_index=1,
                                    grid_subset_index=0,
                                    overwrite=True,
                                )
                                log.info("edge_profiles: patched ggd.electrons.density (product grid)")
                        except Exception as e:
                            log.warning("edge_profiles: failed to patch ggd.electrons.density: %s", e)

                    # Ensure edge_profiles includes GGD node geometry and electrons.temperature values and electrons.temperature values
                    # in standard IMAS backend paths (needed by plot_mhd.py).

                    if _use_h5py_patches(args) and bool(getattr(args, 'ggd_unstructured', False)):
                        conn_kind_ep = str(getattr(args, 'ggd_connectivity', 'none')).lower()
                        use_unstructured_nodes = bool(getattr(args, 'ggd_unstructured_fe_nodes', False)) and conn_kind_ep in ('fe_tri', 'fe_wedge', 'fe_pointcloud')
                        use_fe_hex = (conn_kind_ep == 'hex')

                        if use_unstructured_nodes or use_fe_hex:
                            ovw = not use_fe_hex  # fe_hex: avoid overwriting existing packed leaves
                            try:
                                sp = getattr(args, '_nimrod_species', {}) or {}
                                qe = float(sp.get('qe_c', 1.602176634e-19))
                                z_ions = sp.get('z_ions', None)

                                # Grid definition:
                                #  - unstructured nodes: use native (R,Z) mesh nodes
                                #  - fe_hex: interpolate/bucket onto regular (R,Z) grid (same nb as writer)
                                if use_unstructured_nodes:
                                    Rbase = np.asarray(data.get('R'), dtype=float)
                                    Zbase = np.asarray(data.get('Z'), dtype=float)
                                    shape_rzp = (int(Rbase.shape[0]), int(Rbase.shape[1]), 1)
                                    r_nodes = Rbase.ravel(order='F')
                                    z_nodes = Zbase.ravel(order='F')
                                    phi_nodes = np.zeros_like(r_nodes)
                                    try:
                                        _write_gridggd_space_geometry_vectors_h5(str(entry_dir), 'edge_profiles', occ_base, r_nodes, z_nodes, phi_nodes, log=log)
                                    except Exception as e:
                                        log.warning('edge_profiles space-geometry write failed: %s', e)

                                    def _to_grid(arr2d):
                                        if arr2d is None:
                                            return None
                                        a = np.asarray(arr2d, dtype=float)
                                        if a.shape != Rbase.shape and a.T.shape == Rbase.shape:
                                            a = a.T
                                        return a

                                else:
                                    nb = max(4, int(getattr(args, 'ggd_nbins', 128) or 128))
                                    Rm = np.asarray(data.get('R'), dtype=float)
                                    Zm = np.asarray(data.get('Z'), dtype=float)
                                    r_axis, z_axis, phi_axis, _nodes = _compute_product_grid_axes_nodes({'R': Rm, 'Z': Zm}, args, nb=nb, nphi=1)
                                    # Write product-grid node geometry into grid_ggd.space so readers can locate values.
                                    try:
                                        _write_gridggd_space_geometry_vectors_h5(
                                            str(entry_dir),
                                            'edge_profiles',
                                            occ_base,
                                            _nodes[:, 0],
                                            _nodes[:, 1],
                                            _nodes[:, 2],
                                            log=log,
                                        )
                                    except Exception as e:
                                        log.warning('edge_profiles space-geometry write failed (product grid): %s', e)

                                    rr, zz = np.meshgrid(r_axis, z_axis, indexing='ij')
                                    shape_rzp = (int(rr.shape[0]), int(rr.shape[1]), 1)

                                    def _to_grid(arr2d):
                                        if arr2d is None:
                                            return None
                                        a = np.asarray(arr2d, dtype=float)
                                        if a.shape != Rm.shape and a.T.shape == Rm.shape:
                                            a = a.T
                                        return _interp_mesh_to_points(Rm, Zm, a, rr, zz)

                                # electrons: density, pressure, temperature
                                nq = data.get('nq', None)
                                peq = data.get('peq', None)
                                teq = data.get('teq', None)
                                # If requested, reconstruct full-field edge quantities at phi=0 using modal coefficients.
                                edge_mode_ep = str(getattr(args, 'edge_ggd_values', 'equilibrium') or 'equilibrium').strip().lower()
                                if not bool(getattr(args, '_nonlinear_run', False)):
                                    edge_mode_ep = 'equilibrium'
                                if edge_mode_ep == 'full' and bool(getattr(args, '_nonlinear_run', False)):
                                    fields_ep = data.get('fields', {}) or {}
                                    keff_ep = np.asarray(data.get('keff'), dtype=float) if data.get('keff') is not None else np.arange(int(data.get('nmodes', 0) or 0), dtype=float)
                                    phi0_ep = 0.0
                                    try:
                                        pscl_ep = float(getattr(args, 'pert_scale', 1.0) or 1.0)
                                    except Exception:
                                        pscl_ep = 1.0
                                
                                    def _slice_modes_ep(A, *, spec=None, comp=None):
                                        if A is None:
                                            return None
                                        AA = np.asarray(A)
                                        nm = int(len(keff_ep))
                                        if comp is not None:
                                            c = int(comp)
                                            if AA.ndim == 4:
                                                if AA.shape[-1] == 3 and AA.shape[-2] == nm:
                                                    return AA[:, :, :, c]
                                                if AA.shape[2] == 3 and AA.shape[3] == nm:
                                                    return AA[:, :, c, :]
                                            if AA.ndim == 3 and AA.shape[-1] == 3 * nm:
                                                return AA[:, :, c * nm:(c + 1) * nm]
                                            return None
                                        if spec is not None:
                                            s = int(spec)
                                            if AA.ndim == 4:
                                                if AA.shape[-1] == nm:
                                                    return AA[:, :, s, :] if (0 <= s < AA.shape[2]) else None
                                                if AA.shape[2] == nm:
                                                    return AA[:, :, :, s] if (0 <= s < AA.shape[3]) else None
                                            return None
                                        if AA.ndim == 3 and AA.shape[-1] == nm:
                                            return AA
                                        return None
                                
                                    def _recon_ep(eq2d, rekey, imkey, *, spec=None, comp=None):
                                        reA = _slice_modes_ep(fields_ep.get(rekey), spec=spec, comp=comp)
                                        imA = _slice_modes_ep(fields_ep.get(imkey), spec=spec, comp=comp)
                                        if eq2d is None and (reA is None or imA is None):
                                            return None
                                        return _reconstruct_full_from_modes(eq2d, reA, imA, keff_ep, float(phi0_ep), pert_scale=pscl_ep)
                                
                                    # Override native-mesh equilibrium arrays (before mapping to the chosen GGD grid)
                                    if nq is not None:
                                        try:
                                            nqA_ep = np.asarray(nq, dtype=float) * float(getattr(args, 'n_scale', 1.0) or 1.0)
                                            if nqA_ep.ndim == 2:
                                                ne_native = nqA_ep
                                            elif nqA_ep.ndim >= 3 and nqA_ep.shape[-1] >= 1:
                                                ne_native = nqA_ep[..., 0]
                                            else:
                                                ne_native = None
                                        except Exception:
                                            ne_native = None
                                        if ne_native is not None:
                                            ne_native_full = _recon_ep(ne_native, 'rend', 'imnd', spec=0)
                                            if ne_native_full is not None:
                                                if 'nqA_ep' in locals() and getattr(nqA_ep, 'ndim', 0) >= 3 and nqA_ep.shape[-1] >= 1:
                                                    nqA_ep[..., 0] = np.asarray(ne_native_full, dtype=float)
                                                ne_native = ne_native_full
                                
                                    peq = _recon_ep(peq, 'repe', 'impe') or peq
                                    teq = _recon_ep(teq, 'rete', 'imte') or teq
                                
                                    prq = _recon_ep(data.get('prq', None), 'repr', 'impr') or data.get('prq', None)
                                    tiq = _recon_ep(data.get('tiq', None), 'reti', 'imti') or data.get('tiq', None)
                                
                                    vq_full = None
                                    if data.get('vq', None) is not None:
                                        try:
                                            vqA0 = np.asarray(data.get('vq'), dtype=float)
                                            vq_full = np.zeros_like(vqA0, dtype=float)
                                            for _c in (0, 1, 2):
                                                vq_full[..., _c] = _recon_ep(vqA0[..., _c], 'reve', 'imve', comp=_c) or vqA0[..., _c]
                                        except Exception:
                                            vq_full = None
                                    jq_full = None
                                    if data.get('jq', None) is not None:
                                        try:
                                            jqA0 = np.asarray(data.get('jq'), dtype=float)
                                            jq_full = np.zeros_like(jqA0, dtype=float)
                                            for _c in (0, 1, 2):
                                                jq_full[..., _c] = _recon_ep(jqA0[..., _c], 'reja', 'imja', comp=_c) or jqA0[..., _c]
                                        except Exception:
                                            jq_full = None
                                
                                    if vq_full is not None:
                                        vq = vq_full
                                    if jq_full is not None:
                                        jq = jq_full
                                    data_prq_ep = prq
                                    data_tiq_ep = tiq
                                ne2d = None
                                if nq is not None:
                                    n_scale = float(getattr(args, "n_scale", 1.0) or 1.0)
                                    nqA = locals().get('nqA_ep', (np.asarray(nq, dtype=float) * float(n_scale)))
                                    if nqA.ndim == 2:
                                        ne2d = _to_grid(nqA)
                                    elif nqA.ndim >= 3 and nqA.shape[-1] >= 1:
                                        ne2d = _to_grid(np.asarray(nqA[..., 0], dtype=float))
                                pe2d = _to_grid(peq)

                                if ne2d is not None:
                                    _write_edge_profiles_electrons_density_ggd_h5(
                                        str(entry_dir), occ_base,
                                        values_1d=np.asarray(ne2d, dtype=float).ravel(order='F'),
                                        shape_rzp=shape_rzp,
                                        grid_index=1,
                                        grid_subset_index=0,
                                        overwrite=ovw,
                                    )

                                if pe2d is not None:
                                    _write_edge_profiles_electrons_pressure_ggd_h5(
                                        str(entry_dir), occ_base,
                                        values_1d=np.asarray(pe2d, dtype=float).ravel(order='F'),
                                        shape_rzp=shape_rzp,
                                        grid_index=1,
                                        grid_subset_index=0,
                                        overwrite=ovw,
                                    )

                                te2d = _to_grid(teq)
                                if te2d is None and (pe2d is not None) and (ne2d is not None):
                                    with np.errstate(divide='ignore', invalid='ignore'):
                                        te2d = np.asarray(pe2d, dtype=float) / (np.asarray(ne2d, dtype=float) * qe)

                                if te2d is not None:
                                    _write_edge_profiles_electrons_temperature_ggd_h5(
                                        str(entry_dir), occ_base,
                                        values_1d=np.asarray(te2d, dtype=float).ravel(order='F'),
                                        shape_rzp=shape_rzp,
                                        grid_index=1,
                                        grid_subset_index=0,
                                        overwrite=ovw,
                                    )

                                # ions: density, pressure, temperature, velocity, z_ion, t_i_average
                                prq = locals().get('data_prq_ep', data.get('prq', None))
                                tiq = locals().get('data_tiq_ep', data.get('tiq', None))
                                pi2d = None
                                if (prq is not None) and (peq is not None):
                                    pi2d = _to_grid(np.asarray(prq, dtype=float) - np.asarray(peq, dtype=float))

                                nqA = locals().get('nqA_ep', (np.asarray(nq, dtype=float) if nq is not None else None))
                                nion = 1
                                if nqA is not None and nqA.ndim >= 3 and nqA.shape[-1] >= 2:
                                    nion = int(nqA.shape[-1] - 1)
                                elif z_ions is not None:
                                    try:
                                        nion = max(1, int(np.size(z_ions)))
                                    except Exception:
                                        nion = 1

                                dens_list = []
                                if nqA is not None and nqA.ndim >= 3 and nqA.shape[-1] >= 2:
                                    for k in range(nion):
                                        _ni_native = np.asarray(nqA[..., 1 + k], dtype=float)
                                        if locals().get('edge_mode_ep', 'equilibrium') == 'full':
                                            try:
                                                _ni_full = _recon_ep(_ni_native, 'rend', 'imnd', spec=1 + k)
                                                if _ni_full is not None:
                                                    _ni_native = np.asarray(_ni_full, dtype=float)
                                            except Exception:
                                                pass
                                        dens_list.append(_to_grid(_ni_native))
                                elif ne2d is not None:
                                    dens_list = [np.asarray(ne2d, dtype=float)]
                                else:
                                    dens_list = [np.full(shape_rzp[:2], np.nan, dtype=float)]

                                # t_i_average
                                ti2d = _to_grid(tiq)
                                if ti2d is None and (pi2d is not None):
                                    ni_total = np.zeros(shape_rzp[:2], dtype=float)
                                    for a in dens_list:
                                        ni_total += np.asarray(a, dtype=float)
                                    with np.errstate(divide='ignore', invalid='ignore'):
                                        ti2d = np.asarray(pi2d, dtype=float) / (ni_total * qe)
                                if ti2d is not None:
                                    _write_edge_profiles_t_i_average_ggd_h5(
                                        str(entry_dir), occ_base,
                                        values_1d=np.asarray(ti2d, dtype=float).ravel(order='F'),
                                        shape_rzp=shape_rzp,
                                        grid_index=1,
                                        grid_subset_index=0,
                                        overwrite=ovw,
                                    )
                                else:
                                    ti2d = np.full(shape_rzp[:2], np.nan, dtype=float)

                                # per-ion pressure distribution: proportional to ni_k
                                pres_list = []
                                if pi2d is not None:
                                    ni_total = np.zeros(shape_rzp[:2], dtype=float)
                                    for a in dens_list:
                                        ni_total += np.asarray(a, dtype=float)
                                    with np.errstate(divide='ignore', invalid='ignore'):
                                        for a in dens_list:
                                            frac = np.asarray(a, dtype=float) / ni_total
                                            pres_list.append(np.asarray(pi2d, dtype=float) * frac)
                                else:
                                    pres_list = [np.full(shape_rzp[:2], np.nan, dtype=float) for _ in dens_list]

                                def _cat(vals):
                                    return np.concatenate([np.asarray(v, dtype=float).ravel(order='F') for v in vals], axis=0)

                                _write_edge_profiles_ion_scalar_ggd_h5(
                                    str(entry_dir), occ_base,
                                    field='density',
                                    values_all_1d=_cat(dens_list),
                                    shape_rzp=shape_rzp,
                                    nion=len(dens_list),
                                    grid_index=1,
                                    grid_subset_index=0,
                                    overwrite=ovw,
                                )
                                _write_edge_profiles_ion_scalar_ggd_h5(
                                    str(entry_dir), occ_base,
                                    field='pressure',
                                    values_all_1d=_cat(pres_list),
                                    shape_rzp=shape_rzp,
                                    nion=len(pres_list),
                                    grid_index=1,
                                    grid_subset_index=0,
                                    overwrite=ovw,
                                )
                                _write_edge_profiles_ion_scalar_ggd_h5(
                                    str(entry_dir), occ_base,
                                    field='temperature',
                                    values_all_1d=np.tile(np.asarray(ti2d, dtype=float).ravel(order='F'), len(dens_list)),
                                    shape_rzp=shape_rzp,
                                    nion=len(dens_list),
                                    grid_index=1,
                                    grid_subset_index=0,
                                    overwrite=ovw,
                                )

                                # ion velocity: replicate bulk v for all ions
                                vq = (vq if vq is not None else data.get('vq', None))
                                if vq is not None:
                                    vqA = np.asarray(vq, dtype=float)
                                    if vqA.ndim >= 3 and vqA.shape[-1] >= 3:
                                        vR = _to_grid(vqA[..., 0])
                                        vZ = _to_grid(vqA[..., 1])
                                        vP = _to_grid(vqA[..., 2])
                                        _write_edge_profiles_ion_velocity_component_ggd_h5(
                                            str(entry_dir), occ_base,
                                            component='r',
                                            values_1d=np.asarray(vR, dtype=float).ravel(order='F'),
                                            shape_rzp=shape_rzp,
                                            nion=len(dens_list),
                                            grid_index=1,
                                            grid_subset_index=0,
                                            overwrite=ovw,
                                        )
                                        _write_edge_profiles_ion_velocity_component_ggd_h5(
                                            str(entry_dir), occ_base,
                                            component='z',
                                            values_1d=np.asarray(vZ, dtype=float).ravel(order='F'),
                                            shape_rzp=shape_rzp,
                                            nion=len(dens_list),
                                            grid_index=1,
                                            grid_subset_index=0,
                                            overwrite=ovw,
                                        )
                                        _write_edge_profiles_ion_velocity_component_ggd_h5(
                                            str(entry_dir), occ_base,
                                            component='phi',
                                            values_1d=np.asarray(vP, dtype=float).ravel(order='F'),
                                            shape_rzp=shape_rzp,
                                            nion=len(dens_list),
                                            grid_index=1,
                                            grid_subset_index=0,
                                            overwrite=ovw,
                                        )

                                # z_ion
                                if z_ions is None:
                                    z_ions = np.arange(1, 1 + len(dens_list), dtype=np.int32)
                                _write_edge_profiles_ion_z_ion_ggd_h5(str(entry_dir), occ_base, z_ions=z_ions, overwrite=ovw)
                                # zeff (per-grid for impurities, constant for no-impurity cases) and n_i_total_over_n_e
                                try:
                                    # Electron density on the same RZ grid; if missing, estimate from quasi-neutrality
                                    if ne2d is None:
                                        ne_use = np.zeros_like(ni_total, dtype=float)
                                        if z_ions is None or len(z_ions) != len(dens_list):
                                            z_use = [1.0] * len(dens_list)
                                        else:
                                            z_use = [float(z) for z in z_ions]
                                        for _n, _z in zip(dens_list, z_use):
                                            ne_use += _z * np.asarray(_n, dtype=float)
                                    else:
                                        ne_use = np.asarray(ne2d, dtype=float)
                                
                                    with np.errstate(divide='ignore', invalid='ignore'):
                                        ni_over_ne = np.where(ne_use != 0.0, np.asarray(ni_total, dtype=float) / ne_use, np.nan)
                                        if z_ions is None or len(z_ions) != len(dens_list):
                                            zeff2d = np.ones_like(ni_over_ne, dtype=float)
                                        else:
                                            num = np.zeros_like(ne_use, dtype=float)
                                            for _n, _z in zip(dens_list, [float(z) for z in z_ions]):
                                                num += (_z * _z) * np.asarray(_n, dtype=float)
                                            zeff2d = np.where(ne_use != 0.0, num / ne_use, np.nan)
                                        # Enforce constant Zeff=1 for no-impurity runs (single Z=1 thermal ion)
                                        if z_ions is not None and len(z_ions) == 1 and int(z_ions[0]) == 1:
                                            zeff2d = np.ones_like(ni_over_ne, dtype=float)
                                
                                    _write_edge_profiles_zeff_ggd_h5(str(entry_dir), occ_base,
                                                                   values_1d=zeff2d.ravel(order='F'),
                                                                   shape_rzp=shape_rzp, grid_index=1, grid_subset_index=0, overwrite=ovw)
                                    _write_edge_profiles_n_i_total_over_n_e_ggd_h5(str(entry_dir), occ_base,
                                                                   values_1d=ni_over_ne.ravel(order='F'),
                                                                   shape_rzp=shape_rzp, grid_index=1, grid_subset_index=0, overwrite=ovw)
                                except Exception:
                                    pass

                                # total current density components (R, Z, phi) from equilibrium jq
                                try:
                                    jq = (jq if jq is not None else data.get('jq', None))
                                    if jq is not None:
                                        jqA = np.asarray(jq, dtype=float)
                                        if jqA.ndim >= 3 and jqA.shape[-1] >= 3:
                                            jR2d = _to_grid(jqA[..., 0])
                                            jZ2d = _to_grid(jqA[..., 1])
                                            jP2d = _to_grid(jqA[..., 2])
                                        elif jqA.ndim == 2 and jqA.shape[-1] >= 3:
                                            # flattened points -> reshape back to (nr,nz) using the same ordering as other fields
                                            jR2d = np.asarray(jqA[:, 0], dtype=float).reshape(shape_rzp[:2], order='F')
                                            jZ2d = np.asarray(jqA[:, 1], dtype=float).reshape(shape_rzp[:2], order='F')
                                            jP2d = np.asarray(jqA[:, 2], dtype=float).reshape(shape_rzp[:2], order='F')
                                        else:
                                            jR2d = jZ2d = jP2d = None
                                        if jR2d is not None and jZ2d is not None and jP2d is not None:
                                            _write_edge_profiles_j_total_component_ggd_h5(
                                                str(entry_dir), occ_base, component='r',
                                                values_1d=np.asarray(jR2d, dtype=float).ravel(order='F'),
                                                shape_rzp=shape_rzp, grid_index=1, grid_subset_index=0, overwrite=ovw
                                            )
                                            _write_edge_profiles_j_total_component_ggd_h5(
                                                str(entry_dir), occ_base, component='z',
                                                values_1d=np.asarray(jZ2d, dtype=float).ravel(order='F'),
                                                shape_rzp=shape_rzp, grid_index=1, grid_subset_index=0, overwrite=ovw
                                            )
                                            _write_edge_profiles_j_total_component_ggd_h5(
                                                str(entry_dir), occ_base, component='phi',
                                                values_1d=np.asarray(jP2d, dtype=float).ravel(order='F'),
                                                shape_rzp=shape_rzp, grid_index=1, grid_subset_index=0, overwrite=ovw
                                            )
                                except Exception:
                                    pass


                            except Exception as e:
                                log.warning('edge_profiles GGD required leaves (h5py direct) write failed: %s', e)
                    # If unstructured GGD is requested, we'll mirror edge_profiles electrons.temperature from the mhd GGD
                    # after the mhd IDS is written (see below). This avoids fragile interpolation and keeps node ordering identical.
                    edge_profiles_need_mirror = bool(getattr(args, 'ggd_unstructured', False) and (str(getattr(args, 'edge_ggd_values', 'equilibrium') or 'equilibrium').strip().lower() == 'full'))
            except Exception as e:
                log.error('edge_profiles processing failed: %s', e, exc_info=True)
                raise
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
                    f"dens_pert_order={args.dens_pert_order}; common_fields={include_common}; nonlinear={nonlinear_flag}; "
                    "COCOS: converted NIMROD COCOS=12 -> IMAS COCOS=11 (psi sign flipped; toroidal components flipped; "
                    "modal perturbations mapped for phi-reversal)"
                )
            except Exception:
                pass

            # Best-effort: tag IDS with output COCOS metadata when supported by this DD
            _set_ids_cocos(mhd_ids, COCOS_OUT_DEFAULT)

            _db_put_slice(db, mhd_ids, occ)
        # For nonlinear runs, also write the GGD-based mhd IDS (full fields) alongside mhd_linear.
        if write_mhd and (mhd is not None):
            try:
                # Write one mhd IDS occurrence per species (aligned with mhd_linear):
                #   occ_base+0 : electrons/common
                #   occ_base+1.. : ion species (main + impurities)
                sp = getattr(args, '_nimrod_species', {}) or {}
                nion = 1
                try:
                    z_ions = sp.get('z_ions', []) or []
                    if len(z_ions) > 0:
                        nion = int(len(z_ions))
                    else:
                        nqA = np.asarray(data.get('nq'), dtype=float) if data.get('nq') is not None else None
                        if nqA is not None and nqA.ndim >= 3 and int(nqA.shape[2]) >= 2:
                            nion = int(nqA.shape[2] - 1)
                except Exception:
                    nion = 1
                # Occurrence policy for mhd:
                #   - With impurities (nion>1): occ_base carries electrons/common; occ_base+1.. occ_base+nion carry per-ion fields.
                #   - Without impurities (nion<=1): write ONLY occ_base (no redundant extra occurrence).
                nspec_mhd = (1 if int(nion) <= 1 else (1 + int(nion)))

                mhd0 = None
                for s in range(nspec_mhd):
                    occ = occ_base + int(s)
                    mhd_s = factory.new('mhd') if hasattr(factory, 'new') else factory('mhd')
                    populate_mhd_ggd(mhd_s, data, args, species_index=s)
                    # Store nimrod.in XML (best-effort)
                    try:
                        if nimrod_in_path:
                            try:
                                mhd_s.code.name = 'NIMROD'
                            except Exception:
                                pass
                            mhd_s.code.parameters = _build_nimrod_xml(nimrod_in_path)
                    except Exception:
                        pass
                    # annotate COCOS conversion provenance (dump2imas enforces COCOS=11 output)
                    try:
                        mhd_s.ids_properties.comment = (
                            f"dump2imas: ggd full-field snapshot; ids={ids_name}; occurrence={occ}; species_index={s}; "
                            "COCOS: converted NIMROD COCOS=12 -> IMAS COCOS=11 (psi sign flipped; toroidal components flipped; "
                            "modal perturbations mapped for phi-reversal)"
                        )
                    except Exception:
                        pass
                    # Best-effort: tag IDS with output COCOS metadata when supported by this DD
                    _set_ids_cocos(mhd_s, COCOS_OUT_DEFAULT)

                    _db_put_slice(db, mhd_s, occ)
                    if s == 0:
                        mhd0 = mhd_s

                # Keep mhd pointing to electrons/common occurrence for follow-on HDF5 writes below.
                if mhd0 is not None:
                    mhd = mhd0

                # DD-compliant FE-triangle mode: ensure IMAS-standard space geometry vectors are present.
                # Some bindings/backends only create the scaffolding for grid_ggd.space but not the
                # leaf dataset object[]&geometry; plot_mhd.py expects it.
                try:
                    conn_kind = str(getattr(args, 'ggd_connectivity', 'none') or 'none').lower()
                    use_fe_nodes = (
                        bool(getattr(args, 'ggd_unstructured', False))
                        and bool(getattr(args, 'ggd_unstructured_fe_nodes', False))
                        and conn_kind in ('fe_tri', 'fe_wedge', 'fe_pointcloud')
                    )
                    if use_fe_nodes and _use_h5py_patches(args):
                        import numpy as _np
                        nphi = max(1, int(getattr(args, 'ggd_nphi', 8) or 1))
                        phi_list = _np.linspace(0.0, 2.0*_np.pi, num=nphi, endpoint=False)
                        Rloc = _np.asarray(data.get('R'), dtype=float)
                        Zloc = _np.asarray(data.get('Z'), dtype=float)
                        r2d = Rloc.ravel(order='F')
                        z2d = Zloc.ravel(order='F')
                        nn2d = int(r2d.size)
                        r_nodes = _np.tile(r2d, nphi)
                        z_nodes = _np.tile(z2d, nphi)
                        phi_nodes = _np.repeat(phi_list.astype(float), nn2d)
                        for _s in range(nspec_mhd):
                            _occ_s = occ_base + int(_s)
                            _write_gridggd_space_geometry_vectors_h5(entry_dir, 'mhd', _occ_s, r_nodes, z_nodes, phi_nodes)

                except Exception as _e:
                    _log(f"[warn] Could not write grid_ggd.space geometry vectors via h5py: {_e}", args.quiet)
                # Optional: store unstructured node coordinates/connectivity into a NIMROD-specific
                # auxiliary group. This is used by the packed grid_ggd writer and by lightweight
                # downstream tools.
                if _use_h5py_patches(args) and getattr(args, 'ggd_unstructured', False):
                    try:
                        for _s in range(nspec_mhd):
                            _occ_s = occ_base + int(_s)
                            # Writes IMAS-standard grid_ggd.space vectors + grid_ggd.grid_subset connectivity (packed).
                            _write_unstructured_ggd_aux_h5(entry_dir, 'mhd', _occ_s, data, args)
                        _log('Wrote IMAS-standard unstructured grid_ggd nodes/connectivity into mhd HDF5 (h5py)', args.quiet)
                    except Exception as _e:
                        _log(f"[warn] Could not write unstructured grid_ggd nodes/connectivity: {_e}", args.quiet)
                _log("Populated mhd IDS (GGD full-field snapshot)", args.quiet)
                # Mirror edge_profiles electrons.temperature from mhd GGD ONLY when requested.
                # For --edge-ggd-values equilibrium we want edge_profiles to remain equilibrium-only.
                _edge_mode = str(getattr(args, "edge_ggd_values", "")).strip().lower()
                if (
                    _use_h5py_patches(args)
                    and getattr(args, "ggd_unstructured", False)
                    and _edge_mode in ("full", "mhd", "mirror")
                ):
                    try:
                        _mirror_edge_profiles_from_mhd_h5(entry_dir, occ_base, args=args)
                        mhd_h5, _  = _ids_backend_h5_loc(entry_dir, "mhd", occ_base)
                        edge_h5, _ = _ids_backend_h5_loc(entry_dir, "edge_profiles", occ_base)
                        if os.path.exists(mhd_h5) and os.path.exists(edge_h5):
                            _mirror_edge_profiles_from_mhd_h5(mhd_h5, edge_h5, int(occ_base), int(occ_base), args=args)
                            _log("Mirrored edge_profiles electrons.temperature from mhd GGD (full)", args.quiet)
                    except Exception as _e:
                       _log(f"[warn] edge_profiles mirror failed: {_e}", args.quiet)
            except Exception as e:
                _log(f"[warn] Failed to populate/put mhd IDS (GGD): {e}", args.quiet)

        _log(f"Appended IDS slices for {fn.name}", args.quiet)

    # --- append per-step provenance (workflow + dataset_fair) ---
    try:
        pfiles = [str(p) for p in dump_files]
        cmd = _sanitize_cli_command_common(list(sys.argv), known_files=pfiles)
        _update_workflow_and_dataset_fair_common(
            db, factory,
            component_name="nimrod2imas:dump2imas",
            component_description="Convert NIMROD dumpgll HDF5 outputs to IMAS IDSs (mhd_linear, equilibrium/core_profiles/edge_profiles, and optional mhd GGD snapshot)",
            component_repository="https://github.com/PrincetonUniversity/nimrod2imas",
            component_version=str(__version__),
            exec_command=cmd,
            input_files=pfiles,
            record_checksums=bool(getattr(args, "record_checksums", True)),
            checksum_algorithm=str(getattr(args, "checksum_algorithm", "sha256") or "sha256"),
            workflow_occ=0,
            dataset_fair_occ=0,
            extra_kv={
                "dd": str(args.dd),
                "dd_version": str(dd_version),
                "pulse": str(args.pulse),
                "run": str(args.run),
                "occ_base": str(int(getattr(args, "occ_base", 0) or 0)),
                "ndumps": str(len(dump_files)),
            },
        )
    except Exception as _e:
        _log(f"[warn] provenance update failed: {_e}", args.quiet)
    try:
        db.close()
    except Exception:
        pass

    return 0




def _mirror_edge_profiles_from_mhd_h5(
    mhd_h5: str,
    edge_h5: str,
    mhd_occ: int,
    edge_occ: int,
    args,
    log=None,
) -> None:
    """
    Mirror selected edge-relevant quantities from the mhd IDS HDF5 into the edge_profiles IDS HDF5.

    For IMAS DD >= 4.x, edge_profiles.ggd does **not** use legacy leaves such as
    `current_density_tor`, `velocity_phi`, `pressure_total`, etc.  Instead, current density is
    represented by `ggd(itime)/j_total(i1)/{r,phi,z}(:)` and (optionally) flow by
    `ggd(itime)/electrons/velocity(i1)/{r,phi,z}(:)` or `ion(i1)/velocity(i2)/...`.
    This helper therefore writes DD4-compliant leaves when DD major >= 4.

    For DD < 4, we keep a conservative "best-effort" mirroring using legacy aliases (if present),
    but we never create non-DD nodes.
    """
    if log is None:
        log = logging.getLogger("dump2imas")

    import re as _re
    import h5py as _h5py

    def _parse_dd(v: str) -> tuple[int, int, int]:
        s = str(v or "").strip()
        m = _re.match(r"^\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?\s*$", s)
        if not m:
            return (0, 0, 0)
        return (int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0))

    ddv = str(getattr(args, "dd_version", None) or os.environ.get("IMAS_VERSION", "4.1.1"))
    dd_major, dd_minor, dd_patch = _parse_dd(ddv)

    # --- utilities ---
    def _first_existing(h5, full_paths: list[str]) -> Optional[str]:
        for p in full_paths:
            try:
                if p in h5:
                    return p
            except Exception:
                continue
        return None

    def _read_arr(h5, full_path: str) -> np.ndarray:
        a = np.asarray(h5[full_path])
        return a

    def _scalar_int(a: Optional[np.ndarray], default: int = 0) -> int:
        if a is None:
            return int(default)
        try:
            v = np.asarray(a).ravel()
            if v.size:
                return int(v[0])
        except Exception:
            pass
        return int(default)

    def _as_1d(a: np.ndarray) -> np.ndarray:
        A = np.asarray(a)
        # common IMAS-HDF5: (1, N) or (1,1,N)
        if A.ndim >= 2 and A.shape[0] == 1:
            A = A[0]
            if A.ndim >= 2 and A.shape[0] == 1:
                A = A[0]
        return np.asarray(A).ravel(order="C")

    def _shape_rzp(shape_ds: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if shape_ds is None:
            return None
        s = np.asarray(shape_ds).astype(int).ravel()
        if s.size >= 3:
            return s[-3:]
        return None

    def _ensure_ds(g, name: str, data, dtype=None, overwrite: bool = True):
        try:
            if name in g:
                if overwrite:
                    del g[name]
                else:
                    return
            if dtype is None:
                g.create_dataset(name, data=data)
            else:
                g.create_dataset(name, data=data, dtype=dtype)
        except Exception as e:
            log.debug("HDF5 write failed for %s: %s", name, e)

    def _write_vec_components_dd4(
        g,
        base: str,
        comp_to_src: dict[str, tuple[np.ndarray, Optional[np.ndarray]]],
        aos_shape: Optional[np.ndarray] = None,
        grid_index_arr: Optional[np.ndarray] = None,
        grid_subset_index_arr: Optional[np.ndarray] = None,
        default_grid_index: int = 0,
        default_grid_subset_index: int = 0,
    ):
        """
        Write DD4-compliant AoS leaf `base` with components {r,phi,z} as FLT_1D arrays.

        base examples:
          - "ggd[]&j_total[]"
          - "ggd[]&electrons&velocity[]"
        """
        # AoS bookkeeping (copy from source when possible).
        if aos_shape is None:
            aos_shape_ds = np.array([[1]], dtype=np.int32)
        else:
            aos_shape_ds = np.asarray(aos_shape).astype(np.int32)
            # IMAS-HDF5 typically stores AOS_SHAPE as a 2D int array (ntime x 1) or (1 x 1).
            if aos_shape_ds.ndim == 1:
                aos_shape_ds = aos_shape_ds.reshape(1, -1)

        if grid_index_arr is None:
            gi_ds = np.array([default_grid_index], dtype=np.int32)
        else:
            gi_ds = np.asarray(grid_index_arr).astype(np.int32).ravel()

        if grid_subset_index_arr is None:
            gs_ds = np.array([default_grid_subset_index], dtype=np.int32)
        else:
            gs_ds = np.asarray(grid_subset_index_arr).astype(np.int32).ravel()

        _ensure_ds(g, f"{base}&AOS_SHAPE", aos_shape_ds, dtype="i4", overwrite=True)
        _ensure_ds(g, f"{base}&grid_index", gi_ds, dtype="i4", overwrite=True)
        _ensure_ds(g, f"{base}&grid_subset_index", gs_ds, dtype="i4", overwrite=True)

        for comp, (vals, sh) in comp_to_src.items():
            v1 = _as_1d(vals)
            _ensure_ds(g, f"{base}&{comp}", v1.astype(float), dtype="f8", overwrite=True)
            rzp = _shape_rzp(sh)
            if rzp is not None:
                _ensure_ds(
                    g,
                    f"{base}&{comp}_SHAPE",
                    np.array(rzp, dtype=np.int32).reshape(1, 1, 3),
                    dtype="i4",
                    overwrite=True,
                )

    # --- main ---
    src_root = f"/mhd_{mhd_occ}"
    dst_root = f"/edge_profiles_{edge_occ}"

    with _h5py.File(mhd_h5, "r") as mhd_f, _h5py.File(edge_h5, "r+") as ep_f:
        if dst_root not in ep_f:
            ep_f.create_group(dst_root)
        dst_g = ep_f[dst_root]

        # ------------------------------------------------------------
        # 1) "same-name" mirroring (only if both sides already have it)
        # ------------------------------------------------------------
        if dd_major >= 4:
            # DD4+ edge_profiles thermodynamic leaves are not mirrored from mhd typically,
            # but keep these as harmless no-ops if upstream mhd happens to provide them.
            simple_paths = [
                ("/ggd[]&electrons&temperature[]&values", "/ggd[]&electrons&temperature[]&values"),
                ("/ggd[]&electrons&density[]&values",     "/ggd[]&electrons&density[]&values"),
                ("/ggd[]&electrons&pressure[]&values",    "/ggd[]&electrons&pressure[]&values"),
                ("/ggd[]&t_i_average[]&values",           "/ggd[]&t_i_average[]&values"),
                ("/ggd[]&n_i_total[]&values",             "/ggd[]&n_i_total[]&values"),
            ]
        else:
            # Legacy (DD<4) best-effort alias set (kept small to avoid non-DD nodes)
            simple_paths = [
                ("/ggd[]&electrons&temperature[]&values", "/ggd[]&electrons&temperature[]&values"),
                ("/ggd[]&electrons&density[]&values",     "/ggd[]&electrons&density[]&values"),
                ("/ggd[]&electrons&pressure[]&values",    "/ggd[]&electrons&pressure[]&values"),
                ("/ggd[]&t_i_average[]&values",           "/ggd[]&t_i_average[]&values"),
                ("/ggd[]&n_i_total[]&values",             "/ggd[]&n_i_total[]&values"),
                ("/ggd[]&j_phi[]&values",                 "/ggd[]&j_phi[]&values"),
                ("/ggd[]&j_r[]&values",                   "/ggd[]&j_r[]&values"),
                ("/ggd[]&j_z[]&values",                   "/ggd[]&j_z[]&values"),
                ("/ggd[]&v_phi[]&values",                 "/ggd[]&v_phi[]&values"),
                ("/ggd[]&v_r[]&values",                   "/ggd[]&v_r[]&values"),
                ("/ggd[]&v_z[]&values",                   "/ggd[]&v_z[]&values"),
            ]

        for src_suf, dst_suf in simple_paths:
            src_leaf = f"{src_root}{src_suf}"
            dst_leaf = f"{dst_root}{dst_suf}"
            try:
                if src_leaf not in mhd_f:
                    continue

                # For DD4+, keep density/pressure written from the dump (nq/peq/prq) intact.
                # Only overwrite Te (if provided by mhd) to keep edge temperature consistent.
                overwrite = (dd_major < 4) or (dst_suf == "/ggd[]&electrons&temperature[]&values")

                if (dst_leaf in ep_f) and overwrite:
                    del ep_f[dst_leaf]

                if dst_leaf not in ep_f:
                    ep_f.copy(mhd_f[src_leaf], ep_f, dst_leaf)
                    log.debug("Mirrored %s -> %s", src_leaf, dst_leaf)
            except Exception:
                continue

        # ------------------------------------------------------------
        # 2) DD4+ spec-compliant mapping: mhd j_* -> edge_profiles j_total
        #    and mhd velocity_* -> edge_profiles electrons/velocity
        # ------------------------------------------------------------
        if dd_major >= 4:
            # ----- current density -----
            # Prefer canonical mhd leaves
            jphi_val_p = _first_existing(mhd_f, [f"{src_root}/ggd[]&j_phi[]&values", f"{src_root}/ggd[]&j_tor[]&values"])
            jr_val_p   = _first_existing(mhd_f, [f"{src_root}/ggd[]&j_r[]&values"])
            jz_val_p   = _first_existing(mhd_f, [f"{src_root}/ggd[]&j_z[]&values"])

            # shapes (optional, but helps downstream tools)
            jphi_sh_p = _first_existing(mhd_f, [f"{src_root}/ggd[]&j_phi[]&values_SHAPE", f"{src_root}/ggd[]&j_tor[]&values_SHAPE"])
            jr_sh_p   = _first_existing(mhd_f, [f"{src_root}/ggd[]&j_r[]&values_SHAPE"])
            jz_sh_p   = _first_existing(mhd_f, [f"{src_root}/ggd[]&j_z[]&values_SHAPE"])

            # grid indices (optional)
            j_gi_p = _first_existing(mhd_f, [f"{src_root}/ggd[]&j_phi[]&grid_index", f"{src_root}/ggd[]&j_tor[]&grid_index"])
            j_gs_p = _first_existing(mhd_f, [f"{src_root}/ggd[]&j_phi[]&grid_subset_index", f"{src_root}/ggd[]&j_tor[]&grid_subset_index"])
            j_aos_p = _first_existing(mhd_f, [f"{src_root}/ggd[]&j_phi[]&AOS_SHAPE", f"{src_root}/ggd[]&j_tor[]&AOS_SHAPE"])

            if jphi_val_p or jr_val_p or jz_val_p:
                grid_index = _scalar_int(_read_arr(mhd_f, j_gi_p) if j_gi_p else None, default=0)
                grid_subset_index = _scalar_int(_read_arr(mhd_f, j_gs_p) if j_gs_p else None, default=0)

                comp_map = {}
                if jr_val_p:
                    comp_map["r"] = (_read_arr(mhd_f, jr_val_p), _read_arr(mhd_f, jr_sh_p) if jr_sh_p else None)
                if jphi_val_p:
                    comp_map["phi"] = (_read_arr(mhd_f, jphi_val_p), _read_arr(mhd_f, jphi_sh_p) if jphi_sh_p else None)
                if jz_val_p:
                    comp_map["z"] = (_read_arr(mhd_f, jz_val_p), _read_arr(mhd_f, jz_sh_p) if jz_sh_p else None)

                _write_vec_components_dd4(
                    dst_g,
                    base="ggd[]&j_total[]",
                    comp_to_src=comp_map,
                    aos_shape=_read_arr(mhd_f, j_aos_p) if j_aos_p else None,
                    grid_index_arr=_read_arr(mhd_f, j_gi_p) if j_gi_p else None,
                    grid_subset_index_arr=_read_arr(mhd_f, j_gs_p) if j_gs_p else None,
                    default_grid_index=grid_index,
                    default_grid_subset_index=grid_subset_index,
                )

            # ----- velocity -----
            vphi_val_p = _first_existing(mhd_f, [f"{src_root}/ggd[]&velocity_phi[]&values", f"{src_root}/ggd[]&v_phi[]&values"])
            vr_val_p   = _first_existing(mhd_f, [f"{src_root}/ggd[]&velocity_r[]&values", f"{src_root}/ggd[]&v_r[]&values"])
            vz_val_p   = _first_existing(mhd_f, [f"{src_root}/ggd[]&velocity_z[]&values", f"{src_root}/ggd[]&v_z[]&values"])

            vphi_sh_p = _first_existing(mhd_f, [f"{src_root}/ggd[]&velocity_phi[]&values_SHAPE", f"{src_root}/ggd[]&v_phi[]&values_SHAPE"])
            vr_sh_p   = _first_existing(mhd_f, [f"{src_root}/ggd[]&velocity_r[]&values_SHAPE", f"{src_root}/ggd[]&v_r[]&values_SHAPE"])
            vz_sh_p   = _first_existing(mhd_f, [f"{src_root}/ggd[]&velocity_z[]&values_SHAPE", f"{src_root}/ggd[]&v_z[]&values_SHAPE"])

            v_gi_p = _first_existing(mhd_f, [f"{src_root}/ggd[]&velocity_phi[]&grid_index", f"{src_root}/ggd[]&v_phi[]&grid_index"])
            v_gs_p = _first_existing(mhd_f, [f"{src_root}/ggd[]&velocity_phi[]&grid_subset_index", f"{src_root}/ggd[]&v_phi[]&grid_subset_index"])
            v_aos_p = _first_existing(mhd_f, [f"{src_root}/ggd[]&velocity_phi[]&AOS_SHAPE", f"{src_root}/ggd[]&v_phi[]&AOS_SHAPE"])

            if vphi_val_p or vr_val_p or vz_val_p:
                grid_index = _scalar_int(_read_arr(mhd_f, v_gi_p) if v_gi_p else None, default=0)
                grid_subset_index = _scalar_int(_read_arr(mhd_f, v_gs_p) if v_gs_p else None, default=0)

                # Store bulk flow under ions (DD4.1+: ggd/ion(i1)/velocity(i2)/{r,phi,z}).
                sp = getattr(args, "_nimrod_species", {}) or {}
                z_ions = sp.get("z_ions") or sp.get("z_ion") or []
                try:
                    nion_mirror = int(getattr(args, "edge_mirror_nion", 0) or 0)
                except Exception:
                    nion_mirror = 0
                if nion_mirror <= 0:
                    nion_mirror = int(len(z_ions)) if isinstance(z_ions, (list, tuple)) and len(z_ions) > 0 else 1
                nion_mirror = max(1, int(nion_mirror))

                # infer grid shape (R,Z,Phi) for SHAPE datasets
                shape_rzp = None
                for _sh in (vr_sh_p, vphi_sh_p, vz_sh_p):
                    if _sh:
                        try:
                            shape_rzp = _shape_rzp(_read_arr(mhd_f, _sh))
                            if shape_rzp is not None:
                                break
                        except Exception:
                            pass
                if shape_rzp is None:
                    # fall back to current density shape if available
                    for _sh in (jr_sh_p, jphi_sh_p, jz_sh_p):
                        if _sh:
                            try:
                                shape_rzp = _shape_rzp(_read_arr(mhd_f, _sh))
                                if shape_rzp is not None:
                                    break
                            except Exception:
                                pass
                if shape_rzp is None:
                    # last resort: infer from any 3D-like value array
                    _arr0 = _read_arr(mhd_f, vphi_val_p or vr_val_p or vz_val_p)
                    try:
                        _a0 = np.asarray(_arr0)
                        if _a0.ndim >= 3:
                            shape_rzp = (int(_a0.shape[-3]), int(_a0.shape[-2]), int(_a0.shape[-1]))
                    except Exception:
                        shape_rzp = None

                if shape_rzp is not None:
                    base = "ggd[]&ion[]&velocity[]"
                    aos = np.array([[1, nion_mirror]], dtype=np.int32)
                    _write_arr(dst_g, f"{base}&AOS_SHAPE", aos, dtype="i4", overwrite=True)
                    _write_arr(dst_g, f"{base}&grid_index", np.array([[grid_index] * nion_mirror], dtype=np.int32), dtype="i4", overwrite=True)
                    _write_arr(dst_g, f"{base}&grid_subset_index", np.array([[grid_subset_index] * nion_mirror], dtype=np.int32), dtype="i4", overwrite=True)

                    def _write_comp(_comp: str, _val_path: Optional[str], _sh_path: Optional[str]) -> None:
                        if not _val_path:
                            return
                        vv = _as_1d(_read_arr(mhd_f, _val_path))
                        # replicate the single-fluid field over all ions (when multiple species exist)
                        v_all = np.tile(vv, nion_mirror).astype(np.float64, copy=False)
                        s_rzp = np.asarray(shape_rzp, dtype=np.int32).reshape(1, 3)
                        s_all = np.tile(s_rzp, (nion_mirror, 1))
                        _write_arr(dst_g, f"{base}&{_comp}", v_all.reshape((1, nion_mirror, -1)), dtype="f8", overwrite=True)
                        _write_arr(dst_g, f"{base}&{_comp}_SHAPE", s_all.reshape((1, nion_mirror, 3)), dtype="i4", overwrite=True)

                    _write_comp("r",   vr_val_p,   vr_sh_p)
                    _write_comp("phi", vphi_val_p, vphi_sh_p)
                    _write_comp("z",   vz_val_p,   vz_sh_p)
                else:
                    log.warning("Skipping ion velocity mirroring: could not infer SHAPE for mhd velocity fields")

if __name__ == "__main__":
    raise SystemExit(main())
