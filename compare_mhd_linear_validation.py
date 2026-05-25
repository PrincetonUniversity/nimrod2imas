#!/usr/bin/env python3
"""
Compare selected perturbations between a native NIMROD dumpgll file and the IMAS mhd_linear IDS.

Intended use: produce compact statistical validation metrics for a Scientific Data Data Descriptor.

Supported quantities
-------------------
  n : mass_density_perturbed    (from rend/imnd, converted with ion mass)
  t : temperature_perturbed     (electron: rete/imte, ions: reti/imti)
  p : pressure_perturbed        (explicit if available, otherwise reconstructed as in dump2imas)
  b : b_field_perturbed         (rebe/imbe)
  v : velocity_perturbed        (reve/imve; typically only main-ion occurrence)

Typical examples
----------------
# Electron density perturbation, first non-empty time slice, mode n=1
python compare_mhd_linear_validation.py dumpgll.00020.h5 \
  --entry /path/to/d3d/4/163518/3 \
  --occ 0 --occ-base 0 --quantity n --part real --n-tor 1 \
  --plot n_real_n1.png

# Main-ion velocity phi component in occurrence 1
python compare_mhd_linear_validation.py dumpgll.00020.h5 \
  --entry /path/to/d3d/4/163518/3 \
  --occ 1 --occ-base 0 --quantity v --component phi --part amp --n-tor 1 \
  --plot vphi_amp_n1.png
"""

from __future__ import annotations

import argparse
import hashlib
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from dump2imas import read_and_stitch_dump, _nimrod_species_info
except ImportError:
    from dump2imas import read_and_stitch_dump
    _nimrod_species_info = None
from nimrod2imas import open_dbentry, ids_factory, get_ids

AMU = 1.66053906660e-27
ECHARGE = 1.602176634e-19
PLACEHOLDER_INT = -999999999


# ----------------------------- IMAS helpers -----------------------------

def aos_size(x: Any) -> int:
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
        return 0


def mode_count(ts: Any) -> int:
    try:
        return aos_size(ts.toroidal_mode)
    except Exception:
        return 0


def mode_number(tm: Any) -> Optional[int]:
    for name in ("n_phi", "n_tor"):
        if hasattr(tm, name):
            try:
                v = int(getattr(tm, name))
                return v
            except Exception:
                pass
    try:
        return int(tm.identifier.index)
    except Exception:
        return None


def time_slice_time(ts: Any) -> float:
    """Return time_slice time as float, or NaN if unavailable/placeheld."""
    try:
        return float(ts.time)
    except Exception:
        return float("nan")


def nonempty_time_slice_indices(mhd: Any) -> list[int]:
    """Raw indices of time_slices that contain at least one toroidal_mode entry."""
    nraw = aos_size(mhd.time_slice)
    out: list[int] = []
    for i in range(nraw):
        try:
            if mode_count(mhd.time_slice[i]) > 0:
                out.append(i)
        except Exception:
            pass
    return out


def format_time_slice_table(mhd: Any) -> str:
    """Human-readable table of raw and logical non-empty time indices."""
    nraw = aos_size(mhd.time_slice)
    nonempty = nonempty_time_slice_indices(mhd)
    logical = {raw: j for j, raw in enumerate(nonempty)}
    lines = []
    lines.append("raw_index logical_nonempty_index time n_modes")
    for i in range(nraw):
        try:
            ts = mhd.time_slice[i]
            t = time_slice_time(ts)
            nm = mode_count(ts)
        except Exception:
            t = float("nan")
            nm = 0
        li = logical.get(i, None)
        li_s = "-" if li is None else str(li)
        t_s = "nan" if not np.isfinite(t) else f"{t:.12g}"
        lines.append(f"{i:9d} {li_s:22s} {t_s:>16s} {nm:7d}")
    return "\n".join(lines)


def resolve_time_index(
    mhd: Any,
    ids_time_index: int,
    raw: bool = False,
    ids_time_value: Optional[float] = None,
    ids_time_tolerance: Optional[float] = None,
) -> int:
    """Resolve requested IDS time selection to a raw mhd_linear.time_slice index.

    If ids_time_value is provided, selects the closest time slice by time.  By default,
    the search is restricted to non-empty time_slices, matching the older --time-index
    behavior.  If raw=True, the integer index and time-value search use all raw slices.
    """
    nraw = aos_size(mhd.time_slice)
    if nraw <= 0:
        raise RuntimeError("mhd_linear has no time_slice entries")

    candidates = list(range(nraw)) if raw else nonempty_time_slice_indices(mhd)
    if not candidates:
        candidates = list(range(nraw))

    if ids_time_value is not None:
        target = float(ids_time_value)
        best_i = None
        best_dt = None
        for i in candidates:
            try:
                t = time_slice_time(mhd.time_slice[i])
            except Exception:
                t = float("nan")
            if not np.isfinite(t):
                continue
            dt = abs(t - target)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_i = i
        if best_i is None:
            raise RuntimeError("could not select by IDS time value: no finite time values available")
        if ids_time_tolerance is not None and best_dt is not None and best_dt > float(ids_time_tolerance):
            raise RuntimeError(
                f"closest IDS time differs from requested value by {best_dt:g}, "
                f"which exceeds --ids-time-tolerance={ids_time_tolerance:g}"
            )
        return int(best_i)

    if raw:
        idx = int(ids_time_index)
        if idx < 0:
            idx += nraw
        if idx < 0 or idx >= nraw:
            raise RuntimeError(f"raw IDS time index {ids_time_index} out of range 0..{nraw-1}")
        return idx

    nonempty = candidates
    idx = int(ids_time_index)
    if idx < 0:
        idx += len(nonempty)
    if idx < 0 or idx >= len(nonempty):
        raise RuntimeError(
            f"IDS time index {ids_time_index} out of range for non-empty slices 0..{len(nonempty)-1}; "
            "use --raw-ids-time-index to index raw time_slice entries"
        )
    return int(nonempty[idx])


def select_mode_index(ts: Any, n_tor: Optional[int], mode_index: Optional[int]) -> int:
    nm = mode_count(ts)
    if nm <= 0:
        raise RuntimeError("selected time_slice has no toroidal_mode entries")
    if mode_index is not None:
        if mode_index < 0:
            mode_index += nm
        if mode_index < 0 or mode_index >= nm:
            raise RuntimeError(f"mode-index {mode_index} out of range 0..{nm-1}")
        return mode_index
    if n_tor is None:
        return 0
    vals = []
    for i in range(nm):
        v = mode_number(ts.toroidal_mode[i])
        vals.append(PLACEHOLDER_INT if v is None else v)
    for i, v in enumerate(vals):
        if v == int(n_tor):
            return i
    if all(v == PLACEHOLDER_INT for v in vals):
        # allow direct indexing fallback when n_tor was not populated
        if 0 <= int(n_tor) < nm:
            return int(n_tor)
        if 1 <= int(n_tor) <= nm:
            return int(n_tor) - 1
    raise RuntimeError(f"n_tor={n_tor} not found; available={vals}")


def complex_scalar_array(node: Any, part: str) -> np.ndarray:
    re = np.asarray(node.real, dtype=float)
    im = np.asarray(node.imaginary, dtype=float)
    if part == "real":
        return re
    if part == "imag":
        return im
    if part == "amp":
        return np.sqrt(re * re + im * im)
    raise ValueError(part)


def complex_vector_component(node: Any, component: str, part: str) -> np.ndarray:
    comp_map = {"r": "coordinate1", "z": "coordinate2", "phi": "coordinate3"}
    obj = getattr(node, comp_map[component])
    return complex_scalar_array(obj, part)


def extract_ids_field(
    mhd: Any,
    ids_time_index: int,
    raw_ids_time_index: bool,
    ids_time_value: Optional[float],
    ids_time_tolerance: Optional[float],
    n_tor: Optional[int],
    mode_index: Optional[int],
    quantity: str,
    part: str,
    component: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
    ti = resolve_time_index(
        mhd,
        ids_time_index=ids_time_index,
        raw=raw_ids_time_index,
        ids_time_value=ids_time_value,
        ids_time_tolerance=ids_time_tolerance,
    )
    ts = mhd.time_slice[ti]
    mi = select_mode_index(ts, n_tor=n_tor, mode_index=mode_index)
    tm = ts.toroidal_mode[mi]
    pl = tm.plasma

    # Prefer coordinate_system r/z if present.
    R = np.asarray(pl.coordinate_system.r, dtype=float)
    Z = np.asarray(pl.coordinate_system.z, dtype=float)

    q = quantity.lower()
    if q == "n":
        F = complex_scalar_array(pl.mass_density_perturbed, part)
    elif q == "t":
        F = complex_scalar_array(pl.temperature_perturbed, part)
    elif q == "p":
        F = complex_scalar_array(pl.pressure_perturbed, part)
    elif q == "b":
        if component is None:
            raise RuntimeError("quantity b requires --component r|z|phi")
        F = complex_vector_component(pl.b_field_perturbed, component, part)
    elif q == "v":
        if component is None:
            raise RuntimeError("quantity v requires --component r|z|phi")
        F = complex_vector_component(pl.velocity_perturbed, component, part)
    else:
        raise RuntimeError(f"unsupported quantity {quantity!r}")

    F = np.asarray(F, dtype=float)
    if R.shape != F.shape:
        if R.T.shape == F.shape:
            R = R.T
            Z = Z.T
        else:
            raise RuntimeError(f"IDS grid shape {R.shape} incompatible with field shape {F.shape}")
    return R, Z, F, ti, mi, time_slice_time(ts)


# ----------------------------- dump helpers -----------------------------

def dump_density_fields(data: dict, species_index: int, mode_index: int, ion_mass_amu: float) -> Tuple[np.ndarray, np.ndarray]:
    fields = data["fields"]
    dens_re = fields["rend"]
    dens_im = fields["imnd"]
    s = int(species_index)
    if s >= dens_re.shape[2]:
        s = 0
    mpart = float(ion_mass_amu) * AMU
    return dens_re[:, :, s, mode_index] * mpart, dens_im[:, :, s, mode_index] * mpart


def dump_temperature_fields(data: dict, species_index: int, mode_index: int, electrons_index: int) -> Tuple[np.ndarray, np.ndarray]:
    fields = data["fields"]
    if int(species_index) == int(electrons_index):
        return fields["rete"][:, :, mode_index], fields["imte"][:, :, mode_index]
    return fields["reti"][:, :, mode_index], fields["imti"][:, :, mode_index]


def _species_slice_2d(A: Any, species_index: int) -> np.ndarray:
    """Return a 2D equilibrium array, selecting a species axis when present."""
    arr = np.asarray(A)
    if arr.ndim == 3:
        ssel = int(species_index)
        if ssel < 0 or ssel >= arr.shape[2]:
            ssel = 0
        return arr[:, :, ssel]
    if arr.ndim == 2:
        return arr
    raise RuntimeError(f"expected 2D or 3D equilibrium array, got shape {arr.shape}")


def _mode_slice_2d(A: Any, mode_index: int, species_index: Optional[int] = None) -> np.ndarray:
    """Return a 2D perturbation array from either (R,Z,mode) or (R,Z,species,mode)."""
    arr = np.asarray(A)
    if arr.ndim == 4:
        if species_index is None:
            ssel = 0
        else:
            ssel = int(species_index)
            if ssel < 0 or ssel >= arr.shape[2]:
                ssel = 0
        return arr[:, :, ssel, mode_index]
    if arr.ndim == 3:
        return arr[:, :, mode_index]
    raise RuntimeError(f"expected 3D or 4D perturbation array, got shape {arr.shape}")


def dump_pressure_fields(
    data: dict,
    species_index: int,
    mode_index: int,
    electrons_index: int,
    pressure_source: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the native reference pressure perturbation using dump2imas logic.

    The default ``pressure_source='auto'`` intentionally mirrors the pressure-writing
    branch in ``populate_mhd_linear`` from dump2imasA.py:

      1. electron occurrence: use ``repe/impe`` when present;
      2. if ``repr/impr`` is species-resolved (4D), use the selected species slice;
      3. if ``repr/impr`` is total pressure (3D), use ``repr-repe`` for ions only when
         ``data['nspec_eq'] <= 1``; otherwise use total pressure only for electrons;
      4. if no explicit branch applies, reconstruct
         ``dp = e * (dn*T0 + n0*dT)``.

    This is deliberately different from a purely physical ion-pressure reference. It is
    the reference needed to verify that the deposited IMAS field matches what the current
    converter actually writes.
    """
    fields = data["fields"]
    source = str(pressure_source or "auto").lower()
    valid_sources = {
        "auto", "dump2imas", "explicit", "total", "total_minus_electron",
        "reconstruct", "thermal_only", "density_only",
    }
    if source not in valid_sources:
        raise RuntimeError(f"unknown pressure_source={pressure_source!r}; expected one of {sorted(valid_sources)}")
    if source == "dump2imas":
        source = "auto"

    is_electron = int(species_index) == int(electrons_index)
    have_repe = ("repe" in fields) and ("impe" in fields)
    have_repr = ("repr" in fields) and ("impr" in fields)
    repr_is_species_resolved = have_repr and np.asarray(fields["repr"]).ndim == 4
    nspec_eq = int(data.get("nspec_eq", 0) or 0)

    def electron_pressure() -> Tuple[np.ndarray, np.ndarray]:
        if not have_repe:
            raise RuntimeError("electron pressure perturbation repe/impe is not available")
        return _mode_slice_2d(fields["repe"], mode_index), _mode_slice_2d(fields["impe"], mode_index)

    def total_pressure() -> Tuple[np.ndarray, np.ndarray]:
        if not have_repr:
            raise RuntimeError("total pressure perturbation repr/impr is not available")
        return _mode_slice_2d(fields["repr"], mode_index), _mode_slice_2d(fields["impr"], mode_index)

    def species_pressure_from_repr() -> Tuple[np.ndarray, np.ndarray]:
        if not repr_is_species_resolved:
            raise RuntimeError("repr/impr is not species resolved")
        return (
            _mode_slice_2d(fields["repr"], mode_index, species_index),
            _mode_slice_2d(fields["impr"], mode_index, species_index),
        )

    def ion_pressure_total_minus_electron() -> Tuple[np.ndarray, np.ndarray]:
        dp_tot_re, dp_tot_im = total_pressure()
        dp_e_re, dp_e_im = electron_pressure()
        return dp_tot_re - dp_e_re, dp_tot_im - dp_e_im

    def pressure_linearized_terms() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return thermal and density contributions exactly as dump2imas computes them."""
        nq2d = data.get("nq", None)
        teq2d = data.get("teq", None)
        tiq2d = data.get("tiq", None)
        prq2d = data.get("prq", None)
        peq2d = data.get("peq", None)

        n0 = None
        if nq2d is not None:
            n0 = _species_slice_2d(nq2d, species_index)

        T0 = None
        if is_electron:
            if teq2d is not None:
                T0 = _species_slice_2d(teq2d, 0)
            elif peq2d is not None and n0 is not None:
                pe = _species_slice_2d(peq2d, 0)
                T0 = pe / (np.maximum(n0, 1e-60) * ECHARGE)
        else:
            if tiq2d is not None:
                # dump2imas uses tiq[:,:,0] for a 3D tiq array, not the IMAS occurrence index.
                T0 = _species_slice_2d(tiq2d, 0)
            elif prq2d is not None and n0 is not None:
                # Mirror dump2imas: subtract p_e only when nspec_eq <= 1.  In the
                # single-channel compatibility path, read_and_stitch_dump expands
                # nspec_eq to 2, so the converter uses p_total/n0 here.
                p_tot = _species_slice_2d(prq2d, 0)
                if peq2d is not None and nspec_eq <= 1:
                    p_e = _species_slice_2d(peq2d, 0)
                    T0 = (p_tot - p_e) / (np.maximum(n0, 1e-60) * ECHARGE)
                else:
                    T0 = p_tot / (np.maximum(n0, 1e-60) * ECHARGE)

        # dump2imas uses the same species_index for rend/imnd as for the IMAS occurrence,
        # falling back to 0 only if the requested index is out of range.  Use mpart=1 to
        # recover number-density perturbations from dump_density_fields.
        dn_re, dn_im = dump_density_fields(
            data,
            species_index,
            mode_index,
            ion_mass_amu=1.0 / AMU,
        )
        dT_re, dT_im = dump_temperature_fields(data, species_index, mode_index, electrons_index)
        if n0 is None or T0 is None:
            raise RuntimeError("cannot reconstruct pressure perturbation: missing equilibrium density/temperature")
        thermal_re = n0 * dT_re * ECHARGE
        thermal_im = n0 * dT_im * ECHARGE
        density_re = dn_re * T0 * ECHARGE
        density_im = dn_im * T0 * ECHARGE
        return thermal_re, thermal_im, density_re, density_im

    def reconstruct_species_pressure() -> Tuple[np.ndarray, np.ndarray]:
        th_re, th_im, dn_re, dn_im = pressure_linearized_terms()
        return th_re + dn_re, th_im + dn_im

    def thermal_only_pressure() -> Tuple[np.ndarray, np.ndarray]:
        th_re, th_im, _, _ = pressure_linearized_terms()
        return th_re, th_im

    def density_only_pressure() -> Tuple[np.ndarray, np.ndarray]:
        _, _, dn_re, dn_im = pressure_linearized_terms()
        return dn_re, dn_im

    if source == "total":
        return total_pressure()
    if source == "total_minus_electron":
        return ion_pressure_total_minus_electron()
    if source == "reconstruct":
        return reconstruct_species_pressure()
    if source == "thermal_only":
        return thermal_only_pressure()
    if source == "density_only":
        return density_only_pressure()
    if source == "explicit":
        # Debug helper, not dump2imas exact behavior.
        if is_electron and have_repe:
            return electron_pressure()
        if repr_is_species_resolved:
            return species_pressure_from_repr()
        if have_repr:
            return total_pressure()
        return reconstruct_species_pressure()

    # source == "auto": exact populate_mhd_linear branch order.
    dp_re = None
    dp_im = None

    if is_electron and have_repe:
        dp_re, dp_im = electron_pressure()
    elif repr_is_species_resolved:
        dp_re, dp_im = species_pressure_from_repr()
    elif have_repr:
        dp_tot_re, dp_tot_im = total_pressure()
        if (not is_electron) and have_repe and nspec_eq <= 1:
            dp_e_re, dp_e_im = electron_pressure()
            dp_re = dp_tot_re - dp_e_re
            dp_im = dp_tot_im - dp_e_im
        elif is_electron:
            dp_re, dp_im = dp_tot_re, dp_tot_im

    if dp_re is None or dp_im is None:
        dp_re, dp_im = reconstruct_species_pressure()
    return dp_re, dp_im

def dump_vector_fields(data: dict, quantity: str, component: str, mode_index: int) -> Tuple[np.ndarray, np.ndarray]:
    fields = data["fields"]
    cidx = {"r": 0, "z": 1, "phi": 2}[component]
    if quantity == "b":
        return fields["rebe"][:, :, mode_index, cidx], fields["imbe"][:, :, mode_index, cidx]
    if quantity == "v":
        return fields["reve"][:, :, mode_index, cidx], fields["imve"][:, :, mode_index, cidx]
    raise RuntimeError(quantity)


def extract_dump_field(
    data: dict,
    quantity: str,
    species_index: int,
    mode_index: int,
    part: str,
    component: Optional[str],
    ion_mass_amu: float,
    electrons_index: int,
    pressure_source: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = quantity.lower()
    R = np.asarray(data["R"], dtype=float)
    Z = np.asarray(data["Z"], dtype=float)

    if q == "n":
        re, im = dump_density_fields(data, species_index, mode_index, ion_mass_amu)
    elif q == "t":
        re, im = dump_temperature_fields(data, species_index, mode_index, electrons_index)
    elif q == "p":
        re, im = dump_pressure_fields(data, species_index, mode_index, electrons_index, pressure_source=pressure_source)
    elif q in ("b", "v"):
        if component is None:
            raise RuntimeError(f"quantity {quantity} requires --component r|z|phi")
        re, im = dump_vector_fields(data, q, component, mode_index)
    else:
        raise RuntimeError(f"unsupported quantity {quantity!r}")

    if part == "real":
        F = np.asarray(re, dtype=float)
    elif part == "imag":
        F = np.asarray(im, dtype=float)
    elif part == "amp":
        F = np.sqrt(np.asarray(re, dtype=float) ** 2 + np.asarray(im, dtype=float) ** 2)
    else:
        raise ValueError(part)
    return R, Z, F




# ----------------------------- pressure diagnostics -----------------------------

def _complex_part_from_pair(re: np.ndarray, im: np.ndarray, part: str) -> np.ndarray:
    re = np.asarray(re, dtype=float)
    im = np.asarray(im, dtype=float)
    if part == "real":
        return re
    if part == "imag":
        return im
    if part == "amp":
        return np.sqrt(re * re + im * im)
    raise ValueError(part)


def _safe_add_candidate(cands: dict[str, np.ndarray], name: str, func, part: str) -> None:
    try:
        re, im = func()
        cands[name] = _complex_part_from_pair(re, im, part)
    except Exception as exc:
        cands[name + " [unavailable: " + str(exc).splitlines()[0] + "]"] = None  # type: ignore[assignment]


def pressure_candidate_fields(
    data: dict,
    mode_index: int,
    part: str,
    ids_species_index: int,
    native_species_index: int,
    electrons_index: int,
) -> dict[str, np.ndarray]:
    """Build several plausible native pressure references for debugging.

    This intentionally separates the IMAS occurrence index from the NIMROD native
    species index.  In NIMROD dump files, electrons are usually stored in separate
    arrays such as rete/imte and repe/impe, while ion density arrays such as
    rend/imnd and nq usually start with main ion species index 0.  Therefore,
    occ=2 in IMAS does not automatically mean native species index 1.
    """
    cands: dict[str, np.ndarray] = {}

    def add(name: str, func) -> None:
        _safe_add_candidate(cands, name, func, part)

    # The reference used by the main validation when --pressure-source=auto.
    add(f"dump2imas_auto_ids_species_s{ids_species_index}", lambda: dump_pressure_fields(data, ids_species_index, mode_index, electrons_index=electrons_index, pressure_source="auto"))

    add("repr_total", lambda: dump_pressure_fields(data, native_species_index, mode_index, electrons_index=-999999, pressure_source="total"))
    add("repr_minus_repe", lambda: dump_pressure_fields(data, native_species_index, mode_index, electrons_index=-999999, pressure_source="total_minus_electron"))
    add(f"ion_reconstruct_native_s{native_species_index}", lambda: dump_pressure_fields(data, native_species_index, mode_index, electrons_index=-999999, pressure_source="reconstruct"))
    add(f"ion_thermal_only_native_s{native_species_index}", lambda: dump_pressure_fields(data, native_species_index, mode_index, electrons_index=-999999, pressure_source="thermal_only"))
    add(f"ion_density_only_native_s{native_species_index}", lambda: dump_pressure_fields(data, native_species_index, mode_index, electrons_index=-999999, pressure_source="density_only"))
    add(f"old_reconstruct_ids_species_s{ids_species_index}", lambda: dump_pressure_fields(data, ids_species_index, mode_index, electrons_index=electrons_index, pressure_source="reconstruct"))

    # Also try species 0 and 1 explicitly when possible.
    for s in (0, 1, 2):
        add(f"ion_reconstruct_s{s}", lambda s=s: dump_pressure_fields(data, s, mode_index, electrons_index=-999999, pressure_source="reconstruct"))
        add(f"ion_thermal_only_s{s}", lambda s=s: dump_pressure_fields(data, s, mode_index, electrons_index=-999999, pressure_source="thermal_only"))
        add(f"ion_density_only_s{s}", lambda s=s: dump_pressure_fields(data, s, mode_index, electrons_index=-999999, pressure_source="density_only"))

    # Electron pressure and reconstructed total pressure from electron+main-ion terms.
    add("repe_electron", lambda: dump_pressure_fields(data, 0, mode_index, electrons_index=0, pressure_source="explicit"))
    add("electron_reconstruct", lambda: dump_pressure_fields(data, 0, mode_index, electrons_index=0, pressure_source="reconstruct"))
    add("electron_thermal_only", lambda: dump_pressure_fields(data, 0, mode_index, electrons_index=0, pressure_source="thermal_only"))
    add("electron_density_only", lambda: dump_pressure_fields(data, 0, mode_index, electrons_index=0, pressure_source="density_only"))
    try:
        e_re, e_im = dump_pressure_fields(data, 0, mode_index, electrons_index=0, pressure_source="reconstruct")
        i_re, i_im = dump_pressure_fields(data, native_species_index, mode_index, electrons_index=-999999, pressure_source="reconstruct")
        cands[f"reconstruct_electron_plus_ion_s{native_species_index}"] = _complex_part_from_pair(e_re + i_re, e_im + i_im, part)
    except Exception as exc:
        cands[f"reconstruct_electron_plus_ion_s{native_species_index} [unavailable: {str(exc).splitlines()[0]}]"] = None  # type: ignore[assignment]

    try:
        t_re, t_im = dump_pressure_fields(data, native_species_index, mode_index, electrons_index=-999999, pressure_source="total")
        e_re, e_im = dump_pressure_fields(data, 0, mode_index, electrons_index=0, pressure_source="explicit")
        cands["repr_plus_repe"] = _complex_part_from_pair(t_re + e_re, t_im + e_im, part)
    except Exception as exc:
        cands[f"repr_plus_repe [unavailable: {str(exc).splitlines()[0]}]"] = None  # type: ignore[assignment]

    return cands


def print_pressure_diagnostics(
    R_dump: np.ndarray,
    Z_dump: np.ndarray,
    R_ids: np.ndarray,
    Z_ids: np.ndarray,
    F_ids: np.ndarray,
    data: dict,
    mode_index: int,
    part: str,
    ids_species_index: int,
    native_species_index: int,
    electrons_index: int,
) -> None:
    print("pressure_diagnostics_begin=1")
    print(f"ids_species_index={ids_species_index}")
    print(f"native_species_index={native_species_index}")
    candidates = pressure_candidate_fields(
        data=data,
        mode_index=mode_index,
        part=part,
        ids_species_index=ids_species_index,
        native_species_index=native_species_index,
        electrons_index=electrons_index,
    )
    valid_for_ls: list[tuple[str, np.ndarray]] = []
    for name, arr in candidates.items():
        if arr is None:
            print(f"candidate={name}")
            continue
        try:
            _Rd, _Zd, Fc, _Ri, Fi = align_fields(R_dump, Z_dump, arr, R_ids, Z_ids, F_ids)
            met = compute_metrics(Fc, Fi)
            print(
                "candidate={name} l2_rel={l2:.16e} rmse={rmse:.16e} max_abs={max_abs:.16e} corr={corr:.16e} "
                "cand_l2={cl2:.16e} ids_l2={il2:.16e} cand_min={cmin:.16e} cand_max={cmax:.16e}".format(
                    name=name,
                    l2=met.get("l2_rel", float("nan")),
                    rmse=met.get("rmse", float("nan")),
                    max_abs=met.get("max_abs", float("nan")),
                    corr=met.get("corr", float("nan")),
                    cl2=float(np.linalg.norm(np.asarray(Fc, dtype=float))),
                    il2=float(np.linalg.norm(np.asarray(Fi, dtype=float))),
                    cmin=float(np.nanmin(Fc)),
                    cmax=float(np.nanmax(Fc)),
                )
            )
            if name in ("repr_total", "repr_minus_repe", f"ion_reconstruct_native_s{native_species_index}", "repe_electron"):
                valid_for_ls.append((name, np.asarray(Fc, dtype=float)))
        except Exception as exc:
            print(f"candidate={name} [diagnostic_failed: {str(exc).splitlines()[0]}]")

    # Least-squares fingerprint: if IMAS is a simple linear combination of the common
    # candidate arrays, this often exposes the coefficients immediately.
    if len(valid_for_ls) >= 2:
        mask = np.isfinite(F_ids)
        cols = []
        names = []
        for name, arr in valid_for_ls:
            if arr.shape == F_ids.shape:
                mask &= np.isfinite(arr)
                cols.append(arr)
                names.append(name)
        if cols and np.any(mask):
            A = np.vstack([arr[mask].ravel() for arr in cols]).T
            y = np.asarray(F_ids, dtype=float)[mask].ravel()
            try:
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                yhat = A @ coef
                rel = float(np.linalg.norm(yhat - y) / np.linalg.norm(y)) if np.linalg.norm(y) > 0 else float("nan")
                coef_s = ",".join(f"{n}:{c:.16e}" for n, c in zip(names, coef))
                print(f"least_squares_to_ids={coef_s} rel_residual={rel:.16e}")
            except Exception as exc:
                print(f"least_squares_to_ids_failed={str(exc).splitlines()[0]}")
    print("pressure_diagnostics_end=1")


# ----------------------------- metrics / plotting -----------------------------

def align_fields(R1: np.ndarray, Z1: np.ndarray, F1: np.ndarray, R2: np.ndarray, Z2: np.ndarray, F2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if F1.shape == F2.shape:
        return R1, Z1, F1, R2, F2
    if F1.shape == F2.T.shape:
        return R1, Z1, F1, R2.T, F2.T
    raise RuntimeError(f"field shape mismatch: dump {F1.shape} vs IDS {F2.shape}")


def compute_metrics(ref: np.ndarray, test: np.ndarray) -> dict:
    m = np.isfinite(ref) & np.isfinite(test)
    if not np.any(m):
        raise RuntimeError("no overlapping finite points to compare")
    a = ref[m].ravel().astype(float)
    b = test[m].ravel().astype(float)
    d = b - a
    ref_norm = np.linalg.norm(a)
    rms_ref = np.sqrt(np.mean(a * a))
    out = {
        "n_points": int(a.size),
        "l2_rel": float(np.linalg.norm(d) / ref_norm) if ref_norm > 0 else np.nan,
        "rmse": float(np.sqrt(np.mean(d * d))),
        "nrmse_rms": float(np.sqrt(np.mean(d * d)) / rms_ref) if rms_ref > 0 else np.nan,
        "max_abs": float(np.max(np.abs(d))),
        "max_rel_peak": float(np.max(np.abs(d)) / np.max(np.abs(a))) if np.max(np.abs(a)) > 0 else np.nan,
        "mean_abs": float(np.mean(np.abs(d))),
        "bias": float(np.mean(d)),
        "corr": float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else np.nan,
    }
    return out


def array_signature(a: np.ndarray) -> str:
    aa = np.asarray(a, dtype=np.float64)
    return hashlib.sha256(np.ascontiguousarray(aa).tobytes()).hexdigest()[:16]


def _levels_from_range(vmin: float, vmax: float, n: int = 41) -> np.ndarray:
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return np.linspace(-1.0, 1.0, n)
    if vmax == vmin:
        eps = abs(vmax) * 1e-12 if vmax != 0.0 else 1.0
        return np.linspace(vmin - eps, vmax + eps, n)
    return np.linspace(vmin, vmax, n)


def make_plot(R: np.ndarray, Z: np.ndarray, dump_F: np.ndarray, ids_F: np.ndarray, title: str, out: str, scale_mode: str = "shared") -> None:
    diff = ids_F - dump_F
    finite = np.isfinite(dump_F) & np.isfinite(ids_F)
    if not np.any(finite):
        raise RuntimeError("no finite overlap for plotting")

    scale_mode = str(scale_mode or "shared").lower()
    if scale_mode not in ("shared", "dump", "independent"):
        raise RuntimeError(f"unknown scale_mode={scale_mode!r}; use shared|dump|independent")

    if scale_mode == "shared":
        vmin0 = vmin1 = float(np.nanmin(np.r_[dump_F[finite], ids_F[finite]]))
        vmax0 = vmax1 = float(np.nanmax(np.r_[dump_F[finite], ids_F[finite]]))
    elif scale_mode == "dump":
        vmin0 = vmin1 = float(np.nanmin(dump_F[finite]))
        vmax0 = vmax1 = float(np.nanmax(dump_F[finite]))
    else:  # independent
        vmin0 = float(np.nanmin(dump_F[finite]))
        vmax0 = float(np.nanmax(dump_F[finite]))
        vmin1 = float(np.nanmin(ids_F[finite]))
        vmax1 = float(np.nanmax(ids_F[finite]))

    levels0 = _levels_from_range(vmin0, vmax0, 41)
    levels1 = _levels_from_range(vmin1, vmax1, 41)

    dmax = float(np.nanmax(np.abs(diff[finite])))
    if dmax == 0.0:
        dmax = 1.0e-15
    levels_diff = _levels_from_range(-dmax, dmax, 41)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    cf0 = axs[0].contourf(R, Z, dump_F, levels=levels0, cmap="RdBu_r", extend="both")
    axs[0].set_title("Dump")
    axs[0].set_xlabel("R [m]")
    axs[0].set_ylabel("Z [m]")
    axs[0].set_aspect("equal", adjustable="box")

    cf1 = axs[1].contourf(R, Z, ids_F, levels=levels1, cmap="RdBu_r", extend="both")
    axs[1].set_title("IMAS mhd_linear")
    axs[1].set_xlabel("R [m]")
    axs[1].set_aspect("equal", adjustable="box")

    cf2 = axs[2].contourf(R, Z, diff, levels=levels_diff, cmap="RdBu_r", extend="both")
    axs[2].contour(R, Z, dump_F, levels=12, colors="k", linewidths=0.5, alpha=0.5)
    axs[2].contour(R, Z, ids_F, levels=12, colors="w", linewidths=0.5, alpha=0.7)
    axs[2].set_title("Difference (IMAS - dump)")
    axs[2].set_xlabel("R [m]")
    axs[2].set_aspect("equal", adjustable="box")

    fig.colorbar(cf0, ax=axs[0], shrink=0.92, label="dump value")
    fig.colorbar(cf1, ax=axs[1], shrink=0.92, label="IDS value")
    fig.colorbar(cf2, ax=axs[2], shrink=0.92, label="difference")
    fig.suptitle(title)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- entry-path helper -----------------------------

def resolve_validator_entry_path(args: argparse.Namespace) -> Path:
    """Resolve the IMAS entry directory from either --entry or standard IMAS options.

    This mirrors the directory convention used by dump2imas/input2imas for the
    HDF5 backend: <dbpath>/<dd>/<major-version>/<pulse>/<run>.  It intentionally
    does not depend on nimrod2imas.add_entry_args so the validator remains stable
    if that helper changes.
    """
    if getattr(args, "entry", None):
        return Path(str(args.entry)).expanduser()

    missing = []
    for name in ("dd", "pulse", "run"):
        if getattr(args, name, None) in (None, ""):
            missing.append("--" + name.replace("_", "-"))
    if missing:
        raise RuntimeError(
            "provide either --entry, or standard IMAS entry options "
            "--dbpath --dd --pulse --run; missing " + ", ".join(missing)
        )

    dd_version_dir = getattr(args, "dd_version_dir", None)
    if dd_version_dir in (None, ""):
        ddv = str(getattr(args, "dd_version", "4.1.1") or "4.1.1")
        dd_version_dir = ddv.split(".", 1)[0] if ddv else "4"

    return (
        Path(str(getattr(args, "dbpath", ".") or ".")).expanduser()
        / str(args.dd)
        / str(dd_version_dir)
        / str(int(args.pulse))
        / str(int(args.run))
    )


# ----------------------------- CLI -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare dumpgll perturbations against IMAS mhd_linear and compute validation metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dumpgll", help="Input dumpgll HDF5 file")

    # Entry-location options.  These are intentionally defined locally rather than
    # through nimrod2imas.add_entry_args so this validator keeps the standard
    # dump2imas-style CLI even when the shared helper changes.
    # Use either:
    #   --entry ./d3d/4/163518/5
    # or:
    #   --backend hdf5 --dbpath . --dd d3d --dd-version 4.1.1 --pulse 163518 --run 5
    p.add_argument("--entry", default=None,
                   help="IMAS HDF5 entry directory, e.g. ./d3d/4/163518/5. If omitted, construct it from --dbpath/--dd/--pulse/--run.")
    p.add_argument("--backend", default="hdf5",
                   help="IMAS backend passed to nimrod2imas.open_dbentry")
    p.add_argument("--dbpath", default=".",
                   help="Base IMAS database path used when --entry is omitted")
    p.add_argument("--dd", default=None,
                   help="IMAS database/device name used when --entry is omitted, e.g. d3d")
    p.add_argument("--dd-version", default="4.1.1",
                   help="IMAS Data Dictionary version")
    p.add_argument("--dd-version-dir", default=None,
                   help="Optional IMAS major-version directory override, e.g. 4")
    p.add_argument("--pulse", type=int, default=None,
                   help="IMAS pulse/shot number used when --entry is omitted")
    p.add_argument("--run", type=int, default=None,
                   help="IMAS run number used when --entry is omitted")
    p.add_argument("--occ", type=int, default=1,
                   help="mhd_linear occurrence to compare")

    p.add_argument("--occ-base", type=int, default=1,
                   help="base occurrence used by dump2imas; default matches dump2imas")
    p.add_argument("--quantity", required=True, choices=["n", "t", "p", "b", "v"])
    p.add_argument("--part", default="real", choices=["real", "imag", "amp"])
    p.add_argument("--component", default=None, choices=["r", "z", "phi"])

    # IDS time selection.  --ids-time-index is the preferred spelling; --time-index is kept as an alias.
    p.add_argument("--ids-time-index", "--time-index", dest="ids_time_index", type=int, default=0,
                   help="mhd_linear time slice to compare. By default this indexes non-empty time_slices.")
    p.add_argument("--raw-ids-time-index", "--raw-time-index", dest="raw_ids_time_index", action="store_true",
                   help="interpret --ids-time-index as a raw time_slice index rather than a non-empty-slice index")
    p.add_argument("--ids-time-value", type=float, default=None,
                   help="select the mhd_linear time_slice closest to this time value")
    p.add_argument("--ids-time-tolerance", type=float, default=None,
                   help="optional maximum allowed |t_ids - requested_time| when using --ids-time-value or --match-dump-time")
    p.add_argument("--match-dump-time", action="store_true",
                   help="select the IDS time_slice closest to the time stored in the input dumpgll file")
    p.add_argument("--list-ids-times", action="store_true",
                   help="print available mhd_linear time_slices and exit after opening the IDS")

    p.add_argument("--mode-index", type=int, default=None)
    p.add_argument("--n-tor", type=int, default=None)
    p.add_argument("--dens-pert-order", default="species_major", choices=["species_major", "mode_major"])
    p.add_argument("--ion-mass-amu", type=float, default=2.0,
                   help="ion mass used by dump2imas for mass_density_perturbed conversion")
    p.add_argument("--electrons-index", type=int, default=0,
                   help="legacy index used to identify electron occurrence when --species-role=auto")
    p.add_argument("--species-role", default="auto", choices=["auto", "electron", "ion"],
                   help=("physical role of this IMAS occurrence. In auto mode, occ-occ-base equal to "
                         "--electrons-index is treated as electron; all other occurrences are treated as ion."))
    p.add_argument("--native-species-index", type=int, default=None,
                   help=("diagnostic override for the species index used in native density/pressure references. "
                         "For converter-consistent validation, omit this option; the script then uses occ-occ-base, "
                         "matching dump2imas populate_mhd_linear."))
    p.add_argument("--diagnose-pressure", action="store_true",
                   help="print metrics for several possible native pressure definitions against the IMAS pressure leaf")
    p.add_argument("--pressure-source", default="auto",
                   choices=["auto", "dump2imas", "explicit", "total", "total_minus_electron", "reconstruct", "thermal_only", "density_only"],
                   help=("native-dump pressure reference. 'auto'/'dump2imas' mirrors dump2imas populate_mhd_linear; "
                         "other choices are diagnostic alternatives."))
    p.add_argument("--nimrod-in", default=None,
                   help="optional nimrod.in path; when supplied, read_and_stitch_dump uses the same species metadata as dump2imas")
    p.add_argument("--no-nimrod-in-auto", action="store_true",
                   help="do not automatically look for nimrod.in in the current directory or dump directory")
    p.add_argument("--plot", default=None, help="optional path for a 3-panel contour comparison plot")
    p.add_argument("--scale-mode", default="shared", choices=["shared", "dump", "independent"],
                   help="color scaling for the first two panels: shared, dump, or independent")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.dd_version:
        args.dd_version = "4.1.1"
    ids_species_index = int(args.occ) - int(args.occ_base)
    if ids_species_index < 0:
        raise SystemExit(f"occ={args.occ} is smaller than occ-base={args.occ_base}")
    if args.species_role == "auto":
        species_role = "electron" if ids_species_index == int(args.electrons_index) else "ion"
    else:
        species_role = args.species_role

    # For verification of the current converter, the default species used on the native side
    # must be the same integer that dump2imas used while writing this occurrence.  The older
    # --native-species-index option is retained as an explicit override for diagnostics only.
    converter_species_index = ids_species_index
    compare_species_index = int(args.native_species_index) if args.native_species_index is not None else converter_species_index
    effective_electrons_index = int(args.electrons_index) if species_role == "electron" else -999999

    # Read and stitch the native dump using as much of the same dump2imas context as possible.
    dump_args = SimpleNamespace(time=None, dens_pert_order=args.dens_pert_order, ion_mass_amu=args.ion_mass_amu)
    nimrod_in_path = args.nimrod_in
    if nimrod_in_path is None and not args.no_nimrod_in_auto:
        candidates = [Path("nimrod.in"), Path(args.dumpgll).resolve().parent / "nimrod.in"]
        for c in candidates:
            if c.is_file():
                nimrod_in_path = str(c)
                break
    if nimrod_in_path:
        setattr(dump_args, "_series_nimrod_in_path", str(nimrod_in_path))
        if _nimrod_species_info is not None:
            try:
                setattr(dump_args, "_nimrod_species", _nimrod_species_info(str(nimrod_in_path)))
            except Exception:
                pass
    data = read_and_stitch_dump(Path(args.dumpgll), dump_args)

    # Determine dump mode index using keff/n_tor if possible.
    nm = int(data.get("nmodes", 0) or 0)
    if nm <= 0:
        raise SystemExit("native dump does not appear to contain perturbation modes")
    if args.mode_index is not None:
        dump_mode = int(args.mode_index)
    elif args.n_tor is not None:
        keff = np.ravel(np.asarray(data.get("keff", []), dtype=float))
        matches = np.where(np.rint(keff).astype(int) == int(args.n_tor))[0]
        if matches.size == 0:
            raise SystemExit(f"n_tor={args.n_tor} not found in dump keff={keff.tolist()}")
        dump_mode = int(matches[0])
    else:
        dump_mode = 0
    if dump_mode < 0 or dump_mode >= nm:
        raise SystemExit(f"dump mode index {dump_mode} out of range 0..{nm-1}")

    R_dump, Z_dump, F_dump = extract_dump_field(
        data,
        quantity=args.quantity,
        species_index=compare_species_index,
        mode_index=dump_mode,
        part=args.part,
        component=args.component,
        ion_mass_amu=args.ion_mass_amu,
        electrons_index=effective_electrons_index,
        pressure_source=args.pressure_source,
    )

    # Open IMAS mhd_linear.
    entry = resolve_validator_entry_path(args)
    db, _uri, imas = open_dbentry(args.backend, str(entry), mode="r", dd_version=args.dd_version)
    try:
        factory = ids_factory(imas, args.dd_version or "")
        mhd = get_ids(db, factory, "mhd_linear", int(args.occ))
        if args.list_ids_times:
            print(format_time_slice_table(mhd))
            return
        ids_time_value = args.ids_time_value
        if args.match_dump_time:
            if ids_time_value is not None:
                raise RuntimeError("use either --ids-time-value or --match-dump-time, not both")
            ids_time_value = float(data.get("time", np.nan))
            if not np.isfinite(ids_time_value):
                raise RuntimeError("--match-dump-time requested, but the dump time is not finite")
        R_ids, Z_ids, F_ids, ti_raw, mi, ids_time = extract_ids_field(
            mhd,
            ids_time_index=args.ids_time_index,
            raw_ids_time_index=args.raw_ids_time_index,
            ids_time_value=ids_time_value,
            ids_time_tolerance=args.ids_time_tolerance,
            n_tor=args.n_tor,
            mode_index=args.mode_index,
            quantity=args.quantity,
            part=args.part,
            component=args.component,
        )
    finally:
        try:
            db.close()
        except Exception:
            pass

    R_dump, Z_dump, F_dump, R_ids, F_ids = align_fields(R_dump, Z_dump, F_dump, R_ids, Z_ids, F_ids)
    metrics = compute_metrics(F_dump, F_ids)

    print(f"quantity={args.quantity} part={args.part} component={args.component} occ={args.occ} ids_species_index={ids_species_index}")
    print(f"species_role={species_role}")
    print(f"compare_species_index={compare_species_index}")
    print(f"converter_species_index={converter_species_index}")
    print(f"entry={entry}")
    print(f"backend={args.backend}")
    print(f"dd_version={args.dd_version}")
    print(f"nimrod_in={nimrod_in_path if nimrod_in_path else 'None'}")
    if args.quantity == "p":
        print(f"pressure_source={args.pressure_source}")
    print(f"dump_time={float(data.get('time', np.nan)):.12g}")
    print(f"ids_time={ids_time:.12g}  ids_time_slice_raw_index={ti_raw}  ids_mode_index={mi}")
    print(f"dump_mode_index={dump_mode}")
    print(f"dump_shape={F_dump.shape} ids_shape={F_ids.shape}")
    print(f"dump_min={float(np.nanmin(F_dump)):.16e} dump_max={float(np.nanmax(F_dump)):.16e}")
    print(f"ids_min={float(np.nanmin(F_ids)):.16e} ids_max={float(np.nanmax(F_ids)):.16e}")
    print(f"dump_l2={float(np.linalg.norm(np.asarray(F_dump, dtype=float))):.16e}")
    print(f"ids_l2={float(np.linalg.norm(np.asarray(F_ids, dtype=float))):.16e}")
    print(f"dump_sha16={array_signature(F_dump)}")
    print(f"ids_sha16={array_signature(F_ids)}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}={v:.16e}")
        else:
            print(f"{k}={v}")

    if args.quantity == "p" and args.diagnose_pressure:
        print_pressure_diagnostics(
            R_dump=R_dump,
            Z_dump=Z_dump,
            R_ids=R_ids,
            Z_ids=Z_ids,
            F_ids=F_ids,
            data=data,
            mode_index=dump_mode,
            part=args.part,
            ids_species_index=ids_species_index,
            native_species_index=compare_species_index,
            electrons_index=args.electrons_index,
        )

    if args.plot:
        title = f"{args.quantity} {args.part} occ={args.occ} mode={args.n_tor if args.n_tor is not None else dump_mode}"
        if args.quantity in ("b", "v") and args.component:
            title += f" component={args.component}"
        make_plot(R_dump, Z_dump, F_dump, F_ids, title=title, out=args.plot, scale_mode=args.scale_mode)
        print(f"wrote plot: {args.plot}")


if __name__ == "__main__":
    main()
