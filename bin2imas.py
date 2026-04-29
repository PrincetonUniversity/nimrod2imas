#!/usr/bin/env python3
"""bin2imas.py

Read NIMROD time-history binary diagnostics and map them into IMAS.

Supported inputs:
  - energy*.bin    (step, time, imode, k, E_mag, E_kin, prad)
  - discharge*.bin (step, time, discharge scalars)

Primary mappings:
  - summary IDS (discharge global traces)
  - mhd_linear IDS (per-mode time traces + growthrate from E_mag+E_kin)
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from nimrod2imas import (
    ensure_entry_dir,
    entry_dir,
    open_dbentry,
    ids_factory,
    get_ids,
    put_ids,
    sanitize_cli_command,
    update_workflow_and_dataset_fair,
)

__version__ = "0.1.0"

ENERGY_LABELS = ("step", "time", "imode", "k", "E_mag", "E_kin", "prad")
DISCHARGE_LABELS = (
    "istep",
    "time",
    "divB",
    "totE",
    "totIE",
    "totIEe",
    "totIEi",
    "lnE",
    "lnIE",
    "grate",
    "Itot",
    "Ipert",
    "Vloop",
    "totflux",
    "n0flux",
    "bigF",
    "Theta",
    "magCFL",
    "NLCFL",
    "flowCFL",
)
KPRADEN_LABELS = (
    "istep",
    "time",
    "qlosd",
    "qloso",
    "qlosb",
    "qlosr",
    "qlosl",
    "qlosi",
    "Nz",
    "Ne",
    "Nz+zi",
    "qlost",
    "elosd",
    "eloso",
    "elosb",
    "elosr",
    "elosl",
    "elosi",
    "elost",
)


def _read_fortran_records_f32(path: str | Path, endian: str = ">") -> Iterable[np.ndarray]:
    p = Path(path)
    with p.open("rb") as f:
        while True:
            head = f.read(4)
            if not head:
                break
            if len(head) != 4:
                raise IOError(f"{p}: truncated record header")
            nbytes = struct.unpack(endian + "i", head)[0]
            if nbytes == 0:
                continue
            if nbytes < 0:
                raise IOError(f"{p}: negative record length {nbytes}")
            payload = f.read(nbytes)
            tail = f.read(4)
            if len(payload) != nbytes or len(tail) != 4:
                raise IOError(f"{p}: truncated record payload")
            nbytes2 = struct.unpack(endian + "i", tail)[0]
            if nbytes2 != nbytes:
                raise IOError(f"{p}: record length mismatch ({nbytes} != {nbytes2})")
            if nbytes % 4 != 0:
                raise IOError(f"{p}: record byte length {nbytes} is not divisible by 4")
            yield np.frombuffer(payload, dtype=np.dtype(endian + "f4")).astype(np.float64)


def _rows_from_file(path: Path, ncols: int, endian: str) -> np.ndarray:
    rows: List[np.ndarray] = []
    for rec in _read_fortran_records_f32(path, endian=endian):
        if rec.size == 0:
            continue
        if rec.size != ncols:
            raise ValueError(f"{path}: expected {ncols} float32 values per record, got {rec.size}")
        rows.append(rec)
    if not rows:
        return np.zeros((0, ncols), dtype=np.float64)
    return np.vstack(rows)


def _rows_from_energy_file(path: Path, endian: str) -> np.ndarray:
    """
    Read energy rows accepting both:
      - 6 columns: step,time,imode,k,E_mag,E_kin
      - 7 columns: step,time,imode,k,E_mag,E_kin,prad
    Missing prad is filled with NaN.
    """
    rows: List[np.ndarray] = []
    for rec in _read_fortran_records_f32(path, endian=endian):
        if rec.size == 0:
            continue
        if rec.size == 6:
            rows.append(np.concatenate([rec, np.array([np.nan], dtype=np.float64)]))
            continue
        if rec.size == 7:
            rows.append(rec)
            continue
        raise ValueError(
            f"{path}: expected 6 or 7 float32 values per energy record, got {rec.size}"
        )
    if not rows:
        return np.zeros((0, 7), dtype=np.float64)
    return np.vstack(rows)


def _dedupe_rows(rows: np.ndarray, key_cols: Sequence[int]) -> np.ndarray:
    if rows.size == 0:
        return rows
    keep: Dict[Tuple[float, ...], int] = {}
    for i in range(rows.shape[0]):
        key = tuple(float(rows[i, c]) for c in key_cols)
        keep[key] = i  # keep latest occurrence (restart continuation)
    idx = np.array(sorted(keep.values()), dtype=int)
    return rows[idx, :]


def _concat_sorted(parts: List[np.ndarray], sort_cols: Sequence[int], key_cols: Sequence[int]) -> np.ndarray:
    parts = [p for p in parts if p.size > 0]
    if not parts:
        return np.zeros((0, 0), dtype=np.float64)
    out = np.vstack(parts)
    lex = tuple(out[:, c] for c in reversed(sort_cols))
    out = out[np.lexsort(lex)]
    return _dedupe_rows(out, key_cols=key_cols)


def load_energy(files: Sequence[Path], endian: str = ">") -> np.ndarray:
    parts = [_rows_from_energy_file(fp, endian=endian) for fp in files]
    out = _concat_sorted(parts, sort_cols=(0, 1, 2, 3), key_cols=(0, 1, 2, 3))
    return out if out.size else np.zeros((0, 7), dtype=np.float64)


def load_discharge(files: Sequence[Path], endian: str = ">") -> np.ndarray:
    parts = [_rows_from_file(fp, ncols=20, endian=endian) for fp in files]
    out = _concat_sorted(parts, sort_cols=(0, 1), key_cols=(0, 1))
    return out if out.size else np.zeros((0, 20), dtype=np.float64)


def _rows_from_kpraden_file(path: Path, endian: str) -> np.ndarray:
    """
    Read kpraden rows accepting both:
      - 19 columns: legacy format (istep, time, qlosd, qloso, qlosb, qlosr, qlosl, qlosi, Nz, Ne, Nz+zi, qlost, elosd, eloso, elosb, elosr, elosl, elosi, elost)
      - 22 columns: extended format with 3 additional fields
    Extra columns beyond 19 are truncated.
    """
    rows: List[np.ndarray] = []
    for rec in _read_fortran_records_f32(path, endian=endian):
        if rec.size == 0:
            continue
        if rec.size == 19:
            rows.append(rec)
            continue
        if rec.size == 22:
            rows.append(rec[:19])  # truncate to legacy 19-column format
            continue
        raise ValueError(
            f"{path}: expected 19 or 22 float32 values per kpraden record, got {rec.size}"
        )
    if not rows:
        return np.zeros((0, 19), dtype=np.float64)
    return np.vstack(rows)


def load_kpraden(files: Sequence[Path], endian: str = ">") -> np.ndarray:
    parts = [_rows_from_kpraden_file(fp, endian=endian) for fp in files]
    out = _concat_sorted(parts, sort_cols=(0, 1), key_cols=(0, 1))
    return out if out.size else np.zeros((0, 19), dtype=np.float64)


def _safe_growth_from_energy(Etot: np.ndarray, t: np.ndarray) -> np.ndarray:
    g = np.zeros_like(Etot, dtype=np.float64)
    if Etot.size < 2:
        return g
    dt = np.diff(t)
    e0 = Etot[:-1]
    e1 = Etot[1:]
    good = (dt > 0) & np.isfinite(dt) & (e0 > 0) & (e1 > 0) & np.isfinite(e0) & np.isfinite(e1)
    if np.any(good):
        g[1:][good] = (np.log(e1[good]) - np.log(e0[good])) / (2.0 * dt[good])
    return g


def build_energy_mode_matrices(energy_rows: np.ndarray) -> Dict[str, Any]:
    if energy_rows.size == 0:
        return {
            "time": np.zeros(0, dtype=np.float64),
            "modes": np.zeros(0, dtype=np.int64),
            "mag": np.zeros((0, 0), dtype=np.float64),
            "kin": np.zeros((0, 0), dtype=np.float64),
            "prad": np.zeros(0, dtype=np.float64),
            "growth": np.zeros((0, 0), dtype=np.float64),
        }

    tvals = np.unique(energy_rows[:, 1])
    modes = np.unique(np.rint(energy_rows[:, 3]).astype(np.int64))
    it = {float(t): i for i, t in enumerate(tvals.tolist())}
    im = {int(m): i for i, m in enumerate(modes.tolist())}

    mag = np.full((tvals.size, modes.size), np.nan, dtype=np.float64)
    kin = np.full((tvals.size, modes.size), np.nan, dtype=np.float64)

    for row in energy_rows:
        ti = it.get(float(row[1]))
        mi = im.get(int(round(float(row[3]))))
        if ti is None or mi is None:
            continue
        mag[ti, mi] = float(row[4])
        kin[ti, mi] = float(row[5])

    # pragma is global-by-time in NIMROD output (same per mode each row); keep first finite per t
    prad = np.full(tvals.size, np.nan, dtype=np.float64)
    for i, t in enumerate(tvals.tolist()):
        m = energy_rows[:, 1] == float(t)
        v = energy_rows[m, 6]
        vv = v[np.isfinite(v)]
        if vv.size:
            prad[i] = float(vv[0])

    growth = np.full((tvals.size, modes.size), np.nan, dtype=np.float64)
    for j in range(modes.size):
        etot = mag[:, j] + kin[:, j]
        ok = np.isfinite(etot) & np.isfinite(tvals)
        if np.sum(ok) < 2:
            continue
        sub_g = _safe_growth_from_energy(etot[ok], tvals[ok])
        growth[np.where(ok)[0], j] = sub_g

    return {
        "time": tvals,
        "modes": modes,
        "mag": mag,
        "kin": kin,
        "prad": prad,
        "growth": growth,
    }


def _aos_len(aos: Any) -> int:
    try:
        return len(aos)
    except Exception:
        try:
            return int(aos.size())
        except Exception:
            return 0


def _aos_resize(aos: Any, n: int) -> bool:
    try:
        aos.resize(int(n))
        return True
    except Exception:
        return False


def _set_if_has(obj: Any, attr: str, value: Any) -> bool:
    try:
        if hasattr(obj, attr):
            setattr(obj, attr, value)
            return True
    except Exception:
        return False
    return False


def _resolve_path(obj: Any, path: Sequence[str]) -> Optional[Any]:
    cur = obj
    for p in path:
        if not hasattr(cur, p):
            return None
        try:
            cur = getattr(cur, p)
        except Exception:
            return None
    return cur


def _set_path(obj: Any, path: Sequence[str], value: Any) -> bool:
    if len(path) == 0:
        return False
    parent = _resolve_path(obj, path[:-1]) if len(path) > 1 else obj
    if parent is None:
        return False
    return _set_if_has(parent, path[-1], value)


def _best_set_path(obj: Any, candidates: Sequence[Sequence[str]], value: Any) -> bool:
    for c in candidates:
        if _set_path(obj, c, value):
            return True
    return False


def _set_ids_properties(ids_obj: Any, homogeneous_time: int = 1, comment: Optional[str] = None) -> None:
    ip = getattr(ids_obj, "ids_properties", None)
    if ip is None:
        return
    _set_if_has(ip, "homogeneous_time", int(homogeneous_time))
    if comment:
        try:
            prev = str(getattr(ip, "comment", "") or "").strip()
            txt = (prev + "\n" + comment).strip() if prev else comment
            _set_if_has(ip, "comment", txt)
        except Exception:
            pass


def write_summary_from_discharge(summary: Any, discharge_rows: np.ndarray, *, dd: str, pulse: int) -> Dict[str, bool]:
    status: Dict[str, bool] = {}
    if discharge_rows.size == 0:
        return status

    t = np.asarray(discharge_rows[:, 1], dtype=np.float64)
    _set_if_has(summary, "time", t)
    _set_if_has(summary, "machine", str(dd))
    _set_if_has(summary, "pulse", int(pulse))
    _set_if_has(summary, "description", "NIMROD discharge time history mapped by bin2imas")

    # Known/likely summary paths across DD variants.
    mapping: Dict[str, List[Tuple[Sequence[str], int]]] = {
        "ip": [(("global_quantities", "ip", "value"), 10), (("global_quantities", "ip"), 10)],
        "v_loop": [(("global_quantities", "v_loop", "value"), 12), (("global_quantities", "v_loop"), 12)],
        "w_mhd": [(("global_quantities", "w_mhd", "value"), 3), (("global_quantities", "w_mhd"), 3)],
        "beta_pol": [(("global_quantities", "beta_pol", "value"), 16), (("global_quantities", "beta_pol"), 16)],
        "psi_boundary": [(("global_quantities", "psi_boundary", "value"), 13), (("global_quantities", "psi_boundary"), 13)],
    }

    for name, cands in mapping.items():
        ok = False
        for path, col in cands:
            v = np.asarray(discharge_rows[:, col], dtype=np.float64)
            if _set_path(summary, path, v):
                ok = True
                break
        status[name] = ok

    # Additional energies if available in DD.
    status["energy_thermal"] = _best_set_path(
        summary,
        (("global_quantities", "energy_thermal", "value"), ("global_quantities", "energy_thermal")),
        np.asarray(discharge_rows[:, 4], dtype=np.float64),
    )

    # Keep divB in an optional error slot if present.
    status["divb_error"] = _best_set_path(
        summary,
        (("global_quantities", "error_field_div_b", "value"), ("global_quantities", "error_field_div_b")),
        np.asarray(discharge_rows[:, 2], dtype=np.float64),
    )

    _set_ids_properties(summary, homogeneous_time=1, comment="Generated by nimrod2imas bin2imas.py")
    return status


def _set_mode_number(tm: Any, n: int) -> None:
    if not _set_if_has(tm, "n_phi", int(n)):
        _set_if_has(tm, "n_tor", int(n))


def write_mhd_linear_from_energy(mhd: Any, energy: Dict[str, Any]) -> Dict[str, int]:
    t = np.asarray(energy.get("time", np.zeros(0)), dtype=np.float64)
    modes = np.asarray(energy.get("modes", np.zeros(0)), dtype=np.int64)
    mag = np.asarray(energy.get("mag", np.zeros((0, 0))), dtype=np.float64)
    kin = np.asarray(energy.get("kin", np.zeros((0, 0))), dtype=np.float64)
    growth = np.asarray(energy.get("growth", np.zeros((0, 0))), dtype=np.float64)

    out = {"ntime": int(t.size), "nmodes": int(modes.size), "written_growth": 0, "written_energy": 0}
    if t.size == 0 or modes.size == 0:
        return out

    _set_if_has(mhd, "time", t)
    if not hasattr(mhd, "time_slice"):
        return out

    ts = mhd.time_slice
    if not _aos_resize(ts, t.size):
        return out

    for it in range(int(t.size)):
        tsi = ts[it]
        _set_if_has(tsi, "time", float(t[it]))
        if not hasattr(tsi, "toroidal_mode"):
            continue

        tm = tsi.toroidal_mode
        if not _aos_resize(tm, modes.size):
            continue

        for im in range(int(modes.size)):
            mode = tm[im]
            _set_mode_number(mode, int(modes[im]))

            gval = growth[it, im]
            if np.isfinite(gval) and _set_if_has(mode, "growthrate", float(gval)):
                out["written_growth"] += 1

            wrote_energy = False
            for attr, val in (
                ("energy_magnetic", mag[it, im]),
                ("magnetic_energy", mag[it, im]),
                ("energy_kinetic", kin[it, im]),
                ("kinetic_energy", kin[it, im]),
            ):
                if np.isfinite(val) and _set_if_has(mode, attr, float(val)):
                    wrote_energy = True
            if wrote_energy:
                out["written_energy"] += 1

    _set_ids_properties(mhd, homogeneous_time=1, comment="Generated by nimrod2imas bin2imas.py")
    return out


def write_disruption_from_kpraden(disruption: Any, kpraden_rows: np.ndarray) -> Dict[str, bool]:
    """
    Map kpraden channels to disruption power traces:
      qlosl -> total radiated power
      qloso -> ohmic power
    """
    status: Dict[str, bool] = {}
    if kpraden_rows.size == 0:
        return status

    t = np.asarray(kpraden_rows[:, 1], dtype=np.float64)
    qloso = np.asarray(kpraden_rows[:, 3], dtype=np.float64)
    qlosl = np.asarray(kpraden_rows[:, 6], dtype=np.float64)

    _set_if_has(disruption, "time", t)

    status["power_radiated_total"] = _best_set_path(
        disruption,
        (
            ("global_quantities", "power_radiated_total", "value"),
            ("global_quantities", "power_radiated_total"),
            ("global_quantities", "total_radiated_power", "value"),
            ("global_quantities", "total_radiated_power"),
            ("global_quantities", "power_radiated", "value"),
            ("global_quantities", "power_radiated"),
            ("thermal_quench", "power_radiated_total", "value"),
            ("thermal_quench", "power_radiated_total"),
            ("thermal_quench", "total_radiated_power", "value"),
            ("thermal_quench", "total_radiated_power"),
        ),
        qlosl,
    )

    status["power_ohmic"] = _best_set_path(
        disruption,
        (
            ("global_quantities", "power_ohmic", "value"),
            ("global_quantities", "power_ohmic"),
            ("global_quantities", "ohmic_power", "value"),
            ("global_quantities", "ohmic_power"),
            ("current_quench", "power_ohmic", "value"),
            ("current_quench", "power_ohmic"),
            ("current_quench", "ohmic_power", "value"),
            ("current_quench", "ohmic_power"),
        ),
        qloso,
    )

    _set_ids_properties(disruption, homogeneous_time=1, comment="Generated by nimrod2imas bin2imas.py (kpraden mapping)")
    return status


def _discover_files(input_dir: Path, pattern: str) -> List[Path]:
    return sorted(p.resolve() for p in input_dir.glob(pattern) if p.is_file())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Map NIMROD binary time-history files (energy/discharge) into IMAS summary + mhd_linear",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", default=".", help="Directory containing NIMROD binary files")
    p.add_argument("--energy-pattern", default="energy*.bin", help="Glob for energy files")
    p.add_argument("--discharge-pattern", default="discharge*.bin", help="Glob for discharge files")
    p.add_argument("--kprad-pattern", default="kpraden*.bin", help="Glob for kpraden files (contains qloso/qlosl)")
    p.add_argument("--energy-files", nargs="*", default=None, help="Explicit energy files (overrides --energy-pattern)")
    p.add_argument("--discharge-files", nargs="*", default=None, help="Explicit discharge files (overrides --discharge-pattern)")
    p.add_argument("--kprad-files", nargs="*", default=None, help="Explicit kpraden files (overrides --kprad-pattern)")

    p.add_argument("--dd", required=True, help="DB name (directory name)")
    p.add_argument("--pulse", type=int, required=True)
    p.add_argument("--run", type=int, required=True)
    p.add_argument("--dbpath", default=".", help="IMAS DB root")
    p.add_argument("--backend", default="hdf5", choices=("hdf5",), help="IMAS backend")
    p.add_argument("--dd-version", default="4.1.1", help="IMAS DD version")
    p.add_argument("--mode", default="a", help="DBEntry mode (r/a/w/x; r+/rw -> a)")

    p.add_argument("--summary-occ", type=int, default=0, help="summary IDS occurrence")
    p.add_argument("--mhd-linear-occ", type=int, default=1, help="mhd_linear IDS occurrence")
    p.add_argument("--disruption-occ", type=int, default=0, help="disruption IDS occurrence")
    p.add_argument("--no-summary", action="store_true", help="Do not write summary IDS")
    p.add_argument("--no-mhd-linear", action="store_true", help="Do not write mhd_linear IDS")
    p.add_argument("--no-disruption", action="store_true", help="Do not write disruption IDS")

    p.add_argument("--endian", default=">", choices=(">", "<"), help="Fortran record endianness")
    p.add_argument("--dry-run", action="store_true", help="Parse and report only; do not write IMAS")

    p.add_argument("--no-checksums", dest="record_checksums", action="store_false", help="Disable provenance checksums")
    p.set_defaults(record_checksums=True)
    p.add_argument("--checksum-algorithm", default="sha256", help="Checksum algorithm for provenance")

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    in_dir = Path(args.input_dir).expanduser().resolve()
    if not in_dir.is_dir():
        print(f"ERROR: input directory not found: {in_dir}", file=sys.stderr)
        return 2

    if args.energy_files:
        energy_files = [Path(x).expanduser().resolve() for x in args.energy_files]
    else:
        energy_files = _discover_files(in_dir, args.energy_pattern)

    if args.discharge_files:
        discharge_files = [Path(x).expanduser().resolve() for x in args.discharge_files]
    else:
        discharge_files = _discover_files(in_dir, args.discharge_pattern)

    if args.kprad_files:
        kprad_files = [Path(x).expanduser().resolve() for x in args.kprad_files]
    else:
        kprad_files = _discover_files(in_dir, args.kprad_pattern)

    missing = [str(p) for p in (energy_files + discharge_files + kprad_files) if not p.is_file()]
    if missing:
        print("ERROR: one or more files do not exist:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 2

    if not energy_files and not discharge_files and not kprad_files:
        print("ERROR: no energy/discharge/kpraden files found", file=sys.stderr)
        return 2

    energy_rows = load_energy(energy_files, endian=args.endian) if energy_files else np.zeros((0, 7), dtype=np.float64)
    discharge_rows = load_discharge(discharge_files, endian=args.endian) if discharge_files else np.zeros((0, 20), dtype=np.float64)
    kpraden_rows = load_kpraden(kprad_files, endian=args.endian) if kprad_files else np.zeros((0, 19), dtype=np.float64)
    energy_mats = build_energy_mode_matrices(energy_rows)

    print(f"Parsed energy files: {len(energy_files)} -> rows={energy_rows.shape[0]}, modes={energy_mats['modes'].size}, time_points={energy_mats['time'].size}")
    print(f"Parsed discharge files: {len(discharge_files)} -> rows={discharge_rows.shape[0]}")
    print(f"Parsed kpraden files: {len(kprad_files)} -> rows={kpraden_rows.shape[0]}")

    if args.dry_run:
        if energy_rows.size:
            print(f"Energy time range: {energy_rows[:,1].min():.9e} .. {energy_rows[:,1].max():.9e} s")
        if discharge_rows.size:
            print(f"Discharge time range: {discharge_rows[:,1].min():.9e} .. {discharge_rows[:,1].max():.9e} s")
        if kpraden_rows.size:
            print(f"kpraden time range: {kpraden_rows[:,1].min():.9e} .. {kpraden_rows[:,1].max():.9e} s")
        return 0

    mode = (args.mode or "a").strip().lower()
    if mode in ("r+", "rw"):
        mode = "a"
    if mode not in ("r", "a", "w", "x"):
        mode = "a"

    dd_dir = str(args.dd_version).strip()[:1] if str(args.dd_version).strip() else "4"
    ed = entry_dir(args.dbpath, args.dd, args.dd_version, args.pulse, args.run, dd_version_dir=dd_dir)
    ensure_entry_dir(ed)

    db, _uri, imas_mod = open_dbentry(args.backend, str(ed), mode=mode, dd_version=str(args.dd_version))
    factory = ids_factory(imas_mod, str(args.dd_version))

    summary_status: Dict[str, bool] = {}
    mhd_status: Dict[str, int] = {"ntime": 0, "nmodes": 0, "written_growth": 0, "written_energy": 0}
    disruption_status: Dict[str, bool] = {}

    if not args.no_summary and discharge_rows.size:
        summary = get_ids(db, factory, "summary", int(args.summary_occ))
        summary_status = write_summary_from_discharge(summary, discharge_rows, dd=str(args.dd), pulse=int(args.pulse))
        put_ids(db, summary, int(args.summary_occ))

    if not args.no_mhd_linear and energy_rows.size:
        mhd = get_ids(db, factory, "mhd_linear", int(args.mhd_linear_occ))
        mhd_status = write_mhd_linear_from_energy(mhd, energy_mats)
        put_ids(db, mhd, int(args.mhd_linear_occ))

    if not args.no_disruption and kpraden_rows.size:
        disruption = get_ids(db, factory, "disruption", int(args.disruption_occ))
        disruption_status = write_disruption_from_kpraden(disruption, kpraden_rows)
        put_ids(db, disruption, int(args.disruption_occ))

    pfiles: List[Path] = []
    pfiles.extend(energy_files)
    pfiles.extend(discharge_files)
    pfiles.extend(kprad_files)

    cmd = sanitize_cli_command(list(sys.argv), known_files=pfiles)
    try:
        update_workflow_and_dataset_fair(
            db,
            factory,
            component_name="nimrod2imas:bin2imas",
            component_description="Convert NIMROD binary time-history diagnostics (energy/discharge) to IMAS summary and mhd_linear",
            component_repository="https://github.com/PrincetonUniversity/nimrod2imas",
            component_version=str(__version__),
            exec_command=str(cmd),
            input_files=pfiles,
            record_checksums=bool(getattr(args, "record_checksums", True)),
            checksum_algorithm=str(getattr(args, "checksum_algorithm", "sha256")),
            workflow_occ=0,
            dataset_fair_occ=0,
            extra_kv={
                "dd": str(args.dd),
                "dd_version": str(args.dd_version),
                "pulse": str(args.pulse),
                "run": str(args.run),
                "summary_occ": str(args.summary_occ),
                "mhd_linear_occ": str(args.mhd_linear_occ),
                "disruption_occ": str(args.disruption_occ),
            },
        )
    except Exception as exc:
        print(f"[warn] provenance update failed: {exc}", file=sys.stderr)

    try:
        db.close()
    except Exception:
        pass

    if summary_status:
        mapped = ", ".join([f"{k}={'yes' if v else 'no'}" for k, v in summary_status.items()])
        print(f"summary mapping ({args.summary_occ}): {mapped}")
    if mhd_status:
        print(
            "mhd_linear mapping "
            f"({args.mhd_linear_occ}): ntime={mhd_status['ntime']}, nmodes={mhd_status['nmodes']}, "
            f"growth_points={mhd_status['written_growth']}, energy_points={mhd_status['written_energy']}"
        )
    if disruption_status:
        mapped = ", ".join([f"{k}={'yes' if v else 'no'}" for k, v in disruption_status.items()])
        print(f"disruption mapping ({args.disruption_occ}): {mapped}")

    print(f"Wrote IMAS entry: {ed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
