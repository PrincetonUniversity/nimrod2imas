#!/usr/bin/env python3
"""
gamma2imas.py

Standalone converter to populate IMAS mhd_linear mode scalars from NIMROD diagnostics:

  mhd_linear.time_slice(itime)/toroidal_mode(i1)/growthrate   [Hz]  (IMAS DD units)
  mhd_linear.time_slice(itime)/toroidal_mode(i1)/frequency    [Hz]  (optional)

Growth rates are computed from energy diagnostics:
  - energy.bin : energies (Emag, Ekin) -> gamma = (1/2) d ln(E) / dt
  - logen.bin  : log10(Emag), log10(Ekin) -> same gamma with ln(E)=log10(E)/log10(e)

Mode frequency requires complex time history (optional second positional input):
  - history/nimhist bin (e.g., nimhist01.bin): uses complex B components (Br,Bz,Bphi)
    and computes omega via Dalton's symmetric formula, then converts to Hz:
        omega = 2/dt * Im( (f1 - f0) / (f1 + f0) )    [rad/s]
        frequency = omega / (2*pi)                   [Hz]
    A single frequency per mode is obtained by averaging over the last N samples
    (ignoring zeros / invalid samples) and using the B component with the
    largest mean amplitude over the tail window.

CLI (as requested):
  gamma2imas.py --dd mast --dd-version 4.1.1 --pulse 45272 --run 1 --occ 1 energy.bin
  gamma2imas.py --dd mast --dd-version 4.1.1 --pulse 45272 --run 1 --occ 1 logen.bin

Optional probe file to compute frequency:
  gamma2imas.py ... energy.bin nimhist01.bin
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

__version__ = "0.3.2"

# ------------------------- Fortran sequential record reader -------------------------

def _read_fortran_records_f32(path: str | Path, endian: str = ">") -> Iterable[np.ndarray]:
    """
    Yield payload arrays of float32 from Fortran sequential unformatted records.

    Record layout: [int32 nbytes][payload][int32 nbytes], repeated.
    Records with nbytes==0 are ignored (some NIMROD bins end with two zero markers).
    """
    import struct

    p = Path(path)
    with p.open("rb") as f:
        while True:
            head = f.read(4)
            if not head:
                break
            if len(head) != 4:
                raise IOError(f"{p}: truncated record header ({len(head)} bytes)")
            nbytes = struct.unpack(endian + "i", head)[0]
            if nbytes == 0:
                continue
            if nbytes < 0:
                raise IOError(f"{p}: negative record length {nbytes} at offset {f.tell()-4}")
            payload = f.read(nbytes)
            tail = f.read(4)
            if len(payload) != nbytes or len(tail) != 4:
                raise IOError(f"{p}: truncated record payload/tail at offset {f.tell()}")
            nbytes2 = struct.unpack(endian + "i", tail)[0]
            if nbytes2 != nbytes:
                raise IOError(f"{p}: record length mismatch {nbytes} != {nbytes2}")
            if nbytes % 4 != 0:
                raise IOError(f"{p}: record length {nbytes} not multiple of 4")
            nf = nbytes // 4
            arr = np.frombuffer(payload, dtype=np.dtype(endian + "f4"), count=nf).astype(np.float64)
            yield arr


# ------------------------- energy.bin / logen.bin -------------------------

def read_energy_like(path: str | Path, endian: str = ">") -> np.ndarray:
    """
    Read energy.bin / logen.bin into ndarray rows with columns:
      istep, t, imode, keff, Emag, Ekin
    """
    rows: List[List[float]] = []
    for rec in _read_fortran_records_f32(path, endian=endian):
        if rec.size == 0:
            continue
        if rec.size != 6:
            raise ValueError(f"{path}: expected 6 float32 payload values per record, got {rec.size}")
        rows.append(rec.tolist())
    if not rows:
        raise ValueError(f"{path}: no records read")
    return np.asarray(rows, dtype=np.float64)


def _safe_gamma_from_energy(E: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    gamma_i = (ln(E_i) - ln(E_{i-1})) / (2 * dt_i)
    Invalid intervals return gamma=0 (growth.py convention: 0 means ignore).
    """
    E0 = E[:-1]
    E1 = E[1:]
    dt = np.diff(t)
    gamma = np.zeros_like(dt, dtype=np.float64)

    good = (dt > 0) & np.isfinite(dt) & (E0 > 0) & (E1 > 0) & np.isfinite(E0) & np.isfinite(E1)
    if np.any(good):
        gamma[good] = (np.log(E1[good]) - np.log(E0[good])) / (2.0 * dt[good])
    return gamma


def _safe_gamma_from_log10E(log10E: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    ln(E) = log10(E) / log10(e)
    gamma = (ln(E_i) - ln(E_{i-1})) / (2*dt)
    """
    dt = np.diff(t)
    gamma = np.zeros_like(dt, dtype=np.float64)
    good = (dt > 0) & np.isfinite(dt) & np.isfinite(log10E[:-1]) & np.isfinite(log10E[1:])
    if np.any(good):
        dlnE = (log10E[1:][good] - log10E[:-1][good]) / math.log10(math.e)
        gamma[good] = dlnE / (2.0 * dt[good])
    return gamma


def compute_growth_rates(
    rows: np.ndarray,
    file_kind: str,
    nsteps_use: int = 50,
    component: str = "total",
) -> Dict[float, Dict[str, float]]:
    """
    Compute per-keff growth rate statistics from energy/logen rows.
    Returns dict[keff] = {"gamma_mean","gamma_std","npts"}.
    """
    keff_all = rows[:, 3]
    result: Dict[float, Dict[str, float]] = {}

    for keff in sorted(set(float(k) for k in np.unique(keff_all))):
        m = (keff_all == keff)
        sub = rows[m]
        if sub.shape[0] < 2:
            continue

        # sort by time (and istep to break ties)
        order = np.lexsort((sub[:, 0], sub[:, 1]))
        sub = sub[order]

        # tail window in time points
        if nsteps_use is not None and int(nsteps_use) > 0 and sub.shape[0] > int(nsteps_use):
            sub = sub[-int(nsteps_use):, :]

        tt = sub[:, 1]
        Emag = sub[:, 4]
        Ekin = sub[:, 5]

        if component == "magnetic":
            gamma = _safe_gamma_from_log10E(Emag, tt) if file_kind == "logen" else _safe_gamma_from_energy(Emag, tt)
        elif component == "kinetic":
            gamma = _safe_gamma_from_log10E(Ekin, tt) if file_kind == "logen" else _safe_gamma_from_energy(Ekin, tt)
        else:
            if file_kind == "logen":
                Etot = np.power(10.0, Emag) + np.power(10.0, Ekin)
                gamma = _safe_gamma_from_energy(Etot, tt)
            else:
                Etot = Emag + Ekin
                gamma = _safe_gamma_from_energy(Etot, tt)

        w = (gamma != 0.0) & np.isfinite(gamma)
        npts = float(np.sum(w))
        if npts > 0:
            gm = float(np.average(gamma[w]))
            gs = float(np.sqrt(np.sum((gamma[w] - gm) ** 2) / npts))
        else:
            gm, gs = 0.0, 0.0

        result[keff] = {"keff": float(keff), "gamma_mean": gm, "gamma_std": gs, "npts": npts}

    return result


def print_growth_summary(stats: Dict[float, Dict[str, float]], component: str) -> None:
    print("Growth rates given in units of s^-1")
    print("Var".ljust(13) + " Growth (Re gamma)".ljust(29) + "Npts")
    label = {"total": "E total", "magnetic": "E magnetic", "kinetic": "E kinetic"}[component]
    for keff in sorted(stats.keys()):
        s = stats[keff]
        print("keff =".rjust(8), f"{keff:g}")
        gm = f"{s['gamma_mean']:10.4e}"
        gs = f"{s['gamma_std']:10.4e}"
        npts = f"{int(s['npts']):4d}"
        print(label.ljust(10), ":", gm.rjust(11), "+/-", gs.rjust(10), "(", npts, ")")


# ------------------------- history/nimhist frequency -------------------------

def _infer_nimhist_layout(rec: np.ndarray) -> bool:
    """
    Minimal check that record contains at least:
      istep, t, imode, k, ReBr, ReBz, ReBphi, ImBr, ImBz, ImBphi
    """
    return rec.size >= 10


def compute_frequency_from_history(
    hist_path: str | Path,
    nsteps_use: int = 50,
    endian: str = ">",
) -> Dict[float, Dict[str, float]]:
    """
    Compute per-keff frequency statistics from a NIMROD history/nimhist binary.

    Uses complex B components and Dalton's symmetric formula:
        omega = 2/dt * Im( (f1 - f0)/(f1 + f0) )   [rad/s]
        freq  = omega/(2*pi)                       [Hz]

    Returns dict[keff] = {"freq_mean","freq_std","npts","component"} where
    component is one of {"Br","Bz","Bphi"} selected by largest mean |B|.
    """
    hist_path = Path(hist_path)
    # buffers keyed by keff
    bufs: Dict[float, Dict[str, deque]] = {}

    for rec in _read_fortran_records_f32(hist_path, endian=endian):
        if rec.size == 0:
            continue
        if not _infer_nimhist_layout(rec):
            # Skip unknown record types
            continue

        t = float(rec[1])
        keff = float(rec[3])

        # Complex B components (assumed order as in nimpy.read_bin nimhist_vars)
        br = complex(rec[4], rec[7])
        bz = complex(rec[5], rec[8])
        bphi = complex(rec[6], rec[9])

        if keff not in bufs:
            bufs[keff] = {
                "t": deque(maxlen=int(nsteps_use)),
                "br": deque(maxlen=int(nsteps_use)),
                "bz": deque(maxlen=int(nsteps_use)),
                "bphi": deque(maxlen=int(nsteps_use)),
            }
        b = bufs[keff]
        b["t"].append(t)
        b["br"].append(br)
        b["bz"].append(bz)
        b["bphi"].append(bphi)

    out: Dict[float, Dict[str, float]] = {}
    twopi = 2.0 * math.pi

    for keff, b in bufs.items():
        if len(b["t"]) < 2:
            continue
        t = np.asarray(b["t"], dtype=np.float64)
        br = np.asarray(b["br"], dtype=np.complex128)
        bz = np.asarray(b["bz"], dtype=np.complex128)
        bphi = np.asarray(b["bphi"], dtype=np.complex128)

        # pick the B component with the largest mean amplitude
        amps = {
            "Br": float(np.mean(np.abs(br))),
            "Bz": float(np.mean(np.abs(bz))),
            "Bphi": float(np.mean(np.abs(bphi))),
        }
        comp = max(amps, key=lambda k: amps[k])
        f = {"Br": br, "Bz": bz, "Bphi": bphi}[comp]

        dt = np.diff(t)
        freq = np.zeros_like(dt, dtype=np.float64)

        f0 = f[:-1]
        f1 = f[1:]
        denom = (f1 + f0)

        good = (dt > 0) & np.isfinite(dt) & (np.abs(denom) != 0) & np.isfinite(f0.real) & np.isfinite(f0.imag) & np.isfinite(f1.real) & np.isfinite(f1.imag)
        if np.any(good):
            ratio = (f1[good] - f0[good]) / denom[good]
            omega = (2.0 / dt[good]) * np.imag(ratio)  # rad/s
            freq[good] = omega / twopi                 # Hz

        w = (freq != 0.0) & np.isfinite(freq)
        npts = float(np.sum(w))
        if npts > 0:
            fm = float(np.average(freq[w]))
            fs = float(np.sqrt(np.sum((freq[w] - fm) ** 2) / npts))
        else:
            fm, fs = 0.0, 0.0

        out[float(keff)] = {"keff": float(keff), "freq_mean": fm, "freq_std": fs, "npts": npts, "component": comp}

    return out


def print_frequency_summary(freq_stats: Dict[float, Dict[str, float]]) -> None:
    print("\nFrequencies given in units of Hz (from history/nimhist)")
    print("Var".ljust(13) + " Frequency".ljust(29) + "Npts")
    for keff in sorted(freq_stats.keys()):
        s = freq_stats[keff]
        print("keff =".rjust(8), f"{keff:g}")
        fm = f"{s['freq_mean']:10.4e}"
        fs = f"{s['freq_std']:10.4e}"
        npts = f"{int(s['npts']):4d}"
        label = f"{s.get('component','B')}".ljust(10)
        print(label, ":", fm.rjust(11), "+/-", fs.rjust(10), "(", npts, ")")


# ------------------------- IMAS write -------------------------

def _normalize_mode(mode: str) -> str:
    mode = (mode or "a").strip()
    if mode in ("r+", "rw"):
        return "a"
    if mode not in ("r", "a", "w", "x"):
        return "a"
    return mode


def _aos_len(aos) -> int:
    try:
        return len(aos)
    except Exception:
        try:
            return int(aos.size())
        except Exception:
            return 0


def _ensure_resize(aos, n: int, what: str) -> None:
    cur = _aos_len(aos)
    if cur == n:
        return
    try:
        aos.resize(int(n))
    except Exception as e:
        raise RuntimeError(f"Failed to resize {what} from {cur} to {n}: {e}")


def write_mhd_linear_mode_scalars(
    dd: str,
    dd_version: str,
    dd_version_dir: str,
    pulse: int,
    run: int,
    occ: int,
    growth_stats: Dict[float, Dict[str, float]],
    freq_stats: Optional[Dict[float, Dict[str, float]]] = None,
    dbpath: str = ".",
    backend: str = "hdf5",
    mode: str = "a",
    provenance_files: Optional[List[str | Path]] = None,
    record_checksums: bool = True,
    checksum_algorithm: str = "sha256",
    exec_command: Optional[str] = None,
) -> None:
    """
    Write growthrate (and optionally frequency) into mhd_linear occurrence `occ`.

    Updates ALL existing time slices; creates one time_slice if none exist.
    """
    try:
        from nimrod2imas import entry_dir as entry_dir_common
        from nimrod2imas import ensure_entry_dir
        from nimrod2imas import open_dbentry as open_dbentry_common
        from nimrod2imas import ids_factory as ids_factory_common
        from nimrod2imas import get_ids, put_ids
        from nimrod2imas import update_workflow_and_dataset_fair
        from nimrod2imas import sanitize_cli_command
    except Exception as e:
        raise RuntimeError(f"Failed to import nimrod2imas helpers (needed for consistent IMAS I/O): {e}")

    ed = entry_dir_common(dbpath, dd, dd_version, pulse, run, dd_version_dir=dd_version[0])
    ensure_entry_dir(ed)

    mode_n = _normalize_mode(mode)
    db, _uri, imas_mod = open_dbentry_common(backend, str(ed), mode=mode_n, dd_version=str(dd_version))
    factory = ids_factory_common(imas_mod, str(dd_version))

    mhd = get_ids(db, factory, "mhd_linear", int(occ))

    if not hasattr(mhd, "time_slice"):
        raise RuntimeError("mhd_linear IDS has no time_slice (unexpected schema)")

    if _aos_len(mhd.time_slice) == 0:
        _ensure_resize(mhd.time_slice, 1, "mhd_linear.time_slice")
        try:
            mhd.time_slice[0].time = 0.0
        except Exception:
            pass

    # union of keys for allocating missing toroidal_mode entries
    all_keff = sorted(set(growth_stats.keys()) | (set(freq_stats.keys()) if freq_stats else set()))

    for its in range(_aos_len(mhd.time_slice)):
        ts = mhd.time_slice[its]
        if not hasattr(ts, "toroidal_mode"):
            continue

        if _aos_len(ts.toroidal_mode) == 0 and len(all_keff) > 0:
            _ensure_resize(ts.toroidal_mode, len(all_keff), f"mhd_linear.time_slice[{its}].toroidal_mode")
            for i, keff in enumerate(all_keff):
                tm = ts.toroidal_mode[i]
                nval = int(round(float(keff)))
                for attr in ("n_phi", "n_tor"):
                    if hasattr(tm, attr):
                        try:
                            setattr(tm, attr, nval)
                        except Exception:
                            pass

        # index by n_phi/n_tor
        idx: Dict[int, int] = {}
        for i in range(_aos_len(ts.toroidal_mode)):
            tm = ts.toroidal_mode[i]
            nval = None
            for attr in ("n_phi", "n_tor"):
                if hasattr(tm, attr):
                    try:
                        nval = int(getattr(tm, attr))
                        break
                    except Exception:
                        pass
            if nval is not None:
                idx[int(nval)] = i

        for keff in all_keff:
            nkey = int(round(float(keff)))
            if nkey in idx:
                tm = ts.toroidal_mode[idx[nkey]]
            else:
                cur = _aos_len(ts.toroidal_mode)
                _ensure_resize(ts.toroidal_mode, cur + 1, f"mhd_linear.time_slice[{its}].toroidal_mode")
                tm = ts.toroidal_mode[cur]
                for attr in ("n_phi", "n_tor"):
                    if hasattr(tm, attr):
                        try:
                            setattr(tm, attr, nkey)
                        except Exception:
                            pass
                idx[nkey] = cur

            if keff in growth_stats:
                gr = float(growth_stats[keff]["gamma_mean"])
                if not hasattr(tm, "growthrate"):
                    raise RuntimeError("toroidal_mode has no growthrate field in this IMAS DD/schema")
                tm.growthrate = gr  # Hz in DD units, but gamma is s^-1 (same dimension)

            if freq_stats and keff in freq_stats:
                fr = float(freq_stats[keff]["freq_mean"])
                if not hasattr(tm, "frequency"):
                    raise RuntimeError("toroidal_mode has no frequency field in this IMAS DD/schema")
                tm.frequency = fr  # Hz

    put_ids(db, mhd, int(occ))
    # Append provenance (workflow + dataset_fair). Only basenames are stored.
    try:
        pfiles = list(provenance_files or [])
        cmd = str(exec_command or "")
        if not cmd:
            cmd = sanitize_cli_command(list(__import__("sys").argv), known_files=pfiles)
        update_workflow_and_dataset_fair(
            db, factory,
            component_name="nimrod2imas:gamma2imas",
            component_description="Compute growth rate/frequency from NIMROD diagnostics and store into IMAS mhd_linear",
            component_repository="https://github.com/PrincetonUniversity/nimrod2imas",
            component_version=str(__version__ if "__version__" in globals() else ""),
            exec_command=cmd,
            input_files=pfiles,
            record_checksums=bool(record_checksums),
            checksum_algorithm=str(checksum_algorithm or "sha256"),
            workflow_occ=0,
            dataset_fair_occ=0,
            extra_kv={
                "dd": str(dd),
                "dd_version": str(dd_version),
                "pulse": str(pulse),
                "run": str(run),
                "occ": str(occ),
            },
        )
    except Exception as exc:
        print(f"[warn] provenance update failed: {exc}", file=sys.stderr)

    try:
        db.close()
    except Exception:
        pass


# ------------------------- CLI -------------------------

def _infer_file_kind(path: str | Path, override: Optional[str] = None) -> str:
    if override:
        o = override.strip().lower()
        if o in ("energy", "en"):
            return "energy"
        if o in ("logen", "log"):
            return "logen"
        raise ValueError(f"Unknown --file-kind {override!r} (use energy|logen)")
    b = Path(path).name.lower()
    return "logen" if "logen" in b else "energy"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute NIMROD growth rates from energy.bin/logen.bin and store into IMAS mhd_linear.toroidal_mode[].growthrate; optionally store frequency from history/nimhist.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # dump2imas-compatible core options (names preserved)
    p.add_argument("--dd", required=True, help="DB name (directory name), e.g. mast")
    p.add_argument("--pulse", type=int, required=True)
    p.add_argument("--run", type=int, required=True)
    p.add_argument("--backend", default="hdf5", choices=("hdf5",), help="IMAS backend")
    p.add_argument("--dbpath", default=".", help="DB root path")
    p.add_argument("--dd-version", default="4.1.1", help="IMAS data dictionary version")
    p.add_argument("--mode", default="a", help="DBEntry open mode: r/a/w/x (r+/rw mapped to a)")

    # user-required naming
    p.add_argument("--occ", type=int, default=1, help="Occurrence for mhd_linear IDS")

    # growth-rate knobs
    p.add_argument("-n", "--nsteps", type=int, default=50,
                   help="Tail window (time points) used for statistics (growth and frequency)")
    p.add_argument("--component", choices=("total", "magnetic", "kinetic"), default="total",
                   help="Energy component used for growthrate scalar")
    p.add_argument("--endian", choices=(">", "<"), default=">",
                   help="Endianness for Fortran record markers and float payloads")
    p.add_argument("--file-kind", choices=("energy", "logen"), default=None,
                   help="Override file kind auto-detection (default: infer from filename containing 'logen')")

    # provenance controls
    p.add_argument("--no-checksums", dest="record_checksums", action="store_false",
                   help="Disable provenance file checksums in workflow/dataset_fair IDSs.")
    p.set_defaults(record_checksums=True)
    p.add_argument("--checksum-algorithm", default="sha256",
                   help="Hash algorithm for provenance checksums (sha256, sha1, md5, ...).")

    # positional files (as requested)
    p.add_argument("binfile", help="Input NIMROD energy diagnostic file (energy.bin or logen.bin)")
    p.add_argument("history", nargs="?", default=None,
                   help="Optional NIMROD history/nimhist binary (e.g., history.bin or nimhist01.bin) to compute frequency")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    binfile = Path(args.binfile).expanduser().resolve()
    if not binfile.exists():
        print(f"ERROR: file not found: {binfile}", file=sys.stderr)
        return 2

    file_kind = _infer_file_kind(binfile, args.file_kind)

    rows = read_energy_like(binfile, endian=args.endian)
    growth_stats = compute_growth_rates(rows, file_kind=file_kind, nsteps_use=args.nsteps, component=args.component)

    print_growth_summary(growth_stats, component=args.component)

    freq_stats = None
    if args.history is not None:
        histfile = Path(args.history).expanduser().resolve()
        if not histfile.exists():
            print(f"ERROR: history file not found: {histfile}", file=sys.stderr)
            return 2
        freq_stats = compute_frequency_from_history(histfile, nsteps_use=args.nsteps, endian=args.endian)
        print_frequency_summary(freq_stats)

    # Provenance inputs for workflow/dataset_fair (no absolute paths stored)
    # Absolute paths for checksum robustness
    run_cwd = Path.cwd()

    binfile_abs = Path(binfile).expanduser()
    if not binfile_abs.is_absolute():
        binfile_abs = (run_cwd / binfile_abs).resolve()

    pfiles = [binfile_abs]

    if args.history is not None:
        hist_abs = Path(histfile).expanduser()
        if not hist_abs.is_absolute():
            hist_abs = (run_cwd / hist_abs).resolve()
        pfiles.append(hist_abs)

    # Include companion file if present (energy.bin <-> logen.bin)
    try:
        bname = binfile_abs.name.lower()
        companion = binfile_abs.parent / ("logen.bin" if bname.startswith("energy") else "energy.bin")
        if companion.is_file():
            pfiles.append(companion.resolve())
    except Exception:
        pass

    try:
        from nimrod2imas import sanitize_cli_command as _sanitize_cli_command
        cmd = _sanitize_cli_command(list(sys.argv), known_files=pfiles)
    except Exception:
        cmd = ""

    write_mhd_linear_mode_scalars(
        dd=args.dd,
        dd_version=args.dd_version,
        dd_version_dir=args.dd_version[0],
        pulse=args.pulse,
        run=args.run,
        occ=args.occ,
        growth_stats=growth_stats,
        freq_stats=freq_stats,
        dbpath=args.dbpath,
        backend=args.backend,
        mode=args.mode,
        provenance_files=pfiles,
        record_checksums=bool(getattr(args, "record_checksums", True)),
        checksum_algorithm=str(getattr(args, "checksum_algorithm", "sha256")),
        exec_command=cmd,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
