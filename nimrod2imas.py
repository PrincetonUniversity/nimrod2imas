#!/usr/bin/env python3
"""
nimrod2imas.py — shared helpers for input2imas/dump2imas/gamma2imas.

Exports expected by tools:
  - ensure_entry_dir
  - dd_version_dirname
  - entry_dir
  - open_dbentry
  - ids_factory
  - put_ids
  - value_to_string
  - namelist_file_to_xml
  - compute_file_checksum
  - sanitize_cli_command
  - update_workflow_and_dataset_fair

Workflow provenance is appended by writing workflow.h5 datasets directly (h5py),
to avoid imas-python AoS truncation bugs.
"""

from __future__ import annotations

import os
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

# Default unit scalings used by dump2imas/gamma2imas when converting NIMROD quantities to IMAS SI units.
# NIMROD dumps used in this workflow store:
#   - densities in units of 1e20 m^-3
#   - temperatures in keV
# IMAS expects:
#   - densities in m^-3
#   - temperatures in eV
DEFAULT_N_SCALE = 1e20  # (1e20 m^-3) -> m^-3
DEFAULT_T_SCALE = 1e3   # keV -> eV


# --------------------------- filesystem layout ---------------------------

def ensure_entry_dir(ed: str | Path) -> None:
    Path(ed).mkdir(parents=True, exist_ok=True)


def dd_version_dirname(dd_version: str | None) -> str:
    """
    Return the numeric major DD version directory name, e.g.
      '4.1.1' -> '4'
      '3.39.0' -> '3'
    """
    if not dd_version:
        return "4"
    s = str(dd_version).strip()
    m = re.match(r"^\s*(\d+)", s)
    return m.group(1) if m else "4"


def entry_dir(
    dbroot: str | Path,
    dd: str,
    dd_version: str,
    pulse: int,
    run: int,
    dd_version_dir: str = "4",
) -> Path:
    """Entry directory: <dbroot>/<dd>/<dd_version_dir>/<pulse>/<run>."""
    return Path(dbroot) / str(dd) / str(dd_version_dir) / str(int(pulse)) / str(int(run))


def build_uri(backend: str, entry_dir_path: str | Path) -> str:
    return f"imas:{backend}?path={str(entry_dir_path)}"


# --------------------------- IMAS open / factory / put ---------------------------

def open_dbentry(
    backend: str,
    entry_dir_path: str | Path,
    mode: str = "r",
    dd_version: Optional[str] = None,
):
    """Open an IMAS DBEntry from an entry directory. Returns (db, uri, imas_module)."""
    import imas  # type: ignore

    uri = build_uri(backend, entry_dir_path)

    old_ver = os.environ.get("IMAS_VERSION")
    if dd_version:
        os.environ["IMAS_VERSION"] = str(dd_version)

    db = None
    if dd_version:
        # Signature differs across imas-python versions.
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
        s = str(e).lower()
        if ("already open" not in s) and ("has no attribute" not in s):
            raise

    # Restore env best-effort
    try:
        if dd_version:
            if old_ver is None:
                os.environ.pop("IMAS_VERSION", None)
            else:
                os.environ["IMAS_VERSION"] = old_ver
    except Exception:
        pass

    return db, uri, imas


def ids_factory(imas_module: Any, dd_version: str):
    try:
        return imas_module.IDSFactory(str(dd_version))
    except Exception:
        return imas_module.IDSFactory()

def get_ids(db: Any, factory: Any, ids_name: str, occ: int):
    """Robust IDS getter across imas-python variants.

    Preferred behavior:
      - if the entry exists, return the existing IDS (return-by-value)
      - else return a new IDS instance (factory) so callers can populate it

    Some imas-python builds support both:
      - db.get(ids_name, occ) -> returns IDS
      - db.get(ids_obj, occ)  -> fills in-place
    We try both.
    """
    # Try return-by-value first (safer for AoS-containing IDSs on some builds)
    try:
        obj = db.get(str(ids_name), int(occ))
        if obj is not None:
            return obj
    except Exception:
        pass

    # Otherwise create a new IDS and attempt fill-in-place
    try:
        ids = factory.new(str(ids_name))
    except Exception:
        ids = factory.__getattr__(str(ids_name))()  # type: ignore[attr-defined]

    try:
        db.get(ids, int(occ))
    except Exception:
        # Entry does not exist; return empty IDS
        pass
    return ids



def _infer_homogeneous_time(ids: Any) -> int:
    """Infer a valid ids_properties.homogeneous_time value.

    IMAS DD expects:
      0: heterogeneous time
      1: homogeneous time
      2: independent of time (static)
    """
    try:
        t = getattr(ids, "time", None)
        if t is None:
            return 2
        # numpy arrays / lists
        try:
            if hasattr(t, "__len__") and len(t) > 0:
                return 1
        except Exception:
            pass
        # scalar time
        if isinstance(t, (int, float)) and np.isfinite(t):
            return 1
    except Exception:
        pass
    return 2


def put_ids(db: Any, ids: Any, occ: int) -> None:
    """Put IDS to DB, ensuring mandatory ids_properties fields are valid."""
    try:
        ht = _infer_homogeneous_time(ids)
        _ensure_ids_properties(ids, homogeneous_time=ht)
    except Exception:
        # best-effort; validation will catch if still invalid
        pass

    try:
        db.put(ids, int(occ))
    except Exception as e:
        # One more attempt if validation complains about homogeneous_time
        try:
            _ensure_ids_properties(ids, homogeneous_time=2)
            db.put(ids, int(occ))
            return
        except Exception:
            raise


def _get_or_create(db: Any, factory: Any, ids_name: str, occ: int):
    try:
        obj = db.get(ids_name, int(occ))
        if obj is not None:
            return obj
    except Exception:
        pass
    try:
        return factory.new(ids_name)
    except Exception:
        return factory.__getattr__(ids_name)()  # type: ignore[attr-defined]


def _ensure_ids_properties(ids_obj: Any, homogeneous_time: int = 2) -> None:
    try:
        ip = getattr(ids_obj, "ids_properties", None)
        if ip is None:
            return
        if hasattr(ip, "homogeneous_time"):
            try:
                ip.homogeneous_time = int(homogeneous_time)
            except Exception:
                pass
    except Exception:
        pass


def _append_ids_comment(ids_obj: Any, text: str) -> None:
    try:
        ip = getattr(ids_obj, "ids_properties", None)
        if ip is None:
            return
        prev = ""
        try:
            prev = str(getattr(ip, "comment", "") or "")
        except Exception:
            prev = ""
        new = (prev.rstrip() + "\n" if prev.strip() else "") + str(text).rstrip()
        try:
            ip.comment = new
        except Exception:
            pass
    except Exception:
        pass


# --------------------------- text / namelist utilities ---------------------------

def value_to_string(val: Any) -> str:
    """Convert Python value to Fortran-ish token string for XML."""
    if np is not None and isinstance(val, np.ndarray):
        val = val.tolist()

    if isinstance(val, (list, tuple)):
        return " ".join(value_to_string(v) for v in val)

    if isinstance(val, (bool,)) or (np is not None and isinstance(val, np.bool_)):  # type: ignore[attr-defined]
        return ".true." if bool(val) else ".false."

    if isinstance(val, str):
        s = val
        if s == "":
            return '""'
        if re.search(r"\s|,|\"|\\", s):
            s2 = s.replace("\\", "\\\\").replace('"', r"\"")
            return f'"{s2}"'
        return s

    return str(val)


def namelist_file_to_xml(root_tag: str, path: str | Path) -> str:
    """
    Convert a Fortran namelist file into a simple XML representation.

    Uses f90nml if present; otherwise embeds raw text.
    """
    import xml.etree.ElementTree as ET

    p = Path(path).expanduser()
    if not p.is_file():
        return ""

    root = ET.Element(str(root_tag))

    try:
        import f90nml  # type: ignore
        nml = f90nml.read(str(p))
        # nml is dict-like: group -> dict
        for grp, d in nml.items():
            g = ET.SubElement(root, "group")
            g.set("name", str(grp))
            if isinstance(d, dict):
                for k, v in d.items():
                    it = ET.SubElement(g, "item")
                    it.set("key", str(k))
                    it.text = value_to_string(v)
    except Exception:
        raw = ET.SubElement(root, "raw")
        raw.text = p.read_text(errors="replace")

    return ET.tostring(root, encoding="unicode")


# --------------------------- checksums / command sanitization ---------------------------

def compute_file_checksum(path: str | Path, algo: str = "sha256", chunk_bytes: int = 1024 * 1024) -> str:
    """Compute checksum. Returns '' if missing/unreadable."""
    try:
        p = Path(path).expanduser()
        if not p.is_file():
            return ""
        h = hashlib.new(str(algo or "sha256"))
        with p.open("rb") as f:
            while True:
                b = f.read(int(chunk_bytes))
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return ""


def sanitize_cli_command(argv: List[str], known_files: Optional[List[str | Path]] = None) -> str:
    """Sanitize argv to avoid absolute paths for known input files."""
    import shlex
    kset = set()
    if known_files:
        for p in known_files:
            try:
                kset.add(str(Path(p).expanduser().resolve()))
            except Exception:
                pass
            try:
                kset.add(str(p))
            except Exception:
                pass

    out: List[str] = []
    for tok in (argv or []):
        t = str(tok)
        rp = ""
        try:
            rp = str(Path(t).expanduser().resolve())
        except Exception:
            rp = ""
        if rp and (rp in kset):
            out.append(Path(t).name)
        else:
            out.append(t)
    return " ".join(shlex.quote(x) for x in out)


# --------------------------- workflow.h5 writer (append-only) ---------------------------

def _infer_entry_path_from_db(db: Any) -> Optional[Path]:
    """Infer entry directory from DBEntry URI imas:<backend>?path=<entry_dir>."""
    uri = ""
    for attr in ("uri", "_uri", "__uri", "URI"):
        try:
            if hasattr(db, attr):
                v = getattr(db, attr)
                if callable(v):
                    v = v()
                if v:
                    uri = str(v)
                    break
        except Exception:
            pass
    if not uri:
        try:
            uri = str(db.get_uri())
        except Exception:
            uri = ""

    if not uri or "?" not in uri:
        return None

    try:
        import urllib.parse
        qs = urllib.parse.parse_qs(uri.split("?", 1)[1], keep_blank_values=True)
        p = qs.get("path", [None])[0]
        return Path(p) if p else None
    except Exception:
        return None


def _merge_component_parameters(existing_xml: str, new_execution_el: Any) -> str:
    """Append <execution> to nimrod2imas_provenance XML."""
    import xml.etree.ElementTree as ET

    def make_root():
        return ET.Element("nimrod2imas_provenance")

    if existing_xml:
        try:
            root = ET.fromstring(existing_xml)
        except Exception:
            root = make_root()
            legacy = ET.SubElement(root, "legacy")
            legacy.text = str(existing_xml)
    else:
        root = make_root()

    execs = root.find("executions")
    if execs is None:
        execs = ET.SubElement(root, "executions")
    execs.append(new_execution_el)

    return ET.tostring(root, encoding="unicode")


def _ensure_vlen_str_dset(f: Any, name: str):
    import h5py
    if name in f:
        return f[name]
    dt = h5py.string_dtype(encoding="utf-8")
    return f.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dt)


def _h5_read_str(dset: Any, i: int) -> str:
    try:
        v = dset[i]
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="replace")
        return str(v)
    except Exception:
        return ""


def _workflow_update_h5(
    entry_path: Path,
    *,
    component_name: str,
    component_description: str,
    component_repository: str,
    component_version: str,
    new_execution_el: Any,
) -> bool:
    """Append/update workflow component datasets in workflow.h5 (append-only)."""
    wf_path = Path(entry_path) / "workflow.h5"
    try:
        import h5py  # noqa: F401
    except Exception:
        return False

    import h5py

    ensure_entry_dir(entry_path)
    if not wf_path.exists():
        with h5py.File(wf_path, "w"):
            pass

    with h5py.File(wf_path, "r+") as f:
        d_name = _ensure_vlen_str_dset(f, "time_loop&component[]&name")
        d_par  = _ensure_vlen_str_dset(f, "time_loop&component[]&parameters")
        d_desc = _ensure_vlen_str_dset(f, "time_loop&component[]&description")
        d_repo = _ensure_vlen_str_dset(f, "time_loop&component[]&repository")
        d_ver  = _ensure_vlen_str_dset(f, "time_loop&component[]&version")

        n = max(int(d_name.shape[0]), int(d_par.shape[0]), int(d_desc.shape[0]), int(d_repo.shape[0]), int(d_ver.shape[0]))
        for d in (d_name, d_par, d_desc, d_repo, d_ver):
            if int(d.shape[0]) < n:
                old = int(d.shape[0])
                d.resize((n,))
                for j in range(old, n):
                    d[j] = ""

        idx = -1
        blank = -1
        for i in range(n):
            nm = _h5_read_str(d_name, i).strip()
            pr = _h5_read_str(d_par, i).strip()
            if nm == component_name:
                idx = i
                break
            if blank < 0 and (not nm) and (not pr):
                blank = i

        if idx < 0:
            if blank >= 0:
                idx = blank
            else:
                new_n = n + 1
                for d in (d_name, d_par, d_desc, d_repo, d_ver):
                    d.resize((new_n,))
                    d[new_n - 1] = ""
                idx = new_n - 1

        prev = _h5_read_str(d_par, idx)
        d_par[idx] = _merge_component_parameters(prev, new_execution_el)

        d_name[idx] = component_name
        d_desc[idx] = str(component_description or "")
        d_repo[idx] = str(component_repository or "")
        d_ver[idx]  = str(component_version or "")

    return True


# --------------------------- public provenance entry point ---------------------------

def update_workflow_and_dataset_fair(
    db: Any,
    factory: Any,
    *,
    component_name: str,
    component_description: str,
    component_repository: str,
    component_version: str,
    exec_command: str,
    input_files: Optional[List[str | Path]] = None,
    record_checksums: bool = True,
    checksum_algorithm: str = "sha256",
    workflow_occ: int = 0,
    dataset_fair_occ: int = 0,
    extra_kv: Optional[Dict[str, str]] = None,
) -> None:
    """Append per-step provenance into BOTH workflow and dataset_fair."""
    import xml.etree.ElementTree as ET

    input_files = list(input_files or [])
    algo = str(checksum_algorithm or "sha256").strip() or "sha256"
    ts = datetime.now(timezone.utc).isoformat()

    exec_el = ET.Element("execution")
    exec_el.set("timestamp", ts)

    cmd_el = ET.SubElement(exec_el, "command")
    cmd_el.text = str(exec_command or "").strip()

    files_el = ET.SubElement(exec_el, "inputs")
    files_el.set("checksum_algorithm", algo)

    checksums: List[Tuple[str, str]] = []
    missing: List[str] = []
    for p in input_files:
        name = Path(p).name if hasattr(p, "__fspath__") else str(p)
        f_el = ET.SubElement(files_el, "file")
        f_el.set("name", name)
        if record_checksums:
            h = compute_file_checksum(p, algo=algo)
            if h:
                f_el.set("checksum", h)
                checksums.append((name, h))
            else:
                missing.append(name)

    if extra_kv:
        meta_el = ET.SubElement(exec_el, "metadata")
        for k, v in extra_kv.items():
            kv = ET.SubElement(meta_el, "kv")
            kv.set("key", str(k))
            kv.text = str(v)

    # workflow (append-only)
    entry_path = _infer_entry_path_from_db(db)
    if entry_path is not None:
        try:
            _workflow_update_h5(
                entry_path,
                component_name=component_name,
                component_description=component_description,
                component_repository=component_repository,
                component_version=component_version,
                new_execution_el=exec_el,
            )
        except Exception:
            pass

    # dataset_fair comment append
    try:
        df = _get_or_create(db, factory, "dataset_fair", int(dataset_fair_occ))
        _ensure_ids_properties(df, homogeneous_time=2)

        block: List[str] = []
        block.append(f"nimrod2imas provenance: {component_name}")
        block.append(f"timestamp_utc: {ts}")

        if extra_kv:
            kvs = "; ".join([f"{k}={extra_kv[k]}" for k in sorted(extra_kv.keys())])
            if kvs:
                block.append(f"metadata: {kvs}")

        cmd_line = str(exec_command or "").strip()
        if cmd_line:
            block.append(f"command: {cmd_line}")

        block.append(f"inputs: {len(input_files)} file(s)")

        if record_checksums:
            block.append(f"checksums ({algo}): {len(checksums)} computed; {len(missing)} missing/unreadable")
            cap = 30
            for name, h in checksums[:cap]:
                block.append(f"  {name}: {h}")
            if len(checksums) > cap:
                block.append(f"  ... (+{len(checksums)-cap} more)")
            if checksums:
                hman = hashlib.new(algo)
                for name, h in sorted(checksums, key=lambda x: x[0]):
                    hman.update((name + " " + h + "\n").encode("utf-8"))
                block.append(f"manifest_checksum ({algo}): {hman.hexdigest()}")
        else:
            block.append("checksums: disabled")

        _append_ids_comment(df, "\n".join(block))
        put_ids(db, df, int(dataset_fair_occ))
    except Exception:
        pass
