#!/usr/bin/env python3
"""nimrod2imas.py

Shared utilities for the NIMROD ⇄ IMAS toolchain.

This module centralizes:
  * IMAS entry directory layout under a filesystem DB root
  * Robust DBEntry open logic across imas-python variants
  * Robust IDS retrieval (fill-in-place vs return)
  * Consistent Fortran-like value serialization for XML
  * A lightweight fallback parser for Fortran namelist syntax

Directory layout
----------------
We follow the same layout as dump2imas.py (and update other tools to match):

  <dbpath>/<dd>/<dd_version_dir>/<pulse>/<run>/

Where <dd_version_dir> is usually the *major* DD version (e.g. "3") to
match existing workflows, but can be switched to "full" (e.g. "3.42.0").
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xml.etree.ElementTree as ET
import scipy.integrate as integrate

__version__ = "0.1.0"

# SciPy >= 1.11 removed cumtrapz; OMFIT still expects it
if not hasattr(integrate, "cumtrapz"):
    from scipy.integrate import cumulative_trapezoid
    integrate.cumtrapz = cumulative_trapezoid


# --------------------------- Path / DB helpers ---------------------------

def dd_version_dirname(dd_version: str, mode: str = "major") -> str:
    """Return the directory component to use for a DD version.

    mode:
      - "major": "3.42.0" -> "3"
      - "full" : "3.42.0" -> "3.42.0"
    """
    dv = str(dd_version).strip()
    if not dv:
        return dv
    if mode == "full":
        return dv
    # default: major
    return dv.split(".")[0]


def entry_dir(
    dbpath: str | Path,
    dd: str,
    dd_version: str,
    pulse: int,
    run: int,
    dd_version_dir: str = "major",
) -> Path:
    root = Path(dbpath).expanduser().resolve()
    return root / str(dd) / dd_version_dirname(str(dd_version), dd_version_dir) / str(int(pulse)) / str(int(run))


def ensure_entry_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_uri(backend: str, entry_dir_path: str | Path) -> str:
    return f"imas:{backend}?path={str(entry_dir_path)}"


def open_dbentry(
    backend: str,
    entry_dir_path: str | Path,
    mode: str = "r",
    dd_version: Optional[str] = None,
):
    """Open an IMAS DBEntry from an entry directory.

    Returns (db, uri, imas_module).
    """
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

    # Some versions open in __init__, some require open().
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
    """Robust IDS getter across IMAS python variants."""
    try:
        ids = factory.new(ids_name)
    except Exception:
        ids = factory.__getattr__(ids_name)()  # type: ignore[attr-defined]

    try:
        db.get(ids, int(occ))
        return ids
    except Exception:
        pass

    obj = db.get(ids_name, int(occ))
    return ids if obj is None else obj


def put_ids(db: Any, ids: Any, occ: int) -> None:
    """Robust IDS writer across IMAS python variants."""
    # 1) Preferred: ids.put(db_entry=db, occurrence=occ)
    try:
        ids.put(db_entry=db, occurrence=int(occ))
        return
    except Exception:
        pass
    # 2) Some versions accept ids.put(db, occ)
    try:
        ids.put(db, int(occ))
        return
    except Exception:
        pass
    # 3) DBEntry.put(ids, occ)
    try:
        db.put(ids, int(occ))
        return
    except Exception as e:
        raise RuntimeError(f"Failed to write IDS occurrence={occ}: {e}")


@dataclass
class IMASContext:
    """Filesystem-backed IMAS context used consistently across NIMROD tools."""

    backend: str
    dbpath: str | Path
    dd: str
    dd_version: str
    pulse: int
    run: int
    dd_version_dir: str = "major"

    def entry_dir(self) -> Path:
        return entry_dir(self.dbpath, self.dd, self.dd_version, self.pulse, self.run, dd_version_dir=self.dd_version_dir)

    def uri(self) -> str:
        return build_uri(self.backend, self.entry_dir())

    def open(self, mode: str = "r"):
        p = self.entry_dir()
        if mode in ("a", "w", "x"):
            ensure_entry_dir(p)
        db, uri, imas_mod = open_dbentry(self.backend, p, mode=mode, dd_version=self.dd_version)
        factory = ids_factory(imas_mod, self.dd_version)
        return db, uri, imas_mod, factory


# --------------------------- Namelist helpers ---------------------------

def value_to_string(val: Any) -> str:
    """Convert a Python value to a Fortran-like token string for XML.

    Important: strings that contain whitespace are quoted so they round-trip
    through restore logic without being split into arrays.
    """
    if isinstance(val, np.ndarray):
        val = val.tolist()

    # Arrays/lists: emit space-separated token stream
    if isinstance(val, (list, tuple)):
        return " ".join(value_to_string(v) for v in val)

    # Booleans
    if isinstance(val, (bool, np.bool_)):
        return ".true." if bool(val) else ".false."

    # Strings: quote when needed (whitespace/special chars), using double quotes + backslash escapes
    if isinstance(val, str):
        s = val
        if s == "":
            return '""'
        if re.search(r"\s|,|\"|\\", s):
            s2 = s.replace("\\", "\\\\").replace('"', '\"')
            return f'"{s2}"'
        return s

    return str(val)
def parse_fortran_namelist_text(text: str) -> Dict[str, Dict[str, Any]]:
    """Very small namelist parser.

    This is *not* a full Fortran parser; it's meant as a best-effort fallback
    when f90nml is unavailable. It handles typical NIMROD-style namelists:

      &group
        var = 1,
        flag = .true.
      /

    Arrays like a(1)=..., repeated assignments, and expressions are preserved
    as raw strings.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    if not text:
        return groups

    # Strip comments (! ...)
    lines = []
    for ln in text.splitlines():
        # Keep string literals simple: remove ! only if not in quotes (best-effort)
        if "!" in ln and (ln.count("'") % 2 == 0) and (ln.count('"') % 2 == 0):
            ln = ln.split("!", 1)[0]
        lines.append(ln)
    text2 = "\n".join(lines)

    # Split into group blocks.
    pos = 0
    while True:
        m = _NML_GROUP_RE.search(text2, pos)
        if not m:
            break
        grp = m.group("grp").strip()
        start = m.end()
        # find group terminator: / at line start or &end (rare)
        end_m = re.search(r"(?im)^\s*/\s*$", text2[start:])
        if end_m:
            end = start + end_m.start()
            pos = start + end_m.end()
        else:
            # No '/', consume to end
            end = len(text2)
            pos = len(text2)
        body = text2[start:end]

        # Tokenize assignments in body. We do a conservative split on commas/newlines.
        assigns = re.split(r"[,\n]", body)
        d: Dict[str, Any] = {}
        for a in assigns:
            if "=" not in a:
                continue
            k, v = a.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                continue
            # Normalize logicals
            vl = v.lower().strip()
            if vl in (".true.", "true", "t"):
                d[k] = True
            elif vl in (".false.", "false", "f"):
                d[k] = False
            else:
                # Try numeric scalar
                try:
                    if re.match(r"^[+-]?(\d+\.\d*|\d*\.\d+)([edED][+-]?\d+)?$", v) or re.match(r"^[+-]?\d+([edED][+-]?\d+)?$", v):
                        d[k] = float(v.replace("D", "E").replace("d", "e")) if ("." in v or "e" in vl or "d" in vl) else int(v)
                    else:
                        d[k] = v
                except Exception:
                    d[k] = v
        groups[grp] = d

    return groups


def namelist_file_to_xml(root_tag: str, file_path: Optional[str]) -> str:
    """Convert a Fortran namelist file into the XML format used by input2imas."""
    if not file_path:
        return ""
    p = Path(os.path.expanduser(file_path))
    if not p.is_file():
        return ""

    # Prefer f90nml when available.
    try:
        import f90nml  # type: ignore
        nml = f90nml.read(str(p))
        root = ET.Element(root_tag)
        nml_el = ET.SubElement(root, root_tag.replace("_inputs", "_in"), filename=p.name)
        for grp_name, grp in nml.items():
            g_el = ET.SubElement(nml_el, "group", name=str(grp_name))
            for var_name, val in grp.items():
                v_el = ET.SubElement(g_el, "var", name=str(var_name))
                v_el.text = value_to_string(val)
        return ET.tostring(root, encoding="unicode")
    except Exception:
        pass

    # Fallback: small parser
    try:
        raw = p.read_text(errors="ignore")
    except Exception:
        return ""
    groups = parse_fortran_namelist_text(raw)
    if not groups:
        return ""

    root = ET.Element(root_tag)
    nml_el = ET.SubElement(root, root_tag.replace("_inputs", "_in"), filename=p.name)
    for grp_name, grp in groups.items():
        g_el = ET.SubElement(nml_el, "group", name=str(grp_name))
        for var_name, val in grp.items():
            v_el = ET.SubElement(g_el, "var", name=str(var_name))
            v_el.text = value_to_string(val)
    return ET.tostring(root, encoding="unicode")
