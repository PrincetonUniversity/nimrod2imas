#!/usr/bin/env python3
"""
nimrodInputRestore.py  (patched v4)

Improvements over v3:
  * Restores with customized filenames: <orig>_from_imas.<ext>
      e.g. nimrod_from_imas.in, nimeq_from_imas.in, ...
    If multiple <namelist> blocks exist in XML, writes one file per namelist.
  * Fixes double-quoting artifacts:
      init_type = '"shear alf   mult"'  -> init_type = "shear alf   mult"
      eta_model = '"braginskii n=0"'    -> eta_model = "braginskii n=0"
  * Better heuristic to distinguish scalar strings-with-spaces vs arrays-of-strings:
      - comma-separated => list
      - whitespace-separated tokens that are all numeric/bool => list
      - whitespace-separated tokens that are all identical => list (string array / repetition)
      - otherwise => scalar string (preserve internal spacing)
  * For repeated string arrays, emits Fortran repetition syntax:
      ds_function = 4*'ds_diff'
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, List, Tuple, Optional
from nimrod2imas import __version__, VERSION

import numpy as np

try:
    from imas import IDSFactory
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"ERROR: cannot import IDSFactory from imas: {exc}")

# Shared helpers from your package (original nimrod2imas.py)
from nimrod2imas import entry_dir, open_dbentry, get_ids


_num_re = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eEdD][+-]?\d+)?$")


def _is_bool_token(tok: str) -> bool:
    t = tok.strip().lower()
    return t in ("t", "f", "true", "false", ".true.", ".false.", "1", "0")


def _is_number_token(tok: str) -> bool:
    return bool(_num_re.match(tok.strip()))


def _split_commas_preserving_quotes(s: str) -> List[str]:
    parts, cur = [], []
    in_s = in_d = False
    for ch in s:
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        if ch == "," and (not in_s) and (not in_d):
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p != ""]


def _parse_scalar_token(tok: str) -> Any:
    t = tok.strip()
    tl = t.lower()
    if tl in (".true.", "true", "t", "1"):
        return True
    if tl in (".false.", "false", "f", "0"):
        return False
    if _is_number_token(t):
        try:
            if "d" in t.lower():
                t2 = re.sub(r"[dD]", "E", t)
            else:
                t2 = t
            if "." in t2 or "e" in t2.lower():
                return float(t2)
            return int(t2)
        except Exception:
            return t
    # strip outer quotes
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        return t[1:-1]
    return t


def _parse_xml_value(text: str) -> Any:
    """
    Parse XML <var> text into scalar or list using heuristics.
    """
    raw = "" if text is None else text
    s = raw.strip()

    if s == "":
        return ""

    # comma-separated -> list
    if "," in s:
        tokens = _split_commas_preserving_quotes(s)
        return [_parse_xml_value(tok) for tok in tokens]

    # shlex respects quotes and keeps internal multiple spaces inside quotes
    try:
        toks = shlex.split(s)
    except Exception:
        toks = s.split()

    if len(toks) == 0:
        return ""

    if len(toks) == 1:
        # If the raw value was quoted as a whole, prefer the shlex token (it strips the quotes cleanly)
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return toks[0]
        # otherwise preserve raw (keeps internal spacing if present)
        return raw

    # Multiple tokens:
    # 1) all numeric/bool => list
    if all(_is_number_token(t) or _is_bool_token(t) for t in toks):
        return [_parse_scalar_token(t) for t in toks]

    # 2) all identical => list of strings (enables repetition syntax on output)
    if all(t == toks[0] for t in toks):
        return [toks[0] for _ in toks]

    # 3) otherwise treat as a single scalar string (preserve original spacing)
    return raw


def _quote_string(s: str) -> str:
    """
    Emit Fortran string literal.
    Rule:
      - If string contains whitespace or '=' or ',' -> use double quotes
      - Else use single quotes
    """
    ss = str(s)
    if (ss.startswith('"') and ss.endswith('"')) or (ss.startswith("'") and ss.endswith("'")):
        # strip accidental outer quotes that may have survived
        ss = ss[1:-1]
    if re.search(r"\s|,|=", ss):
        esc = ss.replace('"', '""')
        return f"\"{esc}\""
    esc = ss.replace("'", "''")
    return f"'{esc}'"


def _format_fortran_value(v: Any) -> str:
    # list output with repetition if possible
    if isinstance(v, list):
        if len(v) == 0:
            return ""
        if all(isinstance(x, str) for x in v) and all(x == v[0] for x in v):
            return f"{len(v)}*{_quote_string(v[0])}"
        return ", ".join(_format_fortran_value(x) for x in v)

    if isinstance(v, bool):
        return ".true." if v else ".false."
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    # scalar string (or fallback)
    return _quote_string(str(v))


def _add_suffix(filename: str, suffix: str = "_from_imas") -> str:
    p = Path(filename)
    if p.suffix:
        return p.with_name(p.stem + suffix + p.suffix).name
    return p.name + suffix


def xml_to_namelists(xml_str: str, *, filename_fallback: str = "restored.in") -> List[Tuple[str, str]]:
    """
    Return list of (filename, namelist_text), one per <namelist> block if present.
    """
    if not xml_str or not str(xml_str).strip():
        return []

    root = ET.fromstring(xml_str)

    namelists = root.findall(".//namelist")
    if not namelists:
        # legacy: treat root as the container
        namelists = [root]

    outputs: List[Tuple[str, str]] = []
    for nml in namelists:
        fname = (nml.get("filename") if isinstance(nml, ET.Element) else None) or filename_fallback
        fname = _add_suffix(fname, "_from_imas")

        groups = nml.findall(".//group")
        out_lines: List[str] = []
        for g in groups:
            gname = g.get("name") or "group"
            out_lines.append(f"&{gname}")
            for v_el in g.findall("var"):
                vname = v_el.get("name") or "var"
                val = _parse_xml_value(v_el.text or "")
                out_lines.append(f"  {vname} = {_format_fortran_value(val)}")
            out_lines.append("/")
            out_lines.append("")
        txt = "\n".join(out_lines).rstrip() + "\n"
        outputs.append((fname, txt))
    return outputs


def main():
    p = argparse.ArgumentParser(
        description="Restore NIMROD namelist text file(s) from IMAS IDS code.parameters XML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backend", choices=["hdf5", "mdsplus"], default="hdf5")
    p.add_argument("--dbpath", default=".")
    p.add_argument("--dd", required=True)
    p.add_argument("--dd-version", default=None, help="IMAS DD version (defaults to $IMAS_VERSION)")
    p.add_argument("--dd-version-dir", choices=["major", "full"], default="major")
    p.add_argument("--pulse", type=int, required=True)
    p.add_argument("--run", type=int, required=True)
    p.add_argument("--entry", default=None, help="Explicit entry directory override")
    p.add_argument("--mode", default="r")
    p.add_argument("--occ", type=int, default=0)
    p.add_argument("--ids", choices=["mhd_linear", "mhd", "equilibrium", "core_profiles"], default="mhd_linear",
                   help="IDS to read code.parameters from")
    p.add_argument("--outdir", default=".", help="Write restored file(s) here")
    p.add_argument("--overwrite", action="store_true")

    args = p.parse_args()

    ddv = (args.dd_version or os.environ.get("IMAS_VERSION") or "").strip()
    if not ddv:
        raise SystemExit("ERROR: dd_version is not set. Provide --dd-version or set IMAS_VERSION.")

    entry_path = os.path.abspath(os.path.expanduser(args.entry)) if args.entry else str(
        entry_dir(args.dbpath, args.dd, ddv, args.pulse, args.run, args.dd_version_dir)
    )

    db, uri, _ = open_dbentry(args.backend, entry_path, mode=args.mode, dd_version=ddv)
    print(f"IMAS entry directory: {entry_path}")
    print(f"IMAS URI: {uri}")

    fac = IDSFactory(ddv)
    ids_obj = get_ids(db, fac, args.ids, args.occ)

    try:
        xml = ids_obj.code.parameters
    except Exception:
        xml = ""

    if not xml or not str(xml).strip():
        raise SystemExit(f"ERROR: {args.ids}.code.parameters is empty (occ={args.occ}).")

    outs = xml_to_namelists(str(xml), filename_fallback=f"{args.ids}.in")
    if not outs:
        raise SystemExit("ERROR: Could not parse any <namelist> blocks from code.parameters.")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for fname, txt in outs:
        outpath = outdir / fname
        if outpath.exists() and (not args.overwrite):
            raise SystemExit(f"ERROR: {outpath} exists. Use --overwrite to replace.")
        outpath.write_text(txt)
        print(f"Wrote: {outpath}")

    try:
        db.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
