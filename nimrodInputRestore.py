#!/usr/bin/env python3
"""
nimrodInputRestore.py

Restore FGnimeq (nimeq.in, oculus.in, fluxgrid.in) and NIMROD (nimrod.in)
Fortran namelist files from IMAS, assuming they were stored by
nimrod2imas.py as:

- equilibrium.code.parameters: XML like

    <fgnimeq_inputs>
      <nimeq_in filename="nimeq.in">
        <group name="...">
          <var name="...">value(s)</var>
        </group>
        ...
      </nimeq_in>
      <oculus_in filename="oculus.in"> ... </oculus_in>
      <fluxgrid_in filename="fluxgrid.in"> ... </fluxgrid_in>
    </fgnimeq_inputs>

- mhd.code.parameters (or fallback core_profiles.code.parameters): XML like

    <nimrod_inputs>
      <nimrod_in filename="nimrod.in">
        <group name="...">
          <var name="...">value(s)</var>
        </group>
        ...
      </nimrod_in>
    </nimrod_inputs>

We parse those XML blobs back into f90nml.Namelist objects and rewrite
Fortran namelist files, which you can compare to the originals.

Usage example:

  python3 restore_nimrod_namelists.py \
      --dd nimrod --pulse 1 --run 0 --backend hdf5 \
      --out-nimeq  nimeq_from_imas.in \
      --out-oculus oculus_from_imas.in \
      --out-fluxgrid fluxgrid_from_imas.in \
      --out-nimrod nimrod_from_imas.in
"""

import argparse
import os
import xml.etree.ElementTree as ET

import imas
from imas import imasdef, hli_exception

import f90nml


# ----------------------------------------------------------------------
# parsing utilities (inverse of nimrod2imas4 value_to_string)
# ----------------------------------------------------------------------

def _parse_scalar(token):
    """Parse a single scalar token into bool/int/float/string."""
    token = token.strip()
    if not token:
        return None

    low = token.lower()

    # logicals
    if low in (".true.", "true", ".t.", "t"):
        return True
    if low in (".false.", "false", ".f.", "f"):
        return False

    # quoted string
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return token[1:-1]

    # int
    try:
        return int(token)
    except ValueError:
        pass

    # float
    try:
        return float(token)
    except ValueError:
        pass

    # fallback: bare string
    return token


def _parse_value(text):
    """Parse XML var text back into a Python value suitable for f90nml."""
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None

    # space-separated list?
    if " " in s:
        tokens = [t for t in s.split() if t]
        vals = [_parse_scalar(t) for t in tokens]
        return vals

    # single scalar
    return _parse_scalar(s)


def xml_tag_to_namelist(tag_element):
    """
    Convert an element like:

      <nimeq_in filename="nimeq.in">
        <group name="g1">
          <var name="a">1.0</var>
          <var name="b">.true.</var>
        </group>
        <group name="g2">
          ...
        </group>
      </nimeq_in>

    into a f90nml.Namelist instance:

      &g1
        a = 1.0
        b = .true.
      &g2
        ...
    """
    nml = f90nml.Namelist()
    if tag_element is None:
        return nml

    for g_el in tag_element.findall("group"):
        gname = g_el.get("name")
        if gname is None:
            continue
        group_dict = {}
        for v_el in g_el.findall("var"):
            vname = v_el.get("name")
            if vname is None:
                continue
            value = _parse_value(v_el.text or "")
            group_dict[vname] = value
        nml[gname] = group_dict

    return nml


def _write_namelist(nml, path):
    """Write a f90nml.Namelist to file path, creating dirs as needed."""
    if not path:
        return
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        nml.write(f)


# ----------------------------------------------------------------------
# main logic
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Restore NIMROD / FGnimeq namelist files from IMAS XML."
    )

    parser.add_argument("--dd", "--db", dest="dd", default="nimrod",
                        help="IMAS database name (default: nimrod)")
    parser.add_argument("--pulse", type=int, default=1,
                        help="IMAS pulse number (default: 1)")
    parser.add_argument("--run", type=int, default=0,
                        help="IMAS run number (default: 0)")
    parser.add_argument("--backend", choices=["hdf5", "mdsplus"], default="hdf5",
                        help="IMAS backend (default: hdf5)")

    parser.add_argument("--out-nimeq", help="output nimeq.in", default="nimeq_from_imas.in")
    parser.add_argument("--out-oculus", help="output oculus.in", default="oculus_from_imas.in")
    parser.add_argument("--out-fluxgrid", help="output fluxgrid.in", default="fluxgrid_from_imas.in")
    parser.add_argument("--out-nimrod", help="output nimrod.in", default="nimrod_from_imas.in")

    args = parser.parse_args()

    backend = imasdef.HDF5_BACKEND if args.backend == "hdf5" else imasdef.MDSPLUS_BACKEND

    db = imas.DBEntry(backend, args.dd, args.pulse, args.run)
    db.open()

    # ------------------------------------------------------------------
    # 1. FGnimeq namelists from equilibrium.code.parameters
    # ------------------------------------------------------------------
    try:
        eq_ids = imas.equilibrium()
        eq_ids.get(0, db)

        xml_eq = getattr(eq_ids.code, "parameters", "") or ""
        if xml_eq.strip():
            try:
                root_eq = ET.fromstring(xml_eq)
            except Exception as exc:
                print(f"[warn] Could not parse equilibrium.code.parameters XML: {exc}")
                root_eq = None
        else:
            root_eq = None

        if root_eq is not None and root_eq.tag == "fgnimeq_inputs":
            # nimeq.in
            nimeq_el = root_eq.find("nimeq_in")
            if nimeq_el is not None:
                fname_xml = nimeq_el.get("filename") or "nimeq_from_imas.in"
                out_nimeq = args.out_nimeq or fname_xml
                nml_nimeq = xml_tag_to_namelist(nimeq_el)
                _write_namelist(nml_nimeq, out_nimeq)
                print(f"[info] Wrote nimeq namelist to: {out_nimeq}")

            # oculus.in
            oculus_el = root_eq.find("oculus_in")
            if oculus_el is not None:
                fname_xml = oculus_el.get("filename") or "oculus_from_imas.in"
                out_oculus = args.out_oculus or fname_xml
                nml_oculus = xml_tag_to_namelist(oculus_el)
                _write_namelist(nml_oculus, out_oculus)
                print(f"[info] Wrote oculus namelist to: {out_oculus}")

            # fluxgrid.in
            fluxgrid_el = root_eq.find("fluxgrid_in")
            if fluxgrid_el is not None:
                fname_xml = fluxgrid_el.get("filename") or "fluxgrid_from_imas.in"
                out_flux = args.out_fluxgrid or fname_xml
                nml_flux = xml_tag_to_namelist(fluxgrid_el)
                _write_namelist(nml_flux, out_flux)
                print(f"[info] Wrote fluxgrid namelist to: {out_flux}")

            if (nimeq_el is None and oculus_el is None and fluxgrid_el is None):
                print("[info] equilibrium.code.parameters XML has no "
                      "<nimeq_in>/<oculus_in>/<fluxgrid_in> tags.")
        else:
            print("[info] No FGnimeq XML found in equilibrium.code.parameters.")
    except hli_exception.IDSNotAvailable:
        print("[info] equilibrium IDS not available; skipping FGnimeq restoration.")

    # ------------------------------------------------------------------
    # 2. NIMROD namelist from mhd.code.parameters or core_profiles.code.parameters
    # ------------------------------------------------------------------
    nimrod_xml = ""
    source_label = None

    # try mhd first
    try:
        mhd_ids = imas.mhd()
        mhd_ids.get(0, db)
        nimrod_xml = getattr(mhd_ids.code, "parameters", "") or ""
        if nimrod_xml.strip():
            source_label = "mhd"
    except hli_exception.IDSNotAvailable:
        nimrod_xml = ""
        source_label = None

    # fallback: core_profiles.code.parameters
    if not nimrod_xml.strip():
        try:
            cp_ids = imas.core_profiles()
            cp_ids.get(0, db)
            nimrod_xml = getattr(cp_ids.code, "parameters", "") or ""
            if nimrod_xml.strip():
                source_label = "core_profiles"
        except hli_exception.IDSNotAvailable:
            nimrod_xml = ""
            source_label = None

    if nimrod_xml.strip():
        try:
            root_mhd = ET.fromstring(nimrod_xml)
        except Exception as exc:
            print(f"[warn] Could not parse {source_label}.code.parameters XML: {exc}")
            root_mhd = None

        if root_mhd is not None and root_mhd.tag == "nimrod_inputs":
            nimrod_el = root_mhd.find("nimrod_in")
            if nimrod_el is not None:
                fname_xml = nimrod_el.get("filename") or "nimrod_from_imas.in"
                out_nimrod = args.out_nimrod or fname_xml
                nml_nimrod = xml_tag_to_namelist(nimrod_el)
                _write_namelist(nml_nimrod, out_nimrod)
                print(f"[info] Wrote NIMROD namelist to: {out_nimrod} "
                      f"(source: {source_label}.code.parameters)")
            else:
                print(f"[info] nimrod_inputs XML from {source_label}.code.parameters "
                      "has no <nimrod_in> tag.")
        else:
            print(f"[info] No nimrod_inputs XML found in {source_label}.code.parameters.")
    else:
        print("[info] No NIMROD XML found in mhd/core_profiles code.parameters.")

    db.close()


if __name__ == "__main__":
    main()

