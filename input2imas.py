#!/usr/bin/env python3
"""
input2imas.py

Convert NIMROD input files (GEQDSK + PEQDSK/p-file) to IMAS:

  - equilibrium IDS from GEQDSK (using OMFITgeqdsk, no SciPy aux)
  - core_profiles IDS from PEQDSK (using OMFITpFile)
  - wall IDS from GEQDSK limiter outline

Omega handling:

  - If vtor1 is zero/absent, use omeg to reconstruct VTOR:
      VTOR = omeg * R_mid
  - If vpol1 is zero/absent:
      - If kpol nonzero: VPOL = kpol * Bp_mid
      - else if omegp nonzero: VPOL = omegp * R_mid * Bp_mid / Bt_mid
  - Diamagnetic:
      - If omgpp nonzero: v_dia = omgpp * 1e3 / (2*pi*R_mid)
      - Stored in core_profiles.ion[dia_idx].velocity.diamagnetic
      - On reconstruction, omgpp = 2*pi * R_mid * v_dia / 1e3  (kRad/s)

This version also stores the NIMROD-related input files as XML using f90nml:

  - nimeq.in, oculus.in, fluxgrid.in  -> equilibrium.code.parameters (code.name='fgnimeq')
  - nimrod.in                         -> mhd.code.parameters (code.name='nimrod')
    * If mhd IDS creation fails, falls back to core_profiles.code.parameters

For the namelists, we do:

  - Parse the Fortran namelist file with f90nml
  - Build an XML tree with xml.etree.ElementTree, roughly:

      <fgnimeq_inputs>
        <nimeq_in filename="nimeq.in">
          <group name="...">
            <var name="...">value(s)</var>
          </group>
          ...
        </nimeq_in>
        ...
      </fgnimeq_inputs>

  - Similarly for nimrod.in:

      <nimrod_inputs>
        <nimrod_in filename="nimrod.in">
          ...
        </nimrod_in>
      </nimrod_inputs>
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

import hashlib
from datetime import datetime, timezone
import yaml

import imas
from nimrod2imas import (
    entry_dir,
    open_dbentry,
    put_ids,
    value_to_string as _nimrod_value_to_string,
    update_workflow_and_dataset_fair,
    sanitize_cli_command,
    ids_factory,
    __version__,
    VERSION
)
from imas import IDSFactory

import f90nml
import xml.etree.ElementTree as ET

# Create a global IDS factory for creating IDS objects
_ids_factory = IDSFactory()

from omfit_classes.omfit_eqdsk import OMFITgeqdsk
from omfit_classes.omfit_osborne import OMFITpFile

import scipy.integrate as integrate

# SciPy >= 1.11 removed cumtrapz; OMFIT still expects it
if not hasattr(integrate, "cumtrapz"):
    from scipy.integrate import cumulative_trapezoid
    integrate.cumtrapz = cumulative_trapezoid

# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

_INTERNAL_META_KEY = "__nimrod2imas__"

def all_zero(arr, tol=1e-12):
    """Return True if array is empty or all entries are ~0."""
    a = np.asarray(arr, dtype=float)
    return (a.size == 0) or np.all(np.abs(a) < tol)


def value_to_string(val):
    """Convert Python value to token string for XML (delegates to nimrod2imas.value_to_string)."""
    return _nimrod_value_to_string(val)


def load_metadata_yaml(yaml_path):
    """Load optional metadata YAML.

    Returns an empty dict on missing/invalid YAML, but prints a warning.
    The returned mapping carries an internal block with the original YAML text
    and absolute source path so the full YAML can be preserved in provenance.
    """
    if not yaml_path:
        print("Warning: --input YAML was not provided; summary/dataset_fair/workflow will be written with minimal metadata.")
        return {}
    yaml_path = os.path.abspath(os.path.expanduser(str(yaml_path)))
    if not os.path.isfile(yaml_path):
        print(f"Warning: metadata YAML not found: {yaml_path}. Proceeding without it.")
        return {}
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        data = yaml.safe_load(raw_text) or {}
        if not isinstance(data, dict):
            print(f"Warning: metadata YAML root is not a mapping/dict: {yaml_path}. Proceeding without it.")
            return {}
        data = dict(data)
        internal = data.get(_INTERNAL_META_KEY, {})
        if not isinstance(internal, dict):
            internal = {}
        internal["yaml_path"] = yaml_path
        internal["raw_text"] = raw_text
        data[_INTERNAL_META_KEY] = internal
        return data
    except Exception as exc:
        print(f"Warning: failed to read metadata YAML {yaml_path}: {exc}. Proceeding without it.")
        return {}


def yget(dct, *keys, default=None):
    """Nested dict getter."""
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def internal_metadata(meta):
    """Return the internal metadata block carried alongside parsed YAML."""
    if not isinstance(meta, dict):
        return {}
    internal = meta.get(_INTERNAL_META_KEY, {})
    return internal if isinstance(internal, dict) else {}


def metadata_yaml_path(meta):
    """Return absolute path of the original YAML file, if known."""
    return str(internal_metadata(meta).get("yaml_path", "") or "")


def metadata_yaml_text(meta):
    """Return original YAML file contents, preserving user formatting/comments when possible."""
    raw = internal_metadata(meta).get("raw_text", "")
    return str(raw or "")


def public_metadata_dict(meta):
    """Return metadata dict without internal bookkeeping keys."""
    if not isinstance(meta, dict):
        return {}
    return {k: v for k, v in meta.items() if k != _INTERNAL_META_KEY}


def metadata_yaml_dump(meta):
    """Best-effort YAML serialization of user metadata without internal keys."""
    raw = metadata_yaml_text(meta)
    if raw:
        return raw
    try:
        return yaml.safe_dump(public_metadata_dict(meta), sort_keys=False, allow_unicode=True)
    except Exception:
        return ""


def build_metadata_comment(meta, args=None):
    """Human-readable metadata summary for ids_properties.comment."""
    lines = []
    descr = yget(meta, "dataset", "description", default=None) or yget(meta, "description", default=None)
    identifier = yget(meta, "dataset", "identifier", default=None)
    rights_holder = yget(meta, "dataset", "rights_holder", default=None)
    license_ = yget(meta, "dataset", "license", default=None)
    valid = yget(meta, "dataset", "valid", default=None)
    replaces = yget(meta, "dataset", "replaces", default=None)
    is_replaced_by = yget(meta, "dataset", "is_replaced_by", default=None)
    yaml_path = metadata_yaml_path(meta)

    if descr:
        lines.append(f"dataset.description: {descr}")
    if identifier:
        lines.append(f"dataset.identifier: {identifier}")
    if rights_holder:
        lines.append(f"dataset.rights_holder: {rights_holder}")
    if license_:
        lines.append(f"dataset.license: {license_}")
    if valid:
        lines.append(f"dataset.valid: {valid}")
    if replaces:
        lines.append(f"dataset.replaces: {replaces}")
    if is_replaced_by:
        lines.append(f"dataset.is_replaced_by: {is_replaced_by}")
    if yaml_path:
        lines.append(f"metadata_yaml.path: {yaml_path}")
    if args is not None:
        lines.append(
            f"effective_imas: dd={getattr(args, 'dd', '')}; pulse={getattr(args, 'pulse', '')}; run={getattr(args, 'run', '')}"
        )
    return "\n".join([ln for ln in lines if ln])


def _append_xml_copy(parent, xml_obj, wrapper_tag=None):
    """Append an XML element/string to parent, optionally wrapped in a new tag."""
    if xml_obj is None:
        return None
    target_parent = parent
    if wrapper_tag:
        target_parent = ET.SubElement(parent, wrapper_tag)
    try:
        if isinstance(xml_obj, ET.Element):
            target_parent.append(ET.fromstring(ET.tostring(xml_obj, encoding="unicode")))
        else:
            target_parent.append(ET.fromstring(str(xml_obj)))
        return target_parent
    except Exception:
        text_el = ET.SubElement(target_parent, "text")
        text_el.text = str(xml_obj)
        return target_parent


def build_component_parameters_xml(root_tag, fields=None, *, input_xml=None, meta=None, extra_xml=None):
    """Create XML blob for workflow component parameters."""
    root = ET.Element(root_tag)

    info = ET.SubElement(root, "metadata")
    for key, val in (fields or {}).items():
        if val is None:
            continue
        sval = str(val).strip()
        if not sval:
            continue
        el = ET.SubElement(info, "field", name=str(key))
        el.text = sval

    yaml_path = metadata_yaml_path(meta)
    if yaml_path:
        ET.SubElement(root, "metadata_yaml", path=yaml_path)

    if input_xml is not None:
        _append_xml_copy(root, input_xml, wrapper_tag="input_snapshot")
    if extra_xml is not None:
        _append_xml_copy(root, extra_xml, wrapper_tag="extra")

    return ET.tostring(root, encoding="unicode")


def file_checksum(path, algo="sha256", chunk_bytes=1024 * 1024):
    """Compute file checksum (default sha256). Returns '' on errors."""
    if not path:
        return ""
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        return ""
    try:
        h = hashlib.new(algo)
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk_bytes)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return ""


def build_input2imas_workflow_parameters_xml(args, dd_version, meta, extra_files=None):
    """Build XML string describing input2imas run parameters (incl. optional checksums).

    This also preserves the original YAML file text and an effective metadata snapshot
    where IMAS machine/pulse/run come from the CLI arguments.
    """
    root = ET.Element("nimrod2imas_input2imas")

    def add_text(parent, tag, text, **attrs):
        attrs = {k: str(v) for k, v in attrs.items() if v is not None and str(v) != ""}
        el = ET.SubElement(parent, tag, **attrs)
        el.text = "" if text is None else str(text)
        return el

    add_text(root, "script", os.path.basename(__file__))
    add_text(root, "script_version", __version__)
    add_text(root, "script_repository", "https://github.com/PrincetonUniversity/nimrod2imas")
    add_text(root, "dd_version", dd_version)
    add_text(root, "backend", getattr(args, "backend", ""))
    add_text(root, "entry_path", getattr(args, "entry", "") or "")
    add_text(root, "dbpath", getattr(args, "dbpath", "") or "")
    add_text(root, "dd", getattr(args, "dd", "") or "")
    add_text(root, "pulse", getattr(args, "pulse", "") or "")
    add_text(root, "run", getattr(args, "run", "") or "")
    add_text(root, "occ_inputs", getattr(args, "occ", 0))

    effective = ET.SubElement(root, "effective_imas")
    add_text(effective, "machine", getattr(args, "dd", "") or "")
    add_text(effective, "pulse", getattr(args, "pulse", "") or "")
    add_text(effective, "run", getattr(args, "run", "") or "")
    add_text(effective, "dd_version", dd_version)

    files_el = ET.SubElement(root, "inputs")
    in_files = {
        "metadata_yaml": getattr(args, "input_yaml", None),
        "geqdsk": getattr(args, "geqdsk", None),
        "peqdsk": getattr(args, "peqdsk", None),
        "nimeq_in": getattr(args, "nimeq", None),
        "oculus_in": getattr(args, "oculus", None),
        "fluxgrid_in": getattr(args, "fluxgrid", None),
        "nimrod_in": getattr(args, "nimrod", None),
    }
    if extra_files:
        in_files.update(extra_files)

    for role, pth in in_files.items():
        if pth is None:
            continue
        pth_abs = os.path.abspath(os.path.expanduser(str(pth)))
        el = ET.SubElement(files_el, "file", role=role)
        el.set("path", pth_abs)

    # Optional checksums (controlled by YAML: converter.record_checksums)
    rec_cs = bool(yget(meta, "converter", "record_checksums", default=False))
    algo = str(yget(meta, "converter", "checksum_algorithm", default="sha256") or "sha256").strip() or "sha256"
    if rec_cs:
        cs_el = ET.SubElement(root, "checksums", algorithm=algo)
        for role, pth in in_files.items():
            if not pth:
                continue
            pth_abs = os.path.abspath(os.path.expanduser(str(pth)))
            h = file_checksum(pth_abs, algo=algo)
            if h:
                fel = ET.SubElement(cs_el, "file", role=role)
                fel.set("path", pth_abs)
                fel.text = h

    raw_yaml = metadata_yaml_text(meta)
    yaml_path = metadata_yaml_path(meta)
    if raw_yaml:
        raw_el = ET.SubElement(root, "metadata_yaml_raw")
        if yaml_path:
            raw_el.set("path", yaml_path)
        raw_el.text = raw_yaml

    parsed_public = public_metadata_dict(meta)
    if parsed_public:
        try:
            dumped = yaml.safe_dump(parsed_public, sort_keys=False, allow_unicode=True)
        except Exception:
            dumped = ""
        if dumped:
            parsed_el = ET.SubElement(root, "metadata_yaml_public")
            if yaml_path:
                parsed_el.set("path", yaml_path)
            parsed_el.text = dumped

    applied = ET.SubElement(root, "applied_metadata")
    dataset_block = ET.SubElement(applied, "dataset")
    for key in ("description", "identifier", "rights_holder", "license", "valid", "replaces", "is_replaced_by"):
        val = yget(meta, "dataset", key, default=None)
        if val not in (None, ""):
            add_text(dataset_block, key, val)

    nimrod_block = ET.SubElement(applied, "nimrod")
    for key in ("name", "description", "repository", "comment", "commit", "version"):
        val = yget(meta, "nimrod", key, default=None)
        if val not in (None, ""):
            add_text(nimrod_block, key, val)

    converter_block = ET.SubElement(applied, "converter")
    for key in ("repository", "comment", "commit", "version", "record_checksums", "checksum_algorithm"):
        val = yget(meta, "converter", key, default=None)
        if val not in (None, ""):
            add_text(converter_block, key, val)

    fgnimeq_block = ET.SubElement(applied, "fgnimeq")
    for key in ("name", "description", "repository", "comment", "commit", "version"):
        val = yget(meta, "fgnimeq", key, default=None)
        if val not in (None, ""):
            add_text(fgnimeq_block, key, val)

    return ET.tostring(root, encoding="unicode")


def attach_ids_properties_minimal(ids_obj, provider=None, comment=None, homogeneous_time=2):
    """Best-effort fill ids_properties fields."""
    try:
        ip = getattr(ids_obj, "ids_properties", None)
        if ip is None:
            return
        if hasattr(ip, "homogeneous_time"):
            ip.homogeneous_time = int(homogeneous_time)
        if provider and hasattr(ip, "provider"):
            ip.provider = str(provider)
        if hasattr(ip, "creation_date"):
            try:
                ip.creation_date = datetime.now(timezone.utc).isoformat()
            except Exception:
                pass
        if comment and hasattr(ip, "comment"):
            prev = str(getattr(ip, "comment", "") or "").strip()
            ip.comment = (prev + "\n" + str(comment)).strip() if prev else str(comment)
    except Exception:
        pass


def build_summary_ids(factory, args, meta, geq=None, time0=0.0):
    """Create minimal DD4.1-friendly summary IDS."""
    s = factory.summary()

    machine = (getattr(args, "dd", None) or yget(meta, "imas", "machine", default=None) or yget(meta, "machine", default=None) or "")
    pulse = getattr(args, "pulse", None)
    if pulse is None:
        pulse = yget(meta, "imas", "pulse", default=None)

    descr = yget(meta, "dataset", "description", default=None) or yget(meta, "description", default=None)
    if not descr:
        descr = "NIMROD simulation inputs converted to IMAS (equilibrium, core_profiles, wall) using nimrod2imas input2imas.py"

    # type is an identifier structure; set name if available
    try:
        if hasattr(s, "type") and hasattr(s.type, "name"):
            s.type.name = "simulation"
        elif hasattr(s, "type"):
            s.type = "simulation"
    except Exception:
        pass

    try:
        if hasattr(s, "machine"):
            s.machine = str(machine)
    except Exception:
        pass
    try:
        if hasattr(s, "pulse") and pulse is not None:
            s.pulse = int(pulse)
    except Exception:
        pass
    try:
        if hasattr(s, "description"):
            s.description = str(descr)
    except Exception:
        pass

    provider = yget(meta, "contact", "provider", default=None) or yget(meta, "provider", default=None)
    comment_lines = ["Generated by nimrod2imas input2imas.py"]
    meta_comment = build_metadata_comment(meta, args=args)
    if meta_comment:
        comment_lines.append(meta_comment)
    attach_ids_properties_minimal(s, provider=provider, comment="\n".join(comment_lines), homogeneous_time=2)
    return s


def build_dataset_fair_ids(factory, meta):
    """Create dataset_fair IDS from YAML (DD4.1)."""
    df = factory.dataset_fair()

    identifier = yget(meta, "dataset", "identifier", default="") or ""
    replaces = yget(meta, "dataset", "replaces", default="") or ""
    is_replaced_by = yget(meta, "dataset", "is_replaced_by", default="") or ""
    valid = yget(meta, "dataset", "valid", default="") or ""
    rights_holder = yget(meta, "dataset", "rights_holder", default="") or ""
    license_ = yget(meta, "dataset", "license", default="") or ""

    for attr, val in (("identifier", identifier),
                      ("replaces", replaces),
                      ("is_replaced_by", is_replaced_by),
                      ("valid", valid),
                      ("rights_holder", rights_holder),
                      ("license", license_)):
        try:
            if hasattr(df, attr) and val:
                setattr(df, attr, str(val))
        except Exception:
            pass

    descr = yget(meta, "dataset", "description", default=None) or None
    provider = yget(meta, "contact", "provider", default=None) or yget(meta, "provider", default=None)
    comment_lines = []
    if descr:
        comment_lines.append(str(descr))
    meta_comment = build_metadata_comment(meta)
    if meta_comment:
        comment_lines.append(meta_comment)
    attach_ids_properties_minimal(df, provider=provider, comment="\n".join(comment_lines) if comment_lines else None, homogeneous_time=2)
    return df


def build_workflow_ids(factory, args, dd_version, meta, nimrod_inputs_xml=None, fgnimeq_inputs_xml=None):
    """Create workflow IDS describing NIMROD + conversion tools."""
    wf = factory.workflow()
    provider = yget(meta, "contact", "provider", default=None) or yget(meta, "provider", default=None)
    comment_lines = ["Workflow metadata for NIMROD and nimrod2imas conversion"]
    yaml_path = metadata_yaml_path(meta)
    if yaml_path:
        comment_lines.append(f"metadata_yaml.path: {yaml_path}")
    attach_ids_properties_minimal(
        wf,
        provider=provider,
        comment="\n".join(comment_lines),
        homogeneous_time=2,
    )

    comps = []

    nimrod_params = build_component_parameters_xml(
        "nimrod_component",
        {
            "comment": yget(meta, "nimrod", "comment", default="") or "",
            "commit": yget(meta, "nimrod", "commit", default="") or "",
            "version": yget(meta, "nimrod", "version", default="") or "",
        },
        input_xml=nimrod_inputs_xml,
        meta=meta,
    )

    comps.append({
        "name": yget(meta, "nimrod", "name", default="NIMROD") or "NIMROD",
        "description": yget(meta, "nimrod", "description", default="Extended-MHD code") or "Extended-MHD code",
        "repository": yget(meta, "nimrod", "repository", default="") or "",
        "commit": yget(meta, "nimrod", "commit", default="") or "",
        "version": yget(meta, "nimrod", "version", default="") or "",
        "parameters": nimrod_params,
    })

    # Component: input2imas converter (fixed repo + version from this script)
    conv_commit = yget(meta, "converter", "commit", default="") or ""
    conv_repo = yget(meta, "converter", "repository", default="https://github.com/PrincetonUniversity/nimrod2imas") or "https://github.com/PrincetonUniversity/nimrod2imas"
    conv_params = build_input2imas_workflow_parameters_xml(args, dd_version, meta)

    # Optionally embed the input XML blobs inside parameters for easier provenance tracking
    try:
        root = ET.fromstring(conv_params)
        if nimrod_inputs_xml is not None:
            _append_xml_copy(root, nimrod_inputs_xml, wrapper_tag="nimrod_inputs_snapshot")
        if fgnimeq_inputs_xml is not None:
            _append_xml_copy(root, fgnimeq_inputs_xml, wrapper_tag="fgnimeq_inputs_snapshot")
        conv_params = ET.tostring(root, encoding="unicode")
    except Exception:
        pass

    comps.append({
        "name": "nimrod2imas:input2imas",
        "description": "Convert NIMROD input files (GEQDSK/PEQDSK + namelists) to IMAS",
        "repository": str(conv_repo),
        "commit": str(conv_commit),
        "version": str(yget(meta, "converter", "version", default=__version__) or __version__),
        "parameters": conv_params,
    })

    # Optional component: fgnimeq (if XML exists)
    if fgnimeq_inputs_xml is not None:
        fgnimeq_params = build_component_parameters_xml(
            "fgnimeq_component",
            {
                "comment": yget(meta, "fgnimeq", "comment", default="") or "",
                "commit": yget(meta, "fgnimeq", "commit", default="") or "",
                "version": yget(meta, "fgnimeq", "version", default="") or "",
            },
            input_xml=fgnimeq_inputs_xml,
            meta=meta,
        )
        comps.append({
            "name": yget(meta, "fgnimeq", "name", default="fgnimeq") or "fgnimeq",
            "description": yget(meta, "fgnimeq", "description", default="NIMROD preprocessing inputs (grid/equilibrium mapping)") or "NIMROD preprocessing inputs (grid/equilibrium mapping)",
            "repository": yget(meta, "fgnimeq", "repository", default="") or "",
            "commit": yget(meta, "fgnimeq", "commit", default="") or "",
            "version": yget(meta, "fgnimeq", "version", default="") or "",
            "parameters": fgnimeq_params,
        })

    # Attach to workflow.time_loop.component array
    try:
        comp_arr = wf.time_loop.component
        try:
            comp_arr.resize(len(comps))
        except Exception:
            pass
        for i, c in enumerate(comps):
            comp = comp_arr[i]
            for fld in ("name", "description", "repository", "commit", "version", "parameters"):
                val = c.get(fld, "")
                if val is None:
                    continue
                try:
                    if hasattr(comp, fld) and val != "":
                        setattr(comp, fld, str(val))
                except Exception:
                    pass
    except Exception as exc:
        print(f"Warning: could not populate workflow IDS components: {exc}")

    return wf

def build_namelist_xml(tag, path):
    """Read Fortran namelist file with f90nml and return an XML element.

    Parameters
    ----------
    tag : str
        Name of the outer XML element for this namelist (e.g. 'nimeq_in').
    path : str
        File path. If None or file is missing, returns None.
    """
    if path is None:
        return None
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return None

    nml = f90nml.read(path)
    root = ET.Element(tag, filename=os.path.basename(path))
    for group_name, group in nml.items():
        g_el = ET.SubElement(root, "group", name=str(group_name))
        for var_name, value in group.items():
            v_el = ET.SubElement(g_el, "var", name=str(var_name))
            v_el.text = value_to_string(value)
    return root


def build_fgnimeq_xml(nimeq_path=None, oculus_path=None, fluxgrid_path=None):
    """Construct XML blob describing the FGnimeq inputs (parsed with f90nml)."""
    root = ET.Element("fgnimeq_inputs")
    for tag, path in (("nimeq_in", nimeq_path),
                      ("oculus_in", oculus_path),
                      ("fluxgrid_in", fluxgrid_path)):
        el = build_namelist_xml(tag, path)
        if el is not None:
            root.append(el)

    if not list(root):
        return ""
    return ET.tostring(root, encoding="unicode")


def build_nimrod_xml(nimrod_path=None):
    """Construct XML blob containing the parsed nimrod.in namelist (f90nml)."""
    el = build_namelist_xml("nimrod_in", nimrod_path)
    if el is None:
        return ""
    root = ET.Element("nimrod_inputs")
    root.append(el)
    return ET.tostring(root, encoding="unicode")


def compute_midplane_geometry_from_geq(geq, psin_target):
    # Robust access to psi(R,Z) grid for midplane geometry calculations
    psirz_raw = _geq_get(geq, "PSIRZ", _geq_get(geq, "psirz", None))
    """
    Compute midplane geometry R_mid, Bp_mid, Bt_mid as functions of
    normalized flux psin_target in [0,1].

    - R_mid(psin): midplane major radius
    - Bp_mid(psin): poloidal B (approx)
    - Bt_mid(psin): toroidal B via FPOL = R * Bt

    Inputs:
      geq         : OMFITgeqdsk object (already loaded raw)
      psin_target : 1D array of target normalized psi in [0,1]
    """
    psin_target = np.asarray(psin_target, dtype=float)
    nw = int(geq["NW"])
    nh = int(geq["NH"])

    rdim  = float(geq["RDIM"])
    zdim  = float(geq["ZDIM"])
    rleft = float(geq["RLEFT"])
    zmid  = float(geq["ZMID"])

    rgrid = rleft + np.arange(nw) * rdim / (nw - 1)
    zgrid = (zmid - 0.5 * zdim) + np.arange(nh) * zdim / (nh - 1)

    if psirz_raw is None:
        psirz_raw = _geq_get(geq, "PSIRZ", _geq_get(geq, "psirz", None))
    psirz = np.asarray(psirz_raw).reshape((nh, nw))
    simag = float(geq["SIMAG"])
    sibry = float(geq["SIBRY"])
    dpsi  = sibry - simag if sibry != simag else 1.0

    # midplane index
    j_mid = int(np.argmin(np.abs(zgrid - zmid)))
    psi_mid = psirz[j_mid, :]

    # normalized psi at midplane
    psin_mid = (psi_mid - simag) / dpsi

    # sort in increasing psin
    order = np.argsort(psin_mid)
    psin_mid_sorted = psin_mid[order]
    R_mid_sorted    = rgrid[order]

    # ensure monotonic increasing
    mask = np.isfinite(psin_mid_sorted)
    psin_mid_sorted = psin_mid_sorted[mask]
    R_mid_sorted    = R_mid_sorted[mask]

    if psin_mid_sorted.size < 2:
        # fallback: uniform R
        R_mid = np.interp(psin_target, [0.0, 1.0], [rgrid[0], rgrid[-1]])
    else:
        R_mid = np.interp(psin_target, psin_mid_sorted, R_mid_sorted)

    # Bt from FPOL ~ R * Bt
    fpol = np.asarray(geq["FPOL"], dtype=float)  # vs psi index
    nfp  = len(fpol)
    psin_fpol = np.linspace(0.0, 1.0, nfp)
    Bt_mid = np.interp(psin_target, psin_fpol, fpol) / np.maximum(R_mid, 1e-6)

    # crude Bp from radial derivative of psi at midplane: Bp ~ |dpsi/dR| / R
    dpsi_dR = np.gradient(psi_mid, rgrid)
    dpsi_dR_psin = np.interp(psin_target, psin_mid, dpsi_dR, left=dpsi_dR[0], right=dpsi_dR[-1])
    Bp_mid = np.abs(dpsi_dR_psin) / np.maximum(R_mid, 1e-6)

    return R_mid, Bp_mid, Bt_mid


# ----------------------------------------------------------------------
# GEQDSK -> equilibrium + wall
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Small helpers for robust GEQDSK parsing (some files / OMFIT versions use different key casing)
def _geq_get(geq, key, default=None):
    try:
        return geq[key]
    except Exception:
        return default

def _geq_get_int(geq, *keys):
    for k in keys:
        v = _geq_get(geq, k, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return None

def _geq_get_float(geq, *keys):
    for k in keys:
        v = _geq_get(geq, k, None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return None

def _parse_geqdsk_header_nwnh(geqdsk_path):
    # GEQDSK header (first line) ends with NW NH for standard g-files.
    # For nonstandard variants (q-eqdsk), this may fail; return (None, None).
    try:
        with open(geqdsk_path, "r") as f:
            line1 = f.readline()
        parts = line1.strip().split()
        if len(parts) >= 2:
            nw = int(parts[-2])
            nh = int(parts[-1])
            return nw, nh
    except Exception:
        pass
    return None, None


def geqdsk_to_equilibrium(geqdsk_path, time=0.0):
    """
    Read GEQDSK and build an equilibrium IDS using only raw arrays
    (no SciPy-dependent aux from OMFITgeqdsk).
    """
    geq = OMFITgeqdsk(geqdsk_path)
    geq.load(raw=True, add_aux=False)

    eq = _ids_factory.equilibrium()
    eq.ids_properties.homogeneous_time = 1
    eq.time = np.array([time], dtype=float)
    eq.time_slice.resize(1)
    ts = eq.time_slice[0]
    ts.time = time

    # --- 2D grid and psi(R,Z) ---
    # Some GEQDSK-like files (e.g. q-eqdsk variants) and/or OMFIT versions may not expose
    # NW/NH with uppercase keys. Fall back to PSIRZ shape or header parsing.
    psirz_raw = _geq_get(geq, "PSIRZ", _geq_get(geq, "psirz", None))
    nw = _geq_get_int(geq, "NW", "nw", "NR", "nr")
    nh = _geq_get_int(geq, "NH", "nh", "NZ", "nz")
    if (nw is None) or (nh is None):
        if psirz_raw is not None:
            arr = np.asarray(psirz_raw)
            if arr.ndim == 2:
                nh2, nw2 = arr.shape
                nw = nw if nw is not None else int(nw2)
                nh = nh if nh is not None else int(nh2)
    if (nw is None) or (nh is None):
        nw_h, nh_h = _parse_geqdsk_header_nwnh(geqdsk_path)
        nw = nw if nw is not None else nw_h
        nh = nh if nh is not None else nh_h
    if (nw is None) or (nh is None):
        keys_preview = []
        try:
            keys_preview = list(geq.keys())
        except Exception:
            pass
        raise KeyError(
            "Could not determine NW/NH from GEQDSK. "
            "Missing keys like NW/NH and PSIRZ shape unavailable. "
            f"Keys present (first 80): {keys_preview[:80]}"
        )

    rdim  = _geq_get_float(geq, "RDIM", "rdim")
    zdim  = _geq_get_float(geq, "ZDIM", "zdim")
    rleft = _geq_get_float(geq, "RLEFT", "rleft")
    zmid  = _geq_get_float(geq, "ZMID", "zmid")
    if None in (rdim, zdim, rleft, zmid):
        keys_preview = []
        try:
            keys_preview = list(geq.keys())
        except Exception:
            pass
        raise KeyError(
            "Missing one of RDIM/ZDIM/RLEFT/ZMID in GEQDSK. "
            f"Keys present (first 80): {keys_preview[:80]}"
        )

    rgrid = rleft + np.arange(nw) * rdim / (nw - 1)
    zgrid = (zmid - 0.5 * zdim) + np.arange(nh) * zdim / (nh - 1)

    ts.profiles_2d.resize(1)
    p2 = ts.profiles_2d[0]
    p2.grid_type.index = 1  # 1 = rectangular grid
    p2.grid.dim1 = rgrid
    p2.grid.dim2 = zgrid

    if psirz_raw is None:
        psirz_raw = _geq_get(geq, "PSIRZ", _geq_get(geq, "psirz", None))
    psirz = np.asarray(psirz_raw).reshape((nh, nw))
    # IMAS: psi(R,Z) with shape (len(R),len(Z))
    p2.psi = psirz.T

    # --- 1D profiles vs psi ---
    p1 = ts.profiles_1d
    pres   = np.asarray(geq["PRES"])
    fpol   = np.asarray(geq["FPOL"])
    ffprim = np.asarray(geq["FFPRIM"])
    pprime = np.asarray(geq["PPRIME"])
    qpsi   = np.asarray(geq["QPSI"])

    n_psi = len(pres)
    simag = float(geq["SIMAG"])
    sibry = float(geq["SIBRY"])
    psi_1d = np.linspace(simag, sibry, n_psi)

    p1.psi            = psi_1d
    p1.f              = fpol
    p1.pressure       = pres
    p1.f_df_dpsi      = ffprim
    p1.dpressure_dpsi = pprime
    p1.q              = qpsi

    # --- global quantities ---
    gq = ts.global_quantities
    gq.ip              = float(geq["CURRENT"])
    gq.psi_axis        = float(geq["SIMAG"])
    gq.psi_boundary    = float(geq["SIBRY"])
    gq.magnetic_axis.r = float(geq["RMAXIS"])
    gq.magnetic_axis.z = float(geq["ZMAXIS"])

    # --- boundary ---
    if int(geq["NBBBS"]) > 0:
        ts.boundary.outline.r = np.asarray(geq["RBBBS"])
        ts.boundary.outline.z = np.asarray(geq["ZBBBS"])

    # --- wall/limiter is handled in separate wall IDS ---

    # --- vacuum toroidal field ---
    eq.vacuum_toroidal_field.r0 = float(geq["RCENTR"])
    eq.vacuum_toroidal_field.b0 = np.array([float(geq["BCENTR"])])

    return eq, geq

def geqdsk_to_wall(geq, time=0.0):
    """
    Build wall IDS from GEQDSK limiter outline.

    - Uses LIMITR>0 as a switch that limiter data exist.
    - Enforces RLIM and ZLIM to have identical length by truncating
      both to the minimum length, to satisfy IMAS coordinate rules.
    """

    # --- create wall IDS, preferably via IDSFactory if provided ---
    wall = _ids_factory.wall()

    # homogeneous in time
    try:
        wall.ids_properties.homogeneous_time = 1
    except AttributeError:
        # Older/newer dd variants may not have ids_properties
        pass

    # global time array
    try:
        wall.time = np.array([time], dtype=float)
    except AttributeError:
        pass

    # one 2D description
    wall.description_2d.resize(1)
    desc = wall.description_2d[0]

    # NOTE: DO NOT set desc.time – it does not exist in dd 3.39 / imas-python
    # If you ever move to a dd where it exists, you can safely guard it:
    #
    try:
        desc.time = time
        desc.name = "limiter_wall"
    except AttributeError:
        pass

    # ----- limiter from RLIM/ZLIM -----
    try:
        limitr = int(geq["LIMITR"])
    except Exception:
        limitr = 0

    if limitr > 0:
        try:
            rlim = np.asarray(geq["RLIM"], dtype=float)
            zlim = np.asarray(geq["ZLIM"], dtype=float)
        except KeyError:
            # No usable limiter arrays; just return wall with empty limiter
            print(
                "[input2imas] GEQDSK has LIMITR>0 but missing RLIM/ZLIM; "
                "leaving wall.limiter empty."
            )
            return wall

        if rlim.size == 0 or zlim.size == 0:
            print(
                "[input2imas] RLIM/ZLIM arrays are empty; "
                "leaving wall.limiter empty."
            )
            return wall

        # Enforce same length for r and z (IMAS requires same coordinate size)
        n = min(rlim.size, zlim.size)
        if rlim.size != zlim.size:
            print(
                f"[input2imas] Warning: RLIM({rlim.size}) and ZLIM({zlim.size}) "
                f"lengths differ; truncating both to {n} points for wall IDS."
            )
        rlim = rlim[:n]
        zlim = zlim[:n]

        # Single limiter unit
        desc.limiter.unit.resize(1)
        lim = desc.limiter.unit[0]
        lim.name = "limiter"
        lim.outline.r = rlim
        lim.outline.z = zlim
    else:
        # No limiter defined in GEQDSK
        print(
            "[input2imas] LIMITR<=0 in GEQDSK; wall.limiter not populated."
        )

    return wall

# ----------------------------------------------------------------------
# PEQDSK / p-file -> core_profiles
# ----------------------------------------------------------------------
def fill_core_profiles_from_pfile(cp_ids, pfile_path, geq, time=0.0):
    """
    Fill core_profiles IDS (cp_ids) from an Osborne p-file, matching the
    NIMROD / p-file species convention:

      nspec = len(Z) from the "N Z A" block
      main_idx = nspec - 2  (if nspec >= 2 else 0)
      beam_idx = nspec - 1  (if nspec >= 2 else None)
      impurities: indices 0 .. nspec-3

    Mappings:

      electrons:
        ne  (10^20 m^-3) -> electrons.density_thermal
        te  (keV)        -> electrons.temperature (eV)

      total pressure:
        ptot (kPa) -> 3 * pressure_perpendicular (Pa)

      main ion:
        ni    -> ion[main_idx].density_thermal           (10^20 m^-3 -> m^-3)
        ti    -> ion[main_idx].temperature               (keV -> eV)
        vtor1 -> ion[main_idx].velocity.toroidal         (km/s -> m/s)
        vpol1 -> ion[main_idx].velocity.poloidal         (km/s -> m/s)
        omeg  -> if vtor1 is all-zero/missing, reconstruct VTOR from omeg
        kpol, omegp -> if vpol1 is all-zero/missing, reconstruct VPOL

      beam ion:
        nb -> ion[beam_idx].density_fast                 (10^20 m^-3 -> m^-3)
        pb -> ion[beam_idx].pressure_fast_perpendicular  (kPa -> Pa/3)

      impurities:
        nz{k}   -> ion[i_imp].density_thermal
        vtor{k} -> ion[i_imp].velocity.toroidal
        vpol{k} -> ion[i_imp].velocity.poloidal
        (i_imp runs over impurity indices 0..nspec-3)

      diamagnetic velocity:
        omgpp (kRad/s) -> v_dia stored in ion[dia_idx].velocity.diamagnetic,
        with v_dia = omgpp * 1e3 / (2*pi*R_mid).  We pick dia_idx as the
        first impurity species (0) if there is at least one impurity,
        otherwise the main_idx.
    """
    from omfit_classes.omfit_osborne import OMFITpFile
    import numpy as np
    import math

    p = OMFITpFile(pfile_path)
    p.load()

    cp_ids.ids_properties.homogeneous_time = 1
    cp_ids.time = np.array([time], dtype=float)
    cp_ids.profiles_1d.resize(1)
    prof = cp_ids.profiles_1d[0]
    prof.time = time

    # --- radial grid: use ne.psinorm if available ---
    if "ne" in p:
        rho = np.array(p["ne"]["psinorm"], dtype=float)
    else:
        # fallback: uniform [0,1]
        nw = int(geq["NW"])
        rho = np.linspace(0.0, 1.0, nw)
    prof.grid.rho_tor_norm = rho
    npts = len(rho)

    # "pseudo" normalized coordinate for interpolation/geometry
    s = np.linspace(0.0, 1.0, npts)

    # --- midplane geometry for omega->velocity mapping ---
    R_mid, Bp_mid, Bt_mid = compute_midplane_geometry_from_geq(geq, s)

    # helper: map any 1D data array to the rho grid using pseudo [0,1] coordinate
    def map_to_rho(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return np.zeros_like(rho)
        if arr.size == npts:
            return arr
        if arr.size == 1:
            return np.full_like(rho, arr.item(), dtype=float)
        xp = np.linspace(0.0, 1.0, arr.size)
        x  = s
        return np.interp(x, xp, arr)

    # ------------------------------------------------------------------
    # electrons
    # ------------------------------------------------------------------
    if "ne" in p:
        ne_20 = map_to_rho(p["ne"]["data"])
        prof.electrons.density_thermal = ne_20 * 1.0e20  # 10^20 -> m^-3

    if "te" in p:
        te_keV = map_to_rho(p["te"]["data"])
        prof.electrons.temperature = te_keV * 1.0e3  # keV -> eV

    # ------------------------------------------------------------------
    # total (scalar) pressure
    #
    # p-file ptot is provided in kPa. In IMAS core_profiles the preferred
    # isotropic storage is profiles_1d[].pressure_thermal (or pressure, depending
    # on DD version). We therefore store ptot directly as Pa in that leaf.
    #
    # Backward-compatibility:
    #   - If neither pressure_thermal nor pressure exist in the local IMAS build,
    #     we fall back to pressure_perpendicular assuming an isotropic pressure
    #     tensor:  ptot ~= 3 * p_perp  (so p_perp = ptot/3).
    # ------------------------------------------------------------------
    if "ptot" in p:
        ptot_kPa = map_to_rho(p["ptot"]["data"])
        ptot_Pa = ptot_kPa * 1.0e3
    
        if hasattr(prof, "pressure_thermal"):
            prof.pressure_thermal = ptot_Pa
        elif hasattr(prof, "pressure"):
            prof.pressure = ptot_Pa
        elif hasattr(prof, "pressure_perpendicular"):
            # last-resort fallback (legacy files): isotropic approximation
            prof.pressure_perpendicular = ptot_Pa / 3.0    
    # ------------------------------------------------------------------
    # species composition from N Z A (no hard-coded Z/A)
    # ------------------------------------------------------------------
    if "N Z A" in p:
        Z_arr = np.array(p["N Z A"]["Z"], dtype=float)
        A_arr = np.array(p["N Z A"]["A"], dtype=float)
        nspec = len(Z_arr)
    else:
        # fallback: single ion, Z/A unknown (left as 0.0)
        Z_arr = np.array([0.0], dtype=float)
        A_arr = np.array([0.0], dtype=float)
        nspec = 1

    prof.ion.resize(nspec)
    ions = prof.ion
    for k in range(nspec):
        ion = ions[k]
        if len(ion.element) == 0:
            ion.element.resize(1)
        ion.element[0].z_n = float(Z_arr[k])
        ion.element[0].a   = float(A_arr[k])
        # optional label; safe even if Z/A are zero
        try:
            ion.element[0].label = f"Z{int(round(Z_arr[k]))}A{A_arr[k]:.4g}"
        except Exception:
            pass

    # species index convention: last-2 = main, last = beam
    if nspec >= 2:
        main_idx = nspec - 2
        beam_idx = nspec - 1
    else:
        main_idx = 0
        beam_idx = None

    # pick diamagnetic species index:
    #   - if there is at least one impurity (nspec >= 3), use first impurity (0)
    #   - else use main_idx
    if nspec >= 3:
        dia_idx = 0
    else:
        dia_idx = main_idx

    ion_main = ions[main_idx]
    _ = ion_main.velocity.toroidal
    _ = ion_main.velocity.poloidal

    # ------------------------------------------------------------------
    # main ion: ni, ti, vtor1, vpol1, omeg/kpol/omegp
    # ------------------------------------------------------------------
    if "ni" in p:
        ni_20 = map_to_rho(p["ni"]["data"])
        ion_main.density_thermal = ni_20 * 1.0e20

    if "ti" in p:
        ti_keV = map_to_rho(p["ti"]["data"])
        ion_main.temperature = ti_keV * 1.0e3  # keV -> eV

    # vtor1: if present and nonzero, use directly; otherwise, reconstruct from omeg
    vtor1_from_file = None
    if "vtor1" in p:
        vtor1_from_file = map_to_rho(p["vtor1"]["data"]) * 1.0e3  # km/s -> m/s

    use_vtor1_file = vtor1_from_file is not None and not all_zero(vtor1_from_file)

    if use_vtor1_file:
        ion_main.velocity.toroidal = vtor1_from_file
    elif "omeg" in p:
        # omeg in kRad/s: Omega = omeg * 1e3 rad/s
        omeg_kRad = map_to_rho(p["omeg"]["data"])
        omega_rad = omeg_kRad * 1.0e3
        # VTOR = Omega * R_mid
        vtor_m_s = omega_rad * R_mid
        ion_main.velocity.toroidal = vtor_m_s

    # vpol1: if present and nonzero, use directly; else, use kpol or omegp
    vpol1_from_file = None
    if "vpol1" in p:
        vpol1_from_file = map_to_rho(p["vpol1"]["data"]) * 1.0e3  # km/s -> m/s

    use_vpol1_file = vpol1_from_file is not None and not all_zero(vpol1_from_file)

    if use_vpol1_file:
        ion_main.velocity.poloidal = vpol1_from_file
    else:
        vpol_m_s = np.zeros_like(rho)

        # kpol in km/s/T: VPOL = kpol * Bp
        if "kpol" in p and not all_zero(p["kpol"]["data"]):
            kpol_km_s_T = map_to_rho(p["kpol"]["data"])
            vpol_m_s += kpol_km_s_T * 1.0e3 * Bp_mid

        # omegp in kRad/s: omegp = Bt * VPOL / (R * Bp)
        # -> VPOL = omegp * 1e3 * R * Bp / Bt
        if "omegp" in p and not all_zero(p["omegp"]["data"]):
            omegp_kRad = map_to_rho(p["omegp"]["data"])
            omegp_rad  = omegp_kRad * 1.0e3
            Bt_safe    = np.where(np.abs(Bt_mid) > 1e-6, Bt_mid, 1e-6)
            vpol_from_omegp = omegp_rad * R_mid * Bp_mid / Bt_safe
            vpol_m_s += vpol_from_omegp

        ion_main.velocity.poloidal = vpol_m_s

    # ------------------------------------------------------------------
    # beam ion: nb, pb
    # ------------------------------------------------------------------
    if beam_idx is not None:
        ion_beam = ions[beam_idx]
        _ = ion_beam.velocity.toroidal
        _ = ion_beam.velocity.poloidal

        if "nb" in p:
            nb_20 = map_to_rho(p["nb"]["data"])
            ion_beam.density_fast = nb_20 * 1.0e20

        if "pb" in p:
            pb_kPa = map_to_rho(p["pb"]["data"])
            ion_beam.pressure_fast_perpendicular = pb_kPa * 1.0e3 / 3.0

    # ------------------------------------------------------------------
    # impurities: nz1, nz2, vtor2, vpol2, ...
    # ------------------------------------------------------------------
    n_imp = max(nspec - 2, 0)
    for i_imp in range(n_imp):
        ion_imp = ions[i_imp]
        _ = ion_imp.velocity.toroidal
        _ = ion_imp.velocity.poloidal

        k = i_imp + 1  # nz1, nz2, ...

        nz_key   = f"nz{k}"
        vtor_key = f"vtor{k}"
        vpol_key = f"vpol{k}"

        if nz_key in p:
            nz_20 = map_to_rho(p[nz_key]["data"])
            ion_imp.density_thermal = nz_20 * 1.0e20

        if vtor_key in p:
            vtor_kms = map_to_rho(p[vtor_key]["data"])
            ion_imp.velocity.toroidal = vtor_kms * 1.0e3

        if vpol_key in p:
            vpol_kms = map_to_rho(p[vpol_key]["data"])
            ion_imp.velocity.poloidal = vpol_kms * 1.0e3

    # ------------------------------------------------------------------
    # diamagnetic velocity from omgpp (kRad/s) -> v_dia [m/s]
    # ------------------------------------------------------------------
    if "omgpp" in p and not all_zero(p["omgpp"]["data"]):
        omgpp_kRad = map_to_rho(p["omgpp"]["data"])
        omega_dia_rad = omgpp_kRad * 1.0e3  # kRad/s -> rad/s
        R_safe = np.where(np.abs(R_mid) > 1e-6, R_mid, 1e-6)
        v_dia = omega_dia_rad / (2.0 * np.pi) / R_safe

        ion_dia = ions[dia_idx]
        _ = ion_dia.velocity.diamagnetic
        ion_dia.velocity.diamagnetic = v_dia

    return cp_ids, p


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("geqdsk", help="input GEQDSK file")
    parser.add_argument("peqdsk", help="input P-EQDSK (p-file)")

    parser.add_argument("--input", dest="input_yaml", default=None,
                        help="Optional YAML metadata file for summary/dataset_fair/workflow")

    parser.add_argument("--nimeq", help="nimeq.in input file for FGnimeq", default='nimeq.in')
    parser.add_argument("--oculus", help="oculus.in input file for FGnimeq", default='oculus.in')
    parser.add_argument("--fluxgrid", help="fluxgrid.in input file for FGnimeq", default='fluxgrid.in')
    parser.add_argument("--nimrod", help="nimrod.in input file for NIMROD", default='nimrod.in')

    # IMAS entry selection (consistent with dump2imas)
    parser.add_argument("--dd", required=True, help="IMAS database name (directory name)")
    parser.add_argument("--pulse", type=int, required=True, help="IMAS pulse")
    parser.add_argument("--run", type=int, required=True, help="IMAS run")
    parser.add_argument("--backend", choices=["hdf5", "mdsplus"], default="hdf5",
                        help="IMAS backend (default: hdf5)")
    parser.add_argument("--dbpath", default=".", help="DB root path (output directory)")
    parser.add_argument("--dd-version", default=None, help="IMAS DD version (defaults to $IMAS_VERSION)")
    parser.add_argument("--entry", default=None,
                        help="Explicit entry directory override (otherwise use dbpath/dd/dd-version-dir/pulse/run)")
    parser.add_argument("--mode", default="a", help="DBEntry open mode: r/a/w/x")
    parser.add_argument("--no-checksums", dest="record_checksums", action="store_false",
                        help="Disable provenance file checksums in workflow/dataset_fair IDSs.")
    parser.set_defaults(record_checksums=True)
    parser.add_argument("--checksum-algorithm", default="sha256",
                        help="Hash algorithm for provenance checksums (sha256, sha1, md5, ...).")
    parser.add_argument("--occ", type=int, default=0, help="Occurrence to write")
    parser.add_argument("--occ-base", dest="occ", type=int, help="Alias for --occ")

    args = parser.parse_args()

    # Optional dataset/workflow metadata
    meta = load_metadata_yaml(getattr(args, 'input_yaml', None))

    # Resolve DD version early and align IDSFactory
    dd_version = (args.dd_version or os.environ.get('IMAS_VERSION') or '').strip()
    if not dd_version:
        raise SystemExit('ERROR: dd_version is not set. Provide --dd-version or set IMAS_VERSION.')
    global _ids_factory
    _ids_factory = IDSFactory(dd_version)

    # Resolve entry path (match dump2imas layout)
    if args.entry:
        entry_path = os.path.abspath(os.path.expanduser(args.entry))
    else:
        entry_path = str(entry_dir(args.dbpath, args.dd, dd_version, args.pulse, args.run, args.dd_version[0]))

    time0 = 0.0
    mhd_ids = None  # optional NIMROD mhd IDS

    # --- build equilibrium ---
    eq_ids, geq = geqdsk_to_equilibrium(args.geqdsk, time=time0)

    # --- build core_profiles from p-file ---
    cp_ids = _ids_factory.core_profiles()
    cp_ids, pfile = fill_core_profiles_from_pfile(cp_ids, args.peqdsk, geq, time=time0)

    # NOTE on sign conventions / COCOS:
    # input2imas preserves the original signs from the kinetic PEQDSK (p-file) input.
    # PEQDSK files are device/discharge specific and do not necessarily enforce a single
    # global COCOS convention. Any COCOS normalization/sign handling is performed in
    # dump2imas (for NIMROD dump outputs) rather than here.
    if int(getattr(args, "occ", 0) or 0) == 0:
        try:
            msg = (
                "input2imas: core_profiles(occ=0) preserves the original sign conventions "
                "from the input PEQDSK (kinetic profiles); no COCOS sign normalization is applied."
            )
            ip = getattr(cp_ids, "ids_properties", None)
            if ip is not None and hasattr(ip, "comment"):
                prev = str(getattr(ip, "comment", "") or "").strip()
                ip.comment = (prev + "\n" + msg).strip() if prev else msg
        except Exception:
            pass

    # --- attach code metadata and XML inputs ---
    # FGnimeq inputs (nimeq.in / oculus.in / fluxgrid.in) -> equilibrium.code.parameters
    fgnimeq_xml = build_fgnimeq_xml(args.nimeq, args.oculus, args.fluxgrid)
    if fgnimeq_xml:
        try:
            eq_ids.code.name = "fgnimeq"
            eq_ids.code.parameters = fgnimeq_xml
        except Exception as exc:
            print(f"Warning: could not attach FGnimeq XML to equilibrium.code: {exc}")

    # NIMROD inputs (nimrod.in) -> prefer dedicated mhd IDS, fall back to core_profiles.code
    nimrod_xml = build_nimrod_xml(args.nimrod)
    if nimrod_xml:
        try:
            mhd_ids = _ids_factory.mhd()
            mhd_ids.ids_properties.homogeneous_time = 1
            mhd_ids.time = np.array([time0], dtype=float)
            if hasattr(mhd_ids, "time_slice"):
                try:
                    mhd_ids.time_slice.resize(1)
                    mhd_ids.time_slice[0].time = time0
                except Exception:
                    pass
            mhd_ids.code.name = "nimrod"
            mhd_ids.code.parameters = nimrod_xml
        except Exception as exc:
            print(f"Warning: could not populate mhd IDS for NIMROD input: {exc}")
            mhd_ids = None
            try:
                cp_ids.code.name = "nimrod"
                cp_ids.code.parameters = nimrod_xml
            except Exception as exc2:
                print(f"Warning: could not attach NIMROD XML to core_profiles.code: {exc2}")


    # --- create metadata IDSs (summary, dataset_fair, workflow) ---
    # Note: dataset_fair has maximum occurrences=1; these IDSs are written to occurrence 0.
    try:
        summary_ids = build_summary_ids(_ids_factory, args, meta, geq=geq, time0=time0)
    except Exception as exc:
        print(f"Warning: failed to build summary IDS: {exc}")
        summary_ids = None

    try:
        dataset_fair_ids = build_dataset_fair_ids(_ids_factory, meta)
    except Exception as exc:
        print(f"Warning: failed to build dataset_fair IDS: {exc}")
        dataset_fair_ids = None

    try:
        workflow_ids = build_workflow_ids(
            _ids_factory,
            args,
            dd_version,
            meta,
            nimrod_inputs_xml=nimrod_xml,
            fgnimeq_inputs_xml=fgnimeq_xml,
        )
    except Exception as exc:
        print(f"Warning: failed to build workflow IDS: {exc}")
        workflow_ids = None

    # --- build wall from GEQDSK limiter ---
    wall_ids = geqdsk_to_wall(geq, time=time0)

    # --- write to IMAS DBEntry (same directory layout as dump2imas.py) ---

    db, uri, imas_mod = open_dbentry(args.backend, entry_path, mode=args.mode, dd_version=dd_version)
    factory = ids_factory(imas_mod, dd_version)
    print(f"IMAS entry directory: {entry_path}")
    print(f"IMAS URI: {uri}")

    # Write entry-level metadata IDSs to occurrence 0 (DD4.1 expects single occurrence)
    meta_occ = 0
    if summary_ids is not None:
        put_ids(db, summary_ids, meta_occ)
    if dataset_fair_ids is not None:
        put_ids(db, dataset_fair_ids, meta_occ)
    if workflow_ids is not None:
        put_ids(db, workflow_ids, meta_occ)

    put_ids(db, eq_ids, args.occ)
    put_ids(db, cp_ids, args.occ)
    put_ids(db, wall_ids, args.occ)
    if mhd_ids is not None:
        put_ids(db, mhd_ids, args.occ)

    # --- append per-step provenance (workflow + dataset_fair) ---
    try:
        pfiles = []
        for p in [getattr(args, "input_yaml", None), args.geqdsk, args.peqdsk,
                  getattr(args, "nimeq", None), getattr(args, "oculus", None),
                  getattr(args, "fluxgrid", None), getattr(args, "nimrod", None)]:
            if p and os.path.isfile(str(p)):
                pfiles.append(str(p))

        cmd = sanitize_cli_command(list(sys.argv), known_files=pfiles)
        update_workflow_and_dataset_fair(
            db, factory,
            component_name="nimrod2imas:input2imas",
            component_description="Convert NIMROD input files (GEQDSK/PEQDSK + namelists) to IMAS",
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
                "occ": str(int(getattr(args, "occ", 0) or 0)),
                "metadata_yaml": metadata_yaml_path(meta) or "",
                "metadata_identifier": str(yget(meta, "dataset", "identifier", default="") or ""),
                "nimrod_repository": str(yget(meta, "nimrod", "repository", default="") or ""),
                "nimrod_version": str(yget(meta, "nimrod", "version", default="") or ""),
                "converter_repository": str(yget(meta, "converter", "repository", default="") or ""),
                "converter_version": str(yget(meta, "converter", "version", default="") or __version__),
            },
        )
    except Exception as exc:
        print(f"Warning: provenance update failed: {exc}")
    try:
        db.close()
    except Exception:
        pass

    print(f"Saved summary, dataset_fair, workflow, equilibrium, core_profiles, wall (and mhd if present) to IMAS entry: {entry_path} (occ={args.occ})")


if __name__ == "__main__":
    main()
