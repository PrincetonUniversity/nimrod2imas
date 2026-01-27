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
import numpy as np

import imas
from nimrod2imas import entry_dir, open_dbentry, put_ids, value_to_string as _nimrod_value_to_string
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

def all_zero(arr, tol=1e-12):
    """Return True if array is empty or all entries are ~0."""
    a = np.asarray(arr, dtype=float)
    return (a.size == 0) or np.all(np.abs(a) < tol)


def value_to_string(val):
    """Convert Python value to token string for XML (delegates to nimrod2imas.value_to_string)."""
    return _nimrod_value_to_string(val)
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

    psirz = np.asarray(geq["PSIRZ"]).reshape((nh, nw))
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
    nw = int(geq["NW"])
    nh = int(geq["NH"])

    rdim  = float(geq["RDIM"])
    zdim  = float(geq["ZDIM"])
    rleft = float(geq["RLEFT"])
    zmid  = float(geq["ZMID"])

    rgrid = rleft + np.arange(nw) * rdim / (nw - 1)
    zgrid = (zmid - 0.5 * zdim) + np.arange(nh) * zdim / (nh - 1)

    ts.profiles_2d.resize(1)
    p2 = ts.profiles_2d[0]
    p2.grid_type.index = 1  # 1 = rectangular grid
    p2.grid.dim1 = rgrid
    p2.grid.dim2 = zgrid

    psirz = np.asarray(geq["PSIRZ"]).reshape((nh, nw))
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
    # total pressure: ptot in kPa, core_profiles.pressure_perpendicular in Pa
    # we use 3*pressure_perpendicular ~ total pressure
    # ------------------------------------------------------------------
    if "ptot" in p:
        ptot_kPa = map_to_rho(p["ptot"]["data"])
        prof.pressure_perpendicular = ptot_kPa * 1.0e3 / 3.0

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

    parser.add_argument("--nimeq", help="nimeq.in input file for FGnimeq", default='nimeq.in')
    parser.add_argument("--oculus", help="oculus.in input file for FGnimeq", default='oculus.in')
    parser.add_argument("--fluxgrid", help="fluxgrid.in input file for FGnimeq", default='fluxgrid.in')
    parser.add_argument("--nimrod", help="nimrod.in input file for NIMROD", default='nimrod.in')

    # IMAS entry selection (consistent with dump2imas)
    parser.add_argument("--dd", required=True, help="IMAS database name (tokamak name)")
    parser.add_argument("--pulse", type=int, required=True, help="IMAS pulse")
    parser.add_argument("--run", type=int, required=True, help="IMAS run")
    parser.add_argument("--backend", choices=["hdf5", "mdsplus"], default="hdf5",
                        help="IMAS backend (default: hdf5)")
    parser.add_argument("--dbpath", default=".", help="DB root path (output directory)")
    parser.add_argument("--dd-version", default=None, help="IMAS DD version (defaults to $IMAS_VERSION)")
    parser.add_argument("--dd-version-dir", choices=["major","full"], default="major",
                        help="Directory component for DD version (major or full)")
    parser.add_argument("--entry", default=None,
                        help="Explicit entry directory override (otherwise use dbpath/dd/dd-version-dir/pulse/run)")
    parser.add_argument("--mode", default="a", help="DBEntry open mode: r/a/w/x")
    parser.add_argument("--occ", type=int, default=0, help="Occurrence to write")
    parser.add_argument("--occ-base", dest="occ", type=int, help="Alias for --occ")

    args = parser.parse_args()


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
        entry_path = str(entry_dir(args.dbpath, args.dd, dd_version, args.pulse, args.run, args.dd_version_dir))

    time0 = 0.0
    mhd_ids = None  # optional NIMROD mhd IDS

    # --- build equilibrium ---
    eq_ids, geq = geqdsk_to_equilibrium(args.geqdsk, time=time0)

    # --- build core_profiles from p-file ---
    cp_ids = _ids_factory.core_profiles()
    cp_ids, pfile = fill_core_profiles_from_pfile(cp_ids, args.peqdsk, geq, time=time0)

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

    # --- build wall from GEQDSK limiter ---
    wall_ids = geqdsk_to_wall(geq, time=time0)

    # --- write to IMAS DBEntry (same directory layout as dump2imas.py) ---

    db, uri, _ = open_dbentry(args.backend, entry_path, mode=args.mode, dd_version=dd_version)
    print(f"IMAS entry directory: {entry_path}")
    print(f"IMAS URI: {uri}")

    put_ids(db, eq_ids, args.occ)
    put_ids(db, cp_ids, args.occ)
    put_ids(db, wall_ids, args.occ)
    if mhd_ids is not None:
        put_ids(db, mhd_ids, args.occ)

    try:
        db.close()
    except Exception:
        pass

    print(f"Saved equilibrium, core_profiles, wall (and mhd if present) to IMAS entry: {entry_path} (occ={args.occ})")


if __name__ == "__main__":
    main()

