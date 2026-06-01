#!/usr/bin/env python3
"""
nimrod2imas_bp.py

Convert NIMROD input files (GEQDSK + PEQDSK/p-file) to OMAS ODS and save
as ADIOS BP format using EFFIS shim layer.

This is a variant of nimrod2imas.py that:
  - Builds ODS (OMAS Data Structure) directly instead of IMAS IDS
  - Saves to ADIOS BP format via effis.shim.save_omas_adios()

Data structures populated:
  - equilibrium ODS from GEQDSK (using OMFITgeqdsk, no SciPy aux)
  - core_profiles ODS from PEQDSK (using OMFITpFile)
  - wall ODS from GEQDSK limiter outline

Omega handling:
  - If vtor1 is zero/absent, use omeg to reconstruct VTOR:
      VTOR = omeg * R_mid
  - If vpol1 is zero/absent:
      - If kpol nonzero: VPOL = kpol * Bp_mid
      - else if omegp nonzero: VPOL = omegp * R_mid * Bp_mid / Bt_mid
  - Diamagnetic:
      - If omgpp nonzero: v_dia = omgpp * 1e3 / (2*pi*R_mid)
      - Stored in core_profiles.ion[dia_idx].velocity.diamagnetic

This version also stores the NIMROD-related input files as XML using f90nml:
  - nimeq.in, oculus.in, fluxgrid.in  -> equilibrium.code.parameters (code.name='fgnimeq')
  - nimrod.in                         -> mhd.code.parameters (code.name='nimrod')
"""

import argparse
import os
import numpy as np

import omas

import f90nml
import xml.etree.ElementTree as ET

from omfit_classes.omfit_eqdsk import OMFITgeqdsk
from omfit_classes.omfit_osborne import OMFITpFile


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------

def all_zero(arr, tol=1e-12):
    """Return True if array is empty or all entries are ~0."""
    a = np.asarray(arr, dtype=float)
    return (a.size == 0) or np.all(np.abs(a) < tol)


def value_to_string(val):
    """Convert Python value (scalar or list) to a string for XML."""
    import numpy as _np
    if isinstance(val, _np.ndarray):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        parts = [value_to_string(v) for v in val]
        return " ".join(parts)
    if isinstance(val, bool):
        # Fortran-like logical
        return ".true." if val else ".false."
    return str(val)


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
# GEQDSK -> equilibrium ODS + wall ODS
# ----------------------------------------------------------------------

def geqdsk_to_equilibrium_ods(ods, geqdsk_path, time=0.0):
    """
    Read GEQDSK and populate equilibrium entries in ODS using only raw arrays
    (no SciPy-dependent aux from OMFITgeqdsk).
    
    Returns the loaded geq object for use by other functions.
    """
    geq = OMFITgeqdsk(geqdsk_path)
    geq.load(raw=True, add_aux=False)

    # --- time array ---
    ods['equilibrium.time'] = np.array([time], dtype=float)
    ods['equilibrium.ids_properties.homogeneous_time'] = 1

    # time slice prefix
    ts = 'equilibrium.time_slice.0'
    ods[f'{ts}.time'] = time

    # --- 2D grid and psi(R,Z) ---
    nw = int(geq["NW"])
    nh = int(geq["NH"])

    rdim  = float(geq["RDIM"])
    zdim  = float(geq["ZDIM"])
    rleft = float(geq["RLEFT"])
    zmid  = float(geq["ZMID"])

    rgrid = rleft + np.arange(nw) * rdim / (nw - 1)
    zgrid = (zmid - 0.5 * zdim) + np.arange(nh) * zdim / (nh - 1)

    p2 = f'{ts}.profiles_2d.0'
    ods[f'{p2}.grid_type.index'] = 1  # 1 = rectangular grid
    ods[f'{p2}.grid.dim1'] = np.ascontiguousarray(rgrid)
    ods[f'{p2}.grid.dim2'] = np.ascontiguousarray(zgrid)

    psirz = np.asarray(geq["PSIRZ"]).reshape((nh, nw))
    # IMAS: psi(R,Z) with shape (len(R),len(Z)), must be C-contiguous for ADIOS
    ods[f'{p2}.psi'] = np.ascontiguousarray(psirz.T)

    # --- 1D profiles vs psi ---
    p1 = f'{ts}.profiles_1d'
    pres   = np.asarray(geq["PRES"])
    fpol   = np.asarray(geq["FPOL"])
    ffprim = np.asarray(geq["FFPRIM"])
    pprime = np.asarray(geq["PPRIME"])
    qpsi   = np.asarray(geq["QPSI"])

    n_psi = len(pres)
    simag = float(geq["SIMAG"])
    sibry = float(geq["SIBRY"])
    psi_1d = np.linspace(simag, sibry, n_psi)

    ods[f'{p1}.psi']            = psi_1d
    ods[f'{p1}.f']              = fpol
    ods[f'{p1}.pressure']       = pres
    ods[f'{p1}.f_df_dpsi']      = ffprim
    ods[f'{p1}.dpressure_dpsi'] = pprime
    ods[f'{p1}.q']              = qpsi

    # --- global quantities ---
    gq = f'{ts}.global_quantities'
    ods[f'{gq}.ip']              = float(geq["CURRENT"])
    ods[f'{gq}.psi_axis']        = float(geq["SIMAG"])
    ods[f'{gq}.psi_boundary']    = float(geq["SIBRY"])
    ods[f'{gq}.magnetic_axis.r'] = float(geq["RMAXIS"])
    ods[f'{gq}.magnetic_axis.z'] = float(geq["ZMAXIS"])

    # --- boundary ---
    if int(geq["NBBBS"]) > 0:
        ods[f'{ts}.boundary.outline.r'] = np.asarray(geq["RBBBS"])
        ods[f'{ts}.boundary.outline.z'] = np.asarray(geq["ZBBBS"])

    # --- vacuum toroidal field ---
    ods['equilibrium.vacuum_toroidal_field.r0'] = float(geq["RCENTR"])
    ods['equilibrium.vacuum_toroidal_field.b0'] = np.array([float(geq["BCENTR"])])

    return geq


def geqdsk_to_wall_ods(ods, geq, time=0.0):
    """
    Populate wall entries in ODS from GEQDSK limiter outline.
    """
    ods['wall.ids_properties.homogeneous_time'] = 1
    ods['wall.time'] = np.array([time], dtype=float)

    # limiter from RLIM/ZLIM
    if int(geq["LIMITR"]) > 0:
        ods['wall.description_2d.0.limiter.unit.0.outline.r'] = np.asarray(geq["RLIM"])
        ods['wall.description_2d.0.limiter.unit.0.outline.z'] = np.asarray(geq["ZLIM"])


# ----------------------------------------------------------------------
# PEQDSK / p-file -> core_profiles ODS
# ----------------------------------------------------------------------

def fill_core_profiles_ods_from_pfile(ods, pfile_path, geq, time=0.0):
    """
    Fill core_profiles entries in ODS from an Osborne p-file, matching the
    NIMROD / p-file species convention:

      nspec = len(Z) from the "N Z A" block
      main_idx = nspec - 2  (if nspec >= 2 else 0)
      beam_idx = nspec - 1  (if nspec >= 2 else None)
      impurities: indices 0 .. nspec-3

    Returns the loaded pfile object.
    """
    p = OMFITpFile(pfile_path)
    p.load()

    ods['core_profiles.ids_properties.homogeneous_time'] = 1
    ods['core_profiles.time'] = np.array([time], dtype=float)
    
    prof = 'core_profiles.profiles_1d.0'
    ods[f'{prof}.time'] = time

    # --- radial grid: use ne.psinorm if available ---
    if "ne" in p:
        rho = np.array(p["ne"]["psinorm"], dtype=float)
    else:
        # fallback: uniform [0,1]
        nw = int(geq["NW"])
        rho = np.linspace(0.0, 1.0, nw)
    ods[f'{prof}.grid.rho_tor_norm'] = rho
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
        ods[f'{prof}.electrons.density_thermal'] = ne_20 * 1.0e20  # 10^20 -> m^-3

    if "te" in p:
        te_keV = map_to_rho(p["te"]["data"])
        ods[f'{prof}.electrons.temperature'] = te_keV * 1.0e3  # keV -> eV

    # ------------------------------------------------------------------
    # total pressure: ptot in kPa, core_profiles.pressure_perpendicular in Pa
    # we use 3*pressure_perpendicular ~ total pressure
    # ------------------------------------------------------------------
    if "ptot" in p:
        ptot_kPa = map_to_rho(p["ptot"]["data"])
        ods[f'{prof}.pressure_perpendicular'] = ptot_kPa * 1.0e3 / 3.0

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

    # Set up ion species
    for k in range(nspec):
        ion = f'{prof}.ion.{k}'
        ods[f'{ion}.element.0.z_n'] = float(Z_arr[k])
        ods[f'{ion}.element.0.a']   = float(A_arr[k])
        try:
            ods[f'{ion}.element.0.label'] = f"Z{int(round(Z_arr[k]))}A{A_arr[k]:.4g}"
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

    ion_main = f'{prof}.ion.{main_idx}'

    # ------------------------------------------------------------------
    # main ion: ni, ti, vtor1, vpol1, omeg/kpol/omegp
    # ------------------------------------------------------------------
    if "ni" in p:
        ni_20 = map_to_rho(p["ni"]["data"])
        ods[f'{ion_main}.density_thermal'] = ni_20 * 1.0e20

    if "ti" in p:
        ti_keV = map_to_rho(p["ti"]["data"])
        ods[f'{ion_main}.temperature'] = ti_keV * 1.0e3  # keV -> eV

    # vtor1: if present and nonzero, use directly; otherwise, reconstruct from omeg
    vtor1_from_file = None
    if "vtor1" in p:
        vtor1_from_file = map_to_rho(p["vtor1"]["data"]) * 1.0e3  # km/s -> m/s

    use_vtor1_file = vtor1_from_file is not None and not all_zero(vtor1_from_file)

    if use_vtor1_file:
        ods[f'{ion_main}.velocity.toroidal'] = vtor1_from_file
    elif "omeg" in p:
        # omeg in kRad/s: Omega = omeg * 1e3 rad/s
        omeg_kRad = map_to_rho(p["omeg"]["data"])
        omega_rad = omeg_kRad * 1.0e3
        # VTOR = Omega * R_mid
        vtor_m_s = omega_rad * R_mid
        ods[f'{ion_main}.velocity.toroidal'] = vtor_m_s

    # vpol1: if present and nonzero, use directly; else, use kpol or omegp
    vpol1_from_file = None
    if "vpol1" in p:
        vpol1_from_file = map_to_rho(p["vpol1"]["data"]) * 1.0e3  # km/s -> m/s

    use_vpol1_file = vpol1_from_file is not None and not all_zero(vpol1_from_file)

    if use_vpol1_file:
        ods[f'{ion_main}.velocity.poloidal'] = vpol1_from_file
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

        ods[f'{ion_main}.velocity.poloidal'] = vpol_m_s

    # ------------------------------------------------------------------
    # beam ion: nb, pb
    # ------------------------------------------------------------------
    if beam_idx is not None:
        ion_beam = f'{prof}.ion.{beam_idx}'

        if "nb" in p:
            nb_20 = map_to_rho(p["nb"]["data"])
            ods[f'{ion_beam}.density_fast'] = nb_20 * 1.0e20

        if "pb" in p:
            pb_kPa = map_to_rho(p["pb"]["data"])
            ods[f'{ion_beam}.pressure_fast_perpendicular'] = pb_kPa * 1.0e3 / 3.0

    # ------------------------------------------------------------------
    # impurities: nz1, nz2, vtor2, vpol2, ...
    # ------------------------------------------------------------------
    n_imp = max(nspec - 2, 0)
    for i_imp in range(n_imp):
        ion_imp = f'{prof}.ion.{i_imp}'
        k = i_imp + 1  # nz1, nz2, ...

        nz_key   = f"nz{k}"
        vtor_key = f"vtor{k}"
        vpol_key = f"vpol{k}"

        if nz_key in p:
            nz_20 = map_to_rho(p[nz_key]["data"])
            ods[f'{ion_imp}.density_thermal'] = nz_20 * 1.0e20

        if vtor_key in p:
            vtor_kms = map_to_rho(p[vtor_key]["data"])
            ods[f'{ion_imp}.velocity.toroidal'] = vtor_kms * 1.0e3

        if vpol_key in p:
            vpol_kms = map_to_rho(p[vpol_key]["data"])
            ods[f'{ion_imp}.velocity.poloidal'] = vpol_kms * 1.0e3

    # ------------------------------------------------------------------
    # diamagnetic velocity from omgpp (kRad/s) -> v_dia [m/s]
    # ------------------------------------------------------------------
    if "omgpp" in p and not all_zero(p["omgpp"]["data"]):
        omgpp_kRad = map_to_rho(p["omgpp"]["data"])
        omega_dia_rad = omgpp_kRad * 1.0e3  # kRad/s -> rad/s
        R_safe = np.where(np.abs(R_mid) > 1e-6, R_mid, 1e-6)
        v_dia = omega_dia_rad / (2.0 * np.pi) / R_safe

        ion_dia = f'{prof}.ion.{dia_idx}'
        ods[f'{ion_dia}.velocity.diamagnetic'] = v_dia

    return p


# ----------------------------------------------------------------------
# MHD IDS for NIMROD inputs
# ----------------------------------------------------------------------

def fill_mhd_ods(ods, nimrod_xml, time=0.0):
    """
    Populate mhd entries in ODS with NIMROD input parameters.
    """
    if not nimrod_xml:
        return False
    
    try:
        ods['mhd.ids_properties.homogeneous_time'] = 1
        ods['mhd.time'] = np.array([time], dtype=float)
        ods['mhd.time_slice.0.time'] = time
        ods['mhd.code.name'] = "nimrod"
        ods['mhd.code.parameters'] = nimrod_xml
        return True
    except Exception as exc:
        print(f"Warning: could not populate mhd ODS for NIMROD input: {exc}")
        return False


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert NIMROD input files to OMAS ODS and save as ADIOS BP format"
    )
    parser.add_argument("geqdsk", help="input GEQDSK file")
    parser.add_argument("peqdsk", help="input P-EQDSK (p-file)")

    parser.add_argument("--nimeq", help="nimeq.in input file for FGnimeq", default='nimeq.in')
    parser.add_argument("--oculus", help="oculus.in input file for FGnimeq", default='oculus.in')
    parser.add_argument("--fluxgrid", help="fluxgrid.in input file for FGnimeq", default='fluxgrid.in')
    parser.add_argument("--nimrod", help="nimrod.in input file for NIMROD", default='nimrod.in')

    parser.add_argument("--output", "-o", default="imas.bp",
                        help="Output BP file path (default: imas.bp)")
    parser.add_argument("--consistency-check", action="store_true",
                        help="Enable OMAS consistency checking (default: disabled)")

    args = parser.parse_args()

    time0 = 0.0

    # Create ODS with optional consistency checking
    ods = omas.ODS(consistency_check=args.consistency_check)

    # --- build equilibrium ---
    print(f"Loading GEQDSK from {args.geqdsk}...")
    geq = geqdsk_to_equilibrium_ods(ods, args.geqdsk, time=time0)

    # --- build core_profiles from p-file ---
    print(f"Loading p-file from {args.peqdsk}...")
    pfile = fill_core_profiles_ods_from_pfile(ods, args.peqdsk, geq, time=time0)

    # --- build wall from GEQDSK limiter ---
    print("Building wall from GEQDSK limiter...")
    geqdsk_to_wall_ods(ods, geq, time=time0)

    # --- attach code metadata and XML inputs ---
    # FGnimeq inputs (nimeq.in / oculus.in / fluxgrid.in) -> equilibrium.code.parameters
    fgnimeq_xml = build_fgnimeq_xml(args.nimeq, args.oculus, args.fluxgrid)
    if fgnimeq_xml:
        try:
            ods['equilibrium.code.name'] = "fgnimeq"
            ods['equilibrium.code.parameters'] = fgnimeq_xml
        except Exception as exc:
            print(f"Warning: could not attach FGnimeq XML to equilibrium.code: {exc}")

    # NIMROD inputs (nimrod.in) -> prefer dedicated mhd IDS, fall back to core_profiles.code
    nimrod_xml = build_nimrod_xml(args.nimrod)
    mhd_success = fill_mhd_ods(ods, nimrod_xml, time=time0)
    
    if nimrod_xml and not mhd_success:
        try:
            ods['core_profiles.code.name'] = "nimrod"
            ods['core_profiles.code.parameters'] = nimrod_xml
        except Exception as exc:
            print(f"Warning: could not attach NIMROD XML to core_profiles.code: {exc}")

    # --- Save to ADIOS BP format using EFFIS shim ---
    print(f"Saving ODS to ADIOS BP format: {args.output}...")
    try:
        import effis.shim
        effis.shim.save_omas_adios(ods, args.output)
        print(f"Successfully saved to {args.output}")
    except ImportError as exc:
        print(f"Error: Could not import effis.shim: {exc}")
        print("Please ensure EFFIS is installed: pip install effis")
        print("Or install from: https://github.com/suchyta1/effis")
        return 1
    except Exception as exc:
        print(f"Error saving to ADIOS BP format: {exc}")
        return 1

    print(f"Saved equilibrium, core_profiles, wall (and mhd if present) to ADIOS BP: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
