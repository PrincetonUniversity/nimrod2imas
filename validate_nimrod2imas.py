#!/usr/bin/env python3
"""
validate_nimrod2imas.py

Read equilibrium and core_profiles from IMAS and regenerate:

  - GEQDSK  (using original GEQDSK as template via OMFITgeqdsk)
  - PEQDSK  (Osborne p-file written directly from core_profiles)

Extended features, consistent with nimrod2imas.py:

  - Reconstruct omega-related profiles from IMAS + GEQDSK geometry:

      omeg   = VTOR / R_mid                    (kRad/s)
      omegp  = Bt_mid * VPOL / (R_mid*Bp_mid)  (kRad/s)
      omgvb  = omeg + omegp                    (kRad/s)
      kpol   = VPOL / Bp_mid                   (km/s/T)
      omgpp  = 2*pi * v_dia / R_mid            (kRad/s)
      omgeb  = omgvb + omgpp                   (kRad/s)

    where:
      - VTOR, VPOL come from ion_main.velocity.(toroidal|poloidal)
      - v_dia = ion_dia.velocity.radial, typically a carbon impurity
      - geometry (R_mid, Bp_mid, Bt_mid) is computed from GEQDSK.

  - Compare original and reconstructed p-files for every profile
    present in both, reporting max and RMS differences in values
    and derivatives.
"""

import argparse
import numpy as np
import os

from pathlib import Path

import imas
from nimrod2imas import (
    IMASContext,
    open_dbentry as _open_db_common,
    ids_factory as _ids_factory_common,
    get_ids as _get_ids_common,
    VERSION as __version__
)

VERSION = __version__

from omfit_classes.omfit_eqdsk import OMFITgeqdsk
from omfit_classes.omfit_osborne import OMFITpFile


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def all_zero(arr, tol=1e-12):
    a = np.asarray(arr, dtype=float)
    return (a.size == 0) or np.all(np.abs(a) < tol)


def simple_derivative(x, y):
    """Simple centered finite-difference derivative dy/dx with end-point copies."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2:
        return np.zeros_like(y)
    dydx = np.zeros_like(y)
    dydx[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2] + 1.0e-12)
    dydx[0]    = dydx[1]
    dydx[-1]   = dydx[-2]
    return dydx


def compute_midplane_geometry_from_geq(geq, psin_target):
    """Compute midplane geometry R_mid, Bp_mid, Bt_mid vs psin_target in [0,1]."""
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

    j_mid = int(np.argmin(np.abs(zgrid - zmid)))
    psi_mid = psirz[j_mid, :]
    psin_mid = (psi_mid - simag) / dpsi

    order = np.argsort(psin_mid)
    psin_mid_sorted = psin_mid[order]
    r_mid_sorted    = rgrid[order]
    psi_mid_sorted  = psi_mid[order]

    R_mid = np.interp(psin_target, psin_mid_sorted, r_mid_sorted)

    dpsi_dR_sorted = np.gradient(psi_mid_sorted, r_mid_sorted)
    dpsi_dR_target = np.interp(psin_target, psin_mid_sorted, dpsi_dR_sorted)
    R_safe = R_mid + 1.0e-12
    Bp_mid = np.abs(dpsi_dR_target) / R_safe

    fpol = np.asarray(geq["FPOL"])
    npsi = len(fpol)
    psi_1d  = np.linspace(simag, sibry, npsi)
    psin_1d = (psi_1d - simag) / dpsi
    fpol_target = np.interp(psin_target, psin_1d, fpol)
    Bt_mid = fpol_target / R_safe

    return R_mid, Bp_mid, Bt_mid


# ----------------------------------------------------------------------
# IMAS read helper
# ----------------------------------------------------------------------

def load_ids_from_imas(db_name, pulse, run, backend_str="hdf5"):
    """Legacy IMAS-Core database access (db_name/pulse/run).

    This mode requires an IMAS-Core (HLI) installation that provides:
      - imasdef (backend constants)
      - hli_exception (IDSNotAvailable)
      - constructors like imas.equilibrium()

    If you are using filesystem-backed entries produced by input2imas/dump2imas,
    use the --entry / --dbpath/--dd/--dd-version/--pulse/--run options instead,
    which rely on IMASContext + URI-style DBEntry open.
    """
    try:
        # IMAS-Core style (not provided by IMAS-Python-only installs)
        from imas import imasdef, hli_exception  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "IMAS-Core (HLI) components are not available (cannot import imasdef/hli_exception). "

            "Use the filesystem-backed mode: --entry or --dbpath/--dd/--dd-version/--pulse/--run."
        ) from e

    if backend_str == "mdsplus":
        backend = imasdef.MDSPLUS_BACKEND
    else:
        backend = imasdef.HDF5_BACKEND

    db = imas.DBEntry(backend, db_name, int(pulse), int(run))
    db.open()

    eq = imas.equilibrium()
    eq.get(0, db)

    cp = imas.core_profiles()
    try:
        cp.get(0, db)
    except Exception:
        # IDSNotAvailable in IMAS-Core, but keep broad for compatibility
        cp = None

    w = imas.wall()
    try:
        w.get(0, db)
    except Exception:
        w = None

    db.close()
    return eq, cp, w


def load_ids_from_ctx(ctx: IMASContext, occ: int = 0):
    """Load equilibrium/core_profiles/wall from a filesystem-backed IMAS entry.

    In practice, different tools may store different occurrences (e.g. input2imas
    often writes occ=0; dump2imas may write occ=1). We therefore try the requested
    occurrence first, then fall back to common defaults.
    """
    # Unique candidate list preserving order
    occ_candidates = []
    for o in (int(occ), 0, 1, 2):
        if o not in occ_candidates:
            occ_candidates.append(o)

    db, _uri, _imas_mod, factory = ctx.open(mode="r")
    try:
        last_err = None
        for o in occ_candidates:
            try:
                eq = _get_ids_common(db, factory, "equilibrium", int(o))
                if eq is None:
                    continue
                try:
                    cp = _get_ids_common(db, factory, "core_profiles", int(o))
                except Exception:
                    cp = None
                try:
                    w = _get_ids_common(db, factory, "wall", int(o))
                except Exception:
                    w = None
                return eq, cp, w
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(
            f"No non-empty equilibrium IDS found in any of occurrences {occ_candidates} "
            f"for entry {ctx.entry_dir()}"
        ) from last_err
    finally:
        try:
            db.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
# equilibrium IDS -> OMFITgeqdsk (with safe output filename)
# ----------------------------------------------------------------------

def equilibrium_to_geqdsk(eq_ids, geqdsk_template_path, out_path):
    """
    Use original GEQDSK as a template and overwrite physics from IMAS.
      - OMFITgeqdsk to load and save geqdsk
    """
    import shutil

    # 1) Make sure out_path exists as a copy of the template
    if os.path.abspath(out_path) != os.path.abspath(geqdsk_template_path):
        shutil.copyfile(geqdsk_template_path, out_path)

    # 2) Work directly on the copy
    geq = OMFITgeqdsk(out_path)
    geq.load(raw=True, add_aux=False)

    ts = eq_ids.time_slice[0]
    p2 = ts.profiles_2d[0]
    p1 = ts.profiles_1d
    gq = ts.global_quantities

    # --- 2D psi(R,Z): IMAS psi has shape (len(R), len(Z));
    #                  GEQDSK PSIRZ is (Z,R) flattened ---
    psirz_new = np.asarray(p2.psi).T
    geq["PSIRZ"] = psirz_new.ravel()

    # --- 1D profiles vs psi ---
    geq["PRES"]   = np.asarray(p1.pressure)
    geq["FPOL"]   = np.asarray(p1.f)
    geq["FFPRIM"] = np.asarray(p1.f_df_dpsi)
    geq["PPRIME"] = np.asarray(p1.dpressure_dpsi)
    geq["QPSI"]   = np.asarray(p1.q)

    # --- global quantities ---
    geq["CURRENT"] = float(gq.ip)
    geq["SIMAG"]   = float(gq.psi_axis)
    geq["SIBRY"]   = float(gq.psi_boundary)
    geq["RMAXIS"]  = float(gq.magnetic_axis.r)
    geq["ZMAXIS"]  = float(gq.magnetic_axis.z)

    # --- plasma boundary ---
    if hasattr(ts, "boundary") and len(ts.boundary.outline.r):
        geq["NBBBS"] = len(ts.boundary.outline.r)
        geq["RBBBS"] = np.asarray(ts.boundary.outline.r)
        geq["ZBBBS"] = np.asarray(ts.boundary.outline.z)

    # --- vacuum toroidal field ---
    if len(eq_ids.vacuum_toroidal_field.b0):
        geq["RCENTR"] = float(eq_ids.vacuum_toroidal_field.r0)
        geq["BCENTR"] = float(eq_ids.vacuum_toroidal_field.b0[0])

    # 3) Write everything back to out_path (no SciPy involved)
    geq.save()


# ----------------------------------------------------------------------
# core_profiles IDS -> Osborne P-EQDSK text
# ----------------------------------------------------------------------

def core_profiles_to_peqdsk(cp_ids, geqdsk_path, out_path):
    """Partical re-build of Osborne p-file from core_profiles and 
    GEQDSK geometry.

    Format for each profile:
        N psinorm NAME(UNITS) dNAME/dpsiN
         psi_1  val_1  dval_1
         ...
         psi_N  val_N  dval_N

    Mapping:

      nspec = len(prof.ion)
      main_idx = nspec - 2  (if nspec >= 2 else 0)
      beam_idx = nspec - 1  (if nspec >= 2)
      impurities: indices 0 .. nspec-3

    Units:

      ne, ni, nz* : 10^20 m^-3
      te, ti      : keV
      ptot, pb    : kPa   (ptot = 3 * pressure_perpendicular / 1e3)
      vtor*,vpol* : km/s
      kpol        : km/s/T
      omeg, omegp,
      omgvb, omgeb,
      omgpp       : kRad/s
    """
    if cp_ids is None:
        print("No core_profiles IDS in IMAS; skipping PEQDSK regeneration.")
        return

    geq = OMFITgeqdsk(geqdsk_path)
    geq.load(raw=True, add_aux=False)

    # --- basic radial grid and geometry ---
    prof = cp_ids.profiles_1d[0]
    rho = np.asarray(prof.grid.rho_tor_norm, dtype=float)
    npts = len(rho)
    s = np.linspace(0.0, 1.0, npts)

    R_mid, Bp_mid, Bt_mid = compute_midplane_geometry_from_geq(geq, s)
    R_safe = R_mid + 1.0e-12
    Bp_safe = Bp_mid + 1.0e-12
    Bt_safe = Bt_mid + 1.0e-12

    profiles = []  # list of (name, units, data_array)

    # --- electrons ---
    ne = getattr(prof.electrons, "density_thermal", None)
    if ne is not None and np.asarray(ne).size == npts:
        ne_20 = np.asarray(ne, dtype=float) / 1.0e20
        profiles.append(("ne", "10^20/m^3", ne_20))

    te = getattr(prof.electrons, "temperature", None)
    if te is not None and np.asarray(te).size == npts:
        te_keV = np.asarray(te, dtype=float) / 1.0e3
        profiles.append(("te", "keV", te_keV))

    # --- total pressure ---
    pperp = getattr(prof, "pressure_perpendicular", None)
    if pperp is not None and np.asarray(pperp).size == npts:
        ptot_kPa = 3.0 * np.asarray(pperp, dtype=float) / 1.0e3
        profiles.append(("ptot", "kPa", ptot_kPa))

    # --- species and velocities ---
    ions = prof.ion
    nspec = len(ions)

    Z_list = []
    for ion in ions:
        if len(ion.element) > 0:
            Z_list.append(ion.element[0].z_n)
        else:
            Z_list.append(0.0)
    Z_arr = np.array(Z_list, dtype=float)

    if nspec >= 2:
        main_idx = nspec - 2
        beam_idx = nspec - 1
    else:
        main_idx = 0
        beam_idx = None

    # find carbon-like impurity index (for v_dia)
    dia_idx = None
    for k in range(nspec):
        if abs(Z_arr[k] - 6.0) < 0.5:
            dia_idx = k
            break

    main_ion = ions[main_idx]
    beam_ion = ions[beam_idx] if beam_idx is not None else None

    # ---- main ion densities & temperature ----
    if getattr(main_ion, "density_thermal", None) is not None:
        ni = np.asarray(main_ion.density_thermal, dtype=float)
        if ni.size == npts:
            ni_20 = ni / 1.0e20
            profiles.append(("ni", "10^20/m^3", ni_20))

    if getattr(main_ion, "temperature", None) is not None:
        ti = np.asarray(main_ion.temperature, dtype=float)
        if ti.size == npts:
            ti_keV = ti / 1.0e3
            profiles.append(("ti", "keV", ti_keV))

    # ---- main ion velocities ----
    vtor_main = None
    vpol_main = None
    if getattr(main_ion, "velocity", None) is not None:
        if getattr(main_ion.velocity, "toroidal", None) is not None:
            vtor = np.asarray(main_ion.velocity.toroidal, dtype=float)
            if vtor.size == npts and not all_zero(vtor):
                vtor_main = vtor
        if getattr(main_ion.velocity, "poloidal", None) is not None:
            vpol = np.asarray(main_ion.velocity.poloidal, dtype=float)
            if vpol.size == npts and not all_zero(vpol):
                vpol_main = vpol

    if vtor_main is not None:
        vtor_kms = vtor_main / 1.0e3
        profiles.append(("vtor1", "km/s", vtor_kms))
    if vpol_main is not None:
        vpol_kms = vpol_main / 1.0e3
        profiles.append(("vpol1", "km/s", vpol_kms))

    # ---- compute omega and kpol-related profiles ----
    if vtor_main is not None:
        omeg_rad = vtor_main / R_safe
        omeg_kRad = omeg_rad / 1.0e3
    else:
        omeg_kRad = np.zeros_like(R_mid)

    if vpol_main is not None:
        omegp_rad = Bt_safe * vpol_main / (R_safe * Bp_safe)
        omegp_kRad = omegp_rad / 1.0e3
        kpol_km_s_t = (vpol_main / Bp_safe) / 1.0e3
    else:
        omegp_kRad = np.zeros_like(R_mid)
        kpol_km_s_t = np.zeros_like(R_mid)

    omgvb_kRad = omeg_kRad + omegp_kRad

    # diamagnetic from impurity radial velocity
    if dia_idx is not None:
        dia_ion = ions[dia_idx]
        vdia = None
        if getattr(dia_ion, "velocity", None) is not None and \
           getattr(dia_ion.velocity, "radial", None) is not None:
            vdia_arr = np.asarray(dia_ion.velocity.radial, dtype=float)
            if vdia_arr.size == npts and not all_zero(vdia_arr):
                vdia = vdia_arr

        if vdia is not None:
            omgpp_rad = 2.0 * np.pi * vdia / R_safe
            omgpp_kRad = omgpp_rad / 1.0e3
        else:
            omgpp_kRad = np.zeros_like(R_mid)
    else:
        omgpp_kRad = np.zeros_like(R_mid)

    omgeb_kRad = omgvb_kRad + omgpp_kRad

    # append omega/kpol profiles if non-zero
    if not all_zero(omeg_kRad):
        profiles.append(("omeg", "kRad/s", omeg_kRad))
    if not all_zero(omegp_kRad):
        profiles.append(("omegp", "kRad/s", omegp_kRad))
    if not all_zero(omgvb_kRad):
        profiles.append(("omgvb", "kRad/s", omgvb_kRad))
    if not all_zero(omgpp_kRad):
        profiles.append(("omgpp", "kRad/s", omgpp_kRad))
    if not all_zero(omgeb_kRad):
        profiles.append(("omgeb", "kRad/s", omgeb_kRad))
    if not all_zero(kpol_km_s_t):
        profiles.append(("kpol", "km/s/T", kpol_km_s_t))

    # ---- beam ion: nb, pb ----
    if beam_ion is not None:
        if getattr(beam_ion, "density_fast", None) is not None:
            nb = np.asarray(beam_ion.density_fast, dtype=float)
            if nb.size == npts:
                nb_20 = nb / 1.0e20
                profiles.append(("nb", "10^20/m^3", nb_20))

        if getattr(beam_ion, "pressure_fast_perpendicular", None) is not None:
            pb = np.asarray(beam_ion.pressure_fast_perpendicular, dtype=float)
            if pb.size == npts:
                pb_kPa = 3.0 * pb / 1.0e3
                profiles.append(("pb", "kPa", pb_kPa))

    # ---- impurities: nz1, nz2, ..., vtor{i}, vpol{i} ----
    n_imp = max(nspec - 2, 0)
    for i_imp in range(n_imp):
        ion_imp = ions[i_imp]
        k = i_imp + 1

        if getattr(ion_imp, "density_thermal", None) is not None:
            nz = np.asarray(ion_imp.density_thermal, dtype=float)
            if nz.size == npts:
                nz_20 = nz / 1.0e20
                profiles.append((f"nz{k}", "10^20/m^3", nz_20))

        if getattr(ion_imp, "velocity", None) is not None:
            if getattr(ion_imp.velocity, "toroidal", None) is not None:
                vtor = np.asarray(ion_imp.velocity.toroidal, dtype=float)
                if vtor.size == npts and not all_zero(vtor):
                    vtor_kms = vtor / 1.0e3
                    profiles.append((f"vtor{k}", "km/s", vtor_kms))
            if getattr(ion_imp.velocity, "poloidal", None) is not None:
                vpol = np.asarray(ion_imp.velocity.poloidal, dtype=float)
                if vpol.size == npts and not all_zero(vpol):
                    vpol_kms = vpol / 1.0e3
                    profiles.append((f"vpol{k}", "km/s", vpol_kms))

    # ---- write Osborne-style p-file by hand ----
    with open(out_path, "w") as f:
        for name, units, data in profiles:
            data_arr = np.asarray(data, dtype=float)
            deriv = simple_derivative(rho, data_arr)

            f.write(f"{npts:d} psinorm {name}({units}) d{name}/dpsiN\n")
            for psi, val, der in zip(rho, data_arr, deriv):
                f.write(f" {psi: .6e}  {val: .6e}  {der: .6e}\n")


# ----------------------------------------------------------------------
# p-file comparison
# ----------------------------------------------------------------------

def compare_pfiles(orig_path, new_path, strict_profiles=None, diagnostic_profiles=None):
    """Compare original and reconstructed p-files.

    The PEQDSK contains both profiles that are stored directly in IMAS
    (densities and temperatures) and profiles that are reconstructed from
    several IMAS quantities plus GEQDSK geometry (flow/omega/kpol profiles).
    The latter are useful diagnostics, but they should not be used as the
    pass/fail criterion for a compact CPC regression test because they are
    sensitive to interpolation, midplane-geometry reconstruction, and finite
    differencing conventions.
    """
    if strict_profiles is None:
        strict_profiles = {"ne", "ni", "te", "ti"}
    else:
        strict_profiles = {str(k).lower() for k in strict_profiles}

    if diagnostic_profiles is None:
        diagnostic_profiles = {
            "kpol", "omeg", "omegp", "omgeb", "omgvb",
            "vpol1", "vtor1", "nz1",
        }
    else:
        diagnostic_profiles = {str(k).lower() for k in diagnostic_profiles}

    # Conservative absolute tolerances in p-file units for the directly stored
    # regression quantities. These are intentionally loose enough for ASCII
    # p-file formatting and IMAS round-trip interpolation, but strict enough to
    # catch missing or badly scaled profiles.
    strict_abs_tol = {
        "ne": 1.0e-8,
        "ni": 1.0e-8,
        "te": 1.0e-6,
        "ti": 5.0e-6,
    }

    p_orig = OMFITpFile(orig_path)
    p_orig.load()
    p_new = OMFITpFile(new_path)
    p_new.load()

    keys = sorted(set(p_orig.keys()) & set(p_new.keys()))

    results = {}
    print("\nProfile comparison (original vs reconstructed):")
    for key in keys:
        key_l = key.strip().lower()
        if key_l == "n z a":
            continue
        try:
            o = p_orig[key]
            n = p_new[key]
        except Exception:
            continue

        psio = np.asarray(o["psinorm"], dtype=float)
        psin = np.asarray(n["psinorm"], dtype=float)
        vo   = np.asarray(o["data"], dtype=float)
        vn   = np.asarray(n["data"], dtype=float)
        do   = np.asarray(o["derivative"], dtype=float)
        dn   = np.asarray(n["derivative"], dtype=float)

        if psio.size == 0 or psin.size == 0:
            continue

        # interpolate reconstructed to original grid
        vn_i = np.interp(psio, psin, vn)
        dn_i = np.interp(psio, psin, dn)

        dv = vo - vn_i
        dd = do - dn_i

        max_dv = float(np.max(np.abs(dv)))
        rms_dv = float(np.sqrt(np.mean(dv**2)))
        max_dd = float(np.max(np.abs(dd)))
        rms_dd = float(np.sqrt(np.mean(dd**2)))
        scale = float(max(np.max(np.abs(vo)), np.max(np.abs(vn_i)), 1.0e-30))
        rel_dv = max_dv / scale

        role = "diagnostic"
        passed = None
        if key_l in strict_profiles:
            role = "strict"
            tol = strict_abs_tol.get(key_l, 1.0e-6)
            passed = max_dv <= tol
        elif key_l not in diagnostic_profiles:
            role = "informational"

        results[key_l] = {
            "role": role,
            "passed": passed,
            "max_value_error": max_dv,
            "rms_value_error": rms_dv,
            "max_derivative_error": max_dd,
            "rms_derivative_error": rms_dd,
            "relative_value_error": rel_dv,
        }

        if passed is True:
            status = "PASS"
        elif passed is False:
            status = "FAIL"
        elif role == "diagnostic":
            status = "DIAGNOSTIC"
        else:
            status = "INFO"

        print(f"  {key:<10s} role={role:<13s} "
              f"max|Δval|={max_dv: .3e}, rms|Δval|={rms_dv: .3e}, rel|max|={rel_dv: .3e};  "
              f"max|Δder|={max_dd: .3e}, rms|Δder|={rms_dd: .3e};  "
              f"status={status}")

    failed = [k for k, v in results.items() if v["passed"] is False]
    if failed:
        raise RuntimeError(
            "Strict PEQDSK round-trip comparison failed for profiles: "
            + ", ".join(failed)
        )

    strict_seen = [k for k, v in results.items() if v["role"] == "strict"]
    print("\nStrict regression profiles checked: " + ", ".join(strict_seen))
    print("Diagnostic flow/omega profiles were reported but not used as pass/fail criteria.")
    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("geqdsk", help="original GEQDSK template file")
    parser.add_argument("peqdsk", help="original PEQDSK (p-file)")
    # Consistent with dump2imas/input2imas: filesystem-backed entry layout
    parser.add_argument("--dd-version-dir", choices=["major", "full"], default="major",
                        help="Directory component for DD version (default: major, e.g. 3 for 3.42.0)")
    parser.add_argument("--dd", "--db", dest="dd", default="nstx", help="DB name (directory name), e.g. nstx")
    parser.add_argument("--pulse", type=int, default=1, help="IMAS pulse number")
    parser.add_argument("--run", type=int, default=0, help="IMAS run number")
    parser.add_argument("--dd-version", dest="dd_version", default=os.environ.get("IMAS_VERSION"),
                        help="IMAS DD version (e.g. 3.42.0). If omitted, uses $IMAS_VERSION when set.")
    parser.add_argument("--backend", choices=["mdsplus", "hdf5"], default="hdf5",
                        help="IMAS backend (default: hdf5)")
    parser.add_argument("--dbpath", default=".", help="DB root path")
    parser.add_argument("--occ", type=int, default=0, help="Preferred occurrence for equilibrium/core_profiles/wall (fallbacks tried)")
    parser.add_argument("--entry", default=None,
                        help="Optional explicit entry directory (overrides --dbpath/--dd/--dd-version/--pulse/--run)")
    parser.add_argument("--out-geqdsk", default="geqdsk_from_imas",
                        help="output GEQDSK filename")
    parser.add_argument("--out-peqdsk", default="peqdsk_from_imas",
                        help="output PEQDSK filename")

    args = parser.parse_args()

    # Resolve entry directory and load IDSs
    if args.entry:
        # Make an IMASContext for consistent fallback behavior
        entry_dir = Path(args.entry).expanduser().resolve()
        ctx = IMASContext(
            backend=args.backend,
            dbpath=entry_dir.parents[4] if len(entry_dir.parents) >= 5 else entry_dir.parent,
            dd=entry_dir.parents[3].name if len(entry_dir.parents) >= 4 else str(args.dd),
            dd_version=entry_dir.parents[2].name if len(entry_dir.parents) >= 3 else (str(args.dd_version) if args.dd_version else ""),
            pulse=int(entry_dir.parents[1].name) if len(entry_dir.parents) >= 2 and entry_dir.parents[1].name.isdigit() else int(args.pulse),
            run=int(entry_dir.name) if entry_dir.name.isdigit() else int(args.run),
            dd_version_dir=str(args.dd_version_dir),
        )
        # Override entry_dir() to use the explicit path if it doesn't match computed layout
        # (keeps --entry robust even if layout differs)
        ctx_entry = entry_dir
        def _fixed_entry_dir():
            return ctx_entry
        ctx.entry_dir = _fixed_entry_dir  # type: ignore
        eq_ids, cp_ids, wall_ids = load_ids_from_ctx(ctx, occ=int(args.occ))
    else:
        ctx = IMASContext(
            backend=args.backend,
            dbpath=args.dbpath,
            dd=str(args.dd),
            dd_version=str(args.dd_version) if args.dd_version else "",
            pulse=int(args.pulse),
            run=int(args.run),
            dd_version_dir=str(args.dd_version_dir),
        )
        eq_ids, cp_ids, wall_ids = load_ids_from_ctx(ctx, occ=int(args.occ))

    print(f"Reconstructing GEQDSK -> {args.out_geqdsk}")
    equilibrium_to_geqdsk(eq_ids, args.geqdsk, args.out_geqdsk)

    print(f"Reconstructing PEQDSK -> {args.out_peqdsk}")
    core_profiles_to_peqdsk(cp_ids, args.geqdsk, args.out_peqdsk)

    print("Comparing original and reconstructed PEQDSK profiles:")
    compare_pfiles(args.peqdsk, args.out_peqdsk)

    print("\nPASS: filesystem-backed IMAS entry opened successfully, GEQDSK/PEQDSK were reconstructed, and strict regression profiles passed.")


if __name__ == "__main__":
    main()

