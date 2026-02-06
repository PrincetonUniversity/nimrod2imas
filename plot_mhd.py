#!/usr/bin/env python3
"""plot_mhd.py

Plot contour slices from IMAS mhd IDS (GGD) with a robust HDF5 fallback.

Usage examples
  # IMAS mode (preferred)
  python plot_mhd.py --entry mast/4/45272/5/ --dd-version 4.1.1 --occ 1 --time-index 0 --phi-index 0 --quantity te

  # Force HDF5 mode (uses mhd_<occ>.h5 directly)
  python plot_mhd.py --entry mast/4/45272/5/ --occ 1 --time-index 0 --phi-index 0 --quantity te --hdf5-only

Notes
- The script assumes that the mhd IDS was written by our NIMROD->IMAS pipeline
  and (optionally) includes an auxiliary group:
      /mhd_<occ>/nimrod_unstructured
  with r_axis, z_axis, phi_axis_used, nr, nz, nphi.
- If geometry is only available inside grid_ggd, the HDF5 fallback may not work
  unless you add a detector for your particular layout.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

from pathlib import Path

from nimrod2imas import entry_dir as _entry_dir_common

# Optional: IMAS is only needed for IMAS mode
try:
    import imas
except Exception:
    imas = None

try:
    import h5py
except Exception as e:
    raise SystemExit(f"h5py is required for this script: {e}")


# ----------------------------
# Quantity mapping (HDF5)
# ----------------------------
_H5_VALUES_DS = {
    # electrons
    "te": "ggd[]&electrons&temperature[]&values",
    "temperature": "ggd[]&electrons&temperature[]&values",
    # ions (bulk/average)
    "ti": "ggd[]&t_i_average[]&values",
    "t_i_average": "ggd[]&t_i_average[]&values",
    # total ion density
    "ni": "ggd[]&n_i_total[]&values",
    "n_i_total": "ggd[]&n_i_total[]&values",
}


def _infer_mhd_h5_path(entry: str, occ: int) -> str:
    entry = entry.rstrip("/")
    return os.path.join(entry, f"mhd_{occ}.h5")


def _h5_open_group(h5_path: str, occ: int) -> Tuple[h5py.File, h5py.Group, str]:
    f = h5py.File(h5_path, "r")
    grp_name = f"mhd_{occ}"
    if grp_name not in f:
        # Some backends store as /mhd_1 etc (same)
        raise RuntimeError(f"Group /{grp_name} not found in {h5_path}")
    return f, f[grp_name], grp_name


def _h5_read_aux_unstructured(g: h5py.Group) -> Optional[Dict[str, Any]]:
    """Read auxiliary axes if present.

    Expected layout created by dump2imas (h5py-direct option):
      /mhd_<occ>/nimrod_unstructured/{r_axis,z_axis,phi_axis_used,nr,nz,nphi}
    """
    if "nimrod_unstructured" not in g:
        return None
    auxg = g["nimrod_unstructured"]

    def _get_axis(*names: str) -> Optional[np.ndarray]:
        for n in names:
            if n in auxg:
                return np.asarray(auxg[n][()])
        return None

    r = _get_axis("r_axis", "r")
    z = _get_axis("z_axis", "z")
    phi = _get_axis("phi_axis_used", "phi", "phi_axis")

    # Scalars are stored as datasets
    def _get_scalar(name: str) -> Optional[int]:
        if name in auxg:
            return int(np.asarray(auxg[name][()]).reshape(()))
        return None

    nr = _get_scalar("nr")
    nz = _get_scalar("nz")
    nphi = _get_scalar("nphi")

    if r is None or z is None:
        # Incomplete aux: treat as missing
        return None

    out: Dict[str, Any] = {"r": r, "z": z}
    if phi is not None:
        out["phi"] = phi
    if nr is not None:
        out["nr"] = np.asarray(nr)
    if nz is not None:
        out["nz"] = np.asarray(nz)
    if nphi is not None:
        out["nphi"] = np.asarray(nphi)
    # Optional heavy arrays (present when dump2imas wrote them directly with h5py).
    if "nodes" in auxg:
        out["nodes"] = np.asarray(auxg["nodes"][()])
    if "connectivity" in auxg:
        out["connectivity"] = np.asarray(auxg["connectivity"][()])
    return out


def _h5_read_values_1d(g: h5py.Group, quantity: str, time_index: int) -> np.ndarray:
    key = quantity.lower()
    if key not in _H5_VALUES_DS:
        raise RuntimeError(f"Unsupported quantity '{quantity}'. Supported: {sorted(_H5_VALUES_DS)}")
    ds_name = _H5_VALUES_DS[key]
    if ds_name not in g:
        raise RuntimeError(f"Dataset '{ds_name}' not found under group '{g.name}'.")
    ds = g[ds_name]
    # Common shape: (ntime, 1, nvals) but allow variants
    arr = np.asarray(ds)
    if arr.ndim == 3:
        if time_index >= arr.shape[0]:
            raise RuntimeError(f"time_index {time_index} out of range (ntime={arr.shape[0]})")
        return np.asarray(arr[time_index, 0, :])
    if arr.ndim == 2:
        if time_index >= arr.shape[0]:
            raise RuntimeError(f"time_index {time_index} out of range (ntime={arr.shape[0]})")
        return np.asarray(arr[time_index, :])
    if arr.ndim == 1:
        # single time slice
        if time_index != 0:
            raise RuntimeError("Only one time slice stored; use --time-index 0")
        return np.asarray(arr)
    raise RuntimeError(f"Unexpected dataset rank for {ds_name}: {arr.shape}")


def _reshape_to_zrphi(values_1d: np.ndarray, nr: int, nz: int, nphi: int) -> np.ndarray:
    """Reshape packed 1D values into (nz, nr, nphi).

    This choice matches matplotlib's default meshgrid(r,z) (indexing='xy'), where
    X/Z grids have shape (nz, nr). Keeping data as (nz, nr, nphi) avoids the
    ambiguous transpose issue when nr == nz.
    """
    nvals = values_1d.size
    need = nr * nz * nphi
    if nvals != need:
        raise RuntimeError(f"Cannot reshape: nvals={nvals} != nr*nz*nphi={need} (nr={nr}, nz={nz}, nphi={nphi})")
    return values_1d.reshape((nz, nr, nphi), order="C")


def _slice_phi(a3: np.ndarray, phi_index: int) -> np.ndarray:
    if phi_index < 0 or phi_index >= a3.shape[2]:
        raise RuntimeError(f"phi_index {phi_index} out of range (nphi={a3.shape[2]})")
    return a3[:, :, phi_index]


def _wrap_delta_phi(phi: np.ndarray, phi0: float) -> np.ndarray:
    """Return wrapped (phi-phi0) into [-pi, pi]."""
    return np.angle(np.exp(1j * (phi - phi0)))


def _plane_from_unstructured_nodes(
    vals_1d: np.ndarray,
    aux: Dict[str, Any],
    phi_index: int,
) -> Optional[np.ndarray]:
    """Reconstruct a (nz,nr) plane using per-node coordinates.

    This avoids ambiguous reshape/transpose when nr==nz and is robust to packing order,
    as long as `nimrod_unstructured/nodes` and `nimrod_unstructured/r_axis,z_axis` were written
    and `vals_1d` is aligned with the node ordering.
    """
    nodes = aux.get("nodes")
    if nodes is None:
        return None
    nodes = np.asarray(nodes)
    if nodes.ndim != 2 or nodes.shape[1] < 3:
        return None

    r_axis = np.asarray(aux.get("r"))
    z_axis = np.asarray(aux.get("z"))
    if r_axis.ndim != 1 or z_axis.ndim != 1:
        return None

    nr = int(np.asarray(aux.get("nr", r_axis.size)))
    nz = int(np.asarray(aux.get("nz", z_axis.size)))
    if nr <= 0 or nz <= 0:
        return None

    phi_axis = aux.get("phi")
    if phi_axis is None:
        # derive from nodes
        phi_axis = np.unique(nodes[:, 2])
    phi_axis = np.asarray(phi_axis).ravel()
    if phi_index < 0 or phi_index >= phi_axis.size:
        return None

    # Select nodes belonging to the requested phi plane.
    phi0 = float(phi_axis[phi_index])
    # tolerance: a fraction of nominal spacing
    if phi_axis.size > 1:
        dphi_nom = float(np.min(np.diff(np.sort(np.unique(phi_axis)))))
        tol = max(dphi_nom * 0.25, 1e-8)
    else:
        tol = 1e-6
    mask = np.abs(_wrap_delta_phi(nodes[:, 2], phi0)) <= tol

    if mask.sum() == 0:
        return None

    plane_nodes = nodes[mask]
    plane_vals = np.asarray(vals_1d)[mask]

    # Map each (R,Z) to nearest axis indices.
    # We assume r_axis/z_axis are sorted ascending.
    r = plane_nodes[:, 0]
    z = plane_nodes[:, 1]

    ir = np.searchsorted(r_axis, r)
    ir = np.clip(ir, 1, r_axis.size - 1)
    left = np.abs(r - r_axis[ir - 1])
    right = np.abs(r - r_axis[ir])
    ir = np.where(right < left, ir, ir - 1)

    iz = np.searchsorted(z_axis, z)
    iz = np.clip(iz, 1, z_axis.size - 1)
    leftz = np.abs(z - z_axis[iz - 1])
    rightz = np.abs(z - z_axis[iz])
    iz = np.where(rightz < leftz, iz, iz - 1)

    a2 = np.full((nr, nz), np.nan, dtype=float)
    # Fill; if duplicates occur, last one wins (should not happen for a regular grid).
    a2[ir, iz] = plane_vals

    # If many NaNs remain, fall back (likely mismatch in axes).
    if np.isnan(a2).mean() > 0.05:
        return None
    return a2


def _plot_contour(r: np.ndarray, z: np.ndarray, a2: np.ndarray, title: str, out: Optional[str] = None) -> None:
    import matplotlib.pyplot as plt

    R, Z = np.meshgrid(r, z)  # shapes (nz, nr)

    fig, ax = plt.subplots()
    cf = ax.contourf(R, Z, a2, levels=40)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title)
    fig.colorbar(cf, ax=ax)

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def _try_imas_read(entry: str, dd_version: str, occ: int):
    if imas is None:
        raise RuntimeError("IMAS-Python is not available in this environment.")

    # Use DD specified by the user; required when reading different DD majors.
    factory = imas.IDSFactory(version=dd_version)
    db = imas.DBEntry(backend="hdf5", path=entry, mode="r", ids_factory=factory)
    try:
        ids = db.get("mhd", occ)
    finally:
        db.close()
    return ids


def _imas_extract_axes(ids) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best-effort extraction of r,z,phi from ids.grid_ggd.

    This is intentionally conservative: if we can't confidently locate axes, we
    raise and allow HDF5 fallback.
    """
    if not hasattr(ids, "grid_ggd") or len(ids.grid_ggd) == 0:
        raise RuntimeError("mhd.grid_ggd is missing or empty")

    ggd = ids.grid_ggd[0]
    # Different bindings expose geometry differently; try common patterns.
    # We only implement a minimal extractor here; most users rely on HDF5 aux.
    try:
        # If space objects provide 1D coordinate arrays (rare)
        r = np.asarray(getattr(ggd.space[0], "coordinate", None))
        z = np.asarray(getattr(ggd.space[1], "coordinate", None))
        phi = np.asarray(getattr(ggd.space[2], "coordinate", None))
        if r.size and z.size and phi.size:
            return r, z, phi
    except Exception:
        pass

    raise RuntimeError("Could not extract axes from mhd.grid_ggd")


def main() -> int:
    p = argparse.ArgumentParser(description="Plot contours from mhd IDS (GGD)")
    p.add_argument("--entry", default=None, help="IMAS entry directory (contains master.h5, mhd_*.h5, etc)")
    # Consistent alternative to --entry (match dump2imas/input2imas directory logic)
    p.add_argument("--dbpath", default=".", help="DB root path")
    p.add_argument("--dd", default=None, help="DB name (directory name), e.g. nstx")
    p.add_argument("--dd-version-dir", choices=["major", "full"], default="major",
                   help="Directory component for DD version (default: major, e.g. 3 for 3.42.0)")
    p.add_argument("--pulse", type=int, default=None)
    p.add_argument("--run", type=int, default=None)
    p.add_argument("--dd-version", default=None, help="IMAS DD version for IMAS read (required unless --hdf5-only)")
    p.add_argument("--occ", type=int, required=True, help="Occurrence number")
    p.add_argument("--time-index", type=int, default=0, help="Time slice index")
    p.add_argument("--phi-index", type=int, default=0, help="Phi index")
    p.add_argument("--quantity", required=True, help="Quantity: te, ti, ni (or aliases)")
    p.add_argument("--hdf5-only", action="store_true", help="Force direct HDF5 mode (bypass IMAS)")
    p.add_argument("--output", default=None, help="Output image path (if omitted, show interactively)")
    p.add_argument("--info", action="store_true", help="Print resolved geometry/value shapes and exit")
    #p.add_argument(
    #    "--swap-rz",
    #    action="store_true",
    #    help="Transpose the R/Z plane after slicing (useful if values were packed with swapped r,z ordering).",
    #)

    args = p.parse_args()

    if args.entry:
        entry = str(Path(args.entry).expanduser().resolve()).rstrip("/") + "/"
    else:
        if args.dd is None or args.pulse is None or args.run is None or not args.dd_version:
            raise SystemExit(
                "Provide either --entry, or (--dbpath --dd --dd-version --pulse --run). "
                "Note: --dd-version is also required for IMAS mode."
            )
        entry = str(
            _entry_dir_common(
                args.dbpath,
                str(args.dd),
                str(args.dd_version),
                int(args.pulse),
                int(args.run),
                dd_version_dir=str(args.dd_version_dir),
            )
        ).rstrip("/") + "/"

    # ----------------------------
    # First: try IMAS (unless forced)
    # ----------------------------
    ids = None
    imas_axes: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    imas_failed_reason: Optional[str] = None

    if not args.hdf5_only:
        if not args.dd_version:
            return SystemExit("--dd-version is required unless --hdf5-only is used")
        try:
            ids = _try_imas_read(entry, args.dd_version, args.occ)
            imas_axes = _imas_extract_axes(ids)
        except Exception as e:
            imas_failed_reason = str(e)
            ids = None
            imas_axes = None

    # ----------------------------
    # HDF5 read (always available)
    # ----------------------------
    h5_path = _infer_mhd_h5_path(entry, args.occ)
    if not os.path.exists(h5_path):
        raise SystemExit(f"Cannot find {h5_path}")

    f, g, grp_name = _h5_open_group(h5_path, args.occ)
    try:
        aux = _h5_read_aux_unstructured(g)
        if aux is None:
            raise RuntimeError(
                "Could not find /nimrod_unstructured auxiliary axes in mhd_<occ>.h5; "
                "and IMAS grid_ggd extraction did not succeed."
            )

        r = np.asarray(aux["r"], dtype=float)
        z = np.asarray(aux["z"], dtype=float)
        phi_axis_used = np.asarray(aux.get("phi", np.linspace(0.0, 2.0 * np.pi, int(aux.get("nphi", 1)), endpoint=False)))

        nr = int(r.size)
        nz = int(z.size)
        # Determine nphi
        if "nphi" in aux:
            nphi = int(np.asarray(aux["nphi"]).reshape(()))
        else:
            # infer from values length
            vals_1d_tmp = _h5_read_values_1d(g, args.quantity, args.time_index)
            nphi = int(vals_1d_tmp.size // (nr * nz))

        vals_1d = _h5_read_values_1d(g, args.quantity, args.time_index)

        # Prefer coordinate-driven reconstruction if node coordinates are present.
        a2 = None
        a2_from_nodes = False
        if aux and "nodes" in aux:
            a2 = _plane_from_unstructured_nodes(vals_1d, aux, args.phi_index)
            a2_from_nodes = a2 is not None

        # Fallback: assume values are already packed as a structured (z,r,phi) cube.
        if a2 is None:
            a3 = _reshape_to_zrphi(vals_1d, nr=nr, nz=nz, nphi=nphi)
            a2 = _slice_phi(a3, args.phi_index)
            # Heuristic transpose fix:
            # Old code transposed when a2.shape==(nr,nz); that breaks when nr==nz.
            # We now keep (nz,nr) as canonical; transpose only if shapes clearly swapped.
            if a2.shape == (nr, nz) and (nr != nz):
                a2 = a2.T

        # Info mode
        if args.info:
            print(f"HDF5: {h5_path}")
            print(f"Group: /{grp_name}")
            if args.dd_version:
                print(f"dd_version (IMAS attempted): {args.dd_version}")
            else:
                print("dd_version: None")
            if imas_failed_reason:
                print(f"IMAS mode failed: {imas_failed_reason}")
            print("Geometry dataset used: nimrod_unstructured")
            print(f"Axes: nr={nr} nz={nz} nphi={nphi} phi_len_used={phi_axis_used.size}")
            if a2_from_nodes:
                shape3 = (nz, nr, nphi)
                src = "(from nodes mapping)"
            else:
                shape3 = a3.shape
                src = "(from reshape)"
            print(f"Values: nvals={vals_1d.size} shape3={shape3} (nphi={nphi}) {src}")
            # time
            if "ggd[]&time" in g:
                t = np.asarray(g["ggd[]&time"][()])[args.time_index]
                print(f"Time: {t}")
            else:
                print("Time: (not found)")
            return 0

        # Optional manual transpose (debug/legacy packing)
        #if args.swap_rz:
        #a2 = a2.T

        # Plot
        title = f"mhd_{args.occ}: {args.quantity} time_index={args.time_index} phi_index={args.phi_index}"
        _plot_contour(r, z, a2, title=title, out=args.output)
        return 0

    finally:
        f.close()


if __name__ == "__main__":
    raise SystemExit(main())
