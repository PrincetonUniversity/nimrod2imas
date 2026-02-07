#!/usr/bin/env python3
"""plot_mhd_fixed.py (IMAS-only)

Plot a scalar GGD quantity from an IMAS HDF5 IDS (mhd, edge_profiles, etc.)
produced by dump2imas.

This script intentionally avoids any non-DD auxiliary structures (e.g. no
nimrod_unstructured group). Geometry is obtained from IMAS-standard grid_ggd
encodings:

  1) Packed grid_ggd nodes (preferred, robust across options):
     /<ids>_<occ>/grid_ggd[]&grid_subset[]&element[]&object[]&real
     using grid_subset.dimension==0 (nodes)

  2) grid_ggd space-geometry vectors (if present):
     /<ids>_<occ>/grid_ggd[]&space[]&objects_per_dimension[]&object[]&geometry

Values are read from:
  /<ids>_<occ>/ggd[]&<leaf>[]&values

Examples
  python plot_mhd_fixed.py --dd mast --dd-version 4.1.1 --pulse 45272 --run 9 --occ 1 \
      --ids mhd --quantity ni --phi-index 0 --show --debug

  python plot_mhd_fixed.py --dd mast --dd-version 4.1.1 --pulse 45272 --run 9 --occ 1 \
      --ids edge_profiles --quantity vphi --ion-index 0 --show
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.tri import TriAnalyzer

VERSION = "plot_mhd_fixed_v3"


# ----------------- filesystem helpers -----------------

def _major(ddv: str) -> str:
    try:
        return str(int(ddv.split(".")[0]))
    except Exception:
        return ddv.split(".")[0]


def _base(dd: str, ddv: str, pulse: int, run: int) -> str:
    return os.path.join(dd, _major(ddv), str(pulse), str(run))


def _ids_file(base: str, ids: str, occ: int) -> Tuple[str, str]:
    """Return (filename, group_name)."""
    p1 = os.path.join(base, f"{ids}_{occ}.h5")
    if os.path.exists(p1):
        return p1, f"{ids}_{occ}"
    p2 = os.path.join(base, f"{ids}.h5")
    if os.path.exists(p2):
        return p2, f"{ids}_{occ}"
    raise FileNotFoundError(f"Could not find {p1} or {p2}")


# ----------------- quantity / leaf resolution -----------------

def _leaf_candidates(ids: str, quantity: str) -> List[str]:
    """Return candidate IMAS leaf names (WITHOUT the ggd[]& prefix) for a user quantity."""
    q = (quantity or "").strip()
    ql = q.lower()
    ids_l = (ids or "").strip().lower()

    # If the user passed an IMAS-ish leaf explicitly, accept it.
    # (e.g. "electrons&temperature" or "ion[]&pressure")
    if "&" in q or "ion[]" in q or "electrons" in q:
        return [q]

    # Common short-hands
    base: Dict[str, List[str]] = {
        "ni": ["n_i_total", "n_i", "n_i_total_over_n_e"],
        "ti": ["t_i_average", "t_i"],
        "te": ["electrons&temperature", "t_e", "te"],
        "ne": ["electrons&density", "n_e"],
        "pe": ["electrons&pressure", "p_e"],
        "pi": ["p_i", "ions&pressure"],
        "jphi": ["j_phi", "j_tor", "current_density_phi", "current_density_tor"],
        "jtor": ["j_tor", "j_phi", "current_density_tor", "current_density_phi"],
        "j": ["j_total", "j_phi", "j_tor"],
        "vr": ["velocity_r", "v_r"],
        "vz": ["velocity_z", "v_z"],
        "vphi": ["velocity_phi", "velocity_tor", "v_phi"],
        "vtor": ["velocity_tor", "velocity_phi", "v_phi"],
    }

    cand = base.get(ql, [q])

    # edge_profiles special casing: velocity lives under ion[]&velocity&*
    if ids_l == "edge_profiles":
        if ql in ("vr", "v_r", "velocity_r"):
            cand = ["ion[]&velocity&r", "ion[]&velocity_r"]
        elif ql in ("vz", "v_z", "velocity_z"):
            cand = ["ion[]&velocity&z", "ion[]&velocity_z"]
        elif ql in ("vphi", "v_phi", "vtor", "velocity_phi", "velocity_tor"):
            cand = ["ion[]&velocity&phi", "ion[]&velocity_phi", "ion[]&velocity_tor"]
        elif ql in ("ni", "n_i", "n_i_total"):
            cand = ["n_i_total_over_n_e", "n_i_total"]

    # Also try a few tokenization variants.
    out: List[str] = []
    for c in cand:
        out.append(c)
        if "velocity_" in c:
            out.append(c.replace("velocity_", "v_"))
        if c.startswith("v_"):
            out.append(c.replace("v_", "velocity_"))
    # de-dup while preserving order
    seen = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _find_values_dataset(f: h5py.File, grp: str, leaf_candidates: Sequence[str], debug: bool = False) -> Tuple[str, str]:
    """Return (leaf_used, values_dataset_path)."""
    gprefix = f"/{grp}"

    # Direct attempts
    for leaf in leaf_candidates:
        vpath = f"{gprefix}/ggd[]&{leaf}[]&values"
        if vpath in f:
            return leaf, vpath

    # Fallback: search group keys
    if gprefix not in f:
        raise KeyError(f"Missing group {gprefix} in file")

    keys = list(f[gprefix].keys())
    # candidates based on substring match
    for leaf in leaf_candidates:
        token = f"ggd[]&{leaf}[]&values"
        for k in keys:
            if k.endswith("&values") and token in k:
                return leaf, f"{gprefix}/{k}"

    # looser: any values dataset that contains the last token of leaf
    for leaf in leaf_candidates:
        last = leaf.split("&")[-1]
        for k in keys:
            if k.endswith("&values") and ("ggd[]&" in k) and (last in k):
                if debug:
                    print(f"DEBUG: using loose-match dataset {gprefix}/{k} for leaf '{leaf}'", file=sys.stderr)
                return leaf, f"{gprefix}/{k}"

    head = keys[:80]
    raise KeyError(
        "Could not locate GGD values dataset. Tried leaves: "
        + ", ".join(leaf_candidates)
        + f". Keys under {gprefix} include: {head} ..."
    )


# ----------------- geometry extraction -----------------

def _normalize_phi_units(phi: np.ndarray, debug: bool = False) -> np.ndarray:
    """Return phi in radians. Auto-detect degrees if range suggests so."""
    ph = np.asarray(phi, dtype=float).reshape(-1)
    ph = ph[np.isfinite(ph)]
    if ph.size == 0:
        return np.asarray(phi, dtype=float)
    pmin, pmax = float(np.min(ph)), float(np.max(ph))
    deg_like = (pmax > 2 * math.pi * 1.5) and (pmax <= 360.0 + 1e-6) and (pmin >= -1e-6)
    if deg_like:
        if debug:
            print(f"DEBUG: phi looks like degrees (min={pmin:.6g}, max={pmax:.6g}); converting to radians", file=sys.stderr)
        return np.asarray(phi, dtype=float) * (math.pi / 180.0)
    return np.asarray(phi, dtype=float)


def _packed_nodes_from_gridggd(f: h5py.File, grp: str, debug: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read nodes from packed grid_ggd object[]&real selecting grid_subset.dimension==0."""
    dim_path = f"/{grp}/grid_ggd[]&grid_subset[]&dimension"
    real_path = f"/{grp}/grid_ggd[]&grid_subset[]&element[]&object[]&real"
    if dim_path not in f or real_path not in f:
        return None

    dims = np.asarray(f[dim_path][()])
    real = np.asarray(f[real_path][()])

    # Expect leading dims (nggd, nsubsets, ...)
    if dims.ndim < 2 or real.ndim < 3:
        return None

    g0 = 0
    dims0 = np.asarray(dims[g0]).reshape(-1)

    # Choose subset whose dimension is 0; else fallback to 0
    cand = np.where(dims0 == 0)[0]
    if cand.size == 0:
        sidx = 0
    else:
        sidx = int(cand[0])

    # Slice real to [ggd=0, subset=sidx, ...]
    a = real
    # squeeze ggd axis
    if a.shape[0] > 1:
        a = a[g0]
    else:
        a = a[0]

    # now a has subset axis at 0
    if a.ndim < 2:
        return None
    if sidx >= a.shape[0]:
        sidx = 0
    a = a[sidx]

    # a should now be (nnodes, 3) or (nnodes, >=3) possibly with extra trailing dims
    a = np.asarray(a)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]

    if a.ndim == 2 and a.shape[1] >= 2:
        r = a[:, 0].astype(float, copy=False)
        z = a[:, 1].astype(float, copy=False)
        if a.shape[1] >= 3:
            phi = a[:, 2].astype(float, copy=False)
        else:
            phi = np.zeros_like(r)
        if debug:
            print(f"DEBUG: packed nodes from subset={sidx} nnodes={r.size}", file=sys.stderr)
        return r, z, phi

    return None


def _space_geometry_vectors(f: h5py.File, grp: str, debug: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read per-node space geometry vectors if present."""
    key = f"/{grp}/grid_ggd[]&space[]&objects_per_dimension[]&object[]&geometry"
    if key not in f:
        return None

    arr = np.asarray(f[key][()])
    if debug:
        print(f"DEBUG: space geometry dataset {key} shape={arr.shape}", file=sys.stderr)

    if arr.size < 10:
        return None

    # Find a plausible space axis: first axis with size 2 or 3
    space_ax = None
    for ax, sz in enumerate(arr.shape):
        if sz in (2, 3):
            space_ax = ax
            break
    if space_ax is None:
        return None

    a = np.moveaxis(arr, space_ax, 0)  # a[0]=R, a[1]=Z, a[2]=phi

    if a.shape[0] < 2:
        return None

    # Heuristic: select the first entry along small leading axes (objects_per_dimension/object),
    # then flatten the remaining node axis.
    def _flatten(comp: np.ndarray) -> np.ndarray:
        x = comp
        # peel off small leading dims by taking index 0
        while x.ndim > 1 and x.shape[0] <= 4:
            x = x[0]
        return np.asarray(x, dtype=float).reshape(-1)

    r = _flatten(a[0])
    z = _flatten(a[1])
    phi = _flatten(a[2]) if a.shape[0] >= 3 else np.zeros_like(r)

    m = min(r.size, z.size, phi.size)
    if m < 10:
        return None
    return r[:m], z[:m], phi[:m]


def _extract_geometry(f: h5py.File, grp: str, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    geom = _packed_nodes_from_gridggd(f, grp, debug=debug)
    if geom is not None:
        r, z, phi = geom
        phi = _normalize_phi_units(phi, debug=debug)
        return r, z, phi

    geom = _space_geometry_vectors(f, grp, debug=debug)
    if geom is not None:
        r, z, phi = geom
        phi = _normalize_phi_units(phi, debug=debug)
        return r, z, phi

    raise RuntimeError(
        f"Could not find grid_ggd node geometry in /{grp}.\n"
        "Tried packed nodes: grid_ggd[]&grid_subset[]&element[]&object[]&real\n"
        "and space geometry: grid_ggd[]&space[]&...&geometry"
    )


# ----------------- values extraction -----------------

def _select_time_and_object(arr: np.ndarray, t_index: int) -> np.ndarray:
    """Select time index and object index (0) from typical IMAS packed arrays."""
    a = np.asarray(arr)

    # select time if axis0 looks like time
    if a.ndim >= 1 and a.shape[0] > 1:
        ti = max(0, min(int(t_index), a.shape[0] - 1))
        a = a[ti]
    elif a.ndim >= 1 and a.shape[0] == 1:
        a = a[0]

    # select object if next axis looks like object
    if a.ndim >= 2 and a.shape[0] <= 8:
        a = a[0]

    return np.asarray(a)


def _read_values(
    f: h5py.File,
    grp: str,
    leaf: str,
    values_path: str,
    time_index: int,
    ion_index: int,
    debug: bool = False,
) -> np.ndarray:
    a = _select_time_and_object(f[values_path][()], time_index)

    # Handle ion[] multi-ion arrays if present as (nion, npts)
    if "ion[]" in leaf and a.ndim >= 2 and a.shape[0] > 1:
        ii = max(0, min(int(ion_index), a.shape[0] - 1))
        a = a[ii]

    v = np.asarray(a, dtype=float).reshape(-1)

    # Mask common IMAS fill values (e.g. -9e40) and absurd magnitudes
    bad = (~np.isfinite(v)) | (np.abs(v) > 1.0e30) | (v < -8.0e39)
    if np.any(bad):
        v = v.astype(float, copy=True)
        v[bad] = np.nan

    if debug:
        print(f"DEBUG: values_path={values_path} leaf={leaf} n={v.size}", file=sys.stderr)

    return v


def _match_geometry_to_values(
    r: np.ndarray,
    z: np.ndarray,
    phi: np.ndarray,
    v: np.ndarray,
    phi_index: int,
    phi_tol: float,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """If geometry has replicated phi planes but values only cover one plane, downselect."""
    r = np.asarray(r).reshape(-1)
    z = np.asarray(z).reshape(-1)
    phi = np.asarray(phi).reshape(-1)
    v = np.asarray(v).reshape(-1)

    if r.size == v.size:
        return r, z, phi, v

    # Geometry replicated across planes: r.size = nplanes * v.size
    if v.size > 0 and r.size % v.size == 0:
        nplanes = r.size // v.size
        # identify unique planes
        ph = phi[np.isfinite(phi)]
        if ph.size:
            u = np.unique(np.round(ph, 12))
            u.sort()
            if u.size == nplanes:
                k = max(0, min(int(phi_index), int(u.size) - 1))
                target = float(u[k])
                tol = max(float(phi_tol), 1e-8)
                sel = np.isfinite(phi) & (np.abs(phi - target) <= tol)
                if int(sel.sum()) == int(v.size):
                    if debug:
                        print(
                            f"DEBUG: matched geometry to values by selecting phi plane {k}/{u.size} (phi={target:g})",
                            file=sys.stderr,
                        )
                    return r[sel], z[sel], phi[sel], v

    # Last-resort: trim to min length
    m = min(r.size, v.size)
    if debug:
        print(f"DEBUG: length mismatch geom={r.size} values={v.size}; trimming to {m}", file=sys.stderr)
    return r[:m], z[:m], phi[:m], v[:m]


# ----------------- plotting helpers -----------------

def _select_phi_plane(r, z, phi, v, phi_index=0, phi_tol=1e-6, min_points=200, debug=False):
    phi = np.asarray(phi).reshape(-1)
    v = np.asarray(v).reshape(-1)

    good = np.isfinite(phi)
    if good.sum() == 0:
        raise RuntimeError("No finite phi values found.")

    # Bin by tolerance
    tol = max(float(phi_tol), 1e-12)
    bins = np.round(phi / tol).astype(np.int64)
    uniq_bins, counts = np.unique(bins[good], return_counts=True)
    order = np.argsort(uniq_bins)
    uniq_bins = uniq_bins[order]
    counts = counts[order]

    if uniq_bins.size == 0:
        raise RuntimeError("No phi planes after binning.")

    k = max(0, min(int(phi_index), int(uniq_bins.size) - 1))
    b = int(uniq_bins[k])
    phi_used = b * tol
    sel = good & (bins == b)

    nsel = int(sel.sum())
    if debug:
        print(f"DEBUG: phi planes={uniq_bins.size} selected index={k} nsel={nsel} phi_used~{phi_used:g}", file=sys.stderr)

    if nsel < min_points:
        best_i = int(np.argmax(counts))
        b2 = int(uniq_bins[best_i])
        sel2 = good & (bins == b2)
        if int(sel2.sum()) > nsel:
            if debug:
                print(f"WARNING: phi-index {k} had {nsel} pts; using best plane with n={int(sel2.sum())}", file=sys.stderr)
            sel = sel2
            phi_used = b2 * tol
            nsel = int(sel.sum())


    # Special case: if phi is effectively constant (one plane) but nodes contain repeated
    # (R,Z) blocks (common when phi coordinate isn't populated for replicated planes),
    # infer planes by block and select the requested plane index.
    if uniq_bins.size == 1 and nsel >= min_points:
        idx_all = np.nonzero(sel)[0]
        rr = np.asarray(r, dtype=float).reshape(-1)[idx_all]
        zz = np.asarray(z, dtype=float).reshape(-1)[idx_all]
        if rr.size:
            rspan = float(np.nanmax(rr) - np.nanmin(rr))
            zspan = float(np.nanmax(zz) - np.nanmin(zz))
            span = max(rspan, zspan, 1.0)
            qt = 1.0 / (1e-10 * span)  # scale-invariant quantization
            rq = np.round(rr * qt).astype(np.int64, copy=False)
            zq = np.round(zz * qt).astype(np.int64, copy=False)
            uniq_n = np.unique(np.stack([rq, zq], axis=1), axis=0).shape[0]
            if uniq_n >= 3 and (idx_all.size % uniq_n == 0):
                nplanes = idx_all.size // uniq_n
                if nplanes > 1:
                    kplane = int(phi_index) % int(nplanes)
                    start = kplane * uniq_n
                    end = start + uniq_n
                    if end <= idx_all.size:
                        idx_plane = idx_all[start:end]
                        if debug:
                            print(
                                f"DEBUG: phi constant; inferred nplanes={nplanes} by repeated (R,Z) blocks; "
                                f"selecting plane {kplane} (n={idx_plane.size})",
                                file=sys.stderr,
                            )
                        return np.asarray(r).reshape(-1)[idx_plane], np.asarray(z).reshape(-1)[idx_plane], np.asarray(v).reshape(-1)[idx_plane], phi_used, idx_plane.size, idx_plane
    if nsel < 3:
        raise RuntimeError(f"Not enough points in selected phi plane (n={nsel}).")

    idx = np.nonzero(sel)[0]
    return r[sel], z[sel], v[sel], phi_used, nsel, idx


def _dedup_rz(r, z, v, tol=1e-10, debug=False):
    r = np.asarray(r, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)
    if r.size == 0:
        return r, z, v
    q = 1.0 / tol if tol > 0 else 1e12
    rq = np.round(r * q).astype(np.int64, copy=False)
    zq = np.round(z * q).astype(np.int64, copy=False)
    key = np.stack([rq, zq], axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    if uniq.shape[0] == r.size:
        return r, z, v
    vsum = np.bincount(inv, weights=v)
    vcnt = np.bincount(inv)
    vavg = vsum / np.maximum(vcnt, 1)
    r_u = uniq[:, 0].astype(float) / q
    z_u = uniq[:, 1].astype(float) / q
    if debug:
        print(f"DEBUG: dedup_rz reduced n {r.size}->{uniq.shape[0]} (tol={tol})", file=sys.stderr)
    return r_u, z_u, vavg


def _plot_tricontour(
    r: np.ndarray,
    z: np.ndarray,
    v: np.ndarray,
    title: str,
    outpng: Optional[str] = None,
    show: bool = False,
    triangles: Optional[np.ndarray] = None,
    mask_flat_tris: bool = False,
    min_circle_ratio: float = 0.01,
):
    if triangles is None:
        tri = mtri.Triangulation(r, z)
    else:
        tri = mtri.Triangulation(r, z, triangles=triangles)

    v_ma = np.ma.masked_invalid(v)
    v_mask = np.ma.getmaskarray(v_ma)
    if v_mask.all():
        raise ValueError("All selected values are non-finite (NaN/Inf); cannot plot.")
    # NOTE: we prefer to pre-filter NaN nodes before triangulation; but keep a light
    # safety mask here as well.
    if v_mask.any():
        bad_tris = np.any(v_mask[tri.triangles], axis=1)
        tri.set_mask(bad_tris if tri.mask is None else (tri.mask | bad_tris))

    if mask_flat_tris:
        try:
            ta = TriAnalyzer(tri)
            flat = ta.get_flat_tri_mask(min_circle_ratio=min_circle_ratio)
            tri.set_mask(flat if tri.mask is None else (tri.mask | flat))
        except Exception:
            pass

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.tricontourf(tri, v_ma, levels=40)
    fig.colorbar(cs, ax=ax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_title(title)
    fig.tight_layout()
    if outpng:
        fig.savefig(outpng, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ----------------- connectivity (optional) -----------------

def _load_tri_connectivity(
    f: h5py.File, grp: str, n_nodes: int, debug: bool = False
) -> Optional[np.ndarray]:
    """
    Load triangle connectivity (nv==3) from IMAS packed grid_ggd.

    Important: in some outputs the subset axis contains an entry with degenerate
    connectivity (e.g. all zeros). We select the subset with the lowest fraction
    of degenerate triangles and with indices compatible with n_nodes.

    Returns triangles as (ntri,3) 0-based, with degenerates removed.
    """
    idx_path = f"/{grp}/grid_ggd[]&grid_subset[]&element[]&object[]&index"
    if idx_path not in f:
        return None

    conn = np.asarray(f[idx_path][()])
    conn = np.asarray(conn).squeeze()

    # Peel leading singleton dims (e.g. grid_index axis)
    while conn.ndim > 2 and conn.shape[0] == 1:
        conn = conn[0]

    candidates: list[tuple[float, int, np.ndarray, int, int]] = []  # (deg_ratio, max_idx, tri, min_idx, is_1based)
    def _prep_candidate(craw: np.ndarray) -> None:
        if craw.ndim != 2 or craw.shape[1] != 3:
            return
        c = np.asarray(craw)

        # Determine 1-based vs 0-based heuristically against n_nodes
        cmin = int(np.min(c))
        cmax = int(np.max(c))
        is_1based = 0
        if cmin >= 1 and cmax <= n_nodes:
            is_1based = 1
            c0 = c.astype(np.int64) - 1
        else:
            c0 = c.astype(np.int64)

        # Reject if wildly out of range
        if c0.size == 0:
            return
        if np.min(c0) < 0 or np.max(c0) >= n_nodes:
            return

        # Degenerate triangles (repeated vertices)
        deg = np.count_nonzero((c0[:, 0] == c0[:, 1]) | (c0[:, 0] == c0[:, 2]) | (c0[:, 1] == c0[:, 2]))
        deg_ratio = float(deg) / float(c0.shape[0])

        candidates.append((deg_ratio, int(np.max(c0)), c0, int(np.min(c0)), is_1based))

    if conn.ndim == 3:
        for s in range(conn.shape[0]):
            _prep_candidate(conn[s])
    elif conn.ndim == 2:
        _prep_candidate(conn)
    else:
        return None

    if not candidates:
        if debug:
            print(f"DEBUG: no usable tri connectivity candidates under {idx_path}", file=sys.stderr)
        return None

    # Pick lowest degenerate ratio, then highest max index, then most triangles
    candidates.sort(key=lambda t: (t[0], -t[1], -t[2].shape[0]))
    deg_ratio, max_idx, tri, min_idx, is_1based = candidates[0]

    # Remove degenerate triangles explicitly
    mask = ~((tri[:, 0] == tri[:, 1]) | (tri[:, 0] == tri[:, 2]) | (tri[:, 1] == tri[:, 2]))
    tri2 = tri[mask]

    if debug:
        print(
            f"DEBUG: loaded tri connectivity from {idx_path} "
            f"shape={tri.shape} kept={tri2.shape[0]} deg_ratio={deg_ratio:.3f} "
            f"idx_range=[{min_idx},{max_idx}] one_based={bool(is_1based)}",
            file=sys.stderr,
        )

    if tri2.shape[0] == 0:
        return None
    return tri2


def _restrict_and_remap_connectivity(conn: np.ndarray, keep_nodes: np.ndarray) -> np.ndarray:
    keep_nodes = np.asarray(keep_nodes, dtype=bool).reshape(-1)
    new_index = -np.ones(keep_nodes.size, dtype=np.int64)
    new_index[keep_nodes] = np.arange(int(keep_nodes.sum()), dtype=np.int64)

    conn = np.asarray(conn, dtype=np.int64)
    bad = (conn < 0) | (conn >= keep_nodes.size)
    conn_safe = conn.copy()
    conn_safe[bad] = -1
    conn2 = np.where(conn_safe >= 0, new_index[conn_safe], -1)
    ok = np.all(conn2 >= 0, axis=1)
    return conn2[ok]


def _filter_finite_nodes_and_remap_triangles(
    r: np.ndarray,
    z: np.ndarray,
    v: np.ndarray,
    triangles: Optional[np.ndarray],
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Drop non-finite nodes. If triangles are provided, also drop triangles touching non-finite nodes and remap."""
    r = np.asarray(r, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float).reshape(-1)

    finite = np.isfinite(r) & np.isfinite(z) & np.isfinite(v)
    n_finite = int(finite.sum())
    if debug:
        n_tot = int(r.size)
        vfin = v[finite]
        if vfin.size:
            print(
                f"DEBUG: finite nodes {n_finite}/{n_tot}; v range [{np.nanmin(vfin):.6g},{np.nanmax(vfin):.6g}]",
                file=sys.stderr,
            )
        else:
            print(f"DEBUG: finite nodes {n_finite}/{n_tot}; all values non-finite", file=sys.stderr)

    if n_finite < 3:
        raise ValueError("Not enough finite nodes to plot (need >=3).")

    if triangles is None:
        return r[finite], z[finite], v[finite], None

    tri = np.asarray(triangles, dtype=np.int64)
    if tri.size == 0:
        return r[finite], z[finite], v[finite], None

    # Keep only triangles whose vertices are all finite.
    ok = np.all(finite[tri], axis=1)
    tri_ok = tri[ok]
    if tri_ok.size == 0:
        if debug:
            print("DEBUG: all triangles touch non-finite nodes; falling back to Delaunay", file=sys.stderr)
        return r[finite], z[finite], v[finite], None

    used = np.unique(tri_ok.ravel())
    # Map used global indices -> [0..nused-1]
    new_index = -np.ones(r.size, dtype=np.int64)
    new_index[used] = np.arange(used.size, dtype=np.int64)
    tri_new = new_index[tri_ok]
    if debug:
        print(f"DEBUG: triangles kept {tri_new.shape[0]}/{tri.shape[0]} after finite-mask", file=sys.stderr)
    return r[used], z[used], v[used], tri_new


# ----------------- main -----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dd", required=True, help="Top directory of IMAS HDF5 tree (e.g. mast)")
    ap.add_argument("--dd-version", required=True)
    ap.add_argument("--pulse", required=True, type=int)
    ap.add_argument("--run", required=True, type=int)
    ap.add_argument("--occ", required=True, type=int)
    ap.add_argument("--ids", required=True, help="IDS name (e.g. mhd)")
    ap.add_argument("--time-index", default=0, type=int)
    ap.add_argument("--quantity", required=True, help="Quantity or leaf (e.g. ni, te, ne, jphi, electrons&temperature)")
    ap.add_argument("--ion-index", type=int, default=0, help="Ion index for ion[] quantities (0-based)")
    ap.add_argument("--phi-index", default=0, type=int)
    ap.add_argument("--phi-tol", default=1e-6, type=float)
    ap.add_argument("--min-points", default=200, type=int)
    ap.add_argument("--out", default=None, help="Output PNG path (optional)")
    ap.add_argument("--show", action="store_true", help="Show plot interactively")
    ap.add_argument("--debug", action="store_true")

    ap.add_argument("--dedup-rz", action="store_true", help="De-duplicate near-identical (R,Z) before triangulation")
    ap.add_argument("--dedup-tol", type=float, default=1e-10)

    ap.add_argument("--use-connectivity", dest="use_connectivity", action="store_true", help="Use triangle connectivity if available")
    ap.add_argument("--no-connectivity", dest="use_connectivity", action="store_false", help="Force Delaunay triangulation")
    ap.set_defaults(use_connectivity=True)

    ap.add_argument("--mask-flat-tris", action="store_true")
    ap.add_argument("--min-circle-ratio", type=float, default=0.01)
    args = ap.parse_args()

    base = _base(args.dd, args.dd_version, args.pulse, args.run)
    ids_file, grp = _ids_file(base, args.ids, args.occ)

    if args.debug:
        print(f"[{VERSION}] file={ids_file} group=/{grp}")

    leaf_cands = _leaf_candidates(args.ids, args.quantity)

    with h5py.File(ids_file, "r") as f:
        if f"/{grp}" not in f:
            raise RuntimeError(f"Missing group /{grp} in {ids_file}")

        leaf_used, vpath = _find_values_dataset(f, grp, leaf_cands, debug=args.debug)
        v = _read_values(f, grp, leaf_used, vpath, args.time_index, args.ion_index, debug=args.debug)

        r, z, phi = _extract_geometry(f, grp, debug=args.debug)
        r, z, phi, v = _match_geometry_to_values(r, z, phi, v, args.phi_index, args.phi_tol, debug=args.debug)

        r2, z2, v2, phi_used, nsel, idx_sel = _select_phi_plane(
            r, z, phi, v,
            phi_index=args.phi_index,
            phi_tol=args.phi_tol,
            min_points=args.min_points,
            debug=args.debug,
        )

        if args.dedup_rz:
            r2, z2, v2 = _dedup_rz(r2, z2, v2, tol=args.dedup_tol, debug=args.debug)

        title = f"{args.ids} occ={args.occ} t_idx={args.time_index} {args.quantity} (phi~{phi_used:g}, n={nsel})"

        # Connectivity-based triangles if available and applicable
        triangles = None
        if args.use_connectivity:
            conn = _load_tri_connectivity(f, grp, nsel, debug=args.debug)
            if conn is not None:
                keep = np.zeros(r.size, dtype=bool)
                keep[idx_sel] = True
                conn2 = _restrict_and_remap_connectivity(conn, keep)
                if conn2.size:
                    triangles = conn2

        # Drop NaN/Inf nodes and (if using connectivity) also drop/rewire triangles.
        r3, z3, v3, tri3 = _filter_finite_nodes_and_remap_triangles(
            r2, z2, v2, triangles, debug=args.debug
        )
        triangles = tri3

        if args.debug and triangles is not None:
            print(f"DEBUG: triangles to plot: {triangles.shape[0]}", file=sys.stderr)

        # Update title with final point count.
        title = f"{args.ids} occ={args.occ} t_idx={args.time_index} {args.quantity} (phi~{phi_used:g}, n={r3.size})"

        _plot_tricontour(
            r3, z3, v3, title,
            outpng=args.out,
            show=args.show,
            triangles=triangles,
            mask_flat_tris=args.mask_flat_tris,
            min_circle_ratio=args.min_circle_ratio,
        )

    if args.out and not args.show:
        print(f"Wrote {args.out}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        raise SystemExit(2)
