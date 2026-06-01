#!/usr/bin/env python3
"""plot_mhd.py (IMAS-only)

Plot a scalar GGD quantity from an IMAS HDF5 IDS (mhd, edge_profiles, etc.)
produced by dump2imas.

This script intentionally avoids any non-DD auxiliary structures (e.g. no
nimrod_unstructured group). Geometry is obtained from IMAS-standard grid_ggd
encodings:

  1) Packed grid_ggd nodes (preferred, robust across options):
     /<ids>_<occ>/grid_ggd[]&grid_subset[]&element[]&object[]&space
     using grid_subset.dimension==0 (nodes)

  2) grid_ggd space-geometry vectors (if present):
     /<ids>_<occ>/grid_ggd[]&space[]&objects_per_dimension[]&object[]&geometry

Values are read from:
  /<ids>_<occ>/ggd[]&<leaf>[]&values

Examples
  python plot_mhd.py --dd mast --dd-version 4.1.1 --pulse 45272 --run 9 --occ 1 \
      --ids mhd --quantity ni --phi-index 0 --show --debug

  python plot_mhd.py --dd mast --dd-version 4.1.1 --pulse 45272 --run 9 --occ 1 \
      --ids edge_profiles --quantity vphi --ion-index 0 --show
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.tri import TriAnalyzer
from matplotlib.colors import LogNorm, SymLogNorm

from nimrod2imas import (
    add_entry_args as _add_entry_args_common,
    resolve_entry_path as _resolve_entry_common,
    open_ids_h5 as _open_ids_h5_common,
    normalize_out_and_show as _normalize_out_and_show_common,
    VERSION as __version__
)

VERSION = __version__

# Optional: cmasher colormaps (https://cmasher.readthedocs.io/)
# If installed, importing cmasher registers its colormaps with Matplotlib (names like 'cmr.gothic').
_HAS_CMASher = False
try:
    import cmasher as cmr  # type: ignore  # noqa: F401
    _HAS_CMASher = True
except Exception:
    cmr = None  # type: ignore


# ----------------- filesystem helpers -----------------

def _open_ids_group(entry_dir, ids, occ):
    """Return (h5file, group, h5_path, group_name) for an IDS occurrence."""
    return _open_ids_h5_common(entry_dir, ids, int(occ), mode="r")


# ----------------- quantity / leaf resolution -----------------

# Canonical quantity aliases -> candidate IMAS leaf names (WITHOUT the ggd[]& prefix).
# These are intentionally short and stable; users can always pass an explicit IMAS-ish leaf
# (e.g. 'electrons&temperature' or 'ion[]&pressure') via --quantity.
_H5_Q_LEAF_ALIASES = {
    'ni': ['n_i_total', 'n_i', 'n_i_total_over_n_e'],
    'ti': ['t_i_average', 't_i'],
    'te': ['electrons&temperature', 't_e', 'te'],
    'ne': ['electrons&density', 'n_e'],
    'pe': ['electrons&pressure', 'p_e'],
    'pi': ['p_i', 'ions&pressure'],
    'jphi': ['j_phi', 'j_tor', 'current_density_phi', 'current_density_tor'],
    'jtor': ['j_tor', 'j_phi', 'current_density_tor', 'current_density_phi'],
    'j': ['j_total', 'j_phi', 'j_tor'],
    'vr': ['velocity_r', 'v_r'],
    'vz': ['velocity_z', 'v_z'],
    'vphi': ['velocity_phi', 'velocity_tor', 'v_phi'],
    'vtor': ['velocity_tor', 'velocity_phi', 'v_phi'],
}


def _h5_list_available_ggd_values(g):
    """List available ggd[]&...[]&values datasets in an IDS group.

    Returns dataset names (relative to the group) and excludes bookkeeping datasets such as '*_SHAPE'.
    """
    out = []
    for k in g.keys():
        if not isinstance(k, str):
            continue
        if not k.startswith('ggd[]&'):
            continue
        if not k.endswith('[]&values'):
            continue
        if k.endswith('_SHAPE') or k.endswith('AOS_SHAPE'):
            continue
        out.append(k)
    return sorted(out)


def _print_quantity_help(ids_name, occ, entry, g):
    print('')
    print(f"Available GGD value datasets for ids='{ids_name}', occ={occ}:")
    avail = _h5_list_available_ggd_values(g)
    if not avail:
        print('  (none found under ggd[]&...[]&values in this IDS group)')
    else:
        for k in avail:
            print(f'  - {k}')
    print('')
    print('How to plot:')
    print('  - Use --quantity with an alias (e.g. ni, te, ne, jphi, vphi) or an explicit leaf (e.g. electrons&temperature).')
    print('  - Aliases supported by this script:')
    print('      ' + ', '.join(sorted(_H5_Q_LEAF_ALIASES.keys())))
    print('')
    print('Notes:')
    print('  - For edge_profiles, ion velocities are stored under ion[]&velocity&{r,z,phi}; use --ion-index as needed.')
    print('')


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
    base = _H5_Q_LEAF_ALIASES

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

def _read_values_grid_binding(f: h5py.File, grp: str, leaf: str, values_path: str, debug: bool = False) -> Tuple[Optional[int], Optional[int]]:
    """Return (grid_index, grid_subset_index) for a GGD values leaf when available."""
    base = values_path[:-len('&values')] if values_path.endswith('&values') else values_path.rsplit('&values', 1)[0]
    gidx = None
    gsidx = None
    for suffix, name in (("&grid_index", "grid_index"), ("&grid_subset_index", "grid_subset_index")):
        p = f"{base}{suffix}"
        if p not in f:
            continue
        try:
            arr = np.asarray(f[p][()])
            val = int(np.asarray(arr).reshape(-1)[0])
            if name == 'grid_index':
                gidx = val
            else:
                gsidx = val
        except Exception:
            continue
    if debug:
        print(f"DEBUG: values binding leaf={leaf} grid_index={gidx} grid_subset_index={gsidx}", file=sys.stderr)
    return gidx, gsidx


def _read_full_object_nodes_geometry(f: h5py.File, grp: str, *, grid_index: int = 1, debug: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read node coordinates from DD4+ full-object grid_ggd encoding.

    Recent dump2imas versions store full-object geometry for cylindrical spaces as
    ``(R, phi, Z)`` (``cyl_rpz`` / RPZ order), while some older or custom files may
    use ``(R, Z, phi)``. This helper auto-detects the layout.
    """
    key = f"/{grp}/grid_ggd[]&space[]&objects_per_dimension[]&object[]&geometry"
    sh_key = f"/{grp}/grid_ggd[]&space[]&objects_per_dimension[]&object[]&geometry_SHAPE"
    if key not in f:
        return None
    arr = np.asarray(f[key][()])
    if arr.ndim != 5 or arr.shape[-1] < 2:
        return None
    gi = max(0, min(int(grid_index) - 1, arr.shape[0] - 1))
    geom = np.asarray(arr[gi, 0, 0], dtype=float)
    nobj = int(geom.shape[0])
    if sh_key in f:
        try:
            sh = np.asarray(f[sh_key][()])
            if sh.ndim == 4:
                nz = np.asarray(sh[gi, 0, 0]).reshape(-1)
                nobj = min(nobj, int(np.count_nonzero(nz > 0)))
        except Exception:
            pass
    geom = geom[:nobj]
    if geom.ndim != 2 or geom.shape[1] < 2 or geom.shape[0] == 0:
        return None

    r = geom[:, 0].astype(float, copy=False)
    if geom.shape[1] < 3:
        z = geom[:, 1].astype(float, copy=False)
        phi = np.zeros_like(r)
        return r, z, phi

    c1 = geom[:, 1].astype(float, copy=False)
    c2 = geom[:, 2].astype(float, copy=False)

    # Auto-detect RPZ vs RZP layout.
    def _phi_like(x: np.ndarray) -> tuple[float, float, float]:
        xf = np.asarray(x, dtype=float)
        xf = xf[np.isfinite(xf)]
        if xf.size == 0:
            return (0.0, 0.0, 0.0)
        xmin = float(np.min(xf)); xmax = float(np.max(xf))
        span = xmax - xmin
        score = 0.0
        # radians-like
        if xmin >= -1e-6 and xmax <= 2.0 * math.pi + 1e-3:
            score += 3.0
        # degrees-like
        if xmin >= -1e-3 and xmax <= 360.0 + 1e-3:
            score += 2.0
        # typically smaller span than Z on tokamak grids
        if span <= max(2.0 * math.pi + 1e-3, 360.0 + 1e-3):
            score += 1.0
        return (score, xmin, xmax)

    s1, mn1, mx1 = _phi_like(c1)
    s2, mn2, mx2 = _phi_like(c2)
    if s1 > s2:
        phi = c1
        z = c2
        layout = 'RPZ'
    elif s2 > s1:
        z = c1
        phi = c2
        layout = 'RZP'
    else:
        # Default to RPZ for full-object cylindrical geometry written by dump2imas.
        phi = c1
        z = c2
        layout = 'RPZ(default)'

    if debug:
        print(
            f"DEBUG: full-object node geometry nnodes={r.size} grid_index={grid_index} layout={layout} "
            f"c1=[{mn1:.6g},{mx1:.6g}] c2=[{mn2:.6g},{mx2:.6g}]",
            file=sys.stderr,
        )
    return r, z, _normalize_phi_units(phi, debug=debug)




def _resolve_grid_subset_position(
    f: h5py.File,
    grp: str,
    *,
    grid_index: int = 1,
    grid_subset_index: int | None = None,
    debug: bool = False,
) -> int:
    """Map IMAS ``grid_subset_index`` identifier.index to the positional subset slot.

    In the IMAS files written by ``dump2imas``, ``grid_subset_index`` attached to a
    quantity is an IMAS identifier value (for example 1 for nodes, 5 for cells in
    the unstructured writer), not necessarily the zero-based positional offset in
    ``grid_ggd[]&grid_subset[]``.  Plotting code must therefore resolve the
    identifier to the corresponding subset position before indexing packed subset
    datasets.
    """
    if grid_subset_index is None:
        return 0

    gsi = int(grid_subset_index)
    if gsi < 0:
        return 0

    key = f"/{grp}/grid_ggd[]&grid_subset[]&identifier&index"
    aos_key = f"/{grp}/grid_ggd[]&grid_subset[]&AOS_SHAPE"
    if key not in f:
        return gsi

    try:
        arr = np.asarray(f[key][()])
        if arr.ndim == 0:
            return 0 if int(arr) == gsi else gsi
        gi = max(0, min(int(grid_index) - 1, arr.shape[0] - 1)) if arr.ndim >= 2 else 0
        row = np.asarray(arr[gi] if arr.ndim >= 2 else arr).reshape(-1).astype(np.int64)
        if aos_key in f:
            try:
                aos = np.asarray(f[aos_key][()])
                if aos.ndim >= 2:
                    nsub = int(np.asarray(aos[gi]).reshape(-1)[0])
                    if nsub > 0:
                        row = row[:nsub]
            except Exception:
                pass
        matches = np.where(row == gsi)[0]
        if matches.size:
            pos = int(matches[0])
            if debug:
                print(f"DEBUG: resolved grid_subset_index identifier {gsi} -> subset position {pos}", file=sys.stderr)
            return pos
        # Fallback: if the identifier was already passed as a positional index, keep it.
        if 0 <= gsi < row.size:
            if debug:
                print(f"DEBUG: grid_subset_index={gsi} not found in identifier.index; treating as positional subset", file=sys.stderr)
            return gsi
    except Exception as e:
        if debug:
            print(f"DEBUG: grid subset identifier resolution failed for {key}: {e}", file=sys.stderr)
    return max(0, gsi)

def _extract_geometry_for_subset(
    f: h5py.File,
    grp: str,
    *,
    grid_index: int | None = None,
    grid_subset_index: int | None = None,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return coordinates matching the values subset.

    For node-centered data (subset 0), return node coordinates. For cell-centered
    full-object DD4 grids, return cell centroids.
    """
    gidx = 1 if grid_index is None else int(grid_index)
    gsidx0 = _resolve_grid_subset_position(
        f, grp, grid_index=gidx, grid_subset_index=grid_subset_index, debug=debug
    )

    node_geom = _read_full_object_nodes_geometry(f, grp, grid_index=gidx, debug=debug)
    if node_geom is None:
        node_geom = _packed_nodes_from_gridggd(f, grp, debug=debug)
        if node_geom is None:
            node_geom = _space_geometry_vectors(f, grp, debug=debug)
        if node_geom is None:
            raise RuntimeError(
                f"Could not find grid_ggd node geometry in /{grp}.\n"
                "Tried DD4 full-object nodes, packed nodes, and space geometry leaves."
            )
        r, z, phi = node_geom
        return np.asarray(r), np.asarray(z), _normalize_phi_units(phi, debug=debug)

    r_nodes, z_nodes, phi_nodes = node_geom
    if gsidx0 == 0:
        return r_nodes, z_nodes, phi_nodes

    k_space = f"/{grp}/grid_ggd[]&grid_subset[]&element[]&object[]&space"
    k_dim = f"/{grp}/grid_ggd[]&grid_subset[]&element[]&object[]&dimension"
    k_index = f"/{grp}/grid_ggd[]&grid_subset[]&element[]&object[]&index"
    k_nodes = f"/{grp}/grid_ggd[]&space[]&objects_per_dimension[]&object[]&nodes"
    k_nodes_sh = f"/{grp}/grid_ggd[]&space[]&objects_per_dimension[]&object[]&nodes_SHAPE"
    if not all(k in f for k in (k_space, k_dim, k_index, k_nodes)):
        return r_nodes, z_nodes, phi_nodes

    try:
        ref_space = np.asarray(f[k_space][()])
        ref_dim = np.asarray(f[k_dim][()])
        ref_index = np.asarray(f[k_index][()])
        obj_nodes = np.asarray(f[k_nodes][()])
        obj_nodes_sh = np.asarray(f[k_nodes_sh][()]) if k_nodes_sh in f else None
    except Exception:
        return r_nodes, z_nodes, phi_nodes

    if ref_space.ndim < 4 or ref_dim.ndim < 4 or ref_index.ndim < 4 or obj_nodes.ndim < 5:
        return r_nodes, z_nodes, phi_nodes

    gi = max(0, min(gidx - 1, ref_space.shape[0] - 1))
    sidx = max(0, min(gsidx0, ref_space.shape[1] - 1))
    space_ref = np.asarray(ref_space[gi, sidx]).reshape(-1)
    dim_ref = np.asarray(ref_dim[gi, sidx]).reshape(-1)
    index_ref = np.asarray(ref_index[gi, sidx]).reshape(-1)
    valid = (space_ref > 0) & (dim_ref > 0) & (index_ref > 0)
    if not np.any(valid):
        return r_nodes, z_nodes, phi_nodes

    space_ref = space_ref[valid].astype(np.int64) - 1
    dim_ref = dim_ref[valid].astype(np.int64) - 1
    index_ref = index_ref[valid].astype(np.int64) - 1
    if np.unique(space_ref).size != 1 or np.unique(dim_ref).size != 1:
        return r_nodes, z_nodes, phi_nodes

    sp = int(space_ref[0])
    dm = int(dim_ref[0])
    node_ids = np.asarray(obj_nodes[gi, sp, dm, index_ref], dtype=np.int64)
    if obj_nodes_sh is not None and obj_nodes_sh.ndim >= 4:
        counts = np.asarray(obj_nodes_sh[gi, sp, dm, index_ref]).reshape(-1).astype(np.int64)
    else:
        counts = np.sum(node_ids > 0, axis=1, dtype=np.int64)
    width = int(node_ids.shape[1])
    mask = (np.arange(width, dtype=np.int64)[None, :] < counts[:, None]) & (node_ids > 0)
    node_ids0 = np.where(mask, node_ids - 1, -1)
    rr = np.zeros(node_ids0.shape[0], dtype=float)
    zz = np.zeros(node_ids0.shape[0], dtype=float)
    pp = np.zeros(node_ids0.shape[0], dtype=float)
    for j in range(width):
        mj = node_ids0[:, j] >= 0
        if not np.any(mj):
            continue
        idx = node_ids0[mj, j]
        rr[mj] += r_nodes[idx]
        zz[mj] += z_nodes[idx]
        pp[mj] += phi_nodes[idx]
    denom = np.maximum(counts.astype(float), 1.0)
    rr /= denom
    zz /= denom
    pp /= denom
    if debug:
        print(f"DEBUG: subset geometry from centroids subset={gsidx0} dim={dm} n={rr.size}", file=sys.stderr)
    return rr, zz, pp



def _packed_nodes_from_gridggd(f: h5py.File, grp: str, debug: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read nodes from packed grid_ggd object[]&space selecting grid_subset.dimension==0."""
    dim_path = f"/{grp}/grid_ggd[]&grid_subset[]&dimension"
    real_path = f"/{grp}/grid_ggd[]&grid_subset[]&element[]&object[]&space"
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
    return _extract_geometry_for_subset(f, grp, grid_index=1, grid_subset_index=0, debug=debug)


# ----------------- values extraction -----------------

def _select_time_and_object(arr: np.ndarray, t_index: int, *, ntime: int | None = None, clamp_time: bool = False) -> np.ndarray:
    """Select time index and object index (0) from typical IMAS packed arrays.

    Important: many IDS backends store single-time values as a 1-D vector (npts,)
    rather than (1, npts). In that case, *do not* interpret axis0 as time.
    """
    a = np.asarray(arr)

    # 1-D arrays are assumed to be already flattened per-node/per-point values.
    # If the IDS group advertises a time vector, enforce that time_index is in range,
    # even if values are stored without an explicit leading time axis.
    if a.ndim == 1:
        if ntime is not None:
            ntime_i = int(ntime)
            ti = int(t_index)
            if ti < 0 or ti >= ntime_i:
                if clamp_time:
                    ti = max(0, min(ti, ntime_i - 1))
                else:
                    raise IndexError(f"time-index {ti} out of range [0,{ntime_i-1}] (ntime={ntime_i})")
        return a

    # Select time only when we have an explicit time axis. Prefer matching /<grp>/time length (ntime).
    if ntime is not None and a.ndim >= 2 and a.shape[0] == int(ntime):
        ntime_i = int(ntime)
        ti = int(t_index)
        if ti < 0 or ti >= ntime_i:
            if clamp_time:
                ti = max(0, min(ti, ntime_i - 1))
            else:
                raise IndexError(f"time-index {ti} out of range [0,{ntime_i-1}] (ntime={ntime_i})")
        a = a[ti] if ntime_i > 1 else a[0]
    else:
        # Legacy heuristic fallback (only for multi-d arrays): treat a small leading axis as time.
        if a.ndim >= 2 and a.shape[0] <= 256 and a.shape[0] > 1:
            ti = int(t_index)
            if ti < 0 or ti >= a.shape[0]:
                if clamp_time:
                    ti = max(0, min(ti, a.shape[0] - 1))
                else:
                    raise IndexError(f"time-index {ti} out of range [0,{a.shape[0]-1}] (ntime={a.shape[0]})")
            a = a[ti]
        elif a.ndim >= 2 and a.shape[0] == 1:
            a = a[0]

    # Select object if next axis looks like object (small)
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
    *,
    clamp_time_index: bool = False,
    debug: bool = False,
) -> np.ndarray:
    # Determine number of time slices from /<grp>/time if present.
    ntime: int | None = None
    tpath = f"/{grp}/time"
    if tpath in f:
        try:
            tt = np.asarray(f[tpath][()])
            ntime = int(tt.size) if tt.ndim != 0 else 1
        except Exception:
            ntime = None

    a = _select_time_and_object(f[values_path][()], time_index, ntime=ntime, clamp_time=clamp_time_index)

    # Handle ion[] multi-ion arrays if present as (nion, npts)
    if "ion[]" in leaf and a.ndim >= 2 and a.shape[0] > 1:
        ii = int(ion_index)
        if ii < 0 or ii >= a.shape[0]:
            ii = max(0, min(ii, a.shape[0] - 1))
        a = a[ii]

    a = np.asarray(a, dtype=float)
    # Important: dump2imas commonly writes shaped IMAS values arrays with Fortran-order
    # node packing consistent with geometry built via ravel(order="F"). Flattening here
    # in default C order scrambles values relative to (R,Z,phi) geometry and produces
    # nonphysical striping/oscillatory contours. Preserve Fortran packing for multidim
    # arrays; for 1-D arrays this is identical.
    if a.ndim >= 2:
        v = np.ravel(a, order="F")
    else:
        v = a.reshape(-1)

    # Mask common IMAS fill values (e.g. -9e40) and absurd magnitudes
    bad = (~np.isfinite(v)) | (np.abs(v) > 1.0e30) | (v < -8.0e39)
    if np.any(bad):
        v = v.astype(float, copy=True)
        v[bad] = np.nan

    if debug:
        print(f"DEBUG: values_path={values_path} leaf={leaf} raw_shape={a.shape} n={v.size} ntime={ntime} flatten={'F' if a.ndim >= 2 else '1d'}", file=sys.stderr)

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

    # Periodic wrap case: the same phi bin can appear in multiple contiguous blocks
    # (e.g. first and wrap-around toroidal planes both at phi=0).  If we mix those
    # blocks, duplicate (R,Z) locations with different values produce striping.
    # Prefer a single contiguous block, typically the first physical plane.
    if nsel >= min_points:
        idx_sel_all = np.nonzero(sel)[0]
        if idx_sel_all.size > 0:
            gaps = np.where(np.diff(idx_sel_all) > 1)[0]
            if gaps.size > 0:
                starts = np.r_[0, gaps + 1]
                stops = np.r_[gaps + 1, idx_sel_all.size]
                seg_lengths = stops - starts
                # Choose the longest contiguous block; tie-break to the first block.
                ib = int(np.argmax(seg_lengths))
                if int(seg_lengths[ib]) < nsel:
                    idx_block = idx_sel_all[starts[ib]:stops[ib]]
                    if debug:
                        print(
                            f"DEBUG: selected phi bin contains {len(seg_lengths)} contiguous blocks; "
                            f"using block {ib} with n={idx_block.size} (discarding periodic duplicate blocks)",
                            file=sys.stderr,
                        )
                    return (
                        np.asarray(r).reshape(-1)[idx_block],
                        np.asarray(z).reshape(-1)[idx_block],
                        np.asarray(v).reshape(-1)[idx_block],
                        phi_used,
                        int(idx_block.size),
                        idx_block,
                    )

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


def _available_cmaps():
    """Return available Matplotlib colormap names (sorted).

    If cmasher is installed, its colormaps are included automatically once imported.
    """
    try:
        return sorted(list(plt.colormaps()))
    except Exception:
        try:
            return sorted(list(plt.cm.cmap_d.keys()))  # type: ignore[attr-defined]
        except Exception:
            return []


def _default_cmap_for_data(v, norm_kind='linear'):
    # Sequential by default; diverging if signed.
    try:
        vv = v[np.isfinite(v)]
        if vv.size == 0:
            return 'viridis'
        vmin = float(vv.min())
        vmax = float(vv.max())
    except Exception:
        return 'viridis'
    if str(norm_kind).lower() == 'log':
        return 'viridis'
    if vmin < 0.0 and vmax > 0.0:
        return 'RdBu_r'
    return 'viridis'


def _resolve_cmap(cmap, v, norm_kind='linear'):
    if cmap is None or str(cmap).strip() == '':
        return _default_cmap_for_data(v, norm_kind=norm_kind)
    cmap = str(cmap).strip()
    if cmap.lower().startswith('cmr.') and not _HAS_CMASher:
        raise RuntimeError(
            f"Requested colormap {cmap!r}, but cmasher is not available in this Python environment. "
            "Install it (e.g. 'pip install cmasher') or choose a Matplotlib colormap."
        )
    avail = _available_cmaps()
    if avail and cmap not in avail:
        lower_map = {c.lower(): c for c in avail}
        if cmap.lower() in lower_map:
            return lower_map[cmap.lower()]
        raise RuntimeError(f"Unknown colormap {cmap!r}. Use --list-cmaps to see available names.")
    return cmap


def _build_norm_and_levels(v, nlevels, norm_kind='linear', linthresh=1e-6):
    norm_kind = (norm_kind or 'linear').lower()
    vv = v[np.isfinite(v)]
    if vv.size == 0:
        return None, int(nlevels)

    if norm_kind in ('linear', 'none'):
        return None, int(nlevels)

    if norm_kind == 'log':
        pos = vv[vv > 0]
        if pos.size == 0:
            raise ValueError('Log normalization requires positive data. Use --norm symlog for signed data, or plot a non-negative quantity.')
        vmin = float(pos.min())
        vmax = float(pos.max())
        if not (vmin > 0 and vmax > 0):
            raise ValueError('Log normalization requires positive finite vmin/vmax.')
        if vmax <= vmin:
            vmax = vmin * 10.0
        levels = np.logspace(np.log10(vmin), np.log10(vmax), int(nlevels))
        return LogNorm(vmin=vmin, vmax=vmax), levels

    if norm_kind == 'symlog':
        absmax = float(np.max(np.abs(vv)))
        if absmax == 0.0:
            absmax = 1.0
        lt = float(linthresh) if linthresh and linthresh > 0 else absmax * 1e-3
        return SymLogNorm(linthresh=lt, vmin=-absmax, vmax=absmax), int(nlevels)

    raise ValueError(f"Unknown --norm {norm_kind!r} (use linear|log|symlog).")


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
    cmap: Optional[str] = None,
    norm=None,
    levels=40,
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
    cs = ax.tricontourf(tri, v_ma, levels=levels, cmap=cmap, norm=norm)
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
    f: h5py.File, grp: str, n_nodes: int, debug: bool = False, *, grid_index: int = 1
) -> Optional[np.ndarray]:
    """Load triangle connectivity for node-centered plotting.

    Supports both the older packed grid_subset index tree and the DD4+ full-object
    ``space[].objects_per_dimension[].object[].nodes`` encoding used by recent
    dump2imas versions.
    """
    idx_path = f"/{grp}/grid_ggd[]&grid_subset[]&element[]&object[]&index"
    if idx_path in f:
        conn = np.asarray(f[idx_path][()])
        conn = np.asarray(conn).squeeze()
        while conn.ndim > 2 and conn.shape[0] == 1:
            conn = conn[0]

        candidates: list[tuple[float, int, np.ndarray, int, int]] = []
        def _prep_candidate(craw: np.ndarray) -> None:
            if craw.ndim != 2 or craw.shape[1] != 3:
                return
            c = np.asarray(craw)
            cmin = int(np.min(c))
            cmax = int(np.max(c))
            is_1based = 0
            if cmin >= 1 and cmax <= n_nodes:
                is_1based = 1
                c0 = c.astype(np.int64) - 1
            else:
                c0 = c.astype(np.int64)
            if c0.size == 0:
                return
            if np.min(c0) < 0 or np.max(c0) >= n_nodes:
                return
            deg = np.count_nonzero((c0[:, 0] == c0[:, 1]) | (c0[:, 0] == c0[:, 2]) | (c0[:, 1] == c0[:, 2]))
            deg_ratio = float(deg) / float(c0.shape[0])
            candidates.append((deg_ratio, int(np.max(c0)), c0, int(np.min(c0)), is_1based))

        if conn.ndim == 3:
            for s in range(conn.shape[0]):
                _prep_candidate(conn[s])
        elif conn.ndim == 2:
            _prep_candidate(conn)

        if candidates:
            candidates.sort(key=lambda t: (t[0], -t[1], -t[2].shape[0]))
            deg_ratio, max_idx, tri, min_idx, is_1based = candidates[0]
            mask = ~((tri[:, 0] == tri[:, 1]) | (tri[:, 0] == tri[:, 2]) | (tri[:, 1] == tri[:, 2]))
            tri2 = tri[mask]
            if debug:
                print(
                    f"DEBUG: loaded packed tri connectivity from {idx_path} shape={tri.shape} kept={tri2.shape[0]} "
                    f"deg_ratio={deg_ratio:.3f} idx_range=[{min_idx},{max_idx}] one_based={bool(is_1based)}",
                    file=sys.stderr,
                )
            if tri2.shape[0] > 0:
                return tri2

    nodes_path = f"/{grp}/grid_ggd[]&space[]&objects_per_dimension[]&object[]&nodes"
    sh_path = f"/{grp}/grid_ggd[]&space[]&objects_per_dimension[]&object[]&nodes_SHAPE"
    if nodes_path not in f:
        return None
    try:
        obj_nodes = np.asarray(f[nodes_path][()])
        obj_nodes_sh = np.asarray(f[sh_path][()]) if sh_path in f else None
    except Exception:
        return None
    if obj_nodes.ndim != 5 or obj_nodes.shape[2] < 3:
        return None
    gi = max(0, min(int(grid_index) - 1, obj_nodes.shape[0] - 1))
    faces = np.asarray(obj_nodes[gi, 0, 2], dtype=np.int64)
    if faces.ndim != 2 or faces.size == 0:
        return None
    if obj_nodes_sh is not None and obj_nodes_sh.ndim >= 4:
        counts = np.asarray(obj_nodes_sh[gi, 0, 2]).reshape(-1).astype(np.int64)
    else:
        counts = np.sum(faces > 0, axis=1, dtype=np.int64)
    tris: list[np.ndarray] = []
    sel3 = np.where(counts == 3)[0]
    if sel3.size:
        tris.append(faces[sel3, :3].astype(np.int64) - 1)
    sel4 = np.where(counts == 4)[0]
    if sel4.size:
        pts = faces[sel4, :4].astype(np.int64) - 1
        tris.append(np.stack([pts[:, [0, 1, 2]], pts[:, [0, 2, 3]]], axis=1).reshape(-1, 3))
    if not tris:
        if debug:
            print(f"DEBUG: no 2D face triangles/quads available under {nodes_path}", file=sys.stderr)
        return None
    tri = np.vstack(tris)
    good = np.all((tri >= 0) & (tri < n_nodes), axis=1)
    tri = tri[good]
    tri = tri[~((tri[:, 0] == tri[:, 1]) | (tri[:, 0] == tri[:, 2]) | (tri[:, 1] == tri[:, 2]))]
    if debug:
        print(f"DEBUG: loaded full-object face connectivity from {nodes_path} triangles={tri.shape[0]}", file=sys.stderr)
    return tri if tri.size else None


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
    _add_entry_args_common(ap, include_backend=False, include_ids=True, ids_default='mhd', include_occ=True, occ_default=0)
    ap.add_argument("--time-index", default=0, type=int)
    ap.add_argument("--clamp-time-index", action="store_true", help="Clamp --time-index into available range instead of erroring")
    ap.add_argument("--quantity", required=False, help="Quantity or leaf (e.g. ni, te, ne, jphi, electrons&temperature)")
    ap.add_argument("--help-quantities", action="store_true", help="Print available datasets and aliases for this IDS occurrence and exit")
    ap.add_argument("--levels", type=int, default=40, help="Number of contour levels")
    ap.add_argument("--cmap", default=None, help="Matplotlib colormap name (e.g. viridis, RdBu_r, cmr.gothic)")
    ap.add_argument("--list-cmaps", action="store_true", help="List available colormap names and exit")
    ap.add_argument("--norm", default="linear", choices=["linear", "log", "symlog"], help="Color normalization for contours")
    ap.add_argument("--linthresh", type=float, default=1e-6, help="linthresh for symlog normalization")
    ap.add_argument("--ion-index", type=int, default=0, help="Ion index for ion[] quantities (0-based)")
    ap.add_argument("--phi-index", default=0, type=int)
    ap.add_argument("--phi-tol", default=1e-6, type=float)
    ap.add_argument("--min-points", default=200, type=int)
    ap.add_argument("--out", default="X11", help="Output image path or 'X11' for interactive")
    ap.add_argument("--show", action="store_true", help="Show plot interactively (even if --out is set)")
    ap.add_argument("--debug", action="store_true")

    ap.add_argument("--dedup-rz", action="store_true", help="De-duplicate near-identical (R,Z) before triangulation")
    ap.add_argument("--dedup-tol", type=float, default=1e-10)

    ap.add_argument("--use-connectivity", dest="use_connectivity", action="store_true", help="Use triangle connectivity if available")
    ap.add_argument("--no-connectivity", dest="use_connectivity", action="store_false", help="Force Delaunay triangulation")
    ap.set_defaults(use_connectivity=True)

    ap.add_argument("--mask-flat-tris", action="store_true")
    ap.add_argument("--min-circle-ratio", type=float, default=0.01)
    args = ap.parse_args()

    if args.list_cmaps:
        avail = _available_cmaps()
        if _HAS_CMASher:
            print("cmasher: available (colormaps with prefix 'cmr.')")
        else:
            print("cmasher: not available (install with 'pip install cmasher' to enable 'cmr.*' colormaps)")
        if not avail:
            print("No colormaps discovered (unexpected).")
        else:
            print("Available colormaps:")
            for name in avail:
                print(f"  {name}")
        return 0

    if not args.help_quantities and (args.quantity is None or str(args.quantity).strip() == ''):
        raise RuntimeError('Provide --quantity (or use --help-quantities).')
    entry_dir = _resolve_entry_common(args)
    print(f"IMAS entry directory: {entry_dir}")

    f, g, ids_file, grp = _open_ids_group(entry_dir, args.ids, args.occ)
    try:
        if args.debug:
            print(f"[{__version__}] file={ids_file} group=/{grp}")

        leaf_cands = _leaf_candidates(args.ids, str(args.quantity)) if args.quantity is not None else []

        if f"/{grp}" not in f:
            raise RuntimeError(f"Missing group /{grp} in {ids_file}")

        if args.help_quantities:
            _print_quantity_help(args.ids, args.occ, str(entry_dir), g)
            return 0

        # Resolve/validate requested quantity (aliases -> leaf candidates)
        try:
            leaf_used, vpath = _find_values_dataset(f, grp, leaf_cands, debug=args.debug)
        except Exception as e:
            print(f"ERROR: Could not find datasets for {args.quantity!r}. Tried: {leaf_cands}")
            _print_quantity_help(args.ids, args.occ, str(entry_dir), g)
            raise

        v = _read_values(f, grp, leaf_used, vpath, args.time_index, args.ion_index, clamp_time_index=args.clamp_time_index, debug=args.debug)
        gidx_leaf, gsidx_leaf = _read_values_grid_binding(f, grp, leaf_used, vpath, debug=args.debug)

        r, z, phi = _extract_geometry_for_subset(
            f, grp,
            grid_index=(gidx_leaf if gidx_leaf is not None else 1),
            grid_subset_index=(gsidx_leaf if gsidx_leaf is not None else 0),
            debug=args.debug,
        )
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
        if args.use_connectivity and (gsidx_leaf in (None, 0)):
            conn = _load_tri_connectivity(
                f, grp, r.size, debug=args.debug,
                grid_index=(gidx_leaf if gidx_leaf is not None else 1),
            )
            if conn is not None:
                keep = np.zeros(r.size, dtype=bool)
                keep[idx_sel] = True
                conn2 = _restrict_and_remap_connectivity(conn, keep)
                if conn2.size:
                    triangles = conn2
        elif args.use_connectivity and args.debug:
            print(
                f"DEBUG: skipping connectivity because values are grid_subset_index={gsidx_leaf} (not node-centered)",
                file=sys.stderr,
            )

        # Drop NaN/Inf nodes and (if using connectivity) also drop/rewire triangles.
        r3, z3, v3, tri3 = _filter_finite_nodes_and_remap_triangles(
            r2, z2, v2, triangles, debug=args.debug
        )
        triangles = tri3

        if args.debug and triangles is not None:
            print(f"DEBUG: triangles to plot: {triangles.shape[0]}", file=sys.stderr)

        # Update title with final point count.
        title = f"{args.ids} occ={args.occ} t_idx={args.time_index} {args.quantity} (phi~{phi_used:g}, n={r3.size})"

        cmap_name = _resolve_cmap(args.cmap, v3, norm_kind=args.norm)
        norm, lev = _build_norm_and_levels(v3, int(args.levels), norm_kind=args.norm, linthresh=float(args.linthresh))

        outpng, do_show = _normalize_out_and_show_common(args.out, show=bool(args.show))

        _plot_tricontour(
            r3, z3, v3, title,
            outpng=outpng,
            show=do_show,
            triangles=triangles,
            mask_flat_tris=args.mask_flat_tris,
            min_circle_ratio=args.min_circle_ratio,
            cmap=cmap_name,
            norm=norm,
            levels=lev,
        )

        if outpng and not do_show:
            print(f"Wrote {outpng}")
    finally:
        try:
            f.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        raise SystemExit(2)
