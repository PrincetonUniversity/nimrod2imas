# NIMROD ⇄ IMAS Tools

This repository provides a small, script-oriented Python toolkit for:
- Converting NIMROD inputs (GEQDSK + p-file and selected namelists) to IMAS (`input2imas.py`)
- Converting NIMROD dump files to IMAS (`dump2imas.py`)
- Computing NIMROD linear growth rates (and optional frequency) from `energy.bin`/`logen.bin` and storing them in IMAS `mhd_linear` (`gamma2imas.py`)
- Restoring NIMROD-style input files from IMAS entries (`nimrodInputRestore.py`)
- Plotting IMAS **profiles** (`plot_profiles_1d.py`)
- Plotting IMAS `mhd` and `mhd_linear` content (`plot_mhd.py`, `plot_mhd_linear.py`)
- Validating round-trip reconstruction of GEQDSK / p-file from IMAS (`validate_nimrod2imas.py`)
- Shared helpers for consistent IMAS entry handling and namelist XML encoding/decoding (`nimrod2imas.py`)

The design goal is **workflow consistency**: all scripts can target the same filesystem-backed IMAS entry directory and reuse the same IMAS open/get/put logic and namelist serialization rules.

---

## Requirements

Install the Python dependencies listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The repository also provides a `constraints.txt`, install with constraints so that indirect dependencies are added consistently across environments:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt -c constraints.txt
```

Recommended convention:
- `requirements.txt`: direct project dependencies
- `constraints.txt`: indirect/transitive packages and environment-specific compatibility

This is particularly useful for packages pulled in by `omfit-classes` and related tooling.

`omfit-classes` must be available to read/write GEQDSK and p-files (Osborne format).

> Note: Some scripts are compatible with “IMAS-Python-only” (URI filesystem backend). Others may also support “IMAS-Core/HLI” environments. Prefer the filesystem-backed mode unless you explicitly need HLI.

---

## IMAS entry layout (filesystem backend)

All tools that read/write IMAS via URI expect the same on-disk layout:

```
<dbpath>/<dd>/<dd_version_dir>/<pulse>/<run>/
```

Where:
- `dbpath` is the DB root directory (often `.` or a shared project root)
- `dd` is the database name (often a device name, e.g. `mast`, `nstx`, `d3d`)
- `dd_version_dir` is derived from the DD version as its **major** number (e.g. `4.1.1 → 4`).
  - If your DB uses a non-standard layout (e.g. the full DD version as a directory), use `--entry` to point to the entry explicitly.
- `pulse` and `run` identify the entry

This layout is centralized in `nimrod2imas.py` and should be used by **all** scripts for consistency.

---

## Common CLI options (recommended pattern)

Most scripts follow the same argument set for selecting the entry:

- `--dbpath <path>`: root directory containing the DB layout (default: `.`)
- `--dd <name>`: DB/device directory name
- `--dd-version <x.y.z>`: IMAS Data Dictionary (DD) version to use
- `--pulse <int>` and `--run <int>`
- `--backend {hdf5,mdsplus}`: filesystem backend is typically `hdf5`

Some scripts also support:
- `--entry <path>`: explicit entry directory; overrides `--dbpath/--dd/...`
- `--writer {auto,h5py,imas}`: for scripts such as `dump2imas.py`, select whether IDS content is written via direct HDF5 patching or via native IMAS-Python objects.

---

## Scripts

### 1) `input2imas.py` — inputs → IMAS (equilibrium/core_profiles/wall + namelist XML)

**What it does**
- Reads `GEQDSK` and builds `equilibrium` IDS (+ `wall` from limiter outline)
- Reads `PEQDSK` (Osborne p-file) and builds `core_profiles` IDS
- Optionally parses NIMROD-related namelists and stores them as XML blobs in IMAS code parameters:
  - `nimeq.in`, `oculus.in`, `fluxgrid.in` → `equilibrium.code.parameters` (code.name=`fgnimeq`)
  - `nimrod.in` → ideally `mhd.code.parameters` (code.name=`nimrod`) with fallback to `core_profiles.code.parameters`

**Usage (typical)**
```bash
python input2imas.py GEQDSK PEQDSK \
  --dd mast --dd-version 4.1.1 --pulse 45272 --run 8 \
  --backend hdf5 --dbpath /path/to/dbroot
```

**Namelist input paths**
```bash
python input2imas.py GEQDSK PEQDSK \
  --dd mast --dd-version 4.1.1 --pulse 45272 --run 8 \
  --nimeq nimeq.in --oculus oculus.in --fluxgrid fluxgrid.in --nimrod nimrod.in
```

---

### 2) `dump2imas.py` — dump files → IMAS (equilibrium/core_profiles/edge_profiles + `mhd_linear` / `mhd`)

**What it does**
- Reads NIMROD dump files (e.g. `dumpgll.*.h5`) and writes:
  - `equilibrium` (2D stitched grid: R, Z, ψ, B, …)
  - `core_profiles` **profiles_1d** (flux-surface averages in the closed-flux region)
  - `edge_profiles` **profiles_1d** (flux-surface averages including SOL/PF when present)
  - `mhd_linear` (linear perturbations by toroidal mode) and/or `mhd` (full-field content / GGD for nonlinear runs)
- Stores the parsed `nimrod.in` namelist as XML in IMAS code parameters:
  - For linear runs: `mhd_linear.code.parameters` (code.name=`nimrod`)
  - For nonlinear runs: also store in `mhd.code.parameters`

#### LCFS / separatrix identification (recent change)

When building 1D profiles, the converter needs **ψ_axis** and **ψ_LCFS** to construct normalized coordinates.
The logic is:

1. **`contours.h5` (preferred)**  
   If present, LCFS is determined from the LCFS polyline in the file by sampling the NIMROD ψ(R,Z) on the polyline points.
2. **`peqdsk` (fallback)**  
   If present, Te at normalized poloidal flux **ψ_N≈1** is read from the p-file and used to locate ψ_LCFS on the NIMROD fields.
3. **IMAS occ=0 fallback**  
   If neither file is available, the converter attempts to use ψ_axis/ψ_boundary from `equilibrium` and/or `core_profiles` (occurrence 0).
4. **User-provided Te at separatrix (last resort)**  
   If nothing else is available, a user-provided `--te-sep-ev` is used, with explicit logging. Default: **60 eV**.

**Optional inputs (defaults match common filenames)**
- `--contours contours.h5`
- `--peqdsk peqdsk`
- `--te-sep-ev 60.0`

File resolution order is:
1) explicit CLI value (absolute, or relative to the dump directory), 2) `<dump_dir>/<default_name>`, 3) `./<default_name>`.

#### 1D profile coordinate conventions (recent change)

- **core_profiles.profiles_1d.grid**  
  Stored primarily as a function of **normalized toroidal flux** via `rho_tor_norm` (0 at axis, 1 at LCFS).  
  The converter may also store auxiliary normalized poloidal coordinates (`psi_norm`, `rho_pol_norm`) for plotting/debug if the DD supports them.

- **edge_profiles.profiles_1d.grid**  
  Stored as a function of **normalized poloidal flux** (`psi_norm` / `rho_pol_norm`), with:
  - 0 at the magnetic axis
  - 1 at the LCFS
  - **>1 allowed** to represent **SOL/PF** data (not truncated to 1 and not forced to zeros)

To control how far outside the LCFS the 1D grid extends:
- `--edge-psi-norm-max <float>` (explicit maximum ψ_pol_norm)
- `--edge-psi-norm-quantile <float>` (robust auto-detection from 2D ψ_pol_norm distribution; default 0.9995)

**Usage (typical)**
```bash
python dump2imas.py dumpgll.0000*.h5   --dd d3d --dd-version 4.1.1 --pulse 163518 --run 1   --backend hdf5
```

#### Write implementation selection (recent change)

`dump2imas.py` now separates the **IMAS backend** from the **writer implementation** used to populate IDS content:

- `--backend {hdf5,ascii,netcdf,...}`
  - Selects the IMAS storage backend.
- `--writer {auto,h5py,imas}`
  - `auto`: use direct HDF5 patching only when the backend is HDF5; otherwise use IMAS-Python objects.
  - `h5py`: force direct HDF5 patching (fast, HDF5-only).
  - `imas`: force native IMAS-Python writing (DD-aware; required for non-HDF5 backends, typically slower for large GGD writes).

Examples:

```bash
# Native IMAS-Python path
python dump2imas.py dumpgll.*.h5   --dd d3d --dd-version 4.1.1 --pulse 163518 --run 77   --backend hdf5 --writer imas

# Direct HDF5 patch path
python dump2imas.py dumpgll.*.h5   --dd d3d --dd-version 4.1.1 --pulse 163518 --run 77   --backend hdf5 --writer h5py
```

#### Additional flags (profiles, units, time, and GGD)

**Occurrences**
- `--occ-base <int>`: base occurrence used for `equilibrium`, `core_profiles`, `edge_profiles`, and `mhd`/`mhd_linear` written by this script.
  - For multi-species content, species occurrences use `occ_base + species_index` (see log messages in the script for the exact mapping used).

**1D profile resolution**
- `--nbins <int>`: number of bins used for 1D profile construction (flux-surface averages are accumulated into this number of radial bins).

**Unit / scaling factors**
NIMROD inputs can be in code units or device-dependent units. The following multiplicative factors are applied when mapping dump quantities into IMAS:
- `--p-scale <float>`: pressure-like quantities (example: kPa → Pa uses `1e3`)
- `--T-scale <float>`: temperature-like quantities (example: keV → eV uses `1e3`)
- `--n-scale <float>`: number-density-like quantities (example: cm^-3 → m^-3 uses `1e6`)
- `--B-scale <float>`: magnetic field (example: Gauss → Tesla uses `1e-4`)
- `--L-scale <float>`: length (example: cm → m uses `1e-2`)
- `--v-scale <float>`: velocity (example: cm/s → m/s uses `1e-2`)
- `--j-scale <float>`: current density (example: A/cm^2 → A/m^2 uses `1e4`)

**Time handling**
- `--time <float>`: override time written into IDS time slices (single value applied to all dumps). If not set, the converter uses `dumpTime.vsTime` when available.

**Density perturbation packing and assumptions (linear runs)**
Some dumps store density perturbations packed as `(ny, nx, nspec*nmodes)`.
- `--dens-pert-order {species_major, mode_major}`: how `rend/imnd` are packed.
- `--ion-mass-amu <float>`: ion mass used to convert **number density** perturbation to **mass density** perturbation.
- `--electrons-index <int>`: species index in `nq/rend/imnd` corresponding to electrons (used to pair density with Te perturbation).

**MHD GGD output controls (nonlinear runs)**
These flags control how `mhd.grid_ggd` is constructed when the nonlinear pathway is active.
- `--ggd-nbins <int>`: downsample poloidal-plane fields onto a regular R–Z grid with this many bins in each direction.
- `--ggd-nphi <int>`: number of toroidal angle samples (uniform in `[0,2π)`) used to reconstruct full 3D fields from Fourier modes.
  - Set `--ggd-nphi 1` to store a single toroidal cut at `φ=0`.

**edge_profiles GGD content controls**
- `--edge-ggd-values {equilibrium,full}`:
  - `equilibrium` (default): write equilibrium-like fields.
  - `full`: include perturbations when available.
- `--edge-eq-add-pert`: if set, `edge_profiles.ggd` is constructed as *equilibrium + scaled perturbation* (when perturbations exist).
- `--pert-scale <float>`: scale factor applied to perturbations when `--edge-eq-add-pert` is enabled (primarily for visualization/debug).
- `--edge-pert-phi <float>`: toroidal angle (radians) at which perturbations are sampled when constructing `edge_profiles.ggd` in **structured** mode.

#### Unstructured `ggd` mode (recommended for robust downstream consumption)

By default, the converter may rely on **implicit structured axes** (regular R–Z resampling and toroidal replication).
For workflows that need explicit node coordinates and explicit connectivity (e.g., robust reconstruction of array shapes, ML pipelines, and ParaView/IMAS-ParaView conversion), enable **unstructured GGD**.

- `--ggd-unstructured`
  - Stores explicit per-node coordinates in `grid_ggd.space`.
  - For current full-object output, node geometry is written in cylindrical **`(R, φ, Z)`** ordering.
  - Optionally stores explicit connectivity in `grid_ggd.grid_subset`.

- `--ggd-unstructured-fe-nodes`
  - With `--ggd-unstructured`, export the native stitched NIMROD finite-element node locations as the node set (no poloidal resampling).
  - This is the preferred choice when you want IMAS to preserve the original NIMROD FE geometry as closely as possible.

- `--ggd-connectivity {none,fe_pointcloud,hex,fe_tri,fe_wedge}`
  - `none`: do not write connectivity.
  - `fe_pointcloud`: write nodes only and omit connectivity/cells (good for very large meshes or point-cloud workflows).
  - `fe_tri`: toroidally connected **surface** triangulation based on the stitched `(R,Z)` FE node lattice. This is useful for surface visualization and for constructing `fe_wedge`, but it is **not** a volumetric 3D cell representation.
  - `fe_wedge`: volumetric wedge (triangular-prism) connectivity obtained by extruding the `fe_tri` connectivity between adjacent toroidal planes. This is the recommended option for a true 3D toroidal mesh in ParaView.
  - `hex`: hexahedral connectivity on the reconstructed `(R,Z,φ)` product grid with periodicity in φ. Use this only for regular-grid workflows; it is not the native FE-preserving representation.

- `--ggd-representation {packed,full}`
  - `full` (default): write the DD4-style object-based `grid_ggd.space[].objects_per_dimension[]` representation. This is the recommended mode for ParaView/IMAS-ParaView and for downstream tools that need explicit node/cell objects.
  - `packed`: compact representation using packed subsets/elements. Keep this only for specialized legacy consumers.

- `--ggd-write-full-objects`
  - Explicitly request the IMAS-standard object-based unstructured GGD representation under `grid_ggd.space[0].objects_per_dimension`.
  - This is the recommended switch for IMAS-ParaView compatibility.
  - Use this as an alternative to packed-only output; when changing representation, write to a fresh IMAS entry.

- `--ggd-write-once`
  - Write grid + connectivity only for the first processed dump/time slice.
  - Later GGD field slices are still written, but they reuse/reference the first grid instead of storing another copy.

- `--ggd-reuse-grid`
  - Assume grid and connectivity are invariant over time.
  - For each later processed dump/time slice, copy the first `grid_ggd` grid/connectivity into the new slice instead of reconstructing it.

- `--ggd-reuse-grid-via {auto,h5py,imas}`
  - Select how `--ggd-reuse-grid` performs the copy.
  - `auto`: follow `--writer/--backend`.
  - `h5py`: force HDF5-level copy of the first persisted `grid_ggd`.
  - `imas`: force native IMAS-Python object copying.

**Interaction between `--ggd-write-once` and `--ggd-reuse-grid`**
- The two options are treated as conflicting policies.
- If both are set, **copy semantics win**: `--ggd-reuse-grid` takes precedence and `--ggd-write-once` is ignored.

##### Recommended GGD options for ParaView / IMAS-ParaView

For the most robust ParaView workflow, prefer one of the following:

```bash
# Single stored grid_ggd, later field slices reference it
python dump2imas.py dumpgll.*.h5   --dd <device> --dd-version 4.1.1 --pulse <pulse> --run <run>   --backend hdf5 --writer h5py   --ggd-unstructured --ggd-unstructured-fe-nodes   --ggd-connectivity fe_wedge   --ggd-write-full-objects   --ggd-write-once

# Copy the first grid_ggd into later slices instead of reconstructing it
python dump2imas.py dumpgll.*.h5   --dd <device> --dd-version 4.1.1 --pulse <pulse> --run <run>   --backend hdf5 --writer imas   --ggd-unstructured --ggd-unstructured-fe-nodes   --ggd-connectivity fe_wedge   --ggd-write-full-objects   --ggd-reuse-grid --ggd-reuse-grid-via imas
```

Notes:
- The script inverts the toroidal angle consistent with COCOS=2 to COCOS=17 conversion
- `fe_wedge` is the recommended connectivity for a **3D volumetric torus**.
- `fe_tri` is a **surface** representation only; it should not be expected to produce volumetric cells in ParaView.
- `--ggd-write-full-objects` (or equivalently `--ggd-representation full`) is preferred over packed-only output for `ggd2vtk` / `imas2vtu` conversion.
- When changing GGD layout or connectivity options, write to a **fresh IMAS run/entry**. Reusing an older HDF5 entry can leave stale `grid_ggd` leaves that confuse downstream readers.

##### ParaView / `ggd2vtk` remarks

- ParaView conversion relies on the IMAS-ParaView reader to interpret `grid_ggd` objects. If you see messages such as `vtkGeometryFilter ... Unknown cell type`, first verify that the reader is selecting the 3D cell objects rather than only 2D faces.
- For `fe_wedge`, the VTK output should contain volumetric cells, not only toroidal surfaces. If only surfaces appear, the problem is usually in the reader's interpretation of the GGD cell objects, not in the toroidal-node indexing itself.
- Because ParaView support for unstructured GGD is still sensitive to exact metadata/layout, `fe_wedge + full` is the recommended starting point for debugging and validation.

**Safety / resource control**
- `--mem-limit-gb <float>`: best-effort memory cap for the process in GB (Linux `RLIMIT_AS`). Use to reduce the risk of OS-level OOM for large GGD exports.
- `--quiet`: reduce logging.


---

### 3) `gamma2imas.py` — growth rates (and optional frequency) → IMAS `mhd_linear`

**What it does**
- Reads NIMROD linear energy diagnostics (`energy.bin` or `logen.bin`) and computes a scalar growth rate per toroidal mode (reported as `keff` in the diagnostic file).
- Writes the results into `mhd_linear.time_slice[*].toroidal_mode[*].growthrate` in IMAS.
- Optionally computes a mode frequency from a NIMROD history/nimhist binary (second positional argument) and writes `mhd_linear...frequency`.

**Computation details (matching the script)**
- Growth rate uses: \( \gamma = \frac{1}{2}\, d\ln(E)/dt \) computed on a tail window, with invalid intervals ignored.
- If `logen.bin` is used, the script converts `log10(E)` back to `ln(E)` internally.
- Frequency uses complex magnetic components and Dalton’s symmetric formula to estimate \(\omega\), then converts to Hz.

**CLI options**
Dump2imas-compatible entry selection:
- `--dd`, `--pulse`, `--run`, `--backend`, `--dbpath`, `--dd-version`, `--mode`

Gamma2imas-specific controls:
- `--occ <int>`: occurrence for the `mhd_linear` IDS to update (default: 1)
- `-n/--nsteps <int>`: number of tail time points used for statistics (growth and frequency)
- `--component {total,magnetic,kinetic}`: which energy component is used to compute growth rate
- `--endian {>,<}`: endianness for Fortran record markers and float payloads
- `--file-kind {energy,logen}`: override autodetection of file kind (otherwise inferred from filename containing `logen`)

**Usage examples**
Compute growth rates from `energy.bin` and write to `mhd_linear` occurrence 1:
```bash
python gamma2imas.py --dd mast --dd-version 4.1.1 --pulse 45272 --run 1 --occ 1 energy.bin
```

Compute growth rates from `logen.bin`:
```bash
python gamma2imas.py --dd mast --dd-version 4.1.1 --pulse 45272 --run 1 --occ 1 logen.bin
```

Compute growth rate + frequency (second positional file is a history/nimhist binary):
```bash
python gamma2imas.py --dd mast --dd-version 4.1.1 --pulse 45272 --run 1 --occ 1 energy.bin nimhist01.bin
```

---

### 4) `nimrodInputRestore.py` — IMAS → NIMROD-style inputs

**What it does**
- Opens an existing filesystem-backed IMAS entry and reconstructs:
  - `nimrod.in` from stored XML
  - `nimeq.in`, `oculus.in`, `fluxgrid.in` from stored XML (when present)
- Writes files with **explicit restored names** to avoid clobbering originals:
  - `nimrod_from_imas.in`
  - `nimeq_from_imas.in`
  - `oculus_from_imas.in`
  - `fluxgrid_from_imas.in`

**Usage**
```bash
python nimrodInputRestore.py \
  --dd mast --dd-version 4.1.1 --pulse 45272 --run 8 \
  --backend hdf5 --dbpath /path/to/dbroot
```

#### String reconstruction semantics (important)

When namelists are stored in XML, values are serialized as token streams. For correct round-trip behavior, the shared serializer in `nimrod2imas.py` applies these rules:

1. **Strings containing whitespace are quoted** with double quotes when written to XML  
   Example stored token: `"shear alf   mult"`

2. When restoring:
   - If the value is a *single quoted string*, it is written back as one string:
     ```fortran
     init_type = "shear alf   mult"
     ```
   - If the value is an *array/list of tokens*, it is restored as a list:
     ```fortran
     ds_function = 'ds_diff', 'ds_diff', 'ds_diff', 'ds_diff'
     ```

3. The restore logic removes spurious nested quoting such as:
   ```fortran
   init_type = '"shear alf   mult"'
   ```
   and produces:
   ```fortran
   init_type = "shear alf   mult"
   ```

If a specific variable is still being misclassified (scalar-with-spaces vs array-of-strings), it usually indicates one of:
- the original namelist parser emitted an array instead of a scalar, or
- the XML tokenization step lost quotation boundaries.

Fix this in the shared `nimrod2imas.value_to_string(...)` and the corresponding restore parser so that *all* scripts benefit.

---

### 5) `plot_profiles_1d.py` — 1D profile plots from `core_profiles` / `edge_profiles`

Plots `profiles_1d` from IMAS:
- `core_profiles`: x-axis = **ρ_tor_norm** (normalized toroidal flux coordinate)
- `edge_profiles`: x-axis = **ψ_pol_norm** (normalized poloidal flux coordinate, can exceed 1)

The script also applies human-friendly axis labels (aliases) for common quantities (e.g. `Te`, `ne`, `Ti`, `j_tor`, …).

**Example**
```bash
python plot_profiles_1d.py \
  --dd d3d --dd-version 4.1.1 --pulse 163518 --run 1 --occ 1 \
  --ids core_profiles --quantity te --show
```

---

### 6) `plot_mhd_linear.py` — contour plots from `mhd_linear`

Provides R–Z contour plots of:
- scalar perturbations: `p`, `t`, `n`
- vector perturbations: `b`, `v` with components `r|z|phi`
- parts: `real`, `imag`, `amp`

**Example**
```bash
python plot_mhd_linear.py \
  --dd mast --dd-version 4.1.1 --pulse 45272 --run 8 --occ 1 \
  --quantity b --component r --part real --time-index 0 --n-tor 5
```

Optional: `cmasher` colormaps can be used via `--cmap cmr.gothic` if installed.

---

### 7) `plot_mhd.py` — contour plots from `mhd` (GGD) with HDF5 fallback

Attempts to read axes and connectivity from IMAS `mhd.grid_ggd`; if not possible, falls back to auxiliary HDF5 datasets written by the conversion pipeline (when present).

Recent updates improved compatibility with the DD4-style full-object GGD representation written by `dump2imas.py`:
- node geometry can be read from `grid_ggd.space[].objects_per_dimension[]`
- node-centered and cell-centered data are distinguished through `grid_subset_index`
- full-object node/cell connectivity is preferred when present

When reading newer full-object exports, note that GGD node geometry is stored in cylindrical `(R, φ, Z)` order. `plot_mhd.py` handles this internally, but external readers should not assume `(R, Z, φ)`.

**Example**
```bash
python plot_mhd.py --entry mast/4/45272/8/   --dd-version 4.1.1 --occ 1 --time-index 0 --phi-index 0 --quantity te
```

---

### 8) `validate_nimrod2imas.py` — validate GEQDSK/p-file round trip

Reads `equilibrium` and `core_profiles` from IMAS and regenerates:
- a GEQDSK (using an original template)
- an Osborne p-file

It also compares original vs reconstructed profiles (max/rms differences) and reconstructs omega-related quantities using midplane geometry derived from GEQDSK.

---

## Developer notes

### Avoiding duplication
Common logic (DB open/get/put, directory layout, and namelist value serialization) should live in `nimrod2imas.py`. Scripts should import and reuse these helpers rather than implementing local copies.

### IMAS versioning
Some environments require setting `IMAS_VERSION` or passing `dd_version` directly to `DBEntry`. The `nimrod2imas.open_dbentry(...)` helper handles the most common variants.

---

## Troubleshooting

- **`ImportError: cannot import name 'imasdef'`**  
  You are likely using an IMAS-Python-only environment (no IMAS-Core/HLI). Use the filesystem-backed URI mode (`--dbpath/--dd/...` or `--entry`) rather than HLI-style `imasdef` constants.

- **Missing limiter/wall or empty IDS**  
  Check that GEQDSK includes limiter arrays (`LIMITR`, `RLIM`, `ZLIM`) and that you used the expected occurrence (`--occ`) when reading.

- **ParaView shows isolated toroidal planes or `vtkGeometryFilter ... Unknown cell type`**  
  Start with `--ggd-unstructured --ggd-unstructured-fe-nodes --ggd-connectivity fe_wedge --ggd-write-full-objects`. Verify that the IMAS-ParaView reader is selecting 3D cell objects, not only 2D faces. Also make sure you are writing to a fresh IMAS entry after changing GGD options.

- **`ALBackendException = Unable to extend the existing dataset` on later `mhd` slices**  
  This usually indicates that the entry already contains an incompatible `grid_ggd` layout from an earlier run, or that the GGD policy changed between runs. Write to a fresh IMAS run/entry when switching among reconstruction, `--ggd-write-once`, and `--ggd-reuse-grid` workflows.

- **`plot_mhd.py` shows distorted contours after switching to full-object GGD**  
  Update to the current plotting script so that it correctly interprets cylindrical node geometry ordering and resolves `grid_subset_index` against the actual `grid_subset[].identifier.index` values.

---
