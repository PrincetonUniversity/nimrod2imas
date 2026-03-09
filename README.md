# NIMROD ⇄ IMAS Tools

This repository provides a small, script-oriented Python toolkit for:
- Converting NIMROD inputs (GEQDSK + p-file and selected namelists) to IMAS (`input2imas.py`)
- Converting NIMROD dump files to IMAS (`dump2imas.py`)
- Computing NIMROD linear growth rates (and optional frequency) from `energy.bin`/`logen.bin` and storing them in IMAS `mhd_linear` (`gamma2imas.py`)
- Converting NIMROD binary time-history files (`energy*.bin`, `discharge*.bin`) into IMAS `summary` + `mhd_linear` (`bin2imas.py`)
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
python dump2imas.py dumpgll.0000*.h5 \
  --dd d3d --dd-version 4.1.1 --pulse 163518 --run 1 \
  --backend hdf5  
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
For workflows that need explicit node coordinates and explicit connectivity (e.g., robust reconstruction of array shapes, ML pipelines, or backends that store packed value arrays), enable **unstructured GGD**:

- `--ggd-unstructured`
  - Stores explicit per-node coordinates `(R,Z,φ)` in `grid_ggd.space`.
  - Optionally stores explicit connectivity in `grid_ggd.grid_subset`.

- `--ggd-unstructured-fe-nodes`
  - With `--ggd-unstructured`, export the native stitched NIMROD finite-element node locations `(R,Z)` as the node set (no poloidal resampling).

- `--ggd-connectivity {none,fe_pointcloud,hex,fe_tri,fe_wedge}` (default: `fe_tri`)
  - `none`: do not write connectivity.
  - `fe_pointcloud`: write nodes only and omit connectivity/cells (good for very large meshes).
  - `fe_tri`: triangulated 2D connectivity on the native stitched `(R,Z)` node lattice (two triangles per valid quad cell); replicated per toroidal plane when `--ggd-nphi > 1`.
  - `fe_wedge`: volumetric wedge (triangular-prism) connectivity obtained by extruding the `fe_tri` connectivity between adjacent toroidal planes (periodic in φ).
  - `hex`: hexahedral connectivity on the reconstructed `(R,Z,φ)` product grid with periodicity in φ (legacy/regular-grid mode).

- `--ggd-reuse-grid`
  - Assume grid and connectivity are invariant over time.
  - Write `grid_ggd` geometry/connectivity only for the first dump and reuse it for subsequent time slices (subsequent values reference `grid_index=1`).

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

### 4) `bin2imas.py` — time-history bins → IMAS (`summary` + `mhd_linear` + `disruption`)

**What it does**
- Reads standard NIMROD Fortran-record diagnostics from `energy*.bin` and `discharge*.bin`.
- Optionally reads `kpraden*.bin` and maps:
  - `qlosl` → disruption total radiated power
  - `qloso` → disruption ohmic power
- Merges restart segments by sorting and deduplicating records using `(step,time,mode)` keys.
- Writes discharge traces into `summary` (e.g. `ip`, `v_loop`, `w_mhd` where fields exist in the active DD).
- Writes per-mode time slices into `mhd_linear` and computes per-mode growth rate time history from \(E_{mag}+E_{kin}\).

**CLI options**
- `--input-dir`, `--energy-pattern`, `--discharge-pattern`: file discovery
- `--energy-files`, `--discharge-files`: explicit files (override glob discovery)
- `--summary-occ`, `--mhd-linear-occ`, `--disruption-occ`: IDS occurrences to update
- `--no-summary`, `--no-mhd-linear`, `--no-disruption`: selective output
- `--kprad-pattern` / `--kprad-files`: control `kpraden` discovery/input
- `--dry-run`: parse and report only (no IMAS write)

**Usage example**
```bash
python bin2imas.py \
  --input-dir ~/onedrive/MiRACL/D3D_IAEA_BOTH/output_nimrod \
  --dd d3d --dd-version 4.1.1 --pulse 0 --run 1 \
  --dbpath . --backend hdf5 \
  --summary-occ 0 --mhd-linear-occ 1
```

---

### 5) `nimrodInputRestore.py` — IMAS → NIMROD-style inputs

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

### 6) `plot_profiles_1d.py` — 1D profile plots from `core_profiles` / `edge_profiles`

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

### 7) `plot_mhd_linear.py` — contour plots from `mhd_linear`

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

### 8) `plot_mhd.py` — contour plots from `mhd` (GGD) with HDF5 fallback

Attempts to read axes from IMAS `mhd.grid_ggd`; if not possible, falls back to auxiliary HDF5 datasets written by the conversion pipeline (when present).

**Example**
```bash
python plot_mhd.py --entry mast/4/45272/8/ \
  --dd-version 4.1.1 --occ 1 --time-index 0 --phi-index 0 --quantity te
```

---

### 9) `validate_nimrod2imas.py` — validate GEQDSK/p-file round trip

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

---
