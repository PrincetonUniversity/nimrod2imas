# NIMROD ⇄ IMAS Tools

This repository provides a small, script-oriented Python toolkit for:
- Converting NIMROD inputs (GEQDSK + p-file and selected namelists) to IMAS (`input2imas.py`)
- Converting NIMROD dump files to IMAS (`dump2imas.py`)
- Restoring NIMROD-style input files from IMAS entries (`nimrodInputRestore.py`)
- Plotting IMAS `mhd` and `mhd_linear` content (`plot_mhd.py`, `plot_mhd_linear.py`)
- Validating round-trip reconstruction of GEQDSK / p-file from IMAS (`validate_nimrod2imas.py`)
- Shared helpers for consistent IMAS entry handling and namelist XML encoding/decoding (`nimrod2imas.py`)

The design goal is **workflow consistency**: all scripts can target the same filesystem-backed IMAS entry directory and reuse the same IMAS open/get/put logic and namelist serialization rules.

---

## Requirements

Install the Python dependencies listed in `requirement.txt`:

```bash
python -m pip install -r requirement.txt
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
- `dd_version_dir` is derived from the DD version:
  - `"major"`: `4.1.1 → 4`
  - `"full"` : `4.1.1 → 4.1.1`
- `pulse` and `run` identify the entry

This layout is centralized in `nimrod2imas.py` and should be used by **all** scripts for consistency.

---

## Common CLI options (recommended pattern)

Most scripts follow the same argument set for selecting the entry:

- `--dbpath <path>`: root directory containing the DB layout (default: `.`)
- `--dd <name>`: DB/device directory name
- `--dd-version <x.y.z>`: IMAS Data Dictionary (DD) version to use
- `--dd-version-dir {major,full}`: directory convention for the DD version (default: `major`)
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
  --backend hdf5 --dbpath /path/to/dbroot \
  --dd-version-dir major
```

**Namelist input paths**
```bash
  --nimeq nimeq.in --oculus oculus.in --fluxgrid fluxgrid.in --nimrod nimrod.in
```

---

### 2) `dump2imas.py` — dump files → IMAS (`mhd` and/or `mhd_linear`)

**What it does**
- Reads NIMROD dump files (e.g. `dumpgll.*.h5`) and writes:
  - `mhd_linear` IDS for linear perturbations (toroidal modes)
  - `mhd` IDS for nonlinear / full-field content (depending on your workflow and script settings)
- Stores the parsed `nimrod.in` namelist as XML in IMAS code parameters:
  - For linear runs: in `mhd_linear.code.parameters` (code.name=`nimrod`)
  - For nonlinear runs: also store in `mhd.code.parameters`

**Usage (typical)**
```bash
python dump2imas.py dumpgll.0000*.h5 \
  --dd mast --dd-version 4.1.1 --pulse 45272 --run 8 \
  --backend hdf5 --dbpath /path/to/dbroot \
  --dd-version-dir major
```

---

### 3) `nimrodInputRestore.py` — IMAS → NIMROD-style inputs

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
  --backend hdf5 --dbpath /path/to/dbroot \
  --dd-version-dir major
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

### 4) `plot_mhd_linear.py` — contour plots from `mhd_linear`

Provides R–Z contour plots of:
- scalar perturbations: `p`, `t`, `n`
- vector perturbations: `b`, `v` with components `r|z|phi`
- parts: `real`, `imag`, `amp`

**Example**
```bash
python plot_mhd_linear.py \
  --dd mast --dd-version 4.1.1 --pulse 45272 --run 8 --occ 1 \
  --field b --component r --part real --time-index 0 --n-tor 5
```

Optional: `cmasher` colormaps can be used via `--cmap cmr.gothic` if installed.

---

### 5) `plot_mhd.py` — contour plots from `mhd` (GGD) with HDF5 fallback

Attempts to read axes from IMAS `mhd.grid_ggd`; if not possible, falls back to auxiliary HDF5 datasets written by the conversion pipeline (when present).

**Example**
```bash
python plot_mhd.py --entry mast/4/45272/8/ \
  --dd-version 4.1.1 --occ 1 --time-index 0 --phi-index 0 --quantity te
```

---

### 6) `validate_nimrod2imas.py` — validate GEQDSK/p-file round trip

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
