# NIMROD ⇄ IMAS Utilities

This directory contains three Python utilities to move NIMROD input data into IMAS, validate the round trip, and restore the original Fortran namelists from IMAS.

Scripts:

* `nimrod2imas.py`
* `validate_nimrod2imas.py`
* `nimrodInputRestore.py`
* `dump2imas.py`
* `plot_mhd_linear_contours.py`

---

## 1. `nimrod2imas.py`

### Purpose

Ingest a NIMROD equilibrium and profiles, plus associated input namelists, into IMAS:

* Reads:

  * Magnetic equilibrium from **GEQDSK** (`geqdsk`)
  * Profiles from **PEQDSK / Osborne p-file** (`peqdsk`)
  * FGnimeq namelists: `nimeq.in`, `oculus.in`, `fluxgrid.in`
  * NIMROD namelist: `nimrod.in`
* Writes:

  * `equilibrium` IDS (geometry and equilibrium profiles)
  * `core_profiles` IDS (electron/ion densities, temperatures, flows, beam, impurities, diamagnetic velocity)
  * `wall` IDS (limiter outline from `RLIM/ZLIM` in GEQDSK)
  * optional `mhd` IDS containing the NIMROD code metadata
* Stores namelists in XML form using `f90nml` + `xml.etree.ElementTree` in:

  * `equilibrium.code.parameters` (FGnimeq inputs)
  * `mhd.code.parameters` (or `core_profiles.code.parameters` as fallback, for NIMROD inputs)

No COCOS conversion is applied: GEQDSK arrays are copied “as is” into the equilibrium IDS.

### Dependencies

* Python 3
* `numpy`
* `imas` (IMAS Python bindings)
* `omfit_classes` (`OMFITgeqdsk`, `OMFITpFile`)
* `f90nml`
* `xml.etree.ElementTree` (standard library)

### Typical usage

```bash
python3 nimrod2imas.py geqdsk peqdsk \
    --nimeq    nimeq.in \
    --oculus   oculus.in \
    --fluxgrid fluxgrid.in \
    --nimrod   nimrod.in \
    --dd nimrod --pulse 1 --run 0 --backend hdf5
```

This creates (or overwrites) the IMAS entry `(dd='nimrod', pulse=1, run=0, backend='hdf5')` with populated `equilibrium`, `core_profiles`, `wall`, and `mhd` IDSs.

---

## 2. `validate_nimrod2imas.py`

### Purpose

Validate the GEQDSK/PEQDSK → IMAS → GEQDSK/PEQDSK round trip and quantify differences:

* Reads from IMAS:

  * `equilibrium` IDS
  * `core_profiles` IDS
* Reconstructs:

  * `geqdsk_from_imas` using `equilibrium` (copying fields back into an `OMFITgeqdsk` object)
  * `peqdsk_from_imas` by writing an Osborne-style p-file directly from `core_profiles` (no OMFITpFile on output)
* Compares the **original** and **reconstructed** PEQDSK p-files using `OMFITpFile` and prints, for each profile present in the reconstructed file:

  * `max|Δval|`, `rms|Δval|`
  * `max|Δder|`, `rms|Δder|` (differences in derivatives)

The script also reconstructs kinematic quantities (e.g. `kpol`, `omeg`, `omegp`, `omgpp`, `omgvb`, `omgeb`) from the IMAS fields to check consistency with the original p-file.

### Dependencies

Same as `nimrod2imas.py` (plus `omfit_classes` for p-file handling).

### Typical usage

```bash
python3 validate_nimrod2imas.py geqdsk peqdsk \
    --db nimrod --pulse 1 --run 0 --backend hdf5 \
    --out-geqdsk geqdsk_from_imas \
    --out-peqdsk peqdsk_from_imas
```

Output:

* Writes `geqdsk_from_imas`, `peqdsk_from_imas` in the current directory.
* Prints summary lines like:

  ```
  ne     max|Δval|= ... rms|Δval|= ...  max|Δder|= ... rms|Δder|= ...
  ti     ...
  kpol   ...
  ...
  ```

Use these diagnostics to confirm everything is within acceptable tolerances.

---

## 3. `nimrodInputRestore.py`

### Purpose

Restore the FGnimeq and NIMROD Fortran namelist files from IMAS, to verify the f90nml → XML → f90nml round trip or to reuse the inputs:

* Reads from IMAS:

  * `equilibrium.code.parameters` (expected XML with `<fgnimeq_inputs>`)
  * `mhd.code.parameters` (or fallback `core_profiles.code.parameters`) (expected XML with `<nimrod_inputs>`)
* For FGnimeq:

  * Extracts `<nimeq_in>`, `<oculus_in>`, `<fluxgrid_in>` children
  * Converts each back to a `f90nml.Namelist`
  * Writes Fortran namelist files (`nimeq.in`, `oculus.in`, `fluxgrid.in` by default, or user-specified names)
* For NIMROD:

  * Extracts `<nimrod_in>` child
  * Converts back to `f90nml.Namelist`
  * Writes `nimrod.in` (or user-specified output name)

### Dependencies

* Python 3
* `imas`
* `f90nml`
* `xml.etree.ElementTree`

### Typical usage

```bash
python3 nimrodInputRestore.py \
    --dd nimrod --pulse 1 --run 0 --backend hdf5 \
    --out-nimeq    nimeq_from_imas.in \
    --out-oculus  oculus_from_imas.in \
    --out-fluxgrid fluxgrid_from_imas.in \
    --out-nimrod  nimrod_from_imas.in
```

This recreates the namelists from IMAS so you can `diff` them against the originals.

---
## 4.  `dump2imas.py`

### Purpose

`dump2imas.py` reads a NIMROD `dumpgll.*.h5` file and writes the extracted content into an IMAS database entry.

In its current form, the script **does not** populate the full IMAS physics structures (GGD grids, equilibrium IDS, core_profiles, etc.). Instead, it stores a self-contained record of the dump (metadata + arrays) as **XML embedded in `code.parameters`** in the following IDSs:

- `mhd` (occurrence `0`)
- `mhd_linear` (occurrence `0`, if the installed data dictionary provides this IDS)

Arrays are embedded as **base64-encoded float64** along with their shapes, so they can be reconstructed exactly in post-processing.

### Requirements
- Python 3
- `numpy`
- `h5py`
- IMAS Python bindings (`imas`)

Typical usage is within an environment that already provides IMAS (for example, a TRANSP/IMAS Python environment).

### Input file expectations
The script expects an HDF5 layout consistent with a NIMROD `dumpgll` file that contains:

- A top-level group: `rblocks`
- Within `rblocks`, one group per block (block id as a string, e.g. `"0001"`)
- Within each block group `rblocks/<bid>`, datasets named with the block id suffix:

| Dataset name (per block) | Expected shape | Meaning |
|---|---:|---|
| `rz<bid>`   | `(Nr, Nz, 2)` | Cylindrical coordinates `[R, Z]` |
| `bq<bid>`   | `(Nr, Nz, 3)` | Equilibrium magnetic field `[BR, BZ, Bphi]` |
| `vq<bid>`   | `(Nr, Nz, 3)` | Equilibrium velocity `[vR, vZ, vphi]` |
| `teq<bid>`  | `(Nr, Nz, 1)` or `(Nr, Nz)` | Electron temperature (eV) |
| `tiq<bid>`  | `(Nr, Nz, 1)` or `(Nr, Nz)` | Ion temperature (eV) |
| `nq<bid>`   | `(Nr, Nz, nspec)` | Species densities (m^-3) |

Optional content:
- A top-level group `diff_profiles` containing 1D datasets. These are read and stored as additional named arrays in the XML payload.

Time handling:
- If the file contains a group `dumpTime` with attribute `vsTime`, the script uses that as the dump time (seconds). Otherwise, time defaults to `0.0`.

### What gets written to IMAS
#### IDS targets
- `mhd` (occurrence 0)
- `mhd_linear` (occurrence 0; skipped if not available)

#### Where the data is stored
For each run, the script appends an XML element under:

- `mhd.code.parameters`
- `mhd_linear.code.parameters`

The XML element (`<nimrod_mhd_dump ...>`) contains:
- File name / absolute path
- Time, number of blocks, number of grid points
- A list of fields present
- `<arrays>` with base64 float64 payloads and shapes for:
  - `R`, `Z`
  - `B_eq` (Npoints x 3)
  - `V_eq` (Npoints x 3)
  - `Te_eV`, `Ti_eV`
  - `nq` (Npoints x nspec)
  - Any `diff_profiles` arrays (if present)

The data representation in IMAS is therefore **lossless** with respect to the arrays read from HDF5, but it is not yet “IMAS-native” in terms of schema placement.

### Command-line usage
#### Synopsis
```bash
python dump2imas.py DUMPGLL_FILE \
  --backend hdf5|mdsplus \
  --output-dir OUTDIR \
  --dd DDNAME \
  --pulse PULSE --run RUN
```

#### Arguments
- `dump` (positional): Path to the NIMROD dump file (e.g. `dumpgll.00000.h5`).
- `--backend`: IMAS backend. Supported: `hdf5` (default) or `mdsplus`.
- `--output-dir`: Output directory used to construct the IMAS URI:
  - HDF5: `imas:hdf5?path=<output-dir>`
  - MDSplus: `imas:mdsplus?path=<output-dir>`

  Practical note: to follow a conventional IMAS on-disk layout, you typically set `--output-dir` to the IMAS **entry directory** (for example `./nimrod/3.42.0/201991/1`). The script will create the directory if needed.

- `--dd`, `--pulse`, `--run`: Present for consistency with other workflows. In this script version, the IMAS entry is primarily controlled by `--output-dir` (the URI path). If your local IMAS binding requires dd/pulse/run at DBEntry construction, this script will need to be adapted accordingly.

#### Examples
Write into an entry directory under the current working directory:
```bash
python dump2imas.py dumpgll.00000.h5 \
  --backend hdf5 \
  --output-dir ./nimrod/3.42.0/201991/1 \
  --dd nimrod --pulse 201991 --run 1
```

Append another dump’s XML record into the same entry:
```bash
python dump2imas.py dumpgll.00010.h5 \
  --backend hdf5 \
  --output-dir ./nimrod/3.42.0/201991/1 \
  --dd nimrod --pulse 201991 --run 1
```

### How to decode arrays from `code.parameters`
Below is a minimal example to extract and reconstruct arrays from the XML payload.

```python
import base64
import numpy as np
import xml.etree.ElementTree as ET

xml_str = mhd.code.parameters  # or mhd_linear.code.parameters
root = ET.fromstring(xml_str)

# Find the last appended dump element
last = root.findall('nimrod_mhd_dump')[-1]
arrays = last.find('arrays')

out = {}
for a in arrays.findall('array'):
    name = a.get('name')
    shape = tuple(int(x) for x in a.get('shape').split(','))
    raw = base64.b64decode(a.text)
    out[name] = np.frombuffer(raw, dtype=np.float64).reshape(shape)

R = out['R']
Z = out['Z']
B = out['B_eq']
```

### Known limitations
- The script stores numerical arrays inside XML (`code.parameters`) rather than in IMAS-native nodes (e.g. equilibrium/core_profiles/mhd_linear GGD fields).
- The script writes flattened arrays (`R`, `Z`, etc.) concatenated across blocks; it does not currently reconstruct a stitched 2D block grid.
- Only equilibrium fields are handled (B, v, Te, Ti, n). Perturbation fields (complex mode content) are not written in this script version.

### Troubleshooting
- **`No 'rblocks' group found`**: Your HDF5 dump does not match the expected group structure. Inspect the file with `h5dump -n` or `python -c "import h5py; f=h5py.File('dumpgll.00000.h5'); print(list(f.keys()))"`.
- **`Missing dataset rz<bid>`**: The block group exists but does not include coordinates under the expected naming convention.
- **IMAS import/DBEntry errors**: Confirm that `imas` is importable in your environment and that the chosen `--backend` is supported by your installation.

---

## 5. `plot_mhd_linear_contours.py`

### Purpose

`plot_mhd_linear_contours.py` is a utility for visualizing **2D poloidal-plane contours** of perturbation quantities stored in the IMAS **`mhd_linear`** IDS. It supports selecting:

* pulse/run (IMAS entry),
* IDS occurrence,
* time slice (by index),
* toroidal mode (by `n_tor` or by mode index),
* field and component (for vector quantities),
* real/imag/amplitude parts.

The script is intended for IMAS HDF5 backends (local file-based entries), but the logic is generic enough to work with other backends if configured.

---

### Requirements

* Python 3.9+ (tested in typical IMAS Python venvs)
* `imas-python`
* `numpy`
* `matplotlib`

Optional:

* `h5py` (only used for low-level validation/debugging; not required for normal plotting)

---

### Data model assumptions

The script expects (at minimum):

* `mhd_linear.time_slice[time_index].toroidal_mode[mode_index].grid_2d` with coordinates needed for 2D contour plotting
* Perturbed quantity arrays stored per toroidal mode and time slice

If the data do not contain the requested field, the script will fail with a clear error.

---

### Quick start

#### Print basic IDS information

```bash
python plot_mhd_linear_contours.py \
  --pulse 201991 --run 3 --dd nimrod --dd-version 3.42.0 \
  --occ 0 --info
```

#### Plot pressure perturbation for n=3 at the first time slice

```bash
python plot_mhd_linear_contours.py \
  --pulse 201991 --run 3 --dd nimrod --dd-version 3.42.0 \
  --occ 0 --time-index 0 --n-tor 3 --field p --part real
```

#### Plot amplitude of B-field (radial component) for n=5

```bash
python plot_mhd_linear_contours.py \
  --pulse 201991 --run 3 --dd nimrod --dd-version 3.42.0 \
  --occ 0 --time-index 0 --n-tor 5 --field b --component r --part amp
```

---

### Listing supported fields

To print the list of supported fields and exit:

```bash
python plot_mhd_linear_contours.py --list-fields
```

---

### Field selection

#### Scalar fields

| `--field` value | Aliases        | IDS quantity             |
| --------------- | -------------- | ------------------------ |
| `p`             | `pressure`     | `pressure_perturbed`     |
| `t`             | `temperature`  | `temperature_perturbed`  |
| `n`             | `mass_density` | `mass_density_perturbed` |

### Vector fields (require `--component`)

| `--field` value | Aliases           | IDS quantity                           | Requires       |   |      |
| --------------- | ----------------- | -------------------------------------- | -------------- | - | ---- |
| `b`             | `bfield`          | `b_field_perturbed.coordinate{1,2,3}`  | `--component r | z | phi` |
| `v`             | `vel`, `velocity` | `velocity_perturbed.coordinate{1,2,3}` | `--component r | z | phi` |

#### Part selection (`--part`)

* `real`: real part
* `imag`: imaginary part
* `amp`: amplitude computed as `sqrt(real^2 + imag^2)`

---

### Toroidal mode selection

You can select a mode using either:

* `--n-tor <int>` (recommended): selects by toroidal mode number
* `--mode-index <int>`: selects by array index in `toroidal_mode[]`

If `--n-tor` is requested but the IDS object appears to return placeholder/unset values for `n_tor`, verify that:

* you are opening the DBEntry with the correct `--dd-version`, and
* the script is filling an IDS created by `IDSFactory(dd_version)` (in-place `db.get(ids, occ)`).

A quick low-level validation (optional) is to inspect the raw dataset with `h5py`:

```python
import h5py
with h5py.File(".../mhd_linear.h5","r") as f:
    print(f["/mhd_linear/time_slice[]&toroidal_mode[]&n_tor"][:])
```

---

### Common options

* `--pulse`, `--run`: IMAS shot/run identifiers
* `--dd`, `--dd-version`: data dictionary selection (important for consistent reads)
* `--occ`: IDS occurrence (default typically 0)
* `--time-index`: integer index into `time_slice[]`
* `--field`: field name (see tables above)
* `--component`: `r`, `z`, or `phi` for vector fields
* `--part`: `real`, `imag`, or `amp`
* `--info`: print time/mode summary and exit
* `--list-fields`: print supported fields and exit

---

### Troubleshooting

#### 1) DD major-version mismatch (read or write)

If you see errors about the provided IDS/DD differing from the Data Entry DD major version, ensure:

* the **DBEntry** is opened using the same DD major version as the IDS factory
* `--dd-version` matches the entry you are reading

In some environments it may be necessary to set:

```bash
export IMAS_VERSION=3.42.0
```

before running the script to force the runtime DD.

#### 2) `Requested n_tor=<n> not found`

Run:

```bash
python plot_mhd_linear_contours.py ... --info
```

to see the mode list at the selected time slice. If you still see placeholder values for `n_tor` even though the HDF5 dataset contains valid integers, this indicates the IDS was instantiated with an incompatible DD context. Use `--dd-version` and ensure DBEntry creation respects it.

#### 3) Field not present

Some runs may only populate a subset of perturbations. Use `--info` and then try a different `--field`, or inspect the IDS in a short Python session.

### Notes for development

* Keep `--dd-version` propagation consistent. The most reliable pattern is:

  1. `factory = imas.IDSFactory(dd_version=...)`
  2. `ids = factory.new("mhd_linear")`
  3. `db.get(ids, occ)`  (fill in-place)
* Avoid “best-effort defaults” for DD selection when the entry was written with a specific DD version.
* If adding new fields, update:

  * the field mapping table,
  * `--list-fields` output,
  * `--help` epilog examples.

---

## Recommended workflow

1. **Ingest NIMROD inputs into IMAS**

   ```bash
   python3 nimrod2imas.py geqdsk peqdsk ...options...
   ```

2. **Validate GEQDSK / PEQDSK round trip**

   ```bash
   python3 validate_nimrod2imas.py geqdsk peqdsk ...options...
   ```

   Inspect the reported `max|Δ|` and `rms|Δ|` metrics.

3. **Verify namelist round trip**

   ```bash
   python3 nimrodInputRestore.py ...options...
   diff nimeq.in  nimeq_from_imas.in
   diff nimrod.in nimrod_from_imas.in
   ```

This set of scripts provides a fully traceable path between original NIMROD inputs and their IMAS representation, and back.
