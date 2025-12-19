# NIMROD ⇄ IMAS Utilities

This directory contains three Python utilities to move NIMROD input data into IMAS, validate the round trip, and restore the original Fortran namelists from IMAS.

Scripts:

* `nimrod2imas.py`
* `validate_nimrod2imas.py`
* `nimrodInputRestore.py`

(Your local filenames may have version suffixes, e.g. `nimrod2imas4.py`, `validate_nimrod2imas3.py`.)

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
