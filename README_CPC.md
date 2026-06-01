# CPC program package notes for nimrod2imas

This archive is intended for the Computer Physics Communications "Computer Programs in Physics" submission category.

## Contents

- `*.py`: source scripts for converting NIMROD input and dump data to IMAS and for validation/plotting.
- `requirements.txt`, `constraints.txt`: Python package requirements used by the example.
- `tests/`: reduced example data for a representative NIMROD-to-IMAS conversion.
- `tests/dumpgll.00000.h5`: reduced NIMROD dump used by the example run.
- `tests/g163518.01900`, `tests/p163518.1900`: equilibrium/profile inputs for the example.
- `tests/*.in`: NIMROD/FGNIMEQ-related namelist files used in the example.
- `tests/d3d/4/163518/1/`: expected/generated HDF5 output from a successful example run.
- `Makefile`: reviewer-facing installation, test, validation, cleanup, and archive targets.

The full production-scale dataset discussed in the manuscript is not included in this archive because of its size. It is deposited separately in the public data repository cited in the manuscript. The included reduced example is intended to demonstrate installation and operation of the conversion workflow.

## Quick start

```bash
make venv
make install
make test
```

The example writes an IMAS filesystem-backed entry under:

```text
tests/d3d/4/163518/1/
```

To compare reconstructed GEQDSK and p-file content from the generated IMAS entry, run:

```bash
make validate
```

To remove generated output from the example, run:

```bash
make clean-test
```

## Notes for reviewers

The package requires an IMAS-Python environment compatible with the IMAS Data Dictionary version used in the test, by default `DD_VERSION=4.1.1`. If a local installation uses a different DD version, run for example:

```bash
make test DD_VERSION=4.1.1
```

or set the appropriate `DD_VERSION` value supported by the local environment.

`make validate` performs a reviewer-facing regression check. It opens the generated filesystem-backed IMAS entry, reconstructs GEQDSK and PEQDSK files, and applies strict pass/fail checks to directly stored density and temperature profiles. Flow- and omega-related PEQDSK quantities are printed as diagnostics only because they are reconstructed from multiple quantities and midplane geometry, and are more sensitive to interpolation and finite-difference conventions than the direct round-trip profiles.

A warning about missing `MDSplus` support can be ignored for this example when using the default HDF5 backend.

### Optional mhd_linear validation

The reduced example also contains `mhd_linear` occurrence files generated from the sample NIMROD dump. To check this part of the conversion without requiring a large production data set, run:

```bash
make validate-mhd-linear
```

This target performs a lightweight structural regression check: it verifies that the `mhd_linear` HDF5 files contain time slices, toroidal-mode metadata, grid arrays, and finite perturbation arrays. It is intentionally not a full physics validation of linear-mode amplitudes.

To run both validation checks, use:

```bash
make validate-all
```
