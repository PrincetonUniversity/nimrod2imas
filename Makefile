# Makefile for the nimrod2imas CPC program package.
# It provides a reviewer-friendly way to install dependencies and run
# the reduced example distributed with the program archive.

SHELL := /bin/bash

PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

DD ?= d3d
DD_VERSION ?= 4.1.1
PULSE ?= 163518
RUN ?= 1
TESTDIR ?= tests
DBPATH ?= $(TESTDIR)
BACKEND ?= hdf5
NPHI ?= 4
GGD_CONNECTIVITY ?= fe_wedge

# Prefer the non-empty p-file included in the test directory.
PFILE ?= $(shell if [ -s $(TESTDIR)/p163518.1900 ]; then echo $(TESTDIR)/p163518.1900; elif [ -s $(TESTDIR)/p163518.01900 ]; then echo $(TESTDIR)/p163518.01900; else echo $(TESTDIR)/p163518.1900; fi)

.PHONY: help venv install check test test-input test-dump validate validate-mhd-linear validate-all clean clean-test dist

help:
	@echo "nimrod2imas CPC package targets"
	@echo "  make venv        Create local Python virtual environment ($(VENV))"
	@echo "  make install     Install Python dependencies into $(VENV)"
	@echo "  make check       Check that required Python modules import"
	@echo "  make test        Run the reduced NIMROD-to-IMAS example"
	@echo "  make validate    Reconstruct GEQDSK/p-file from IMAS and compare"
	@echo "  make validate-mhd-linear  Check mhd_linear output structure"
	@echo "  make validate-all  Run all validation checks"
	@echo "  make clean-test  Remove generated IMAS output from the example"
	@echo "  make dist        Create a clean source archive"
	@echo ""
	@echo "Common overrides: DD_VERSION=4.1.1 RUN=1 NPHI=4 VENV=/path/to/venv"

venv:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -U pip setuptools wheel

install: venv
	$(PIP) install -r requirements.txt -c constraints.txt

check:
	$(PY) -c 'import importlib.util; mods=["numpy","scipy","h5py","f90nml","yaml","imas","omfit_classes"]; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print("Missing modules: "+str(missing)) if missing else print("All required modules found."); raise SystemExit(1 if missing else 0)'

test: test-input test-dump
	@echo "Reduced example completed. Output is under $(DBPATH)/$(DD)/4/$(PULSE)/$(RUN)/"

test-input:
	@if [ ! -s "$(PFILE)" ]; then echo "Missing p-file: $(PFILE)"; exit 1; fi
	@mkdir -p "$(DBPATH)"
	$(PY) input2imas.py \
		--dd "$(DD)" --dd-version "$(DD_VERSION)" --pulse "$(PULSE)" --run "$(RUN)" \
		--backend "$(BACKEND)" --dbpath "$(DBPATH)" \
		--nimeq "$(TESTDIR)/nimeq.in" \
		--oculus "$(TESTDIR)/oculus.in" \
		--fluxgrid "$(TESTDIR)/fluxgrid.in" \
		--nimrod "$(TESTDIR)/nimrod.in" \
		--input nimrod.yaml \
		"$(TESTDIR)/g163518.01900" "$(PFILE)"

test-dump:
	$(PY) dump2imas.py \
		--dd "$(DD)" --dd-version "$(DD_VERSION)" --pulse "$(PULSE)" --run "$(RUN)" \
		--backend "$(BACKEND)" --dbpath "$(DBPATH)" \
		--occ-base 1 --sanity-print \
		--peqdsk "$(PFILE)" \
		--ggd-unstructured \
		--ggd-nphi "$(NPHI)" \
		--ggd-connectivity "$(GGD_CONNECTIVITY)" \
		--ggd-write-full-objects \
		--ggd-reuse-grid \
		--edge-ggd-values full \
		"$(TESTDIR)/dumpgll.00000.h5"

validate:
	$(PY) validate_nimrod2imas.py \
		"$(TESTDIR)/g163518.01900" "$(PFILE)" \
		--dd "$(DD)" --dd-version "$(DD_VERSION)" --pulse "$(PULSE)" --run "$(RUN)" \
		--backend "$(BACKEND)" --dbpath "$(DBPATH)" --occ 0 \
		--out-geqdsk "$(TESTDIR)/geqdsk_from_imas" \
		--out-peqdsk "$(TESTDIR)/peqdsk_from_imas"

validate-mhd-linear:
	$(PY) validate_mhd_linear.py \
		--entry "$(DBPATH)/$(DD)/4/$(PULSE)/$(RUN)"

validate-all: validate validate-mhd-linear

clean-test:
	rm -rf "$(DBPATH)/$(DD)/4/$(PULSE)/$(RUN)" \
		"$(TESTDIR)/geqdsk_from_imas" \
		"$(TESTDIR)/peqdsk_from_imas"

clean: clean-test
	rm -rf build dist *.egg-info

# Create an archive suitable for upload to the CPC program files area.
dist:
	mkdir -p dist
	tar --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
		-czf dist/nimrod2imas-cpc.tar.gz \
		--transform 's,^,nimrod2imas/,' \
		CHANGELOG.md LICENSE README.md README_CPC.md Makefile \
		requirements.txt constraints.txt nimrod.yaml *.py tests
