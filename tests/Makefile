# Convenience Makefile for reviewers who enter the tests directory.
#
# The executable test and validation targets are defined in the package-level
# Makefile.  This file forwards reviewer-facing targets to that Makefile so
# the reduced example can be exercised directly from tests/.
#
# The distributed test case is intentionally low resolution.  It is intended
# to test input/dump conversion and the structure of the generated IMAS data,
# not to provide a physically converged linear-stability calculation.
# Consequently, production NIMROD history files such as energy.bin, logen.bin,
# and nimhist*.bin are not included.  Growth rates/frequencies from the reduced
# case would not constitute meaningful physics validation.

.PHONY: 	help venv install check 	test test-input test-dump 	validate validate-mhd-linear validate-all 	gamma-info 	clean-test clean

# Forward the standard package-level targets.
help venv install check test test-input test-dump validate validate-mhd-linear validate-all clean-test clean:
	$(MAKE) -C .. $@

# Explain why gamma2imas is not exercised by the reduced repository test.
gamma-info:
	@echo "gamma2imas.py is not exercised by the reduced test case."
	@echo "The included NIMROD simulation is intentionally low resolution and"
	@echo "is provided only for regression testing of the conversion workflow."
	@echo "Production history files (energy.bin, logen.bin, nimhist*.bin) are"
	@echo "therefore not distributed with this test case, because growth rates"
	@echo "and mode frequencies from this under-resolved calculation would not"
	@echo "be physically meaningful."
	@echo ""
	@echo "Use 'make validate-mhd-linear' for the repository-contained structural"
	@echo "validation of mhd_linear output generated from the sample dump."
