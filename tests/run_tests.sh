#!/usr/bin/env bash
set -euo pipefail

# Reviewer-friendly wrapper for the reduced example.
# Run from either the package root or the tests directory.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

make -C "${ROOT_DIR}" test "$@"
