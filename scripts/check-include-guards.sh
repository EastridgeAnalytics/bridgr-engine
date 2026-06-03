#!/bin/sh

# Fail if any C/C++ HEADER under $1 lacks a "#pragma once" guard.
#
# Vendored third-party header trees are skipped: the GVE-Leiden headers under
# .../gve/ (puzzlef/leiden-communities-openmp, MIT) are kept pristine, ship
# non-header files (LICENSE, README, CITATION.cff), and include at least one
# header that omits the guard. Anything under a third_party/ directory is
# likewise exempt.
#
# (Was: `! grep -rL "^#pragma once" $1 | grep ""`, which scanned EVERY file and
#  so flagged those vendored non-headers and the guard-less vendored header.)
! grep -rL "^#pragma once" \
    --include='*.h' --include='*.hpp' --include='*.hxx' --include='*.hh' --include='*.cuh' \
    "$1" \
  | grep -v '/gve/' \
  | grep -v '/third_party/' \
  | grep ""
