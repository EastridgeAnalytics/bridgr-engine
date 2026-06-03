#!/bin/sh

# Fail if any source file under $1 includes <assert.h>/<cassert>.
#
# Vendored third-party trees are exempt — the GVE-Leiden headers under .../gve/
# (puzzlef/leiden-communities-openmp, MIT) use <cassert> and are kept pristine,
# as is anything under a third_party/ directory.
! grep --color=auto -r "include <assert.h>\|include <cassert>" "$1" \
  | grep -v '/gve/' \
  | grep -v '/third_party/'
