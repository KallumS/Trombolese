#!/bin/bash
# Prepare a Claude Code on the web container for Trombolese work.
#
# Two things are needed and neither is in the base image:
#   * the Python scientific stack the model and its tests are built on
#   * the Faust compiler, for the DSP port in dsp/
#
# Faust lives in Ubuntu's universe repository, so `apt-get update` has to run
# first -- without it the install fails with a bare "unable to locate package",
# which looks like the package not existing at all.
set -euo pipefail

# Local machines are assumed to be set up already; only the web container needs this.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_DIR"

echo "Trombolese: installing Python dependencies..."
pip install --quiet --upgrade pip >/dev/null 2>&1 || true
pip install --quiet numpy scipy matplotlib soundfile pytest

echo "Trombolese: installing the Faust compiler..."
# Third-party PPAs in the base image may be unreachable behind the network
# policy; their failures are warnings, so do not let them abort the run.
apt-get update >/dev/null 2>&1 || true
if ! command -v faust >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends faust \
    >/dev/null 2>&1 || echo "Trombolese: faust unavailable; dsp/ cannot be compiled"
fi

# src/ layout: make the package importable without an editable install.
echo "export PYTHONPATH=\"${PROJECT_DIR}/src\${PYTHONPATH:+:\$PYTHONPATH}\"" \
  >> "${CLAUDE_ENV_FILE:-/dev/null}"

if command -v faust >/dev/null 2>&1; then
  echo "Trombolese: ready ($(faust --version 2>&1 | head -1))"
else
  echo "Trombolese: ready (no faust)"
fi
