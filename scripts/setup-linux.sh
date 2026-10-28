#!/usr/bin/env bash
# Ubuntu 22.04+, Python 3.10+. Native build dependencies must already be installed.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv-linux
source .venv-linux/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install 'numpy<2' 'pybind11==2.13.6' dm-env 'gym==0.26.2' 'gymnasium==0.29.1' optree packaging
python engine/fetch.py "$PWD"
python engine/prepare.py "$PWD"
cmake -S engine -B build/engine -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCORE_DIR="$PWD/vendor/ygopro-core-current" -DENV_DIR="$PWD/vendor/ygo-agent"
cmake --build build/engine -j 4
python -m pip install -e vendor/ygo-agent/ygoenv
python -m yugioh_gto.cli preflight
