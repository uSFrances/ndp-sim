#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RAMULATOR_ROOT="${REPO_ROOT}/third_party/ramulator2"
VENV_DIR="${REPO_ROOT}/.venv"

cd "${REPO_ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required in WSL." >&2
  exit 1
fi

if ! python3 -m venv --help >/dev/null 2>&1; then
  echo "python3-venv support is required." >&2
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
. "${VENV_DIR}/bin/activate"

python -m pip install -U pip setuptools wheel cmake ninja
python -m pip install -e "${REPO_ROOT}"

if [ ! -d "${RAMULATOR_ROOT}" ]; then
  echo "Ramulator root not found at ${RAMULATOR_ROOT}" >&2
  exit 1
fi

cd "${RAMULATOR_ROOT}"
cmake -S . -B build-linux -G Ninja -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build-linux -j

cd "${REPO_ROOT}"
cat <<EOF
WSL setup complete.

Activate the virtual environment:
  source .venv/bin/activate

Smoke-test Ramulator:
  ./third_party/ramulator2/build-linux/ramulator2 -f third_party/ramulator2/example_config.yaml

Run unit tests:
  python -m unittest discover -s tests -v

Run validation:
  python -m address_remapping.cli run-validation \\
    examples/graphs/ring_gemm_bias.json \\
    --output outputs/performance/ring_gemm_bias/ring_gemm_bias_performance.json \\
    --config examples/configs/performance_config.json \\
    --ramulator-root third_party/ramulator2
EOF
