#!/usr/bin/env bash
# Set up the LIBERO simulator in the dedicated pixi environment.
#
# LIBERO pins numpy 1.22 / robosuite 1.4 / gym 0.25 and only runs on Python 3.10,
# while LeRobot (and therefore the SurgicalVLA trainer) needs Python >= 3.12.
# The two cannot share an interpreter, so the simulator gets its own environment
# (`pixi run -e libero ...`) and the RL trainer drives it across a process
# boundary -- see src/surgicalvla/rl/libero_server.py.
#
# Usage:
#   scripts/setup_libero.sh [/path/to/LIBERO]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBERO_DIR="${1:-$REPO_ROOT/LIBERO}"
PY="$REPO_ROOT/.pixi/envs/libero/bin/python"
export PATH="$HOME/.pixi/bin:$PATH"
# The system Python 3.10 user site and the ROS humble PYTHONPATH both shadow the
# pinned numpy/robosuite stack, so keep this environment hermetic.
export PYTHONNOUSERSITE=1
export PYTHONPATH=

if [[ ! -x "$PY" ]]; then
  echo "==> creating the 'libero' pixi environment (Python 3.10)"
  (cd "$REPO_ROOT" && pixi install -e libero)
fi

if [[ ! -d "$LIBERO_DIR/.git" ]]; then
  echo "==> cloning LIBERO into $LIBERO_DIR"
  git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
fi

mkdir -p "$LIBERO_DIR/datasets"

echo "==> writing the non-interactive LIBERO config"
# Importing `libero` prompts on stdin when ~/.libero/config.yaml is missing, which
# breaks any headless run, so create it up front.
"$PY" - "$LIBERO_DIR" <<'PY'
import os, sys, yaml

lib = os.path.abspath(sys.argv[1])
root = os.path.join(lib, "libero", "libero")
cfg_dir = os.path.expanduser("~/.libero")
os.makedirs(cfg_dir, exist_ok=True)
cfg = {
    "benchmark_root": root,
    "bddl_files": os.path.join(root, "bddl_files"),
    "init_states": os.path.join(root, "init_files"),
    "assets": os.path.join(root, "assets"),
    "datasets": os.path.join(lib, "datasets"),
}
with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
    yaml.safe_dump(cfg, f)
print(f"    wrote {cfg_dir}/config.yaml")
PY

echo "==> generating robosuite's private macros"
# The script prompts on stdin when a private macro file already exists.
yes "" | "$PY" "$REPO_ROOT/.pixi/envs/libero/lib/python3.10/site-packages/robosuite/scripts/setup_macros.py" >/dev/null || true

echo "==> installing LIBERO into the 'libero' environment"
# LIBERO's setup.py uses find_packages() while `libero/` is a *namespace* dir
# without __init__.py, so modern PEP 660 editable installs map no modules at all.
# `editable_mode=compat` writes a .pth with the checkout root, which is exactly
# what the legacy `setup.py develop` used to do.
"$PY" -m pip install --quiet --no-deps --no-build-isolation --index-url https://pypi.org/simple \
  --config-settings editable_mode=compat -e "$LIBERO_DIR"

echo "==> smoke test"
"$PY" - <<'PY'
import os

import gym
import numpy
import robosuite
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv  # noqa: F401  (import-only check)

print(f"    numpy {numpy.__version__}  robosuite {robosuite.__version__}  gym {gym.__version__}")
bddl = get_libero_path("bddl_files")
print(f"    bddl suites: {sorted(os.listdir(bddl))}")
PY

echo "==> done.  Run the simulator server with: pixi run -e libero python -m surgicalvla.rl.libero_server"
