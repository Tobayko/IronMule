#!/bin/sh
# IronMule installer. No admin rights, nothing outside your home directory:
#   curl -fsSL https://raw.githubusercontent.com/Tobayko/IronMule/main/install.sh | sh
# It installs uv (https://docs.astral.sh/uv/) if it is missing, then IronMule as an isolated
# command-line tool with its own Python. Linux machines with an NVIDIA GPU get the CUDA build,
# other Linux machines the CPU build (slow, but it answers; CPU4).
set -eu

source_spec="${IRONMULE_SOURCE:-git+https://github.com/Tobayko/IronMule}"  # override: a local checkout
case "$(uname -s)/$(uname -m)" in
  Darwin/arm64) package="ironmule" ;;
  Linux/*)
    if command -v nvidia-smi >/dev/null 2>&1; then
      package="ironmule[cuda]"
    else
      echo "No NVIDIA GPU found (no nvidia-smi): installing the CPU build, which is much slower." >&2
      package="ironmule[cpu]"
    fi ;;
  *)
    echo "IronMule needs an Apple Silicon Mac or Linux, not $(uname -s)/$(uname -m)." >&2
    exit 1 ;;
esac

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="$HOME/.local/bin:$PATH"
fi

uv tool install --force --python 3.12 "$package @ $source_spec"
command -v ironmule >/dev/null 2>&1 || uv tool update-shell

echo
echo "IronMule is installed. Start chatting with:"
echo "  ironmule start"
echo "(Open a new terminal first if the command is not found.)"
