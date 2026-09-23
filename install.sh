#!/bin/sh
# IronMule installer. No admin rights, nothing outside your home directory:
#   curl -fsSL https://raw.githubusercontent.com/Tobayko/IronMule/main/install.sh | sh
# It installs uv (https://docs.astral.sh/uv/) if it is missing, then IronMule as an isolated
# command-line tool with its own Python. Linux machines with an NVIDIA GPU get the CUDA build.
set -eu

source_spec="git+https://github.com/Tobayko/IronMule"
case "$(uname -s)/$(uname -m)" in
  Darwin/arm64) package="ironmule" ;;
  Linux/*)
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      echo "IronMule needs an Apple Silicon Mac or Linux with an NVIDIA GPU; no nvidia-smi was found." >&2
      exit 1
    fi
    package="ironmule[cuda]" ;;
  *)
    echo "IronMule needs an Apple Silicon Mac or Linux with an NVIDIA GPU, not $(uname -s)/$(uname -m)." >&2
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
