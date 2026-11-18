"""Bundle the scoped KG4 experiment's Python sources from one Git revision for the pinned CUDA capsule."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
MODULES = ("ironmule", "ironmule_product", "friday_evidence", "ironmule_cli.py", "ironmule_inventory.py",
           "ironmule_devices.py", "ironmule_controller.py", "lab/portable_gpu", "lab/kg4/scoped_native.py",
           "lab/kg4/scoped_http.py", "lab/kg4/scoped_campaign.py")
REQUIRED = ("ironmule/kg4_scope.py", "lab/portable_gpu/mixed_traffic.py")
LAUNCH = b"""#!/bin/bash
set -euo pipefail
umask 077
overlay="$(cd -- "$(dirname -- "$0")" && pwd)"
(cd -- "$overlay" && sha256sum --quiet -c SHA256SUMS)
exec python3 "$overlay/src/lab/kg4/scoped_campaign.py" "$@"
"""


def build(revision, output):
    sha = subprocess.check_output(["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=ROOT, text=True).strip()
    snapshot = subprocess.check_output(["git", "archive", "--format=tar", sha, *MODULES], cwd=ROOT)
    files = {"revision.txt": (sha + "\n").encode(), "run.sh": LAUNCH}
    with tarfile.open(fileobj=io.BytesIO(snapshot)) as source:
        for member in source:
            if member.isfile():
                files["src/" + member.name] = source.extractfile(member).read()
    if not all("src/" + name in files for name in REQUIRED):
        raise ValueError("revision is missing package dependencies")
    checks = [f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(files.items())]
    files["SHA256SUMS"] = "".join(checks).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for name, data in sorted(files.items()):
                info = tarfile.TarInfo("scoped-overlay/" + name)
                info.size = len(data)
                info.mode = 0o755 if name == "run.sh" else 0o644
                tar.addfile(info, io.BytesIO(data))
    return sha


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    revision = build(args.revision, args.output)
    print(revision, args.output.stat().st_size, hashlib.sha256(args.output.read_bytes()).hexdigest())
