"""Restore one exact measured evaluator in a fresh detached checkout of the public product base."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def restore(capsules, family, checkout):
    manifest = json.loads((capsules / "manifest.json").read_bytes())
    files = manifest["families"][family]
    if checkout.exists():
        raise ValueError("destination must be a new worktree")
    prepared = {}
    for name, actual in files.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
            raise ValueError("invalid source path")
        row = manifest["files"][actual]
        base = (
            subprocess.check_output(["git", "show", f"{row['commit']}:{name}"])
            if row["commit"]
            else b""
        )
        if sha(base) != row["base_sha256"]:
            raise ValueError("different repository source")
        if row.get("patch"):
            patch = (capsules / "diffs" / (actual + ".patch")).read_bytes()
            if sha(patch) != row["patch"]:
                raise ValueError("different source delta")
            with tempfile.TemporaryDirectory() as temp:
                original, target = Path(temp) / "original", Path(temp) / "actual"
                original.write_bytes(base)
                subprocess.run(
                    ["patch", "-s", "--batch", "--fuzz=0", "-o", target, original],
                    input=patch,
                    check=True,
                    cwd=temp,
                )
                base = target.read_bytes()
        if sha(base) != actual:
            raise ValueError("could not reproduce measured source")
        prepared[path] = base
    subprocess.run(
        ["git", "worktree", "add", "--detach", checkout, manifest["base_commit"]],
        check=True,
    )
    for name, data in prepared.items():
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("capsules", type=Path)
    arg.add_argument("family")
    arg.add_argument("checkout", type=Path)
    args = arg.parse_args()
    restore(args.capsules, args.family, args.checkout.resolve())
