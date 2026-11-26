"""Stage two complete Git archive exports for an EPYC trial without modifying any existing tree."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

SOURCE = "ironmule/ggml_engine.py"


def stage(archive, candidate, target, *, reference_sha, candidate_sha):
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != candidate_sha:
        raise ValueError("candidate upload differs from its pinned source")
    with tarfile.open(archive, mode="r:gz") as tar:
        members = tar.getmembers()
        regular = {}
        for entry in members:
            path = PurePosixPath(entry.name)
            if path.is_absolute() or ".." in path.parts or not (entry.isfile() or entry.isdir()):
                raise ValueError("unsupported entry in frozen Git archive")
            if entry.isfile():
                if entry.name in regular:
                    raise ValueError("duplicate archive entry")
                regular[entry.name] = entry
        if hashlib.sha256(tar.extractfile(regular[SOURCE]).read()).hexdigest() != reference_sha:
            raise ValueError("archive does not contain expected incumbent")
        for arm in ("base", "candidate"):
            destination = target / arm
            with tempfile.TemporaryDirectory(dir=target) as tmp:
                new = not destination.exists()
                tree = Path(tmp) / arm if new else destination
                if tree.is_symlink():
                    raise ValueError("tree must be a real directory")
                if not new and {str(f.relative_to(tree)) for f in tree.rglob("*") if f.is_file() or f.is_symlink()} != set(regular):
                    raise ValueError("existing frozen tree differs from the complete archive")
                for name, entry in regular.items():
                    file = tree / name
                    with (candidate.open("rb") if arm == "candidate" and name == SOURCE else tar.extractfile(entry)) as src:
                        if new:
                            file.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                            with file.open("xb") as output:
                                shutil.copyfileobj(src, output)
                            file.chmod(0o555 if entry.mode & 0o111 else 0o444)
                        else:
                            if (file.is_symlink() or file.stat().st_mode & 0o022
                                    or (os.geteuid() == 0 and file.stat().st_uid != 0)):
                                raise ValueError(f"mutable existing input: {name}")
                            with file.open("rb") as existing:
                                if hashlib.file_digest(src, "sha256").digest() != hashlib.file_digest(existing, "sha256").digest():
                                    raise ValueError(f"existing source differs: {name}")
                if any(d.is_symlink() or d.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH)
                       or (os.geteuid() == 0 and d.stat().st_uid != 0) for d in (tree, *tree.rglob("*")) if d.is_dir()):
                    raise ValueError("mutable input directory")
                if new:
                    tree.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "candidate", "target"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("archive-sha", "reference-sha", "candidate-sha"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    with args.archive.open("rb") as fp:
        if hashlib.file_digest(fp, "sha256").hexdigest() != args.archive_sha:
            raise ValueError("different archive bytes")
    stage(args.archive, args.candidate, args.target, reference_sha=args.reference_sha, candidate_sha=args.candidate_sha)


if __name__ == "__main__":
    main()
