"""Fetch or verify the official, immutable Qwen3-32B AWQ checkpoint for a lab comparison."""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import http.client
import json
import os
import re
import stat
import urllib.request
from pathlib import Path

REPO = "Qwen/Qwen3-32B-AWQ"
REVISION = "0499c3ac83fdef8810b907a23894ba91e95eddd8"
TOKENIZER_REPO = "Qwen/Qwen3-32B"
TOKENIZER_COMMIT = "9216db5781bf21249d130ec9da846c4624c16137"
# The four tokenizer files (including the chat template) have the same Git blob and
# SHA-256 at the two revisions. The AWQ config.json is deliberately different.
FILES = {
    "LICENSE": (11544, "5de36594c10839788a8c589443a8ef9d8b8d17c65a1b5807206ae037fc36c6bd"),
    "config.json": (907, "0e0c09d430b12e84773ef83ca83dba7c9174caf5a94bb940cd895b82df6c6528"),
    "generation_config.json": (239, "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"),
    "merges.txt": (1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "tokenizer.json": (11422654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    "tokenizer_config.json": (9732, "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
    "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    "model.safetensors.index.json": (131598, "7d0427c7d238f401b5d38ec08dca07cc8a921aa69d73769b2b7983520de504a1"),
    "model-00001-of-00004.safetensors": (4966384688, "ab7988215e3b087c884e0c14ce1994a64aac31dc9a37a4bcf29f5b45f3506e2e"),
    "model-00002-of-00004.safetensors": (4998723112, "46faff349f1dd18ffe76993edd3d9fd32941d042331dc590525409bbe3d37edf"),
    "model-00003-of-00004.safetensors": (4949671152, "6e9f763d8c31ad19f1cd67b0b599a38edffd76a4465a22ff5f7fcac46d820e63"),
    "model-00004-of-00004.safetensors": (4410702792, "87f180889fead12b00b1bdf4dca3a3a14e4510e1e200bf56e31898e120b993d8"),
}


def _manifest() -> dict:
    files = {name: {"bytes": size, "sha256": digest} for name, (size, digest) in FILES.items()}
    return {"repo": REPO, "commit": REVISION, "files": files}


def _manifest_digest() -> str:
    encoded = json.dumps(_manifest(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stamp(path: Path) -> list[int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"checkpoint entry must be a regular file: {path.name}")
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _file_digest(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest


def _check(path: Path) -> list[int]:
    expected_size, expected_digest = FILES[path.name]
    before = _stamp(path)
    if before[2] != expected_size:
        raise ValueError(f"unexpected byte count for {path.name}")
    digest = _file_digest(path).hexdigest()
    if digest != expected_digest or _stamp(path) != before:
        raise ValueError(f"checkpoint file differs from its pinned SHA-256: {path.name}")
    return before


def _partial_name(name: str) -> str:
    return f".{name}.incomplete"


def _contents(directory: Path, downloading: bool = False) -> set[str]:
    names = {path.name for path in directory.iterdir()}
    allowed = set(FILES)
    if downloading:
        allowed.update(_partial_name(name) for name in FILES)
    if names - allowed:
        raise ValueError(f"unexpected checkpoint entries: {sorted(names - allowed)}")
    if not downloading and names != set(FILES):
        raise ValueError(f"missing checkpoint entries: {sorted(set(FILES) - names)}")
    return names


def verify(directory: Path) -> dict:
    """Hash every file, including every weight shard. Return a JSON-serializable receipt."""
    directory = Path(directory)
    _contents(directory)
    stamps = {name: _check(directory / name) for name in FILES}
    result = {
        **_manifest(), "verified": True, "revision": REVISION, "manifest_sha256": _manifest_digest(),
        "total_bytes": sum(size for size, _ in FILES.values()),
        "shard_bytes": sum(size for name, (size, _) in FILES.items() if name.endswith(".safetensors")),
        "tokenizer_repo": TOKENIZER_REPO, "tokenizer_commit": TOKENIZER_COMMIT,
        "tokenizer_files_match": True, "file_stamps": stamps,
    }
    if not unchanged(directory, result):
        raise ValueError("checkpoint changed during verification")
    return result


def unchanged(directory: Path, verified: dict) -> bool:
    """Recheck the file stamps of a locally obtained verification receipt after serving."""
    if verified.get("verified") is not True or verified.get("manifest_sha256") != _manifest_digest():
        return False
    try:
        directory = Path(directory)
        return {name: _stamp(directory / name) for name in _contents(directory)} == verified.get("file_stamps")
    except (OSError, ValueError):
        return False


def _transfer(partial: Path, name: str) -> None:
    size, expected = FILES[name]
    digest = hashlib.sha256()
    offset = 0
    if partial.exists() or partial.is_symlink():
        offset = _stamp(partial)[2]
        if offset > size:
            raise ValueError(f"oversized partial checkpoint file: {name}")
        digest = _file_digest(partial)

    if offset < size:
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true"
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        with urllib.request.urlopen(request, timeout=45) as response:
            if response.status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if not match or (int(match[1]), int(match[3])) != (offset, size) or not offset <= int(match[2]) < size:
                    raise ValueError(f"unexpected HTTP range for {name}")
            elif response.status == 200:
                offset, digest = 0, hashlib.sha256()
            else:
                raise OSError(f"unexpected HTTP status {response.status} for {name}")
            with partial.open("ab" if offset else "wb") as output:
                for block in iter(lambda: response.read(4 << 20), b""):
                    offset += len(block)
                    if offset > size:
                        raise ValueError(f"response exceeds the pinned size for {name}")
                    output.write(block)
                    digest.update(block)
        if offset < size:
            raise OSError(f"incomplete HTTP transfer for {name}")
    if digest.hexdigest() != expected:
        partial.unlink(missing_ok=True)
        raise ValueError(f"wrong downloaded SHA-256 for {name}")
    os.replace(partial, partial.parent / name)


def _fetch_file(directory: Path, name: str) -> None:
    path, partial = directory / name, directory / _partial_name(name)
    if path.exists() or path.is_symlink():
        _check(path)
        return
    for attempt in range(3):
        try:
            _transfer(partial, name)
            return
        except (OSError, http.client.HTTPException):
            if attempt == 2:
                raise


def fetch(directory: Path) -> Path:
    """Download into the exact checkpoint directory, resuming interrupted shard transfers."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        _contents(directory, downloading=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda name: _fetch_file(directory, name), FILES))
        _contents(directory)
        return directory
    finally:
        os.close(fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify(args.directory), sort_keys=True))
    else:
        print(fetch(args.directory))
