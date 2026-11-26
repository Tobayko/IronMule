"""Authenticate the CPU-only MLX dependency used beside the separately qualified CUDA GGUF backend."""
import hashlib
import re


def content_hash(root):
    entries = sorted(root.rglob("*"))
    if any(p.is_symlink() and (not p.resolve(strict=True).is_relative_to(root) or p.is_dir()) for p in entries):
        raise ValueError("unexpected external or indirect Python overlay directory")
    names = [p for p in entries if p.is_file()]
    if not names or not any(p.parent.name == "mlx" and p.name.startswith("core.") and p.suffix == ".so" for p in names):
        raise ValueError("incomplete MLX CPU overlay")
    digest = hashlib.sha256()
    for path in names:
        relative = path.relative_to(root).as_posix()
        if (not path.resolve().is_relative_to(root) or "\n" in relative
                or not re.fullmatch(r"mlx(?:/.*|[-_][A-Za-z0-9._-]+\.dist-info/.*|_cpu\.libs/.*)", relative)
                or path.suffix == ".pyc" or "__pycache__" in path.parts):
            raise ValueError("unexpected Python overlay entry")
        with path.open("rb") as stream:
            file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        digest.update(f"{file_hash}  {relative}\n".encode())
    return digest.hexdigest()


def bind(worker_root, record):
    from lab.rp.restricted_worker import immutable

    path = (worker_root.parent / "runtime/product-mlx-cpu").resolve(strict=True)
    if str(path) != record.get("root") or not re.fullmatch(r"[a-f0-9]{64}", str(record.get("sha256"))):
        raise ValueError("wrong protected MLX CPU overlay")
    immutable(path, tree=True)
    if content_hash(path) != record["sha256"]:
        raise ValueError("MLX CPU overlay bytes changed")
    return path
