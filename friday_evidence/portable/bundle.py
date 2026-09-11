"""Build a small private/offline Kaggle dataset and its pinned entry script."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import textwrap
import zipfile

from .contracts import (ContractError, code_digest, code_paths, file_sha256, safe_file,
                        spec_digest, validate_spec, write_json_new)


def prepare_bundle(spec: dict, data_dir: Path, destination: Path, *, owner: str) -> dict:
    validate_spec(spec)
    if spec["backend"] == "mlx":
        raise ContractError("mlx_collection_is_local")
    if not isinstance(owner, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,63}", owner):
        raise ContractError("invalid_kaggle_owner")
    if spec.get("code_sha256") != code_digest():
        raise ContractError("code_changed_since_plan")
    destination = Path(destination)
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    dataset = destination / "dataset"
    notebook = destination / "notebook"
    dataset.mkdir(mode=0o700)
    notebook.mkdir(mode=0o700)
    slug = f"data1-{spec['run_id']}"
    resource_id = f"{owner}/{slug}"
    copied = set()
    total = 0
    for case in spec["cases"]:
        if case.get("export_policy") != "private_public_workload_only":
            raise ContractError("capture_export_not_approved")
        for letter in ("a", "b"):
            name = case[f"{letter}_file"]
            source = safe_file(data_dir, name)
            if file_sha256(source) != case[f"{letter}_sha256"]:
                raise ContractError("input_changed_since_plan")
            if name not in copied:
                total += source.stat().st_size
                if total > 256 * 1024 * 1024:
                    raise ContractError("upload_size_budget_exceeded")
                shutil.copyfile(source, dataset / name)
                os.chmod(dataset / name, 0o600)
                copied.add(name)
    write_json_new(dataset / "spec.json", spec)
    evidence_root = Path(__file__).resolve().parents[1]
    archive = dataset / "portable-source.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for source in code_paths():
            bundle.write(source, str(Path("friday_evidence") / source.relative_to(evidence_root)))
    os.chmod(archive, 0o600)
    archive_sha = file_sha256(archive)
    entry_body = f'''import hashlib
from pathlib import Path
import sys
import zipfile

matches = [p for p in Path('/kaggle/input').rglob('portable-source.zip')
           if hashlib.sha256(p.read_bytes()).hexdigest() == {archive_sha!r}]
if len(matches) != 1:
    raise RuntimeError('DATA1 input bundle unavailable or ambiguous')
source = matches[0]
workspace = Path('/kaggle/working/data1-source')
workspace.mkdir(exist_ok=False)
with zipfile.ZipFile(source) as bundle:
    for name in bundle.namelist():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise RuntimeError('Invalid DATA1 bundle member')
    bundle.extractall(workspace)
sys.path.insert(0, str(workspace))
from friday_evidence.portable.contracts import load_json, spec_digest
from friday_evidence.portable.runner import run_experiment
spec = load_json(source.parent / 'spec.json')
if spec_digest(spec) != {spec_digest(spec)!r}:
    raise RuntimeError('DATA1 spec identity mismatch')
result = run_experiment(spec, source.parent, Path('/kaggle/working/results'))
print({{'run_id': result['run_id'], 'status': result['status'], 'performance_claim': False}})
'''
    # multiprocessing's spawn imports __main__ again. Bootstrap must run only
    # in the original process, otherwise extraction would run twice.
    entry = ("# Private DATA1 job; no internet or installations.\ndef main():\n"
             + textwrap.indent(entry_body, "    ")
             + "\nif __name__ == '__main__':\n    main()\n")
    entry_path = notebook / "data1.py"
    with entry_path.open("x", encoding="utf-8") as output:
        output.write(entry)
    os.chmod(entry_path, 0o600)
    write_json_new(dataset / "dataset-metadata.json", {"id": resource_id,
        "title": f"IronMule DATA1 {spec['run_id']}", "licenses": [{"name": "other"}],
        "description": "Private operator captures from public DATA1 workloads. Original model terms apply. No public redistribution."})
    write_json_new(notebook / "kernel-metadata.json", {"id": resource_id,
        "title": f"IronMule DATA1 {spec['run_id']}", "code_file": "data1.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_internet": False, "enable_gpu": spec["backend"] == "cuda",
        "enable_tpu": spec["backend"] == "tpu", "dataset_sources": [resource_id]})
    return {"dataset_id": resource_id, "kernel_id": resource_id, "run_slug": slug,
            "dataset_dir": str(dataset), "notebook_dir": str(notebook),
            "archive_sha256": archive_sha, "input_bytes": total, "submitted": False}
