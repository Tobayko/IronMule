"""Prepare a private, offline T4 job from a committed checkout. Does not submit a job."""

import argparse
import base64
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGES = ("smoke", "steps", "http")
BUDGET = {"smoke": 90, "steps": 360, "http": 600}


def package(args):
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.commit}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("expected a full commit SHA")
    if args.through != "smoke":
        if not args.gguf or not (args.dataset or args.kernel_source):
            raise ValueError("attach the cached inputs and specify their GGUF path")
        if args.kg2_archive:
            if args.lib_dir or args.bindings or not (args.llama_source or args.preload):
                raise ValueError("the KG2 archive needs a probe DSO or its pinned llama.cpp sources")
        elif not (args.lib_dir and args.bindings):
            raise ValueError("supply a KG2 archive or the extracted library and bindings directories")
    if args.llama_source and args.preload:
        raise ValueError("select a prebuilt probe or the sources to compile it")
    budget = args.budget_s if args.budget_s is not None else BUDGET[args.through]
    if not 1 <= budget <= BUDGET[args.through]:
        raise ValueError(f"budget for {args.through} must be between 1 and {BUDGET[args.through]} seconds")

    paths = ["native/ggml_patches/kg5", "lab/kg5/bench.py", "lab/kg5/worker.py", "lab/kg5/kaggle_session.py"]
    if args.through != "smoke":
        paths += ["ironmule", "ironmule_product", "friday_evidence", "ironmule_cli.py",
                  "ironmule_controller.py", "ironmule_inventory.py", "lab/row1/facts.py"]
    archive = subprocess.check_output(["git", "archive", "--format=zip", commit, *paths], cwd=ROOT)
    config = {"commit": commit, "through": args.through, "budget_s": budget}
    for name in ["gguf", "kg2_archive", "lib_dir", "bindings", "llama_source", "kg_lib", "preload", "wheelhouse", "gate_cache"]:
        path = getattr(args, name)
        if path:
            if not path.is_relative_to("/kaggle/input") or ".." in path.parts:
                raise ValueError(f"{name} must be a mounted path under /kaggle/input")
            config[name] = str(path)
    slug = f"ironmule-kg5-{args.through}"
    metadata = {"id": f"{args.owner}/{slug}", "title": slug, "code_file": "run.py", "kernel_type": "script",
                "language": "python", "is_private": True, "enable_internet": False, "enable_gpu": True,
                "machine_shape": "NvidiaTeslaT4", "dataset_sources": args.dataset,
                "kernel_sources": args.kernel_source, "competition_sources": [], "model_sources": []}
    entry = f'''import base64
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

with tempfile.TemporaryDirectory() as directory:
    with zipfile.ZipFile(io.BytesIO(base64.b64decode({base64.b64encode(archive).decode()!r}))) as bundle:
        bundle.extractall(directory)
    job = Path(directory) / "job.json"
    job.write_text({json.dumps(config)!r})
    sys.exit(subprocess.call([sys.executable, "lab/kg5/kaggle_session.py", str(job)], cwd=directory))
'''
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("the output directory must be empty")
    (args.output / "run.py").write_text(entry)
    (args.output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {metadata['id']} at {commit}, {budget}s workload limit.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True, help="local Git revision whose files will run on Kaggle")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--through", choices=STAGES, default="smoke")
    parser.add_argument("--budget-s", type=int)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--kernel-source", action="append", default=[])
    for name in ["gguf", "kg2-archive", "lib-dir", "bindings", "llama-source", "kg-lib", "preload", "wheelhouse", "gate-cache"]:
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    try:
        package(args)
    except (ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
