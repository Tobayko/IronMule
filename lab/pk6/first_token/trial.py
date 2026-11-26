"""Run the frozen ABBA warm HTTP test in an isolated checkout with the registered patch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # driver tree; children use checkout via PYTHONPATH
from lab.pk6 import warm_repeat as warm  # noqa: E402

HERE = Path(__file__).resolve().parent
BASE = "f3a69741f7a466ffc144a87b0395f4bfdb6e9b50"
ARMS = ("R1", "C1", "C2", "R2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkout", "models", "output", "state"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.checkout = args.checkout.resolve()
    if args.output.exists():
        raise ValueError("supply a new trial directory")
    if Path.cwd() != args.checkout or os.environ.get("PYTHONPATH") != str(args.checkout):
        raise ValueError("use the measured checkout for both driver and worker imports")
    patch = HERE / "lead.patch"
    git = ["git", "-C", str(args.checkout)]
    if warm.output(*git, "rev-parse", "HEAD") != BASE or warm.output(*git, "status", "--porcelain", "-uno"):
        raise ValueError("need the clean registered base before patching")
    device = warm.board()
    model = args.models / "Qwen3-32B-Q4_K_M.gguf"
    if warm.output("sha256sum", str(model)).split()[0] != warm.GGUF:
        raise ValueError("incorrect GGUF")
    args.output.mkdir(mode=0o700, parents=True)
    subprocess.run([*git, "apply", str(patch)], check=True)
    try:
        warm.save(args.output / "target.json", {"base": BASE, "hardware": device, "model_sha256": warm.GGUF,
                  "candidate_patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                  "protocol_sha256": hashlib.sha256((HERE / "protocol.md").read_bytes()).hexdigest(),
                  "arguments": list(ARMS)})
        for i, arm in enumerate(ARMS):
            os.environ["IRONMULE_LAB_PREFILL_LEAD"] = str(int(arm.startswith("C")))
            warm.arm(args, arm, device, source=BASE, seed=260113 if i < 2 else 260177)
        warm.save(args.output / "completed.json", {"hardware": warm.board(), "order": list(ARMS)})
    finally:
        subprocess.run([*git, "apply", "--reverse", str(patch)], check=True)


if __name__ == "__main__":
    sys.exit(main())
