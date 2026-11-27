"""Record elapsed broker-to-worker startup alongside the unmodified complete Qwen workload."""

import argparse
import os
from pathlib import Path
import sys


def main():
    import restart_arm
    from native_trial import checksum
    from lab.portable_gpu.bench import save
    import json

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--broker-start-ns", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args, retained = p.parse_known_args()
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    prior = int(fields[19]) / os.sysconf("SC_CLK_TCK") - args.broker_start_ns / 1e9
    if prior < -0.02 or prior > 600:
        raise ValueError("implausible broker to process time")
    sys.argv = [sys.argv[0], "--output", str(args.output), *retained]
    restart_arm.main()
    row = json.loads(args.output.read_bytes())
    row.update(
        broker_to_process_s=max(0.0, prior),
        broker_start_ns=args.broker_start_ns,
        followup_harness_sha256=checksum(__file__),
    )
    save(args.output, row)


if __name__ == "__main__":
    main()
