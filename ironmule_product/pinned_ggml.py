"""Qualify one --gpus worker after its launcher has selected the CUDA device."""
import json
import sys
from argparse import Namespace
from contextlib import redirect_stdout

from .cli import _local_ggml_options
from .state import ProductStore
from .types import ModelSpec


def main():
    payload = json.load(sys.stdin)
    with redirect_stdout(sys.stderr):
        if bus := payload.get("rocm_bus"):
            from .placement_devices import verify_rocm_worker
            verify_rocm_worker(bus)
        options = _local_ggml_options(Namespace(**payload["options"]), ModelSpec.from_dict(payload["spec"]),
                                      ProductStore(payload["root"]), pinned=payload.get("pinned", True))
    print(json.dumps(options, allow_nan=False))


if __name__ == "__main__":
    main()
