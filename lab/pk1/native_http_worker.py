"""Entry inside the restricted target. Keep the normal product tokenizer, worker protocol and GGUF bridge."""
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

import native_trial as trial


def main(argv):
    from ironmule import ggml_engine as ge
    from ironmule.ggml_binary import identity

    mode, gguf, profile_path, *rest = argv
    if mode == "gate":
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
    data = json.loads(Path(profile_path).read_bytes())
    profile = data["binding"]["profile"]
    lib = Path(os.environ["IRONMULE_KG_LIB"])
    expected = data["binary"]
    if (trial.checksum(lib) != expected or trial.checksum(os.environ["IRONMULE_W4_TABLE"]) != data["table"]
            or ge.file_sha256(gguf) != data["binding"]["model_sha256"] or identity() != data["binding"]["native"]
            or ge.gpu_layers() != 999 or any(os.environ.get("IRONMULE_" + k) != v for k, v in trial.FLAGS.items())):
        raise ValueError("changed HTTP worker binding")
    if mode == "gate":
        row = ge.gate(gguf, gguf, cache=Path.cwd() / "gate")
        if row.get("passes") is not True or row.get("settings") != data["binding"]["engine"]:
            raise ValueError("the product quality gate did not admit the measured settings")
        row["kernel_profile"] = {"profile": "explicit", "flags": {k: os.environ[k] for k in data["flags"]},
                                 "gguf_sha256": ge.file_sha256(gguf), "machine": ge.machine_key(),
                                 "native_sha256": identity()["sha256"], "kernel_sha256": expected,
                                 "layers": 999, "fallback_reason": None}
        Path(rest[0]).write_text(json.dumps(row, allow_nan=False))
        return 0
    if mode != "serve":
        raise ValueError("unknown worker mode")

    # Freeze the same masks and reference measurements used for both arms of the merged trial.
    plugin = ctypes.CDLL(str(lib))
    getters = []
    for suffix, bits in zip(("", "6"), profile["mask_bits"], strict=True):
        setter = getattr(plugin, "ironmule_kg_set_mask" + suffix)
        setter.argtypes = [ctypes.c_uint32]
        setter(bits)
        getter = getattr(plugin, "ironmule_kg_get_mask" + suffix)
        getter.restype = ctypes.c_uint32
        getters.append(getter)

    class MeasuredEngine(trial.fixture_engine(ge, profile)):
        def __init__(self, model, **settings):
            super().__init__(model, **settings, kernel_mask=False)
            effective = {"mask_bits": [fn() for fn in getters], "padding": {str(k): v for k, v in self.pad_to.items()},
                         "pass_seconds": {str(k): v for k, v in self.pass_costs.points} if self.pass_costs else {},
                         "pilot_costs": dict(self.pilot.costs)}
            if (self.seqs != 32 or self.seq_ctx != 2048 or self.settings != data["binding"]["engine"]
                    or effective != profile or self.threads != data["binding"]["threads"] or not self.speculate
                    or trial.checksum(self.pilot.weights_path) != data["binding"]["pilot"]
                    or sorted(os.sched_getaffinity(0)) != data["binding"]["affinity"]):
                raise ValueError("product worker and measured native execution profiles differ")

    # This process has one model and exits with its worker. All ordinary serving flows remain in GgmlBridge.
    ge.GgmlEngine = MeasuredEngine
    from ironmule_product.worker import main as worker_main
    return worker_main(rest)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
