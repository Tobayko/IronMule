"""Protected entry for the ordinary GGUF worker using the released startup method and its local cache."""
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys


def main(argv):
    mode, gguf, data_path, *rest = argv
    from ironmule import ggml_engine as ge
    from ironmule.ggml_binary import identity
    from ironmule.kernel_gen import fingerprint
    from native_trial import checksum, mask_bits
    from restart_arm import mapped_vendor

    data = json.loads(Path(data_path).read_bytes())
    expected = data["runtime"]
    if overlay := data.get("python_overlay"):
        import mlx.core as mx
        if not Path(mx.__file__).resolve().is_relative_to(Path(overlay["root"])) or mx.cuda.is_available():
            raise ValueError("unexpected MLX dependency in the protected GGUF worker")
    lib = os.environ["IRONMULE_KG_LIB"]
    if (ge.file_sha256(gguf) != expected["model_sha256"] or checksum(lib) != expected["w4_sha256"]
            or checksum(os.environ["IRONMULE_W4_TABLE"]) != expected["table_sha256"] or identity() != expected["native"]
            or checksum(ge.__file__) != data["source"] or any(os.environ.get(k) != v for k, v in expected["flags"].items())):
        raise ValueError("different product worker source, CUDA, weights or settings")
    if mode == "gate":
        # The existing numeric/factual gate also independently probes the streams profile.
        # Use its full evidence for the exact requested settings, even if autotuning prefers another.
        if overlay:
            from restart_gate_broker import install_driver
            channel = os.fdopen(os.dup(1), "w", buffering=1)
            install_driver(ge, channel)
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
        from ironmule import ggml_gate_diagnostics as diagnostics
        original_failure = diagnostics.failed_child

        def capture_failure(stage, stderr, status=None, exception=None):
            # A private worker-owned file keeps the exact failing child output; only the normal diagnostic is exported.
            if isinstance(stderr, str) and stderr:
                path = Path.cwd() / "gate-child-stderr.json"
                with path.open("a", encoding="utf-8") as log:
                    log.write(json.dumps({"stage": stage, "status": status, "stderr": stderr[-16384:]}) + "\n")
            return original_failure(stage, stderr, status, exception)

        diagnostics.failed_child = capture_failure
        try:
            report = ge.gate(gguf, gguf, cache=Path.cwd() / "gate")
        finally:
            diagnostics.failed_child = original_failure
        matching = [r for r in [*report.get("tried", []), report.get("streams")] if r
                    and r["settings"] == expected["settings"] and r.get("passes") is True]
        passed = report.get("passes") is True and bool(matching)
        record = {**report, "passes": passed, "source_sha256": data["source"], "preferred_settings": report.get("settings")}
        if passed and data.get("original_product_width_selection"):
            from ironmule_product.cli import _ggml_streams
            actual = _ggml_streams(report, bool(ge.gpu_layers()))
            if ge.SEQS != 8 or actual["settings"] != expected["settings"]:
                record.update(passes=False,reason="ordinary product selected different numerical settings")
                passed = False
            else:
                width = ge.choose_sequences(gguf, actual["settings"], cache=Path.cwd()/"gate", experience_root=Path.cwd())
                record["qualified_width"] = width
                if width.get("sequences") != expected["seqs"] or "32" not in width.get("rates", {}):
                    record.update(passes=False,reason="ordinary independent width calibration did not select measured width 32")
                    passed = False
        if passed:
            result = matching[0]
            record.update(settings=result["settings"], ratio=result["ratio"], interval=result["interval"], chunks=result["chunks"],
                          qualified_profile=result, kernel_profile={"profile": "explicit", "flags": {k: os.environ[k] for k in data["flags"]},
                          "gguf_sha256": expected["model_sha256"], "machine": ge.machine_key(), "native_sha256": expected["native"]["sha256"],
                          "kernel_sha256": data["binary"], "layers": 999, "fallback_reason": None})
        elif "reason" not in record:
            record["reason"] = "the exact requested streams profile did not pass the product gate"
        Path(rest[0]).write_text(json.dumps(record, allow_nan=False))
        return 0
    if mode != "serve":
        raise ValueError("unknown restart worker mode")
    original = ge.GgmlEngine

    class CheckedEngine(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            stat = Path(gguf).stat()
            actual = {"model_sha256": ge.file_sha256(gguf), "model_stat": [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns],
                      "native": identity(), "fp": fingerprint(), "w4_sha256": checksum(lib),
                      "table_sha256": checksum(os.environ["IRONMULE_W4_TABLE"]), "settings": self.settings,
                      "seqs": self.seqs, "seq_ctx": self.seq_ctx, "threads": self.threads, "max_rows": self.max_rows,
                      "speculate": self.speculate, "gpu_layers": self.gpu_layers, "affinity": sorted(os.sched_getaffinity(0)),
                      "flags": {k: os.environ.get(k) for k in expected["flags"]},
                      "measure_kg_sha256": hashlib.sha256(inspect.getsource(original.measure_kg).encode()).hexdigest()}
            plugin = ctypes.CDLL(lib)
            bits = []
            for getter in (plugin.ironmule_kg_get_mask, plugin.ironmule_kg_get_mask6):
                getter.restype = ctypes.c_uint32
                bits.append(getter())
            if (actual != expected or mask_bits(self.kg) != bits or bits != data["mask_bits"]
                    or self.kg.get("restart_profile_reused", False) is not data["candidate"] or self.pilot is None
                    or checksum(self.pilot.weights_path) != data["pilot"] or mapped_vendor() != data["vendor"]):
                raise ValueError("ordinary product startup has a different calibrated profile or cache decision")

    # Only add observations after the real source constructor returns. Its normal measurement/replay remains active.
    ge.GgmlEngine = CheckedEngine
    from ironmule_product.worker import main as worker_main
    return worker_main(rest)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
