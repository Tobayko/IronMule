"""A real-provider smoke that uploads code only and makes no speed claim."""
from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path
import re
import time
import uuid

from friday_evidence.canonical import canonical_sha256

from .contracts import code_digest, load_json, write_json_new
from .kaggle import KaggleCli, KaggleCliError
from .quota import AccountPreflight, QuotaController


class ProbeError(RuntimeError):
    pass


_REMOTE = r'''# IronMule DATA1 provider smoke: code-only, private, no performance claim.
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

BACKEND = __BACKEND__
RUN_ID = __RUN_ID__

def operands():
    left = ((np.arange(8192, dtype=np.float32) % 97) - 48).reshape(64, 128) / 97
    right = ((np.arange(8192, dtype=np.float32) % 89) - 44).reshape(128, 64) / 89
    return left, right

def run_cuda(left, right):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_unavailable")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    if "t4" not in props.name.lower():
        raise RuntimeError("unexpected_cuda_device")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    a = torch.from_numpy(left).to(device=device, dtype=torch.float32)
    b = torch.from_numpy(right).to(device=device, dtype=torch.float32)
    observed = (a @ b)
    torch.cuda.synchronize(device)
    result = observed.detach().cpu().numpy()
    return result, {"backend":"cuda", "accelerator":str(props.name),
        "compute_capability":list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes":int(props.total_memory), "framework":"torch",
        "framework_version":str(torch.__version__), "cuda_runtime":str(torch.version.cuda),
        "tf32":False}

def run_tpu(left, right):
    import jax
    import jax.numpy as jnp
    devices = [device for device in jax.devices() if device.platform == "tpu"]
    if not devices:
        raise RuntimeError("tpu_unavailable")
    device = devices[0]
    a = jax.device_put(left, device)
    b = jax.device_put(right, device)
    with jax.default_matmul_precision("float32"):
        observed = jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)
    result = np.asarray(jax.device_get(observed.block_until_ready()), dtype=np.float32)
    return result, {"backend":"tpu", "accelerator":str(device),
        "device_kind":str(getattr(device, "device_kind", "unknown")),
        "platform":str(device.platform), "framework":"jax",
        "framework_version":str(jax.__version__),
        "matmul_semantics":"float32_inputs_highest_available_precision"}

started = time.monotonic()
report = {"schema":"ironmule.provider-smoke.v1", "run_id":RUN_ID,
          "backend":BACKEND, "status":"failed", "hardware_verified":False,
          "correctness_verified":False, "performance_claim":False}
try:
    left, right = operands()
    expected = left @ right
    result, hardware = run_cuda(left, right) if BACKEND == "cuda" else run_tpu(left, right)
    exact_shape = result.shape == expected.shape
    finite = bool(np.isfinite(result).all())
    close = bool(np.allclose(result, expected, rtol=2e-4, atol=2e-4))
    if not (exact_shape and finite and close):
        raise RuntimeError("matmul_correctness_failed")
    report.update({"status":"passed", "hardware_verified":True,
                   "correctness_verified":True, "hardware":hardware})
except BaseException as exc:
    report["error_code"] = str(exc) if str(exc) in {
        "cuda_unavailable", "unexpected_cuda_device", "tpu_unavailable",
        "matmul_correctness_failed"} else type(exc).__name__
report["diagnostic_wall_seconds"] = time.monotonic() - started
target = Path("/kaggle/working/probe-result.json")
descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
    json.dump(report, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
if report["status"] != "passed":
    raise SystemExit(1)
'''


GEMMA3_1B_HANDLE = "google/gemma-3/transformers/gemma-3-1b-it/1"
WORKLOADS = ("matmul", "gemma3-1b")

_GEMMA_REMOTE = r'''# IronMule DATA2 inference smoke: private, no internet, no performance claim.
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

RUN_ID = __RUN_ID__
MODEL_HANDLE = __MODEL_HANDLE__
MAX_NEW_TOKENS = 32
GUEST_LIMIT_S = 60.0
PROMPT = "Explain in two sentences why the sky is blue."
ERRORS = {"cuda_unavailable", "unexpected_cuda_device", "model_not_attached",
          "unexpected_model_type", "unexpected_model_dtype", "generate_nondeterministic",
          "manual_generate_token_mismatch", "guest_deadline_exceeded"}


class Failure(Exception):
    pass


def locate_model(root):
    found = [p.parent for p in sorted(Path(root).rglob("config.json"))
             if (p.parent / "tokenizer.json").exists() or (p.parent / "tokenizer.model").exists()]
    if len(found) != 1:
        raise Failure("model_not_attached")
    return found[0]


def model_identity(path):
    small = {}
    for name in ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer.model", "tokenizer_config.json"):
        if (path / name).is_file():
            small[name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
    weights = {p.name: p.stat().st_size for p in sorted(path.glob("*.safetensors"))}
    # Kaggle model versions are immutable; hashing gigabytes would spend the guest budget.
    return {"handle": MODEL_HANDLE, "small_file_sha256": small,
            "weight_file_bytes": weights, "weight_sha256": "not_computed_guest_budget"}


def run_inference(path, device, deadline):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    import transformers

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def check():
        if time.monotonic() > deadline:
            raise Failure("guest_deadline_exceeded")

    tokenizer = AutoTokenizer.from_pretrained(path)
    loaded = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager")
    model = model.to(device).eval()
    sync()
    load_seconds = time.monotonic() - loaded
    if not str(model.config.model_type).startswith("gemma3"):
        raise Failure("unexpected_model_type")
    if next(model.parameters()).dtype != torch.float32:
        raise Failure("unexpected_model_dtype")
    check()

    encoded = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, return_dict=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        eos_token_id=eos, pad_token_id=tokenizer.pad_token_id, cache_implementation="dynamic")
    prompt_len = int(input_ids.shape[1])

    runs = []
    with torch.inference_mode():
        for _ in range(2):
            sync()
            start = time.perf_counter()
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                    generation_config=config)
            sync()
            runs.append((output[0, prompt_len:].tolist(), time.perf_counter() - start))
            check()
        if runs[0][0] != runs[1][0]:
            raise Failure("generate_nondeterministic")

        def forward(ids, past):
            try:
                return model(input_ids=ids, past_key_values=past, use_cache=True, logits_to_keep=1)
            except TypeError:
                return model(input_ids=ids, past_key_values=past, use_cache=True)

        sync()
        start = time.perf_counter()
        result = forward(input_ids, None)
        token = result.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        sync()
        ttft = time.perf_counter() - start
        manual = [int(token)]
        past = result.past_key_values
        while len(manual) < MAX_NEW_TOKENS and manual[-1] not in eos:
            result = forward(token, past)
            past = result.past_key_values
            token = result.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            manual.append(int(token))
        sync()
        total = time.perf_counter() - start
    check()
    if manual != runs[0][0]:
        raise Failure("manual_generate_token_mismatch")
    decode = (len(manual) - 1) / (total - ttft) if len(manual) > 1 and total > ttft else None
    return {
        "framework": "torch", "framework_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "model_type": str(model.config.model_type), "dtype": "float32",
        "attn_implementation": "eager", "prompt_tokens": prompt_len,
        "generated_token_ids": manual, "generated_text": tokenizer.decode(manual),
        "diagnostics": {"load_seconds": load_seconds, "generate_seconds": [r[1] for r in runs],
                        "manual_ttft_seconds": ttft, "manual_total_seconds": total,
                        "manual_decode_tokens_per_second": decode},
    }


def main():
    started = time.monotonic()
    report = {"schema": "ironmule.inference-smoke.v1", "run_id": RUN_ID, "backend": "cuda",
              "workload": "gemma3-1b", "status": "failed", "hardware_verified": False,
              "correctness_verified": False, "performance_claim": False}
    try:
        import torch
        if not torch.cuda.is_available():
            raise Failure("cuda_unavailable")
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        if "t4" not in props.name.lower():
            raise Failure("unexpected_cuda_device")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        report["hardware"] = {"accelerator": str(props.name),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(props.total_memory), "cuda_runtime": str(torch.version.cuda),
            "tf32": False}
        report["hardware_verified"] = True
        path = locate_model("/kaggle/input")
        report["model"] = model_identity(path)
        report.update(run_inference(path, device, started + GUEST_LIMIT_S))
        report["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
        report.update({"status": "passed", "correctness_verified": True})
    except BaseException as exc:
        report["error_code"] = str(exc) if str(exc) in ERRORS else type(exc).__name__
    report["diagnostic_wall_seconds"] = time.monotonic() - started
    target = Path("/kaggle/working/probe-result.json")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
'''


def _source(backend: str, run_id: str, workload: str = "matmul") -> str:
    if (backend not in {"cuda", "tpu"} or workload not in WORKLOADS
            or (workload == "gemma3-1b" and backend != "cuda")
            or not re.fullmatch(r"[0-9a-f]{32}", run_id)):
        raise ProbeError("invalid_probe_identity")
    if workload == "gemma3-1b":
        return _GEMMA_REMOTE.replace("__RUN_ID__", repr(run_id)).replace(
            "__MODEL_HANDLE__", repr(GEMMA3_1B_HANDLE))
    return _REMOTE.replace("__BACKEND__", repr(backend)).replace("__RUN_ID__", repr(run_id))


def _kernel_metadata(*, owner: str, run_id: str, backend: str, accelerator: str,
                     workload: str) -> dict:
    kind = "inference" if workload == "gemma3-1b" else "provider"
    # Kaggle derives the final slug from the title even when another id is
    # supplied. Keep both identical so status/output address the real kernel.
    metadata = {
        "id": f"{owner}/ironmule-{kind}-smoke-{run_id[:8]}",
        "title": f"IronMule {kind} smoke {run_id[:8]}",
        "code_file": "probe.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_internet": False,
        "enable_gpu": backend == "cuda", "enable_tpu": backend == "tpu",
        "machine_shape": accelerator,
    }
    if workload == "gemma3-1b":
        metadata["model_sources"] = [GEMMA3_1B_HANDLE]
    return metadata


def _write_text_new(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def load_account_preflight(path: Path) -> AccountPreflight:
    value = load_json(path)
    allowed = {field.name for field in fields(AccountPreflight)}
    values = {key: item for key, item in value.items() if key in allowed}
    if isinstance(values.get("supported_free_skus"), list):
        values["supported_free_skus"] = tuple(values["supported_free_skus"])
    return AccountPreflight(**values)


def run_provider_smoke(*, state_dir: Path, backend: str, owner: str, accelerator: str,
                       account: AccountPreflight, executable: str, workload: str = "matmul") -> dict:
    if workload not in WORKLOADS or (workload == "gemma3-1b" and backend != "cuda"):
        raise ProbeError("invalid_probe_workload")
    if backend == "cuda" and accelerator != "NvidiaTeslaT4":
        raise ProbeError("cuda_probe_requires_t4")
    if backend == "tpu" and not accelerator.startswith("Tpu"):
        raise ProbeError("tpu_probe_requires_verified_tpu")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,63}", owner):
        raise ProbeError("invalid_kaggle_owner")
    now = time.time()
    account.validate(now)
    resource = "gpu" if backend == "cuda" else "tpu"
    client = KaggleCli(executable)
    before = client.quota(resource=resource)
    controller = QuotaController(state_dir / "quota.sqlite3")
    controller.preflight(before, account)
    run_id = uuid.uuid4().hex
    metadata = _kernel_metadata(owner=owner, run_id=run_id, backend=backend,
                                accelerator=accelerator, workload=workload)
    kernel_id = metadata["id"]
    reservation = controller.reserve(before, account, run_slug=kernel_id.split("/", 1)[1],
                                     timeout_seconds=180, smoke=True)
    root = state_dir / "provider-probes" / run_id
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    project = root / "notebook"
    project.mkdir(mode=0o700)
    _write_text_new(project / "probe.py", _source(backend, run_id, workload))
    write_json_new(project / "kernel-metadata.json", metadata)
    write_json_new(root / "reservation.json", reservation.to_dict())
    started = time.monotonic()
    try:
        client.push(project, accelerator=accelerator, timeout_seconds=180, account=account)
    except BaseException:
        controller.mark_submission_unknown(reservation)
        raise
    terminal = None
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        terminal = client.job_state(kernel_id, metadata_path=project)
        if terminal in {"complete", "failed", "cancelled"}:
            break
        time.sleep(5)
    if terminal not in {"complete", "failed", "cancelled"}:
        controller.mark_terminal_unknown(reservation)
        raise ProbeError("provider_probe_terminal_unconfirmed")
    try:
        after = client.quota(resource=resource)
    except BaseException:
        controller.mark_quota_unknown(reservation)
        raise
    reconciliation = controller.reconcile_terminal(
        reservation, after, actual_elapsed_seconds=time.monotonic() - started, terminal=True)
    output = root / "output"
    output.mkdir(mode=0o700)
    client.output(kernel_id, output, metadata_path=project)
    reports = list(output.rglob("probe-result.json"))
    report = load_json(reports[0]) if len(reports) == 1 else None
    passed = bool(terminal == "complete" and report and report.get("run_id") == run_id
                  and report.get("backend") == backend and report.get("status") == "passed"
                  and report.get("workload", "matmul") == workload
                  and report.get("hardware_verified") is True
                  and report.get("correctness_verified") is True
                  and report.get("performance_claim") is False)
    if not passed:
        controller.mark_provider_failed(reservation)
    outcome = {"schema":"ironmule.provider-smoke-outcome.v1", "run_id":run_id,
        "backend":backend, "workload":workload, "kernel_id":kernel_id,
        "status":"passed" if passed else "failed",
        "provider_terminal_state":terminal, "report":report, "quota":reconciliation,
        "code_sha256":code_digest(), "performance_claim":False}
    write_json_new(root / "outcome.json", outcome)
    return outcome


__all__ = ["GEMMA3_1B_HANDLE", "ProbeError", "WORKLOADS", "load_account_preflight", "run_provider_smoke"]
