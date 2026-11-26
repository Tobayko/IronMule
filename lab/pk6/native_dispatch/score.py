"""Replay paired Qwen3-32B HTTP comparisons from complete immutable process receipts."""

import argparse
from bisect import bisect_left
import gzip
import json
import math
from pathlib import Path

from lab.pk6.mixed_progress import grade
from lab.pk6.native_dispatch.run import SEEDS
from lab.pk6.warm_repeat import GGUF
from lab.portable_gpu.fetch_awq import _manifest_digest


def read(path):
    return json.loads(path.read_bytes())


def waves(folder, arm):
    path = folder / f"waves-{arm}.jsonl"
    raw = (
        path.read_bytes()
        if path.is_file()
        else gzip.decompress(path.with_suffix(".jsonl.gz").read_bytes())
    )
    return [json.loads(l) for l in raw.splitlines()]


def exclusion(arm, proof):
    if arm.startswith("B") and proof.get("fallback") == "stock":
        return "verified recovery to stock after explicit cuBLAS rejection"
    if arm.startswith("D"):
        return "instrumented discovery"
    if arm.startswith("P"):
        return "explicit cuBLAS threshold incorrectly qualified as identical stock; see PR 70"
    if (
        arm.startswith(("F", "V", "I", "M", "Z", "T", "X", "J", "G"))
        and proof.get("fallback") == "stock"
    ):
        return "product rejected the proposed numeric profile"
    if arm.startswith("G") and any(
        proof.get("profile_flags", {}).get(k) != "1"
        for k in ("IRONMULE_PK6", "IRONMULE_PS1")
    ):
        return "requested PS1 and PK6 profile did not serve"
    if arm.startswith("H") and not proof.get("verified_compact_pool"):
        return "requested pool capacity not witnessed in the serving worker"
    if arm.startswith(("F", "V", "I", "M", "Z", "T", "X", "J")) and (
        not any(proof["prompt_plugin_dispatches_in_http_by_width"].values())
        or not proof["prompt_plugin_dispatches_in_http_quality"]
    ):
        return "candidate path absent from counted performance or quality waves"
    if arm.startswith(("N", "G")) and (
        proof["shared_pool"] or proof["streams_pass"] is not True
    ):
        return "candidate split streams did not serve"
    if arm.startswith(("V", "U")) and not proof["raised_budget_in_http"]:
        return "candidate admission rule did not serve"
    return None


def wave_energy(samples, wave):
    """Integrate linearly interpolated board power. Missing endpoints stay unknown."""
    times = [r[0] for r in samples]
    a, b = wave["unix_start"], wave["unix_end"]
    left, right = bisect_left(times, a), bisect_left(times, b)
    if left == 0 or right >= len(times):
        return None
    joules = 0.0
    for i in range(left, right + 1):
        t0, _, p0 = samples[i - 1]
        t1, _, p1 = samples[i]
        if not 0 < t1 - t0 <= 2.5:
            return None
        lo, hi = max(t0, a), min(t1, b)
        rate = (p1 - p0) / (t1 - t0)
        joules += (p0 + 0.5 * rate * (lo + hi - 2 * t0)) * (hi - lo)
    return joules


def inspect(folder, arm, target, telemetry):
    start, ready, end = [
        read(folder / f"{part}-{arm}.json") for part in ("start", "ready", "end")
    ]
    if not (
        target["hardware"] == start["hardware"] == end["hardware"]
        and start["unix"] < end["unix"]
        and start["arm"] == end["arm"] == arm
    ):
        raise ValueError("invalid process receipt")
    rows = waves(folder, arm)
    seeds = SEEDS if arm == "vllm" else (SEEDS[int(arm[1:]) - 1],)
    expected = [
        (s, r, w)
        for s in seeds
        for r in range(4)
        for w in ((1, 8, 32) if r < 3 else (32,))
    ]
    if [(r["seed"], r["repeat"], r["width"]) for r in rows] != expected:
        raise ValueError("incomplete or reordered waves")
    end_prior = start["unix"]
    for r in rows:
        if not (
            r["arm"] == arm
            and end_prior <= r["unix_start"] < r["unix_end"] <= end["unix"]
        ):
            raise ValueError("out-of-process or overlapping requests")
        end_prior = r["unix_end"]
        grade.measure(r)
    metrics = {
        str(w): grade.summarize(
            [grade.measure(r) for r in rows if r["width"] == w and r["repeat"] < 3]
        )
        for w in (1, 8, 32)
    }
    facts = {
        str(r["seed"]): sum(x["correct"] is True for x in r["requests"])
        for r in rows
        if r["repeat"] == 3
    }
    sample = grade.sampled_resources(telemetry, rows)
    process_samples = [v for t, v, _ in telemetry if start["unix"] <= t <= end["unix"]]
    sample["process_peak_gpu_mib"] = max(process_samples) if process_samples else None
    energy = {}
    for w in (1, 8, 32):
        block = [r for r in rows if r["width"] == w and r["repeat"] < 3]
        estimates = [wave_energy(telemetry, r) for r in block]
        tokens = sum(x["tokens"] for r in block for x in r["requests"])
        energy[str(w)] = (
            sum(estimates) / tokens if all(e is not None for e in estimates) else None
        )
    sample["estimated_board_joules_per_output_token"] = energy
    quality = ready["health"].get("gate")
    if arm != "vllm" and not (
        quality and quality["passes"] is True and quality["interval"][1] <= 1.005
    ):
        raise ValueError("missing native quality gate")
    proof = {}
    if arm != "vllm":
        projection = grade.bindings(folder, arm, rows)
        trace = folder / f"trace-{arm}.txt"
        raw = (
            trace.read_bytes()
            if trace.is_file()
            else gzip.decompress(trace.with_suffix(".txt.gz").read_bytes())
        )
        marks = []
        budgets = []
        capacities = []
        for line in raw.decode().splitlines():
            if line.startswith("# physical_pool "):
                fields = dict(v.split("=") for v in line.split()[2:] if "=" in v)
                capacities.append(
                    [int(fields[k]) for k in ("cells", "allocated", "cap")]
                )
            if line.startswith("# shared_budget "):
                budgets.append(float(line.split()[-1]))
            if line.startswith("# prompt_kernel "):
                event = dict(part.split("=") for part in line.split()[2:])
                marks.append((float(event["t"]), int(event["launches"])))
        proof = {
            "profile_flags": ready["health"].get("kernel_profile", {}).get("flags", {}),
            "cpu_policy": target.get("cpu_policy"),
            "verified_compact_pool": bool(
                target.get("pool_plan")
                and ready["health"].get("shared_pool")
                and capacities
                and capacities[-1][:2] == [target["pool_plan"]["cells"]] * 2
                and capacities[-1][2] == ready["health"]["shared_pool"]["cap"]
            ),
            "observed_pool_capacities": capacities,
            "raised_budget_in_http": sum(
                1
                for r in rows
                if r["repeat"] < 3
                for t in budgets
                if min(x["t_send"] for x in r["requests"])
                <= t
                <= max(x["t_done"] for x in r["requests"])
            ),
            "selected_profile": ready["health"]
            .get("kernel_profile", {})
            .get("profile"),
            "fallback": ready["health"].get("kernel_fallback", {}).get("serving"),
            "matched_native_requests": sum(c["matched"] for c in projection["cells"]),
            "native_raw_sha256": projection["trace_sha256"],
            "shared_pool": bool(ready["health"].get("shared_pool")),
            "streams_pass": ready["health"]
            .get("stream_qualification", {})
            .get("passes"),
            "prompt_plugin_dispatches_in_http_by_width": {
                str(w): sum(
                    count
                    for r in rows
                    if r["width"] == w and r["repeat"] < 3
                    for t, count in marks
                    if min(x["t_send"] for x in r["requests"])
                    <= t
                    <= max(x["t_done"] for x in r["requests"])
                )
                for w in (1, 8, 32)
            },
            "prompt_plugin_dispatches_in_http_quality": sum(
                count
                for r in rows
                if r["repeat"] == 3
                for t, count in marks
                if min(x["t_send"] for x in r["requests"])
                <= t
                <= max(x["t_done"] for x in r["requests"])
            ),
        }
    summary = {
        "start": start["unix"],
        "end": end["unix"],
        "startup_s": ready["startup_s"],
        "quality": facts,
        "metrics": metrics,
        "resources": sample,
        "gate": quality,
        "excluded_from_promotion": exclusion(arm, proof),
        "proof": proof,
    }
    return rows, summary


def run(root):
    # Each process directory contains its actual target manifest. A missing end receipt is a separate rejected/incomplete arm.
    finished = sorted(root.rglob("end-*.json"))
    data, arms, binding = {}, {}, None
    for end in finished:
        arm = end.stem.split("-", 1)[1]
        folder = end.parent
        target = read(folder / "target.json")
        pin = [
            target[k]
            for k in (
                "hardware",
                "source",
                "gguf_sha256",
                "awq_manifest_sha256",
                "vllm_version",
                "seeds",
            )
        ]
        if (
            pin[0]["name"] != "NVIDIA L40S"
            or pin[1] != "2e3c1bb69b8cedc2b889156e6dd0912bc1a6235d"
            or pin[2:] != [GGUF, _manifest_digest(), "0.34.0", list(SEEDS)]
            or (binding is not None and pin != binding)
            or arm in arms
            or arm not in target["order"]
        ):
            raise ValueError(
                "different device, source, model, protocol, or duplicate arm"
            )
        binding = pin
        for d in (folder, folder.parent):
            csv_path = d / (d.name + ".csv")
            if csv_path.exists():
                break
        telemetry = grade.telemetry_samples(csv_path)
        data[arm], arms[arm] = inspect(folder, arm, target, telemetry)
    windows = sorted((a["start"], a["end"]) for a in arms.values())
    if any(a[1] > b[0] for a, b in zip(windows, windows[1:])):
        raise ValueError("shared measurement device was used concurrently")
    pairs = {}
    for arm, rows in data.items():
        if arm == "vllm":
            continue
        for ref in dict.fromkeys(
            ["vllm", "R" + arm[1:]] + (["A" + arm[1:]] if arm.startswith("T") else [])
        ):
            if ref == arm or ref not in data:
                continue
            seed = rows[0]["seed"]
            control = [r for r in data[ref] if r["seed"] == seed]
            if (
                not control
                or rows[-1]["input_digest"] != control[-1]["input_digest"]
                or [x["case"] for x in rows[-1]["requests"]]
                != [x["case"] for x in control[-1]["requests"]]
            ):
                raise ValueError("quality comparison changed")
            for w in (1, 8, 32):
                pair = grade.paired(
                    [r for r in control if r["width"] == w and r["repeat"] < 3],
                    [r for r in rows if r["width"] == w and r["repeat"] < 3],
                )
                pair["clear_win"] = pair["same_fixed_work"] and all(
                    v is not None
                    and math.isfinite(v)
                    and (v >= 1.1 if k == "tok_s" else v <= 0.9)
                    for k, v in pair["candidate_over_reference"].items()
                )
                pairs[f"{arm}/{ref}/{w}"] = pair
    complete = {}
    for name in sorted({a[0] for a in arms if a != "vllm"}):
        labels = [name + str(i) for i in (1, 2)]
        if not all(a in arms for a in labels):
            continue
        cells = [pairs.get(f"{a}/vllm/{w}") for a in labels for w in (1, 8, 32)]
        complete[name] = {
            "both_seeds_32_facts": all(
                set(arms[a]["quality"].values()) == {32} for a in labels
            ),
            "clear_on_every_timing_metric": all(
                c is not None and c["clear_win"] for c in cells
            ),
            "excluded": any(arms[a]["excluded_from_promotion"] for a in labels),
        }
    return {
        "binding": binding,
        "arms": arms,
        "pairs": pairs,
        "two_seed_evaluation": complete,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.root), sort_keys=True, indent=2, allow_nan=False))
