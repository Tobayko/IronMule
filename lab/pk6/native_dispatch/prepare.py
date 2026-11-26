"""Instrument an isolated PR 57 checkout. Only explicit lab switches change serving behavior."""

import argparse
from pathlib import Path

from lab.pk6.replay_http import checksum

BASE_ENGINE = "c8d388abff8fb24515c08622b40c6f41fd76f8f627d793d2f98c3bf1ffc66382"


def substitute(source, old, new):
    if source.count(old) != 1:
        raise ValueError("unrecognized native evaluator source")
    return source.replace(old, new)


def prepare(root, candidates=True, draft_mix=False):
    engine = root / "ironmule/ggml_engine.py"
    flags = root / "ironmule/ggml_profiles.py"
    if checksum(engine) != BASE_ENGINE:
        raise ValueError("unexpected native evaluator revision")
    code = engine.read_text()
    cache = flags.read_text()
    if candidates:
        code = substitute(
            code,
            "        params.n_ctx, params.n_batch, params.n_ubatch =",
            """        lab_shared_profile = os.environ.get("IRONMULE_LAB_SHARED_PROFILE", "")
        if lab_shared_profile and shared and self.gpu_layers:
            if lab_shared_profile != "wide":
                raise ValueError("unknown experimental shared-pool profile")
            physical = min(logical, 2048)
        params.n_ctx, params.n_batch, params.n_ubatch =""",
        )
        code = substitute(
            code,
            '                mark("pilot_admit")',
            """                if (self.pool is not None and not PROMPT_BUDGET and len(pending) >= 4
                        and os.environ.get("IRONMULE_LAB_SHARED_PROFILE") == "wide"
                        and not any(job.hardware_probe for job in pending)):
                    limit = min(self.seq_ctx, self.batch_cap, 2048) - len(active)
                    if budget < limit:
                        trace_note(f"shared_budget from={budget} to={limit} queued={len(pending)}")
                        budget = limit
                mark("pilot_admit")""",
        )
        code = substitute(
            code,
            '        streams = probe({**chosen["settings"], "flash_attn": True, "kv_unified": False}, "streams")',
            '        streams = probe({**chosen["settings"], "flash_attn": True, "kv_unified": False}, "streams")\n'
            '        if os.getenv("IRONMULE_LAB_STREAM_ALTERNATIVE") == "plain" and not streams.get("passes"):\n'
            "            flash_result = streams\n"
            '            streams = probe({**chosen["settings"], "flash_attn": False, "kv_unified": False}, "plain_streams")\n'
            '            streams = {**streams, "previous_flash_result": flash_result}',
        )
        # Ignore low-row W4 calibration only for a plugin that declares a prompt-only ABI.
        code = substitute(
            code,
            "            lib.ironmule_kg_set_mask.argtypes = [ctypes.c_uint32]\n            lib.ironmule_kg_table.restype = ctypes.c_char_p",
            """            if hasattr(lib, "im_prompt_threshold"):
                lib.im_prompt_threshold.restype = ctypes.c_int
                return {"family":"prompt_only", "minimum_rows":lib.im_prompt_threshold(), "tokens_served":[]}
            lib.ironmule_kg_set_mask.argtypes = [ctypes.c_uint32]
            lib.ironmule_kg_table.restype = ctypes.c_char_p""",
        )
        code += """
if os.environ.get("IRONMULE_LAB_KERNEL_TRACE") and TRACE:
    from lab.pk6.native_dispatch.capture import count_dispatches
    count_dispatches(GgmlEngine, TRACE)
"""
    code += """
if os.environ.get("IRONMULE_LAB_ACTIVITY"):
    from lab.pk6.native_dispatch.capture import install
    install(GgmlEngine)
"""
    if candidates:
        # Tile selection is part of the measured native path, including default attention.
        code = substitute(
            code,
            '"IRONMULE_CUBLAS_MIN_ROWS": "0"} if stock else env)',
            '"IRONMULE_CUBLAS_MIN_ROWS": "0", "IRONMULE_LAB_MMQ_TILE": ""} if stock else env)',
        )
        code = substitute(
            code,
            'if experimental or prefill_batch or os.environ.get("IRONMULE_KG_LIB"):',
            'if experimental or prefill_batch or os.getenv("IRONMULE_LAB_MMQ_TILE") or os.environ.get("IRONMULE_KG_LIB"):',
        )
        code = substitute(
            code,
            "and not prefill_batch:\n            tried.append(",
            'and not prefill_batch and not os.getenv("IRONMULE_LAB_MMQ_TILE"):\n            tried.append(',
        )
        code = substitute(
            code,
            '    default_alternative = kernel == ""',
            '    default_alternative = kernel == "" and not os.getenv("IRONMULE_LAB_MMQ_TILE")',
        )
    cache = substitute(
        cache,
        "    return key",
        """    if os.getenv("IRONMULE_CUBLAS_MIN_ROWS") or os.getenv("IRONMULE_LAB_FP8"):
        key.extend([os.getenv("IRONMULE_CUBLAS_MIN_ROWS"), os.getenv("IRONMULE_LAB_FP8")])
    return key""",
    )
    if candidates:
        cache = substitute(
            cache,
            "    return key",
            """    if os.getenv("IRONMULE_LAB_SHARED_PROFILE"):
        key.extend(["shared-physical-budget", os.environ["IRONMULE_LAB_SHARED_PROFILE"]])
    return key""",
        )
    if candidates:
        cache = substitute(
            cache,
            "    return key",
            '    if os.getenv("IRONMULE_LAB_STREAM_ALTERNATIVE"):\n'
            '        key.extend(["stream-alternative", os.environ["IRONMULE_LAB_STREAM_ALTERNATIVE"]])\n    return key',
        )
    if candidates:
        cache = substitute(
            cache,
            "    return key",
            '    if os.getenv("IRONMULE_LAB_MMQ_TILE"):\n'
            '        key.extend(["qualified-native-tile", os.environ["IRONMULE_LAB_MMQ_TILE"]])\n    return key',
        )
    if candidates:
        code = substitute(
            code,
            "        params.n_ctx, params.n_batch, params.n_ubatch = seqs * seq_ctx, logical, physical",
            """        cells = seqs * seq_ctx
        if shared and self.gpu_layers and os.environ.get("IRONMULE_LAB_POOL_CELLS"):
            requested = int(os.environ["IRONMULE_LAB_POOL_CELLS"])
            if requested < 2 * seq_ctx or requested % 256:
                raise ValueError("invalid experimental unified pool size")
            cells = min(cells, requested)
        params.n_ctx, params.n_batch, params.n_ubatch = cells, logical, physical""",
        )
        code = substitute(
            code,
            "self.pool = SharedPool(self, seqs * seq_ctx, self.cap, ids, MAX_DRAFT_SLACK) if shared else None",
            """self.pool = SharedPool(self, cells, self.cap, ids, MAX_DRAFT_SLACK) if shared else None
        if shared and TRACE and os.environ.get("IRONMULE_LAB_POOL_CELLS"):
            trace_note(f"physical_pool cells={self.pool.cells} allocated={self.L.llama_n_ctx(self.ctx)} cap={self.pool.cap}")""",
        )
        cache = substitute(
            cache,
            "    return key",
            '    if os.getenv("IRONMULE_LAB_POOL_CELLS"):\n'
            '        key.extend(["unified-pool-cells", os.environ["IRONMULE_LAB_POOL_CELLS"]])\n    return key',
        )
    if draft_mix:
        code = substitute(
            code,
            "                modelled: dict[int, list[int]] = {}",
            """                modelled: dict[int, list[int]] = {}
                local_drafts = {}
                draft_mix = os.environ.get("IRONMULE_LAB_DRAFT_MIX") if self.drafter is not None else None""",
        )
        code = substitute(
            code,
            "                            wants[seq] = (job.full_context(), room)",
            """                            if (draft_mix == "context_first" and self.policy is not None
                                    and len(order) <= self.drafter.max_running):
                                offer = job.context.draft(room)
                                if self.collective is not None and len(offer[0]) < 2:
                                    shared_offer = self.collective.draft(job.full_context(), room)
                                    if len(shared_offer[0]) > len(offer[0]):
                                        offer = shared_offer
                                if len(offer[0]) >= 2:
                                    local_drafts[seq] = offer
                                    continue
                            wants[seq] = (job.full_context(), room)""",
        )
        code = substitute(
            code,
            "                    if seq in modelled:\n                        offers.append((modelled[seq], DRAFT_BUCKET))",
            """                    if seq in local_drafts:
                        offers.append(local_drafts[seq])
                        continue
                    if seq in modelled:
                        offers.append((modelled[seq], DRAFT_BUCKET))""",
        )
        code = substitute(
            code,
            "                        self.policy.observe_drafts(bucket, r, min(r, len(new) - 1))",
            """                        self.policy.observe_drafts(bucket, r, min(r, len(new) - 1))
                    if draft_mix and TRACE and r and (seq in modelled or seq in local_drafts):
                        trace_note(f"draft_verified source={'model' if seq in modelled else 'context'} rows={r} accepted={min(r, len(new)-1)}")""",
        )
    engine.write_text(code)
    flags.write_text(cache)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--draft-mix", action="store_true")
    args = p.parse_args()
    prepare(args.root, not args.baseline, args.draft_mix)
