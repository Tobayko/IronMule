from types import SimpleNamespace

from lab.pk6.mixed_progress.schedule import serve


def test_partial_prefills_share_real_decodes_and_keep_output_ownership():
    jobs = {
        n: SimpleNamespace(ids=[n + 100] * k, tokens=[], cancelled=None)
        for n, k in enumerate([3, 21, 4, 19, 5, 20, 6, 18])
    }
    kept = {s: s % 2 for s in jobs}
    kv = {s: list(j.ids[: kept[s]]) for s, j in jobs.items()}
    active = dict(reversed(jobs.items()))
    calls = []

    def decode(rows):
        assert len(rows) <= 19
        calls.append(rows)
        picks = []
        for token, pos, seq, logits in rows:
            assert pos == len(kv[seq])
            expected = (
                jobs[seq].ids[pos] if pos < len(jobs[seq].ids) else jobs[seq].tokens[-1]
            )
            assert token == expected
            kv[seq].append(token)
            if logits:
                picks.append(1000 * seq + len(kv[seq]))
        return picks

    def take(seq, job, pick):
        assert pick == 1000 * seq + len(job.ids) + len(job.tokens)
        job.tokens.append(pick)
        if len(job.tokens) == 3:
            del active[seq]

    e = SimpleNamespace(
        _decode=decode,
        pack_calls=0,
        pack_rows=0,
        pack_prompts=0,
        observer=None,
        batch_cap=40,
    )
    serve(e, [(s, j, kept[s]) for s, j in jobs.items()], active, take, None, 19)
    assert all(j.tokens for j in jobs.values())
    assert e.pack_rows == sum(len(j.ids) - kept[s] for s, j in jobs.items())
    assert any(
        any(pos >= len(jobs[s].ids) for _, pos, s, _ in call)
        and any(pos < len(jobs[s].ids) for _, pos, s, _ in call)
        for call in calls
    )


def test_cancellation_of_partially_prefilled_job_discards_its_cache():
    jobs = {
        i: SimpleNamespace(ids=[i] * n, tokens=[], cancelled=None)
        for i, n in enumerate((3, 4, 40, 60))
    }
    group = [(s, j, 0) for s, j in jobs.items()]
    live = jobs.copy()
    cancelled = []
    rows = []
    jobs[2].cancelled = lambda: any(s == 2 for _, _, s, _ in rows)

    def decode(step):
        rows.extend(step)
        return [s for _, _, s, logits in step if logits]

    def finish(s, j, reason, keep):
        assert reason == "cancelled" and not keep
        cancelled.append(s)
        del live[s]

    e = SimpleNamespace(
        _decode=decode,
        pack_calls=0,
        pack_rows=0,
        pack_prompts=0,
        observer=None,
        batch_cap=32,
    )
    serve(e, group, live, lambda s, j, t: j.tokens.append(t), finish, 16)
    assert cancelled == [2] and not jobs[2].tokens and jobs[3].tokens


def test_short_output_or_absent_eighth_chunk_cannot_become_a_fixed_work_win():
    from lab.pk6.mixed_progress.grade import measure, paired

    def wave(i):
        return {
            "width": 1,
            "repeat": i,
            "seed": 1,
            "input_digest": "same",
            "elapsed_s": 0.09,
            "requests": [
                {
                    "server_id": str(i),
                    "t_send": 1.0,
                    "t_done": 1.08,
                    "chunk_times": [1.02, 1.05],
                    "ttft_ms": 20.0,
                    "visible_ttft_ms": 20.0,
                    "tokens": 3,
                    "prompt_tokens": 33,
                    "finish_reason": "stop",
                    "output_sha256": "x",
                }
            ],
        }

    rows = [wave(i) for i in range(3)]
    assert measure(rows[0])["eighth"] is None
    compared = paired(rows, rows)
    assert (
        compared["same_fixed_work"] is False
        and compared["candidate_over_reference"]["eighth"] is None
    )


def test_forecast_and_replay_bind_the_choice_to_the_historical_a100_observations():
    import json
    from lab.pk6.first_token.phases import RESULTS
    from lab.pk6.mixed_progress.plan import plan

    historic = json.loads(
        (RESULTS / "mixed-prefill-progress" / "plan.json").read_bytes()
    )
    assert plan() == historic
    assert historic["selected_rows"] == 2048 and len(historic["simulated"]) == 6


def test_simulated_row_budgets_remain_grouped_and_outside_the_label_set():
    from training.experience_archive import ROOT, canonical_sha256
    from training.mixed_progress_evidence import experiences

    source = "lab/autonomous/results/mixed-prefill-progress/plan.json"
    rows = list(
        experiences(ROOT, source, canonical_sha256((ROOT / source).read_bytes()))
    )
    assert len(rows) == 6 and len({r["group"] for r in rows}) == 1
    assert all(r["provenance"] == "simulated" and not r["label_eligible"] for r in rows)
    assert [r["profiles"] for r in rows if r["decision"]] == [[{"row_budget": 2048}]]


def test_step_model_updates_only_on_the_training_start():
    from lab.pk6.mixed_progress.learning import fit

    def call(p, d, s):
        return {"prompt_rows": p, "decode_rows": d, "seconds": s}

    train = [call(1000, 10, 0.32), call(1500, 8, 0.48)] * 3
    prior = [0.09, 0, 0.29, 0]
    slow = fit(train, [call(1000, 10, 1.0)], prior)
    fast = fit(train, [call(1000, 10, 0.1)], prior)
    assert slow["updated"] == fast["updated"] != fast["prior"]
    assert (
        slow["holdout_mean_absolute_error_s"] != fast["holdout_mean_absolute_error_s"]
    )
    assert fit([], train, prior)["status"] == "insufficient_observed_mixed_calls"


def test_w4_screen_uses_only_whole_calibrated_shapes_within_the_stock_margin():
    from lab.pk6.mixed_progress.footprint import choose

    inventory = {
        "measured_shapes": [
            {"code": 12, "N": 32, "K": 256, "total_weight_bytes": 2**20},
            {"code": 14, "N": 64, "K": 256, "total_weight_bytes": 5 * 2**20},
        ]
    }
    selection = choose(inventory, 100, 90)
    assert selection["allowed_shapes"] == [[12, 32, 256]]
    assert selection["estimated_bytes"] <= selection["budget_bytes"]
    assert choose(inventory, 100, 99)["allowed_shapes"] == []


def test_shared_markers_belong_to_the_measured_http_window():
    from lab.pk6.mixed_progress.pool_score import markers

    wave = {
        "width": 1,
        "repeat": 0,
        "seed": 1,
        "requests": [{"t_send": 10.0, "t_done": 11.0}],
    }
    (count,) = markers(
        "# shared_budget from=128 to=1024 queued=4 9.5\n"
        "# shared_budget from=128 to=1024 queued=4 10.25",
        [wave],
    )
    assert count["changes"] == 1
