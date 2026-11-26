"""Experimental packed native calls that advance generating jobs while a burst is prefilling."""
from __future__ import annotations

import time


def slices(waiting, lengths, positions, room, *, whole=False):
    """Consume at most room prompt rows, favour shorter uncached suffixes in stable order."""
    initial_room = room
    for seq in waiting:
        start = positions[seq]
        if whole and room < initial_room and lengths[seq] - start > room:
            break
        end = min(lengths[seq], start + room)
        if end > start:
            yield seq, start, end
            room -= end - start
        if not room:
            break


def serve(engine, group, active, take, finish, budget, *, whole=False):
    """All group slots are already owned by run(). Its take/finish manage tokens and KV lifetime."""
    jobs = {seq: job for seq, job, _ in group}
    lengths = {seq: len(job.ids) for seq, job in jobs.items()}
    positions = {seq: keep for seq, _, keep in group}
    waiting = sorted(jobs, key=lambda seq: lengths[seq] - positions[seq])
    if not len(group) < budget <= engine.batch_cap:
        raise ValueError("row budget must leave room for prompt progress and fit the native context")
    while waiting:
        for seq in list(active):
            job = jobs[seq]
            if job.cancelled and job.cancelled():
                finish(seq, job, "cancelled", keep=False)
        waiting = [seq for seq in waiting if seq in active]
        if not waiting:
            break
        producing = [seq for seq in active if jobs[seq].tokens]
        segments = {seq: (start, end) for seq, start, end in
                    slices(waiting, lengths, positions, budget - len(producing), whole=whole)}
        rows, outputs = [], []
        for seq in sorted(set(producing) | segments.keys()):
            job = jobs[seq]
            if seq in segments:
                start, end = segments[seq]
                rows.extend((job.ids[i], i, seq, i + 1 == lengths[seq]) for i in range(start, end))
                if end == lengths[seq]:
                    outputs.append(seq)
            else:
                rows.append((job.tokens[-1], lengths[seq] + len(job.tokens) - 1, seq, True))
                outputs.append(seq)
        began = time.monotonic()
        picks = engine._decode(rows)
        elapsed = time.monotonic() - began
        prompt = len(rows) - len(producing)
        if engine.observer is not None:
            engine.observer(len(producing), prompt,
                            sum(positions[s] + len(jobs[s].tokens) for s in active), elapsed)
        engine.pack_calls += 1
        engine.pack_rows += prompt
        for seq, (_, end) in segments.items():
            positions[seq] = end
        for seq, token in zip(outputs, picks, strict=True):
            take(seq, jobs[seq], token)
        waiting = [seq for seq in waiting if positions[seq] < lengths[seq] and seq in active]
    engine.pack_prompts += len(group)
