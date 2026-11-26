"""Exercise the compact KV plan with distinct near-limit documents and subsequent shared-prefix bursts."""

import hashlib
import random

from ironmule_product import compare
from lab.pk6.native_dispatch import run


def prompts(seed, width, count):
    rng = random.Random(seed)
    topics = ("alder", "birch", "cedar", "elm", "fir", "oak", "pine", "willow")
    # Alternate a distinct document per client with a long shared prefix across a whole burst.
    # Every request ends in a different lookup, independently of its prefix.
    common = " ".join(rng.choice(topics) for _ in range(900))
    group = []
    for i in range(count):
        unique = hashlib.sha256(f"{seed}/{i}".encode()).hexdigest()[:12]
        body = (
            common if i % 3 == 0 else " ".join(rng.choice(topics) for _ in range(900))
        )
        group.append(
            f"Inventory {'' if i % 3 == 0 else unique} {body}\nWrite four complete sentences on the inventory for lot {unique}."
        )
    return group


if __name__ == "__main__":
    import fcntl
    import os
    from pathlib import Path

    compare.performance_prompts = prompts
    with Path(
        os.environ.get("IRONMULE_LAB_GPU_LOCK", "/work/native-dispatch.lock")
    ).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run.main()
