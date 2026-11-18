"""IronMule's local model of the benefit of a CPU thread action.

Weights start at zero. After a real replay, normalized gradient descent fits
log speedup from the measured time ratio. A token mismatch is a negative reward.
Hardware- and binding-specific weights supplement shared action and workload
features. The model ranks experiments; only the caller's local hardware gate
can activate an action.
"""
from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path

SCHEMA = 1


class HardwarePolicy:
    def __init__(self, path: Path, binding: str, hardware: str, cores: int, costs: dict[str, float],
                 profile: dict | None = None):
        self.path, self.binding, self.hardware, self.cores = path, binding, hardware, cores
        self.costs = costs
        self.profile = profile or {}
        self.seen = 0
        self.rejected = 0
        self.parameters = 0
        self.last = None
        self.initialized = False

    def _database(self):
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=0.1)
        if self.initialized:
            return db
        try:
            db.execute("""CREATE TABLE IF NOT EXISTS profiles
                       (binding TEXT PRIMARY KEY, hardware TEXT NOT NULL, details TEXT NOT NULL)""")
            db.execute("INSERT OR IGNORE INTO profiles VALUES(?,?,?)",
                       (self.binding, self.hardware, json.dumps(self.profile, sort_keys=True)))
            db.execute("CREATE TABLE IF NOT EXISTS weights (name TEXT PRIMARY KEY, value REAL NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS observations
                       (id INTEGER PRIMARY KEY, binding TEXT NOT NULL, hardware TEXT NOT NULL, regime TEXT NOT NULL,
                        threads INTEGER NOT NULL, stage TEXT NOT NULL, ratio REAL NOT NULL, matched INTEGER NOT NULL,
                        accepted INTEGER NOT NULL, aa_spread REAL, prediction REAL NOT NULL, reward REAL NOT NULL,
                        measured_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
            db.execute("CREATE INDEX IF NOT EXISTS by_action ON observations(binding, regime, threads)")
            db.execute("CREATE INDEX IF NOT EXISTS by_result ON observations(accepted)")
            db.commit()
            self.initialized = True
        except BaseException:
            db.close()
            raise
        return db

    def _features(self, regime: str, threads: int) -> dict[str, float]:
        ctx, base, rows, load = map(int, regime.split(":"))
        cores = self.cores
        delta = math.log2(threads / base)
        direction = "up" if delta > 0 else "down"
        # The continuous features transfer across core counts; sparse terms let
        # different hardware and measured regimes correct the shared estimate.
        step = max(1e-6, self.costs.get("step_one", 0.1))
        row = self.costs.get("row", step)
        kv = self.costs.get("kv_1k", step)
        return {f"action/{direction}": 1.0, "thread_ratio": delta, "thread_ratio_squared": delta * delta,
                "available_cores": delta * (cores - load - base) / cores,
                "context": delta * ctx / 8, "decode_rows": delta * math.log2(rows + 1),
                "row_price": delta * math.log1p(min(100, row / step)),
                "kv_price": delta * math.log1p(min(100, kv / step)),
                f"hardware/{self.hardware}/{direction}": 1.0,
                f"shape/{ctx}/{rows}/{load}/{direction}": 1.0,
                f"local/{self.binding}/{regime}/{threads}": 1.0}

    @staticmethod
    def _values(db, features):
        placeholders = ",".join("?" for _ in features)
        return dict(db.execute(f"SELECT name,value FROM weights WHERE name IN ({placeholders})", list(features)))

    def _refresh_counts(self, db):
        self.seen, self.rejected = db.execute("""SELECT (SELECT count(*) FROM observations),
                                        (SELECT count(*) FROM observations WHERE accepted=0)""").fetchone()
        self.parameters = db.execute("SELECT count(*) FROM weights").fetchone()[0]

    def rank(self, regime: str, candidates: list[int]) -> list[int]:
        """Trial priority includes a small bonus for sparsely tested local actions."""
        with closing(self._database()) as db:
            self._refresh_counts(db)
            attempts = dict(db.execute("""SELECT threads,count(*) FROM observations
                                      WHERE binding=? AND regime=? GROUP BY threads""", (self.binding, regime)))
            scores = {}
            for action in candidates:
                features = self._features(regime, action)
                values = self._values(db, features)
                prediction = sum(values.get(k, 0.0) * x for k, x in features.items())
                scores[action] = prediction + 0.15 / math.sqrt(1 + attempts.get(action, 0))
            return sorted(candidates, key=scores.get, reverse=True)

    def record(self, regime: str, threads: int, stage: str, ratio: float, matched: bool,
               accepted: bool, aa_spread: float | None = None):
        """One transaction remembers the outcome and its gradient, also across concurrent workers."""
        if not math.isfinite(ratio) or ratio <= 0:
            return
        features = self._features(regime, threads)
        reward = max(-1.0, min(1.0, -math.log(ratio))) if matched else -1.0
        if not accepted:
            reward = min(reward, 0.0)
        with closing(self._database()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            values = self._values(db, features)
            prediction = sum(values.get(k, 0.0) * x for k, x in features.items())
            step = 0.35 * (reward - prediction) / sum(x * x for x in features.values())
            db.executemany("""INSERT INTO weights(name,value) VALUES(?,?)
                           ON CONFLICT(name) DO UPDATE SET value=excluded.value""",
                           ((k, values.get(k, 0.0) + step * x) for k, x in features.items()))
            db.execute("""INSERT INTO observations(binding,hardware,regime,threads,stage,ratio,matched,accepted,
                       aa_spread,prediction,reward) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                       (self.binding, self.hardware, regime, threads, stage, ratio, int(matched),
                        int(accepted), aa_spread, prediction, reward))
            self._refresh_counts(db)
        self.last = {"prediction": prediction, "reward": reward, "stage": stage, "accepted": accepted}

    def status(self):
        return {"schema": SCHEMA, "observations": self.seen, "negative_or_unresolved": self.rejected,
                "parameters": self.parameters, "last_update": self.last}
