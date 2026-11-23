"""Build once and call the same C++17 trainer and scorer online and offline."""
from __future__ import annotations

import ctypes as C
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

SOURCE = Path(__file__).with_name("placement_model.cpp")
W = C.c_double * 5


class NativeModel:
    def __init__(self, cache: Path):
        cache.mkdir(parents=True, exist_ok=True)
        self.version = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        library = cache / f"placement-{self.version[:16]}-{os.uname().machine}.so"
        try:
            self.lib = C.CDLL(str(library.resolve()))
        except OSError:
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                output = Path(tmp) / "model.so"
                subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O3", "-shared", "-fPIC",
                                str(SOURCE), "-o", str(output)], check=True, timeout=120)
                os.replace(output, library)
            self.lib = C.CDLL(str(library.resolve()))
        ptr = C.POINTER(C.c_double)
        self.lib.placement_fit.argtypes = [ptr, C.c_size_t, ptr]
        self.lib.placement_fit.restype = C.c_int
        self.lib.placement_ms.argtypes = [ptr, C.c_double, C.c_double, C.c_double]
        self.lib.placement_ms.restype = C.c_double
        self.lib.placement_plan.argtypes = [ptr, ptr, C.c_int] + [C.c_double] * 5
        self.lib.placement_plan.restype = C.c_int

    def fit(self, rows):
        data = (C.c_double * (4 * len(rows)))(*(value for row in rows for value in row))
        weights = W()
        if not self.lib.placement_fit(data, len(rows), weights):
            raise ValueError("invalid native placement training data")
        return list(weights)

    def predict(self, weights, chars, limit, queue):
        return self.lib.placement_ms(W(*weights), chars, limit, queue)

    def choose(self, weights, chars, limit, queues, future, future_weight):
        lanes = len(queues)
        data = (C.c_double * (5 * lanes))(*(v for lane in weights for v in lane))
        backlog = (C.c_double * lanes)(*queues)
        return self.lib.placement_plan(data, backlog, lanes, chars, limit, *future, future_weight)
