# Lab profiler: with IRONMULE_YAPPI_DIR set, every Python process profiles all threads (wall clock) and dumps the
# function table every 20 s to <dir>/<pid>.txt (the server is stopped with SIGTERM, so no atexit)
import os
d = os.environ.get("IRONMULE_YAPPI_DIR")
if d:
    try:
        import threading, time, yappi
        yappi.set_clock_type(os.environ.get("IRONMULE_YAPPI_CLOCK", "cpu"))
        yappi.start(builtins=False, profile_threads=True)
        def dump():
            while True:
                time.sleep(20)
                try:
                    stats = yappi.get_func_stats()
                    stats.sort("tsub", "desc")
                    with open(os.path.join(d, f"{os.getpid()}.txt"), "w") as fh:
                        fh.write(" ".join(os.sys.argv)[:200] + "\n")
                        for s in list(stats)[:80]:
                            fh.write(f"{s.tsub:10.3f} {s.ttot:10.3f} {s.ncall:9d}  {s.module}:{s.lineno} {s.name}\n")
                        fh.write("--- threads\n")
                        for t in yappi.get_thread_stats():
                            fh.write(f"{t.ttot:10.3f} {t.sched_count:8d} {t.name} {t.id}\n")
                except Exception as exc:
                    open(os.path.join(d, f"{os.getpid()}.err"), "w").write(repr(exc))
        threading.Thread(target=dump, daemon=True, name="yappi-dump").start()
    except Exception:
        pass
_sw = os.environ.get("IRONMULE_SWITCH")
if _sw:
    import sys
    sys.setswitchinterval(float(_sw))
