"""Run a command while H busy processes keep H cores of this machine busy (a shared server). Usage: busy.py H CMD..."""
import multiprocessing as mp, subprocess, sys, time
def hog(stop):
    x = 0
    while not stop.is_set():
        for _ in range(100000): x += 1
if __name__ == "__main__":
    h, cmd = int(sys.argv[1]), sys.argv[2:]
    stop = mp.Event(); ps = [mp.Process(target=hog, args=(stop,), daemon=True) for _ in range(h)]
    [p.start() for p in ps]; time.sleep(1.0)
    try:
        code = subprocess.call(cmd)
    finally:
        stop.set(); [p.join() for p in ps]
    sys.exit(code)
