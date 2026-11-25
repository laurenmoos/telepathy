"""One log line: seconds since the start, then the message."""
import time

T0 = time.time()


def log(*x):
    print(f"[{time.time() - T0:7.0f}s]", *x, flush=True)
