import sys
import time


class ProgressLogger:
    """Minimal progress logger for long-running batch runs — no external deps."""

    def __init__(self, total: int, label: str = "run"):
        self.total = total
        self.label = label
        self.start = time.time()
        self.n = 0

    def step(self) -> None:
        self.n += 1
        elapsed = time.time() - self.start
        rate = self.n / elapsed if elapsed > 0 else 0.0
        print(
            f"[{self.label}] {self.n}/{self.total} ({rate:.2f}/s, {elapsed:.0f}s elapsed)",
            file=sys.stderr,
        )
