from __future__ import annotations

from pathlib import Path
import random
import string
import time

from app.model import FinScribeEngine

BASE_DIR = Path(__file__).resolve().parent
CATEGORIES_PATH = BASE_DIR / "data" / "categories.yaml"
EXEMPLARS_PATH = BASE_DIR / "data" / "exemplars.json"


def random_txn(n_words: int = 3) -> str:
    merchants = ["AMZN", "STARBUCKS", "UBER", "SHELL", "NETFLIX", "BIGBAZAAR"]
    extras = ["ORD", "PAYMENT", "TRIP", "PUMP", "SUBS", "GROCERY"]
    return " ".join(
        random.choice(merchants + extras + ["".join(random.choices(string.ascii_uppercase, k=4))])
        for _ in range(n_words)
    )


def main():
    engine = FinScribeEngine(CATEGORIES_PATH, EXEMPLARS_PATH)

    n = 1000
    texts = [random_txn() for _ in range(n)]

    print(f"Running batch benchmark on {n} synthetic transactions...")

    start = time.time()
    for t in texts:
        engine.predict(t)
    total = time.time() - start

    avg_ms = (total / n) * 1000.0
    tps = n / total

    print(f"Total time: {total:.3f} s")
    print(f"Avg latency: {avg_ms:.2f} ms per transaction")
    print(f"Throughput: {tps:.1f} transactions per second")


if __name__ == "__main__":
    main()
