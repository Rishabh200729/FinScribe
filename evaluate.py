from __future__ import annotations

from pathlib import Path
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from app.model import FinScribeEngine

BASE_DIR = Path(__file__).resolve().parent
CATEGORIES_PATH = BASE_DIR / "data" / "categories.yaml"
EXEMPLARS_PATH = BASE_DIR / "data" / "exemplars.json"
DATA_PATH = BASE_DIR / "evaluation" / "synthetic_data.csv"
OUT_DIR = BASE_DIR / "evaluation"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    engine = FinScribeEngine(CATEGORIES_PATH, EXEMPLARS_PATH)

    y_true = df["label"].tolist()
    y_pred = []
    conf_scores = []

    start_time = time.time()
    for text in df["text"]:
        result = engine.predict(text)
        y_pred.append(result["category_id"])
        conf_scores.append(result["confidence"])
    total_time = time.time() - start_time

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    avg_conf = float(np.mean(conf_scores))
    avg_latency_ms = (total_time / len(df)) * 1000.0

    print("=== Classification Report ===")
    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    print(report)

    print(f"Macro F1: {macro_f1:.3f}")
    print(f"Average confidence: {avg_conf:.3f}")
    print(f"Average latency per transaction: {avg_latency_ms:.2f} ms")

    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    plt.colorbar(im)
    plt.tight_layout()
    cm_path = OUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    print("Saved confusion matrix to", cm_path)

    metrics = {
        "macro_f1": macro_f1,
        "average_confidence": avg_conf,
        "avg_latency_ms": avg_latency_ms,
        "n_samples": len(df),
    }
    metrics_path = OUT_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to", metrics_path)

    report_path = OUT_DIR / "classification_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report)
    print("Saved detailed report to", report_path)


if __name__ == "__main__":
    main()
