from __future__ import annotations

from pathlib import Path

from app.model import FinScribeEngine

BASE_DIR = Path(__file__).resolve().parent.parent
CATEGORIES_PATH = BASE_DIR / "data" / "categories.yaml"
EXEMPLARS_PATH = BASE_DIR / "data" / "exemplars.json"


def main():
    engine = FinScribeEngine(CATEGORIES_PATH, EXEMPLARS_PATH)

    examples = [
        "AMZN MKTP ORD 99X",
        "Starbuxs Mombay",
        "UBR TRP BLR",
        "SHELL FUL PMP",
        "Netflx.com subscrption",
    ]

    print("=== Robustness to Noisy Inputs ===")
    for text in examples:
        res = engine.predict(text)
        print(f"\nInput: {text}")
        print(f"  Predicted: {res['prediction']} (id={res['category_id']})")
        print(f"  Confidence: {res['confidence']:.3f}")
        print(f"  Needs review: {res['needs_review']}")
        print("  Top-3:")
        for c in res["top_3"]:
            print(f"    - {c['category_label']} (score {c['score']:.3f})")


if __name__ == "__main__":
    main()
