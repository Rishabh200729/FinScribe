from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from app.model import FinScribeEngine

BASE_DIR = Path(__file__).resolve().parent.parent
CATEGORIES_PATH = BASE_DIR / "config" / "categories.yaml"
EXEMPLARS_PATH = BASE_DIR / "config" / "exemplars.json"
OUT_DIR = BASE_DIR / "evaluation"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = FinScribeEngine(CATEGORIES_PATH, EXEMPLARS_PATH)
    labels_map = engine.taxonomy.get_all_labels()

    # Use all category label embeddings already in engine
    embs = engine.category_embs  # shape (n_categories, dim)
    ids = engine.category_ids

    pca = PCA(n_components=2)
    points = pca.fit_transform(embs)

    x = points[:, 0]
    y = points[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y)

    for i, cid in enumerate(ids):
        ax.text(x[i] + 0.02, y[i] + 0.02, labels_map[cid], fontsize=8)

    ax.set_title("Category Embeddings (PCA projection)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()

    out_path = OUT_DIR / "category_embeddings_pca.png"
    plt.savefig(out_path)
    print("Saved embedding visualization to", out_path)


if __name__ == "__main__":
    main()
