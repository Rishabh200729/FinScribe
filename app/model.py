from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Any
import json

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

from .config_loader import CategoryTaxonomy


class ExemplarStore:
    def __init__(self, path: str | Path, model: SentenceTransformer):
        self.path = Path(path)
        self.model = model
        self.entries: List[Dict[str, str]] = []
        self.embeddings: np.ndarray | None = None
        self.index: faiss.Index | None = None
        self._load()
        self._build_index()

    def _load(self) -> None:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                self.entries = json.load(f)
        else:
            self.entries = []

    def _save(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)

    def _build_index(self) -> None:
        if not self.entries:
            self.embeddings = None
            self.index = None
            return
        texts = [e["text"] for e in self.entries]
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.embeddings = embs.astype("float32")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def add_exemplar(self, text: str, category_id: str) -> None:
        self.entries.append({"text": text, "category_id": category_id})
        self._save()
        self._build_index()

    def search(self, text: str, k: int = 5) -> List[Tuple[float, Dict[str, str]]]:
        if self.index is None or self.embeddings is None or not self.entries:
            return []

        query_vec = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(query_vec, min(k, len(self.entries)))
        results: List[Tuple[float, Dict[str, str]]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            entry = self.entries[idx]
            results.append((float(score), entry))
        return results


class FinScribeEngine:
    def __init__(
        self,
        categories_path: str | Path,
        exemplars_path: str | Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self.taxonomy = CategoryTaxonomy(categories_path)
        
        # --- ERROR HANDLING FOR MISSING CATEGORIES ---
        if not self.taxonomy.get_all_labels():
            raise ValueError(f"No categories loaded from {categories_path}. Check YAML format.")
            
        self.exemplar_store = ExemplarStore(exemplars_path, self.model)

        self.category_ids = list(self.taxonomy.get_all_labels().keys())
        labels = [self.taxonomy.get_all_labels()[cid] for cid in self.category_ids]
        self.category_embs = self.model.encode(labels, convert_to_numpy=True, normalize_embeddings=True).astype(
            "float32"
        )

        dim = self.category_embs.shape[1]
        self.category_index = faiss.IndexFlatIP(dim)
        self.category_index.add(self.category_embs)

    def reload_taxonomy(self) -> None:
        self.taxonomy.reload()
        self.category_ids = list(self.taxonomy.get_all_labels().keys())
        labels = [self.taxonomy.get_all_labels()[cid] for cid in self.category_ids]
        self.category_embs = self.model.encode(labels, convert_to_numpy=True, normalize_embeddings=True).astype(
            "float32"
        )
        dim = self.category_embs.shape[1]
        self.category_index = faiss.IndexFlatIP(dim)
        self.category_index.add(self.category_embs)

    def _predict_base_category(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        query_vec = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, idxs = self.category_index.search(query_vec, min(top_k, len(self.category_ids)))
        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            cid = self.category_ids[idx]
            results.append((cid, float(score)))
        return results

    def _fuzzy_boost(self, text: str, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        boosted = []
        labels = self.taxonomy.get_all_labels()
        for cid, score in candidates:
            label = labels[cid]
            fuzz_score = fuzz.partial_ratio(text.lower(), label.lower()) / 100.0
            boosted.append((cid, score + 0.1 * fuzz_score))
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    def predict(
        self,
        text: str,
        low_conf_threshold: float = 0.5, # Lowered slightly since we removed normalization
        k_exemplars: int = 5,
    ) -> Dict[str, Any]:
        base_candidates = self._predict_base_category(text, top_k=5)
        base_candidates = self._fuzzy_boost(text, base_candidates)

        exemplar_matches = self.exemplar_store.search(text, k=k_exemplars)

        scores: Dict[str, float] = {}
        for cid, score in base_candidates:
            scores[cid] = scores.get(cid, 0.0) + (0.4 * score) # Reduced semantic weight

        for sim, entry in exemplar_matches:
            print(sim, entry)
            cid = entry["category_id"]
            # Increased exemplar weight to favor memory over fuzzy semantics
            scores[cid] = scores.get(cid, 0.0) + (0.6 * sim) 

        if not scores:
            return {
                "prediction": None,
                "confidence": 0.0,
                "needs_review": True,
                "top_3": [],
                "exemplars": [],
                "explanation_terms": [],
            }

        # --- REMOVED NORMALIZATION ---
        # Do not divide by max_score. Let raw scores compete.
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_scores[:3]
        best_cid, raw_score = top_3[0]
        best_score = min(raw_score, 1.0)  # Clamp to 1.0 for confidence
        
        labels = self.taxonomy.get_all_labels()
        
        # BETTER EXPLAINABILITY LOGIC
        explanation_terms = []
        pred_label_words = labels[best_cid].lower().replace('&', '').split()
        input_words = text.lower().split()
        
        # Match against label
        for word in input_words:
            # Simple containment check
            if any(l_word in word for l_word in pred_label_words if len(l_word)>2): 
                explanation_terms.append(word)
                
        # Match against top exemplar
        if exemplar_matches:
            top_ex_words = exemplar_matches[0][1]['text'].lower().split()
            for word in input_words:
                if word in top_ex_words and word not in explanation_terms:
                    explanation_terms.append(word)
        
        if not explanation_terms:
            explanation_terms = ["Semantic Similarity"]

        result = {
            "prediction": labels[best_cid],
            "category_id": best_cid,
            "confidence": float(best_score),
            "needs_review": best_score < low_conf_threshold,
            "top_3": [
                {
                    "category_id": cid,
                    "category_label": labels[cid],
                    "score": float(min(1,score)),
                }
                for cid, score in top_3
            ],
            "exemplars": [
                {
                    "text": e["text"],
                    "category_id": e["category_id"],
                    "category_label": labels.get(e["category_id"], e["category_id"]),
                    "similarity": float(sim),
                }
                for sim, e in exemplar_matches
            ],
            "explanation_terms": list(set(explanation_terms)),
        }
        return result

    def add_feedback(self, text: str, category_id: str) -> None:
        self.exemplar_store.add_exemplar(text, category_id)