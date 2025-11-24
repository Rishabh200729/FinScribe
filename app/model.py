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
    from rapidfuzz import fuzz

    def build_explanation(self, text, prediction, scores, nearest_exemplars):
        """
        Local explainability engine (No GenAI).
        
        Args:
            text: raw transaction string
            prediction: predicted category_id
            scores: dict with semantic_score, exemplar_score, fuzzy_score
            nearest_exemplars: list of (text, category, similarity)
        
        Returns:
            dict: explanation object
        """

        # 1. Token-level fuzzy importance
        tokens = text.split()
        token_scores = []
        for tok in tokens:
            token_scores.append({
                "token": tok,
                "fuzzy_score": max(
                    fuzz.partial_ratio(tok, prediction),
                    max((fuzz.partial_ratio(tok, ex[0]) for ex in nearest_exemplars), default=0)
                )
            })

        # 2. Nearest exemplars formatted
        exemplar_info = [
            {
                "exemplar_text": ex_text,
                "exemplar_category": ex_cat,
                "similarity": float(sim)
            }
            for ex_text, ex_cat, sim in nearest_exemplars
        ]

        # 3. Convert raw scores to user-friendly language
        def explain_score(label, value):
            if value > 0.80: return f"Strong match with {label}"
            if value > 0.60: return f"Moderate match with {label}"
            return f"Weak match with {label}"

        friendly = {
            "semantic": explain_score("semantic patterns", scores.get("semantic_score", 0)),
            "exemplar": explain_score("past examples", scores.get("exemplar_score", 0)),
            "fuzzy": explain_score("text similarity", scores.get("fuzzy_score", 0))
        }

        # Final explanation dictionary
        return {
            "predicted_category": prediction,
            "scores": scores,
            "token_importance": token_scores,
            "nearest_exemplars": exemplar_info,
            "friendly_summary": friendly,
        }

    def predict(
        self,
        text: str,
        low_conf_threshold: float = 0.5,
        k_exemplars: int = 5,
    ) -> Dict[str, Any]:
        # 1. BASE CATEGORY PREDICTION (semantic)
        base_candidates = self._predict_base_category(text, top_k=5)

        # 2. FUZZY BOOST
        base_candidates = self._fuzzy_boost(text, base_candidates)

        # Extract semantic scores for explanation
        base_score_map = {cid:score for cid,score in base_candidates}

        # 3. EXEMPLAR SEARCH
        exemplar_matches = self.exemplar_store.search(text, k=k_exemplars)

        # Exemplar scoring
        exemplar_score_map = {}
        nearest_exemplars_list = []  # For explanation module

        for sim, entry in exemplar_matches:
            cid = entry["category_id"]
            exemplar_score_map[cid] = exemplar_score_map.get(cid, 0.0) + sim
            nearest_exemplars_list.append((entry["text"], entry["category_id"], sim))

        # 4. COMBINE SCORES
        scores: Dict[str, float] = {}

        # Semantic weight 0.4
        for cid, score in base_score_map.items():
            scores[cid] = scores.get(cid, 0.0) + (0.4 * score)

        # Exemplar weight 0.6
        for cid, score in exemplar_score_map.items():
            scores[cid] = scores.get(cid, 0.0) + (0.6 * score)

        # 5. NO SCORES → fallback
        if not scores:
            return {
                "prediction": None,
                "confidence": 0.0,
                "needs_review": True,
                "top_3": [],
                "exemplars": [],
                "explanation_terms": [],
            }

        # 6. SORT AND PICK BEST
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_scores[:3]

        best_cid, best_raw_score = top_3[0]
        best_confidence = float(min(best_raw_score, 1.0))

        labels = self.taxonomy.get_all_labels()

        # 7. FUZZY SCORE FOR EXPLANATION
        fuzzy_score = max(
            fuzz.partial_ratio(text.lower(), labels[best_cid].lower()) / 100.0,
            max((fuzz.partial_ratio(text.lower(), ex[0].lower()) for ex in nearest_exemplars_list), default=0) / 100.0
        )

        # Extract semantic + exemplar score for best category
        best_score_semantic = base_score_map.get(best_cid, 0.0)
        best_score_exemplar = exemplar_score_map.get(best_cid, 0.0)

        # 8. BASIC EXPLANATION TERMS
        explanation_terms = []
        pred_words = labels[best_cid].lower().replace("&", "").split()
        input_words = text.lower().split()

        for w in input_words:
            if any(p in w for p in pred_words):
                explanation_terms.append(w)

        if nearest_exemplars_list:
            exemplar_tokens = nearest_exemplars_list[0][0].lower().split()
            for w in input_words:
                if w in exemplar_tokens and w not in explanation_terms:
                    explanation_terms.append(w)

        if not explanation_terms:
            explanation_terms = ["semantic pattern match"]

        # 9. CALL EXPLAINABILITY ENGINE (your new local function)
        explanation = self.build_explanation(
            text=text,
            prediction=best_cid,
            scores={
                "semantic_score": float(best_score_semantic),
                "exemplar_score": float(best_score_exemplar),
                "fuzzy_score": float(fuzzy_score),
            },
            nearest_exemplars=nearest_exemplars_list
        )

        # 10. CONSTRUCT FINAL RESULT
        return {
            "prediction": labels[best_cid],
            "category_id": best_cid,
            "confidence": best_confidence,
            "needs_review": best_confidence < low_conf_threshold,
            "top_3": [
                {
                    "category_id": cid,
                    "category_label": labels[cid],
                    "score": float(min(1, score)),
                }
                for cid, score in top_3
            ],
            "exemplars": [
                {
                    "text": ex_text,
                    "category_id": ex_cat,
                    "category_label": labels.get(ex_cat, ex_cat),
                    "similarity": float(sim),
                }
                for ex_text, ex_cat, sim in nearest_exemplars_list
            ],
            "explanation_terms": explanation_terms,
            "explanation": explanation,
        }

    def add_feedback(self, text: str, category_id: str) -> None:
        self.exemplar_store.add_exemplar(text, category_id)