import yaml
import json
import os
import torch
from sentence_transformers import SentenceTransformer, util

class FinScribeEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2', data_dir='data'):
        print("🔄 Loading Model and Data... (This may take a few seconds)")
        
        # 1. Load the AI Model
        # We use 'cpu' here to make sure it runs on any laptop without a GPU
        self.model = SentenceTransformer(model_name, device='cpu')
        
        # 2. Paths to your data
        self.categories_path = os.path.join(data_dir, 'categories.yaml')
        self.exemplars_path = os.path.join(data_dir, 'exemplars.json')
        
        # 3. Load Data
        self.categories = self._load_yaml(self.categories_path)
        self.exemplars = self._load_json(self.exemplars_path)
        print(self.categories)
        print(self.exemplars)
        # 4. Pre-Compute Embeddings (The "Fast Lane")
        # We embed the descriptions ONCE so we don't have to do it for every request.
        self.category_embeddings = self._embed_categories()
        self.exemplar_embeddings = self._embed_exemplars()
        
        print("✅ FinScribe Engine is Ready!")

    def _load_yaml(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _load_json(self, path):
        if not os.path.exists(path):
            return {}
        with open(path, 'r') as f:
            return json.load(f)

    def _embed_categories(self):
        """
        Embeds the VALUES (descriptions) from the YAML file.
        Returns: Dictionary {Category_Name: Embedding_Vector}
        """
        embeddings = {}
        for category, keywords in self.categories.items():
            # We join keywords into a rich sentence for the AI to understand
            # e.g. "restaurants, fast food, cafes, coffee shops..."
            description_text = ", ".join(keywords)
            embeddings[category] = self.model.encode(description_text, convert_to_tensor=True)
        return embeddings

    def _embed_exemplars(self):
        """
        Embeds the specific examples from JSON.
        Returns: Dictionary {Category_Name: Tensor_of_Embeddings}
        """
        embeddings = {}
        for category, examples in self.exemplars.items():
            if examples:
                # Encode all examples for this category at once
                embeddings[category] = self.model.encode(examples, convert_to_tensor=True)
            else:
                embeddings[category] = None
        return embeddings

    def predict(self, transaction_str):
        """
        The Core Logic: Hybrid Matching
        1. Zero-Shot: Compare against Category Description
        2. Few-Shot: Compare against Past Exemplars
        """
        # Encode the incoming transaction
        query_embedding = self.model.encode(transaction_str, convert_to_tensor=True)
        
        scores = {}

        for category in self.categories.keys():
            # --- Score 1: Semantic Match (Description) ---
            cat_emb = self.category_embeddings[category]
            # util.cos_sim returns a list of lists, we take [0][0]
            semantic_score = util.cos_sim(query_embedding, cat_emb).item()
            
            # --- Score 2: Exemplar Match (Memory) ---
            exemplar_score = 0.0
            if category in self.exemplar_embeddings and self.exemplar_embeddings[category] is not None:
                # Compare input vs ALL exemplars for this category
                # We find the single closest match (max value)
                ex_sims = util.cos_sim(query_embedding, self.exemplar_embeddings[category])
                exemplar_score = torch.max(ex_sims).item()
            
            # --- Hybrid Logic ---
            # We give higher weight to exemplars if there is a very strong match (>0.85)
            # Otherwise we rely on the semantic description.
            if exemplar_score > 0.85:
                final_score = exemplar_score
            else:
                # Weighted average: 70% Semantic, 30% Exemplar boost
                final_score = (semantic_score * 0.7) + (exemplar_score * 0.3)
                
            scores[category] = final_score

        # Sort by score (highest first)
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        # Return Top Match and Top 3 Breakdown
        best_category, best_confidence = sorted_scores[0]
        return {
            "category": best_category,
            "confidence": round(best_confidence, 4),
            "top_3": sorted_scores[:3]
        }

    def learn(self, transaction_str, correct_category):
        """
        Instant Learning: Adds a new example to memory and updates embeddings immediately.
        """
        # 1. Update the In-Memory List
        if correct_category not in self.exemplars:
            self.exemplars[correct_category] = []
        
        if transaction_str not in self.exemplars[correct_category]:
            self.exemplars[correct_category].append(transaction_str)
            
            # 2. Save to Disk (Persistent Memory)
            with open(self.exemplars_path, 'w') as f:
                json.dump(self.exemplars, f, indent=2)
            
            # 3. Update Embeddings (Instant "Re-training")
            # We only need to re-embed this ONE category, not everything.
            new_embedding = self.model.encode(self.exemplars[correct_category], convert_to_tensor=True)
            self.exemplar_embeddings[correct_category] = new_embedding
            
            return True
        return False

# --- Quick Test (Run this file directly to check) ---
if __name__ == "__main__":
    engine = FinScribeEngine()
    
    test_tx = "Blinkit"
    print(f"\nTesting: {test_tx}")
    result = engine.predict(test_tx)
    print(f"Prediction: {result['category']} ({result['confidence']})")
    print(f"Details: {result['top_3']}")