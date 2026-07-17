import numpy as np
from collections import defaultdict
from .index import VectorIndex
from .model import FashionCLIPModel

class FashionSearchEngine:
    def __init__(self, index: VectorIndex, model_handler: FashionCLIPModel):
        self.index = index
        self.model = model_handler
        self.heads = {}
        self.top_k_config = {"category": 5, "attribute": 20, "style": 5}

    def set_semantic_heads(self, categories: list, attributes: list, styles: list):
        self.heads["category"] = {"labels": categories, "embeddings": self.model.encode_texts(categories)}
        self.heads["attribute"] = {"labels": attributes, "embeddings": self.model.encode_texts(attributes)}
        self.heads["style"] = {"labels": styles, "embeddings": self.model.encode_texts(styles)}

    def _top_k_mask(self, scores, k):
        masked = np.zeros_like(scores)
        idx = np.argpartition(scores, -k)[-k:]
        masked[idx] = scores[idx]
        s = masked.sum()
        if s > 0:
            masked /= s
        return masked

    def retrieve(self, query: str, image_embeddings: np.ndarray, k: int = 5, rerank_k: int = 30):
        query_embedding = self.model.encode_texts([query])[0]
        
        labels, distances = self.index.index.knn_query(query_embedding, k=rerank_k)
        candidate_ids = labels[0]
        image_similarity = 1 - distances[0]
        candidate_embeddings = image_embeddings[candidate_ids]

        query_projections = {}
        for head_name, head_data in self.heads.items():
            scores = (query_embedding[None] @ head_data["embeddings"].T)[0]
            scores = self._top_k_mask(scores, self.top_k_config[head_name])
            query_projections[head_name] = {"scores": scores, "active": np.flatnonzero(scores)}

        rankings = {"image": np.argsort(image_similarity)[::-1]}
        for head_name, head_data in self.heads.items():
            candidate_scores = candidate_embeddings @ head_data["embeddings"].T
            active = query_projections[head_name]["active"]
            
            if len(active) == 0:
                scores = np.zeros(candidate_scores.shape[0])
            else:
                scores = candidate_scores[:, active] @ query_projections[head_name]["scores"][active]
                
            rankings[head_name] = np.argsort(scores)[::-1]

        weights = {
            "image": 0.85,
            "category": 0.05,
            "attribute": 0.05,
            "style": 0.05,
        }

        rrf_scores = defaultdict(float)
        for name, ranking in rankings.items():
            w = weights[name]
            for rank, idx in enumerate(ranking):
                rrf_scores[idx] += w / (60 + rank + 1)

        final_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for candidate_idx, score in final_ranked[:k]:
            image_idx = candidate_ids[candidate_idx]
            results.append({
                "image_id": int(image_idx),
                "path": self.index.all_paths[image_idx],
                "score": float(score),
            })
        return results