import hnswlib
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from .model import FashionEmbedder
from .dataset import create_dataloader
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

class FashionSearchEngine:
    def __init__(self, embedder: FashionEmbedder, dimension: int = 768):
        self.embedder = embedder
        self.dimension = dimension
        self.index = None
        self.all_paths = []

    def build_index(self, image_paths: list, batch_size: int = 32):
        loader = create_dataloader(image_paths, self.embedder.preprocess_val, batch_size=batch_size)
        all_embeddings = []
        self.all_paths = []

        device_type = self.embedder.device.type
        use_amp = device_type == 'cuda'
        autocast_device = 'cuda' if use_amp else 'cpu'
        
        with torch.inference_mode(), torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
            for images, paths in tqdm(loader, desc="Building Search Index"):
                images = images.to(self.embedder.device)
                embeddings = self.embedder.model.encode_image(images, normalize=True)
                all_embeddings.append(embeddings.cpu().numpy())
                self.all_paths.extend(paths)

        embeddings_np = np.vstack(all_embeddings).astype(np.float32)
        num_elements = embeddings_np.shape[0]

        self.index = hnswlib.Index(space="cosine", dim=self.dimension)
        self.index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        self.index.add_items(embeddings_np, np.arange(num_elements))
        self.index.set_ef(50)
        print(f"Indexed {num_elements} images successfully.")

    def save(self, filepath: str):
        self.index.save_index(filepath)
        metadata = {"paths": self.all_paths,  "dimension": self.dimension,}
        with open(filepath + ".meta.pkl", "wb") as f:
            pickle.dump(metadata, f)

        print(f"Saved index to {filepath}")

    def load(self, filepath: str):
        meta_file = filepath + ".meta.pkl"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Index file not found: {filepath}")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")
        with open(meta_file, "rb") as f:
            metadata = pickle.load(f)
            
        self.dimension = metadata["dimension"]
        self.all_paths = metadata["paths"]
        self.index = hnswlib.Index(space="cosine", dim=self.dimension)
        self.index.load_index(filepath)

        print(f"Loaded {len(self.all_paths)} indexed images.")

    def _retrieve_initial(self, query_text: str, k: int = 30) -> list:
        query_embedding = self.embedder.encode_text(query_text)
        indices, distances = self.index.knn_query(query_embedding, k=k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                "path": self.all_paths[idx],
                "score": float(1.0 - dist)
            })
        return results

    @staticmethod
    def _load_image(path):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    def query(self, query_text: str, k_initial: int = 30, k_final: int = 10, batch_size: int = 8) -> list:
        if self.index is None:
            raise ValueError("Index is not loaded or built.")

        initial_results = self._retrieve_initial(query_text, k=k_initial)
        reranked_results = []

        paths = [res["path"] for res in initial_results]

        with ThreadPoolExecutor(max_workers=min(8, len(paths))) as executor:
            loaded_images = list(executor.map(self._load_image, paths))

        valid_images = []
        valid_results = []

        for image, res in zip(loaded_images, initial_results):
            if image is None:
                print(f"Skipping unreadable image {res['path']}")
                continue

            valid_images.append(image)
            valid_results.append(res)

        device_type = self.embedder.device.type
        use_amp = device_type == "cuda"
        autocast_device = "cuda" if use_amp else "cpu"

        with torch.inference_mode(), torch.amp.autocast(
            device_type=autocast_device,
            enabled=use_amp,
        ):
            for start in range(0, len(valid_images), batch_size):

                batch_images = valid_images[start:start + batch_size]
                batch_results = valid_results[start:start + batch_size]

                inputs = self.embedder.cross_processor(
                    images=batch_images,
                    text=[query_text] * len(batch_images),
                    return_tensors="pt",
                    padding=True
                )

                inputs = {k: v.to(self.embedder.device) for k, v in inputs.items()}

                outputs = self.embedder.cross_model(**inputs)

                itm_scores = torch.nn.functional.softmax(outputs.itm_score, dim=1)
                match_probabilities = itm_scores[:, 1].cpu().tolist()

                for res, score in zip(batch_results, match_probabilities):
                    reranked_results.append({
                        "path": res["path"],
                        "bi_encoder_score": res["score"],
                        "score": float(score)
                    })

        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        return reranked_results[:k_final]