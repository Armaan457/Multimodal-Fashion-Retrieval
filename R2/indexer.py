import hnswlib
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from .model import FashionEmbedder
from .dataset import create_dataloader

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
        
        with torch.no_grad(), torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
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
        np.save(filepath + ".paths.npy", np.array(self.all_paths))

    def load(self, filepath: str):
        self.index = hnswlib.Index(space="cosine", dim=self.dimension)
        self.index.load_index(filepath)
        self.all_paths = np.load(filepath + ".paths.npy", allow_pickle=True).tolist()

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

    def query(self, query_text: str, k_initial: int = 30, k_final: int = 10) -> list:
        if self.index is None:
            raise ValueError("Index is not loaded or built.")
        
        initial_results = self._retrieve_initial(query_text, k=k_initial)
        reranked_results = []
        
        with torch.no_grad():
            for res in initial_results:
                try:
                    image = Image.open(res["path"]).convert("RGB")
                    
                    inputs = self.embedder.cross_processor(
                        image, query_text, return_tensors="pt"
                    ).to(self.embedder.device)  
                    
                    outputs = self.embedder.cross_model(**inputs)
                    
                    itm_scores = torch.nn.functional.softmax(outputs.itm_score, dim=1)
                    match_probability = itm_scores[:, 1].item() 
                    
                    reranked_results.append({
                        "path": res["path"],
                        "bi_encoder_score": res["score"],
                        "score": match_probability 
                    })
                except Exception as e:
                    print(f"Skipping unreadable image {res['path']}: {e}")
                    continue
                    
        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        return reranked_results[:k_final]