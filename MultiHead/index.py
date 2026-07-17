from pathlib import Path
import pickle
import hnswlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .model import FashionCLIPModel
from .dataset import ImageDataset

class VectorIndex:
    def __init__(self, dim: int = 512, space: str = "cosine"):
        self.dim = dim
        self.space = space
        self.index = None
        self.all_paths = []

    def build_from_dir(self, dataset_dir: Path, model_handler: FashionCLIPModel, batch_size: int = 32):
        dataset = ImageDataset(dataset_dir)
        
        def collate_fn(batch):
            images, paths = zip(*batch)
            inputs = model_handler.processor(images=list(images), return_tensors="pt")
            return inputs, paths

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        
        all_embeddings = []
        self.all_paths = []

        for inputs, paths in tqdm(loader, desc="Extracting visual features"):
            embeddings = model_handler.get_image_features(inputs)
            all_embeddings.append(embeddings.cpu())
            self.all_paths.extend(paths)

        image_embeddings = torch.cat(all_embeddings).numpy().astype(np.float32)
        
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(max_elements=len(image_embeddings), ef_construction=200, M=32)
        self.index.add_items(image_embeddings, np.arange(len(image_embeddings)))
        self.index.set_ef(100)
        
        return image_embeddings

    def save(self, filepath: str | Path, image_embeddings: np.ndarray):
        bin_path = Path(filepath)
        bin_path.parent.mkdir(parents=True, exist_ok=True)

        self.index.save_index(str(bin_path))

        meta_path = bin_path.with_suffix(".pkl")
        metadata = {
            "all_paths": self.all_paths,
            "image_embeddings": image_embeddings,
            "dim": self.dim,
            "space": self.space,
        }

        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)


    def load(self, filepath: str | Path):
        bin_path = Path(filepath)
        meta_path = bin_path.with_suffix(".pkl")

        if bin_path.exists() and meta_path.exists():
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)

            self.dim = metadata["dim"]
            self.space = metadata["space"]
            self.all_paths = metadata["all_paths"]
            image_embeddings = metadata["image_embeddings"]

            self.index = hnswlib.Index(space=self.space, dim=self.dim)
            self.index.load_index(str(bin_path))

            print(f"Loaded existing index and {len(self.all_paths)} paths.")
            return image_embeddings

        print("No local index found. Making one")
        return None