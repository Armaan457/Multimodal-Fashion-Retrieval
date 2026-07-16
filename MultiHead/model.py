import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from tqdm.auto import tqdm

os.environ["HF_HOME"] = str(Path(__file__).parent.parent / "models")

class FashionCLIPModel:
    def __init__(self, model_name: str = "patrickjohncyh/fashion-clip"):
        self.device = self._get_device()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            print("Using device: CUDA")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Using device: MPS")
            return torch.device("mps")
        else:
            print("Using device: CPU")
            return torch.device("cpu")

    @torch.no_grad()
    def get_image_features(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.get_image_features(**inputs)
        return F.normalize(outputs.pooler_output, dim=-1) 

    @torch.no_grad()
    def encode_texts(self, texts: list, batch_size: int = 64):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.processor(
                text=batch, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model.get_text_features(**inputs)
            emb = F.normalize(outputs.pooler_output, dim=-1) 
            all_embeddings.append(emb.cpu())
            
        return torch.cat(all_embeddings).numpy().astype(np.float32)