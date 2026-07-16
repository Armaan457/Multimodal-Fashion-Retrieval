import os
from pathlib import Path
import numpy as np
import torch
import open_clip
from transformers import BlipProcessor, BlipForImageTextRetrieval

os.environ["HF_HOME"] = str(Path(__file__).parent.parent / "models")

class FashionEmbedder:
    def __init__(self, 
                 bi_model_name: str = 'hf-hub:Marqo/marqo-fashionSigLIP',
                 cross_model_name: str = 'Salesforce/blip-itm-base-coco'):
        
        self.device = torch.device("mps" if torch.backends.mps.is_available() 
                                   else "cuda" if torch.cuda.is_available() 
                                   else "cpu")
        print(f"Using device: {self.device}")
        
        self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(bi_model_name)
        self.tokenizer = open_clip.get_tokenizer(bi_model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.cross_processor = BlipProcessor.from_pretrained(cross_model_name)
        self.cross_model = BlipForImageTextRetrieval.from_pretrained(cross_model_name)
        self.cross_model.to(self.device)
        self.cross_model.eval()

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        embedding = self.model.encode_text(tokens, normalize=True)
        return embedding.cpu().numpy().astype(np.float32)