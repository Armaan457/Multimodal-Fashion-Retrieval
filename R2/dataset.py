from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch

class ImageDataset(Dataset):
    def __init__(self, image_paths: list):
        self.image_paths = [Path(p) for p in image_paths]
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            return image, str(self.image_paths[idx])
        except Exception as e:
            raise RuntimeError(f"Error loading image {self.image_paths[idx]}: {e}")

def create_dataloader(image_paths: list, preprocess_fn, batch_size: int = 32, num_workers: int = 0) -> DataLoader:
    dataset = ImageDataset(image_paths)
    
    def collate_fn(batch):
        images, paths = zip(*batch)
        images = torch.stack([preprocess_fn(img) for img in images])
        return images, list(paths)

    is_mps = torch.backends.mps.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=not is_mps 
    )