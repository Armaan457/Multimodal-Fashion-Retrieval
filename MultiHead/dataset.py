from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class ImageDataset(Dataset):
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.image_paths = sorted([
            p for p in self.dataset_dir.iterdir()
            if p.suffix.lower() in VALID_EXTENSIONS
        ])
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return image, str(path)