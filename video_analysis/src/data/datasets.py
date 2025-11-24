import numpy as np, cv2, torch
from torch.utils.data import Dataset
import albumentations as A

class AlbumentationsTransform:
    def __init__(self, augment=True, size=256):
        if augment:
            self.tf = A.Compose([
                A.RandomResizedCrop(size, size, scale=(0.85, 1.0), ratio=(0.9, 1.1), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.6),
                A.GaussNoise(p=0.2),
                A.MotionBlur(blur_limit=(3,5), p=0.15),
                A.JpegCompression(quality_lower=70, quality_upper=95, p=0.2),
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
            ])
        else:
            self.tf = A.Compose([
                A.SmallestMaxSize(max_size=size),
                A.CenterCrop(size, size),
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
            ])
    def __call__(self, image):
        return self.tf(image=image)['image']

class ImageBinaryDataset(Dataset):
    def __init__(self, paths, labels, augment=False, size=256):
        self.paths = paths
        self.labels = labels
        self.tf = AlbumentationsTransform(augment=augment, size=size)
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]; y = self.labels[idx]
        img = cv2.imread(p)
        if img is None:
            size = self.tf.tf.transforms[-2].max_size if hasattr(self.tf.tf.transforms[-2], "max_size") else 256
            img = np.zeros((size, size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tf(img)
        img = torch.from_numpy(img).permute(2,0,1).float()
        return img, torch.tensor(y, dtype=torch.float32)
