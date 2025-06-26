# /media/salim/inner_disk/AI_workspace/DDPM_DDIM/datasets/hes.py
import os, glob
from PIL import Image
from torch.utils.data import Dataset

class HESDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        # on part de : root_dir/{train,test,val}/*_patches/*.*
        pattern = os.path.join(root_dir, split, '*_patches', '*.*')
        self.img_paths = glob.glob(pattern)
        if len(self.img_paths) == 0:
            raise RuntimeError(f"Aucune image trouvée avec le pattern {pattern}")
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0
