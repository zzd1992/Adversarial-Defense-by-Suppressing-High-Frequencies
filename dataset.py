from torch.utils.data import Dataset
from PIL import Image
import os

class DatasetFile(Dataset):
    def __init__(self, root, file, transform=None, target_transform=None):
        self.samples = []
        self.root = root
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img, label = line.strip().split(',')
                self.samples.append((img, int(label)))

        self.transform = transform
        self.target_transform = target_transform

    def loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(os.path.join(self.root, path))
        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.samples)

