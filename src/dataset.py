import os

import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class ProlivDataset(Dataset):
    def __init__(self, img_dir, labels, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        if isinstance(labels, str):
            self.labels = pd.read_csv(labels, sep=" ", names=["filename", "label"])
        else:
            self.labels = self._generate_labels(labels)

    def _generate_labels(self, labels):
        filenames = os.listdir(self.img_dir)

        return pd.DataFrame({
            "filename": filenames,
            "label": labels
        })

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])

        try:
            image = read_image(img_path)
        except Exception as e:
            raise RuntimeError(f"Error reading the image file {img_path}: {e}")

        label = self.labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def show(self, indexes):
        fig, axes = plt.subplots(len(indexes), len(indexes[0]), figsize=(10, 8))
        axes = axes.reshape(-1)

        for ax, idx in zip(axes, [idx for sublist in indexes for idx in sublist]):
            img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
            image = read_image(img_path).permute(1, 2, 0)

            if self.transform:
                image = self._inverse_transform(image)

            label = self.labels.iloc[idx, 1]
            ax.imshow(image)
            ax.set_title(f"Label: {label}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def _inverse_transform(self, image):
        return image
