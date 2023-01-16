"""
    gen.py
    -------
    Encapsulates utilities for generating abstract 2D objects dataset
"""

from __future__ import annotations

from typing import Generator, List, Dict, Any, Tuple
from dataclasses import dataclass
import yaml
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate

@dataclass
class DataGenerator:

    obj_dim: int
    angle_start: int
    angle_end: int
    rotation_interval: int
    padding: int

    @classmethod
    def from_config(cls, config: str = "config.yaml") -> DataGenerator:
        """Instantiates abstract dataset generator from config"""
        cfg: Dict[str, Any]
        with open(file=config, mode="r") as stream:
            cfg = yaml.safe_load(stream=stream)
        return cls(**cfg["data_gen"])

    @property
    def abstract_obj_image(self) -> np.ndarray:
        """Creates abstract object image"""
        img: np.ndarray = np.ones(shape=(self.obj_dim, self.obj_dim))
        diag_len: int = len(np.diag(img) // 2)
        pad_img: np.ndarray = np.pad(
            array=img, 
            pad_width=diag_len + self.padding
        )
        lsp: np.ndarray = np.linspace(start=-1, stop=1, num=pad_img.shape[0])
        xv, yv = np.meshgrid(lsp, lsp)
        pad_img[(xv - 0.1)**2 + (yv - 0.2)**2 < 0.01] = 2
        return pad_img

    def plot_samples(self, 
        num_samples: int = 5, 
        figsize: Tuple[int, int] = (15, 15)) -> None:
        """Plots a subsection of the abstract object dataset
        Args:
            num_samples: Number of samples to be plotted
            figsize: Figure size for the plot
        """
        samples: List[np.ndarray] = list(self())
        _, axes = plt.subplots(nrows=1, ncols=num_samples, figsize=figsize)
        for idx in range(num_samples):
            axes[idx].imshow(samples[idx]);
            axes[idx].axis('off');

    def to_array(self) -> np.ndarray:
        """Convenient method to yield the dataset as an ndarray"""
        return np.array(list(self()))

    def create_sinogram(self) -> np.ndarray:
        pass

    @property
    def dataset_shape(self) -> Tuple[int, int, int]:
        """Convenient method to yield the shape of the dataset"""
        padded_dim: int = (2 * self.obj_dim) + self.obj_dim + (2 * 10)
        length: int = np.arange(
            start=self.angle_start, 
            stop=self.angle_end, 
            step=self.rotation_interval).shape[0]
        return (length, padded_dim, padded_dim)

    def __call__(self) -> Generator[np.ndarray, None, None]:
        """Yields abstract dataset as a generator expression"""
        angles: np.ndarray = np.arange(
            start=self.angle_start, 
            stop=self.angle_end, 
            step=self.rotation_interval) * (np.pi / 180)
        samples: List[np.ndarray] = [
            rotate(image=self.abstract_obj_image, angle=angle * 180/np.pi) 
            for angle in angles
        ]
        for sample in samples:
            yield sample
        
        
if __name__ == "__main__":
    pass

    