"""
    data.py
    -------
    Implements abstract data source
"""

from __future__ import annotations

from typing import Generator, List, Dict, Any
from dataclasses import dataclass
import yaml
import numpy as np
from skimage.transform import rotate


@dataclass
class DataGenerator:

    obj_dim: int
    angle_start: int
    angle_end: int
    rotation_interval: int
    padding: int

    @classmethod
    def generator_from(cls, config: str) -> DataGenerator:
        cfg: Dict[str, Any]
        with open(file=config, mode="r") as stream:
            cfg = yaml.safe_load(stream=stream)
        return cls(**cfg["data_gen"])

    @property
    def base_image(self) -> np.ndarray:
        """Computes base object image"""
        img: np.ndarray = np.ones(shape=(self.obj_dim, self.obj_dim))
        diag_len: int = len(np.diag(img) // 2)
        pad_img: np.ndarray = np.pad(
            array=img, 
            pad_width=diag_len + self.padding
        )
        radial_values: np.ndarray = np.linspace(
            start=-1, 
            stop=1, 
            num=pad_img.shape[0])
        xv, yv = np.meshgrid(radial_values, radial_values)
        pad_img[(xv - 0.1)**2 + (yv - 0.2)**2 < 0.01] = 2
        return pad_img

    def __call__(self) -> Generator[np.ndarray, None, None]:
        angles: np.ndarray = np.arange(
            start=self.angle_start, 
            stop=self.angle_end, 
            step=self.rotation_interval) * (np.pi / 180)
        samples: List[np.ndarray] = [
            rotate(image=self.base_image, angle=angle * 180/np.pi) 
            for angle in angles
        ]
        for sample in samples:
            yield sample
        
        


    