"""
    recon.py
    --------
    Encapsulates utilities for reconstruction of abstract 2D object
"""

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

class Reconstructor:

    dataset: List[np.ndarray]
    rotation_profile: Tuple[int, int, int]
    
    def __init__(self, 
        dataset: List[np.ndarray], 
        rotation_profile: Tuple[int, int, int]) -> None:
        self.dataset = dataset
        self.rotation_profile = rotation_profile

    @property
    def _linear_space(self) -> np.ndarray:
        """Convenient function to compute linear space across the object. This 
        would be required to compute the differential over linear space"""
        assert(len(self.dataset) > 0)
        return np.linspace(start=-1, stop=1, num=self.dataset[0].shape[0])

    def create_sinogram(self) -> np.ndarray:
        """Computes line integrals across all rotated samples from the 
        abstract object dataset"""
        diff_space: np.ndarray = np.diff(self._linear_space)[0]
        return np.array(
            [rotation.sum(axis=0) * diff_space for rotation in self.dataset])

    def plot_projections(self, 
        num_samples: int = 5, 
        figsize: Tuple[int, int] = (15, 2)) -> None:
        angles: np.ndarray = np.arange(
            start=self.rotation_profile[0], 
            stop=self.rotation_profile[1], 
            step=self.rotation_profile[2])[:num_samples]
        sgs: np.ndarray = self.create_sinogram()
        _, axes = plt.subplots(nrows=1, ncols=len(angles), figsize=figsize)
        for idx in range(len(angles)):
            axes[idx].plot(self._linear_space, sgs[idx, :]);
            axes[idx].set_xlabel(f'Angle: {angles[idx]}')
            axes[idx].set_ylabel('$\ln({I_0}/I)$')


if __name__ == "__main__":
    pass
