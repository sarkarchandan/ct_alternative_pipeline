# pylint: disable=unnecessary-semicolon
"""
    recon.py
    --------
    Encapsulates utilities for reconstruction of abstract 2D object
"""

from typing import List, Tuple, Callable
from enum import Enum, auto
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from scipy.fft import fft, ifft


class Reconstruction(Enum):
    """Defines variants for back projection"""
    REGULAR_BACK_PROJECTION = auto()
    FILTERED_BACK_PROJECTION = auto()


class Reconstructor:
    """Encapsulates utilities for reconstructing the abstract 2D object from a
    set of rotated samples. In this demo implementation we assume the samples
    themselves as the rotated attenuation profiles of the 2D abstract object."""

    dataset: List[np.ndarray]
    rotation_profile: Tuple[int, int, int]

    def __init__(self,
                 dataset: List[np.ndarray],
                 rotation_profile: Tuple[int, int, int]) -> None:
        """
        Args: 
            dataset: A list of rotated samples (attenuation profiles of the 
            abstract 2D object)
            rotation_profile: A tuple object describing the range and interval 
            of rotation
        """
        self.dataset = dataset
        self.rotation_profile = rotation_profile

    @property
    def linear_space(self) -> np.ndarray:
        """Linear space across the object. This would be required to compute 
        the differential over the same"""
        assert (len(self.dataset) > 0)
        return np.linspace(start=-1, stop=1, num=self.dataset[0].shape[0])

    @property
    def angles_rad(self) -> np.ndarray:
        """Angles of rotation (in radians)"""
        return np.arange(
            start=self.rotation_profile[0],
            stop=self.rotation_profile[1],
            step=self.rotation_profile[2]) * (np.pi / 180)

    def create_sinogram(self) -> np.ndarray:
        """Computes line integrals across all rotated samples from the rotated 
        samples"""
        diff_space: np.ndarray = np.diff(self.linear_space)[0]
        return np.array(
            [rotation.sum(axis=0) * diff_space for rotation in self.dataset])

    def show_projections(self,
                         num_samples: int = 5,
                         figsize: Tuple[int, int] = (15, 2)) -> None:
        """Displays given number of projections (line integrals) for the samples

        Args:
            num_samples: (Optional)Number of samples to display
            figsize: (Optional)Size of the PyPlot figure object
        """
        angles: np.ndarray = np.arange(
            start=self.rotation_profile[0],
            stop=self.rotation_profile[1],
            step=self.rotation_profile[2])[:num_samples]
        sgs: np.ndarray = self.create_sinogram()
        _, axes = plt.subplots(nrows=1, ncols=len(angles), figsize=figsize)
        for idx in range(len(angles)):
            axes[idx].plot(self.linear_space, sgs[idx, :]);
            axes[idx].set_xlabel(f"Angle: {angles[idx]}");
            axes[idx].set_ylabel("$\ln({I_0}/I)$");

    @staticmethod
    def _interpolate(
            x: float,
            y: float,
            lsp: np.ndarray,
            angles: np.ndarray,
            source: np.ndarray) -> np.ndarray:
        """Defines an interpolation function for sampling of values during 
        reconstruction of the abstract 2D image object. This function is 
        subjected to be vectorized for indexing.

        Uses RectBivariateSpline to interpolate values from a rectangular 
        grid.

        Args:
            (x, y): Coordinates from a rectangular mesh grid
            lsp: Linear space across the object
            angles: Angles of rotation
            source: Inverse 2D fourier space to sample the values from
        """
        diff_angles: np.ndarray = np.diff(angles)[0]
        interpolation: Callable[[float, float], float] = RectBivariateSpline(
            x=lsp,
            y=angles,
            z=source)
        return interpolation(
            x * np.cos(angles) + y * np.sin(angles), angles,
            grid=False).sum() * diff_angles

    def reconstruct(self, with_strategy: Reconstruction) -> np.ndarray:
        """Reconstructs the abstract object using the projections sinogram 
        with a given strategy

        Args:
            with_strategy: Reconstruction strategy
        """
        lsp: np.ndarray = self.linear_space
        angles: np.ndarray = self.angles_rad
        vec_interpolate: Callable = np.vectorize(
            self._interpolate, excluded=["lsp", "angles", "source"])
        sinogram: np.ndarray = self.create_sinogram()
        grid_x, grid_y = np.meshgrid(lsp, lsp)
        if with_strategy is Reconstruction.REGULAR_BACK_PROJECTION:
            # Interpolate to sample values with the size of the image using 
            # sinogram as source
            return vec_interpolate(
                x=grid_x,
                y=grid_y,
                lsp=lsp,
                angles=angles,
                source=sinogram.transpose())
        # Fourier transform of 1D sinogram for all rotations
        p_fft: np.ndarray = fft(sinogram.transpose(), axis=0)
        # Define filter entity in frequency domain 
        abs_filter: np.ndarray = np.abs(
            np.fft.fftfreq(p_fft.shape[0], d=np.diff(lsp)[0])
        )
        integrand: np.ndarray = abs_filter.transpose() * p_fft.transpose()
        # Inverse fourier transform of filtered 1D projection
        p_ifft: np.ndarray = np.real(ifft(integrand.transpose(), axis=0))
        # Interpolate to sample values with the size of the image using 
        # inverse fourier transform of filtered 1D projection as source
        return vec_interpolate(
            x=grid_x,
            y=grid_y,
            lsp=lsp,
            angles=angles,
            source=p_ifft)


if __name__ == "__main__":
    pass
