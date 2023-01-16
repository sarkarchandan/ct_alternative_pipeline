"""
    vis.py
    ------
    Encapsulates visualization utilities
"""

from enum import Enum, auto
import numpy as np
import matplotlib.pyplot as plt
import cv2


class Filtering(Enum):
    """Defines filtering variants for frequency domain analysis"""
    HIGH_PASS = auto()
    LOW_PASS = auto()


def magnitude_spectrum_for(img: np.ndarray) -> None:
    """Visualizes magnitude spectrum generated from the frequency-domain 
    transformation of the provided image. Uses OpenCV discrete fourier 
    transform.

    Args:
        img: Image object as ndarray
    """
    freq_trn: np.ndarray = cv2.dft(
        src=np.float32(img), 
        flags=cv2.DFT_COMPLEX_OUTPUT
    )
    # Shifts the zero frequency components to the center
    freq_trn = np.fft.fftshift(freq_trn)
    mag_spc: np.ndarray = 20 * np.log(
        cv2.magnitude(
            x=freq_trn[:, :, 0], 
            y=freq_trn[:, :, 1]
        )
    )
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 15))
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Input Image')
    ax2.imshow(mag_spc, cmap='gray')
    ax2.set_title('DFT - Magnitude Spectrum')
    plt.show();

def fourier_analysis_for(img: np.ndarray, filtering: Filtering) -> None:
    """Implements a filtering strategy for the frequency domain transformation 
    of the provided image.

    Args:
        img: Image object as ndarray
        filtering: Type of filtering to apply
    """
    freq_trn: np.ndarray = cv2.dft(
        src=np.float32(img), 
        flags=cv2.DFT_COMPLEX_OUTPUT
    )
    # Shifts the zero frequency components to the center
    freq_trn = np.fft.fftshift(freq_trn)
    # Create appropriate mask depending on the filtering type
    rows, cols = img.shape
    center_x, center_y = rows // 2, cols // 2
    mask: np.ndarray
    mask_radius: int
    if filtering is Filtering.HIGH_PASS:
        mask = np.ones(shape=(rows, cols, 2), dtype=np.uint8)
        mask_radius = 30
    else:
        mask = np.zeros(shape=(rows, cols, 2), dtype=np.uint8)
        mask_radius = 50
    grid_x, grid_y = np.ogrid[:rows, :cols]
    mask_area: np.ndarray = (
        (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2) <= mask_radius ** 2
    if filtering is Filtering.HIGH_PASS:
        mask[mask_area] = 0
    else:
        mask[mask_area] = 1
    # Apply the filtering mask on shifted frequency domain transformation
    masked_fft :np.ndarray = freq_trn * mask
    masked_fft_mag: np.ndarray = 20 * np.log(
        cv2.magnitude(
            x=masked_fft[:, :, 0], 
            y=masked_fft[:, :, 1]
        )
    )
    # Inverse fourier transform to bring the image back from frequency domain 
    # to spatial domain
    masked_ifft: np.ndarray = np.fft.ifftshift(x=masked_fft)
    filtered_img: np.ndarray = cv2.idft(src=masked_ifft)
    filtered_img = cv2.magnitude(
        x=filtered_img[:, :, 0], y=filtered_img[:, :, 1]
    )
    # Plot the original image and high-pass filtered image side by side
    _, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original Image')
    ax2.imshow(masked_fft_mag, cmap='gray')
    if filtering is Filtering.HIGH_PASS:
        ax2.set_title('High-pass filter mask')
    else:
        ax2.set_title('Low-pass filter mask')
    ax3.imshow(filtered_img, cmap='gray')
    ax3.set_title('Filtered Image')
    plt.show();


if __name__ == "__main__":
    pass
