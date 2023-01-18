# pylint: disable=unnecessary-semicolon
"""
    vis.py
    ------
    Encapsulates data visualization utilities
"""

import base64
from io import BytesIO
from typing import List, Tuple
from enum import Enum, auto
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
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
    ax1.imshow(img, cmap="gray");
    ax1.set_title("Input Image");
    ax2.imshow(mag_spc, cmap="gray");
    ax2.set_title("DFT - Magnitude Spectrum");
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
    mask_area: np.ndarray = (grid_x - center_x) ** 2 + (grid_y - center_y) \
                            ** 2 <= mask_radius ** 2
    if filtering is Filtering.HIGH_PASS:
        mask[mask_area] = 0
    else:
        mask[mask_area] = 1
    # Apply the filtering mask on shifted frequency domain transformation
    masked_fft: np.ndarray = freq_trn * mask
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
    ax1.imshow(img, cmap="gray")
    ax1.set_title("Original Image")
    ax2.imshow(masked_fft_mag, cmap="gray")
    if filtering is Filtering.HIGH_PASS:
        ax2.set_title("High-pass filter mask")
    else:
        ax2.set_title("Low-pass filter mask")
    ax3.imshow(filtered_img, cmap="gray")
    ax3.set_title("Filtered Image")
    plt.show();


def imshow(source: np.ndarray, **kwargs) -> None:
    """Convenient function for general visualization of images or sinograms

    Args:
        source: Source to visualize
        kwargs: Optional keyword arguments
            - title: Plot title
            - x_label: Label for x axis
            - y_label: Label for y axis
            - pcolor: Use of pseudo color plot 
            - lsp: Linear space of the scanner
            - angles: Angles of rotation
    """
    title: str = kwargs.get("title", None)
    x_label: str = kwargs.get("x_label", None)
    y_label: str = kwargs.get("y_label", None)
    pcolor: str = kwargs.get("pcolor", None)
    lsp: np.ndarray = kwargs.get("lsp", None)
    angles: np.ndarray = kwargs.get("angles", None)
    if pcolor is not None:
        assert (angles is not None and lsp is not None)
        plt.pcolor(angles, lsp, source);
    else:
        plt.imshow(source);
    if title is not None:
        plt.title(title);
    if x_label is not None:
        plt.xlabel(x_label, fontsize=20);
    if y_label is not None:
        plt.ylabel(y_label, fontsize=20);


def serialize_multiple(images: List[np.ndarray], **kwargs) -> bytes:
    """Serializes a list of images as base64 encoded byte array to be 
    visualized on the browser

    Args:
        images: List of images
        kwargs: Keyword arguments

            dpi: Estimated dpi of the screen, default 192
            figsize: a tuple denoting figure size, default (600, 400)
            rows: Number of images in row direction, default 1
            cols: Number of images in column direction, default 1
    """
    dpi: int = kwargs.get("dpi", 192)
    figsize: Tuple[int, int] = kwargs.get("figsize", (600, 400))
    rows: int = kwargs.get("rows", 1)
    cols: int = kwargs.get("cols", 1)
    fig: Figure = Figure(figsize=(figsize[0] / dpi, figsize[1] / dpi))
    axs: Axes = fig.subplots(nrows=rows, ncols=cols)
    for idx in range(len(images)):
        axs[idx].imshow(images[idx]);
        axs[idx].axis("off");
    buf: BytesIO = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    return base64.b64encode(buf.getbuffer()).decode("ascii")


def serialize(source: np.ndarray, **kwargs) -> bytes:
    """Serializes a single ndarray either as colormap or a single image

    Args:
        source: NDArray to be plotted
        kwargs: keyword arguments
            
            dpi: Estimated dpi of the screen, default 192
            pcolor: Whether colormap should be used instead of imshow
            lsp: Linear space spanned on the surface of the scanner
            angles: Angles of rotation
            x_label: Label for x-axis
            y_label: Label for y-axis
            title: Tile for the plot
    """
    dpi: int = kwargs.get("dpi", 192)
    # figsize: Tuple[int, int] = kwargs.get("figsize", (600, 400))
    pcolor: bool = kwargs.get("pcolor", False)
    lsp: np.ndarray = kwargs.get("lsp", None)
    angles: np.ndarray = kwargs.get("angles", None)
    x_label: str = kwargs.get("x_label", None)
    y_label: str = kwargs.get("y_label", None)
    title: str = kwargs.get("title", None)
    fig: Figure = Figure()
    axs: Axes = fig.subplots()
    if pcolor:
        axs.pcolor(angles, lsp, source);
    else:
        axs.imshow(source)
        axs.axis("off");
    if x_label is not None:
        axs.set_xlabel(xlabel=x_label)
    if y_label is not None:
        axs.set_ylabel(ylabel=y_label)
    if title is not None:
        axs.set_title(title)
    buf: BytesIO = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    return base64.b64encode(buf.getbuffer()).decode("ascii")


if __name__ == "__main__":
    pass
