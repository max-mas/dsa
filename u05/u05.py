# external imports
import numpy as np
import scipy.signal as sig
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure, binary_erosion
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# STL imports
import sys

# mpl defaults
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{parskip}\usepackage{braket}",
    "axes.labelsize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.titlesize": 24,
    "figure.titlesize": 24,
    "font.family": "serif",
    "figure.dpi": 300,
    "figure.figsize": [8, 6]
})
sns.color_palette("colorblind")

def set_bit(value, n):
    return value | (1 << n)

def main():
    message = np.loadtxt("message.txt")
    N = len(message)

    data_fs = [697, 770, 852, 941, 1209, 1336, 1477, 1633]

    win = sig.windows.hamming(400, sym=True)
    SFT = sig.ShortTimeFFT(win=win, hop=400, fs=8000, fft_mode="onesided2X", scale_to="psd") # 400 = 50 ms
    stft = SFT.stft(message)
    psd = np.abs(stft)

    fig, ax = plt.subplots(figsize=(10,7))
    im = ax.imshow(psd, aspect="auto", origin="lower", extent=SFT.extent(N), cmap="Greens")
    ax.hlines(data_fs, xmin=0, xmax=N/8000,  colors="indianred", linestyles="dashed", label="Bit Frequencies")
    ax.set_ylabel("Frequency $f$ (Hz)")
    ax.set_xlabel("Time $t$ (s)")
    fig.colorbar(im, label="Spectral Density (a.u.)")
    ax.set_ylim(500, 2000)
    ax.legend()

    fig.savefig("test.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,7))
    mask = psd>0.035
    im = ax.imshow(mask, aspect="auto", origin="lower", extent=SFT.extent(N), cmap="Greens")
    ax.hlines(data_fs, xmin=0, xmax=N/8000,  colors="indianred", linestyles="dashed", label="Bit Frequencies")
    ax.set_ylabel("Frequency $f$ (Hz)")
    ax.set_xlabel("Time $t$ (s)")
    fig.colorbar(im, label="Spectral Density (a.u.)")
    ax.set_ylim(500, 2000)
    ax.legend()


    fig.savefig("test2.pdf")

    ticks = psd.shape[1]
    indices = [int(data_f/4020 * psd.shape[0]) for data_f in data_fs]
    print(indices)

    message = ""
    for i in range(0, ticks-1, 2):
        char = 0
        first_half = mask[indices, i]
        second_half = mask[indices, i+1]
        for loc in range(0, 4):
            val = first_half[-(4-loc)]
            char = set_bit(char, loc) if val else char
            val = second_half[-(4-loc)]
            char = set_bit(char, loc+4) if val else char
        print(bin(char))
        message += chr(char)

    print(message)
        

    

if __name__ == "__main__":
    main()