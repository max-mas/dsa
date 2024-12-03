# external imports
import numpy as np
import scipy as scp
from scipy.signal import hilbert
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# STL imports

# mpl defaults
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{parskip}\usepackage{braket}",
    "axes.labelsize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.titlesize": 24,
    "figure.titlesize": 24,
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "figure.figsize": [8, 6]
})
sns.color_palette("colorblind")

def y(t): # t in s
    return 10.0 * np.sin(2*np.pi *   2 * t) \
         +  5.0 * np.sin(2*np.pi *  32 * t) \
         +  2.0 * np.sin(2*np.pi * 128 * t)

def G_xx(ft, dt):
    return 2 / (len(ft) * dt) * np.real(ft)**2 + np.imag(ft)**2 # eqn (170) in the lecture notes

def plot_G(fs, Gs, rates):
    fig, ax = plt.subplots()

    for f, G, rate in zip(fs, Gs, rates):
        ax.plot(f, G, label=f"Sampling Rate {rate} Hz")
    ax.axvline(2, color="indianred", ls="--", alpha=0.7, label="Expected Peak Locations")
    ax.axvline(32, color="indianred", ls="--", alpha=0.7)
    ax.axvline(128, color="indianred", ls="--", alpha=0.7)

    ax.set_xlabel("Frequency $f$ (1/s)")
    ax.set_ylabel("$G_{xx}$")
    ax.grid()
    ax.legend()
    ax.set_yscale("log")
    ax.set_ylim(bottom=1)
    ax.set_xscale("log")
    ax.set_xlim(left=1)    

    fig.tight_layout()
    return fig

def get_spectral_density(n):
    t = np.linspace(0, 1, n)
    y_t = y(t)
    ft = np.fft.fft(y_t)
    # fft[0:n//2] are the zero- and positive-frequency section of the FT
    return G_xx(ft[:n//2], 1/n) # indexing here only correct for even n

def main():
    N = [50, 100, 200, 256] # all even. assumed above
    Gs = []
    fs = []
    for n in N:
        Gs.append(get_spectral_density(n))
        fs.append(np.linspace(0, n/2, n//2))    

    fig = plot_G(fs, Gs, N)
    fig.savefig("G.pdf")
    plt.close(fig)

if __name__ == "__main__":
    main()