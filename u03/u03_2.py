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

def y(t):
    return 10.0 * np.sin(2*np.pi * 2 * t) \
        + 5.0 * np.sin(2*np.pi * 32 * t) \
        + 2.0 * np.sin(2*np.pi* 100 * t)

def G_xx(ft, dt):
    return 2 / (len(ft) * dt) * np.real(ft)**2 + np.imag(ft)**2

def plot_G(fs, Gs, rates):
    fig, ax = plt.subplots()

    for f, G, rate in zip(fs, Gs, rates):
        ax.plot(f, G, label=f"Sampling Rate {rate} Hz")

    ax.set_xlabel("Frequency $f$ (1/s)")
    ax.set_ylabel("$G_{xx}$")
    ax.grid()
    ax.legend()
    ax.set_ylim(bottom=0)    

    fig.tight_layout()
    return fig

def main():
    t_50  = np.linspace(0, 1, 50)
    t_100 = np.linspace(0, 1, 100)
    t_256 = np.linspace(0, 1, 256)
    t_512 = np.linspace(0, 1, 512)

    y_50 = y(t_50)
    y_100 = y(t_100)
    y_256 = y(t_256)
    y_512 = y(t_512)

    ft_50 = np.fft.fft(y_50)
    ft_100 = np.fft.fft(y_100)
    ft_256 = np.fft.fft(y_256)
    ft_512 = np.fft.fft(y_512)

    G_50 = G_xx(y_50, 2/100)
    G_100 = G_xx(y_100, 1/100)
    G_256 = G_xx(y_256, y_256[1] - y_256[0])
    G_512 = G_xx(y_512, y_512[1] - y_512[0])

    fig = plot_G(
        [#np.linspace(0, 1/(2*0.02), 50), 
         #np.linspace(0, 1/(2*0.01), 100), 
         np.linspace(0, 1/(2*(t_256[1] - t_256[0])), 256),
         np.linspace(0, 1/(2*(t_256[1] - t_256[0])), 512)], 
        [#G_50, G_100, 
            G_256,
            G_512], 
        [#50, 100, 
            256,
            512])
    fig.savefig("G.pdf")
    plt.close(fig)


    

if __name__ == "__main__":
    main()