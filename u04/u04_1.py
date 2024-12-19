# external imports
import numpy as np
import scipy.signal as sig
from scipy.signal import hilbert
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import emd
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
    "font.family": "serif",
    "figure.dpi": 300,
    "figure.figsize": [8, 6]
})
sns.color_palette("colorblind")

def G_xx(ft, dt):
    return 2 / (len(ft) * dt) * np.real(ft)**2 + np.imag(ft)**2 # eqn (170) in the lecture notes

def get_spectral_density(s):
    n = len(s)
    if n % 2 != 0:
        raise ValueError("Signal must have even length.")
    ft = np.fft.fft(s)    
    # fft[0:n//2] are the zero- and positive-frequency section of the FT
    return G_xx(ft[:n//2], 1/n) # indexing here only correct for even n

signal = np.loadtxt("windowing.txt")
dt = 1 / 1e6 # (s)
n = len(signal)

spectrum = get_spectral_density(signal)

signal_windowed = signal * np.hanning(n)
spectrum_windowed = get_spectral_density(signal_windowed)

t = np.linspace(0, n*dt, n)
f = np.linspace(0, n/2 - 1, n//2) / (n) # MHz

nonzero_freqs = f[np.nonzero(spectrum)]
nonzero_freqs_windowed = f[np.nonzero(spectrum_windowed)]
print("Non-zero frequencies:", nonzero_freqs, "MHz")
print("Non-zero frequencies (Hanning):", nonzero_freqs_windowed, "MHz")

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
ax1.plot(1e3*t, signal, label="Raw")
ax1.plot(1e3*t, signal_windowed, label="Hanning")
ax1.set_xlabel("Time $t$ (ms)")
ax1.set_ylabel("Signal $s(t)$ (a.u.)")
ax1.grid()
ax1.set_ylim(-2, 2)
ax1.set_xlim(0, 1000*n*dt)
ax1.legend()

ax2.plot(f, spectrum)
ax2.plot(f, spectrum_windowed)
ax2.set_xlabel("Frequency $f$ (MHz)")
ax2.set_ylabel("Spectral Density $G_{xx}$ (a.u.)")
ax2.grid()
ax2.set_xlim(0, 0.05)
ax2.set_yscale("log")

fig.tight_layout()
fig.savefig("signal.pdf")
plt.close(fig)

