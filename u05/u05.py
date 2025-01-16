# external imports
import numpy as np
import scipy.signal as sig
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
    "font.family": "serif",
    "figure.dpi": 300,
    "figure.figsize": [8, 6]
})
sns.color_palette("colorblind")

def main():
    message = np.loadtxt("message.txt")
    N = len(message)

    win = sig.windows.hamming(160, sym=True)
    SFT = sig.ShortTimeFFT(win=win, hop=50, fs=1/8000, fft_mode="onesided2X", scale_to="psd")
    stft = SFT.stft(message)

    fig, ax = plt.subplots()
    ax.imshow(np.abs(stft))
    t_lo, t_hi = SFT.extent(N)[:2]
    ax.set( xlabel=f"Time $t$ in seconds ({SFT.p_num(N)} slices, " + rf"$\Delta t = {SFT.delta_t:g}\,$s)", 
            ylabel=f"Freq. $f$ in Hz ({SFT.f_pts} bins, " + rf"$\Delta f = {SFT.delta_f:g}\,$Hz)", 
            xlim=(t_lo, t_hi))

    fig.savefig("test.pdf")

if __name__ == "__main__":
    main()