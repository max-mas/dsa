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
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "figure.figsize": [8, 6]
})
sns.color_palette("colorblind")

def plot_funcs(timeseries: list[np.ndarray], labels: list[str], xlabel: str, ylabel: str, xs: None | list[np.ndarray] = None) -> plt.Figure:
    fig, ax = plt.subplots()
    i = 0
    for ts, label in zip(timeseries, labels):
        if xs is None:
            ax.plot(ts, label=label, alpha=0.8)
        else:
            ax.plot(xs[i], ts, label=label, alpha=0.8)
            i += 1
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend(fontsize=12)
    ax.set_xlim(
        0 if xs is None else xs[-1][0], 
        len(timeseries[-1]) if xs is None else xs[-1][-1]
    )
    fig.tight_layout()

    return fig

def G_xx(ft, dt):
    return 2 / (len(ft) * dt) * np.real(ft)**2 + np.imag(ft)**2 # eqn (170) in the lecture notes

def plot_G(fs, Gs):
    fig, ax = plt.subplots()

    for i, (f, G) in enumerate(zip(fs, Gs)):
        ax.plot(f, G, label=f"Time Series {i+1}", alpha=0.8)

    ax.set_xlabel("Frequency $f$ (a.u.)")
    ax.set_ylabel("$G_{xx}$")
    ax.grid()
    ax.legend()
    #ax.set_yscale("log")
    #ax.set_ylim(bottom=1)
    ax.set_xscale("log")
    ax.set_xlim(left=1e-4)    

    fig.tight_layout()
    return fig

def get_spectral_density(f):
    n = len(f)
    t = np.arange(0, n)
    ft = np.fft.fft(f)
    # fft[0:n//2] are the zero- and positive-frequency section of the FT
    return G_xx(ft[:n//2], 1/n) # indexing here only correct for even n

def main():
    ts1 = np.loadtxt("zeitreihe1.txt")
    ts2 = np.loadtxt("zeitreihe2.txt")
    n1 = len(ts1)
    n2 = len(ts2)

    fig = plot_funcs([ts1, ts2], ["Time Series 1", "Time Series 2"], "Time $t$ (a.u.)", "Time series $S$ (a.u.)")
    fig.savefig("ts.pdf")
    plt.close(fig)

    ac1 = sig.correlate(ts1, ts1)
    ac2 = sig.correlate(ts2, ts2)
    lags1 = sig.correlation_lags(n1, n1)
    lags2 = sig.correlation_lags(n2, n2)
    fig = plot_funcs([ac1, ac2], ["Time Series 1", "Time Series 2"], "Lag (a.u.)", "Autocorrelation Function", xs=[lags1, lags2])
    fig.savefig("ac.pdf")
    plt.close(fig)

    cc = sig.correlate(ts1, ts2)
    lags_c = sig.correlation_lags(n1, n2)
    fig = plot_funcs([cc], [""], "Lag (a.u.)", "Crosscorrelation Function", xs=[lags_c])
    fig.savefig("cc.pdf")
    plt.close(fig)

    G1 = get_spectral_density(ts1)
    G2 = get_spectral_density(ts2)
    fig = plot_G([np.linspace(0, 0.5, n1//2), np.linspace(0, 0.5, n2//2)], [G1, G2])
    fig.savefig("G_03.pdf")
    plt.close(fig)



if __name__ == "__main__":
    main()