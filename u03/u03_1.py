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

def plot_analytic_sig(sig, rho, phi, omega):
    
    fig, [sig_ax, rho_ax, phi_ax, omega_ax] = plt.subplots(nrows=4, ncols=1, figsize=(8, 16), sharex=True)

    t = np.linspace(0, 1, len(sig))
    sig_ax.plot(t, sig)
    rho_ax.plot(t, rho)
    phi_ax.plot(t, phi/(2*np.pi))
    omega_ax.plot(t, omega/(2*np.pi))

    sig_ax.grid()
    rho_ax.grid()
    phi_ax.grid()
    omega_ax.grid()

    omega_ax.set_xlabel("Time $t$ (a.u.)")
    sig_ax.set_ylabel("Signal $s(t)$ (a.u.)")
    rho_ax.set_ylabel("Envelope $\\rho$ (a.u.)")
    phi_ax.set_ylabel("Phase $\\varphi$ $(2\\pi)$")
    omega_ax.set_ylabel("Ang. Freq. $\\omega$ ($2\\pi$/a.u.)")

    fig.tight_layout()
    return fig

def main():
    # load signal
    sig = np.loadtxt("signal.txt")
    sig_x = sig[:, 0]
    sig_y1 = sig[:, 1]
    sig_y2 = sig[:, 2]

    # a)
    analytic_sig = hilbert(sig_x)
    rho = np.abs(analytic_sig)
    phi = np.unwrap(np.angle(analytic_sig)) # remove jumps in phase
    omega = np.gradient(phi)

    fig = plot_analytic_sig(sig_x, rho, phi, omega)
    fig.savefig("analytic_sig.pdf")
    plt.close(fig)
    

if __name__ == "__main__":
    main()