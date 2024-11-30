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

def plot_wave(mag, delta, eps, phi):
    
    fig, [mag_ax, del_ax, eps_ax, phi_ax] = plt.subplots(nrows=4, ncols=1, figsize=(8, 16), sharex=True)

    t = np.linspace(0, 1, len(mag))
    mag_ax.plot(t, mag)
    del_ax.plot(t, delta/(2*np.pi))
    eps_ax.plot(t, eps)
    phi_ax.plot(t, phi/(2*np.pi))

    mag_ax.grid()
    del_ax.grid()
    eps_ax.grid()
    phi_ax.grid()

    phi_ax.set_xlabel("Time $t$ (a.u.)")
    mag_ax.set_ylabel("Wave Amplitude (a.u.)")
    del_ax.set_ylabel("Phase Difference $\\delta$ $(2\\pi)$")
    eps_ax.set_ylabel("Ellipticity $\\epsilon$")
    phi_ax.set_ylabel("Azimuth $\\varphi$ $(2\\pi)$")

    fig.tight_layout()
    return fig

def analytic_signal(sig):
    a_s = hilbert(sig)
    rho = np.abs(a_s)
    phi = np.unwrap(np.angle(a_s)) # remove jumps in phase
    omega = np.gradient(phi)

    return rho, phi, omega

def main():
    # load signal
    sig = np.loadtxt("signal.txt")
    sig_x = sig[:, 0]
    sig_y1 = sig[:, 1]
    sig_y2 = sig[:, 2]

    # a)
    rho, phi, omega = analytic_signal(sig_x)

    fig = plot_analytic_sig(sig_x, rho, phi, omega)
    fig.savefig("analytic_sig.pdf")
    plt.close(fig)

    # b)
    rho_y1, phi_y1, omega_y1 = analytic_signal(sig_y1)
    rho_y2, phi_y2, omega_y2 = analytic_signal(sig_y2)

    # from lecture notes:
    abs1 = np.sqrt(rho**2 + rho_y1**2)
    delta1 = phi - phi_y1 # removes omega*t component
    azimuth_1 = 0.5 * np.arctan2(2 * rho * rho_y1 * np.cos(delta1), (rho**2 - rho_y1**2)) # x1 / x2
    ellipticity_1 = np.tan(0.5 * np.arcsin(2 * rho * rho_y1 / (rho**2 + rho_y1**2) * np.sin(delta1)))

    fig = plot_wave(abs1, delta1, ellipticity_1, azimuth_1)
    fig.savefig("wave1.pdf")

    abs2 = np.sqrt(rho**2 + rho_y2**2)
    delta2 = phi - phi_y2 # removes omega*t component
    azimuth_2 = 0.5 * np.arctan2(2 * rho * rho_y2 * np.cos(delta2), (rho**2 - rho_y2**2)) # x1 / x2
    ellipticity_2 = np.tan(0.5 * np.arcsin(2 * rho * rho_y2 / (rho**2 + rho_y2**2) * np.sin(delta2)))

    fig = plot_wave(abs2, delta2, ellipticity_2, azimuth_2)
    fig.savefig("wave2.pdf")
    

if __name__ == "__main__":
    main()