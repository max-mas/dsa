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

def analytic_signal(sig):
    a_s = hilbert(sig)
    rho = np.abs(a_s)
    phi = np.unwrap(np.angle(a_s)) # remove jumps in phase
    omega = np.gradient(phi)

    return rho, phi, omega

t_y = np.loadtxt("signal_emd.txt")
t = t_y[:, 0]
y = t_y[:, 1]

imf = emd.sift.sift(y)

fig, [sum_ax, imf_ax, resid_ax] = plt.subplots(nrows=3, ncols=1, figsize=(8, 16), sharex=True)

sum_ax.plot(t, y, label="Signal", ls="--", color="forestgreen")
sum_ax.plot(t, np.sum(imf, axis=1), label="IMF Sum", ls="-.", color="indianred")
sum_ax.grid()
sum_ax.legend(loc="upper right")
sum_ax.set_ylabel("Signal / Reconstruction (a.u.)")

for i in range(len(imf[0])):
    imf_ax.plot(t, imf[:, i], label=f"IMF {i}", alpha=0.7)
imf_ax.grid()
imf_ax.legend(loc="upper right")
imf_ax.set_ylabel("IMF (a.u.)")

resid_ax.plot(t, np.abs(y - np.sum(imf, axis=1)))
#resid_ax.set_yscale("log")
resid_ax.set_ylabel("Residual (a.u.)")
resid_ax.set_xlabel("Time (a.u.)")
resid_ax.set_xlim(t[0], t[-1])
resid_ax.grid()

fig.tight_layout()
fig.savefig("imf.pdf")
plt.close(fig)

rhos = []
phis = []
omegas = []
for i in range(len(imf[0])):
    rho, phi, omega = analytic_signal(imf[:, i])
    rhos.append(rho)
    phis.append(phi)
    omegas.append(omega)

fig, [sig_ax, rho_ax, phi_ax, omega_ax] = plt.subplots(nrows=4, ncols=1, figsize=(8, 16), sharex=True)

for i, (rho, phi, omega) in enumerate(zip(rhos, phis, omegas)):
    sig_ax.plot(t, imf[:, i], label=f"IMF {i}")
    rho_ax.plot(t, rho)
    phi_ax.plot(t, phi/(2*np.pi))
    omega_ax.plot(t, omega/(2*np.pi))

sig_ax.grid()
sig_ax.legend()
rho_ax.grid()
rho_ax.set_ylim(0, 7)
phi_ax.grid()
phi_ax.set_ylim(0, 70)
omega_ax.grid()
omega_ax.set_ylim(0, 0.04)

omega_ax.set_xlabel("Time $t$ (a.u.)")
sig_ax.set_ylabel("Signal $s(t)$ (a.u.)")
rho_ax.set_ylabel("Envelope $\\rho$ (a.u.)")
phi_ax.set_ylabel("Phase $\\varphi$ $(2\\pi)$")
omega_ax.set_ylabel("Ang. Freq. $\\omega$ ($2\\pi$/a.u.)")

fig.tight_layout()
fig.savefig("imf_analytic_sig.pdf")
plt.close(fig)