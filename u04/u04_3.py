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

t = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # s
y = np.array([109.4, 187.5, 267.5, 331.9, 386.1, 428.4, 452.2, 498.1, 512.3, 513]) # m
n = len(t)

def f(t, m):
    return m[0] + m[1] * t - 0.5 * m[2] * t**2

def main():
    A = np.zeros((n, 3))
    A[:, 0] = t**0
    A[:, 1] = t**1
    A[:, 2] = -0.5 * t**2
    
    # QR solution would be better, but at least we don't calculate the inverse explicitly
    coeffs = np.linalg.solve(A.T @ A, A.T @ y)
    print("Coefficients: [m1, m2, m3] =", coeffs)
    y_t = lambda t : f(t, coeffs)
    
    t_cont = np.linspace(0, 11, 1000)
    y_cont = y_t(t_cont)

    # plot
    fig, ax = plt.subplots()
    ax.plot(t_cont, y_cont, ls="--", color="forestgreen", alpha=0.8, label="Fit")
    ax.plot(t, y, color="indianred", marker="*", ls="", label="Data")
    ax.legend()
    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel("Height $y$ (m)")
    ax.grid()
    ax.set_xlim(0, 11)
    ax.set_ylim(bottom=0)
    
    fig.tight_layout()
    fig.savefig("lstsq.pdf")
    plt.close(fig)

    # variance
    resid = y - y_t(t)
    resid_var = 1/(n - 3) * np.sum(resid**2)
    var = np.var(y)
    print("Ratio of residual variance:", np.round(resid_var / var, 4))




if __name__ == "__main__":
    main()