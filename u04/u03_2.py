# external imports
import numpy as np
import scipy.signal as sig
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
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "figure.figsize": [8, 6]
})
sns.color_palette("colorblind")

t_y = np.loadtxt("signal_emd.txt")
t = t_y[:, 0]
y = t_y[:, 1]

imf = emd.sift.sift(y)
ax: plt.Axes = emd.plotting.plot_imfs(imf)
fig = ax.get_figure()
fig.tight_layout()
fig.savefig("test.pdf")