# external imports
import numpy as np
import scipy as scp
import scipy.optimize as opt
import scipy.stats as stats
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

def plot_timeseries(timeseries: np.ndarray, label=str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    ax.plot(timeseries, label=label)
    ax.set_xlabel("Time $t$ (a.u.)")
    ax.set_ylabel("Time series $S$ (a.u.)")
    ax.grid()
    ax.legend(fontsize=12)
    ax.set_xlim(0, len(timeseries))

    return fig, ax

def plot_timeseries_and_trend(timeseries: np.ndarray, timeseries_trend: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    ax.plot(timeseries, label="Raw time series")
    ax.plot(timeseries_trend, ls="--", label="Time series trend")
    ax.set_xlabel("Time $t$ (a.u.)")
    ax.set_ylabel("Time series $S$ (a.u.)")
    ax.grid()
    ax.legend(fontsize=12)
    ax.set_xlim(0, len(timeseries))

    return fig, ax

def plot_timeseries_hist(timeseries: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    ax.hist(timeseries, bins=100, density=True, zorder=5)
    ax.grid(zorder=1)
    ax.set_xlabel("Time series $S$ (a.u.)")
    ax.set_ylabel("Normalised histogram")

    return fig, ax

def centered_empirical_moment(empirical_distribution: np.ndarray, mean: float, order: int, xmin: float, xmax: float):
    x = np.linspace(xmin, xmax, len(empirical_distribution))
    dx = (xmax - xmin) / len(empirical_distribution)
    # binned integration
    m = dx * np.dot((x - mean)**order, empirical_distribution)

    return m

# entrypoint
def main():
    timeseries_path = "timeseries.txt"
    timeseries = np.loadtxt(timeseries_path)

    # a) plot raw time series
    fig, ax = plot_timeseries(timeseries, label="Time series")
    print("Ex. 2 a):\n\tSaving plot...")
    fig.savefig("timeseries_plot.pdf")
    plt.close(fig)

    # b) detrend time series
    x = np.arange(len(timeseries))
    linreg = stats.linregress(x, timeseries)
    # print regression parameters and statistics
    print("Ex. 2 b):\n\tPerforming linear regression...")
    print(f"\tLinreg params: slope = {np.round(linreg.slope, 8)} pm {np.round(linreg.stderr, 8)}, intercept = {np.round(linreg.intercept, 3)} pm {np.round(linreg.intercept_stderr, 3)}, r2 = {np.round(linreg.rvalue, 3)}")
    # init linreg callable
    trend_fn = lambda x: linreg.slope * x + linreg.intercept
    # trend array
    timeseries_trend = trend_fn(x)
    fig, ax = plot_timeseries_and_trend(timeseries, timeseries_trend)
    print("\tSaving plot of time series and trend...")
    fig.savefig("timeseries_trend_plot.pdf")
    plt.close(fig)
    # detrend
    timeseries_detrended = timeseries - timeseries_trend
    fig, ax = plot_timeseries(timeseries_detrended, label="Detrended time series")
    print("\tSaving plot of detrended time series...")
    fig.savefig("timeseries_detrended.pdf")
    plt.close(fig)

    # c) plot histograms
    print("Ex. 2 c)")
    fig, ax = plot_timeseries_hist(timeseries)
    print("\tSaving plot of raw time series histogram using 100 bins...")
    fig.savefig("timeseries_hist.pdf")
    plt.close(fig)
    fig, ax = plot_timeseries_hist(timeseries_detrended)
    print("\tSaving plot of detrended time series using 100 bins...")
    fig.savefig("timeseries_hist_detrended.pdf")
    plt.close(fig)

    # d) calculate moments
    print("Ex. 2 d)")
    N = 1000
    print(f"\tCalculating empirical distributions with {N} bins...")
    dist_raw, _ = np.histogram(timeseries, bins=N, density=True)
    dist_detrended, _ = np.histogram(timeseries_detrended, bins=N, density=True)
    raw_mean = np.mean(timeseries)
    detrended_mean = np.mean(timeseries_detrended)
    print(f"\tCalculating moments...")
    for order in range(5):
        m_raw = centered_empirical_moment(dist_raw, raw_mean, order, np.min(timeseries), np.max(timeseries))
        m_detrended = centered_empirical_moment(dist_detrended, detrended_mean, order, np.min(timeseries_detrended), np.max(timeseries_detrended))
        print(f"\t\tOrder {order}:")
        print(f"\t\t\tRaw series: moment = {np.round(m_raw, 6)}")
        print(f"\t\t\tDetrended series: moment = {np.round(m_detrended, 6)}")


if __name__ == "__main__":
    main()