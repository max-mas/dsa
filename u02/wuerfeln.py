import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def z_alpha(alpha):
    a = sp.special.erfinv(alpha) * np.sqrt(2)
    return a

def confidence_interval(alpha, data, i):
    return(7 + z_alpha(alpha) * np.std(data) / np.sqrt(n_mean[i]) * np.array([-1, 1]))

data = np.loadtxt('wuerfel.txt')
n_bins = round(10 / 0.01)  # because the biggest and smallest possible are 12 and 2
n_mean = np.array([1, 3, 5, 10, 30])

fig, axs = plt.subplots(5, 1, tight_layout=False, dpi=300)

for i in range(5):
    axs[i].hist(data[:, i], bins=n_bins, range=(2,12))


fig.tight_layout()
fig.show()
fig.savefig('wuerfeln.png')

# schreibe in ein Dokument: Mittelwert, Standardabweichung, Interval 99 %, Interval 95%
with open('results.txt', 'w') as f:
    for i in range(5):
        f.write(str(np.mean(data[:, i])) + '\t' + str(np.std(data[:, i])) + '\t' + \
                str(confidence_interval(0.95, data[:, i], i)) + '\t' + str(confidence_interval(0.99, data[:, i], i)) + '\n')