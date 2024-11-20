import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt('wuerfel.txt')
n_bins = round(10 / 0.01) # because the biggest and smallest possible are 12 and 2

fig, axs = plt.subplots(5, 1, tight_layout=False)

for i in range(5):
    axs[i].hist(data[:, i], bins=n_bins, range=(2,12))

fig.show()

for i in range(5):
    print(np.std(data[:, i]))