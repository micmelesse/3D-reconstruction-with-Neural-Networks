import numpy as np
import matplotlib.pyplot as plt
from lib.dataset import construct_path_lists

i = 4
l = construct_path_lists("aws", "loss.npy")[i]
print(l)
p = np.load(l)
print(p.shape)
plt.plot(p)
plt.show()
