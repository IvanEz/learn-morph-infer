import numpy as np
np.random.seed(4242123)

t1gd = 0.35 * np.random.rand(88000) + 0.5
flair = 0.45 * np.random.rand(88000) + 0.05

np.savez_compressed("scanthresholds", t1gd=t1gd, flair=flair)