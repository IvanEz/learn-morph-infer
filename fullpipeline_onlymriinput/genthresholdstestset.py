import numpy as np
np.random.seed(1234242)

t1gd = 0.35 * np.random.rand(12000) + 0.5
flair = 0.45 * np.random.rand(12000) + 0.05
necrotic = 0.05 * np.random.rand(12000) + 0.95

np.savez_compressed("scanthresholds", t1gd=t1gd, flair=flair, necrotic=necrotic)
