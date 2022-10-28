import numpy as np
from mayavi import mlab

d = np.load("8.0-150.0-4.0-Data_0001.npz")['data'][:,:,:,0]
d2 = np.load("12.0-30.0-3.5-Data_0001.npz")['data'][:,:,:,0]

f = mlab.figure(); mlab.volume_slice(d, figure=f);
f = mlab.figure(); mlab.volume_slice(d2, figure=f);
f = mlab.figure(); mlab.volume_slice((d >= 0.7).astype(np.float32), figure=f);
f = mlab.figure(); mlab.volume_slice((d2 >= 0.75).astype(np.float32), figure=f);
mlab.show()
