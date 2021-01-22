import numpy as np
from mayavi import mlab

d = np.load("7.0-150.0-5.0-Data_0001.npz")['data'][:,:,:,0]
#d2 = np.load("11.0-50.0-3.5-Data_0001.npz")['data'][:,:,:,0]
#d3 = np.load("12.0-30.0-3.5-Data_0001.npz")['data'][:,:,:,0]
d4 = np.load("12.0-30.0-3.7-Data_0001.npz")['data'][:,:,:,0]

f = mlab.figure(); mlab.volume_slice(d, figure=f, slice_index=48);
#f = mlab.figure(); mlab.volume_slice(d2, figure=f);
#f = mlab.figure(); mlab.volume_slice(d3, figure=f);
f = mlab.figure(); mlab.volume_slice(d4, figure=f, slice_index=48);
f = mlab.figure(); mlab.volume_slice(0.5 * (d >= 0.8).astype(np.float32) + 0.5 * (d >= 0.13).astype(np.float32), figure=f, slice_index=48);
f = mlab.figure(); mlab.volume_slice(0.5 * (d4 >= 0.7).astype(np.float32) + 0.5 * (d4 >= 0.45).astype(np.float32), figure=f, slice_index=48);

f = mlab.figure(); mlab.volume_slice((d >= 0.8)*d, figure=f, slice_index=48);
f = mlab.figure(); mlab.volume_slice((d4 >= 0.7)*d4, figure=f, slice_index=48);
mlab.show()
