import numpy as np
#https://github.com/deepmind/surface-distance/blob/master/surface_distance/metrics.py
def dice(gt, sim, thresholds):
    result = []
    gt_thresholded = (gt >= thresholds[0])
    sim_thresholded = (sim >= thresholds[1])

    total = gt_thresholded.sum() + sim_thresholded.sum()
    # assert total != 0

    intersect = (gt_thresholded & sim_thresholded).sum()
    if total != 0:
        print((2 * intersect) / total)
    else:
        print((threshold, np.nan))

dice(np.load("8.0-150.0-4.0-Data_0001.npz")['data'][:,:,:,0], np.load("12.0-30.0-3.5-Data_0001.npz")['data'][:,:,:,0], [0.7, 0.75])
