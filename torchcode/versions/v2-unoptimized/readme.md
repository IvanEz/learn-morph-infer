Updated v2 code but uses same principles as v1 code: lr_update is every epoch and not every batch (i.e. "schedstepepoch"), no avgpool, no dynamic thresholding.
These three are the optimizations that v2 code benefits compared to this version.

Training with this versions are marked with "schedstepepoch" or "v2-unoptimized"

Use this to train on two thresholds but at the same time get directly comparable results to v1 code.