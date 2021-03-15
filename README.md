# BAMOT: Bundle Adjustment for Multiple-Object Tracking - a Stereo-based Approach

This is the code for my Master's Thesis at the [Dynamic Vision and Learning Group](https://dvl.in.tum.de) under the supervision of [Prof. Dr. Laura Leal-Taixe](https://dvl.in.tum.de/team/lealtaixe/)
and the advisory of [Dr. Aljosa Osep](https://aljosaosep.github.io) and Nikolaus Demmel.
My thesis is also part of the repository (in the `thesis` directory).

<p align="center"><img src="visualization_example.gif"/></p>

## Abstract
_Note that citations are adjusted from the abstract of the thesis to include links instead of papers_

This thesis addresses the problem of 3D multi-object tracking for RGB-based systems.
More specifically, we propose a method that performs sparse feature-based object-level Bundle Adjustment for accurate object track localization.
Using the 2D object detector and tracker [TrackR-CNN](https://github.com/VisualComputingInstitute/TrackR-CNN), we introduce a procedure for stereo object detections and improve TrackR-CNN's trained association ability.
We achieve superior association via a multi-stage association pipeline that combines appearance and 3D localization similarity.
Additionally, we leverage a priori knowledge of object shapes and dynamics for both association and keeping a sparse outlier-free point cloud representation of objects.

We evaluate our proposal on the [KITTI tracking dataset](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) via the traditional CLEAR <sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup> and the recently introduced HOTA <sup><a id="fnr.2" class="footref" href="#fn.2">2</a></sup> metrics.
However, as the official KITTI tracking benchmark only includes 2D MOT evaluation, and the extended 3D evaluation from [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT) only supports CLEAR via 3D IoU, we implement a customized
tracking ability assessment.
The evaluation introduces a normalized 3D GIoU <sup><a id="fnr.3" class="footref" href="#fn.3">3</a></sup> detection similarity score to the official [HOTA evaluation scripts](https://github.com/JonathonLuiten/TrackEval).
We compare our performance to the LiDAR-based AB3DMOT for which 3D tracking results are readily available and demonstrate promising results, especially w.r.t. association and rigid, moving objects.
Furthermore, we show the contribution of various features of our system on an overall performance increase of 17 % for cars and 27 % for pedestrians.


### Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> <http://jivp.eurasipjournals.com/content/2008/1/246309>

<sup><a id="fn.2" href="#fnr.2">2</a></sup> <http://link.springer.com/10.1007/s11263-020-01375-2>

<sup><a id="fn.3" href="#fnr.3">3</a></sup> <http://arxiv.org/abs/1902.09630>


## Running the code
### Building g2opy
### Installing python dependencies
### Running KITTI scenes

## Results


