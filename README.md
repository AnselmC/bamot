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
### Installing custom g2opy
First, make sure you have cloned this repo recursively, i.e.
```bash
git submodule update --init --recursive
```

Then as per [g2opy's readme](https://github.com/uoip/g2opy#installation), build and install g2opy like this:
```bash
cd bamot/thirdparty/g2opy
mkdir build
cd build
cmake ..
make -j8 # or however many processors you have
cd ..
python setup.py install
```

If you're using a virtualenv such as `anaconda`, make sure you have the environment active before installation.


### Installing python dependencies

To manage dependencies, this project uses [poetry](https://github.com/python-poetry/poetry) - see installation instructions there.

Then, install the dependencies via (in the projects home directory):
```bash
poetry install
```

For convenience, a `requirements.txt` is also given so that dependencies can be installed via `pip`:
```bash
pip install -r requirements.txt
```

### The KITTI dataset
Currently, only the [KITTI tracking dataset](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) is supported (see linked page for download).
You need to download the left and right color images, the GPS/IMU data for ego poses, the camera calibration matrices, and (for the training data) the ground truth label data. 

Additionally, you'll need to create 2D object detections and store them in the KITTI `.txt` format using a 2D object tracker.
For convenience, the detections used in this work are given in the `detections` directory.

Ideally, you'll want to run `TrackR-CNN` or your own tracker on the entire dataset (all train and test scenes).

While the location of the dataset (and its sub-parts) is configurable, by default the following directory structure is expected:
```
bamot/data/KITTI
└── tracking
    ├── testing
    │   ├── ...
    └── training
        ├── calib --> contains camera calibration data
        │   ├── 0000.txt
        │   ├── ...
        │   └── 0019.txt
        ├── detections --> contains 2D detections from detector/tracker (e.g. TrackR-CNN)
        ├── stereo_detections --> contains pseudo stereo detections with updated ids (see next step)
        │   ├── image_02
        │   │   ├── 0000.txt
        │   │   ├── ...
        │   │   └── 0019.txt
        │   ├── image_03
        ├── image_02 --> contains left (rectified) image data
        │   ├── 0000
        │   │   ├── 000001.png
        │   │   ├── ...
        │   │   └── 000153.png
        │   ├── ...
        │   └── 0019
        ├── image_03 --> contains right image data
        ├── label_02 --> contains ground truth 3D object detections (only available for training data)
        │   ├── 0000.txt
        │   ├── ...
        │   └── 0019.txt
        └── oxts --> contains ego poses
            ├── 0000.txt
            ├── ...
            └── 0019.txt
```

### Creating pseudo-stereo detections
Once you have the dataset setup (including the 2D detections/tracklets), you can compute the pseudo 2D stereo detections as follows:

```bash
poetry run python create_2d_stereo_detections.py -s all --no-view
```

__Note:__ Use the `--help` flag to see all available options.

Again, for convenience pre-computed pseudo stereo detections are given in this repo under `stereo_detections`.
If you want to use them you'll need to move or symlink them to the appropriate place as noted in the previous section.

### Running BAMOT on the train or test set
Once you've created the pseudo-stereo detections you can run `BAMOT` on the train or test set (however, you won't be able to evaluate the latter qualitatively).

For example, to run scene 6 on the train set, execute:
```bash
CONFIG_FILE=configs/config-best.yaml poetry run python run_kitti_train_mot.py --scene 6 --classes car -mp -c
```

__Note:__ Use the `--help` flag to see all available options (e.g. disable viewer, run non-continuously, etc.).

__Note:__ To run a custom configuration, take a look at `bamot/config.py` and `configs/config-best.yaml`.






