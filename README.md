# [Video](https://ohxu.github.io/MPR_GAN_web/)
## Environments
PyTorch 1.7 <br>
Python 3.6 <br>
## Data downloads
Download [KITTI datasets](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) into corresponding 'dataset/KITTI/...' folders, including velodyne, image, calib.
## Transformation
Before training, there are several steps required to transform 3D point cloud into 2D images
```
cd ./data
python transform.py
```
## Training
Some parameters can be modified in the options.py
```
python train.py
```
