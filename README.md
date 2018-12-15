# F-CNN Pose Machines (Quadruped Subjects)

This is a Python and TensorFlow implementation of Convolutional Pose Machines to infer joint locations from RGB images of quadruped subjects. The general pose-estimation problem can be described as below:

<img src="images/pose_estimation.PNG" width=550>

Fully-Convolutional Neural Networks are trained to regress belief maps over the image area which indicate the probability of a specific joint to be in a specific pixel. Therefore, for example, for 12 joints, there are 12 heatmaps.

The ground trtuth data was procedurally generated using 3D modelling software MAYA. The parameters that vary are:
* Quadruped Pose
  * Leg joints angles
  * Hip Orientation
  * Head Orientation
  * Mouth opening
* Viewing Angle

## Requisites

This project makes use of the following frameworks/libraries
* Python 3
* TensorFlow 1.4+
* OpenCV 2
* Matplotlib
* Numpy
* Pillow

## How it works

The way the code works is simple.
We have 


There are a couple different models which incorporate the following:
- Different feature extraction stages: Original, VGG16, SLIM
- 

<img src="images/results">
<img src="images/architecture.png">
![alt text](http://url/to/img.png)
