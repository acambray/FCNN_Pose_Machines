# F-CNN Pose Machines (Quadruped Subjects)

This is a Python and TensorFlow implementation of Convolutional Pose Machines to infer joint locations from RGB images of quadruped subjects. The general pose-estimation problem can be described as below:
<p align="center">
 <img src="images/pose_estimation.PNG" width=550><br>
 <img src="images/walk_cycle.gif" width=200>
 <img src="images/walk-cycle side.gif" width=200>
 <img src="images/angular_range.gif" width=200>
 
</p>

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



## F-CNN Convolutional Pose Machine Architecture
There are a couple different models under /models which incorporate the following:
* Different feature extraction stages: Original, VGG16, SLIM
* Different stage sub-module structures (different models

Models can be modified or one can create a new model. Base it off the existing models.
The image below gives an idea of how the staged approach in this project works. Each stage can contain its own feature extractor and heatmap regressor. Each stage works on the input image and the feature maps from previous stages. Due to local supervision, each stages refines the probability heatmap estimate for each joint.
<p align="center">
 <img src="images/General_Framework.PNG">
</p>


## Dataset
<p align="center">
 <img src="images/viewpoint_variation.PNG" width=450><br>
 <img src="images/viewpoint_variation2.PNG" width=650>
 <img src="images/dataset_input+groundtruth.PNG">
</p>

 
