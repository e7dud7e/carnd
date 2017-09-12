# Self-Driving Car Engineer Nanodegree
# Overview of projects

This is an overview of self driving car term 1 projects that I completed for the Udacity Self-Driving Car Engineer Nanodegree.

Term 1 covers computer vision using traditional methods as well as with neural networks.

[//]: # (Image References)

[overlay_lanes]: ./LaneDetection/pipeline_images/whiteCarLaneSwitch_overlay_lanes.jpg

[img_traffic]: ./TrafficSigns/TrafficSigns_screenshot.png

[img_clone]: ./behavioral-cloning/img_cloning.png

[img_lane_adv]: ./LaneAdvanced/output_images/draw_normal.jpg "draw normal"

[image_heat14]: ./VehicleDetection/examples/detect14.jpg

### Install
The projects use either OpenCV or TensorFlow.  See the individual README of each project for setup details.

## Lane Detection
See the project writeup ./LaneDetection/LaneDetectionWriteup.md

See also the videos the lane detection here: LaneDetection/test_videos_output/

![overlay full lanes][overlay_lanes]

Use OpenCV to detect lane lines in video from a mounted camera of a driving car.

## Traffic Sign Classification with CNNs

![img_traffic]

- See the project writeup ./TrafficSigns/project_writeup.md.
- Built convolutional neural networks to classify traffic signs, and reach 97.55 test accuracy.
- Used image augmentation and batch normalization to improve the model.

## Behavioral Cloning with CNNs

![img_clone]

- See the project writeup in writeup_report.md
- See the video.mp4 for a test drive based on the trained network (in a simulator).
- Used Keras to build a convolutional neural network that predicts steering angles using video images.


## Advanced Lane Detection

![img_lane_adv]

- See ./LaneAdvanced/advanced_lane_writeup.md
- See the video ./LaneAdvanced/output_video/project_video_out.mp4

- Used OpenCV to undistort camera images, used perspective warp to generate an aerial view of the road.
- Used gradients and color spaces to detect lane lines and also detect the entire lane within lane lanes.


## Vehicle Detection

![alt text][image_heat14]

- See the project writeup in ./VechicleDetection/project_writeup.md
- See the videos in ./VehicleDetection/output_video
- Used histogram of gradients to extract features from images.
- Used a rolling window and SVM to classify areas of the image as a car or not.
- Tracked detected images over time in video.