
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image_vehicle]: ./examples/vehicle_train.png
[image_nonvehicle]: ./examples/nonvehicle_train.png
[image_test7]: ./examples/test7.jpg
[image_find7]: ./examples/find_cars7.jpg
[image_detect7]: ./examples/detect7.jpg
[image_heat7]: ./examples/heat7.png
[image_test14]: ./examples/test14.jpg
[image_find14]: ./examples/find_cars14.jpg
[image_detect14]: ./examples/detect14.jpg
[image_heat14]: ./examples/heat14.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for histogram of gradients is in the function get_hog_features. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt][image_vehicle]
![alt][image_nonvehicle]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried using HLS and YCrCb color spaces, and in an effort to reduce the feature set for faster training, I tried larger pixels per cell and cells per block, but ended up with lower test accuracy after training a linear support vector machine.  I settled on spatial features of 32 x 32, histogram of color channels with 64 bins, and for the histogram of gradients, I chose 9 orientations, 8 pixels per cell, and 2 cells per block.

I used a decision tree to extract the importance of features.  Even though the histogram of color channels (with 64 bins) had the highest feature importance, I kept all features because removing HOG features reduced the test accuracy of the trained support vector machine classifier.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using C = 0.01, as this had a test accuracy of 0.9893 when using the Ycrcb color space (slightly better compared to HLS).  This is in the code under the header "Save Chosen SVC".

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I searched for images with a smaller scale closer to the horizon (farther away), and searched with larger windows for regions closer to the camera.  I colored the search regions in red, green and blue.  The detected vehicle bounding boxes are also colored based on which region and scale they were using.  The code is in the function `find_cars`.

Here is an example of the original image, the image with the detected bounding boxes.

![alt text][image_test7]
![alt text][image_find7]


Here is another example
![alt text][image_test14]
![alt text][image_find14]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I used sample images from the video to test my pipeline, and adjusted parameters to improve detection.  

---

### Video Implementation

#### 1. Provide a link to your final video output.
Here's a [link to my video result, project_video_out.mp4](./output_video/project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used a heatmap to aggregate the bounding boxes, and then thresholded that map so that isolated bounding boxes are not considered detections, whereas close clusters of bounding boxes are identified as cars.  I used `scipy.ndimage.measurements.label()` to identify individual clusters in the heatmap. I  assumed each label was a detected vehicle. I created a bounding box around each cluster to draw onto the image.


### Here are some frames and their corresponding heatmaps:

Here are the found boxes and their corresponding heat map
![alt text][image_find7]
![alt text][image_heat7]

Here is another example of boxes with the thresholded heat map.
Notice that the box on the far left is thresholded and does not appear in the heat map.
![alt text][image_find14]
![alt text][image_heat14]


### Here are the resulting vehicle detection boxes:

![alt text][image_detect7]

![alt text][image_detect14]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I tracked vehicles over time by using a Vehicle class.  I created vehicles for newly detected windows, and assigned windows to existing vehicles when the window and vehicle position were close.  When a vehicle hasn't been assigned a newly detected window after a set number of frames, I mark the vehicle as 'gone', and remove it from the list of tracked vehicles.

In order to match a list of windows to a list of vehicles, I calculated the linear distance in pixels between the center of the window and the center of the vehicle.  I first assign the two closest window and vehicle pair, then continue with the remaining pairs until they're either all assigned, or they're considered too far away to be paired together.  Also, there can be more vehicles than windows, or more windows than vehicles. I remove vehicles that haven't been matched to a window after a set number of frames.  For windows that aren't assigned to a vehicle, I add a new vehicle object and assign it to that window.

I also set limits for how far a window can be from an existing vehicle.  If the window is too far away, it can't be assigned to that vehicle.

My pipeline misses the white car, and detects the black cars more readily.  It also merges two cars that are closer to one another. I'll need to either try other color spaces to detect white and black cars equally, or consider removing the histogram of color channels, so that the classifier relies more on the spatial and histogram of gradient features instead.