## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[img_calibrate]: ./output_images/undistort_chess.png "calibration"
[img_undistort]: ./output_images/undistort_lane.png "Undistorted"
[img_binary_xy]: ./output_images/gradx_grady.png "binarize x y"
[img_binary_md]: ./output_images/grad_mag_dir.png "binarize mag dir"
[img_binary_rs]: ./output_images/red_saturation.png "binarize red saturation"
[img_binary_co]: ./output_images/binary_comb.png "binarize saturation or gradx"
[img_perspective]: ./output_images/transform_perspective.png "transform perspective"

[img_aerial]: ./output_images/draw_aerial.jpg "draw aerial"
[img_normal]: ./output_images/draw_normal.jpg "draw normal"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

The code can be found in Advanced_Lane.ipynb

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for camera calibration is in the cell below the header "Camera Calibration", in function "undistort_image"

To calibrate the camera, and undistort images, I find points in "real world" coordinates and the same points within the raw image.  Once I identify points in these two spaces, I can convert between the two.  This helps us to undistort images, transforming the raw images into the "real world" coordinates that we would see without looking through a camera lens.

![calibration][img_calibrate]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

With the camera calibration, we can undistort any images taken with that camera, including images of the lane:
![undistort][img_undistort]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

See the functions under the header "Binarize Image": abs_sobel_thresh, mag_thresh, dir_threshold.
I tried gradients in the x and y direction, gradient magnitude, and gradient orientation.  I also used the red and saturation channels.  


![binary x y][img_binary_xy]
![binary mag dir][img_binary_md]


The original image has shadows on the lane.  Saturation is able to avoid detecting shadows, and I choose that over using red thresholding.  In addition, I see that the gradient in the x direction is best able to avoid detecting non-lane lines.  At the same time, the x gradient detects some lane line points that are not detected by saturation alone.  I use an "or" operand to display points that are either detected by saturation, or by the x gradient (or both).


![binary red saturation][img_binary_rs]
![binary combined ][img_binary_co]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Under the header "Perspective Transform", see the function "transform_perspective".
Similar to the principle behind calibrating the camera, I first find two sets of points, one with coordinates in the normal camera view, and then the same points' coordinates in an aerial transformed (or "warped") view.  We can see the normal and transformed perspective, with the corresponding points used for the transformation connected by red lines.

![transform perspective][img_perspective]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

See under the header "Find Lane Lines".  I define a Line class and Lane class (which contains a left Line object and right Line object).  I create a function "lane_line_detect", which uses two functions for searching for lane lines: "lane_line_window_search" and "lane_line_window_search" and "lane_line_given_previous".  I also create two functions for drawing the pixels that are detected as lane lines, and the polynomial that is fitted based on the found points: "draw_aerial" and "draw_normal".
The "compare_fits" function compares the most recent fitted line with the average of previous recent lines.

The lane_line_window_search function first uses a histogram to identify the left and right lines at the point closest to the car (at the bottom of the image).  From there, it uses a series of rectangles to search within for thresholded pixels.  It then fits those points to a second order polynomial.

The lane_line_given_previous function uses an existing fitted polynomial, and searches for thresholded pixels to the left and right of this "previous" fitted line.  It then fits the found pixels to get the new fitted polynomial.

The lane_line_detect function first uses previous fits to find new fits, but also uses window search when the first method gets results that are significantly different from previous images.  If neither method appears to give a result somewhat within the range of the previous fits, it assumes that the fit is bad for this image, and skips it to keep the line from jumping erratically.

![draw aerial][img_aerial]



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature under the header "Curvature", in the function "curvature".

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is what the detected points and lane look like on the original undistorted image:

![draw aerial][img_normal]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

My pipeline function is under the header "Apply to Video", within the VideoProcessor class, and the pipeline_function function.

The pipeline function takes an image, undistorts it, uses thresholding to binarize it, then uses lane line detection to fit a polynomial to the left and right lines of the lane.  Then it draws the detected left and right line pixels (red and blue), and also draws the detected lane in green.

Here's a link to the final result [link to my video result](./output_video/project_video_out.mp4).  Or see the subfolder "output_video/project_video_out.mp4".

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had difficulty dealing with shadows and turns, as these caused the lane detection to jump off the actual lane.  Even though I throw away fitted lines that appear too different from previous recent lanes, if I throw away too many fits, then the new images are no longer similar to the recently saved fits, which means that the lane detection may freeze at older fits and never accept new fits.  To avoid this, I tuned the lane detection so that it won't throw away bad fits too often, and will keep trying to fit the most recent image.  

Another issue that arises is when the lane detection jumps wildly and starts thinking that something else is the left or right line.  For instance, the lane detection might jump to the left too much, and start detecting the shoulder of the road as the left lane line.  To avoid this, I kept the size of the detection windows fairly narrow, so that they won't detect other edges nearby that are not actually lane lines.

I used various metrics to compare the current fit with recent fits.  I used radius of curvature, the car's offset from the lane center, and percent differences of the polynomial coefficients.  

In all those cases, I ended up detecting large differences and then discarding recent fits.  This was a problem, because when I discard too many recent fits, the car has already moved onto new parts of the road that no longer match the saved previous fits.  

I settled on using the constant coefficient of the left and right lines, as the metric for comparing the new and previous fits.  This is relatively stable, and represents the line's position along the x axis (horizontal direction).  The higher order coefficients for y^2 and y are less stable, because they represent the curvature of the line, which might vary quite a lot from frame to frame even when the fit appears to match the image fairly well.

