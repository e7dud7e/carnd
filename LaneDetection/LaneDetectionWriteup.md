#**Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 7 steps.  

1. crop out a region of interest
2. convert to grayscale
3. blur image to smooth lines
4. find edges with Canny edge detection
5. find lines in Hough space
6. Draw one of the following:
    - Draw detected lane lines
    - Draw full and extrapolatd lanes
7. Overlay lanes over original image

For cropping out the region of interest, I drew a quadrilateral that looks like a pyramid with 
the top chopped off.  This reduces the image to the lane lines and other objects near the lane.

Converting to grayscale makes edge detection simpler, because the image is just varying levels of brightness
as opposed to varying levels of three colors.

Blurring the image makes edge detection and also line detection better able to identify major lines, as opposed to smaller lines that make up the general outline of a shape.  For instance, if we wanted to detect the edges of a face, blurring helps us focus on detecting the outline of the face instead of capturing the small wrinkles and freckles.

Canny edge detection finds edges by finding places where the pixel value changes significantly.  I think of it as walking up and down various hills, and noting when the difference between the bottom and top of the hill is significantly steep.  The function's high_threshold is the slope of that hill for which anything as steep.  Any slopes that are in-between the two threshold are still added to the edge if they are adjacent to a pixel that is at or above the high_threshold.  I chose 200 and 100 for the high and low thresholds, but other ranges seem to turn out fine as well.

After detecting edges, I use Hough space (parameter space) to determine when several adjacent pixels form a line.  If we think of parameter space as representing lines with a single pair of values (radius and angle), then we can notice that when we see pixels that make up a line in the original image, that line can be represented as a single point (radius, angle) in parameter space.  The pixels themselves are wavy lines in parameter space.  These wavy lines all intersect at the radius and angle that is also the line in the original image that connects those pixels.

I set `rho` (the smallest unit for the radius in parameter space) to 1, and `theta` (the angle) resolution to 1 degree (pi/180 radians), to have a high resolution representation of the image.  I set the `threshold` (minimum number of pixels that fall within the same line) to be 20, to avoid picking up small stray lines that are not lanes.  I set `min_line_length` to 5, which is the minimum number of adjacent pixels on the line.  This eliminates smaller detected lines as well.  I set `max_line_gap` to 10, as a higher value ends up detecting longer lines even if there are some gaps in-between the detected edge points.

Most of my original work focused on processing the lines found in Hough space, and determining which lines represent the left lane, the right lane, or not a lane.

First, I get some detailed information about the lines.  I get the x,y coordinates of the two vertices that make up each line and fit them to get the slope and intercept.  I also notice that cropping the image to focus on the region of interest also produces pronounced lines on the left and right, which are not lane lines.  Since I know where the region's edges are based on the original region of interest, I remove lines that fall near the region's boundaries, effectly further cropping the image with a slightly smaller region.  I just label these with a flag `is_region_edge` instead of discarding the region edge lines.

Next, I attempt to determine which lines represent the left lane or the right lane.

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


###3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...