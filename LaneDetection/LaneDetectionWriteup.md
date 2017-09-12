# Finding Lane Lines on the Road

[//]: # (Image References)

[region_of_interest]: ./pipeline_images/whiteCarLaneSwitch_region.jpg

[gray]: ./pipeline_images/whiteCarLaneSwitch_gray.jpg

[blur_gray]: ./pipeline_images/whiteCarLaneSwitch_blur_gray.jpg

[edges]: ./pipeline_images/whiteCarLaneSwitch_edges.jpg

[hough]: ./pipeline_images/whiteCarLaneSwitch_hough.jpg

[region_edge]: ./pipeline_images/whiteCarLaneSwitch_region_edge.jpg

[other_lines]: ./pipeline_images/whiteCarLaneSwitch_other_lines.jpg

[lane_lines]: ./pipeline_images/whiteCarLaneSwitch_lane_lines.jpg

[full_lanes]: ./pipeline_images/whiteCarLaneSwitch_full_lanes.jpg

[overlay_lines]: ./pipeline_images/whiteCarLaneSwitch_overlay_lines.jpg

[overlay_lanes]: ./pipeline_images/whiteCarLaneSwitch_overlay_lanes.jpg

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 8 steps.  

1. crop out a region of interest
2. convert to grayscale
3. blur image to smooth lines
4. find edges with Canny edge detection
5. find lines in Hough space
6. get line details (left or right of image, slope, intercept)
7. Draw one of the following:
    - Draw detected lane lines
    - Draw full and extrapolatd lanes
8. Overlay lanes over original image

For cropping out the region of interest, I drew a quadrilateral that looks like
 a pyramid with the top chopped off.  This reduces the image to the lane lines 
 and other objects near the lane.
![region of interest][region_of_interest]

Converting to grayscale makes edge detection simpler, because the image is 
just varying levels of brightness as opposed to varying levels of three colors.
![gray][gray]

Blurring the image makes edge detection and also line detection better able 
to identify major lines, as opposed to smaller lines that make up the 
general outline of a shape.  For instance, if we wanted to detect the edges 
of a face, blurring helps us focus on detecting the outline of the face 
instead of capturing the small wrinkles and freckles.
![blur gray][blur_gray]

Canny edge detection finds edges by finding places where the pixel value 
changes significantly.  I think of it as walking up and down various hills, 
and noting when the difference between the bottom and top of the hill 
is significantly steep.  The function's high_threshold is the slope of 
that hill for which anything as steep.  Any slopes that are in-between 
the two threshold are still added to the edge if they are adjacent to a 
pixel that is at or above the high_threshold.  I chose 200 and 100 for the 
high and low thresholds, but other ranges seem to turn out fine as well.
![edge detection][edges]

After detecting edges, I use Hough space (parameter space) to determine 
when several adjacent pixels form a line.  If we think of parameter space 
as representing lines with a single pair of values (radius and angle), 
then we can notice that when we see pixels that make up a line in the 
original image, that line can be represented as a single point 
(radius, angle) in parameter space.  The pixels themselves are wavy lines 
in parameter space.  These wavy lines all intersect at the radius and 
angle that is also the line in the original image that connects those pixels.

I set `rho` (the smallest unit for the radius in parameter space) to 1, 
and `theta` (the angle) resolution to 1 degree (pi/180 radians), 
to have a high resolution representation of the image.  
I set the `threshold` (minimum number of pixels that fall within 
the same line) to be 20, to avoid picking up small stray lines 
that are not lanes.  I set `min_line_length` to 5, which is the 
minimum number of adjacent pixels on the line.  This eliminates 
smaller detected lines as well.  I set `max_line_gap` to 10, 
as a higher value ends up detecting longer lines even if there are some 
gaps in-between the detected edge points.
![hough lines][hough]

Most of my original work focused on processing the lines found in 
Hough space, and determining which lines represent the left lane, 
the right lane, or not a lane.

First, I get some detailed information about the lines.  
I get the x,y coordinates of the two vertices that make up each line 
and fit them to get the slope and intercept.  
I also notice that cropping the image to focus on the region of interest a
lso produces pronounced lines on the left and right, which are not lane lines.  
Since I know where the region's edges are based on the original region of 
interest, I remove lines that fall near the region's boundaries, 
effectly further cropping the image with a slightly smaller region.  
I just label these with a flag `is_region_edge` instead of discarding 
the region edge lines.

![region edge to be removed][region_edge]

Next, I attempt to determine which lines represent the left lane or 
the right lane.  I label each line as part of a lane if it meets all of 
these criteria: 
- It is not an edge created by cropping the region of interest
- The line has a slope that is generated by fitting a line 
(note that this has a shortcoming in that vertical lines are excluded)
- The line is not too horizontal (slope is somewhat close to 0).  

Here is an example of some lines that do not meet the criteria for a lane line:
![other lanes that are not lanes][other_lines]

I addition, the slope of the line should be consistent with its position 
(left or right) in the image:
- If the line is on the left side, its slope should be negative 
(from bottom left to top right; recall that the y axis increases 
as we move downward).
- If the line is on the right side, its slope should be positive 
(from bottom right to top left).

Once I have the lines that represent lanes, I can draw them directly (in red).  
![lane lines][lane_lines]

To show more complete lanes, I also have another function that considers 
the detected lane lines and draws single line for the left and right lane.  
To extrapolate lane lines, I first want to find lines that are similar 
in slope and intercept, and fit a line to only those most similar lines. 
So I do the following:
- Use k-means clustering to find the group of left lane lines that are 
most similar in slope and intercept.  Likewise, do this separately 
for right lane lines.  This removes some lines that are possibly 
not lane lines.  I make the assumption that the cluster with the 
most lines has the lines that best represent the lane line.
- Use linear regression to fit a line to the subset of lane lines from 
the largest cluster.  Do this separately for left and right lane lines.
- Extrapolate by using the regression model to predict the x coordinate 
when choosing a y coordinate.  For the two end points of the lane line, 
I assume that the lane line should extend to the bottom of the image 
(close to the car).  The other end of the lane line should either extend up 
to the highest vertex that's considered a detected lane line, 
but no further than a certain height that I set.  This threshold is meant 
to keep the extrapolated lane line from jumping too much in the video, 
and to handle cases when the highest lane line detected is still 
not actually a lane line.  
The function draws the extrapolated lane lines in green.

![full lane lines][full_lanes]

Finally, I overlay the detected lane lines or the extrapolated lane lines 
over the original image.
![overlay lane lines][overlay_lines]
![overlay full lanes][overlay_lanes]


###2. Identify potential shortcomings with your current pipeline

One shortcoming of the lane detector is that it still draws a straight line 
when faced with a curved road. Also, the region of interest is 
not sufficient for removing other none-lane lines that end up 
being interpreted as lane lines.  For instance, a wall or rail 
next to the lane gets detected as a lane line because it has similar 
slope relative to the lane.


###3. Suggest possible improvements to your pipeline

To handle curved roads, I could try fitting the lane lines to a curve.  
Since the detected hough lines are straight lines, I may want to adjust 
the parameters so that it will detect shorter lines.  
These shorter lines may work better when trying to fit a single lane line 
on the left and right sides.  

To handle lines from walls or rails that are parallel to the road, 
I can cluster groups of lines by their slope.  The lines that are closer 
to perpendicular (higher absolute slope) and closer to the middle 
of the horizontal x-axis will be the lane lines.  The other lines that 
get detected outside of the actual lane will be further from the mid-line, 
and will have a less steep slope.