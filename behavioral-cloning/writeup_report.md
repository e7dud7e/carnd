#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[run01]: ./examples/history_loss_run01.png "run 1"
[run02]: ./examples/history_loss_run02.png "run 2"
[run03]: ./examples/history_loss_run03.png "run 3"
[run04]: ./examples/history_loss_run04.png "run 4"
[run05]: ./examples/history_loss_run05.png "run 5"
[run06]: ./examples/history_loss_run06.png "run 6"

[recovery01]: ./examples/off_road_right.jpg "Recovery Image"
[recovery02]: ./examples/off_road_left.jpg "Recovery Image"
[original]: ./examples/original.jpg "Original Image"
[flipped]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* I also made a video recording, discussing my [project](https://www.youtube.com/edit?o=U&video_id=zTbS5nF_Ieo)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64.

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer after the convolutional layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. This includes data from normal driving, recovering from driving off the road, driving the track in the opposite direction, and driving on the second track. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovering from the left and right sides of the road.  Other data sets, such as driving in the opposite direction, or driving on the advanced track, did not appear to improve the model, so they were left out to avoid adding noise.  Sometimes adding additional data would make the car have more erratic swerving behavior.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate because they've already tested and found a model that works well.  In general, the first layer has the largest filter size, and the smallest depth (number of channels).  As we move along to later layers, the filter size gets smaller, and the depth increases.  The reason for the larger filter size at the beginning, is that the original image is large, and relevant features may only be found with larger "windows" (filters).  Each layer after that is looking at a collection of filters from the previous layer, so even a small filter at the last layer is actually observing a much larger range of the original image.

The reason for smaller depths at earlier layers is in part to make sure the calculations don't exhaust all the memory.  The higher depths at later layers is intended to allow for more combinations of basic features to form the more complex features.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

The first model I ran only used data from normal driving, twice around the track, counter-clockwise.  The validation set's mean squared error loss was actually lower than that of the training set, which is unusual.  At least this means that the model did not overfit to the training data.  When running in autonomous mode, the car drives smoothly and makes some turns properly.  However, when it starts to drift towards the edge of the road, the car had not learned to correct itself to stay in off the edges, and eventually drifts off the side of the road.


![run01][run01]


I ran successive models by adding on additional data sets.  

In a second training set adds training from driving clock-wise (in the opposite direction).  The resulting mse loss for both validation and training still decreased, but not as much as when there was only the normal driving (counter-clockwise) training data.  In autonomous mode, the car still drifts off the side of the road early on.

![run01][run02]


Similarly, I cumulatively added these data sets: normal driving, driving in the track in the other direction, driving onto the road when the car is positioned on the dirt or grass, completely off the road, and driving on the second advanced track.  Adding more data appears to make the car drive less smoothly, but the car still did not learn not to drift off the side of the road.


![run01][run03]


![run01][run04]


In order to fix this, I recorded another data set, in which the car is drifting on the edge of the lane (but still on the road), and recorded examples of the car steering more sharply towards the center, to get off the shoulder.  The car did pull itself towards the center when drifting onto the edge, but tended to pull itself towards the right side whenever it entered the bridge.

![run05][run05]


To fix this, I cleaned the off road data by removing the training data related to the bridge, which was improperly teaching the car to veer to the the right side.

I then trained a sixth model, using only the normal driving behavior and the examples of getting off the sides of the road.  The model's validation loss was again less than teh training loss, and both decreased with successive epochs.  In autonomous mode, the car was able to navigate one loop the track without falling off the road.  The car also learned to pull itself towards the center whenever it was drifting on either the right or left sides of the road.

![run 06 good][run06]

#### 2. Final Model Architecture

The final model architecture (model.py lines 51-63) consisted of a convolution neural network with the following layers and layer sizes:

Two pre-processing "layers" to crop the image (to remove features unrelated to the road), and normalization (to set the input range from -1 to 1).  This is followed by 5 convolutional layers, a flatten layer, and 3 fully connected layers and an output layer.

Convolution 1: window 5x5 and depth 24
Convolution 2: window 5x5 and depth 36
Convolution 3: window 5x5 and depth 48
Convolution 4: window 3x3 and depth 64
Convolution 5: window 3x3 and depth 64
Flatten Layer
fully connected 1: size 100
fully connected 1: size 50
fully connected 1: size 10
output layer 1: size 1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![normal][normal]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to pull itself towards the center whenever it started driving on the shoulder.

![recovery01][recovery01]

![recovery02][recovery02]



To augment the data sat, I also flipped images and angles so that every left turn example is balanced with an equivalent right turn example. For example, here is an image that has then been flipped:

![original][original]

![flipped][flipped]

After the collection process, I had X number of data points. I then preprocessed this data by cropping out the features unrelated to the road (the bottom part containing the hood of the car, and the top part containing the sky and scenery above the road).  I normalized the image so that pixel ranges of 0 to 255 are rescaled to -1 and 1.

I finally randomly shuffled the data set for each batch and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4, as in most runs, the rate of validation loss decrease started to flatten out at 4 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
