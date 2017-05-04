#**Traffic Sign Recognition** 

## Project description and review


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image_traffic_sign_images_by_type]: ./readme_images/traffic_sign_images_by_type.png "signs by type"
[image_distribution_train]: ./readme_images/distribution_train.png "training data count per type"
[image_distribution_valid]: ./readme_images/distribution_valid.png "validation data count per type"
[image_distribution_test]: ./readme_images/distribution_test.png "test data count per type"
[image_transformed_images]: ./readme_images/transformed_images.png "Rotated and translated images"
[image_equalize]: ./readme_images/equalize.png "equalized histogram"
[image_sign01]: ./new_signs/bicycles_crossing.jpg "Traffic Sign 1"
[image_sign02]: ./new_signs/children_crossing.jpg "Traffic Sign 2"
[image_sign03]: ./new_signs/dangerous_curve_left.jpg "Traffic Sign 3"
[image_sign04]: ./new_signs/dangerous_curve_right.jpg "Traffic Sign 4"
[image_sign05]: ./new_signs/end_speed_passing_limits.jpg "Traffic Sign 5"
[image_sign06]: ./new_signs/right_of_way.jpg "Traffic Sign 6"
[image_sign07]: ./new_signs/roundabout_mandatory.jpg "Traffic Sign 7"
[image_sign08]: ./new_signs/wild_animals_crossing.jpg "Traffic Sign 8"
[image_signs_rescaled]: ./readme_images/new_signs_rescaled.png "Traffic Signs rescaled"
[image_sign01_pred]: ./readme_images/bicycle_pred.png "Prediction"
[image_sign02_pred]: ./readme_images/children_pred.png "Prediction"
[image_sign03_pred]: ./readme_images/left_pred.png "Prediction"
[image_sign04_pred]: ./readme_images/right_pred.png "Prediction"
[image_sign05_pred]: ./readme_images/end_pred.png "Prediction"
[image_sign06_pred]: ./readme_images/right_of_way_pred.png "Prediction"
[image_sign07_pred]: ./readme_images/roundabout_pred.png "Prediction"
[image_sign08_pred]: ./readme_images/wild_pred.png "Prediction"
[image_comparison01]: ./readme_images/comparison1.png "Comparison"
[image_comparison02]: ./readme_images/comparison2.png "Comparison"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### README

Here is a link to my [Traffic Sign Classifier code](https://github.com/e7dud7e/carnd/tree/master/TrafficSigns)

###Data Set Summary & Exploration

#### 1. Summary of the data set

I used numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is 32 x 32 x 3 (height, width, color channels)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory Analysis

Here is a sample of training images categorized by their sign type.

![traffic sign images by type][image_traffic_sign_images_by_type]

Here is the distribution of the training set, validation set, and test set (count of sample images for each sign type).

![training data distribution][image_distribution_train]

![validation data distribution][image_distribution_valid]

![testing data distribution][image_distribution_test]

### Design and Test a Model Architecture

#### 1. Pre-processing data


To add more variety to the data, and to avoid overfitting (especially for signs that have fewer samples), I rotate and translate the images.  I do this with randomized angles and shift sizes.  Instead of generating a new static data set, I will randomly generate new images for a fraction of each batch, during training.  This allows the data sets to be a bit different for each epoch.

![rotated and translated images][image_transformed_images]

I also equalize the histogram for each image, which means that I stretch out the distribution of each image's red, green, blue values so that their min and max are 0 and 255.  This increases contrast, and should help reduce the differences caused by different lighting conditions.

![equalized histogram][image_equalize]

I then normalize the images so that the 0 to 255 scale is converted to a -1 to 1 scale.  Since each image's histogram was previously normalized so that each has a min and max of 0 and 255, the result is that each image has a min and max of -1 and +1.  Normalizing to these smaller values makes it easier to train the model during back propagation, because back propagation involves a series of multiplications, and we want to keep the products from getting too large or too small.


#### 2. Model Design

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							|
| Convolution 1x1 (0)   | 1x1 stride, same padding, outputs 32x32x3   	|
| batch normalization(0)| outputs 32x32x3   	                        |
| Relu (0)              | outputs 32x32x3   	                        | 
| Convolution 3x3 (1a)  | 1x1 stride, same padding, outputs 32x32x32 	|
| Convolution 3x3 (1b)  | 1x1 stride, same padding, outputs 32x32x32 	|
| Batch normalization (1)| outputs 32x32x32   	                        |
| Relu (1)			    | outputs 32x32x32   	                        |
| Max pooling (1)      	| 2x2 stride,  outputs 16x16x32 				|
| Dropout (1)           | 50% keep probability, outputs 32x32x32        |
| Convolution 3x3 (2a)    	| 1x1 stride, same padding, outputs 16x16x64 	|
| Convolution 3x3 (2b)   	| 1x1 stride, same padding, outputs 16x16x64 	|
| batch normalization (2)	| outputs 16x16x64   	                        |
| Relu (2)				| outputs 16x16x64   	                        |
| Max pooling (2)      	| 2x2 stride,  outputs 8x8x64 				|
| Dropout (2)              | 50% keep probability, outputs 16x16x64        |
| Convolution 3x3 (3a)     	| 1x1 stride, same padding, outputs 8x8x128 	|
| Convolution 3x3 (3b)    	| 1x1 stride, same padding, outputs 8x8x128 	|
| batch normalization (3)	| outputs 8x8x128   	                        |
| Relu (3)					| outputs 8x8x128   	                        |
| Max pooling 	      	| 2x2 stride,  outputs 16x16x32 				|
| Dropout (3)              | 50% keep probability, outputs 8x8x128       |
| Flat (1)                | flatten dropout (1)                           |
| Flat (2)                | flatten dropout (2)                           |
| Flat (3)                | flatten dropout (3)                           |
| Flatten layer (4)          | Concatenate flat (1), flat (2), flat (3)      |
| Fully Connected (5)     | Outputs 1024                                  |
| Batch normalization (5) | Outputs 1024                                  |
| ReLu (5)                | Outputs 1024                                  |
| Dropout (5)             | 50% keep probability, outputs 1024            |
| Fully Connected (6)     | Outputs 1024                                  |
| Batch normalization (6) | Outputs 1024                                  |
| ReLu (6)                | Outputs 1024                                  |
| Dropout (6)             | 50% keep probability, outputs 1024            |
| Logits (7)              | Outputs 43 (for 43 traffic sign types)        |
| Softmax (8)			  | Outputs 43; converts logits to probablities   |
 

#### 3. Training procedure

I use cross entropy to conver the softmax probabilities into a cost (loss).  Then I use Adam Optimizer to perform gradient descent and back propagation.

For gradient descent, I decided to stay with a learning rate of 0.01 for a reasonably fast runtime (3 hours or less).  Using smaller learning rates would probably have improved validation accuracy, but the time to train sometimes took 8 hours on some attempts, and took too long for me to test and refine the model.  

I used 20 epochs, and a batch size of 512.  For the first epoch, I randomly choose 90% of the batch, and for each original image in that fraction, I generate 3 new images with rotation and translation. I then append the new data with all of the original data.  This means that for the first epoch, each batch size is 512 + (0.90 * 512) * 3, or 512 + 1382.4 = 1894.4.  This stays below the upper limit of 2048 that I found would stay within memory constraints.  For each successive epoch, the fraction of data to use for generating new data decreases by 0.10, so the second epoch transforms 80% of each batch, etc.  The 9th batch generates 10% new data, and after that, the later epochs train with the original data only.


#### 4. Model improvement process

My final model results were:
* training set accuracy of 0.9999
* validation set accuracy of 0.9882 
* test set accuracy of 0.9755

#### Initial model designs and their problems

I pre-processed the data by scaling each image to have a min and max of -1 to +1. My goal was to reduce differences in lighting by scaling each individual image to have the same range.  This seemed to help, but I later chose to use opencv's histogram equalization to stretch the distribution of images, for the same goal of taking out differences due to lighting.

I initially tried 3 convolutional layers followed by 3 fully connected layers.  I chose layer sizes that were rather large, took a long time to train (4 to 6 hours), and overfit the training data while not sufficiently improving validation accuracy.

#### New data generation

Next, I tried transforming the data and trying to make the distribution of each class equal.  I first tried to generate multiple images using translation and rotation, so that there were 2,500 total number of images for each of the 43 classes.  This meant generating a lot of images for those classes that started with just a few hundred images.  This made the data set larger, but I probably generated too many copies of the same image, and had validation accuracy of 92%.  

Next, I tried generating images for only the classes that had very few to start with (less than 800).  This made the distribution more balanced without generating too many copies of the same image.  I ended up with a 94.69 validation accuracy.  I tried the same with a learning rate of .001 instead of .01, but only reached 89% validation accuracy at epoch 20.  It was going to take much longer to train.

I then tried creating batches using random sampling of the whole data set for each batch.  Then I transformed a fraction of each batch, and swapped out each original image for its transformed version.  I ended up with worse performance (around 80% validation accuracy).  I think the problem was that by randomly sampling for each batch, the entire epoch may end up seeing the same image more than once, and not see some images at all.  Also, as I found out later, swapping out an original image for its transformed image did worse than if I kept both the original new new images for training.  I think that making the training data too unstable will make training more difficult, because then the weights are being pushed and pulled in a new direction at each epoch.

Another design choice I made was to shuffle the training set at the start of each epoch, so that batch 1 in epoch 1 is not the same data as batch 1 in epoch 2.  This at least did not hurt performance.  I was trying to keep the model from overfitting the training data.

Since greyscale did not improve my results, I tried adding a 1x1 convolution layer that would let the model choose what weight to give each of the 3 color channels.  This seemed to make things worse.

#### Changes to model layers

Since I was getting worse results with transformed data, I went back to using original data only to use as a baseline for improving the design.

I also tried making the layers as large as possible, which is what I had tried in a previous CNN project.  This only got 92% validation accuracy, and took 8 hours to train.

I found that by using 2 fully connected layers instead of 3, and by making the last fully connected layer smaller than the first, validation accuracy improved.  So, for instance, a fully connected layer of 2048 followed by another of size 1024 does better than a fully connected layer of 2048 followed by another of size 2048. This improved accuracy from 92% to 94%.  This reminds me of advice I once received that the hidden layers tend to work better when the middle layers are bigger, while the later layers get smaller as they get closer to the output layer.

After a few attempts at swapping out original images with transformed images made validation accuracy worse, I went back to using original data again, and modified the architecture.  I based it off of Vivek Yadav's medium post, which looks more similar to the VGG net architecture.  There are two convolutional layers followed by pooling and dropout. I created 3 sets of these.  There is also a colormap layer at the beginning, which is a 1x1 convolution that chooses how to weight the 3 input color channels.  Also, similar the "inception module" concept, I take the outputs of each of the 3 sets of convolutions (the dropout layers with depths 32, 64 and 128), and flatten, then concatenate them all to create the flatten layer.  So the fully connected layer receives inputs from all 3 levels of convolutions.  The idea is that the fully connected layer should be able to see the more detailed features of the earlier convolutions, as well as the more general features of the later layers.  With original data only, I reached 94.56% validation accuracy.

I then made a couple attempts to use transformed data, swapping out original images with their transformed images.  This made validation accuracy worse, around 92% validation accuracy.

#### Histogram equalization

Finally, at trial 25, I changed the image pre-processing.  Instead of scaling each image to have a min and max of -1 and +1, I used histogram equalization, which stretches the distribution of each image to 0 to 255.  Then I scaled all images to have a global min and max of -1 and +1.  The end result is that each image has a range of -1 and +1, but for all the values in-between, opencv's histogram equalization may be different from a directly linear scaling. 

#### Keeping all original data when generating new images

The second change I made was to append the transformed data to all of the original data in each batch, instead of swapping out the original images for the new ones.  This meant that the actual batch sizes would be larger, depending on what fraction of the batch I want to use to generate new images.  For epoch 1, I sample 90% of each batch, and for each image in that 90%, generate 1 new image and append it to the batch for training.  For epoch 2, the fraction transformed is 80%, then 70% down to 0% by epoch 10 and later.  Validation accuracy reached 97.8%.

#### Final Model

For epoch 26, I generated 3 new images for each original image. This resulted in a 98.82% validation accuracy at epoch 19.  It already reached 98% validation accuracy at epoch 9, so I probably could have reduced the learning rate if I wanted to run 20 epochs.

I saved the session for the last 3 epochs, so I used epoch 19 (with the highest validation accuracy) for testing.

### Test model on test data set
Running the final model on test data, I get a test accuracy of 97.55%.

### Test Model on New Images

#### 8 Downloaded traffic sign images and their predicted signs

Here are some German traffic signs that I found on the web:

![alt text][image_sign01] ![alt text][image_sign02] ![alt text][image_sign03] 
![alt text][image_sign04] ![alt text][image_sign05] ![alt text][image_sign06] 
![alt text][image_sign07] ![alt text][image_sign08] 

These images have water marks with text covering the signs.  This may be an issue, as patterns that aren't prominent for human eyes may be considered more significant by the network.  Another problem is that when I rescale the images down to 32x32, the images become very grainy, and the details on the signs become blurred and distorted.

![alt text][image_signs_rescaled]

#### Predictions on downloaded images

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bicyle Crossing          | Keep Left                                  | 
| Children Crossing        | Children Crossing                          |
| dangerous curve left     | U-turn                                     |
| dangerous curve right	   | Speed Limit 70 km/h                        |
| end speed passing limits | Priority Road	                            |
| right of way			   | Right of way                               |
| roundabout mandatory     | Keep Right                                 |
| wild animal crossing     | Speed Limit 80 km/h                        |


The model was able to correctly guess 2 of the 8 traffic signs, which gives an accuracy of 25%. This is pretty low compared to 97% accuracy on the test set.

#### Certainty levels for each prediction

![alt text][image_sign01_pred]
![alt text][image_sign02_pred]
![alt text][image_sign03_pred]
![alt text][image_sign04_pred]
![alt text][image_sign05_pred]
![alt text][image_sign06_pred]
![alt text][image_sign07_pred]
![alt text][image_sign08_pred]

### Comparing actual to predicted
Here are sample images of the first pairs of actual and predicted signs.
The bicycle crossing was predicted as keep left, possibly because the diagonal side of the triangle was matched to the diagonal arrow in the 'keep left' sign.

Children crossing was predicted as children crossing.

Dangerous left turn was predicted as speed limit 70 km/h.  This is possibly because after rescaling the image to 32x32, the curved arrow looks like the number 7.

The Dangerous right turn sign was predicted as 'traffic signals.'  It may be because the images both look like triangles with a vertical line inside.


![alt text][image_comparison01]

Here are samples for the other pairs of actual vs. predicted signs.
The 'end all speed and passing limits' was predicted as 'priority road.' It may be that the diagonal line through the circle was matched with a diagonal line in the diamond.

'right of way next intersection' was correctly predicted.

'Roundabout mandatory' was predicted as 'keep right.'  These are both blue circles with white arrows inside them.

'Wild animal crossing' was predicted as 'speed limit 80km/h.'  I'm not sure why it could be interpreted as a speed limit sign.

![alt text][image_comparison02]


