# **Traffic Sign Recognition** 

## Writeup

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 by 32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a plot of a sample traffic sign in the dataset

![Sample traffic sign from the dataset][./examples/traffic_sign.png]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I converted the image to grayscale to filter out the color effect (as light condition and materials may affect this).

![Sample grayscale traffic sign from the dataset][./examples/gray_traffic_sign.png]

As a final step, I normalized the image data to avoid slow and unstable learning process since the weights were initialized with identical normalization distribution.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x8 	|
| RELU and dropout					|												|
| Max pooling	      	| 2x2 stride, valid padding  outputs 14x14x8 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16				|
| RELU and dropout					|												|
| Max pooling	      	| 2x2 stride, valid padding  outputs 5x5x16 				|
| Fully connected		| input flattened(5x5x16 = 400) outputs 240								|
| Fully connected				| input 240 output 80       									|
| Softmax						| input 80 output 43												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an epoche number of 100 (to improve the accuracy), batch size of 128 (to improve the learning speed), dropout probability of 0.8 (to avoid overfitting), and learning rate of 0.0008. These parameters were tuned based on grid search from the initial value of the LeNet introduced in the lessons. However, I found there were unintuitive scenarios that increasing batch size and reducing learning rate reduces accuracy. These were something suspicous and not explained (actually oppose) yet by anything learnt in the lessons.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.942
* test set accuracy of 0.912

The first architecture was a converlution layer to filter the images into small piece. It was good to identify simply patterns. The problem of this architecture was that it could be too small to identify patterns or too large to lose its track on the patterns. 

The LeNet architecture was chosen. It was using two convonlution layers to identify simple patterns and combined patterns. Similarly, the traffic signs composites patterns (lines, curves, shapes, etc.) that are identified by the brain to interpret. The LeNet architecture simulates the brain work in this way and fit this traffic sign identification application. The training set accuracy suggested that the model was well trained with the training set (accuracy > 0.99). The validation set accuracy suggested that the model was good, though the validation set was given to the model. The test set accuracy suggested that the model could predict brand new images on 0.912 accuracy, which is more reliable when evaluating the model prediction accuracy.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Sign 1][./examples/new_sign_1.png] ![Sign 2][./examples/new_sign_2.png] ![Sign 3][./examples/new_sign_3.png] 
![Sign 4][./examples/new_sign_4.png] ![Sign 5][./examples/new_sign_5.png]

The first image might be difficult to classify because the contrast was very low. The third image might be difficult to classify as the resolution was very low.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)      		| Priority road   									| 
| Right-of-way at the next intersection     			| Right-of-way at the next intersection 										|
| Road work					| Speed limit (80km/h)											|
| Go straight or right	      		| Go straight or right					 				|
| Roundabout mandatory			| End of speed limit (80km/h)      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This disagrees with the accuracy on the test set of 91%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th and 15th cell of the Ipython notebook.

For the first image, the model is not sure that this is a priority road sign (probability of 0.38), and the image contains 'Speed limit (70km/h)'. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .38         			| Priority road   									| 
| .15     				| Speed limit (70km/h) 										|
| .08					| Speed limit (100km/h)											|
| .07	      			| End of all speed and passing limits					 				|
| .06				    | Speed limit (120km/h)      							|


For the second image, the model is pretty sure that this is a 'Right-of-way at the next intersection' sign (probability of 0.99), and the image does contain the 'Right-of-way at the next intersection' sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Right-of-way at the next intersection  									| 
| .00     				| Beware of ice/snow 										|
| .00					| Double curve											|
| .00	      			| Slippery road					 				|
| .00				    | Dangerous curve to the left      							|

For the third image, the model is pretty sure that this is a 'Speed limit (80km/h)' sign (probability of 0.99), and the image contains the 'Road work' sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (80km/h) 									| 
| .00     				| Wild animals crossing 										|
| .00					| Speed limit (50km/h)											|
| .00	      			| Speed limit (100km/h)					 				|
| .00				    | Speed limit (60km/h) 							|

For the forth image, the model is pretty sure that this is a 'Go straight or right' sign (probability of 1.00), and the image does contain the 'Go straight or right' sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Go straight or right									| 
| .00     				| Speed limit (60km/h) 										|
| .00					| Keep right											|
| .00	      			| Turn left ahead					 				|
| .00				    | Dangerous curve to the right 							|

For the fifth image, the model is relatively sure that this is a 'End of speed limit (80km/h)' sign (probability of 0.61), and the image contains the 'Roundabout mandatory' sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .61         			| End of speed limit (80km/h)									| 
| .26     				| End of no passing by vehicles over 3.5 metric tons 										|
| .08					| End of no passing											|
| .05	      			| End of all speed and passing limits					 				|
| .04				    | Roundabout mandatory 							|
