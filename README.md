# Build a Traffic Sign Recognition Project
## The goals / steps of this project are the following:
* Using the tools discussed for Deep and Convolutional Neural Networks develop a pipeline to to classify traffic signs.
* Train and validate a model so it can classify traffic sign images using the German Traffic Sign Dataset

Steps
* Load the dataset
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Images and Results

### Visualization of the Dataset one Image from each Class:
<p align="center">
  <img width="500" alt="image_first_in_class" src="https://user-images.githubusercontent.com/28680734/29999787-0497ebcc-900a-11e7-84c0-f57db4b57a9b.png">
</p>

Test Images
<p align="center">
  <img width="500" alt="image_test" src="https://user-images.githubusercontent.com/28680734/29999822-4e67866c-900b-11e7-9686-561499d865cd.png">
</p>

Predicited Signs
<p align="center">
  <img width="500" alt="image_predicted_sign" src="https://user-images.githubusercontent.com/28680734/29999835-d0937790-900b-11e7-9595-4359bbf42601.png">
</p>

## Rubric Points
The writeup herein addresses the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

#### here is a link to my [project code](https://github.com/pharidm/CarND-Traffic-Signs-Classifier/blob/master/Report_Traffic_Sign_Classifier.pdf)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. The code and analysis uses python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python and numpy library to calculate summary statistics of the traffic
signs data set:

* Number of training examples is 34799.
* The size of the validation set is 4410. 
* Number of testing examples is 12630.
* The shape of a traffic sign image is (32, 32, 3).  32 pixels by 32 pixels, with the third value representing 3 channels for a color image. 
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the Training Data Set by class.  It also shows the minimum number of images per class and the maximum number of images per class.  

Distribution of Training Set Data
<p align="center">
  <img width="500" alt="image_distribution_training_set" src="https://user-images.githubusercontent.com/28680734/29999811-c17c90c6-900a-11e7-8c8e-bf32c869cb51.png">
</p>

### Design and Test a Model Architecture

#### 1. Preprocessing the image included the following steps
* Step 1: The Training set was shuffled to improve the accuracy results using sklearn.utils.shuffle 
* Step 2: Each image was converted to gray scale to minimize the variability do to the different quality of pictures and cameras.  In addition we can amplify the features that are most interesting to our classification model. Handling colored images involves a matrix for each of the three colors, making the demand for memory and computations even larger. 

* Step 3: Each image was then normalized in order to help our model reach convergence; the point where our predictions return the lowest error possible. Since all the values are operating on the same scale our model should return the results faster. Min, Max scaling was employed on the gray scaled image.  The image below shows the results of our Training, Validation and Test dataset pre and post normalization. The function defined to normalize the data is below as well. Numpy Newaxis was used to reduce the dimensions of the data for use in a Tensor. 

<p align="center">
<img width="500" alt="normalization_function" src="https://user-images.githubusercontent.com/28680734/30006642-be991f94-90b1-11e7-9e7e-9cb245defc4f.png">
</p>


<p align="center">
<img width="500" alt="pre_post_processing" src="https://user-images.githubusercontent.com/28680734/30006677-89e6d916-90b2-11e7-8e9f-22cbc8e6d46c.png">
</p>

#### 2. My final model was built using the LeNet-5 Architecture.   It consists of the following layers:
<center>

| Layer         		|     Description	        					    |
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Gray scale image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	|1x1 stride,  outputs 10x10x16       									|
| RELU			|                								| Convolution 3x3     	| 2x2 stride, same padding, outputs 5x5x16 	|
| Flatten 		| output 400           |
| Fully Connected        | output 120          |
| RELU Fully Connected|
| Fully Connected        | output 84         |  
| RELU Fully Connected  |       |
| Fully Connected        | output 43 |
|Return Logits|           |

</center>

####  3. A iterative approach was used to train the model starting with the basic LeNet architecture. 

ADAM optimizer was picked a good starting point to update the model network weights.  Instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data. “Adam combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems.” [Reference Jason Brownlee] (https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)

My model returned results close to the required 0.93 accuracy on the first run with 10-15 EPOCHs.  An EPOCK is a single forward and backward pass of the whole dataset. This is used to increase the accuracy of the model without requiring more data. I increased the EPOCHs and played with the learning rate and dropout.

In order to prevent over fitting I employed a 50% dropout rate.  The key idea is to randomly drop units (along with their connections) from the neural network during training.  A learning rate of 0.001 was picked as it appears to be a “balanced” default value, were model should not overshoot nor converge too slowly.

Reference: [A Simple Way to prevent Neural Networks from Overtraining] (http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

The key parameters of my model is as follows:

* EPOCHS     = 20
* BATCH_SIZE = 128
* keep_prob  = tf.placeholder(tf.float32)
* rate = 0.001
* dropout = 0.5

####4. Architecture Approach
A iterative approach was used to train the model starting with the basic LeNet architecture. Improvements to this architecture would have been to employ GoogleNet, but due to time constraints it was not considered.  Using GoogleNet would have returned a high accuracy much faster than using 20 EPOCHs.  LeNet was choose because it was discussed in class to provide a good starting point and high level of accuracy without much modification.  Because we are implementing a classical vision system model on a text book case of traffic signs, our LeNet architecture performed well.  Moving on to none basic classification problems, or pictures of low quality might need a more improved model.  

####4.1 My final model results were:
* Training set accuracy of 0.986
* Validation set accuracy of 0.942
* Test set accuracy of 0.933

####4.2 Accuracy Assessment 

The Training set is used to to fit the parameters [i.e., weights]

The Validation set is to tune the parameters [i.e., architecture].  The validation set is an extension of the training set, by keeping them separate to prevent overfitting.  

The Test set is used to assess the performance i.e., generalization and predictive power of our model. 

All three results showed a high degree of accuracy at the final 20th EPOCH over 0.93 % indicating our model is highly accurate. It took 10 EPOCH runs or more before the minimum accuracy appeared stable.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|




