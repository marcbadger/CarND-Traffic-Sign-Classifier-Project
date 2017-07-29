**Build a Traffic Sign Recognition Project Writeup**

The goals of this project were the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[imageAllClasses]: ./writeup_files/class_examples_2.png "One example of each sign class."
[imageClassDistribution]: ./writeup_files/frequency_of_classes_training_set.png "Distribution of training examples."
[imageClassDistributionValid]: ./writeup_files/frequency_of_classes_validation_set.png "Distribution of validation examples."
[imageHistogramEqualization]: ./writeup_files/histogram_equalization_example.png "Distribution of validation examples."
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

# Writeup

The following is a writeup of my work on the Traffic Sign Classifier project for the Self Driving Car Nanodegree program.  Here is a link to my [project code](https://github.com/marcbadger/CarND-Traffic-Sign-Classifier-Project/blob/master/Badger_2017_Traffic_Sign_Classifier.ipynb)

## Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the numpy shape and unique functions to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is [32, 32, 3].
* The number of unique classes/labels in the data set is 43.

#### 2. Exploratory visualization of the dataset.

Here is an example image from each class in the data set.  As you can tell, the quality of some of the images is pretty terrible (pixellated, occlusions, blurry, dark, etc.) and I'm amazed that the network gets most of these right!  The brighness of the images varies a lot, so there's a good chance that some sort of histogram equalization pre-processing will help.  It's not implemented here, but an alternative, more powerful, approach is to let the neural network determine the best pre-processing for us.  To do so we will need to augment the training data with copies of the training set images with altered brightness, translation, and affine transformations.

![alt text][imageAllClasses]

Another interesting observation is that the frequency of classes varies by an order of magnitude among classes. Presumably the distribution in the training set is representative of traffic signs in reality, so giving our classifier a bias towards common signs is probably be a good thing unless there is a case of an uncommon sign that is very very important to not mis-classify (i.e. a false negative of an uncommon but important sign would be bad). So we'll have to look closely at the confusion matrix to make sure rare, but important (such as wrong way signs) aren't ignored.

![alt text][imageClassDistribution]

The validation set follows a similar distribution:

![alt_text][imageClassDistributionValid]

## Design and Test a Model Architecture

#### 1. Preprocessed the image data.

The images in the validation set differed a lot in their brightness, so I used histogram equalization (from OpenCV) to equalize each color channel of the images.  I then normalized the image data by subtracted 128 from each pixel value and divided the result by 128 so that all pixel values were between -1 and 1.  I chose to keep all three color channels because sign color should be a very informative cue, and it is somewhat surprising that the original LeNet sign classification paper found that the network performed just as well on grayscale images. I chose not to generate additional training data because the network achieved the desired performance on the validation set without it.  Note that if the validation set is not representative of the data we would see in the "deployed environment" then we would probably want to focus more on augmenting the training set.

Here is an example of a traffic sign image before and after histogram equalization:

![alt text][imageHistogramEqualization]

As a last step, I normalized the image data so that gradient descent will converge faster.

#### 2.Final model architecture.



My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

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


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


