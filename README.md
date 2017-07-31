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
[imageNetworkDiagram]: ./writeup_files/network_diagram.png "Diagram of the network used to classifiy German street signs."
[imageConfusionMatrix]: ./writeup_files/confusion_matrix_norm_testset.png "Confusion matrix for the final model on the test set."
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

My final model consisted of two 5x5 convolution layers with valid padding, each of which was followed by a relu and 2x2 max pooling layer.  Outputs passed through two fully connected layers, the first of which was followed by a dropout layer (p = 0.5 during training), and finally onto a softmax layer.  The outputs of each convoldution/relu/maxpooling layer were flattened and concatenated together before input into the fully connected layer (following the idea of skip-layer connections or multi-scale features).  The model is shown in the following diagram:

![alt_text][imageNetworkDiagram]
 
#### 3-4. Description of model training and model selection approach.

The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I first started out by implementing dropout on the vanilla LeNet architecture because dropout generally works very well at preventing overfitting.  All models were trained with the Adam optomizer with an initial learning rate of 0.001 on batches containing 128 training examples.  After 60 training epochs, it reached 92 percent accuracy on the vlidation set, but seemed to be plateuing.  I then reduced the training rate to 0.0001 and trained for another 60 more epochs to reach 94.3 percent accuracy.

I then implemented a model including skip-layer connections (see diagram above) and re-trained the network from scratch following the same training procedure.  Doing so brought the accuracy up to 95.7 percent on the validation set after 150 training iterations.

Using the "evaluate" function, my final model results were:
* training set accuracy of 1.000.
* validation set accuracy of 0.957.
* test set accuracy of 0.941.

The accuracy on the training set is much higher than the accuracy on the validation set, indicating that the model is overfitting.  To further improve the model, additional dropout layers could be added after each convolution layer an l2 regularization on the model weights could be introduced into the loss function.

I also investigated the confusion matrix of the classifier on the test set to get a sense of whether the mistakes it made were inconsequential or definitely-not-ready-to-be-deployed-terrible.  The confusion matrix is shown below:


![alt_text][imageConfusionMatrix]

First, note that stop signs are not confused with anything else and visa versa (i.e. stop signs have good precion and good recall). A mildly concerning error is that "End of no passing" sings are frequently (15%) labeled as "No passing signs", but this is much better than the reverse (i.e. passing when the sign told you not to)! "Pedestirans" signs are also sometimes misclassified as "Right of way at the next intersection" signs, which is a bit troubling.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

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


