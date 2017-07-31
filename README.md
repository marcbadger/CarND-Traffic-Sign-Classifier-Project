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
[imageOwnExamples]: ./writeup_files/own_example_collage.png "German street signs from the wild."
[imageExamplesTop5Predictions]: ./writeup_files/top_5_probs_own_examples.png "Top 5 predictions for each new example image."

# Writeup

The following is a writeup of my work on the Traffic Sign Classifier project for the Self Driving Car Nanodegree program.  Here is a link to my [project code](https://github.com/marcbadger/CarND-Traffic-Sign-Classifier-Project/blob/master/Badger_2017_Traffic_Sign_Classifier.ipynb), and [an HTML version](https://github.com/marcbadger/CarND-Traffic-Sign-Classifier-Project/blob/master/Badger_2017_Traffic_Sign_Classifier.html).

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

#### 1. "Wild" German traffic signs from Google Streetview

Here are ten German traffic signs that I found using Google Streetview:

![alt text][imageOwnExamples]

I expect most of these signs to be easy as they are higher quality than many of those in the training data. The "Yield" sign (Class 13 above) might be slightly more difficult because it is at a slight angle rather than from straight on. (No augmentation of the training data was performed. In particular, no affine transformations were performed.) Other difficult signs could be the "General caution" sign with the "No entry" sign in the background, the "No entry" sign that is partially occluded by a "General caution" sign and the "Ahead only" sign that is partially occluded by the "No entry" sign. I have also included an impossible sign "No Roundabout" which isn't even in the training dataset.

#### 2. Discussion of the model's predictions and performance on the new traffic signs

The model achieved 90 percent accuracy on these new examples. However, given that one of them was "impossible", the accuracy on the signs that were actually in the training set was 100 percent.  But given that we only tested on 9 new images, the probability of getting all 9 right assuming the true accuracy was 94.1 percent (as it was on the test set) is (0.941^9) = 0.58, meaning that we are likely to have gotten all 9 right even if the model's performance really was 94.1 percent.  In fact given only 9 samples, the confidence interval for the accuracy (assuming 9 independent Bernoulli trials) is [0.82, 1.00] using [the Wilson score interval](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval).  If you're looking to test how your model does in some new context, you should collect a much larger test set than just 10 images.  See the next section for visualizations of the model's predictions on each sign.

#### 3. Assessing model certainty

The code for making predictions on my final model is located in the "Analyzing Performance" section of the Ipython notebook, and the image below shows the top 5 probabilities for each image.

![alt text][imageExamplesTop5Predictions]

Looking at the top-5 probabilities for each of the new signs, the network (which achieved 95% accuracy on the validation set) got every new example correct except the "No Roundabout" sign, which isn't even represented in the training data. It's a bit concerning that it classified it as nearly 100% "Speed limit 50km/h". If you look at the probabilities below, it is the only one with two entries with greater than 0.00e+00 probability. Ideally, the classifier would be confident when it is correct and unconfident when it is incorrect. The classifier also seems to have a lot of trouble with speed limit signs. It's second best guess for most signs is some version of a speed limit sign. I bet that most of the speed limit signs have relatively low precision because the network is giving a lot of false positivies.  See below for the top 5 probabilities for each of the 10 new signs:

TopKV2(values=array([[  1.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00],
       [  1.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00],
       [  1.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00],
       [  1.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00],
       [  1.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00],
       [  1.00e+00,   6.61e-07,   0.00e+00,   0.00e+00,   0.00e+00],
       [  1.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00],
       [  1.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00],
       [  1.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00],
       [  1.00e+00,   0.00e+00,   0.00e+00,   0.00e+00,   0.00e+00]], dtype=float32), indices=array([[17,  0,  1,  2,  3],
       [12,  0,  1,  2,  3],
       [18,  0,  1,  2,  3],
       [37,  0,  1,  2,  3],
       [13,  0,  1,  2,  3],
       [ 2,  9,  0,  1,  3],
       [33,  0,  1,  2,  3],
       [38,  0,  1,  2,  3],
       [17,  0,  1,  2,  3],
       [35,  0,  1,  2,  3]]))

