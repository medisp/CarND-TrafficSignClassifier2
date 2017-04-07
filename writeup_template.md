#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---
###Writeup /

####1. 
link to my project: https://github.com/medisp/CarND-TrafficSignClassifier2/blob/master/Traffic_Sign_Classifier.ipynb

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy and python to calculate data set summary:

Image Shape: (32, 32, 3)

Training Set:   34799 samples
Validation Set: 4410 samples
Test Set:       12630 samples

####2. Include an exploratory visualization of the dataset.

Please find these visualizations of the data:
![alt tag](https://github.com/medisp/CarND-TrafficSignClassifier2/blob/master/vis1.PNG)
![alt tag](https://github.com/medisp/CarND-TrafficSignClassifier2/blob/master/vis2.PNG)
###Design and Test a Model Architecture

### Question 1
Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

Answer:
Technique:Histogram Equalization Reason: Images often lacked sharpness, and adequate differences in brightness and contrast so this enhance the contrance differences.
Technique: crop,rescale images Reason: there is often backgrounds around signs that need not be considered, so resized images to 26 pixes.
Technique: Affine Transformations Reason: In order to jitter out images so essential features in hidden layers get picked up, Affine transformation shuffles up image orientation with transformation, rotation and shearing while preserving parallel lines.
Technique:Data Generation. Reason: I generated additional data with this processing to make sure each image has multiple, slightly varied versions of them. This gave rise to total data size of 661181 images to help the network learn better how to classify, as opposed to the original size of 34799 training samples.

Here is an example of the processing steps of histogram normalization, brightness augmentation, and affine transformations.
![alt tag](https://github.com/medisp/CarND-TrafficSignClassifier2/blob/master/vis3.PNG)

 
## Model Architecture

### Question 2

Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

### Answer:

The CNN Architecture is based heavily on the LeNet Architecturefrom Udacity Self Driving Car Term 1 lab.
Input: 26 x 26 x 3
Layer 1: Convolutional Bayer 5 x 5 with 3 depth input, 32 depth output, Valid Padding 
         Relu Activation
         MaxPool with stride of 2, Valid Padding
         
Layer 2: Convolutional Layer 4 x 4 with depth input of 32, 64 depth output, Valid Padding
         Relu Activation
         MaxPool with stride of 2, Valid Padding

Flatten Layer: Input of 1024 and output of 120, Relu Activation

Fully Connected 1: input 120 and output 84, Relu Activation

Fully Connected 2: input of 84 and output 43 to 43 classes

Dropouts below did not work and somehow caused the model to give 0.054 accuracy for both training and validation without improvements.

Epochs cut short as soon as model achieved accuracy higher than 0.935 on validation set.


## Question 3
Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

### Answer:
Batch Size: 128
Epochs: 40 with a stop in the training function when validation accuracy exceeded 0.935. 

This was to fix an incident in which the model reaches 0.94+ but in the last epoch, number 40: dropped down to 0.929.
I initialized the weights with truncated norm with mean = 0 standard deviation = 0.1

Learn Rate: 0.0007 with a process to decrease it by multiplying by 0.75 every 5 epochs.

I used the Adam optimizer with softmax cross entropy calculations for measuring probabilities.

My final Accuracy measurements:

Training Accuracy = 0.970

Validation Accuracy = 0.937

Test Accuracy = 0.916

## Question 4

Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

### Answer:
1) I have largely worked in iterative methods to slowly get the accuracy of the model up. I initially used all of the same settings from LeNet Architecture, except changing the output classes to 43 to match this dataset. Validation and Training Accuracy ~ 0.7

2) The accuracy was drastically low, and I decided to create more data to see if atleast the training accuracy can improve. I started seeing improvements but saw overfitting of training because the training accuracy was high but the validation accuracy was low.  Training Accuracy ~ 0.95 Validation Accuracy ~ 0.85
 
3) I began to modify the network architecture to see if features can better be assessed from this type of data. I modified the learning rate by decreasing it and having a process slowly decrease the learnrate as the number of epochs increased. Validation Accuracy ~ 0.91 . I benchmarked different convolution net sizes and layer choices while creating more data.

4) Finally settling with current settings and stopping the learning process after a cutoff accuracy  was what I have now. I plan to disable this stopping feature and use inception models to get to accuracy 0.95.

5) I wanted to use Dropouts but had issues when introducing dropouts. my test and train accuracies plummetted and no learning was happening with epochs. So I disabled them and will experiment in the future to prevent training overfit.

I believe a convolutional layer was the right approach due to how low level features and higher level generalizations is exactly how to best identify and classify signs. This is because lower level features like likes can then come together to form specific shapes such as triangles or left turn arrows etc.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
Here are the new images I found on the web:

![alt tag](https://github.com/medisp/CarND-TrafficSignClassifier2/blob/master/vis4.png)
![alt tag](https://github.com/medisp/CarND-TrafficSignClassifier2/blob/master/vis5.png)

### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set

The model had accuracy of 80% when predicting the right class when using new images from the link provided: https://www.embedded-vision.com/sites/default/files/technical-articles/CadenceCNN/Figure15.jpg
This was an image represnting the different images in this dataset that I pulled and parsed into new images.

These images may be particularly difficult due to the weather and lighting on them. For instance, I thought the 30 km/hr speed limit sign would be most difficult for the model to classify.

![alt tag](https://github.com/medisp/CarND-TrafficSignClassifier2/blob/master/vis6.png)

## Question 3
Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The softmax graphs are an intuitive way to assess how the model was classifying the images. Some images showed 100% accuracy without predicting other softmax values which is good. for the speed limit 20 image above, I believe the preprocessing of the image or clipping of the image caused some artifacts that confused the model. Its second best choice was the right answer, it even had some probability for the speed limit 120km/h sign which is interesting and intuitive because the convolutional layers have both use the 2 to identify those two signs. 


