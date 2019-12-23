# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the training dataset (validation and testing dataset distribution could be found [here](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/tree/master/explorations)). It is a colored bar chart showing how the data is distributed across the different labels/sign names.
   #### Training DataSet Distribution
![Training DataSet Distribution](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/explorations/Training%20Explored.png) 

---
   #### Validation DataSet Distribution
![Validation DataSet Distribution](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/explorations/Validation%20Explored.png) 

---
   #### Test DataSet Distribution
![Test DataSet Distribution](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/explorations/Test%20Explored.png)

Here is a visualization of 15 random images from the training dataset for each 43 classes. Due to weather conditions, time of the day, lighting, and image orientation we can notice big differences in appearance between each image.

---
   #### Random images from the training dataset
![Random images from the training dataset](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/explorations/randomImagesPerLabel.png)

### Design and Test a Model Architecture

#### 1. After some experiments I decided to apply 1)Histogram Equalization 2)GrayScaling, and 3)Normalization in the same order of processing.
   **Histogram Equalization**
   Histogram Equalization is one of the fundamental tools in the image processing toolkit. It’s a technique for adjusting the pixel values in an image to enhance the contrast by making those intensities more equal across the board. Typically, the histogram of an image will have something close to a normal distribution, but equalization aims for a uniform distribution. Resources: [Link1](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html), [Link2](https://hackernoon.com/histogram-equalization-in-python-from-scratch-ebb9c8aa3f23)
   
   **Gray Scaling**
   Then I convert the images into grayscale, because color channels didn't seem to improve the model, but it tends to be just more complex at the end and slows down the training.
   
   **Normalization**
   Neural networks process inputs using small weight values, and inputs with large integer values can disrupt or slow down the learning process. As such it is good practice to normalize the pixel values so that each pixel value has a value between 0 and 1. Resource: [Link1](https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/)
   #### Sample image from training dataset
![Sample image from training dataset](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/explorations/TrainingImagePreprocessing.png)

   #### Sample image from validation dataset
![Sample image from validation dataset](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/explorations/ValidationImagePreprocessing.png)

   #### Sample image from test dataset
![Sample image from test dataset](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/explorations/TestImagePreprocessing.png)

I decided not to generate additional data for now because the accuracy obtained on the test dataset is quite statifying (96%). As a future improvement, I could manually look at the individual images for which the model makes a wrong judgement. So that, I then can augment the dataset accordingly.

#### 2. I decided to use a deep neural network classifier as a model, which was inspired by LeNet-5. It is composed of 7 layers: 4 convolutional layers for feature extraction and 3 fully connected layer as a classifier.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 image  							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 15x15x32 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 13x13x64			|
| RELU     |												|
|Max pooling|	2x2 stride, outputs 7x7x64|
|Dropout|	Keep probability 0.5|
|Convolution| 3x3	1x1 stride, valid padding, outputs 5x5x128|
|RELU	||
|Max pooling|	2x2 stride, outputs 3x3x128|
|Convolution| 3x3	1x1 stride, valid padding, outputs 1x1x256|
|RELU	||
|Max pooling|	2x2 stride, outputs 1x1x256|
|Dropout|	Keep probability 0.5|
|Fully connected|	Inputs 3392 (7x7x64 + 256), outputs 1024|
|RELU	||
|Dropout|	Keep probability 0.5|
|Fully connected|	Inputs 1024, outputs 512|
|RELU	||
|Dropout|	Keep probability 0.5|
|Fully connected	|Inputs 512, outputs 43|
|Softmax	||
 
Additional maxpooling, dropout, and L2-regularization techniques have been used to reduce overfitting.

#### 3. To train the model, I used the AdamOptimizer with a learning rate of 0.001. The epochs used was 10 while the batch size was 128. Other important parameters I tuned were the probabilty of keeping neurons for dropout, and the beta parameter for L2-regularization.

#### 4. 
I considered a well known architecture, LeNet-5 from Yann Le Cun, because of simplicity of implementation. It was a good starting point but it suffered from under fitting. The accuracy on both sets, training and validation, was very low (around 50%).

So I modified the network and added more convolutional and fully-connected layers, and I used dropout as well. I've also played with the layer parameters (filter size, strides, padding, max pooling strides, fully connected outputs size) to find which changes give the best results.

But after a few runs with this new architecture I noticed that the model still tended to overfit to the original training dataset. With the best run, I obtained a 5% gap in accuracy between the training and validation sets. Hence, I added dropout between the convolution layers, and I also used L2-regularization.

With these 2 techniques, the gap between training and validation accuracies has been reduced to 3.3%. Furthermore, the accuracy increased for the validation and testing datasets. The performance of the model has been increased and generalizes better to unseen data which is the ultimate goal.

Hence, my final model results were:

training set accuracy of 0.999
validation set accuracy of 0.978
test set accuracy of 0.961
Find below the learning curves. Both training and validation curves are converging toward 100%, however a gap between the two is still remaining. Future improvements could include techniques such as data augmentation to reduce overfitting even more.

   #### Neural network learning curve
![Neural network learning curve](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/explorations/learningCurve.png)

### Test a Model on New Images

#### 1. Chose ten German traffic signs found on the web. Before processing I made sure the the shape of the images is (32,32,3).

Here are ten German traffic signs that I found on the web:

![image12](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/newImages/1.png) ![image13](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/newImages/2.png) ![image14](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/newImages/3.png) ![image15](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/newImages/4.png) ![image16](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/newImages/5.png) ![image17](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/newImages/6.png) ![image18](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/newImages/7.png) ![image19](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/newImages/8.png) ![image20](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/newImages/9.png) ![image21](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/newImages/10.png)

#### 2. The model's predictions on these new traffic signs and compare the results to predicting on the test set. 

Here are the results of the prediction:

| Image			        						|     Prediction	        					| 
|:---------------------------------------------:|:---------------------------------------------:| 
| Yield      									| Yield   										| 
| No passing for vehicles over 3.5 metric tons	| Ahead only									|
| Speed limit (20km/h)							| Speed limit (80km/h)							|
| General caution	      						| General caution				 				|
| Bumpy road									| Bumpy road     								|
| No vehicles									| Speed limit (80km/h)							|
| Priority road									| Priority road     							|
| Ahead only									| Ahead only     								|
| Vehicles over 3.5 metric tons prohibited		| Vehicles over 3.5 metric tons prohibited		|
| Right-of-way at the next intersection			| Right-of-way at the next intersection			|
| Go straight or left							| Go straight or left							|
| Speed limit (30km/h)							| Speed limit (30km/h)							|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Predicting on each of the new images by looking at the softmax probabilities for each prediction. The top 5 softmax probabilities for each image along with the sign type of each probability.

The model is relatively sure that it predicts the right sign (probabilities range from 98% to 100%). However for the three of the images model makes a wrong prediction, we can notice the model is not very sure about it (75%).

![image11](https://github.com/snehalmparmar/CarND-Traffic-Sign-Classifier-Project/blob/master/explorations/top5Probabilities.png)
