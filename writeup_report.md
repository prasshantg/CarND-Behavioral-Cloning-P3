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

[image1]: ./examples/test_center_image.jpg "Normal Image"
[image2]: ./examples/test_left_image.jpg "Left Camera Image"
[image3]: ./examples/test_right_image.jpg "Right Camera Image"
[image4]: ./examples/test_flipped_image.jpg "Flipped Image"
[image5]: ./examples/test_cropped_image.jpg "Cropped Image"
[image6]: ./examples/training_data.png "Training Data Histogram"
[image7]: ./examples/validation_data.png "Valiation Data Histogram"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 for video recorded with trained model on track one

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network is based on Nvidia network with 5x5 filters with stride 2 and 3x3 filter sizes and depths between 32 and 128 (model.py lines 231-235) 

https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. (model.py line 230)

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. (model.py line 238 and line 240)

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 253).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, forward and backward driving ...

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the https://github.com/commaai/research/blob/master/train_steering_model.py I thought this model might be appropriate because it was implemented for same purpose and verified.

Model performed good in terms of loss for training and validation dataset. No changes required for overfitting or underfitting.

The final step was to run the simulator to see how well the car was driving around track one. It worked fine for straight road but failed on turns. Then I switched to Nvidia model as it had more convolution layers. I added dropout layers to it to avoid overfitting.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 228-243) consisted of a convolution neural network with the following layers and layer sizes

Convolution layer1, kernels 24, kernel size 5x5, stride 2x2 
Activation layer1, ReLU 
Convolution layer2, kernels 36, kernel size 5x5, stride 2x2 
Activation layer2, ReLU 
Convolution layer3, kernels 48, kernel size 5x5, stride 2x2 
Activation layer3, ReLU 
Convolution layer4, kernels 64, kernel size 3x3, stride 1x1 
Activation layer4, ReLU 
Convolution layer5, kernels 64, kernel size 3x3, stride 1x1 
Activation layer5, ReLU 
Flatten, 1164 
Dense layer1, 100 
Dropout layer1, rate 0.5 
Dense layer2, 50 
Dropout layer2, rate 0.5 
Dense layer3, 10 
Dense layer4, 1 

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps each for forward and backward drive on track one using center lane driving. Here is an example image of center lane driving:

![Center Camera Image][image1]

Also, I recorded data only for turns skipping straight roads to get more data for turns as it was failing on some turns.

To augment the data sat, I also flipped images and angles thinking that this would give additional dataset. For example, here is an image that has then been flipped:

![Normal Image][image1]
![Flipped Image][image4]

After the collection process, I had 13289 number of data points. I replicated lines with angles greater than 0.1 and less than -0.1 to get more data for turns. It helped to get good distribution across all angles.

![Training data histogram][image6]
![Validation data histogram][image7]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I also used images from right and left cameras with correction of 0.2 in angle. Did not perform any experimentation with it as this worked fine.

I used this training data for training the model by cropping the image to remove unnecessary part from image. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![Normal Image][image1]
![Cropped Image][image5]
