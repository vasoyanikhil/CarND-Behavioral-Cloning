# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/architecture.PNG "Model Visualization"
[image8]: ./examples/ NVIDIA.jpg      "Model layers"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/left_2019_12_12_10_01_47_583.jpg "Recovery Image"
[image4]: ./examples/right_2019_12_12_10_04_34_734.jpg "Recovery Image"
[image5]: ./examples/right_2019_12_12_10_04_20_645.jpg "Recovery Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I decided to test the model provided by NVIDIA as suggested by Udacity. The model architecture is described by NVIDIA here. As an input this model takes in image of the shape (60,266,3) but our dashboard images/training images are of size (160,320,3). 
I decided to keep the architecture of the remaining model same but instead feed an image of different input shape which I will discuss later. 

Here is a NVIDIA of the architecture
![alt text][image8]

#### 2. Attempts to reduce over-fitting in the model

The model was trained and validated on different data sets to ensure that the model was not over-fitting (code line 107-115). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 117).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to not over fitting and under fitting and validation loss and training loss should be very less.

My first step was to use a convolution neural network model similar to the NVIDIA I thought this model might be appropriate because already they used real car. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
#### Loading Data

* I am using OpenCV to load the images, by default the images are read by OpenCV in BGR format but we need to convert to RGB as in drive.py it is processed in RGB format.

* Since we have a steering angle associated with three images we introduce a correction factor for left and right images since the steering angle is captured by the center angle.
* I decided to introduce a correction factor of 0.2
* For the left images I increase the steering angle by 0.2 and for the right images I decrease the steering angle by 0.2

#### Preprocessing

* I decided to crop the images and after i did normalization of the images because after cropping the images less process i need to do and model process also fast. after I pass normalization images to model then model.h5 generated.

* The final step was to run the simulator to see how well the car was driving around track one. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Final Model Architecture

* I made a little changes to the original NVIDIA architecture, my final architecture looks like in the image below.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process
* I took total 4 laps with among that one recovery lap and 3 normal laps.Here is an example image of center lane driving:

  ![alt text][image2]
  
  I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to recover if drive in off side. These images show what a recovery looks like starting from ... :

  ![alt text][image3]
  ![alt text][image4]
  ![alt text][image5]
  
  so I was satisfied with the data and decided to move on.
  
* I decided to split the dataset into training and validation set using sklearn preprocessing library.

* I decided to keep 20% of the data in Validation Set and remaining in Training Set

* I am using generator to generate the data so as to avoid loading all the images in the memory and instead generate it at the run time in batches of 32. Even Augmented images are generated inside the generators.



* I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.I used an loss for 'mse' After I generated model.h5.


