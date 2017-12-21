# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This project is based on the deep neural networks and convolutional neural networks to clone driving behavior and drive the car around the track autonomously. In this project I have build the model using Keras. The model take image of the road as input and output a steering angle to an autonomous vehicle. The whole idea of this project is to predict steering angles that will allow the car to drive autonomously without using other features like lane line detection etc...

The data was collected using the simulator. Collected two types of data
1. The data was collected steering the car at the center of the road.
2. The data was collected letting the car to steer off the road and recover back to the center.

Udacity provided sample data, which was also used as part of my testing

### Files Submitted : 
* model.py (script used to create and train the model).
* drive.py (script to drive the car - feel free to modify this file).
* model.h5 (a trained Keras model).
* a report writeup file (either markdown or pdf).
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap).


### Rubric
---
[rubric points](https://review.udacity.com/#!/rubrics/432/view) 

### Project Goals
---
The goals / steps of this project are the following:
* #### Use the simulator to collect data of good driving behavior 
    * Collected data from the simualator. I tried to test the model with
      * training data provided by Udacity.
      * Collected the data from track1 by running the car mostly positioned in the center.
      * Collected the data from track1 by running the car get off the center and trying to recover.
    * For my final model I have used the data from Step 3 as it helped me to recover from hitting the lane lines and obstacles.
    * The simulation creats training data with IMG folder with all the images and driving_log.csv. The training data,  was read from driving_log.csv. Each line of driving_log.csv corresponded to one sample. Each sample contained left, and right camera images, as well as the current driving angle, throttle, brake, and speed data.

Example image from the center camera.
![center camera][center]
Example image from the right camera.
![left camera][right]
Example image from the left camera.
![right camera][left]


  
* ##### Design, train and validate a model that predicts a steering angle from image data
  #### Model Architecture and Training Strategy
    * ###### An appropriate model architecture has been employed
    My initial approach was to use [LeNet](http://yann.lecun.com/exdb/lenet/), but it was hard to have the car inside the street with three epochs (this model could be found [here](clone.py#L81-L94)). After this, I decided to try the [nVidia Autonomous Car Group](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) model, and the car drove the complete first track after just three training epochs (this model could be found [here](model.py#L108-L123)).

A model summary is as follows:

```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 43, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 20, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 8, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 6, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 4, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 8448)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           844900      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================

```

#### 2. Attempts to reduce overfitting in the model
To avoid overfitting I have used Dropout and I decided to keep the training epochs low to only three epochs. Intially I satrted with 10 epochs but after 3 epochs the Loss started increasing , same was observed for 5 epochs. 
In addition to that, I split my sample data into training and validation data. Using 80% as training and 20% as validation.

#### 3. Model parameter tuning

The model used an Adam optimizer.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with LeNet model and used the training data provided by Udacity. On the first track, the car went straight out of track and was looping around. To fix the problem I started preprocessing the data. Added `Lambda` layer to normalize the input images to zero means. This step improved a little bit but was not convincing. As second step in preprocessing added  `Cropping`, and 'Augmentation' the data by adding the same image flipped with a negative angle.  stil the model was not convincing.
After reading through the nVidia paper, I added the  [nVidial model: [nVidia Autonomous Car Group](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) with five convoulution layers and have a single output as it was required. This time the car almost made the whole track, but when it was about to make right turn towards the end of loop it went staright into lake.   To fix it  the left and right camera images where introduced with a correction factor of 0.125 was introduced which helped the car go back to the lane. I tried correction factors of 0.10, 0.15, 0.20, all these made the car to drift over the yello lines. 

#### 2. Final Model Architecture
