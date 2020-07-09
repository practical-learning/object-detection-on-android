# object-detection-on-android

Object detection has many use cases in computer vision applications. Like tracking objects in real time , face detection and recognition are some of the examples where object detection is used.

In this tutorial you will learn how to train object detection model using tensorflow on your own custom data.

What you will learn:

1. Data Annotation : Here you will learn how to label the training data on which you will train the model.

2. Generating TFRecord : As we are using Tensorflow library for training the model we can't feed png or jpg images to the training script. First we have to convert them to TFRecord. TFRecord is the TensorFlow's own data storage format.

3. Mode Training

4. Run trained model on Android

First of all collect the images on which you want to train the model. Here I am going to train the object detection model on the dataset of apples. And trained model will detect good and damaged apples . Create one folder named "Dataset" and in this folder create two folders named train and test. Train folder will have the data which we will be using for training and test folder will have all the data which we will use for validating the model. General rule for dividing the data into train and test is 80-20 means 80% of the total images will be used for training and 20% of the total images will be used for testing.

All the images for training and testing will be in folder named images under the train and test folder. If you want to use same dataset which I am using you can download the dataset from this link https://github.com/OlafenwaMoses/AppleDetection/releases/
