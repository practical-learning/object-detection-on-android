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

If you have your own dataset you can use that also.

Next we will label the dataset. To label the dataset we will label image tool follow the installation istruction here https://github.com/tzutalin/labelImg

After installing Label Image tool open the tool and from the view menu option enable Auto save option , as this will save a lot of time while labelling the images.

Select open dir option and open the images for train folder . First we will be labelling our training images and then we will label test images. After that select "change save dir" option and make one new folder inside train folder named annotations and open it. This folder will save all the labelled files.

To start labelling press "W" key or select "create rectangle" option. It will highlight the cursor with two lines. Mark the desired object which you want to detect. After that enter the name of the object, this will be the label of this object. Same way mark all the other objects in image. After marking all the objects press "D" key and this will save the image and open next image.

If you open the annotations folder in train folder you will see the xml files with same name as the name of image. Same way label all the images in train and test folder.

Next we will setup Tensorflow Object detection API.

Make sure you have Python 3 installed on your system. As tensorflow 2.0 version have some compatibilty issues with Object Detection API so we will use Tensorflow v1.15. 
Install Tensorflow using this command ```pip3 install tensorflow==1.15```

Install these dependencies :

```
pip3 install --user Cython
pip3 install --user contextlib2
pip3 install --user pillow
pip3 install --user lxml
pip3 install --user jupyter
pip3 install --user matplotlib
pip3 install --user tf_slim
pip3 install --user pycocotools
```

Install Protobuf 

For Mac

```brew install protobuf```

Clone the object detection models repository

```git clone https://github.com/tensorflow/models.git```

Compile Protobuf 

```# From models/research/
protoc object_detection/protos/*.proto --python_out=.
```

Change PYTHONPATH variable:

Add this to the bottom of .bash_profile file

export PYTHONPATH=$PYTHONPATH:"<path to the models folder>/research":"<path to the models folder>/research/slim"
  
To test the installation of object detection API run below command from models/research folder

```python3 object_detection/builders/model_builder_tf1_test.py```


