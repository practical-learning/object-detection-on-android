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

Edit xml_to_csv.py file , set the path of annotations folder in train folder and name of the csv file to "train.csv" save the file and run it from terminal using python3. This command will generate csv file from annotated xml files. After generating train.csv again edit xml_to_csv.py file and set the path to annotations folder in test folder and name the csv file as test.csv. Save the file and run it using python3 and this will generate test.csv file.

Edit generate_tfrecord.py file , goto function named class_text_to_int and change row label to the label which you use while labelling the data. As I have used two labels apple and damaged_apple so I will add them here. Int valueswhich this function will return will be 1 for first label , 2 for second label and so on. As I have only two labels "apple" and "damaged_apple" so this function will return 1 and 2.
Save the changes in generate_tfrecord.py file and run it from terminal with Python3 with below command

```
python3 generate_tfrecord.py --csv_input=<path of train.csv file>  --output_path=<path of the output directory>/train.record --image_dir=<path to the train images folder>
``` 

Replace the <path ...> with the correct paths.

Above command will generate train.record file from training images and it take information about the bounding boxes from csv file which we pass in argument.

Same way generate test.record file.

```
python3 generate_tfrecord.py --csv_input=<path of test.csv file>  --output_path=<path of the output directory>/test.record --image_dir=<path to the test images folder>
```

To train the model we will use pretrained model as our initial checkpoint.This way we are not training our model from scratch and it will take less time for our new model to get trained. Create new folder named pretrained_model and in this folder download the pretrained ssd mobilenet v2 model from this link https://github.com/practical-learning/object-detection-on-android/releases/download/v1.0/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
and extract the downloaded file.

Create label_map.pbtxt file in dataset folder. This file will have the mapping of our labels with int ids. Check the content of label_map.pbtxt file from this link https://github.com/practical-learning/object-detection-on-android/blob/master/label_map.pbtxt  change it according to your labels.

Next open the pipeline.config file in the pre-trained SSD model folder which you have downloaded in previous step. In this file first change the num_classes variable value to the numbers of classes or number of labels in your dataset. Dataset which I am using have only 2 labels "apple" and "damaged_apple" so num of classes are 2 , change it according to your datatset.

Next change the finetune checkpoint path to the path of the model checkpoint file in pretrained SSD model folder.

In the train input reader change the label map path to the label_map.pbtxt file in dataset folder and change the input path to the path of train.record file in dataset folder.

In the eval input reader change the label map path to lable_map.pbtxt file which is same as we used above and change the input path to test.record file in dataset folder.
Last in the quantization chnage the delay value to some lower number like I used 4800.
After making all the chnages save the file.

To start training open the terminal in research folder or change the path in terminal to research folder and run this command. 

```
python3 object_detection/legacy/train.py --logtostderr --train_dir=<training fodler path>  --pipeline_config_path=<path of pipeline.config file>
```
For training folder path in above command , make new folder where you want to save your trained model.
