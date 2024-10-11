This is a tutorial about the requirements to use YOLO in a Cluster.

The goal of this tutorial is to make it possible to evaluate the engineering drawings based on a custom dataset.

Specially focused on object detection of the components (symbols, labels, and specifiers).

# You Only Look Once (YOLO)

YOLO is an algorithm that detects and recognizes objects in pictures. YOLO employs convolutional neural networks (CNN) to detect objects, the algorithm requires only a single forward propagation through a neural network to detect objects. This means that prediction in the entire image is done in a single algorithm run. The CNN is used to predict various class probabilities and bounding boxes.

There are many variations of YOLO, some of the latest developed by [Ultralytics](https://github.com/ultralytics/ultralytics). In the latest version, the last letter refers to the size of the model (nano, small, median, large, and extra large). The published results shows that lager models may have better precision, and small models are faster. This trade off makes the variation of the size of the model became important depending on the application.

# Compute YOLO in the Cluster

## Create the Environment

The first step to compute YOLO in the cluster is to create the environment:

```
# Enter in the folder of your project
cd ~~~ 

# Check the environments available
conda env list

# Create a new environment
conda create --name yolo-env python=3.9

# Activate the environment to install packges
conda activate yolo-env

# Install the requirements
pip install ultralytics

# If something goes wrong and you need to remove the environmen
conda env remove -n yolo-env
```

You will need to match the version of PyTorch with the available or defined GPU. 
If there are updates you may install extra packages.

## Upload the YOLO to the Cluster

The second step is to download the YOLOv5 and then upload it to the cluster.
I recommend doing that from the official developer [Ultralytics](https://github.com/ultralytics/yolov5/). 
This version is based on [PyTorch](https://pytorch.org/) and works in the cluster and in [Google Colab](https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb).

To upload the YOLOv5 in your project folder you can follow this example (using your path):
```
# Example of how to upload the YOLOv5 in the Cluster
scp -r C:/Users/user/Desktop/yolov5/ diclub:/home/sfrizzostefenon/yolo/
```

## Test the YOLOv5 in the Cluster

To test the YOLOv5 in the cluster you need to upload your dataset.
You can do that by following this example (using your path):
```
# Example of how to upload your dataset in the Cluster
scp -r C:/Users/user/Desktop/dataset/ diclub:/home/sfrizzostefenon/dataset/
```

### Organize Your Dataset

There are tree ways to organize your personalized dataset.

The first way is using a different path for training and validation (works for YOLOv5 and YOLOv7).
```    
dataset/train/images
dataset/train/labels
dataset/valid/images
dataset/valid/labels
```

The second way is using a different path for training and validation (works specially for YOLOv6).
```    
dataset/images/train
dataset/images/val
dataset/labels/train
dataset/labels/val
```

The third way is to place everything in the same place and call it by a `.txt` file (works only for YOLOv5).
```
dataset/train.txt
dataset/valid.txt
```

**I highly recommend that you use the second way, as it's going to be easier to change what you are using for training and testing.**

In the `train.txt` you will have the path of all your pictures, one by one like:
```
diclub:/home/user/dataset/RFI_640_110c10_0.jpg
```

When you have decided how to organize your data, you will need to change how the model loads it.

This is going to be in the file that you use to call the script that loads your data.

In the `train.py` there is (where you define how the data will be loaded):
```
parser.add_argument('--data', type=str, default=ROOT / 'data/mydata.yaml', help='dataset.yaml path') 
```

This file is going to be in the data folder inside the YOLO model.

Depending on how you decided to organize the data the `mydata.yaml` will look like this:
```
path: ../dataset
train: train/images
val: valid/images
```
or
```
path: ../dataset
train: train.txt
val: val.txt
```

**If you use this structure the model will load the labels automatically based on their names.**

OBS: Here the test is optional because it will be performed after training.
```
# test images (optional)
```

# Create a Custom Dataset

To create a custom dataset with the goal of object detection it is necessary to use an image labeling algorithm or software.

I recommend to use the [labelImg](https://github.com/heartexlabs/labelImg), it's based on Python, so it's light and easy to use.
LabelImg is a graphical image annotation tool written in Python.

After you download the algorithm you can run the `labelImg.py`

In the `data/predefined_classes.txt` you can define the classes that you are going to use. 
This will allocate the spaces in the memory, therefore the number in the annotation will follow this order.

Later on, the classes that you have created have to match with `mydata.yaml`, [like this](https://gitlab.fbk.eu/dsip/dsip_dlresearch/rfi_acc/-/blob/main/Algorithms/YOLO/Setup%20for%20the%20Cluster/rfi_my.yaml), for the [RFI pallet](https://gitlab.fbk.eu/dsip/dsip_dlresearch/rfi_acc/-/blob/main/Extra%20Information/Componenti_RFI.xlsx) components.

## RFI Project

As the images of the engineering drawings are big, it is necessary to perform a cropout.

To be able to train the YOLO the crop outs need to be standardized, this can be done by [this algorithm](https://gitlab.fbk.eu/dsip/dsip_dlresearch/rfi_acc/-/blob/main/Algorithms/Build%20Dataset/Slide_Window_Crop_Out.py).

As the crop outs may divide an element, it's possible to build an overlapping dataset by including the required shift [available here](https://gitlab.fbk.eu/dsip/dsip_dlresearch/rfi_acc/-/blob/main/Algorithms/Build%20Dataset/Slide_Window_Crop_Out.py).

If you need to perform a binary classification or use different inputs, it is possible to change all classes based on a single run using [this algorithm](https://gitlab.fbk.eu/dsip/dsip_dlresearch/rfi_acc/-/blob/main/Algorithms/Build%20Dataset/Change_the_Classes_for_the_Input_of_YOLO.py).

**OBS: Be careful with this algorithm since is going to change all your annotation. I recommend saving the output in a different path.**

# Final comments



---

Wrote by **Stefano Frizzo Stefenon**

Fondazione Bruno Kessler

Trento, Italy, June 06, 2024
