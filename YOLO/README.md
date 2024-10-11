This is a tutorial about the requirements to use YOLO in a Cluster.

The goal of this tutorial is to make it possible to evaluate the engineering drawings based on a custom dataset.

Specially focused on object detection of the components (symbols, labels, and specifiers).

# You Only Look Once (YOLO)

YOLO is an algorithm that detects and recognizes objects in pictures. YOLO employs convolutional neural networks (CNN) to detect objects, the algorithm requires only a single forward propagation through a neural network to detect objects. This means that prediction in the entire image is done in a single algorithm run. The CNN is used to predict various class probabilities and bounding boxes.

There are many variations of YOLO, some of the latest developed by [Ultralytics](https://github.com/ultralytics/ultralytics). In the latest version, the last letter refers to the size of the model (nano, small, medium, large, and extra large). The published results show that larger models may have better precision, and small models are faster. This trade-off makes the variation of the size of the model became important depending on the application.

## Create the Environment

The first step to compute YOLO in the cluster is to create the environment:

```
# Enter in the folder of your project
cd ~~~ 

# Check the environments available
conda env list

# Create a new environment
conda create --name yolo-env python=3.9

# Activate the environment to install packages
conda activate yolo-env

# If something goes wrong and you need to remove the environment
conda env remove -n yolo-env
```

You will need to match the version of PyTorch with the available or defined GPU. 
If there are updates you may install extra packages.

## Upload the YOLO to the Cluster

The second step is to download the YOLO and then upload it to the cluster.
I recommend doing that from the official developer [Ultralytics](https://github.com/ultralytics/ultralytics). 
This version is based on [PyTorch](https://pytorch.org/).

To download the YOLO directly in your project folder on the cluster using `wget` or you can copy `scp`.

```
# Example of how to copy the data in the Cluster
scp -r C:/Users/user/Desktop/yolo/ diclub:/home/user/yolo/
```

After having the YOLO saved, you can install the requirements of the model.

```
# Install the requirements
pip install ultralytics
```

### Organize Your Dataset

There are three ways to organize your personalized dataset.

The first way is using a different path for training and validation (works for YOLOv5 and YOLOv8).
```    
dataset/train/images
dataset/train/labels
dataset/valid/images
dataset/valid/labels
```

The second way is using a different path for training and validation (which works especially for YOLOv6).
```    
dataset/images/train
dataset/images/val
dataset/labels/train
dataset/labels/val
```

The third way is to place everything in the same place and call it by a `.txt` file (works only for YOLOv5 or YOLOv8).
```
dataset/train.txt
dataset/valid.txt
```

**I highly recommend that you use the third way, as it's going to be easier to change what you are using for training and testing, just by changing the `.txt` file.**

When you have decided how to organize your data, you will need to change how the model loads it.

This is going to be in the file that you use to call the script that loads your data.

In the `run-yolov8.py` there is (where you define how the data will be loaded):
```
dataset="../yolov8/data.yaml"
model.train(data=dataset)
```

This file is going to be in the data folder inside the YOLO model.

Depending on how you decided to organize the data the `data.yaml`, it will look like this:
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

To create a custom dataset with the goal of object detection it is necessary to use image labeling software.

I recommend using the [labelImg](https://github.com/heartexlabs/labelImg), it's based on Python, so it's light and easy to use.
LabelImg is a graphical image annotation tool written in Python.

After you download the algorithm you can run the `labelImg.py`

In the `data/predefined_classes.txt` you can define the classes that you are going to use. 
This will allocate the spaces in the memory, therefore the number in the annotation will follow this order.

Later on, the classes that you have created have to match with `data.yaml`

```
nc: 3
names: ['C00', 'C01', 'C02']
```

## RFI Project

As the images of the engineering drawings are big, it is necessary to perform a cropout.

To be able to train the YOLO the cropouts need to be standardized, this can be done by [this algorithm](https://gitlab.fbk.eu/dsip/dsip_dlresearch/rfi_acc/-/blob/main/Algorithms/Build%20Dataset/Slide_Window_Crop_Out.py).

If you need to perform a binary classification or use different inputs, it is possible to change all classes based on a single run using [this algorithm](https://gitlab.fbk.eu/dsip/dsip_dlresearch/rfi_acc/-/blob/main/Algorithms/Build%20Dataset/Change_the_Classes_for_the_Input_of_YOLO.py).

**OBS: Be careful with this algorithm since is going to change all your annotation. I recommend saving the output in a different path.**

# Final Comments

For YOLOv8 you can compute a Python file to run the experiments, using the SBATCH:

```
#!/bin/bash
#SBATCH --job-name=YOLO_exp
#SBATCH --output=YOLO_output.txt
#SBATCH --error=YOLO_error.txt
#SBATCH --partition=gpu-A40
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --cores-per-socket=4
#SBATCH --nice=0
eval "$(conda shell.bash hook)"
cd ~/yolov8/
conda activate yolo
python run-yolov8.py
```
Then, the `run-yolov8.py` Python file would be:

```
import re
import numpy as np
from ultralytics import YOLO
import time
import wandb
import os

wandb.init(project="YOLO_project", name=f"EXP1")

# Specify the save directory for training runs
name = '/storage/yolo_results/'
dataset="../yolov8/data.yaml"

# To compute multiple experiments
for runs in range(0,10): 
        start = time.time()
        # Load a model
        model = YOLO("yolov8n.pt")  # load a pre-trained model (recommended for training)

        # Use the model
        model.train(data=dataset, seed=int(runs), epochs=300, patience=50, val=True, name=name)
        metrics = model.val()  # evaluate model performance on the validation set

        end = time.time()
        time_s = end - start
        print(f'Time (s): {time_s} seconds')
```


---

Wrote by **Stefano Frizzo Stefenon**

Fondazione Bruno Kessler

Trento, Italy, June 06, 2024
