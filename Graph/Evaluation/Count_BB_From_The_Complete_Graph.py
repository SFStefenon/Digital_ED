
''' 
Algoritm wrote by Stefano Frizzo Stefenon

Fondazione Bruno Kessler
Trento, May 15, 2023.

'''

# Import libs
import cv2
import glob
import numpy as np
from numpy import asarray
from PIL import Image 
import pandas as pd
import math

# Main path
main = str('C:/Users/ffriz/Dropbox/FBK/')
path_main = str('../PrePro/Complete_Graph/')

show_images = False
save_images = True
use_othogonal = False
connect_symbols = True
connect_labels = False

###############################################################################
###############################################################################
# Functions
###############################################################################
###############################################################################
# Function to read the text file and create an array
def replace_n(line):
  return line.replace('\n','').split(' ')

# Function to convert the bounding boxes (bb) coordinates to pixels coord.
def conv_bb(line):
  # Here it is step + the BB
  x_1 = int(float(line[1]))*640 + ((int((float(line[1])-int(float(line[1]))) * 640)) - int(((float(line[3])-int(float(line[3]))) * 640)/2))
  y_1 = int(float(line[2]))*640 + ((int((float(line[2])-int(float(line[2]))) * 640)) - int(((float(line[4])-int(float(line[4]))) * 640)/2))
  x_2 = int(float(line[3]))*640 + ((int((float(line[1])-int(float(line[1]))) * 640)) + int(((float(line[3])-int(float(line[3]))) * 640)/2))
  y_2 = int(float(line[4]))*640 + ((int((float(line[2])-int(float(line[2]))) * 640)) + int(((float(line[4])-int(float(line[4]))) * 640)/2))
  coord_bb = [[x_1, y_1], [x_2, y_2]]
  return coord_bb

# Funcion to have the average of the bb
def m_bb(line):
  coord_bb_m = [int((c_bb[0][0] + c_bb[1][0])/2), int((c_bb[0][1] + c_bb[1][1])/2)]
  return coord_bb_m

# Function to use the bb / bboxA = [0, 0, 2, 2]
def bb_intersection_over_union(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou # https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

# Over bounding boxes (o_bb)
def o_bb(line, thr_bb_x, thr_bb_y):
  coord_bb_o = [[c_bb[0][0]-thr_bb_x, c_bb[0][1]-thr_bb_y], [c_bb[1][0]+thr_bb_x, c_bb[1][1]+thr_bb_y]]
  return coord_bb_o
 
# Fonts for IDs
font = cv2.FONT_HERSHEY_SIMPLEX

###############################################################################
###############################################################################
# Load all data
###############################################################################
###############################################################################
# Load image data
path = path_main + str('*.jpg')
image_names = glob.glob(path, recursive=True)

# GLoad all label names for the txt files
label_names: list = []
for image_name in image_names:
    label = image_name.replace('images','labels', 1).replace('.jpg','.txt', 1)
    label_names.append(label)

# Load main file for the segments
main_segments = '../PrePro/Complete_Graph/I016II_SDO_110b_all_seg.txt'
with open(main_segments,'r') as f_seg:
    all_segments = f_seg.readlines()

# Create a vector of segments
new_elements_f: list = []
for segm in all_segments:
    segm = segm.replace('\n','').split(' ')
    segments = [[int(segm[0]), int(segm[1])], [int(segm[2]), int(segm[3])]]
    new_elements_f.append(segments)
    # print(segments)

# Create a vector of BBs
BBs: list = []
for idx, label in enumerate(label_names): 
    with open(label,'r') as f:      # Load all images
        label = f.readlines()
    for line in label:
        line = replace_n(line)  # Change str information to vector
        c_bb = conv_bb(line)    # Convert the coordinates of the BB
        BBs.append(c_bb)     

iS_obj=0
symbol=0
if connect_symbols == True:
    for idx, label in enumerate(label_names):
        with open(label,'r') as f:      # Load all images
            label = f.readlines()
        thr_s_x = 5
        thr_s_y = 5
        ID_obj: int = 0
        symbol: int = 0
        ID_seg: int = 20000
        edges_class_all = []
        for line in label: 
            line = replace_n(line)  # Change str information to vector
            c_bb = conv_bb(line)    # Convert the coordinates of the BB  
            # Definition of the considered objects to connect to the segments
            # Consireding symbols and dots
            if int(line[0]) >= 76:
                symbol +=1


# int(line[0]) >= 76:

# int(line[0]) <= 61:

# (int(line[0]) >= 74 and int(line[0]) <= 75):

# (int(line[0]) >= 62 and int(line[0]) <= 73): 
    

    