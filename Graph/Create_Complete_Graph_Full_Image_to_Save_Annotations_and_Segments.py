
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
save_images = False
use_othogonal = False
connect_symbols = False
connect_labels = True

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

###############################################################################
###############################################################################
# Join orthogonal segments and create the whole picture
###############################################################################
############################################################################### 

lines_updated = new_elements_f
if use_othogonal == True:
    # Iterations to cluster segments
    for k in range(0,2):
        new_elements_t1=[]
        thr = 10
        for i in range(0,len(lines_updated)):    
          for j in range(0,len(lines_updated)):
            # If it is not the same segment 
            if (i != j):
              # The same starting position of the segments are close
              d_x1 = abs(lines_updated[i][0][0] - lines_updated[j][0][0])
              d_y1 = abs(lines_updated[i][0][1] - lines_updated[j][0][1]) 
              if (d_x1<thr and d_y1<thr): 
                n1 = [[lines_updated[j][1][0], lines_updated[j][1][1]], [lines_updated[i][1][0], lines_updated[i][1][1]]]
                new_elements_t1.append(n1)
              # Oposite position of the segments are close
              d_x2 = abs(lines_updated[i][1][0] - lines_updated[j][0][0])
              d_y2 = abs(lines_updated[i][1][1] - lines_updated[j][0][1])
              if (d_x2<thr and d_y2<thr):
                n2 = [[lines_updated[j][1][0], lines_updated[j][1][1]], [lines_updated[i][0][0], lines_updated[i][0][1]]]
                new_elements_t1.append(n2)        
        lines_updated = new_elements_t1
        print('Iteration: ',k+1,' Total Segments: ',len(new_elements_f),' New Segments: ',len(lines_updated))
    # Update the original array of segments
    lines_updated = np.append(new_elements_f, lines_updated)
    new_elements_f = lines_updated.reshape(-1,2,2)
    print('Finished segment updates: ', len(new_elements_f))

###############################################################################
###############################################################################
# Create images with IDs from PHT + YOLO
###############################################################################
###############################################################################

if save_images == True:
    output_path_ID = path_main + str('Processing/IDs')
    for idx, label in enumerate(label_names):
        img0_ID = cv2.imread(image_names[idx])
        ID_obj: int = 0
        ID_seg: int = 2000
    
        # Draw the segment and IDs
        for line_ht in new_elements_f:
          p0, p1 = line_ht
          cv2.line(img0_ID, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0,255,0), 2)              
          org = ((int((p0[0] + p1[0])/2)), (int((p0[1] + p1[1])/2)))              
          cv2.putText(img0_ID, str(ID_seg), org, 1, 0.7, (255, 0, 0), 1, font)
          ID_seg +=1
    
        if use_othogonal == True: # Draw the new segments highlighted
            for line_ht in new_elements_t1: 
                p0, p1 = line_ht            
                cv2.line(img0_ID, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0,0,255), 2) 
                org = ((int((p0[0] + p1[0])/2)), (int((p0[1] + p1[1])/2)))              
                cv2.putText(img0_ID, str(ID_seg), org, 1, 0.7, (255, 0, 0), 1, font)
                ID_seg +=1
         
        if show_images == True:
              cv2.imshow("Img", img0_ID)
              cv2.waitKey(10) 
            
        # Draw de object IDs
        with open(label,'r') as f:  # Load all labels
            label = f.readlines()
        for line in label:
            line = replace_n(line)  # Change str information to vector
            c_bb = conv_bb(line)    # Convert the coordinates of the BB    
            coord_bb_m = m_bb(line) 
           
            img0_ID = cv2.rectangle(img0_ID, (c_bb[0][0], c_bb[0][1]), (c_bb[1][0], c_bb[1][1]), (0,  0, 255), 1)
            
            cv2.putText(img0_ID, str(ID_obj), coord_bb_m, 1, 0.7, (0,  0, 255), 1, font)
            r = Image.fromarray(img0_ID, "RGB")    
            back_im_ID = r.copy()
            img0_ID = asarray(back_im_ID)
            if show_images == True:
                cv2.imshow("Img", img0_ID)
                cv2.waitKey(10)       
            ID_obj +=1                     
        cv2.waitKey(500)
        cv2.destroyAllWindows()
    
        # Write image result with IDS
        filename: str = output_path_ID + '/' + 'IDs_' + str(image_names[idx][25:-4]) + '.jpg'
        cv2.imwrite(filename, img0_ID)

###############################################################################
###############################################################################
# Graph of the segment connections (Object to segments) - Method 1
###############################################################################
###############################################################################
iS_obj=0
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
            el = new_elements_f        
            line = replace_n(line)  # Change str information to vector
            c_bb = conv_bb(line)    # Convert the coordinates of the BB  
            # Definition of the considered objects to connect to the segments
            # Consireding symbols and dots
            if int(line[0]) >= 76 or (int(line[0]) >= 62 and int(line[0]) <= 73):   
              for i in range (0, len(el)): 
                c_left = False; c_right = False; c_top = False; c_bottom = False
                c_o_left = False; c_o_right = False; c_o_top = False; c_o_bottom = False
                  
                # Is close to the left of the BB and is not far apart from the top and down borders of the BB
                if ((np.absolute(c_bb[0][0]-el[i][1][0]))<thr_s_x) and (c_bb[0][1]<el[i][1][1]) and (c_bb[1][1]>el[i][1][1]): 
                    c_left = True
                # Is close to the right of the BB and is not far apart from the top and down borders of the BB
                if ((np.absolute(c_bb[1][0]-el[i][0][0]))<thr_s_x) and (c_bb[0][1]<el[i][0][1]) and (c_bb[1][1]>el[i][0][1]): 
                    c_right = True
                # Is close to the top of the BB and is not far apart from the left and right borders of the BB
                if ((np.absolute(c_bb[0][1]-el[i][1][1]))<thr_s_y) and (c_bb[0][0]<el[i][1][0]) and (c_bb[1][0]>el[i][1][0]):
                    c_top = True   
                # Is close to the bottom of the BB and is not far apart from the left and right borders of the BB
                if ((np.absolute(c_bb[1][1]-el[i][0][1]))<thr_s_y) and (c_bb[0][0]<el[i][0][0]) and (c_bb[1][0]>el[i][0][0]):
                    c_bottom = True
    
                # To consider orthogonal segments it is necessary to consider that they could have other direction
                if use_othogonal == True: # Are the same rules just considering other direction of the segment
                    if ((np.absolute(c_bb[0][0]-el[i][0][0]))<thr_s_x) and (c_bb[0][1]<el[i][0][1]) and (c_bb[1][1]>el[i][0][1]):  
                        c_o_left = True
                    if ((np.absolute(c_bb[1][0]-el[i][1][0]))<thr_s_x) and (c_bb[0][1]<el[i][1][1]) and (c_bb[1][1]>el[i][1][1]): 
                        c_o_right = True
                    if ((np.absolute(c_bb[0][1]-el[i][0][1]))<thr_s_y) and (c_bb[0][0]<el[i][0][0]) and (c_bb[1][0]>el[i][0][0]):
                        c_o_top = True      
                    if ((np.absolute(c_bb[1][1]-el[i][1][1]))<thr_s_y) and (c_bb[0][0]<el[i][1][0]) and (c_bb[1][0]>el[i][1][0]):
                        c_o_bottom = True
                        
                if (c_left == True or c_right == True or c_top == True or c_bottom == True or c_o_left == True or c_o_right == True or c_o_top == True or c_o_bottom == True):
                    edges_class = ID_obj, i+ID_seg, line[0] 
                    #print(edges_class, c_bb, el[i])
                    edges_class_all.append(edges_class)
                                          
            ID_obj +=1
               
        # Save only object to object
        edges_symbols = []
        for i in range(0,len(edges_class_all)):
          for j in range(0,len(edges_class_all)):
            # Connect object to object   
            if (edges_class_all[i][1] == edges_class_all[j][1]) and (edges_class_all[i][0] != edges_class_all[j][0]):
              edges = edges_class_all[i][0], edges_class_all[j][0]#, edges_class_all[i][2], edges_class_all[j][2], edges_class_all[i][1]
              # Do not create extra duplicate egdes (102, 104) and (104, 102)
              if edges_class_all[i][0]>edges_class_all[j][0]:
                  edges_symbols.append(edges)
        
        # Remove duplicated edges 
        edges_symbols = np.unique(edges_symbols, axis=0)  
        
        # Save the graph of the segment connections
        df = pd.DataFrame(edges_symbols)
        #df.to_csv(r''+path_main[10:]+'Graph_segments/Objects/05_05/M1_' + str(image_names[idx][25:-4]) + '.csv')  
        edges = edges_symbols
    
   
###############################################################################
###############################################################################
# Graph of the letters connections (letter to letter) - Considering the Projec.
###############################################################################
###############################################################################
if connect_labels == True:
    for idx, label in enumerate(label_names):
        with open(label,'r') as f:      # Load all images
            label = f.readlines()
            
        thr_x = 15
        thr_y = 15
        ID_obj: int = 0
        edges_class_all = []
        c_bb_m_all = []
        c_bb_all=[]
        class_line = [] 
        ALL_ID = []
        ID_obj_b = ID_obj  
        
        # Save all variables
        for line in label:         
            line = replace_n(line)                                    
            c_bb = conv_bb(line)                                              
            c_bb_all.append(c_bb)   
            c_bb_m = m_bb(line)  
            c_bb_m_all.append(c_bb_m)
            class_line.append(int(line[0]))
            
        for line in label:
            line = replace_n(line)                                    
            c_bb = conv_bb(line)                                                
            c_bb_m = m_bb(line)     
                                       
            if int(line[0]) <= 61:
                for i in range (0, len(c_bb_all)):   
                    if class_line[i] <= 61:

                      # Check where is the BBs and 
                      check_same_bb = (c_bb[0][0]-c_bb_all[i][0][0])!=0 or ((c_bb[0][1]-c_bb_all[i][0][1])!=0) # check if the same BB is not used
                      check_r_side = ((np.absolute(c_bb[1][0]-c_bb_all[i][0][0]))<(np.absolute(c_bb[0][0]-c_bb_all[i][1][0]))) # right side
                      check_ver = ((np.absolute(c_bb[0][1]-c_bb_all[i][0][1]))<(np.absolute(c_bb[1][1]-c_bb_all[i][1][1]))) # bottom
        
                      # Distance between BB's
                      x = c_bb_m[0]-c_bb_m_all[i][0]
                      y = c_bb_m[1]-c_bb_m_all[i][1]
                      delta = int(math.sqrt(pow(x,2)+pow(y,2)))
            
                      if check_r_side and check_ver:
                        if (np.absolute(c_bb[1][0]-c_bb_all[i][0][0])<thr_x) and ((np.absolute(c_bb[0][1]-c_bb_all[i][0][1]))<thr_y) and check_same_bb:
                          edges_class = ID_obj, (i+ID_obj_b)#, int(line[0]), class_line[i], 'right bottom', c_bb, c_bb_all[i]
                          edges_class_all.append(edges_class) #, 'right bottom')
            
                      elif check_r_side and (not check_ver):
                        if (np.absolute(c_bb[1][0]-c_bb_all[i][0][0])<thr_x) and ((np.absolute(c_bb[1][1]-c_bb_all[i][1][1]))<thr_y) and check_same_bb:
                          edges_class = ID_obj, (i+ID_obj_b)#, int(line[0]), class_line[i], "right top", c_bb, c_bb_all[i]
                          edges_class_all.append(edges_class) #, "right top")
                      
                      elif (not check_r_side) and check_ver:
                        if ((np.absolute(c_bb[0][0]-c_bb_all[i][1][0]))<thr_x) and ((np.absolute(c_bb[0][1]-c_bb_all[i][0][1]))<thr_y) and check_same_bb:
                          edges_class = ID_obj, (i+ID_obj_b)#, int(line[0]), class_line[i], "left bottom", c_bb, c_bb_all[i]
                          edges_class_all.append(edges_class) #, "left bottom")
            
                      elif (not check_r_side) and (not check_ver):
                        if ((np.absolute(c_bb[0][0]-c_bb_all[i][1][0]))<thr_x) and ((np.absolute(c_bb[1][1]-c_bb_all[i][1][1]))<thr_y) and check_same_bb:
                          edges_class = ID_obj, (i+ID_obj_b)#, int(line[0]), class_line[i], "left top", c_bb, c_bb_all[i]
                          edges_class_all.append(edges_class) #, "left top")   
                      
                ID_obj +=1

    # Save the graph of the segment connections
    df = pd.DataFrame(edges_class_all)
    print(f'Saved: {str(image_names[idx][-22:-4])}')
    #df.to_csv(r''+path_main[10:]+'Graph_segments/Letters/110b/' + 'right_' + str(image_names[idx][-19:-4]) + '.csv')
    edges = edges_class_all
