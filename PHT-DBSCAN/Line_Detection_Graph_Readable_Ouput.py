
''' 
Algoritm wrote by Stefano Frizzo Stefenon

Fondazione Bruno Kessler
Trento, May 15, 2023.

'''

# Import libs
import cv2
import glob
from numpy import asarray
from PIL import Image 
from skimage.transform import probabilistic_hough_line
import numpy as np
import math
from numpy import mean
import sys
import pandas as pd

# Main path
main = str('C:/Users/ffriz/Dropbox/FBK/')
path_main = str('../PrePro/Dataset_Graphs/')

# Setup algorithm
CV_method = 'adaptive' # sobel' 'adaptive' 'binarization' 'Otsu_Riddler-Calvard'
cluster_method = 'DBSCAN' # 'KMeans' 'OPTICS' 'Agglomerative'
use_othogonal = False

# This definition is according to the file name
save_all_bb = False
save_seg_all = False
save_seg = False # Works if save_seg_all is false

# Check
if save_seg_all == True and save_seg == True:
    save_seg = False  # Works if save_seg_all is false cuz they are on the same loop
    
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
  x_1 = (int((float(line[1])) * 640)) - int(((float(line[3])) * 640)/2)
  y_1 = (int((float(line[2])) * 640)) - int(((float(line[4])) * 640)/2)
  x_2 = (int((float(line[1])) * 640)) + int(((float(line[3])) * 640)/2)
  y_2 = (int((float(line[2])) * 640)) + int(((float(line[4])) * 640)/2)
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
# Load all image names
###############################################################################
###############################################################################

# Load data
path = path_main + str('*.jpg')
image_names = glob.glob(path, recursive=True)

# Define output path for the first step (outputs with white boxes)
output_path1 = path_main + str('Processing/Without_simbols')
time = 1 # Time step to preent the images examples

# GLoad all label names
label_names: list = []
for image_name in image_names:
    label = image_name.replace('images','labels', 1).replace('.jpg','.txt', 1)
    label_names.append(label)

###############################################################################
###############################################################################
# Save All BB in one txt file
###############################################################################
###############################################################################

if save_all_bb == True:
    orig_stdout = sys.stdout
    output_path2 = path_main + str('Processing/I016II_SDO_110b_all_bb.txt')
    W_bb = open(output_path2, 'w')
    sys.stdout = W_bb
    for idx, label in enumerate(label_names):
        img = cv2.imread(image_names[idx])        
        # Position definition based on sliding window
        x_pos = int(image_names[idx][39:-4])
        y_pos = int(image_names[idx][37:-7])
        with open(label,'r') as f:
            label = f.readlines()
            for line in label:
                img0 = img.copy()       # Create a image copy
                line = replace_n(line)  # Change str information to vector
                print(line[0], "%5.6f" %(float(line[1])+x_pos), "%5.6f" %(float(line[2])+y_pos), "%5.6f" % (float(line[3])+x_pos), "%5.6f" % (float(line[4])+y_pos))
    sys.stdout = orig_stdout
    W_bb.close()

###############################################################################
###############################################################################
# Save All segments in one txt file
###############################################################################
###############################################################################

if save_seg_all == True:
    orig_stdout = sys.stdout
    file_all = path_main + 'Processing/I016II_SDO_110d_all_seg.txt'
    W_s_all = open(file_all, 'w')
    sys.stdout = W_s_all

###############################################################################
###############################################################################
# Show how the model read the output of YOLO
###############################################################################
###############################################################################
'''
for idx, label in enumerate(label_names):
    img = cv2.imread(image_names[idx])
    with open(label,'r') as f:
        label = f.readlines()
        line_counter: int = 0
        for line in label:
            img0 = img.copy()       # Create a image copy
            line = replace_n(line)  # Change str information to vector
            c_bb = conv_bb(line)    # Convert the coordinates of the BB           
            color = (255, 0, 255)   # Color of the rectangle       
            thickness = 1           # Line thickness 
            img0 = cv2.rectangle(img, (c_bb[0][0], c_bb[0][1]), (c_bb[1][0], c_bb[1][1]), color, thickness)
            cv2.imshow("Img", img0)
            cv2.waitKey(time*2)       
            line_counter +=1        # Counter increment
        cv2.destroyAllWindows()      
'''
###############################################################################
###############################################################################
# Save the outputs with white boxes 
###############################################################################
###############################################################################

for idx, label in enumerate(label_names):
    img = cv2.imread(image_names[idx])
    with open(label,'r') as f:
        label = f.readlines()
        line_counter: int = 0
        for line in label:
            img0 = img.copy()       # Create a image copy
            line = replace_n(line)  # Change str information to vector
            c_bb = conv_bb(line)    # Convert the coordinates of the BB
            color = (255, 255, 255) # Color of the rectangle
            thickness = -1          # Thickness of -1 will fill the entire shape
            imag0 = cv2.rectangle(img, (c_bb[0][0], c_bb[0][1]), (c_bb[1][0], c_bb[1][1]), color, thickness)
            #cv2.imshow("Img", imag0)
            #cv2.waitKey(time)       
            line_counter +=1        # Counter increment
        cv2.destroyAllWindows()      
    filename: str = output_path1 + '/' + 'Prepro_' + str(image_names[idx][25:-4]) + '.jpg'
    cv2.imwrite(filename, imag0)   
              
###############################################################################
###############################################################################
# Probabilistic Hough Line + Object Detection
###############################################################################
###############################################################################

input_path = path_main + str('Processing/Without_simbols/*.jpg') # Load all images (Pre-processed)
image_names_pre = glob.glob(input_path, recursive=True)

# Define output path for the second step (final results)
output_path_rec = path_main + str('Processing/Reconstructed')
output_path_ID = path_main + str('Processing/IDs')

# Import the image (Object Library)
# exec(open('library.py').read())
image_size = 640        # Image size
max_angle = 5           # Max angles of the segments

for idx, label in enumerate(label_names):
    img = cv2.imread(image_names[idx])
    img_o = cv2.imread((image_names_pre[idx]))
    image = cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY)

    # Definition of the Computer Vision Method
    if CV_method == 'canny':
        from skimage.feature import canny
        edges = canny(image, 2, 1, 25) 
    if CV_method == 'sobel':   
        sobelX = cv2.Sobel(image, cv2.CV_64F, 2, 0)
        sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 2)
        sobelX = np.uint8(np.absolute(sobelX))
        sobelY = np.uint8(np.absolute(sobelY))
        sobel = cv2.bitwise_or(sobelX, sobelY)
        edges = np.vstack([sobel])        
    if CV_method == 'adaptive':
        edges = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 5)
    if CV_method == 'binarization':      
        (T, binI) = cv2.threshold(image, 225, 255, cv2.THRESH_BINARY_INV)
        edges = np.vstack([np.hstack([cv2.bitwise_and(image, image, mask = binI)])])
    if CV_method == 'Otsu_Riddler-Calvard':        
        import mahotas
        T = mahotas.thresholding.otsu(image)
        temp = image.copy()
        temp[temp > T] = 255
        temp[temp < 255] = 0
        temp = cv2.bitwise_not(temp)
        T = mahotas.thresholding.rc(image)
        temp2 = image.copy()
        temp2[temp2 > T] = 255
        temp2[temp2 < 255] = 0
        temp2 = cv2.bitwise_not(temp2)
        edges = np.vstack([np.hstack([temp2])]) 
        
    # Probabilistic Hough Line
    lines = probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3) 
    lines_p = np.array(lines)
    for k in range(2):
      if k == 0:
        lines = np.flip(lines_p, 2)
      else:
        lines = lines_p
      # Sort the array based on the "y" cordinates (considering the mean of the cordinates y1, y2)
      lines_sorted_y = lines[np.argsort(lines[:,:,1].mean(axis=1))]
      # Caculate the angles of the segments fot the cordinate "x"
      def angle(x):
        ag = math.atan2(np.abs(x[1]-x[3]), np.abs(x[0]-x[2])) * (180.0 / math.pi)
        return ag
      # Maximum angle accepted for x (5°)
      lines_sorted_y = lines_sorted_y[np.where(np.apply_along_axis(angle, 1, lines_sorted_y.reshape(-1,4))<max_angle)]
      
      # Apply the clustering method if there are acceptable segments 
      if len(lines_sorted_y)>3:
          # Posible clustering methods
          if cluster_method == 'KMeans':
              from sklearn.cluster import KMeans
              n_c = int(len(lines_sorted_y)*0.6)
              db = KMeans(n_clusters=n_c, random_state=0, n_init="auto").fit((lines_sorted_y.reshape(-1,4))/image_size)
          if cluster_method == 'Agglomerative':
              from sklearn.cluster import AgglomerativeClustering
              n_c = int(len(lines_sorted_y)*0.6)
              db = AgglomerativeClustering(n_clusters=n_c).fit((lines_sorted_y.reshape(-1,4))/image_size)             
          if cluster_method == 'OPTICS':
              from sklearn.cluster import OPTICS
              db = OPTICS(min_samples=5).fit((lines_sorted_y.reshape(-1,4))/image_size)
          if cluster_method == 'DBSCAN':
              from sklearn.cluster import DBSCAN
              db = DBSCAN(eps=0.006, min_samples=2).fit(lines_sorted_y[:,:,1].mean(axis=1).reshape(-1,1)/image_size)              
          
          # List of unique_labels
          labels = db.labels_
          unique_labels = list(set(labels))
          n=0
          new_elements=[]
          new_elements_thr=[]
          # Goes through the labels
          for c in unique_labels:
            # Cluster the lines that have similar horizontal position"
            cluster_y = lines_sorted_y[np.where(labels == c)]
            c_sorted_along_x = np.apply_along_axis(sorted, 1, cluster_y)
            # Sort the segments along the cluster
            c_sorted_along_x = c_sorted_along_x[np.argsort(c_sorted_along_x[:,0,0])]
            # Save the x cordinate of each segment
            x_elements = c_sorted_along_x[:,:,0].copy()
            y_elements = c_sorted_along_x[:,:,1].copy()
            i = 0
            # If there is more than one element in the bucket
            if len(x_elements)>1:
              # While "i" is lower than the number of elements in the bucket
              while i<len(x_elements):
                # If the difference between the element is lower than 10
                # It means that they are close apart or overwritten
                if (x_elements[i+1,0] - x_elements[i,1]) <= 10:
                  # The new element has the first coordinate + the highest coordinate
                  el = [x_elements[i,0], x_elements[i:i+2,1].max()]
                  # Update the element with the new element
                  x_elements[i] = el
                  # Delete the previous elements that generate the new element
                  x_elements = np.delete(x_elements, i+1, axis=0)
                # If there is not close apart or overwritten element skip it
                else:
                  i += 1
                # If there is only one element in the bucket stop 
                if len(x_elements[i:]) == 1:
                  break
            for x_el in x_elements: # New segment (based on the mean of y)
              y_el = int(mean((y_elements[0,1], y_elements[-1,1])))
              new_elements.append([(x_el[0], y_el), (x_el[1], y_el)])
              if ((x_el[1]-x_el[0])>(image_size*0.02)):
                new_elements_thr.append([(x_el[0], y_el), (x_el[1], y_el)])             
          if k == 0:
            new_elements_thr_vertical = np.flip(new_elements_thr, 2)
          else:
            new_elements_thr_horizontal = new_elements_thr
    
      # If there is not a single segment detected create a default    
      else: # OBS: this is not goin to be consider in the graph
        new_elements_thr_horizontal=[(0,0), (0,1)]
        new_elements_thr_vertical=[(0,0), (1,0)]
    
    f_h = np.array(new_elements_thr_horizontal).flatten()   # Flatten 
    f_v = np.array(new_elements_thr_vertical).flatten()  
    conc_hv = np.concatenate((f_h, f_v))
    new_elements_f = np.resize(conc_hv,(int((len(conc_hv))/4),2,2))
    for line_ht in new_elements_f:
        # The lines are the final result of Probabilistic Hough Transform
        p0, p1 = line_ht
        cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0,255,0), 2)      
    # Load all images
    with open(label,'r') as f:
        label = f.readlines()
        line_counter: int = 0
        img0 = img.copy()           # Create a image copy     

    ###############################################################################
    ###############################################################################
    # Connect orthogonal segments 
    ###############################################################################
    ###############################################################################    
     
    if use_othogonal == True:
        lines_updated = new_elements_f
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
        lines_updated = np.append(lines_updated, new_elements_t1)
        lines_updated = lines_updated.reshape(-1,2,2)
        # Update the original array of segments
        new_elements_f = lines_updated
    
    ###############################################################################
    ###############################################################################
    # Save segments in txt files
    ###############################################################################
    ###############################################################################
    
    if save_seg == True:
        orig_stdout = sys.stdout
        file = path_main + 'Processing/Segments/' + str(image_names[idx][25:-4])+'.txt'
        W_s = open(file, 'w')
        sys.stdout = W_s
        for line_ht in new_elements_f:
          print(line_ht.flatten()[0], line_ht.flatten()[1], line_ht.flatten()[2], line_ht.flatten()[3])
        sys.stdout = orig_stdout
        W_s.close()

    if save_seg_all == True:
        for line_ht in new_elements_f:
          # Position definition based on sliding window
          x_pos = int(image_names[idx][39:-4])
          y_pos = int(image_names[idx][37:-7])
          print(int(line_ht.flatten()[0]+(x_pos*640)), int(line_ht.flatten()[1]+(y_pos*640)), int(line_ht.flatten()[2]+(x_pos*640)), int(line_ht.flatten()[3]+(y_pos*640)))

    ###############################################################################
    ###############################################################################
    # Use the result of Probabilistic Hough Line + YOLO to create (reconstructed a new images) 
    ###############################################################################
    ###############################################################################
    '''
    for line in label:
        line = replace_n(line)  # Change str information to vector
        c_bb = conv_bb(line)    # Convert the coordinates of the BB 
        coord_bb_m = m_bb(line)            
        m_1 = coord_bb_m[0]
        n_1 = coord_bb_m[1]
        r = Image.fromarray(img0, "RGB") # Generate an Image
        back_im = r.copy()
        # Import the image Conditions (Conditions Library)
        exec(open('library_conditions.py').read())
        img0 = asarray(back_im)
        cv2.imshow("Img", img0)
        cv2.waitKey(time*5)       
        line_counter +=1        # Counter increment
    cv2.waitKey(time*5)
    cv2.destroyAllWindows() 
    # Write image result of the reconstructed drawing
    filename: str = output_path_rec + '/' + 'REC_' + str(image_names[idx][-19:-4]) + '.jpg'
    cv2.imwrite(filename, img0)
    '''
    ###############################################################################
    ###############################################################################
    # Create images with IDs from PHT + YOLO
    ###############################################################################
    ###############################################################################
    
    ID_obj: int = 100
    ID_seg: int = 200
    img0_ID = img.copy()           # Create a image copy
    #cv2.imshow("Img", img0_ID)
    for line_ht in new_elements_f:
      p0, p1 = line_ht
      cv2.line(img0_ID, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0,255,0), 2)              
      org = ((int((p0[0] + p1[0])/2)), (int((p0[1] + p1[1])/2)))              
      cv2.putText(img0_ID, str(ID_seg), org, 1, 0.7, (255, 0, 0), 1, font)
      #cv2.imshow("Img", img0_ID)
      #cv2.waitKey(time*10) 
      ID_seg +=1
    for line in label:
        line = replace_n(line)  # Change str information to vector
        c_bb = conv_bb(line)    # Convert the coordinates of the BB    
        coord_bb_m = m_bb(line) 
        # print(str(ID_obj))
        cv2.putText(img0_ID, str(ID_obj), coord_bb_m, 1, 0.7, (0,  0, 255), 1, font)
        r = Image.fromarray(img0_ID, "RGB")    
        back_im_ID = r.copy()
        img0_ID = asarray(back_im_ID)
        #cv2.imshow("Img", img0_ID)
        #cv2.waitKey(time*10)       
        ID_obj +=1                     
    #cv2.waitKey(time*5000)
    cv2.destroyAllWindows()
    # Write image result with IDS
    filename: str = output_path_ID + '/' + 'IDs_' + str(image_names[idx][25:-4]) + '.jpg'
    cv2.imwrite(filename, img0_ID)
    
    ###############################################################################
    ###############################################################################
    # Graph of the segment connections (Object to segments) - Method 1
    ###############################################################################
    ###############################################################################
    
    thr_s_x = 5
    thr_s_y = 5
    ID_obj: int = 0
    ID_seg: int = 2000
    edges_class_all = []
    for line in label: 
        el = new_elements_f        
        line = replace_n(line)  # Change str information to vector
        c_bb = conv_bb(line)    # Convert the coordinates of the BB  
        # Definition of the considered objects to connect to the segments
        # Consireding symbols and dots
        if int(line[0]) >= 76 or (int(line[0]) >= 62 and int(line[0]) <= 73):  
          for i in range (0, len(el)):

            # Is close to the left of the BB and is not far apart from the top and down borders of the BB
            if ((np.absolute(c_bb[0][0]-el[i][1][0]))<thr_s_x) and (c_bb[0][1]<el[i][1][1]) and (c_bb[1][1]>el[i][1][1]):                
              edges_class = ID_obj, i+ID_seg, line[0]
              edges_class_all.append(edges_class)
            # Is close to the right of the BB and is not far apart from the top and down borders of the BB
            if ((np.absolute(c_bb[1][0]-el[i][0][0]))<thr_s_x) and (c_bb[0][1]<el[i][0][1]) and (c_bb[1][1]>el[i][0][1]):                
              edges_class = ID_obj, i+ID_seg, line[0]
              edges_class_all.append(edges_class)
            # Is close to the top of the BB and is not far apart from the left and right borders of the BB
            if ((np.absolute(c_bb[0][1]-el[i][1][1]))<thr_s_y) and (c_bb[0][0]<el[i][1][0]) and (c_bb[1][0]>el[i][1][0]):
              edges_class = ID_obj, i+ID_seg, line[0] 
              edges_class_all.append(edges_class)      
            # Is close to the botton of the BB and is not far apart from the left and right borders of the BB
            if ((np.absolute(c_bb[1][1]-el[i][0][1]))<thr_s_y) and (c_bb[0][0]<el[i][0][0]) and (c_bb[1][0]>el[i][0][0]):
              edges_class = ID_obj, i+ID_seg, line[0] 
              edges_class_all.append(edges_class)

            # To consider orthogonal segments it is necessary to consider that they could have other direction
            if use_othogonal == True: # Are the same rules just considering other direction of the segment
                if ((np.absolute(c_bb[0][0]-el[i][0][0]))<thr_s_x) and (c_bb[0][1]<el[i][0][1]) and (c_bb[1][1]>el[i][0][1]):                
                  edges_class = ID_obj, i+ID_seg, line[0]
                  edges_class_all.append(edges_class)
                if ((np.absolute(c_bb[1][0]-el[i][1][0]))<thr_s_x) and (c_bb[0][1]<el[i][1][1]) and (c_bb[1][1]>el[i][1][1]):                
                  edges_class = ID_obj, i+ID_seg, line[0]
                  edges_class_all.append(edges_class)
                if ((np.absolute(c_bb[0][1]-el[i][0][1]))<thr_s_y) and (c_bb[0][0]<el[i][0][0]) and (c_bb[1][0]>el[i][0][0]):
                  edges_class = ID_obj, i+ID_seg, line[0] 
                  edges_class_all.append(edges_class)         
                if ((np.absolute(c_bb[1][1]-el[i][1][1]))<thr_s_y) and (c_bb[0][0]<el[i][1][0]) and (c_bb[1][0]>el[i][1][0]):
                  edges_class = ID_obj, i+ID_seg, line[0] 
                  edges_class_all.append(edges_class)                    
        ID_obj +=1
           
    # Save only object to object
    edges_symbols = []
    for i in range(0,len(edges_class_all)):
      for j in range(0,len(edges_class_all)):
        # Connect object to object   
        if (edges_class_all[i][1] == edges_class_all[j][1]) and (edges_class_all[i][0] != edges_class_all[j][0]):
          edges = edges_class_all[i][0], edges_class_all[j][0], edges_class_all[i][2], edges_class_all[j][2], edges_class_all[i][1]
          # Do not create extra duplicate egdes (102, 104) and (104, 102)
          if edges_class_all[i][0]>edges_class_all[j][0]:
              edges_symbols.append(edges)
    
    # Remove duplicated edges 
    edges_symbols = np.unique(edges_symbols, axis=0)  
    
    # Save the graph of the segment connections
    df = pd.DataFrame(edges_symbols)
    df.to_csv(r''+path_main[10:]+'Graph_segments/Objects/01_01/M1_' + str(image_names[idx][25:-4]) + '.csv')  
    
    #cv2.waitKey(time*1000)
    #cv2.destroyAllWindows()
    
    ###############################################################################
    ###############################################################################
    # Graph of the letters connections (letter to letter) - Considering the Projection
    ###############################################################################
    ###############################################################################
    '''
    thr_x = 15
    thr_y = 15
    ID_obj: int = 100
    ID_seg: int = 200
    
    edges_class_all = []
    c_bb_m_all = []
    c_bb_all=[]
    class_line = [] 
     
    for line in label: 
        line = replace_n(line) 
        class_line.append(int(line[0]))
    ID_obj_b = ID_obj  
    
    for line in label:         
        line = replace_n(line)                                    
        c_bb = conv_bb(line)                                              
        c_bb_all.append(c_bb)   
        c_bb_m = m_bb(line)  
        c_bb_m_all.append(c_bb_m)
                                                
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
                      edges_class = ID_obj, (i+ID_obj_b), delta, 0, 'right bottom', line[0], class_line[i], c_bb, c_bb_all[i]
                      edges_class_all.append(edges_class) #, 'right bottom')
        
                  elif check_r_side and (not check_ver):
                    if (np.absolute(c_bb[1][0]-c_bb_all[i][0][0])<thr_x) and ((np.absolute(c_bb[1][1]-c_bb_all[i][1][1]))<thr_y) and check_same_bb:
                      edges_class = ID_obj, (i+ID_obj_b), delta, 0, "right top", line[0], class_line[i], c_bb, c_bb_all[i]
                      edges_class_all.append(edges_class) #, "right top")
        
                  elif (not check_r_side) and check_ver:
                    if ((np.absolute(c_bb[0][0]-c_bb_all[i][1][0]))<thr_x) and ((np.absolute(c_bb[0][1]-c_bb_all[i][0][1]))<thr_y) and check_same_bb:
                      edges_class = ID_obj, (i+ID_obj_b), delta, 1, "left bottom", line[0], class_line[i], c_bb, c_bb_all[i]
                      edges_class_all.append(edges_class) #, "left bottom")
        
                  elif (not check_r_side) and (not check_ver):
                    if ((np.absolute(c_bb[0][0]-c_bb_all[i][1][0]))<thr_x) and ((np.absolute(c_bb[1][1]-c_bb_all[i][1][1]))<thr_y) and check_same_bb:
                      edges_class = ID_obj, (i+ID_obj_b), delta, 1, "left top", line[0], class_line[i], c_bb, c_bb_all[i]
                      edges_class_all.append(edges_class) #, "left top")   
    
            ID_obj +=1 
        edges_class_all

    # Save the graph of the segment connections
    df = pd.DataFrame(edges_class_all)
    print(f'Saved: {str(image_names[idx][-22:-4])}')
    df.to_csv(r''+path_main[10:]+'Graph_segments/Letters/110d2/' + str(thr_x) + '_' + str(thr_y) + '/' + 'D1_' + str(image_names[idx][-19:-4]) + '.csv')
    '''
    ###############################################################################
    ###############################################################################
    # Graph of the letters connections (letter to letter) - Considering the IoU
    ###############################################################################
    ###############################################################################
    '''
    thr_x = 5
    thr_y = 0
    ID_obj: int = 100
    ID_seg: int = 200
    
    edges_class_all = []
    c_bb_all=[]
    c_bb_m_all=[]
    c_bb_o_all=[]
    class_line = []
    
    for line in label: 
        line = replace_n(line) 
        class_line.append(int(line[0]))

    for line in label:
      line = replace_n(line)
      c_bb = conv_bb(line)
      c_bb_all.append(c_bb)
      c_bb_m = m_bb(line)
      c_bb_m_all.append(c_bb_m)
      c_bb_o = o_bb(line, thr_bb_x = thr_x, thr_bb_y = thr_y)
      c_bb_o_all.append(c_bb_o)  
    
    for line in label:
        line = replace_n(line)                                                      # Change str information to vector
        c_bb = conv_bb(line)
        c_bb_m = m_bb(line)
        c_bb_o = o_bb(line, thr_bb_x = thr_x, thr_bb_y = thr_y)
        
        # Evaluation of letters only
        if int(line[0]) <= 61:            
            for i in range (0, len(c_bb_o_all)):
                
              # Distance between BB's
              
              #c_bb_o_all = m_bb(c_bb_o_all)
              
              x = c_bb_m[0]-c_bb_m_all[i][0]
              y = c_bb_m[1]-c_bb_m_all[i][1]
              delta = int(math.sqrt(pow(x,2)+pow(y,2)))
               
              # IoU  
              iou = bb_intersection_over_union(list(np.array(c_bb_o).flatten()), list(np.array(c_bb_o_all[i]).flatten()))
              
              # Has IoU but it is not itself
              if iou>0.1 and iou!=1:
                  # Should be a letter close to another letter
                  if class_line[i]<=61:
                      edges_class = ID_obj, (i+ID_obj), delta, iou, line[0], class_line[i]#, c_bb, c_bb_all[i]
                      edges_class_all.append(edges_class)
        ID_obj +=1
    edges_class_all
    
    # Save the graph of the segment connections
    df = pd.DataFrame(edges_class_all)
    print(f'Saved: {str(image_names[idx][25:-4])}')
    #df.to_csv(r''+path_main[10:]+'Graph_segments/Letters/IoU/' + str(thr_x) + '_' + str(thr_y) + '/' + 'D1_' + str(image_names[idx][25:-4]) + '.csv')
    '''

 
# read graph
# mydata = np.loadtxt('Output_0_0.csv', delimiter=",", dtype=np.float32, skiprows=1, usecols=(1, 2))

# read ground_truth
# ground_truth = np.loadtxt('RFI_640_110b0_7.csv', delimiter=",", dtype=np.float32, skiprows=0)
###############################################################################
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
###############################################################################
    '''
    ###############################################################################
    ###############################################################################
    # Create an output that can be read by NORMA Tool
    ###############################################################################
    ###############################################################################

    #str(image_names[idx][25:-4])
    import sys
    orig_stdout = sys.stdout
    path_dia = main + path_main[3:] + str('Dia/out.dia')
    f = open(path_dia, 'w')
    sys.stdout = f
    
    print('<?xml version="1.0" encoding="UTF-8"?>')
    print('<dia:diagram xmlns:dia="http://www.lysator.liu.se/~alla/dia/">')
    print('  <dia:diagramdata>')
    print('    <dia:attribute name="background">')
    print('      <dia:color val="#ffffffff"/>')
    print('    </dia:attribute>')
    print('    <dia:attribute name="pagebreak">')
    print('      <dia:color val="#000099ff"/>')
    print('    </dia:attribute>')
    print('    <dia:attribute name="paper">')
    print('      <dia:composite type="paper">')
    print('        <dia:attribute name="name">')
    print('          <dia:string>#Letter#</dia:string>')
    print('        </dia:attribute>')
    print('        <dia:attribute name="tmargin">')
    print('          <dia:real val="2.5399999618530273"/>')
    print('        </dia:attribute>')
    print('        <dia:attribute name="bmargin">')
    print('          <dia:real val="2.5399999618530273"/>')
    print('        </dia:attribute>')
    print('        <dia:attribute name="lmargin">')
    print('          <dia:real val="2.5399999618530273"/>')
    print('        </dia:attribute>')
    print('        <dia:attribute name="rmargin">')
    print('          <dia:real val="2.5399999618530273"/>')
    print('        </dia:attribute>')
    print('        <dia:attribute name="is_portrait">')
    print('          <dia:boolean val="true"/>')
    print('        </dia:attribute>')
    print('        <dia:attribute name="scaling">')
    print('          <dia:real val="1"/>')
    print('        </dia:attribute>')
    print('        <dia:attribute name="fitto">')
    print('          <dia:boolean val="false"/>')
    print('        </dia:attribute>')
    print('      </dia:composite>')
    print('    </dia:attribute>')
    print('    <dia:attribute name="grid">')
    print('      <dia:composite type="grid">')
    print('        <dia:attribute name="dynamic">')
    print('          <dia:boolean val="true"/>')
    print('        </dia:attribute>')
    print('        <dia:attribute name="width_x">')
    print('          <dia:real val="1"/>')
    print('        </dia:attribute>')
    print('        <dia:attribute name="width_y">')
    print('          <dia:real val="1"/>')
    print('        </dia:attribute>')
    print('        <dia:attribute name="visible_x">')
    print('          <dia:int val="1"/>')
    print('        </dia:attribute>')
    print('        <dia:attribute name="visible_y">')
    print('          <dia:int val="1"/>')
    print('        </dia:attribute>')
    print('        <dia:composite type="color"/>')
    print('      </dia:composite>')
    print('    </dia:attribute>')
    print('    <dia:attribute name="color">')
    print('      <dia:color val="#d8e5e5ff"/>')
    print('    </dia:attribute>')
    print('    <dia:attribute name="guides">')
    print('      <dia:composite type="guides">')
    print('        <dia:attribute name="hguides"/>')
    print('        <dia:attribute name="vguides"/>')
    print('      </dia:composite>')
    print('    </dia:attribute>')
    print('    <dia:attribute name="display">')
    print('      <dia:composite type="display">')
    print('        <dia:attribute name="antialiased">')
    print('          <dia:boolean val="false"/>')
    print('        </dia:attribute>')
    print('        <dia:attribute name="snap-to-grid">')
    print('          <dia:boolean val="true"/>')
    print('        </dia:attribute>')
    print('        <dia:attribute name="snap-to-object">')
    print('          <dia:boolean val="true"/>')
    print('        </dia:attribute>')
    print('        <dia:attribute name="show-grid">')
    print('          <dia:boolean val="true"/>')
    print('        </dia:attribute>')
    print('        <dia:attribute name="show-connection-points">')
    print('          <dia:boolean val="true"/>')
    print('        </dia:attribute>')
    print('      </dia:composite>')
    print('    </dia:attribute>')
    print('  </dia:diagramdata>')
    print('  <dia:layer name="Background" visible="true" connectable="true" active="true">')

    ID_obj: int = 100
    ID_seg: int = 200
    conv = 0.055#0.026458#333333

    for line in label: 
        el = new_elements_f        
        line = replace_n(line)  
        c_bb = conv_bb(line)

        if int(line[0]) >= 76  or (int(line[0]) >= 62 and int(line[0]) <= 73):  
            print(f'    <dia:object type="Flowchart - Box" version="0" id="O{ID_obj}">')
            print('      <dia:attribute name="obj_pos">')
            print(f'        <dia:point val="{c_bb[0][0]*conv},{c_bb[0][1]*conv}"/>')
            print('      </dia:attribute>')
            print('      <dia:attribute name="obj_bb">')
            print(f'        <dia:rectangle val="{c_bb[0][0]*conv},{c_bb[0][1]*conv};{c_bb[1][0]*conv},{c_bb[1][1]*conv}"/>')
            print('      </dia:attribute>')
            print('      <dia:attribute name="elem_corner">')
            print(f'        <dia:point val="{c_bb[0][0]*conv},{c_bb[0][1]*conv}"/>')
            print('      </dia:attribute>')
            print('      <dia:attribute name="elem_width">')
            print(f'        <dia:real val="{(c_bb[1][0]-c_bb[0][0])*conv}"/>')  # BB width
            print('      </dia:attribute>')
            print('      <dia:attribute name="elem_height">')
            print(f'        <dia:real val="{(c_bb[1][1]-c_bb[0][1])*conv}"/>')  # BB height
            print('      </dia:attribute>')
            print('      <dia:attribute name="show_background">')
            print('        <dia:boolean val="true"/>')
            print('      </dia:attribute>')
            print('      <dia:attribute name="padding">')
            print('        <dia:real val="0.1"/>')                              # BB scale
            print('      </dia:attribute>')
            print('      <dia:attribute name="text">')
            print('        <dia:composite type="text">')
            print('          <dia:attribute name="string">')
            #print(f'            <dia:string>#{line[0]}#</dia:string>')          # BB name
            print(f'            <dia:string>#{ID_obj}#</dia:string>')          # BB name
            print('          </dia:attribute>')
            print('          <dia:attribute name="font">')
            print('            <dia:font family="sans" style="0" name="Helvetica"/>')
            print('          </dia:attribute>')
            print('          <dia:attribute name="height">')
            print('            <dia:real val="1"/>')                            # Text height
            print('          </dia:attribute>')
            print('          <dia:attribute name="pos">')
            print(f'            <dia:point val="{c_bb[1][0]*conv},{c_bb[1][1]*conv}"/>') # Text position
            print('          </dia:attribute>')
            print('          <dia:attribute name="color">')
            print('            <dia:color val="#000000ff"/>')
            print('          </dia:attribute>')
            print('          <dia:attribute name="alignment">')
            print('            <dia:enum val="1"/>')                            # Text alignment
            print('          </dia:attribute>')
            print('        </dia:composite>')
            print('      </dia:attribute>')
            print('    </dia:object>')  
            

        if int(line[0]) <= 61:

            if int(line[0]) >= 10 and int(line[0]) <= 35:
                line[0] = chr(97+(int(line[0])-10)) # 97 is to convert and 10 is because start from it
            elif int(line[0]) >= 36:
                line[0] = chr(97-32+(int(line[0])-36)) # 97-32 is to convert CAPITAL and 36 is because start from it

            print(f'    <dia:object type="Standard - Outline" version="0" id="O{ID_obj}">')
            print('      <dia:attribute name="obj_pos">')
            print(f'        <dia:point val="{c_bb[0][0]*conv},{c_bb[0][1]*conv}"/>')
            print('      </dia:attribute>')
            print('      <dia:attribute name="obj_bb">')
            print(f'        <dia:rectangle val="{c_bb[0][0]*conv},{c_bb[0][1]*conv};{c_bb[1][0]*conv},{c_bb[1][1]*conv}"/>')
            print('      </dia:attribute>')
            print('      <dia:attribute name="meta">')
            print('        <dia:composite type="dict"/>')
            print('      </dia:attribute>')
            print('      <dia:attribute name="name">')
            print(f'        <dia:string>#{line[0]}#</dia:string>')
            print('      </dia:attribute>')
            print('      <dia:attribute name="rotation">')
            print('        <dia:real val="0"/>')
            print('      </dia:attribute>')
            print('      <dia:attribute name="text_font">')
            print('        <dia:font family="sans" style="0" name="Helvetica"/>')
            print('      </dia:attribute>')
            print('      <dia:attribute name="text_height">')
            print(f'        <dia:real val="{(c_bb[1][1]-c_bb[0][1])*conv*0.5}"/>')
            print('      </dia:attribute>')
            print('      <dia:attribute name="line_width">')
            print('        <dia:real val="0"/>')
            print('      </dia:attribute>')
            print('      <dia:attribute name="line_colour">')
            print('        <dia:color val="#000000ff"/>')
            print('      </dia:attribute>')
            print('      <dia:attribute name="fill_colour">')
            print('        <dia:color val="#ffffffff"/>')
            print('      </dia:attribute>')
            print('      <dia:attribute name="show_background">')
            print('        <dia:boolean val="false"/>')
            print('      </dia:attribute>')
            print('    </dia:object>')
        
        ID_obj +=1
 
    for i in range (0, len(el)):
        for k in range (0, int(len(edges_symbols.flatten())/5)):
            
            if (int(edges_symbols[k][4])==(i+ID_seg)): 
                print(f'<dia:object type="Standard - Line" version="0" id="{i+ID_seg}">')
                print('  <dia:attribute name="obj_pos">')
                print('    <dia:point val="{el[i][0][0]*conv},{el[i][0][1]*conv}"/>')
                print('  </dia:attribute>')
                print('  <dia:attribute name="obj_bb">')
                print('    <dia:rectangle val="{el[i][0][0]*conv},{el[i][0][1]*conv};{el[i][1][0]*conv},{el[i][1][1]*conv}"/>')
                print('  </dia:attribute>')
                print('  <dia:attribute name="conn_endpoints">')
                print('    <dia:point val="{el[i][0][0]*conv},{el[i][0][1]*conv}"/>')
                print('    <dia:point val="{el[i][1][0]*conv},{el[i][1][1]*conv}"/>')
                print('  </dia:attribute>')
                print('  <dia:attribute name="numcp">')
                print('    <dia:int val="1"/>')
                # This is to use arrows
                #print('  </dia:attribute>')
                #print('  <dia:attribute name="end_arrow">')
                #print('    <dia:enum val="1"/>') # Arrow is 22
                #print('  </dia:attribute>')
                #print('  <dia:attribute name="end_arrow_length">')
                #print('    <dia:real val="0.5"/>')
                #print('  </dia:attribute>')
                #print('  <dia:attribute name="end_arrow_width">')
                #print('    <dia:real val="0.5"/>')
                print('  </dia:attribute>')
                print('  <dia:connections>')
                print(f'        <dia:connection handle="0" to="O{edges_symbols[k][0]}" connection="16"/>')
                print(f'        <dia:connection handle="1" to="O{edges_symbols[k][1]}" connection="16"/>')
                print('      </dia:connections>')
                print('    </dia:object>')          

    print('  </dia:layer>')
    print('</dia:diagram>')
    sys.stdout = orig_stdout
    f.close()
    '''

# Save all segments
if save_seg_all == True:
    sys.stdout = orig_stdout
    W_s_all.close()      
