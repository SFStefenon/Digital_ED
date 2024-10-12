
''' 
Algoritm wrote by Stefano Frizzo Stefenon

Fondazione Bruno Kessler
Trento, May 15, 2023.

'''
'''
Parameters
----------
    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image, where
                n is uniform noise with specified mean & variance.
'''

# Import libs
import cv2
import glob
import numpy as np
from numpy import mean
import os
import cv2
import random

# Main path
main = str('C:/Users/ffriz/Dropbox/FBK/')
path_main = str('../PrePro/Noise_data/Done_Experiment_1/')
   
###############################################################################
###############################################################################
# Load all image names
###############################################################################
###############################################################################

# Load data
path = path_main + str('*.jpg')
image_names = glob.glob(path, recursive=True)
time = 1 # Time step to preent the images examples

###############################################################################
###############################################################################
# Function ot create noise
###############################################################################
###############################################################################

# Function to include noise
def noisy(noise_typ,image):
   image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   row,col = image.shape
   if noise_typ == "gauss_white": # Create gaussian noise
      Mean = 0.001; var = 1; sigma = np.sqrt(var)
      n = np.random.normal(loc=Mean, scale=sigma, size=(row,col))
      noisy = 255*(image + n) # Not save the background
      return noisy
   if noise_typ == "gauss": # Create gaussian noise
      image = image/255; Mean = 0; var = 0.01
      sigma = np.sqrt(var)
      n = np.random.normal(loc=Mean, scale=sigma, size=(row,col))
      noisy=255*(image + n) # Not save the background
      return noisy
   elif noise_typ == "s&p":
       output = np.zeros(image.shape,np.uint8)
       prob=0.005; thres = 1 - prob
       for i in range(image.shape[0]):
           for j in range(image.shape[1]):
               rdn = random.random()
               if rdn < prob:
                   output[i][j] = 0
               elif rdn > thres:
                    output[i][j] = 255
               else:
                    output[i][j] = image[i][j]
       return output
   elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 0.8 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
   elif noise_typ =="speckle":
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)        
        noisy = image + image * gauss
        return noisy

###############################################################################
###############################################################################
# Save the outputs with noise
###############################################################################
###############################################################################

for idx in image_names:
    img = cv2.imread(idx)        
    img0 = img.copy()       # Create a image copy
    noise_img = noisy("speckle", img0)
    filename: str = path_main + 'Output/n' + str(idx[39:-4]) + '.jpg'
    cv2.imwrite(filename, noise_img)  

###############################################################################
###############################################################################
# Rename txt files
###############################################################################
###############################################################################       
'''

import os
path = path_main + str('*.txt')
label_names = glob.glob(path, recursive=True)

for idx in label_names:
    labelname: str = path_main + 'Output/n' + idx[21:]
    print(labelname)
    os.rename(idx, labelname)
'''

###############################################################################
###############################################################################
# View the Gaussian
###############################################################################
###############################################################################    
'''
for idx in image_names:
    img = cv2.imread(idx)        
    img0 = img.copy()       # Create a image copy
    f = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    # Consider background
    f = f/255 
    # create gaussian noise
    x, y = f.shape
    Mean = 0
    var = 0.01
    sigma = np.sqrt(var)
    n = np.random.normal(loc=Mean, scale=sigma, size=(x,y))
    # add a gaussian noise
    g = f + n
    # Not save the background
    g=g*255
    filename: str = path_main + 'Output/n' + str(idx[21:-4]) + '.jpg'
    cv2.imwrite(filename, g)  
    
# Plot Gaussian
import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(n, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - 0)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()
'''
