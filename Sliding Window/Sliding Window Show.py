import argparse
import cv2

##############################################################################
# Define the image to be used
image = "I016II_SDO_110b.png"

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default=image, help="path")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

# Size of the image
len_y = len(image)
len_x = len(image[0])

# Slide window
size_y = 640
size_x = 640

max_y = len_y//size_y
max_x = len_x//size_x

for i in range(max_y):
    ##############################################################################
    # Define the size of the output pictures
    # Height of the picture
    line = i                    # first line is zero
    y2 = size_y + (line*size_y) # y2 maximum height
    y1 = y2 - size_y            # y1 minimum height
    
    # width of the pictures
    x2 = size_x                 # x2 maximum width
    x1 = x2 - size_x            # x1 minimum width
    yy = 0                      # step y (only if I wanna make with angle)
    xx = size_x                 # As I'm not interesed in over-writing 
                                # the second starts where the first ends 
    maxi = max_x                # number of results (for each line)
    
    ##############################################################################
    # Present and save the results
    ins=0
    for j in range(0,maxi):
        print(f'x{j+1} y{i+1}')
        # image output with background
        ins = cv2.rectangle(image,(x1+(j*xx),y1+(j*yy)),(x2+(j*xx),y2+(j*yy)),(50*j,0,250-10*j),2)
        
        # image output
        # ins = image[(y1+(j*yy)):y2+(j*yy), x1+(j*xx):x2+(j*xx)]
        
        cv2.imshow("FRI", ins)
        cv2.waitKey(50)
        
        # save the results
        #cv2.imwrite('RFI_%00i.jpg' %i, ins)
cv2.destroyAllWindows()


    
       

