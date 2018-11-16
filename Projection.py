import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import operator

diretorio = 'Images/extracted/'           # Images Directory
files = glob.glob(diretorio+'*.jpg');     # Put all images into a list

plate = diretorio+'17_extracted.jpg'       # Name of the current image being processed
img = cv2.imread(plate)
img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # Turn plate image into grayscale

(thresh, im_bw) = cv2.threshold(img_grayscale, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # Binarization

im_bw = 1-im_bw

vertical_projection = np.sum(im_bw,axis=0,dtype=np.uint16)      # Array with sum of collumns
horizontal_projection = np.sum(im_bw,axis = 1,dtype=np.uint16)  # Array with sum of lines

#print(vertical_projection)

########################### PLOTS VERTICAL AND HORIZONTAL PROJECTION ############################################
#plt.figure(1)
#plt.plot(1-vertical_projection)

x = np.arange(0, len(horizontal_projection),1)
#plt.figure(2)
#plt.plot(1-horizontal_projection, x)
#plt.show()
############################ DEFINE VALLEYS POSITIONS ###############################################

square = np.array([2**16] * len(vertical_projection))
valley_position_vert = [];                                      # Save the valleys positions
mean = np.mean(vertical_projection)
mean_plot = np.mean(vertical_projection)
mean_plot = mean_plot*1.05
mean_array = np.array(([mean_plot]) * len(vertical_projection))
status = 1
dict_samples = {}

for i in range(0, len(vertical_projection)):
    if(status == 1):    # Must verify if the current sample is below average
        if(vertical_projection[i] > mean*1.05):
            dict_samples.update({i:vertical_projection[i]})

        elif(bool(dict_samples) == True):
            pos = max(dict_samples.items(), key=operator.itemgetter(1))[0]
            val = max(dict_samples.items(), key=operator.itemgetter(1))[1]
            valley_position_vert.append(pos)
            square[pos] -= val
            dict_samples.clear()
            status = 0
        else:
            status = 0
    elif(status == 0):
        if(vertical_projection[i] > mean):
            status = 1

print(valley_position_vert)
#plt.figure(1)
#plt.plot(vertical_projection, 'b',mean_array, 'g--')

# Repeat same procedure for the horizontal projection
#square = np.array([2**16] * len(vertical_projection))
valley_position_hor = [];                                      # Save the valleys positions
mean = np.mean(horizontal_projection)
mean_plot = np.mean(1-horizontal_projection)

mean_array_hor = np.array([mean_plot] * len(horizontal_projection))
#plt.plot(1-horizontal_projection, 'b',mean_array_hor, 'g--')
plt.show()

status = 0
dict_samples.clear()

for i in range(0, len(horizontal_projection)):
    if(status == 1):    # Must verify if the current sample is below average
        if(horizontal_projection[i] > mean):
            dict_samples.update({i:horizontal_projection[i]})
        elif(bool(dict_samples) == True):       # Verify if dictionary is not empty
            pos = max(dict_samples.items(), key=operator.itemgetter(1))[0]
            valley_position_hor.append(pos)
            dict_samples.clear()
            status = 0
        else:
            status = 0
    elif(status == 0):
        if(horizontal_projection[i] > mean):
            status = 1

print(valley_position_hor)
######################################### CROP IMAGE ###########################################################
fig,ax = plt.subplots(1)

#print(valley_position_hor)
pos = len(valley_position_hor)-1
start_h = pos - 1
diff_vert = valley_position_hor[pos] - valley_position_hor[start_h]
cont = 1
while(diff_vert < 10):
    if(len(valley_position_hor) > 1+cont):
        diff_vert = valley_position_hor[pos] - valley_position_hor[pos-cont-1]
        start_h = pos-cont-1
        cont+=1
    else:
        break
for i in range(0, len(valley_position_vert)-1):

    diff_hor = valley_position_vert[i+1] - valley_position_vert[i]
    # Always put the sqyuare in the image
    rect = patches.Rectangle((valley_position_vert[i], valley_position_hor[start_h]),diff_hor,diff_vert,linewidth=2,edgecolor='r',facecolor='none')
    ax.add_patch(rect)  # ADD rectangle to image
    if(diff_hor < 11 and i < len(valley_position_vert)-1):
        # If diff_hor don't correspond to a valid space between valley, put another square
        diff_hor = valley_position_vert[i+1] - valley_position_vert[i]
        rect = patches.Rectangle((valley_position_vert[i], valley_position_hor[len(valley_position_hor)-2]),diff_hor,diff_vert,linewidth=2,edgecolor='r',facecolor='none')
        ax.add_patch(rect)  # ADD rectangle to image
diff_hor = len(img[0]) - valley_position_vert[len(valley_position_vert)-1]
rectF = patches.Rectangle((valley_position_vert[len(valley_position_vert)-1], valley_position_hor[start_h]),diff_hor,diff_vert,linewidth=2,edgecolor='r',facecolor='none')
ax.add_patch(rectF)

ax.imshow(img)                  # Show plate
plt.show()
