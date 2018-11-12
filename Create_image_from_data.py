from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

im_height = 224
im_width = 284

filename = 'Images/data/saida.txt'
# Open file
with open(filename) as f:
    text = f.read()

# Put file into a array. Already converted to decimal
item = text.split('\n')
image_data = []
for i in range(0,(im_height*im_width)):
    image_data.append(int(item[i],16))

"""filename_w = "Images/data/test.txt"
F = open(filename_w,'w')
for i in range(0,len(image_data)):
    if i == len(image_data)-1:
        F.write(str(image_data[i]))
    else:
        F.write(str(image_data[i]) + "\n")
"""

#image_data = np.fromfile(filename_w,np.uint8,-1,"\n")
image_data = np.array(image_data)
image_data = image_data.reshape(im_height,im_width)

img = Image.fromarray(image_data)
#img.save('Images/data/im_saida_test.png')
img.show()


#plt.imshow(image_data)
#plt.show()




