from scipy.ndimage import morphology
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im_height = 220
im_width = 280

dir = 'Images/data/'
filename = 'Images/data/Output_morph_open2.hex'
# Open file
with open(filename) as f:
    text = f.read()

item = text.split('\n')
image_data =[]

for i in range(0,im_width*im_height):
    a = item[i][-4:-2]
    b = int(a,16)
    image_data.append(b*255)

image_data = np.array(image_data)
image_data = image_data.reshape(im_height,im_width)
#img = Image.fromarray(image_data)
#img.save('Images/data/Img_output_Coprocessor.png')
#img.show()

plt.imshow(image_data,cmap='gray')
plt.show()

#structuring_element_2 = np.ones((3,3),np.uint8)
#opening_2 = morphology.binary_opening(image_data)
#plt.imshow(opening_2, cmap='gray')
#plt.show()
