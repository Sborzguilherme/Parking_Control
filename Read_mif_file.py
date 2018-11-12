import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt

im_height = 224
im_width = 284

# Open file
with open('Images/data/car_close_284_224.mif') as f:
    text = f.read()

# Separate lines
lines = text.split(':')
result = []
for i in range(1,len(lines)):
    data = lines[i].split(';\n')
    data = int(data[0])
    result.append(data * 255)           # If read a binary image

result = np.array(result)
result = result.reshape(im_height,im_width)

img = Image.fromarray(result)
#img.save('Images/data/original_input_test.png')
img.show()
