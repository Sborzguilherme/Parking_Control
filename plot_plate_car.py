import cv2
import numpy as np
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
cont = 0
plate_file = 'Images/data/car_320_close.jpg'

img = cv2.imread(plate_file)
img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                               # Grayscale image
structuring_element = np.ones((9,19),np.uint8)
opening = cv2.morphologyEx(img_grayscale, cv2.MORPH_OPEN, structuring_element)      # Morphological opening
subtract = cv2.subtract(img_grayscale, opening)

(thresh, im_bw) = cv2.threshold(subtract, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # Binarization

structuring_element_2 = np.ones((3,3),np.uint8)
opening_2 = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, structuring_element_2)          # Morphological opening
"""structuring_element_close = np.ones((5,15),np.uint8)
plate = cv2.morphologyEx(opening_2, cv2.MORPH_CLOSE, structuring_element_close)

n_regions,plate2 = cv2.connectedComponents(plate, 4)
im,contours, hierarchy = cv2.findContours(plate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
max_area = 1
j=0
blob_num = 0
for i in range(0, n_regions-1):
    x,y,w,h = cv2.boundingRect(contours[i])
    area = cv2.contourArea(contours[i])
    ratio = w/h
    if(((ratio>=2.7) and (ratio<=4))and((w>=100 and h>=30)))and((w<=200 and h <=60)):   # Brazilian plate values
        if max_area <= (area):
            max_area = (area)
            blob_num = i

x_b, y_b,w_b,h_b = cv2.boundingRect(contours[blob_num])
plate_extracted = img_grayscale[y_b:y_b+h_b,x_b:x_b+w_b]        # [y_min:y_max,x_min:x_max]"""


"""im_height = 224
im_width = 284

filename = 'Images/data/Output_test_image_fsm_changed.hex'
# Open file
with open(filename) as f:
    text = f.read()

item = text.split('\n')
image_data =[]

for i in range(0,im_width*im_height):
    a = item[i][-4:-2]
    image_data.append(int(a,16))

image_data = np.array(image_data)
image_data = image_data.reshape(im_height,im_width)
#img = Image.fromarray(image_data)
#img.save('Images/data/Img_output_Coprocessor.png')
#img.show()
"""


#cv2.drawContours(img, contours, blob_num, (255,0,0), 2)

#  ------ Medium stages ------------------------------------------------ #
#cv2.imshow('Color image', img)
#cv2.imshow('Grayscale', img_grayscale)
#cv2.imshow('Morphological Opening',opening)#[8:224][18:284])
#cv2.imshow('Subtraction', subtract)
cv2.imshow('Binarization', im_bw[18:284+18][8:224+8])
#cv2.imshow('Morphological Opening 2', opening_2)
#cv2.imshow('Morphological Close', plate)
#cv2.imshow('Plate Extracted', plate_extracted)
diretorio_escrita = 'Images/Result/binarize_close_320.jpg'
#cv2.imwrite(diretorio_escrita,im_bw[8:224+8][18:284+18])
cv2.waitKey()

print(im_bw)

#plt.figure(1)
#plt.imshow(image_data,cmap='gray')

#plt.figure(2)
#plt.imshow(opening[18:284][8:224],cmap='gray')
#plt.show()

#diff = []
#opening = opening[18:285][:]
#print(len(opening))

"""for i in range(0,len(image_data[0])):
    for j in range(0,len(image_data)):
        diff.append((abs(image_data[j][i] - opening[j][i])))

diff = np.array(diff)
diff = diff.reshape(im_height,im_width)
plt.imshow(diff,cmap='gray')
plt.show()
"""
