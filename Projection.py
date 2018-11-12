import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

diretorio = 'Images/extracted/'                                  # Diretório com as imagens
files = glob.glob(diretorio+'*.jpg');     # Adiciona a uma lista todos os arquivos jpg

plate = diretorio+'2_extracted.jpg'
img = cv2.imread(plate)
img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                               # Grayscale image

(thresh, im_bw) = cv2.threshold(img_grayscale, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # Binarization

im_bw = 1-im_bw

vertical_projection = np.sum(im_bw,axis=0,dtype=np.uint16)
horizontal_projection = np.sum(im_bw,axis = 1,dtype=np.uint16)
#print(1-vertical_projection)
#plt.plot(1-vertical_projection)

#print(len(vertical_projection))
#print(len(horizontal_projection))

#print(1-vertical_projection)
square = np.zeros(len(vertical_projection))
#plt.plot(1-vertical_projection, 'b')
plt.plot(1-horizontal_projection)

x = np.arange(0, len(horizontal_projection),1)

valley_position_vert = [];
th_vert = (65475 - 2**16)
current_value = 'L'

for i in range(0, len(vertical_projection)):
    if(((1- vertical_projection[i]) < th_vert) and current_value == 'L'):
        valley_position_vert.append(i)
        current_value = 'H'
        square[i] = 65475
    elif((1- vertical_projection[i] > th_vert) and current_value == 'H'):
        valley_position_vert.append(i)
        current_value = 'L'
        square[i] = 65475
        #print(i)
    else:
        square[i] = 2**16
#print(vertical_projection)

valley_position_hor = [];
th_hor = (65293 - 2**16)
square_hor = np.zeros(len(horizontal_projection))

for i in range(0, len(horizontal_projection)):
    if(((1- horizontal_projection[i]) < th_hor) and current_value == 'L'):
        valley_position_hor.append(i)
        current_value = 'H'
        square_hor[i] = 65293
    elif((1- horizontal_projection[i] > th_hor) and current_value == 'H'):
        valley_position_hor.append(i)
        current_value = 'L'
        square_hor[i] = 65293
        #print(i)
    else:
        pass
        square_hor[i] = 2**16



fig,ax = plt.subplots(1)
ax.imshow(img)

#print(valley_position_hor)
#print(valley_position_vert)
diff_vert = valley_position_hor[len(valley_position_hor)-1] - valley_position_hor[0]
diff_hor = valley_position_vert[3] - valley_position_vert[1]

for i in range(0, len(valley_position_vert)-2,2):
    if(i == 6):
        continue
    rect = patches.Rectangle((valley_position_vert[i], valley_position_hor[0]),diff_hor,diff_vert,linewidth=2,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

#im = im.crop((1, 1, 98, 33))
#im.save('_0.png')

#plt.plot(1-horizontal_projection, x, 'b', square_hor, x, 'rs'); plt.title('PROJEÇÃO HORIZONTAL')  # PLOT HORIZONTAL PROJECTION
#plt.title('PROJEÇÃO VERTICAL')               # PLOT VERTICAL   PROJECTION
plt.show()
