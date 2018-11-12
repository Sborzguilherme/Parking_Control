from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im_height = 224
im_width = 284

filename = 'Images/data/Output_test_image_fsm_changed.hex'
filename2 = 'Images/data/Output_PLL_80MHZ.hex'
# Open file
with open(filename) as f:
    text = f.read()

with open(filename2) as f2:
    text2 = f2.read()


item = text.split('\n')
item2 = text2.split('\n')
image_data =[]
image_data2 = []
for i in range(0,im_width*im_height):
    a = item[i][-4:-2]
    b = item2[i][-4:-2]
    image_data.append(int(a,16))
    image_data2.append(int(b,16))

diff = []
for i in range(0,len(image_data)):
    diff.append(abs(image_data[i] - image_data2[i]))

diff = np.array(diff)
diff = diff.reshape(im_height,im_width)
#img = Image.fromarray(image_data)
#img.save('Images/data/Img_output_Coprocessor.png')
#img.show()

plt.imshow(diff,cmap='gray')
plt.show()
