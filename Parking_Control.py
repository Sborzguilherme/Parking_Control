import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import operator
import pytesseract
import re
import time

############################################## FIND PLATE ##############################################################
def findPlate(image):
    img = cv2.imread(image)
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                               # Grayscale image

    structuring_element = np.ones((15,30),np.uint8)
    opening = cv2.morphologyEx(img_grayscale, cv2.MORPH_OPEN, structuring_element)      # Morphological opening
    subtract = cv2.subtract(img_grayscale, opening)

    (thresh, im_bw) = cv2.threshold(subtract, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # Binarization

    structuring_element_2 = np.ones((3,3),np.uint8)
    opening_2 = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, structuring_element_2)          # Morphological opening
    structuring_element_close = np.ones((5,15),np.uint8)
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
    plate_extracted = img_grayscale[y_b:y_b+h_b,x_b:x_b+w_b]        # [y_min:y_max,x_min:x_max]
    return plate_extracted

############################################# PROJECTION ##############################################################
def projection(plate):
    img = plate
    #img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # Turn plate image into grayscale
    img_grayscale = plate
    (thresh, im_bw) = cv2.threshold(img_grayscale, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # Binarization

    im_bw = 1-im_bw

    vertical_projection = np.sum(im_bw,axis=0,dtype=np.uint16)      # Array with sum of collumns
    horizontal_projection = np.sum(im_bw,axis = 1,dtype=np.uint16)  # Array with sum of lines

    valley_position_vert = [];                                      # Save the valleys positions
    mean = np.mean(vertical_projection)
    mean_plot = np.mean(vertical_projection)
    mean_plot = mean_plot*1.05
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
                dict_samples.clear()
                status = 0
            else:
                status = 0
        elif(status == 0):
            if(vertical_projection[i] > mean):
                status = 1

    # Repeat same procedure for the horizontal projection
    valley_position_hor = [];                                      # Save the valleys positions
    mean = np.mean(horizontal_projection)
    mean_plot = np.mean(1-horizontal_projection)

    mean_array_hor = np.array([mean_plot] * len(horizontal_projection))

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

    # ------------------------------------- HORIZONTAL CROP  ---------------------------------
    pos = len(valley_position_hor)-1

    if(len(valley_position_hor) < 1):
        diff_vert = len(img)
        start_h = 0
    elif(len(valley_position_hor) < 2):
        start_h = valley_position_hor[0]
        diff_vert = len(img) - start_h
    else:
        start_h = valley_position_hor[pos - 1]
        diff_vert = valley_position_hor[pos] - start_h
    cont = 1
    while(diff_vert < 20):
        if(len(valley_position_hor) > 1+cont):
            diff_vert = valley_position_hor[pos] - valley_position_hor[pos-cont-1]
            start_h = valley_position_hor[pos-cont-1]
            cont+=1
        else:
            start_h = valley_position_hor[0]
            diff_vert = len(img) - start_h


    # ---------------------------------------- VERTICAL CROP -------------------------------

    if(len(img[0]) - valley_position_vert[len(valley_position_vert)-1] < 15):
        diff_hor = valley_position_vert[len(valley_position_vert)-1] - valley_position_vert[0]
    else:
        diff_hor = len(img[0]) - valley_position_vert[0]

    rect = patches.Rectangle((valley_position_vert[0], start_h),diff_hor,diff_vert,linewidth=2,edgecolor='r',facecolor='none')

    crop_image = im_bw[start_h:start_h+diff_vert, valley_position_vert[0]:valley_position_vert[0] + diff_hor]

    resized_image = cv2.resize(crop_image, (450, 150))

    (thresh, crop_image) = cv2.threshold(crop_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # Binarization

    return crop_image
######################################### CORRECT OUTPUT TEXT #########################################################
def test_string(result):
    aux = result

    # print('F = {} and L= {}'.format(result[:3], result[4:8]))
    for char in result:
        if char == ')':
            result = result.replace(')', 'i')
        if char == ':':
            result = result.replace(':', 'j')
        if char == '/':
            result = result.replace('/', '1')
        if char == '!':
            result = result.replace('!', '1')
        if char == '_':
            result = result.replace('_', '')
        if char == ' ':
            result = result.replace(' ', '')
        if char == '.':
            result = result.replace('.', '')

    if re.findall('[^A-Za-z]', result[:3]):
        for char in result[:3]:
            if char == '1':
                result = result[:3].replace('1', 'i')
                result = result + aux[3:8]
            if char == '0':
                result = result[:3].replace('0', 'q')
                result = result + aux[3:8]
            if char == '8':
                result = result[:3].replace('8', 'b')
                result = result+aux[3:8]
            if char == ')':
                result = result[:3].replace(')', 'i')
                result = result+aux[3:8]
            if char == ':':
                result = result[:3].replace(':', 'j')
                result = result+aux[3:8]
        #print('AJUSTE LETRA')
        aux = result
    if re.findall('[^z0-9]', result[4:8]):
        for char in result[4:8]:
            if char == ')':
                result = result[4:8].replace(')', '1')
                result = aux[:4] + result
            if char == '!':
                result = result[4:8].replace('!', '1')
                result = aux[:4]+result
            if char == 'i':
                result = result[4:8].replace('i', '1')
                result = aux[:4]+result
            if char == '/':
                result = result[4:8].replace('/', '1')
                result = aux[:4]+result
        #print('AJUSTE NUMERO')

    if len(result) == 8:
        if result[3] != '-':
            result = result[:3]+'-'+result[4:]
            #print('AJUSTE TRACO')
        print('Placa = {}'.format(result.upper()))
    else:
        result = result.replace(result[2], "")
        print('Placa = {}'.format(result.upper()))
        #print('ERRO TAMANHO')
    #return
########################################## CHAR RECOGNITION ###########################################################
def plateRecognition(image):
    config = ("-l eng --oem 1 --psm 7")
    text = pytesseract.image_to_string(image, config=config)
    test_string(text)
    return text

########################################## MAIN ###########################################################
diretorio = 'venv/Images/'                                  # Diretório com as imagens
files = glob.glob(diretorio+'Fotos_carros/*.jpg');     # Adiciona a uma lista todos os arquivos jpg
cont = 1
acc_fp = 0
acc_pro = 0
acc_pr = 0
for plate_file in files:                             # Percorre a lista com as imagens

    start_fp = time.process_time()
    plate = findPlate(plate_file)                    # Find plate in original image
    end_fp = time.process_time()

    start_pro = time.process_time()
    chars = projection(plate)                        # Find chars to be recognized
    end_pro = time.process_time()

    start_pr = time.process_time()
    plate_str = plateRecognition(chars)
    end_pr = time.process_time()
    cont+=1

    acc_fp += (end_fp - start_fp)
    acc_pro += (end_pro - start_pro)
    acc_pr += (end_pr - start_pr)

time_fp = acc_fp/cont
time_pro = acc_pro/cont
time_pr = acc_pr/cont

print('TIME FP:', time_fp)
print('TIME PRO:', time_pro)
print('TIME PR:', time_pr)
    #print("\n")
    #print(plate_str)
#plt.imshow(chars, cmap='gray')
#plt.show()
