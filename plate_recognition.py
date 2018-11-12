import cv2
import numpy as np
import pytesseract
import glob
from PIL import Image
cont = 0
diretorio = 'Images/'                                  # Diretório com as imagens
files = glob.glob(diretorio+'Fotos_carros/*.jpg');     # Adiciona a uma lista todos os arquivos jpg
plate_str = list()

for plate_file in files:                             # Percorre a lista com as imagens

    img = cv2.imread(plate_file)
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

    #cv2.drawContours(img, contours, blob_num, (255,0,0), 2)

    #  ------ Medium stages ------------------------------------------------ #
    #cv2.imshow('Color image', img)
    #cv2.imshow('Grayscale', img_grayscale)
    #cv2.imshow('Morphological Opening',opening)
    #cv2.imshow('Subtraction', subtract)
    #cv2.imshow('Binarization', im_bw)
    #cv2.imshow('Morphological Opening 2', opening_2)
    #cv2.imshow('Morphological Close', plate)
    #cv2.imshow('Plate Extracted', plate_extracted)
    diretorio_escrita = diretorio+'extracted/'+str(cont)+'_extracted.jpg'
    cv2.imwrite(diretorio_escrita,plate_extracted)
"""
    #  ------ Char recognition ------------------------------------------------ #

    largura, altura = plate_extracted.size

    print("A:", altura, "L: ", largura)

    phrase = pytesseract.image_to_string(Image.open(diretorio_escrita))
    plate_str.append(phrase)
    cont+=1

with open('Plate_extrated.txt', 'w',encoding = 'utf8') as f:
    for item in plate_str:
        f.write("%s\n" % item)
"""
