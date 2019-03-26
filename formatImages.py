import cv2
import os
import numpy as np

def erode(filename, path, tag):
    img = cv2.imread(path+"/"+filename,0)
    _, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations = 1)
    cv2.imwrite("{}/{}formated/{}".format(folderpath, tag, filename), thresh) #create folder with name {tag}formated (exclude brackets) before running file

folderpath = "extracted_images/-"
tag = "-"
c = 0
for file in os.listdir(folderpath):
    if file.split(".")[-1] in ["jpg", "jpeg", "png"]:
        erode(file, folderpath, tag)
        print("{}done".format(c+1))
        c+=1