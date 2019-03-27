import cv2
import numpy as np
import os

folderpath ="D:\AutoCalculate-segmentation\AutoCalculate\-formated"
path = "D:\AutoCalculate-segmentation\AutoCalculate"
tag = "/"
for file in os.listdir(folderpath):
    if file.split(".")[-1] in ["jpg", "jpeg", "png"]:
            img = cv2.imread(folderpath+"/"+file,0)
            ro = cv2.getRotationMatrix2D((14,14), 45, 1.0)
            img2 = cv2.warpAffine(img, ro, img.shape[1::-1], flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
            cv2.imwrite("{}\{}formated\{}".format(path, tag, file),img2)
            print("Done")