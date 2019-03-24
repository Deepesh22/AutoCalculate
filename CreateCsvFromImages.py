import pandas as pd
import cv2
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description= "Script to make csv file from given images in a folder.")
parser.add_argument("folderPath", help="Enter relative path to folder containing images")
parser.add_argument("label", help="Label of images")
args = parser.parse_args()

images = os.listdir(args.folderPath)

count = 0
data = []
for img in images:
    if img.split(".")[-1] in ["jpg", "jpeg", "png"]:
        path = args.folderPath + "/" + img
        image = cv2.imread(path)
        cv2.imshow("", image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #removing rgb channel
        image = cv2.resize(image, (28,28)) #resizing
        data.append([*list(np.array(image).flatten()), args.label])
        count +=1
        print("{} done".format(count))

df = pd.DataFrame(data)
filename = "{} dataset.csv".format(args.label)
df.to_csv(filename)
