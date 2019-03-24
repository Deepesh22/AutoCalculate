import cv2
import numpy as np 
import keras

model = keras.models.load_model("CNN_Scored_99.357.model")

path = "IMG20190316224800-min.jpg"

img = cv2.imread(path)
img = cv2.resize(img, (512, 512))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#thresholding
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

_, contours, heir = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #contour detection

#cv2.imshow("t", thresh)

for index, contour in enumerate(contours):
	if heir[0][index][-1] == -1: #check if contour has no parent 
		cv2.drawContours(img, [contour], -1, (0,0,255))
		x,y,w,h = cv2.boundingRect(contour) 
		n = 12
		x, y , w, h = x-n, y-n, x+w+n, y+h+n
		num = thresh[y : h, x : w]
		cv2.imshow(str(index),num)
		num = cv2.resize(num, (28,28))
		num = np.array(num)/255
		num = num.reshape(1, 28, 28, 1)
		answer = dict()
		answer[np.max(model.predict(num), axis =1)[0]] = np.argmax(model.predict(num), axis =1)
		k = max(answer.keys())
		answer = answer[k]
		cv2.rectangle(img, (x, y), (w, h), (255,0,0)) 
		cv2.putText(img, str(answer), (x+2, y+2),cv2.FONT_HERSHEY_COMPLEX , 0.7, (0, 0, 0), 1)

cv2.imshow("Window",img)
cv2.waitKey()
