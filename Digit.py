import cv2
import numpy as np 
import keras

model = keras.models.load_model("CNN_Scored_99.357.model") #load Model
path = "0_4JRXMGNuA7FLG2KR.png" #path to image

#load and convert to gray
img = cv2.imread(path) 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#thresholding
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Add dilation to thicken
kernel = np.ones((3,3),np.uint8)
thresh = cv2.dilate(thresh, kernel, iterations = 1)

_, contours, heir = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #contour detection

for index, contour in enumerate(contours):
	if heir[0][index][-1] == -1: #check if contour has no parent 
		cv2.drawContours(img, [contour], -1, (0,0,255))
		x,y,w,h = cv2.boundingRect(contour) 
		x, y , w, h = x, y, x+w, y+h
		num = thresh[y : h, x : w]

		# Add black surrounding to increase accuracy
		top = int(0.4 * num.shape[1])
		bottom = top
		left = int(0.4 * num.shape[0])
		right = left
		num = cv2.copyMakeBorder(num, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0,0,0))

		#prepare for prediction
		num = cv2.resize(num, (28,28))
		cv2.imshow(str(index),num) ##this image will be passed to model
		num = np.array(num)/255 #scaling pixel values
		num = num.reshape(1, 28, 28, 1)
		answer = np.argmax(model.predict(num), axis =1) #prediction
		#print(answer)
		cv2.rectangle(img, (x, y), (w, h), (255,0,0)) 
		cv2.putText(img, str(answer), (x+2, y+2),cv2.FONT_HERSHEY_COMPLEX , 0.7, (0, 0, 0), 1)

cv2.imshow("Window",img)
cv2.waitKey()
