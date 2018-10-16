import cv2 
import numpy as np

#Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

ret,frame = cap.read()
ret,frame = cap.read()
ret,frame = cap.read()
ret,frame = cap.read()

while True:
	ret,frame = cap.read()

	if ret==False:
		continue
	
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	
	if len(faces)==0:
		continue

	for face in faces:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),)1

	cv2.imshow("Frame",frame)
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()






