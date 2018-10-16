import numpy as np
import cv2 as cv2
import os

def getDist(x1,x2):
    x2 = np.reshape(x2,x1.shape)
    d = np.absolute(x1-x2)
    d = sum(d)
    return d

def knn(training_data,testing_data,k=5):	
	dist = []
	for key in training_data.keys():
		for i in range(training_data[key].shape[0]):
			dist.append((getDist(training_data[key][i],testing_data),key))
	dist = np.array(dist)
	dist = dist[dist[:,0].argsort()]
	dist = dist[:k]
	dist = np.array(np.unique(dist[:,1],return_counts=True))
	dist = dist[0,np.argmax(dist[1,:])]
	return dist



#Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
a = os.listdir('./data')

train = {}
for arr in a:
    if arr.endswith('.npy'):
        data = np.load('./data/'+arr)
        arr = arr[:-4]
        train[arr] = data

print "Data Loaded"


while True:
	ret,frame = cap.read()

	if ret==False:
		continue
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	if len(faces)==0:
		continue
	faces = sorted(faces,key=lambda f:f[2]*f[3])

	# Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

	#Extract (Crop out the required face) : Region of Interest
	offset = 10
	face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
	face_section = cv2.resize(face_section,(100,100))

	face_data = np.asarray(face_section)
	face_data = face_data.reshape((1,-1))

	name = knn(train,face_data)
	cv2.putText(frame,name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
	cv2.imshow("Frame",frame)
	
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()






