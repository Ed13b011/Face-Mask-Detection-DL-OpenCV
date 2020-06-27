# import modules
from keras.models import load_model
import cv2
import numpy as np
from tensorflow import *

## load model and haarcascade frontal face file
model = load_model('model-010.model')
face_cls = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# capture videos and show rectangular box
source = cv2.VideoCapture(0)
label_dict = {0: 'Mask', 1: 'No Mask'}
color_dict = {0:(0,255,0), 1:(0,0,255)}

while(True):
    ret, img = source.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# converting colored image to grayscale
    faces = face_cls.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # detectMultiScale-> method to search for the face rectangle coordinates 
    # scale factor-> decreses the shape of the value by 20% untill the face is found
    # minNeighbors-> cascade classifier uses slides window proces to remove false positives and detect true positives


    for (x,y,w,h) in faces:
        face_img = gray[y:y+w, x:x+w]
        resized = cv2.resize(face_img, (100, 100))
        normalized = resized/255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped) # predictions
        label = np.argmax(result, axis=1)[0] # labels (0 or 1)

        cv2.rectangle(img, (x,y), (x+w, y+h), color_dict[label], 2)
        cv2.rectangle(img, (x,y-40), (x+w,y), color_dict[label], -1)
        cv2.putText(img, label_dict[label], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)
        # create rectangle and (255, 255, 255) for RGB value of rectangle outline and 2 is for width of the rectangle


    cv2.imshow('LIVE', img)
    # wait for key-> q to stop the program
    #k = cv2.waitKey(30)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

print('came out of the loop')
cv2.destroyAllWindows()
source.release()
