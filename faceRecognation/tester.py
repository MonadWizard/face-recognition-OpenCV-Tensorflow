# pip install opencv-contrib-python

import os
import cv2
import numpy as np
import faceRecognition as fr


# for detect from picture
test_img = cv2.imread('\\media\\monad\\DataScience\\code\\opencv\\faceRecognation\\TestImages\\test.jpg') 
faces_detected, gray_img = fr.faceDetection(test_img)
print('face_detected:', faces_detected)


"""
for (x,y,w,h) in faces_detected:
    cv2.rectangle(test_img, (x,y), (x+w,y+h),(255,0,0),thickness=5)
    
resized_img = cv2.resize(test_img,(1000,700))
cv2.imshow("resize face detection happend",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

'''
# call label for training data for training face id for train new face and save to yml

faces, faceID = fr.labels_for_training_data("D:\\code\\opencv\\faceRecognation\\TrainingImages")
face_recognizer = fr.train_classifier(faces,faceID)
face_recognizer.write('trainingData.yml')
'''
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('\\media\\monad\\DataScience\\code\\opencv\\faceRecognation\\trainingData.yml')




name={0:"Rakib", 1:"Rakib2"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]

    if(confidence<37):#If confidence more than 37 then don't print predicted face text on screen
        continue

    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,1000))
cv2.imshow("face dtecetion tutorial",resized_img)
cv2.waitKey(0)#Waits indefinitely until a key is pressed
cv2.destroyAllWindows()

    



