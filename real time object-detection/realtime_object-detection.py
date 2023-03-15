# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:19:08 2023

@author: Cihat Kaya
"""
from imageai.Detection import ObjectDetection
import cv2

modelpath = "yolov3.pt"


model = ObjectDetection()
model.setModelTypeAsYOLOv3()
model.setModelPath(modelpath)
model.loadModel()


cam = cv2.VideoCapture(0) 
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)


while True:
    ## read frames
    ret, img = cam.read()
    ## predict yolo
    detections = model.detectObjectsFromImage(input_image=img,
                      minimum_percentage_probability=70
                      )
    ## display predictions
    for detect in detections:
        x = detect["box_points"][0]
        y = detect["box_points"][1]
        w = detect["box_points"][2]
        h = detect["box_points"][3]
        text = detect["name"] + ": " + str(detect["percentage_probability"])
        cv2.rectangle(img, (x,y), (w,h),(255,0,0),2)
        cv2.putText(img, text, (x,y-30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
    cv2.imshow("video", img)
    ## press q or Esc to quit    
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break
## close camera
cam.release()
cv2.destroyAllWindows()




