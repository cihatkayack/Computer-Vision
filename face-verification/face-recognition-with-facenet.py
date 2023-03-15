from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import PIL
import cv2
import os
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

print("Tensorflow version " + tf.__version__)


from keras.models import load_model


def triplet_loss(y_true, y_pred, alpha = 0.2):

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    #Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis=-1)
    #Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),axis=-1)
    #subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)
    #Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss,0.0))
    
    return loss


def img_to_encoding(image, model):
    target_size=(160, 160, 3)
    img = np.resize(image,target_size)
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)

def image_processing_for_database(image_path, cascade_face):
    if type(image_path) == str:
        img = cv2.imread(image_path)
    else:
        img = np.array(image_path)
    face = cascade_face.detectMultiScale(img, 
        scaleFactor = 1.3,
        minNeighbors = 5,
        minSize = (30,30)
    )
    new_image = img
    for (x,y,w,h) in face:
        
        new_image = img[y:y+h+50, x-30:x+h+50]
    img = Image.fromarray(new_image)
    return img


def database():
    database = {}
    image_list = os.listdir("images/")
    for image in image_list:
        if "test" not in image:
            name = image.split(".")[0]
            image = image_processing_for_database("images/{}".format(image),cascade_face)
            database[name] = img_to_encoding(image, FRmodel)
    return database


def who_is_it(image_path, database, model):
    #Compute the target "encoding" for the image.
    # image_path = np.flip(image_path)
    img = image_processing_for_database(image_path, cascade_face)
    encoding = img_to_encoding(img, model)
    
    
    ##Find the closest encoding 
    min_dist = 100
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name 
    
        
    return min_dist, identity


# load the model
model = load_model('facenet_keras.h5')
print(model.summary())
print(model.inputs)
print(model.outputs)

FRmodel = model
cascade_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

database_ = database()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2000)

while True:
    ret, img = cap.read()
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = cascade_face.detectMultiScale(
        g,
        scaleFactor = 1.3,
        minNeighbors = 5,
        minSize = (30,30)
        )
    
    for (x,y,w,h) in f:
        cv2.rectangle(img, (x,y-20), (x+w,y+h+20),(255,0,0),2)
        gray_r = g[y:y+h, x:x+h]
        identify = who_is_it(img, database_, FRmodel)
        text = str(identify[0]) + " " + str(identify[1])
        cv2.putText(img, text, (x,y-30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
        
    cv2.imshow("video", img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.relesase()
cv2.destroyAllWindows()









































































