# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:14:32 2023

@author: Cihat Kaya
"""
# For running inference on the TF-Hub module.
import tensorflow as tf
import tensorflow_hub as hub

# For measuring the inference time.
import time

# For processing
from Image import ImageProcess
from Image import Boxes




def load_img(path):
    # read the file
    img = tf.io.read_file(path)
    
    # convert to a tensor
    img = tf.image.decode_jpeg(img, channels=3)
    
    return img


def run_detector(detector, path):
    '''
    Runs inference on a local file using an object detection model.
    
    Args:
        detector (model) -- an object detection model loaded from TF Hub
        path (string) -- path to an image saved locally
    '''
    
    # load an image tensor from a local file path
    img = load_img(path)

    # add a batch dimension in front of the tensor
    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    
    # run inference using the model
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    # save the results in a dictionary
    result = {key:value.numpy() for key,value in result.items()}

    # print results
    print("Found %d objects." % len(result["detection_scores"]))
    print("Inference time: ", end_time-start_time)

    # draw predicted boxes over the image
    image_with_boxes = Boxes().draw_boxes(
      img.numpy(), result["detection_boxes"],
      result["detection_class_entities"], result["detection_scores"])

    # display the image
    ImageProcess().display_image(image_with_boxes)






# Print Tensorflow version
print(tf.__version__)


print("This part takes a few minutes to download model")
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
model = hub.load(module_handle)
print("model signature keys:", model.signatures.keys())

detector = model.signatures['default']


url = "https://upload.wikimedia.org/wikipedia/commons/5/58/2018_IMG_8253_Helsinki%2C_Finland_%2840249531641%29.jpg"
downloaded_image_path = ImageProcess().download_and_resize_image(url, 1000, 1000,display = True)

run_detector(detector, downloaded_image_path)

















