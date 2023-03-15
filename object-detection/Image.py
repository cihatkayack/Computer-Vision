# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:15:23 2023

@author: Cihat Kaya
"""

import random

# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

# For measuring the inference time.
import time

class ImageProcess():
    
    def display_image(self,image):
        """
        Displays an image 
        """
        figure = plt.figure(figsize=(15,15))
        plt.imshow(image)
    
    def download_and_resize_image(self,url, new_width = 256, new_height = 256, display = False):
        
        #create a temporary file
        _, filename = tempfile.mkstemp(suffix = "jpg")
        
        response = urlopen(url)
        image_data = response.read()
        
        #puts the image data in memory buffer
        image_data = BytesIO(image_data)
        
        pil_image = Image.open(image_data)
        
        #resize image
        pil_image = ImageOps.fit(pil_image,(new_width,new_height), Image.ANTIALIAS)
        
        #save image
        pil_image_rgb = pil_image.convert("RGB")
        pil_image_rgb.save(filename, format="JPEG", quality = 90)
        
        print("Image dowloaded at ", filename)
        
        if display:
            self.display_image(pil_image)
        
        return filename
    
    
 
class Boxes():
    
    def draw_box_on_image(self,image,y1,x1,y2,x2,color,font, thickness = 4, display_str_list =()):
        """
        Args:
            image -- the image object
            y1 -- bounding box coordinate
            x1 -- bounding box coordinate
            y2 -- bounding box coordinate
            x2 -- bounding box coordinate
            color -- color for the bounding box edges
            font -- font for class label
            thickness -- edge thickness of the bounding box
            display_str_list -- class labels for each object detected

        """
        #The ImageDraw module provides simple 2D graphics for Image objects. 
        #You can use this module to create new images, annotate or retouch existing images
        draw = ImageDraw.Draw(image)
        image_width, image_heigth = image.size
        
        # scale the bounding box coordinates
        (left, right, top, bottom) = (x1 * image_width, x2 * image_width,
                                      y1 * image_heigth, y2 * image_heigth)
        
        # define the four edges of the detection box
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                 (left, top)],
                width=thickness,
                fill=color)
        
        #string
        display_str_heights = [font.getsize(string)[1] for string in display_str_list]
        
        # Each display_str has a top and bottom margin of 0.05x
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
        
        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = top + total_display_str_height
            
        # Reverse list and print from bottom to top
        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle([(left, text_bottom-text_height-2*margin),
                            (left + text_width, text_bottom)],
                           fill = color)
            draw.text((left + margin, text_bottom - text_height - margin),
                      display_str,
                      fill = "black",
                      font=font)
            
            text_bottom -= text_height - 2 * margin
            
            
            
        
        
        
    
    def draw_boxes(self,image,boxes,class_names, scores,max_boxes = 10, min_score = 0.001):
        colors = list(ImageColor.colormap.values())
        font = ImageFont.load_default()
        
        print(f"boxes shape: {boxes.shape[0]}, , max boxes: {max_boxes}")
        for i in range(min(boxes.shape[0], max_boxes)):
            if scores[i] >= min_score:
                y1, x1, y2, x2 = tuple(boxes[i])
                print_classname = "{}: {}%".format(class_names[i].decode("ascii"),
                                                   int(100*scores[i]))
                color = random.choice(colors)
                image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
                self.draw_box_on_image(image_pil,
                                           y1,
                                           x1,
                                           y2,
                                           x2,
                                           color,
                                           font,
                                           display_str_list=[print_classname])
                #draw one bounding box and overplay the class label and score onto the the image
        
        np.copyto(image, np.array(np.array(image_pil)))
        
        return image
                
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    