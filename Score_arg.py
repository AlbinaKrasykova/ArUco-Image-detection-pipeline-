import argparse

import cv2 
from PIL import Image
import numpy as np

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters 
detector = cv2.aruco.ArucoDetector(dictionary)

#0. Function that Generates dictionary with images from the file 
# Key: 9_angle_6_.png, Value: <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1920x1080 at 0x178C0FF6BF0>

import os
from PIL import Image

def load_images(directory):
    image_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            try:
                image = Image.open(image_path)
                image_dict[filename] = image
            except OSError:
                print(f"Unable to open image: {filename}")
    return image_dict

image_directory = "D:\AI research internship\opencv_scripts\data_set"



#2 CREATE an ARRAY with Tag original ID's by parcing the key string, as getting first digits of it and save to an array 'Original ID' - Key: 9_angle_6_.png - > 9
def original_id(image_dict):
    digit_array = []
    for key in image_dict.keys():
        first_digit = next((char for char in key if char.isdigit()), None)
        if first_digit:
            digit_array.append(int(first_digit))
    return digit_array




#3 CREATE an ARRAY with Tags that were predicted 

def predicted_id2_c(image_dict):
    p_id_arr = []
    for ids, images in image_dict.items():
        img = images
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        clache = cv2.createCLAHE(clipLimit=40)
        frame_clache = clache.apply(gray)           
        th3 = cv2.adaptiveThreshold(frame_clache, 125, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 51, 1)
        blurred = cv2.GaussianBlur(th3, (21, 21), 0)
        flipped = cv2.flip(blurred, 1)
        _, ids, _ = detector.detectMarkers(flipped)
        p_id_arr.append(ids)
    return p_id_arr


#print(p_id_arr = predicted_id(image_dict))

def combined_2(img1, img2):


 height = max(img1.shape[0], img2.shape[0])
 img1 = cv2.resize(img1, (int(img1.shape[1] * height / img1.shape[0]), height))
 img2 = cv2.resize(img2, (int(img2.shape[1] * height / img2.shape[0]), height))

# Create a new image with double width
 combined_image = Image.new("RGB", (img1.shape[1] + img2.shape[1], height))

# Paste the images side by side
 combined_image.paste(Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)), (0, 0))
 combined_image.paste(Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), (img1.shape[1], 0))

 return combined_image



def score(original_id, predicted_id):
  scores = 0
  ratio = 0
  img_count = len(original_id)
  for id in  original_id:
      
      for ids in predicted_id:
          
          if id == ids:
           scores+=1
           
  ratio = img_count/scores  
  return img_count,scores,ratio


def score_c(original_id, predicted_id):
 scores = 0
 ratio = 0
 img_count = len(original_id)

 for id in original_id:
    for ids in predicted_id:
        if id == ids:
            scores += 1

 if scores != 0:
    ratio = img_count / scores

 return img_count, scores, ratio


def info(img_count,score,ratio):
     img_count = img_count
     score = score
     ratio = ratio
     print(f"Image processing pipeline scored at {score}")
     print(f"Out of {img_count} images {score} were predicted")

     


def processed(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clache = cv2.createCLAHE(clipLimit=40)
    frame_clache = clache.apply(gray)
    th3 = cv2.adaptiveThreshold(frame_clache, 125, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 51, 1)
    blurred = cv2.GaussianBlur(th3, (21, 21), 0)
    flipped = cv2.flip(blurred, 1)
    return flipped



cap = cv2.VideoCapture(0)









parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
parser.add_argument('--score', action='store_true')
parser.add_argument('--score_with_images', action='store_true')
#parser.add_argument('score', type=str, help='Score the pipeline' )
#parser.add_argument('score with showing images', type=str, help='Scores the pipeline, shwoing all the images')
args = parser.parse_args()
if args.score:
# 1 -  Load the data

  image_dict = load_images(image_directory)


  original_ids = original_id(image_dict)


  predicted_ids = predicted_id2_c(image_dict)

 #Code Check predicted_ids - works 

  print(predicted_ids)

  img_count,scores,ratio = score_c(original_ids,predicted_ids)

  info(img_count,scores,ratio)


elif args.score_with_images:
 
   image_dict = load_images(image_directory)


   original_ids = original_id(image_dict)


   predicted_ids = predicted_id2_c(image_dict)

 #Code Check predicted_ids - works 

   print(predicted_ids)

   img_count,scores,ratio = score_c(original_ids,predicted_ids)

   info(img_count,scores,ratio)
 
   while(True):
     ret, frame = cap.read()
     for key, values in image_dict.items():
         img1 = cv2.imread(key)
         img2 = processed(img1)
         
         combined = combined_2(img1, img2)
         combined.show()
   
    
     
     if cv2.waitKey(1) & 0xFF == ord('x'):
         break
     
     
cap.release()
cv2.destroyAllWindows()
