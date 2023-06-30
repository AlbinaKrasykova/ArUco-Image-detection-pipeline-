import argparse

import cv2 
from PIL import Image
import numpy as np
from shadow_highlight_correction import correction

calibration_frame = None


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters 
detector = cv2.aruco.ArucoDetector(dictionary)



import os
from PIL import Image


directory = r'D:\AI research internship\opencv_scripts\data_set' 

def image_generator(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            try:
                image = cv2.imread(image_path)
                yield filename, image
            except OSError:
                print(f"Unable to open image: {filename}")

def add_images_to_dictionary(directory):
    image_dict = {}
    for filename, image in image_generator(directory):
        image_dict[filename] = image

    return image_dict

image_dict = add_images_to_dictionary(directory)



import re

def original_id(image_dict):
    digit_array = []
    pattern = r'^\d{1,2}'
    
    for key in image_dict.keys():
         match = re.match(pattern, key)
         if match:
             first_digits = int(match.group())
             digit_array.append(first_digits)
    print(digit_array) 
    print(image_dict.keys())   
    print('len of the array is', len(digit_array)) 
    return digit_array

def contrast(image, clahe):
    clahe_out = clahe.apply(image)
    return clahe_out


#blur

def blur(image, kernel):
    return cv2.blur(image, kernel)

#threshold


def threshold(image, px_val):
    # ret, thresholded = cv2.threshold(image, px_val, 255, cv2.THRESH_TOZERO)
    # ret, thresholded = cv2.threshold(thresholded, px_val, 1, cv2.THRESH_TOZERO_INV)
    thresholded = cv2.adaptiveThreshold(image, 
                                        255, 
                                        cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 
                                        89, 
                                        2)
    return thresholded

#THRESHOLD_PX_VAL 

THRESHOLD_PX_VAL = 100

CLIP_LIMIT = 20.0 


def drawMarkers(img, corners, ids, borderColor=(255,0,0), thickness=25):
    if ids == []:
        ids = ["R"] * 100
    for i, corner in enumerate(corners):
        if ids[i] == 17:
            continue
        corner = corner.astype(int)
        cv2.line(img, (corner[0][0][0], corner[0][0][1]), (corner[0][1][0], corner[0][1][1]), borderColor, thickness)
        cv2.line(img, (corner[0][1][0], corner[0][1][1]), (corner[0][2][0], corner[0][2][1]), borderColor, thickness)
        cv2.line(img, (corner[0][2][0], corner[0][2][1]), (corner[0][3][0], corner[0][3][1]), borderColor, thickness)
        cv2.line(img, (corner[0][3][0], corner[0][3][1]), (corner[0][0][0], corner[0][0][1]), borderColor, thickness)
        cv2.putText(img, str(ids[i]), (corner[0][0][0], corner[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5, cv2.LINE_AA)
    return img

clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(5, 5)) 
def invert(img):
    image_not = cv2.bitwise_not(img)
    return image_not


def A_detect(image, draw_rejected = False):
    (corners, ids, rejected) =  detector.detectMarkers(image)
    (corners_inv, ids_inv, rejected_inv) =  detector.detectMarkers(invert(image))
    (corners_hflip, ids_hflip, _) = detector.detectMarkers(invert(cv2.flip(image,1)))

    back_to_color = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    if draw_rejected:
        detected = drawMarkers(back_to_color.copy(), rejected, [], borderColor=(255, 0, 0))
        detected = drawMarkers(detected.copy(), rejected_inv, [], borderColor=(255, 0, 0))
        detected = drawMarkers(detected.copy(), corners, ids, borderColor=(83, 235, 52))
        detected = drawMarkers(detected.copy(), corners_inv, ids_inv, borderColor=(83, 235, 52))
    else:
        detected = drawMarkers(back_to_color.copy(), corners, ids, borderColor=(83, 235, 52))
        detected = drawMarkers(detected.copy(), corners_inv, ids_inv, borderColor=(83, 235, 52))
        detected = drawMarkers(cv2.flip(detected.copy(), 1), corners_hflip, ids_hflip, borderColor=(83, 235, 52))

    
    return ids_hflip

''''
def predict(image_dict, pipeline):
    for ids, img in image_dict:
        predicted = pipeline(img)
        # comparison ... 
'''

def A_predicted_id2(image_dict):
    p_id_arr = []
    for ids, img in image_dict.items():
         
        img_corrected = correction(img, 0, 0, 0, 0.6, 0.6, 30, .3)
        img_gray = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2GRAY)
        if calibration_frame is not None:
            img_norm = img_gray - calibration_frame
        else:
            img_norm = img_gray

        img_contrast_enhanced = contrast(img_norm, clahe)
        img_blurred = blur(img_contrast_enhanced, (5, 5))
        img_thresholded = threshold(img_blurred, THRESHOLD_PX_VAL)
        # with no flip  - I got 9% score
        # with flip - I got 180%, 41 out of 74
        flipped = cv2.flip(img_thresholded, 1)
        ids = A_detect(flipped)
        p_id_arr.append(ids)
        #print(type(ids))
        #print(ids)
    print('Amrits array with predicted id is: ', p_id_arr)
    print(len(p_id_arr))
    return p_id_arr

def score1(original_id, predicted_id):
 scores = 0
 ratio = 0
 img_count = len(original_id)

 for id in original_id:
    for ids in predicted_id:
        if id == ids:
            scores += 1

 if scores != 0:
    ratio =  scores/img_count

 return img_count, scores, ratio

def score(original_id, predicted_id):
    scores = 0
    img_count = len(original_id)

    for id in original_id:
        if id in predicted_id:
            scores += 1

    ratio = scores / img_count if img_count != 0 else 0

    return img_count, scores, ratio

#6 Display ratio based on the precious scoring subfunction  - ✔


def info(img_count,score,ratio):
     img_count = img_count
     score = score
     ratio = ratio
     print(f"Image processing pipeline scored at {int(ratio*100)} %")
     print(f"Out of {img_count} images {score} were predicted")
     print('Score:', ratio*100, '%')

#image_dict = load_images(image_directory)


original_ids = original_id(image_dict)

   #A_ppln - 

A_predicted_ids = A_predicted_id2(image_dict)

img_count,scores,ratio = score(original_ids,A_predicted_ids)

print(' ')
   
print('Printing Amrits ppln score')
   
print(' ')

info(img_count,scores,ratio)
print(original_ids)
print(A_predicted_ids)

print(len(original_ids),len(A_predicted_ids))