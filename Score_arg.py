import argparse

import cv2 
from PIL import Image
import numpy as np
from shadow_highlight_correction import correction
import math

calibration_frame = None

#I was trying to import functions but I got work more of functions I importing like these which imports images 
#from Amrits_ppln import correction , contrast, threshold, calibration_frame, clahe, blur, THRESHOLD_PX_VAL



''''
Image processing pipeline scored at 180 %
Out of 41 images 74 were predicted
Score: 180.4878048780488 %
#FIXED Monday 7/3 - changed scoring function 
#changed tabs order 

Total images 41
Predicted ing: 41
29

Printing Amrits ppln score

Image processing pipeline scored at 70 %
Out of 41 images 29 were predicted
Score: 70.73170731707317 %
'''



#from Fahads_ppln import Fdetection, Ftransformation  - for some rrn I got the img wth an id as well, so its imports everything I have ✔
#from save_frame  import detect_F, process_frame_F ✔


#GOAL 1: score function which scores ing pppln according to how well it perfomed  ✔

#Result: does run, but 't detect images BUT needs a confusion matrix, grid search and laod images in batches (as for 2K dataset) ? 

#2: takes both 2dprinted tags(test) and 3d printed tags and detects the id's correctl, scores it and return the ratio ✔

#3 : check the datatype whch is return by the function, rewrite all the fucntion manually - Done - ✔

#4: check/score 2 pplns - Fahds (4.1) - ✔ & Amrit (4.2) - ✔

#5 : Implement My, F, A pipelines as an option in the command line - ✔

#6: Build a better datset, test pplns on it 2500 ✔

#7: Dataset from different angles/calculating the distance - ✔

#  image processing in batches, implement and rewrite functions 

# NEEDS TO BE DONE 7: Confusion matrix implnt (precision and recall)




#ERRORS: 


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters 
detector = cv2.aruco.ArucoDetector(dictionary)



import os
from PIL import Image

#0. Function that Generates dictionary with images from the file - ✔
# Key: 9_angle_6_.png, Value: <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1920x1080 at 0x178C0FF6BF0>

# Function Requires n Output : path to the folder, folder with images-files 

""

''''
def load_images(directory):
    image_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            try:
                image = cv2.imread(image_path)
                image_dict[filename] = image
            except OSError:
                print(f"Unable to open image: {filename}")
           # finally:
           #     del image  # Release the memory for the image

    return image_dict

'''

directory = r'D:\AI research internship\opencv_scripts\n_updwn_angl'

""""

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

"""

#1 Loads images in batches 

def load_images_in_batches(directory, batch_size, batch_index=0):
    
    image_files = os.listdir(directory)
    total_images = len(image_files)
    num_batches = math.ceil(total_images / batch_size)

    for i in range(num_batches):
        batch_index +=1
        start_index = i * batch_size
        end_index = min(start_index + batch_size, total_images)
        batch_files = image_files[start_index:end_index]
        
        batch_images = {}
        for file in batch_files:
            
            image_path = os.path.join(directory, file)
            image = cv2.imread(image_path)
            image_name = os.path.splitext(file)[0]  # Extract name without extension
            batch_images[image_name] = image
        print()
        print('Batch numebr: ', batch_index)
        print()
        
        yield batch_images



# p_dataset
# data_set





#2 CREATE an ARRAY with Tag original ID's by parcing the key string, as getting first digits of it and save to an array 'Original ID' - Key: 9_angle_6_.png - > 9

#check type - ✔ returns array of ints

#Info: Original id's of the images which is an array of ints  

#Requirments: image_dict, returns an array of ints,
''''
def original_id(image_dict):
    digit_array = []
    for key in image_dict.keys():
        first_digit = next((char for char in key if char.isdigit()), None)
        if first_digit:
            #convert from str to int 
            first_digit_int = int(first_digit)
            digit_array.append(int(first_digit_int))
            #check datatype 
            print(type(first_digit_int))
            

    return digit_array
    '''


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
    return digit_array



#3 CREATE an ARRAY with Tags that were predicted - ✔

def My_predicted_id(image_dict):
    p_id_arr = []
    for ids, img in image_dict.items():
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clache = cv2.createCLAHE(clipLimit=40)
        frame_clache = clache.apply(gray)           
        th3 = cv2.adaptiveThreshold(frame_clache, 125, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 51, 1)
        blurred = cv2.GaussianBlur(th3, (21, 21), 0)
        flipped = cv2.flip(blurred, 1)
        _, ids, _ = detector.detectMarkers(flipped)
        p_id_arr.append(ids)

    
        #print(type(ids))
    print('My array with predicted id is: ', p_id_arr)
    return p_id_arr

# F_ppln CREATE an ARRAY with Tags that were predicted - ✔

def F_predicted_id2(image_dict):
    p_id_arr = []
    for ids, images in image_dict.items():
        transformation = cv2.cvtColor((images), cv2.COLOR_BGR2GRAY)
        transformation = cv2.bitwise_not(transformation)
        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16, 16))
        transformation = clahe.apply(transformation)
        transformation = cv2.GaussianBlur(transformation, (21, 21), 0)
        _, transformation = cv2.threshold(transformation, 150, 255, cv2.THRESH_BINARY)
        flipped = cv2.flip(transformation, 1)
        _, ids, _ = detector.detectMarkers(flipped)
        p_id_arr.append(ids)
        print(type(ids))
    
    print('Fahads array with the predicted id is: ', p_id_arr)
    return p_id_arr

# A_ppln CREATE an ARRAY with Tags that were predicted - 

#correction

#calibration_frame

#contrast

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
    return p_id_arr










#print(p_id_arr = predicted_id(image_dict))


#4 Function that combines and displays 2 images side by side  - ✔

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


#5 Scoring Function - ✔ +precision and recall 

def calc_p_r(original_ids, predicted_ids):
    true_positive = 0
    false_negative = 0
    false_positive = 0

    for i in range(len(original_ids)):
        if original_ids[i] == predicted_ids[i]:
            true_positive += 1
        else:
            false_negative += 1

    false_negative = len(original_ids) - true_positive

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    return precision, recall



def score(original_id, predicted_id):
    scores = 0
    total = len(original_id)
    predicted_id_count = len(predicted_id)
    print()
    print('Total images:', total)
    print('Predicted images:', predicted_id_count)

    for id in original_id:
        if isinstance(
            
            
            
            
            
            
            id, int) and id in predicted_id:
            scores += 1

    precision, recall = calc_p_r(original_id, predicted_id)
    ratio = (scores / total) * 100
    print('Scores:', scores)
    return scores, total, ratio, precision, recall

#6 Display ratio based on the precious scoring subfunction  - ✔


def info(score,total,ratio,precision, recall):
     total = total
     score = score
     ratio = ratio
     precision=precision
     recall = recall
     print(f"Image processing pipeline scored at {int(ratio)} %")
     print(f"Out of {total} images {score} were predicted")
     print('Score:', ratio, '%')
     print('precision:', precision, '%')
     print('recall:', recall, '%')

     

 

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
parser.add_argument('--my_score', action='store_true')
parser.add_argument('--Fahad_score', action='store_true')
parser.add_argument('--Amrit_score', action='store_true')
parser.add_argument('--score_with_images', action='store_true')
#parser.add_argument('score', type=str, help='Score the pipeline' )
#parser.add_argument('score with showing images', type=str, help='Scores the pipeline, shwoing all the images')
args = parser.parse_args()


if args.my_score:


    batch_size = 150
    directory = r'D:\AI research internship\opencv_scripts\n_updwn_angl'


    print('My ppln score: ')   
    print('')
    for batch in load_images_in_batches(directory, batch_size):
        original_ids = original_id(batch)
        predicted_ids = My_predicted_id(batch)
        img_count, scores, ratio,precision, recall = score(original_ids, predicted_ids)
        info(img_count, scores, ratio,precision, recall)
       
        # Perform further processing on the image or store it as needed
    print("Batch complete")

# 1 -  Load the data - ✔ 

    #image_dict = load_images(image_directory)


 #   original_ids = original_id(image_dict)

# note: MY ppln  Uncomment once switxh to my ppln, and comment F_ppln 
#predicted_ids = predicted_id2_c(image_dict)

#F_ppln - ✔

 #   predicted_ids = My_predicted_id(image_dict)


#    img_count,scores,ratio = score(original_ids,predicted_ids)

#    print(' ')

#    print('Printing My ppln score') 

#   print(' ')

#    info(img_count,scores,ratio)



if args.Fahad_score:
   
   # image_dict = load_images(image_directory)

    #Check 41 img dataset 
   #  directory = r'D:\AI research internship\opencv_scripts\data_set'
    directory = r'D:\AI research internship\opencv_scripts\data_set'
    print('Fahd ppln score: ')   
    print('')
    #original_ids = original_id(image_dict)

# note: MY ppln  Uncomment once switxh to my ppln, and comment F_ppln 
#predicted_ids = predicted_id2_c(image_dict)

#F_ppln - ✔
    for batch in load_images_in_batches(directory, batch_size):
        original_ids = original_id(batch)
        predicted_ids = F_predicted_id2(batch)
        img_count, scores, ratio,precision, recall = score(original_ids, predicted_ids)
        info(img_count, scores, ratio,precision, recall)
   

if args.Amrit_score:
   
   # image_dict = load_images(image_directory)
   
    print('Amrit ppln score: ')  
    print('')

    batch_size = 150

   # Check 41 img dataset 
    directory = r'D:\AI research internship\opencv_scripts\data_set'

    def A_predicted_id(image_dict):
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
            flipped = cv2.flip(img_thresholded, 1)
            ids = A_detect(flipped)
            p_id_arr.append(ids)
        print('Amrits array with predicted id is: ', p_id_arr)
        print(len(p_id_arr))
        return p_id_arr

    for batch in load_images_in_batches(directory, batch_size):
        original_ids = original_id(batch)
        print(original_ids)
        predicted_ids = A_predicted_id(batch)
       
        new_arr_predicted_ids = [int(x[0, 0]) if x is not None else None for x in predicted_ids]
        print(new_arr_predicted_ids)
        img_count, scores, ratio,precision, recall = score(new_arr_predicted_ids,original_ids)
        info(img_count, scores, ratio,precision, recall)
       
        # Perform further processing on the image or store it as needed
    print("Batch complete")



# note: MY ppln  Uncomment once switxh to my ppln, and comment F_ppln 
#predicted_ids = predicted_id2_c(image_dict)

#F_ppln - ✔
''''

    predicted_ids = F_predicted_id2(image_dict)


    img_count,scores,ratio = score_c(original_ids,predicted_ids)

    info(img_count,scores,ratio)
    '''
   

if args.score_with_images:
 
   #image_dict = load_images(image_directory)


   original_ids = original_id(image_dict)


   predicted_ids = predicted_id(image_dict)

 #Code Check predicted_ids - works 

   print(predicted_ids)

   img_count,scores,ratio = score(original_ids,predicted_ids)

   info(img_count,scores,ratio)
 
   while(True):
    #  ret, frame = cap.read()
     for key, values in image_dict.items():
         img1 = cv2.imread(key)
         img2 = processed(img1)
         
         combined = combined_2(img1, img2)
         combined.show()
   
    
     
     if cv2.waitKey(1) & 0xFF == ord('x'):
         combined.close()
         break
     
     
cap.release()
cv2.destroyAllWindows()



def score(original_id, predicted_id):
    scores = 0
    total = len(original_id)
    predicted_id_count = len(predicted_id)
    print()
    print('Total images:', total)
    print('Predicted images:', predicted_id_count)

    for id in original_id:
        if isinstance( id, int) and id in predicted_id:
            scores += 1

    precision, recall = calc_p_r(original_id, predicted_id)
    ratio = (scores / total) * 100
    print('Scores:', scores)
    return scores, total, ratio, precision, recall

original = [1, 1, 1,
        1, 22, 22, 22, 22, 22]


pred = [1, 1, None, 1, 22, 22, None, None, 22]

scores, total, ratio, precision, recall = score(pred, original)
print(info(scores, total, ratio, precision, recall))

# Testing the scoring fucntion  


#A1 = [1,2,3,44,22,33]
#A2 = [1,None, 3, None, None,33]




#Test worked, converted array of array to an int, flipped orig, to predict. 



#AA = A_predicted_id2(image_dict)




