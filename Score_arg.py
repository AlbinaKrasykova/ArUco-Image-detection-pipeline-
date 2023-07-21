# importing librraies 

import argparse
import cv2 
import numpy as np
from shadow_highlight_correction import correction
import math
import os
import re
from PIL import Image


#GOAL 1: score function which scores ing pppln according to how well it perfomed  ✔

#Result: does run, but 't detect images BUT needs a confusion matrix, grid search and laod images in batches (as for 2K dataset) ? FIXED

#2: takes both 2dprinted tags(test) and 3d printed tags and detects the id's correctl, scores iT ✔

#3 : check the datatype whch is return by the function, rewrite all the fucntion manually - Done - ✔ 
# (Original id return array of ints, predict ints array return arr of int in array - [[1]])

#4: check/score 2 pplns - Fahds (4.1) - ✔ & Amrit (4.2) - ✔

#5 : Implement My, F, A pipelines as an option in the command line - ✔

#6: Build a better datset, test pplns on it 2500 ✔

#7: Dataset from different angles/calculating the distance - ✔

#  image processing in batches, implement and rewrite functions - ✔

# NEEDS TO BE DONE 7: Implement (precision and recall) - ✔

#8 Predicts 2 tags, return 2 tags  

#9 function that takes all the pplns 

#10 saves the result to the file 

#11 TASKS for July 11 rewite function for 2ID just for a 1 set 
#Original Id and scoring functim 

#12 #Function: inserting  any pipelines and getting the result 

##CLean the code rn - repeating values, implenet 2 id's to every ppln 
# Score the string and add the classification to the scoring function
# label each predicted id as option(FP,TP,TN,FN), compere to exel while manually created 

#I was trying to import functions but I got work more of functions I importing like these which imports images 
#from Amrits_ppln import correction , contrast, threshold, calibration_frame, clahe, blur, THRESHOLD_PX_VAL 



directory  = r'D:\AI research internship\opencv_scripts\2_id'
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters 
detector = cv2.aruco.ArucoDetector(dictionary)


#0. Function that Generates dictionary with images from the file - ✔
# Key: 9_angle_6_.png, Value: <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1920x1080 at 0x178C0FF6BF0>



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

#2 CREATE an ARRAY with Tag original ID's by parcing the key string, as getting first digits of it and save to an array 'Original ID' - Key: 9_angle_6_.png - > 9

#check type - ✔ returns array of ints

#Info: Original id's of the images which is an array of ints  

def original_id(image_dict):
    digit_array = []
    pattern = r'^\d{1,2}'
    
    for key in image_dict.keys():
        match = re.match(pattern, key)
        if match:
            first_digits = int(match.group())
            digit_array.append(first_digits)
    print(digit_array)        
    return digit_array


#3 CREATE an ARRAY with Tags that were predicted - ✔

def My_ppln(image_dict):
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

    print('My array with predicted id is: ', p_id_arr)
    return p_id_arr

# F_ppln CREATE an ARRAY with Tags that were predicted - ✔

def F_ppln(image_dict):
    p_id_arr = []
    for ids, images in image_dict.items():
        transformation = cv2.cvtColor((images), cv2.COLOR_BGR2GRAY)
        transformation = cv2.bitwise_not(transformation)
        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16, 16))
        transformation = clahe.apply(transformation)
        transformation = cv2.GaussianBlur(transformation, (21, 21), 0)
        _, transformation = cv2.threshold(transformation, 150, 255, cv2.THRESH_BINARY)
        # flipped = cv2.flip(transformation, 1)
        _, ids, _ = detector.detectMarkers(transformation)
        p_id_arr.append(ids)
        
    
    print('Fahads array with the predicted id is: ', p_id_arr)
    return p_id_arr

def pplns(ppln):
 batch_size = 150
 for batch in load_images_in_batches(directory, batch_size):
        original_ids = original_id(batch)
        print(original_ids)
        predicted_ids = ppln(batch)
       
        new_arr_predicted_ids = [int(x[0, 0]) if x is not None else None for x in predicted_ids]
        print(new_arr_predicted_ids)
        img_count, scores, ratio,precision, recall = score(new_arr_predicted_ids,original_ids)
        info(img_count, scores, ratio,precision, recall)

# A_ppln CREATE an ARRAY with Tags that were predicted - > reimplemnt function with array of sets 

calibration_frame = None

THRESHOLD_PX_VAL = 100

CLIP_LIMIT = 20.0 

def contrast(image, clahe):
    clahe_out = clahe.apply(image)
    return clahe_out

def blur(image, kernel):
    return cv2.blur(image, kernel)

def threshold(image, px_val):

    thresholded = cv2.adaptiveThreshold(image, 
                                        255, 
                                        cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 
                                        89, 
                                        2)
    return thresholded


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




def A_ppln(image_dict):
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
    return p_id_arr

#4

''''
def pplns_output(ppln):

 for batch in load_images_in_batches(directory, batch_size):
        original_ids = original_id(batch)
        print(original_ids)
        predicted_ids = ppln(batch)
       
        new_arr_predicted_ids = [int(x[0, 0]) if x is not None else None for x in predicted_ids]
        print(new_arr_predicted_ids)
        img_count, scores, ratio,precision, recall = score(new_arr_predicted_ids,original_ids)
        info(img_count, scores, ratio,precision, recall)
'''

#5 Function that combines and displays 2 images side by side  - ✔


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

#6 Calculates the precison and recall - ✔ 
#if original is empty set(), and predicted is empty TN++  EX:  (set() - > set()) - TN++
#id the original is empty set(, and predicted is something else then empty FP++ EX: (set() -> {17}) - FP++ 
#ALso if the set is one value, and th epredicted is anothe value, FP++ EX: ({23,40} -> {17}) - FP++ 


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

    precision = 0
    recall = 0

    if true_positive + false_positive != 0:
        precision = true_positive / (true_positive + false_positive)
    
    if true_positive + false_negative != 0:
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
        if isinstance(id, int) and id in predicted_id:
            scores += 1

    precision, recall = calc_p_r(original_id, predicted_id)
    ratio = (scores / total) * 100
    print('Scores:', scores)
    return scores, total, ratio, precision, recall

#7 Display ratio based on the precious scoring subfunction  - ✔

#TO ADD: toal FP/TN/FN/TP
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



'''
def processed(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clache = cv2.createCLAHE(clipLimit=40)
    frame_clache = clache.apply(gray)
    th3 = cv2.adaptiveThreshold(frame_clache, 125, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 51, 1)
    blurred = cv2.GaussianBlur(th3, (21, 21), 0)
    flipped = cv2.flip(blurred, 1)
    return flipped


'''

cap = cv2.VideoCapture(0)




parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
parser.add_argument('--my_score', action='store_true')
parser.add_argument('--Fahad_score', action='store_true')
parser.add_argument('--Amrit_score', action='store_true')
parser.add_argument('--score_2id', action='store_true')
parser.add_argument('--score_all_cases', action='store_true')

args = parser.parse_args()

output_filename = 'output.doc'  
# Save the output image to a file


if args.my_score:

    batch_size = 150
    directory = r'D:\AI research internship\opencv_scripts\n_l_r_angl'


    def My_ppln(image_dict):
        p_id_arr = []
        for ids, img in image_dict.items():
            
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            clache = cv2.createCLAHE(clipLimit=40)
            frame_clache = clache.apply(gray)           
            th3 = cv2.adaptiveThreshold(frame_clache, 125, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV, 51, 1)
            blurred = cv2.GaussianBlur(th3, (21, 21), 0)
            #flipped = cv2.flip(blurred, 1)
            _, ids, _ = detector.detectMarkers(blurred)
            print('id dtype is ', type(ids))
            p_id_arr.append(ids)

        print('My array with predicted id is: ', p_id_arr)
        return p_id_arr
    
    
    for batch in load_images_in_batches(directory, batch_size):
        original_ids = original_id(batch)
        print(original_ids)
        predicted_ids = My_ppln(batch)
       
        new_arr_predicted_ids = [int(x[0, 0]) if x is not None else None for x in predicted_ids]
        print(new_arr_predicted_ids)
        img_count, scores, ratio,precision, recall = score(new_arr_predicted_ids,original_ids)
        info(img_count, scores, ratio,precision, recall)


    print('My ppln score: ')   
    print('')
    
    print("Batch complete")


#F_ppln with args - ✔


if args.Fahad_score:
   
  
    batch_size = 150
  
  
    directory = r'D:\AI research internship\opencv_scripts\n_l_r_angl'
    print('Fahd ppln score: ')   
    print('')
    
    pplns(F_ppln)
     

if args.Amrit_score:
    
    batch_size = 150
    directory = r'D:\AI research internship\opencv_scripts\n_l_r_angl'

    print('Amrit ppln score: ')  
    print('')
 
    
    pplns(A_ppln)

# Dataset 2ID's  

if args.score_2id:
    
    
    print('score for 2 ids')
    directory  = r'D:\AI research internship\opencv_scripts\2_id'

    

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
        return image_dict


    img_dict = load_images(directory) 

  
    

    def original_id_2(image_dict):
        
        arr = []
        
        pattern = r'^(\d{1,2})_(\d{1,2})'

        for key in image_dict.keys():
            digit_set = set()
            match = re.match(pattern, key)
            if match:
                first_digits = int(match.group(1))
                second_digits = int(match.group(2))
                digit_set.add(first_digits)
                digit_set.add(second_digits)
                arr.append(digit_set)
            else:
                print(f"Key '{key}' does not match the pattern.")
        

        print(arr)
        return arr
    


    
    



    def A_ppln_2(image_dict):
        arr = []
        p_id_set = set()
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
            
            p_id_set = set()
            
             
            if ids is not None:
                for inner_arr in ids:
                    for i in inner_arr:
                        p_id_set.add(i)
                    
            arr.append(p_id_set)
        print('Array for 2 ids is ', arr)    
        return arr
    
    #A_ppln that handles none values 


     #Add count to each of classification param
     #print oroginal value and predict value and lable as any od 4 values 

    def calc_p_r(original_ids, predicted_ids):
        #I did - > it predicted 
        true_positive = 0
        #I did -> it didn't predicted 
        false_negative = 0
        #I didn't do -> it predicted 
        false_positive = 0
        #I didn't do - >  it didn't predict 
        true_negative = 0

        #Q:  do i have to create set with empty values as for possible false positive/true negative cases 

        for i in range(len(original_ids)):
            if original_ids[i] == predicted_ids[i]:
                true_positive += 1
            else:
                false_negative += 1

        false_negative = len(original_ids) - true_positive

        precision = 0
        recall = 0

        if true_positive + false_positive != 0:
            precision = true_positive / (true_positive + false_positive)

        if true_positive + false_negative != 0:
            recall = true_positive / (true_positive + false_negative)

        return precision, recall
    

    
    def score_3_debug(s1, s2):
            score = 0
            for set1, set2 in zip(s1, s2):
                if not set2:
                    continue
                print('set1:', set1, 'set2:', set2)

                for id1 in set1:
                    if id1 in set2:
                        
                        score+=1

            return score
#the case for #1 data_set (1ids), #2 data_set (noid, multiple id's) 
if args.score_all_cases:
# Working function that prints clearly putput of scoring function for the 2 datasets -----------------------------------------------------------------------------------------------
#(implement in argline as info score for 2 datasets  -> 41_7)
#issue i am running into none is not itterable - solved - create an empty set instead of none 
#  out of 91 images saves just 2 because sets saves n stores






    directory = r'D:\AI research internship\opencv_scripts\data_set'


    #cleans a string, count ids, ignores the last digit(angle)



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

        return image_dict




    def clean_string(string):
        digits = re.findall(r'\d+', string)
        if len(digits) > 1:
            digits = digits[:-1]  
        cleaned_string = '_'.join(digits)
        count = len(digits)
        
        return count, cleaned_string



    def original_id_2(image_dict):
        arr = []

        for key in image_dict.keys():
            count, key = clean_string(key)
            #print('Count of ids is', count, 'the key is', key)
            pattern = r'^(\d{1,2})'  # Create the pattern based on the ID count

            for _ in range(1, count):
                pattern += r'_(\d{1,2})'

            digit_set = set()
            match = re.match(pattern, key)
            if match:
                for group in match.groups():
                    digit_set.add(int(group))
                arr.append(digit_set)
            else:
                print(f"Key '{key}' does not match the pattern.")

        return arr






    

    def A_ppln_2(image_dict):
            arr = []
            p_id_set = set()
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
                
                p_id_set = set()
                
                if ids is not None:
                    for inner_arr in ids:
                        for i in inner_arr:
                            p_id_set.add(i)
                        
                arr.append(p_id_set) 
            return arr
        

    #small dataset 41 img - 1id 
    directory = r'D:\AI research internship\opencv_scripts\data_set'
    img_dict = load_images(directory) 
    #small dataset 91 img  - 2 id
    #directory2 = r'D:\AI research internship\opencv_scripts\2_id'





 
    def calc_p_r(original_ids, predicted_ids):
            #I did - > it predicted 
            true_positive = 0
            #I did -> it didn't predicted  FN
            false_negative = 0
            #I didn't do -> it predicted 
            false_positive = 0
            #I didn't do - >  it didn't predict 
            true_negative = 0

            #Q:  do i have to create set with empty values as for possible false positive/true negative cases 

            for i in range(len(original_ids)):
                if original_ids[i] == predicted_ids[i]:
                    true_positive += 1
                else:
                    false_negative += 1

            false_negative = len(original_ids) - true_positive

            precision = 0
            recall = 0

            if true_positive + false_positive != 0:
                precision = true_positive / (true_positive + false_positive)

            if true_positive + false_negative != 0:
                recall = true_positive / (true_positive + false_negative)

            return true_positive, true_negative, false_negative, precision, recall


    def calc_p_r_all(original_ids, predicted_ids):
            #I did - > it predicted 
            true_positive = 0
            #I did -> it didn't predicted  FN
            false_negative = 0
            #I didn't do -> it predicted 
            false_positive = 0
            #I didn't do - >  it didn't predict 
            true_negative = 0

            #Q:  do i have to create set with empty values as for possible false positive/true negative cases 

            for i in range(len(original_ids)):
                # 1 - handle empty set's cases
                if original_id[i] is set():
                    # set() -> set() - true_negative
                    if  original_id[i] == original_id[i]:
                     true_negative+=1
                    # set() -> {20} - false_positive
                    if original_id[i] != original_id[i]:
                        false_positive+=1
                # 1 - handle sets with the values 
                if original_ids[i] == predicted_ids[i]:
                    # set{10}->{10}
                    true_positive += 1
                else:
                    false_negative += 1

            false_negative = len(original_ids) - true_positive

            precision = 0
            recall = 0

            if true_positive + false_positive != 0:
                precision = true_positive / (true_positive + false_positive)

            if true_positive + false_negative != 0:
                recall = true_positive / (true_positive + false_negative)

            return true_positive, true_negative, false_negative, precision, recall






    def score3(s1, s2):
                score = 0
                for set1, set2 in zip(s1, s2):
                    if not set2:
                        continue
                    

                    for id1 in set1:
                        if id1 in set2:
                            
                            score+=1

                return score

                   
#Dataset

    orig_set = original_id_2(img_dict)

    predict_set = A_ppln_2(img_dict)

    score = score3(orig_set, predict_set)


    def info(original, predicted):

     
        total = len(original)
        score = score3(original, predicted)
        ratio = score/total*100
        true_positive, true_negative, false_negative, precision, recall =  calc_p_r(original, predicted)
        print()
        print('---------------------------------------------------')
        print(f"Image processing pipeline scored at {int(ratio)} %")
        print(f"Out of {total} images {score} were predicted")
        print('Score:', ratio, '%')
        print('precision:', precision, '%')
        print('recall:', recall, '%')
        print('true_positive total:', true_positive)
        print('false_negative total:', false_negative)
        print('true_negative total:', true_negative)
        print('---------------------------------------------------')
        print()


    a_orig_set = [set(),{40,30},{5},{40,30},{6},{20,30},set()]

    a_predicted_set = [set(),{17},{5},{40,30},set(),{20},{5}]

    directory = r'D:\AI research internship\opencv_scripts\data_set'



    print()
    print('---------------------------------------------------')
    print('Info for the 7 img  dataset (multiple cases: no ids, 1 id, 2+ ids)')
    print('---------------------------------------------------')
    info(a_orig_set, a_predicted_set)


    print('---------------------------------------------------')
    print('Info for the 41 img dataset (1 case: 1 id)')
    print('---------------------------------------------------')
    print()
    info(orig_set, predict_set)

#TEST  -----------------------------------------------------------------------------------------------



#Test function calc precision and score 





original_ids = [ {30}, {30,40,50},set(), set(),{30,40}]

predicted_ids = [{30} , {30}, set(),{30}, {17}]

true_positive = 0
true_negative = 0
false_negative = 0
false_positive = 0
score = 0



# true_positive - I did - > it predicted 
#false_negative - I did -> it didn't predicted  FN
# false_positive  - I didn't do -> it predicted 
# true_negative  - #I didn't do - >  it didn't predict 
        

#empty case handling 

#Handle multiple cases 

for i in range(len(original_ids)):
        
        # 1 handle all the empty case 
        #score count 1/len(original_ids[i])
        if original_ids[i] == set():
            print('Handling the empty case', original_ids[i],' ',  predicted_ids[i] )

            # - fales positive case 
            if original_ids[i] != predicted_ids[i]:
                print('Handeling false positive cases', original_ids[i],' ',  predicted_ids[i] )
                false_positive+=1

            if original_ids[i] == predicted_ids[i]:
                print('Handeling true negative cases', original_ids[i],' ',  predicted_ids[i] )
                true_negative+=1

        #2 handle vales cases

        if original_ids[i] != set():
          if original_ids[i] == predicted_ids[i]:
            print('Handeling true postive cases', original_ids[i],' ',  predicted_ids[i])
            true_positive+=1
            print('1/len(original_ids[i]): ', len(original_ids[i]))
            score = 1/len(original_ids[i])

          if predicted_ids[i]==set():
              print('Handeling false negative cases', original_ids[i],' ',  predicted_ids[i] )
              false_negative+=1

print('score', score)
print('true_negative: ', true_negative)
print('false_positive: ', false_positive)
print('false_negative' , false_positive)
print('true positive' , true_positive)
           


            


''''
        #Handle all the case with the values 
        print('Handling cases with values in it')
        # 1- true positive case iwth values
        if original_ids[i]!= set() and original_ids[i] == predicted_ids[i]:
            print('Handling true positive')
            score = 1/len(original_ids[i])
            true_positive+=1
        #2 - false positive case with values
        if original_ids[i] !=predicted_ids[i]:
            print('Handling false positive')
            false_positive+=1
          
         #3 - false_negative - I did -> it didn't predicted  FN
        if predicted_ids[i] == set():
            print('Handling false negative')
            false_positive+=1
            


































































3 #CODE for the reuse 

# Function Requires n Output : path to the folder, folder with images-files 

""
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
''''
    for batch in load_images_in_batches(directory, batch_size):
        original_ids = original_id(batch)
        print(original_ids)
        predicted_ids = A_ppln(batch)
       
        new_arr_predicted_ids = [int(x[0, 0]) if x is not None else None for x in predicted_ids]
        print(new_arr_predicted_ids)
        img_count, scores, ratio,precision, recall = score(new_arr_predicted_ids,original_ids)
        info(img_count, scores, ratio,precision, recall)
        save_to_file(img_count, scores, ratio,precision, recall)
       
        
    print("Batch complete")
'''
#TEST
''''
def info_ppln_small(ppln):
 directory_s= r'D:\AI research internship\opencv_scripts\data_set'

 for batch in load_images_in_batches(directory_s, batch_size):
        original_ids = original_id(batch)
        print(original_ids)
        predicted_ids = ppln(batch)
       
        new_arr_predicted_ids = [int(x[0, 0]) if x is not None else None for x in predicted_ids]
        print(new_arr_predicted_ids)
        img_count, scores, ratio,precision, recall = score(new_arr_predicted_ids,original_ids)
        return img_count, scores, ratio,precision, recall
 
def info_ppln_big(ppln):
 directory_b= r'D:\AI research internship\opencv_scripts\n_l_r_angl'
#directory_2= r'D:\AI research internship\opencv_scripts\data_set

 for batch in load_images_in_batches(directory_b, batch_size):
        original_ids = original_id(batch)   
        print(original_ids)
        predicted_ids = ppln(batch)
       
        new_arr_predicted_ids = [int(x[0, 0]) if x is not None else None for x in predicted_ids]
        # print(new_arr_predicted_ids)
        img_count, scores, ratio,precision, recall = score(new_arr_predicted_ids,original_ids)
        return img_count, scores, ratio,precision, recall


A_img_count_s, A_scores_s, A_ratio_s,A_precision_s, A_recall_s= info_ppln_small(A_ppln)
My_img_count_s, My_scores_s, My_ratio_s, My_precision_s, My_recall_s = info_ppln_small(My_ppln)
F_img_count_s, F_scores_s, F_ratio_s,F_precision_s, F_recall_s = info_ppln_small(F_ppln)

A_img_count_b, A_scores_b, A_ratio_b,A_precision_b, A_recall_b= info_ppln_big(A_ppln)
My_img_count_b, My_scores_b, My_ratio_b, My_precision_b, My_recall_b = info_ppln_big(My_ppln)
F_img_count_b, F_scores_b, F_ratio_b,F_precision_b, F_recall_b = info_ppln_big(F_ppln)


 
def save_to_file():


 output_filename = "all_ppl_scoreo.txt"  # Specify the filename and extension for the output text file

    # Open the file in write mode ('w') and write the information to it
 with open(output_filename, 'w') as file:
        file.write(f"Image processing  for Amrit's ppln & small dataset scored at {int(A_ratio_s)} %\n")
        file.write(f"Out of {A_img_count_s} images, {A_scores_s} were predicted\n")
        file.write(f"Score: {A_ratio_s}%\n")
        file.write(f"Precision: {A_precision_s}%\n")
        file.write(f"Recall: {A_recall_s}%\n")

        file.write(f"Image processing pipeline for my ppln & small dataset  scored at {int(My_ratio_s)} %\n")
        file.write(f"Out of {My_img_count_s} images, {My_scores_s} were predicted\n")
        file.write(f"Score: {My_scores_s}%\n")
        file.write(f"Precision: {My_precision_s}%\n")
        file.write(f"Recall: {My_recall_s}%\n")

        file.write(f"Image processing pipeline for Fahd's ppln &  small dataset scored at {int(F_ratio_s)} %\n")
        file.write(f"Out of {F_img_count_s} images, {F_scores_s} were predicted\n")
        file.write(f"Score: {F_ratio_s}%\n")
        file.write(f"Precision: {F_precision_s}%\n")
        file.write(f"Recall: {F_recall_s}%\n")

#big_data
        file.write(f"Image processing pipeline for Amrit's ppln & big dataset scored at {int(A_ratio_b)} %\n")
        file.write(f"Out of {A_img_count_b} images, {A_scores_b} were predicted\n")
        file.write(f"Score: {A_ratio_b}%\n")
        file.write(f"Precision: {A_precision_b}%\n")
        file.write(f"Recall: {A_recall_b}%\n")

        file.write(f"Image processing pipeline for My ppln & big dataset scored at {int(My_ratio_b)} %\n")
        file.write(f"Out of {My_img_count_b} images, {My_scores_b} were predicted\n")
        file.write(f"Score: {My_scores_b}%\n")
        file.write(f"Precision: {My_precision_b}%\n")
        file.write(f"Recall: {My_recall_b}%\n")

        file.write(f"Image processing pipeline for Fahad's ppln & big dataset scored scored at {int(F_ratio_b)} %\n")
        file.write(f"Out of {F_img_count_b} images, {F_scores_s} were predicted\n")
        file.write(f"Score: {F_ratio_b}%\n")
        file.write(f"Precision: {F_precision_b}%\n")
        file.write(f"Recall: {F_recall_b}%\n")

print("Information saved to", output_filename)



save_to_file()

'''
''''
if args.score_with_images:


   original_ids = original_id(image_dict)


   predicted_ids = predicted_id(image_dict)

   print(predicted_ids)

   img_count,scores,ratio = score(original_ids,predicted_ids)

   info(img_count,scores,ratio)
 
   while(True):
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

'''