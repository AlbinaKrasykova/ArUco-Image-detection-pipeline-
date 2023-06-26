
import cv2 

#import function from the main img p ppln 
#from Fahads_ppln import Fdetection, Ftransformation

 
#import The_Main_imgP_ppln from function

#GOAL-1: saves the frames specifically which the img p ppln was able to detect (if the img p pln saw the id - > save it )
#GOAL-2: First score test for the pipelines, - > return an image, and printes an id of the aruco tag

#RESULT : test-score - 1 (basic) check's the ppln by return an an image with an id 


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters 
detector = cv2.aruco.ArucoDetector(dictionary)



def detect(frame):
    corners,ids,_ = detector.detectMarkers(frame)
    detected_markers = cv2.aruco.drawDetectedMarkers(frame, corners, ids)   
    return ids,detected_markers
    ''''
    if ids is None:
        detected_flag = False
    else:
        detected_flag = True
    return detected_markers, detected_flag
    '''

def process_frame0(frame):
     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
     clache = cv2.createCLAHE(clipLimit = 40) 
     frame_clache = clache.apply(gray)  
     th3 = cv2.adaptiveThreshold(frame_clache,125,cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,51,1) 
     blurred = cv2.GaussianBlur(th3,(21,21),0) 

     detected_frame, detected_flag = detect(frame,blurred)
     #if detected_flag:
        ### save this frame
        #cv2.imwrite("save_frame1", frame)
        #print("SAVED")
     return detected_frame 

#Just a process frame is returned 
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    clache = cv2.createCLAHE(clipLimit = 40) 
    frame_clache = clache.apply(gray)  
    th3 = cv2.adaptiveThreshold(frame_clache,125,cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,51,1) 
    blurred = cv2.GaussianBlur(th3,(21,21),0) 
    flipped = cv2.flip(blurred, 1)
    return flipped


# For a first check test of pipeline (score) - ppln_check
# I need a frame processing fucntion from an image processing pipeline(A,F,GridSearcg genereted ones), and a raw image of the tG 
# return id and image 
def ppln_check(img, process_frame,detect):
 p_image = process_frame(img)
 ids, img = detect(p_image)
 return ids,img



    


cap = cv2.VideoCapture(0)

#from Fahads_ppln import Fdetection, Ftransformation

while(True):
    
    #Basic Check for the ppln, and returns the image of the id   
    img = cv2.imread('id_3_passed.jpg')
    id, img_check = ppln_check(img, process_frame, detect)
    cv2.imshow('img_check',img_check)

    if cv2.waitKey(1000) & 0xFF == ord('x'):
        cv2.imwrite('img_check.jpg', img_check)
        print('Detect id is:', id)
        break
     
     
cap.release()
cv2.destroyAllWindows()






