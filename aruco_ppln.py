
import cv2 

#DETECTs printed tags 

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters 
detector = cv2.aruco.ArucoDetector(dictionary)



def detect(raw,frame):
    corners,ids,_ = detector.detectMarkers(frame)
    detected_markers = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    return detected_markers 

def process_frame(frame):
   
     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
     clache = cv2.createCLAHE(clipLimit = 40) #5 - Limit is 40
     frame_clache = clache.apply(gray)
    
     
     #adaptiveThreshold
    
     th3 = cv2.adaptiveThreshold(frame_clache,125,cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,51,1)  # start with 11, 2
    # 51,1 detected the best 

     blurred = cv2.GaussianBlur(th3,(21,21),0) #7,7

     

     

     # define the contrast and brightness value
     #contrast = 1. # Contrast control ( 0 to 127)
     #brightness = 1. # Brightness control (0-100)

     #out = cv2.addWeighted(th3, contrast, th3, 0, brightness)

    
    # 0  start values 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV , 21, 10
    # 1 blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 10 - F
    # 1 blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 36  
    # 1 blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 106

     flipped = cv2.flip(blurred, 1)
     detected = detect(frame,blurred)
     return detected 






cap = cv2.VideoCapture(0)




while(True):
     ret, frame = cap.read()

     processed_frame = process_frame(frame)
     cv2.imshow('frame', processed_frame)
    
     
     if cv2.waitKey(1) & 0xFF == ord('x'):
         break
     
     
cap.release()
cv2.destroyAllWindows()