import cv2


#Detects    PRINTED TAGS 
# First step setting up the variables for the functions 
# get dictionary, parameters, detector 
#detector = cv2.pythonaruco.ArucoDetector(dictionary, detectorParameters=parameters)
# parameters.refinementStrategy = cv2.cornerHarris
 

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

parameters = cv2.aruco.DetectorParameters 
detector = cv2.aruco.ArucoDetector(dictionary)








# Step 2 I create a function which I call detect, with my input parameters raw, frame 
# the first lime inside of the function I make an instanceof detector variable-object. 
# then I use function on it detectMarkers with the input frame which gives ne parameters that is needed to the next fuxtion
#in order to actually detect the the makers 

# the last line returning detected markers 


# image detection pipiline 


def detect(raw, frame):
      corners, ids, _ = detector.detectMarkers(frame)
      detected_markers = cv2.aruco.drawDetectedMarkers(raw, corners, ids)
      return detected_markers 

# processing pipline 


#first line creating the process frame function with the frame fucntion with the input - frame 
# the next, first line of the function start with processing the fram into a grey color 
# processed_frame = frame - ?
# call the detect function for marker detection 

def process_frame(frame):
      processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
      
      detected = detect(frame, processed_frame)
      return detected 




import cv2
# reading the camera
cap  = cv2.VideoCapture(0)  




while(True):
  ret, frame = cap.read()


  processed_frame = process_frame(frame)

  cv2.imshow('frame',  processed_frame)

  if cv2.waitKey(1) & 0xFF == ord('x'):
       break

cap.release()
cv2.destroyAllWindows()






#dictionary = cv2.aruco.getPredifinedDictionary(c2.aruc)


print(cv2.__version__)  


