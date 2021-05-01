import numpy as np
import argparse
import sys
import cv2
from math import pow, sqrt
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import time
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from os import listdir
from os.path import isfile, join
from numpy.core.records import array
from platform import python_version
from PIL import Image

print("Python version: ", python_version())
print("OpenCV version: ", cv2.__version__)
print("Numpy version: ", np.version.version)
print("Tensorflow version: ", tf.__version__)
print("Pickle version: ", pickle.format_version)
print(sys.version)

# Parse the arguments from command line
parser = argparse.ArgumentParser()

parser.add_argument('-v', '--video', type = str, default = '', help = 'Video file path. If no path is given, video is captured using device.')

parser.add_argument('-m', '--model', default = 'SSD_MobileNet.caffemodel', help = "Path to the pretrained model.")
    
parser.add_argument('-p', '--prototxt', default = 'SSD_MobileNet_prototxt.txt', help = 'Prototxt of the model.')

parser.add_argument('-l', '--labels', default = 'class_labels.txt', help = 'Labels of the dataset.')

parser.add_argument('-y', '--cfg', default = 'yolov3.cfg', help = 'Path_to_yolo_caffemodel')

parser.add_argument('-w', '--weights', default = 'yolov3.weights', help = 'Prototxt file for yolo')

parser.add_argument('-x', '--excel', default = 'label_names.csv', help = 'CSV file for Traffic_Sign_Detection')

parser.add_argument('-c', '--confidence', type = float, default = 0.9, help='Set confidence for detecting objects')

args = parser.parse_args(args=[])


# Loading mean image to use for preprocessing further; Opening file for reading in binary mode
with open('mean_image_rgb.pickle', 'rb') as f:
    mean = pickle.load(f, encoding='latin1')  # dictionary type

labels = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow","diningtable",
            "dog","horse", "motorbike","person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(labels), 3))
# Read the csv file for traffic-sign and print first five records
tf_labels = pd.read_csv(args.excel)
print(tf_labels.head())

print("Streaming video using device...\n")

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_features/haarcascade_frontalface_default.xml')
profile_classifier = cv2.CascadeClassifier('haarcascade_features/haarcascade_profileface.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_features/haarcascade_eye.xml')
print("Loading HAAR classifiers...\n")


# Function to detect face
def face_detector(img, size=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale( gray, 1.3, 5, minSize = (30,30))
    # If face not found return blank region
    if faces == ():
        return [img, [], None]
    # Obtain Region of face
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))        
        profile = profile_classifier.detectMultiScale(img, 1.3,5)
        for (px,py,pw,ph) in profile:
            cv2.rectangle(img,(px,py),(px+pw,py+ph), (0,255,255),2)         
        eyes = eye_classifier.detectMultiScale(img, 1.3,4)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh), (0,255,255),2) 
    return [img, roi, faces[0]]   

# Capture video from file or through webcam
if args.video:
    cap = cv2.VideoCapture(args.video)    
else:
    cap = cv2.VideoCapture(0)    
#initialize the FPS counter
fps = FPS().start()
#Load the Caffe model 
print("Loading model...\n")
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)
d_net = cv2.dnn.readNetFromDarknet(args.cfg, args.weights)

# To use with GPU
d_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
d_net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
# Getting names of all YOLO v3 layers
layers_all = d_net.getLayerNames()
# Getting only detection YOLO v3 layers that are 82, 94 and 106
layers_names_output = [layers_all[i[0] - 1] for i in d_net.getUnconnectedOutLayers()]

# Facial Recognition model training 
models = {"Komal": {"data_path": "face/komal/","files": [],"model": None},
          "Ananya": {"data_path": "face/ananya/","files": [],"model": None},
          "Arunima": {"data_path": "face/arunima/","files": [],"model": None},
          "Ibrahim": {"data_path": "face/ibrahim/","files": [],"model": None}
         }
for key in models:
    print("Started training model for " + key)
    models[key]["files"] = [f for f in listdir(models[key]["data_path"]) if isfile(join(models[key]["data_path"], f))]
    Training_Data, Labels = [], []

    for i, files in enumerate(models[key]["files"]):
        image_path = models[key]["data_path"] + models[key]["files"][i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append( np.asarray( images, dtype=np.uint8))
        Labels.append(i)

    # Create a numpy array for both training data and labels
    Labels = np.asarray(Labels, dtype=np.int32)

    # Initialize facial recognizer
    models[key]["model"] =  cv2.face.LBPHFaceRecognizer_create()
    # NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
    # Let's train our model
    models[key]["model"].train(np.asarray(Training_Data), np.asarray(Labels))
    print("Model trained successfully for " + key)

while True:  
    ret, frame = cap.read()
    ar = face_detector(frame)
    face=ar[1] 
    pos=ar[2]
    time.sleep(0.06)
    if not ret:
        break   

    # grab the frame from the threaded video stream and resize it to have a maximum width of 600 pixels    
    frame = imutils.resize(frame, width=600)
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
    # Blob from current frame of traffic sign video
    tf_blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    d_net.setInput(tf_blob)
    detections = net.forward()
    tf_detections = d_net.forward(layers_names_output)
       
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > args.confidence:
            # extract the index of the class label from the`detections`, then compute the (x, y)coordinates of the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")            
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(labels[idx],confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 1)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startY, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 1)  
            print(label)
    
    pos_dict = dict()
    coordinates = dict()
    # Focal length (in cm)
    F = 50    
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")  
    coordinates[i] = (startX, startY, endX, endY)
    # Mid point of bounding box
    x_mid = round((startX+endX)/2,4)
    y_mid = round((startY+endY)/2,4)
    height = round(endY-startY,4)

    # Distance from camera based on triangle similarity
    distance = round(((165 * F)/height)/30.48,2)
    print("Distance:{dist}".format(dist = distance), "feet")
    
    # Mid-point of bounding boxes (in cm) based on triangle similarity technique
    x_mid_cm = (x_mid * distance) / F
    y_mid_cm = (y_mid * distance) / F
    pos_dict[i] = (x_mid_cm,y_mid_cm,distance)
    
    # Distance between every object detected in a frame
    close_objects = set()
    for i in pos_dict.keys():
        for j in pos_dict.keys():
            if i < j:
                dist = sqrt(pow(pos_dict[i][0]-pos_dict[j][0],2) + pow(pos_dict[i][1]-pos_dict[j][1],2) + pow(pos_dict[i][2]-pos_dict[j][2],2))

                # Check if distance less than 1 feet (300 mm approx):
                if dist < 30:
                    close_objects.add(i)
                    close_objects.add(j)
    for i in pos_dict.keys():
        if i in close_objects:
            COLOR = (0,0,255)
        else:
            COLOR = (0,255,0)     
        (startX, startY, endX, endY) = coordinates[i]
        cv2.rectangle(frame,(startX,startY), (endX, endY), COLOR, 1)
        y = startY - 15 if startY - 15 > 15 else startY + 15        
        # Convert mms to feet
        cv2.putText(frame, "Distance: {i} ft".format(i=round(pos_dict[i][2]/30.48,4)), (y, startY),cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR, 1)
        cv2.namedWindow('Frame',cv2.WINDOW_NORMAL) 

    # Lists for detected bounding boxes, confidences and class's number
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Going through all output layers after feed forward pass
    for traffic_result in tf_detections:
        # Going through all detections from current output layer
        for detected_objects in traffic_result:
            # Getting 80 classes' probabilities for current detected object
            scores = detected_objects[8:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]
            # Minimum probability to eliminate weak detections
            probability_minimum = 0.9
            # Setting threshold to filtering weak bounding boxes by non-maximum suppression
            threshold = 0.8
            
            # Eliminating weak predictions by minimum probability
            if confidence_current > probability_minimum:
                # Scaling bounding box coordinates to the initial frame size
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Getting top left corner coordinates
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)                

    # Implementing non-maximum suppression of given bounding boxes
    tf_results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

    # Checking if there is any detected object been left
    if len(tf_results) > 0:
        # Going through indexes of results
        for i in tf_results.flatten():
            # Bounding box coordinates, its width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]             
            # Cut fragment with Traffic Sign
            c_ts = frame[y_min:y_min+int(box_height), x_min:x_min+int(box_width), :]            
            if c_ts.shape[:1] == (0,) or c_ts.shape[1:2] == (0,):
                pass
            else:
                # Getting preprocessed blob with Traffic Sign of needed shape
                blob_ts = cv2.dnn.blobFromImage(c_ts, 1 / 255.0, size=(32, 32), swapRB=True, crop=False)
                blob_ts[0] = blob_ts[0, :, :, :] - mean['mean_image_rgb']
                blob_ts = blob_ts.transpose(0, 2, 3, 1)
                               
                prediction = np.argmax(scores)
                
                # Drawing bounding box on the original current frame
                cv2.rectangle(frame, (x_min, y_min),(x_min + box_width, y_min + box_height),(0,0,255), 1)

                # Preparing text with label and confidence for current bounding box
                box = '{}: {:.4f}'.format(tf_labels['SignName'][prediction],confidences[i]*100)

                # Putting text with label and confidence on the original image
                cv2.putText(frame, box, (x_min, y_min - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,200), 1)
            print(box)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        foundFace = False
        user = None
        confidence = 82
        for key in models:
            if foundFace == True:
                break
            results = models[key]["model"].predict(face)
            if results[1] < 500:
                confidence = int( 100 * (1 - (results[1])/500) )
                if confidence > 82:
                    user = key
                    foundFace = True        
        posX = pos[0] + 5
        posY = pos[0] - 5
        cv2.putText(frame, "Face Detected " + str(confidence) + "%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        if foundFace == True:
            cv2.putText(frame, user, (posX, posY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)   
            print(user)
            print("Face Detected " + str(confidence) + "%")
        else:
            cv2.putText(frame, "Unknown ", (posX, posY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,153,255), 2)

        cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
    # Raise exception in case, no image is found
    except Exception as e:
        cv2.putText(frame, "Accuracy 0% (No face detected)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,155,255), 1)
        cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
        pass
    
    # Show the output frame
    cv2.imshow('Frame', frame)
    cv2.resizeWindow('Frame',800,600)

    key = cv2.waitKey(1) & 0xFF
    break
    if key == ord("q"):
        break
    #update the FPS counter
    fps.update()  
#stop the timer and display FPS count 
fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approximate FPS: {:.2f}".format(fps.fps()))

import glob
import pytesseract
for img in glob.glob("*.jpg"):
    cv_img = cv2.imread(img)
demo = Image.fromarray(cv_img)
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
text_reader = pytesseract.image_to_string(demo, lang = 'eng')
print(text_reader)    
# Clean
cap.release()
cv2.destroyAllWindows()