# import the necessary packages
from typing import List
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5)  # Set confidence value
ap.add_argument("-t", "--threshold", type=float, default=0.3)  # Set threshold value
args = vars(ap.parse_args())

kernel = np.ones((5, 5), np.uint8)  # Kernel setting for filters

# Load the COCO class labels for YOLO
labelsPath = "labels/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
# Get colours for the bounding boxes, one colour per label
np.random.seed(50)
COLOURS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


# Set weights and config file paths
weightsPath = "weights/yolov3.weights"
configPath = "cfg/yolov3.cfg"

# Load YOLO with these files and the command below
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Get input video location and set up for writer and frame dimensions
intput = input("Enter video directory location\n")
cap = cv2.VideoCapture(intput)  # Capture video
# From here we can make changes to edit video properties:
# Uncomment only one to apply that effect
# cap = cv2.GaussianBlur(cap, (7, 7), 0) # Blurring
# cap = cv2.dilate(cap, kernel, iterations=1) # Dilation
# cap = cv2.erode(cap, kernel, iterations=1) # Erosion

writer = None
(W, H) = (None, None)

# Find total frames
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("{} frames in video".format(total))

frame_times = []  # Array for network passthrough time (per frame)
func_times = []  # Array for function time
confid = []  # Array of confidence values
confil = []  # Array of labels for corresponding confidence values

# Loop to get each frame
while True:
    start1 = time.time()
    (ret, frame) = cap.read()
    # If ret is false then at last frame of video
    if not ret:
        break
    # Grab frame dimensions if these have not been set yet (at start)
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Make blob of frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    net.setInput(blob)  # Input blob into YOLO network
    start = time.time()  # Time before detection
    layerOutputs = net.forward(ln)  # Forward pass
    end = time.time()  # Time after detection

    boxes = []
    confidences = []
    classIDs = []

    # Loop over layerOutputs
    for output in layerOutputs:
        # Loop over detections
        for detection in output:
            # Record relevant values in initialised arrays
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Apply confidence check to filter out weaker detections
            if confidence > args["confidence"]:

                # Determine bounding box dimensions and apply to frame
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # Find bottom left corner of bounding box using center and box width/height
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # Record values in arrays
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Non-maxima suppression function to remove replica boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])
    # If one detection exists
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Get bounding box values from boxes array for frame
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # Draw and label bounding box on frame
            colour = [int(c) for c in COLOURS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            confil.append(LABELS[classIDs[i]])  # Get labels for frame
            confid.append(confidences[i])  # Get confidence value for frame
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    # Uncomment section below to save video:
    #if writer is None:
        #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        #writer = cv2.VideoWriter('Resources/(name_here).mp4', fourcc, 30,
        #                         (frame.shape[1], frame.shape[0]), True)
    #writer.write(frame)



    # Display live frame detections
    # If executing with Julia notebook or Google Colab this will need to be changed
    # to cv2_imshow

    elap = (end - start)
    frame_times.append(elap)
    end1 = time.time()
    ftime = (end1 - start1)
    func_times.append(ftime)
    print("Network time: {:.4f} seconds".format(elap))
    print("Function time: {:.4f} seconds".format(ftime))

    cv2.imshow('detection', frame)

    # Cancel program early with q press
    if cv2.waitKey(1) == ord('q'):
        break

# release the file pointers
writer.release()
cap.release()
d = np.asarray([frame_times, func_times])
j = np.asarray([confil, confid])
np.savetxt("(savelocation)/(name).csv", d, delimiter=",")  # save times as csv file
np.savetxt("(savelocation)/(name).csv", j, delimiter=",", fmt="%s")  # save confidences as csv file
print("Done")