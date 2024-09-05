import cv2
import numpy as np

# Load the image
image = cv2.imread('R:/ARUCO/tester.jpg')
image = cv2.resize(image, (960, 540))

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Create the ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Detect the markers
corners, ids, rejected = detector.detectMarkers(gray)
ids = ids.flatten()

# Print the detected markers
print("Detected markers: ", ids)
print('\n', corners)

# Initialize variables for the marker corners
topleft, topright, bottomright, bottomleft = None, None, None, None

# Loop through detected markers and assign corners based on marker IDs
for markercorner, markerid in zip(corners, ids):
    if markerid == 41:
        topleft = markercorner[0][0]
    elif markerid == 42:
        topright = markercorner[0][1]
    elif markerid == 43:
        bottomright = markercorner[0][2]
    elif markerid == 44:
        bottomleft = markercorner[0][3]

# If all corners are detected, draw lines connecting them
if topleft is not None and topright is not None and bottomright is not None and bottomleft is not None:
    # Convert points to integer tuples
    topleft = tuple(map(int, topleft))
    topright = tuple(map(int, topright))
    bottomright = tuple(map(int, bottomright))
    bottomleft = tuple(map(int, bottomleft))

    # Draw the lines across the box formed by ArUco markers
    cv2.line(image, topleft, topright, (0, 255, 0), 2)
    cv2.line(image, topright, bottomright, (0, 255, 0), 2)
    cv2.line(image, bottomright, bottomleft, (0, 255, 0), 2)
    cv2.line(image, bottomleft, topleft, (0, 255, 0), 2)

# Draw the detected markers on the image
if ids is not None:
    cv2.aruco.drawDetectedMarkers(image, corners, ids)

# Show the image with detected markers and lines
cv2.imshow('Detected Markers', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
