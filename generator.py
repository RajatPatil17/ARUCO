import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the dictionary we want to use
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Generate a marker
marker_size = 200  # Size in pixels
marker_image1 = cv2.aruco.generateImageMarker(aruco_dict, 41, marker_size)
marker_image2 = cv2.aruco.generateImageMarker(aruco_dict, 42, marker_size)

marker_image3 = cv2.aruco.generateImageMarker(aruco_dict, 43, marker_size)

marker_image4 = cv2.aruco.generateImageMarker(aruco_dict, 44, marker_size)


cv2.imwrite('marker_41.png', marker_image1)
cv2.imwrite('marker_42.png', marker_image2)

cv2.imwrite('marker_43.png', marker_image3)

cv2.imwrite('marker_44.png', marker_image4)


