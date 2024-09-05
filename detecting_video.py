import cv2
import numpy as np

# Initialize the video capture (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Define the ArUco dictionary and detector parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Create the ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

while True:
    # Capture frame-by-frame from the video
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Resize the frame (optional)
    frame = cv2.resize(frame, (960, 540))

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the markers
    corners, ids, rejected = detector.detectMarkers(gray)

    # Initialize variables for the marker corners
    topleft, topright, bottomright, bottomleft = None, None, None, None

    if ids is not None:
        ids = ids.flatten()

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
            cv2.line(frame, topleft, topright, (0, 255, 0), 2)
            cv2.line(frame, topright, bottomright, (0, 255, 0), 2)
            cv2.line(frame, bottomright, bottomleft, (0, 255, 0), 2)
            cv2.line(frame, bottomleft, topleft, (0, 255, 0), 2)

        # Draw the detected markers on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # Display the frame with detected markers and lines
    cv2.imshow('Detected Markers', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
