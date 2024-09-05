import cv2 as cv
import sys
import os.path
import numpy as np

input_video_path = 'R:/ARUCO/test.mp4'  
output_video_path = 'output_ar.mp4'  


aruco = cv.aruco

cap = cv.VideoCapture(input_video_path)
print("Storing it as:", output_video_path)

# Get the video writer initialized to save the output video
vid_writer = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 28, 
                            (round(2 * cap.get(cv.CAP_PROP_FRAME_WIDTH)), 
                             round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

img = cv.imread("new_scenery.jpg")

while cv.waitKey(1) < 0:
    try:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print("Output file is stored as", output_video_path)
            cv.waitKey(1)
            break

        dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
        parameters =  cv.aruco.DetectorParameters()
        detector = cv.aruco.ArucoDetector(dictionary, parameters)

        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

        # Identify the specific markers (IDs 25, 33, 30, 23) to calculate the corners for the homography
        index = np.squeeze(np.where(markerIds == 25))
        refPt1 = np.squeeze(markerCorners[index[0]])[1]
        
        index = np.squeeze(np.where(markerIds == 33))
        refPt2 = np.squeeze(markerCorners[index[0]])[2]

        # Calculate distance between two reference points to maintain scaling
        distance = np.linalg.norm(refPt1 - refPt2)
        scalingFac = 0.02
        
        # Define the destination points based on the detected markers
        pts_dst = [
            [refPt1[0] - round(scalingFac * distance), refPt1[1] - round(scalingFac * distance)],
            [refPt2[0] + round(scalingFac * distance), refPt2[1] - round(scalingFac * distance)]
        ]
        
        index = np.squeeze(np.where(markerIds == 30))
        refPt3 = np.squeeze(markerCorners[index[0]])[0]
        pts_dst.append([refPt3[0] + round(scalingFac * distance), refPt3[1] + round(scalingFac * distance)])

        index = np.squeeze(np.where(markerIds == 23))
        refPt4 = np.squeeze(markerCorners[index[0]])[0]
        pts_dst.append([refPt4[0] - round(scalingFac * distance), refPt4[1] + round(scalingFac * distance)])

        # Source points from the image being used for augmentation
        pts_src = [[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]]
        
        # Convert points to NumPy arrays for further processing
        pts_src_m = np.asarray(pts_src)
        pts_dst_m = np.asarray(pts_dst)

        # Calculate the Homography matrix to warp the source image to the destination points
        h, status = cv.findHomography(pts_src_m, pts_dst_m)
        
        # Warp the source image to fit within the destination points in the video frame
        warped_image = cv.warpPerspective(img, h, (frame.shape[1], frame.shape[0]))
        
        # Prepare a mask representing the region to copy from the warped image into the original frame
        mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8)
        cv.fillConvexPoly(mask, np.int32([pts_dst_m]), (255, 255, 255), cv.LINE_AA)

        # Erode the mask to avoid copying boundary effects from the warping
        element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        mask = cv.erode(mask, element, iterations=3)

        # Copy the mask into 3 channels for further processing
        warped_image = warped_image.astype(float)
        mask3 = np.zeros_like(warped_image)
        for i in range(3):
            mask3[:, :, i] = mask / 255

        # Blend the warped image with the original frame using the mask
        warped_image_masked = cv.multiply(warped_image, mask3)
        frame_masked = cv.multiply(frame.astype(float), 1 - mask3)
        im_out = cv.add(warped_image_masked, frame_masked)
        
        # Showing the original image and the new output image side by side
        concatenatedOutput = cv.hconcat([frame.astype(float), im_out])
        cv.imshow("AR using Aruco markers", concatenatedOutput.astype(np.uint8))
        
        # Write the processed frame to the output video
        vid_writer.write(concatenatedOutput.astype(np.uint8))

    except Exception as e:
        print(e)

# Release resources and close windows
cv.destroyAllWindows()
cap.release()
vid_writer.release()
