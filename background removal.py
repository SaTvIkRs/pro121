import cv2
import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting frame width and frame height as 640 x 480
camera.set(3, 640)
camera.set(4, 480)

while True:
    # read a frame from the attached camera
    status, frame = camera.read()

    # if we got the frame successfully
    if status:
        # flip it
        frame = cv2.flip(frame, 1)

        # convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # perform thresholding
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # invert the mask
        mask = cv2.bitwise_not(thresh)

        # bitwise and operation to extract foreground / person
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # show the resulting image
        cv2.imshow('Final Image', result)
        cv2.imshow('Original Frame', frame)

        # wait for 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code == 32:  # Break loop when spacebar is pressed
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
