import cv2
import os
import HandTrackingModule as htm  # Import the custom HandTrackingModule

# Set the width and height of the webcam feed
wCam, hCam = 640, 480

# Load the finger images (0.jpg, 1.jpg, etc.) and resize them
folderPath = "FingerImages"
myList = os.listdir(folderPath)
imgList = [cv2.resize(cv2.imread(f'{folderPath}/{imgPath}'), (200, 300)) for imgPath in myList]

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize hand detector
detector = htm.handDetector(detectionCon=0.75)

# Tip IDs for the 5 fingers
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    # Find the hands in the image
    img = detector.findHands(img)

    # Get the landmark positions
    lmList = detector.findPosition(img, draw=False)

    # If landmarks are found, count fingers
    if len(lmList) != 0:
        fingers = []

        # Thumb (special case since it moves horizontally)
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # For other fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Total fingers counted
        totalFingers = fingers.count(1)

        # Display the corresponding finger count image
        h, w, c = imgList[totalFingers].shape
        img[0:h, 0:w] = imgList[totalFingers]

    # Show the image
    cv2.imshow('Image', img)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
