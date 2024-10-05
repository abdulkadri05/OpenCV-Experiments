import cv2  # OpenCV library for image and video processing
import mediapipe as mp  # MediaPipe for hand detection
import time  # Time library to calculate FPS

# Initialize video capture (webcam feed)
cap = cv2.VideoCapture(0)

# Initialize MediaPipe's hand detection module
mpHands = mp.solutions.hands  # Access MediaPipe hands solution
hands = mpHands.Hands()  # Create a Hands object to detect and track hand landmarks
mpDraw = mp.solutions.drawing_utils  # Drawing utility to draw landmarks and connections on the image
                                                                                        
# Initialize variables to calculate frame rate (FPS)
pTime = 0  # Previous time
cTime = 0  # Current time

# Loop to process frames in real-time
while True:
    # Capture frame from the webcam
    success, img = cap.read()

    # If the frame is not captured, display an error message and continue the loop
    if not success:
        print("Failed to capture image")
        continue

    # Convert the captured frame to RGB format (required by MediaPipe)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the frame for hand detection and get the results
    results = hands.process(imgRGB)

    # Print the landmarks of detected hands (None if no hands are detected)
    print(results.multi_hand_landmarks)

    # If hands are detected
    if results.multi_hand_landmarks:
        # Loop over each detected hand
        for handLms in results.multi_hand_landmarks:
            # Loop over each landmark in the detected hand
            for id, lm in enumerate(handLms.landmark):
                # Get the image dimensions (height, width, channels)
                h, w, c = img.shape
                # Convert normalized landmark coordinates (x, y) to pixel coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Print the landmark ID and its corresponding pixel coordinates
                print(id, cx, cy)

            # Draw the hand landmarks and connections on the image
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calculate frames per second (FPS)
    cTime = time.time()  # Get the current time
    fps = 1 / (cTime - pTime) if pTime > 0 else 0  # FPS calculation
    pTime = cTime  # Update previous time to current time

    # Display the FPS on the image
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 6)

    # Show the image in a window
    cv2.imshow('Image', img)

    # Press 'q' to exit the loop and close the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

"""
Explanation of the Code:

1. **Imports:**
    - `cv2`: OpenCV library is imported to handle video capture, image manipulation, and displaying results.
    - `mediapipe as mp`: MediaPipe is used for hand-tracking functionality.
    - `time`: Used to calculate the frames per second (FPS).

2. **Video Capture:**
    - `cap = cv2.VideoCapture(0)`: Opens the webcam to capture real-time video feed.
    - If it fails to capture, the loop continues without processing.

3. **MediaPipe Initialization:**
    - `mpHands = mp.solutions.hands`: Access the hands module from MediaPipe.
    - `hands = mpHands.Hands()`: Create an object to track hand landmarks.
    - `mpDraw = mp.solutions.drawing_utils`: A utility that allows drawing landmarks and connections on the image.

4. **FPS Calculation Variables:**
    - `pTime`: Stores the previous frame's time.
    - `cTime`: Stores the current frame's time.
    - These are used to calculate the FPS in real-time.

5. **Main Loop:**
    - The `while True` loop runs indefinitely, capturing each frame from the webcam.
    - The frame is read from the webcam using `cap.read()`. If successful, it is processed further, otherwise, an error message is printed.

6. **Image Processing:**
    - Convert the image from BGR (OpenCV format) to RGB (`imgRGB`) because MediaPipe requires RGB.
    - `hands.process(imgRGB)`: This method processes the frame and detects any hand landmarks present.

7. **Hand Landmark Processing:**
    - The program checks if any hand landmarks are detected (`results.multi_hand_landmarks`).
    - If detected, it iterates over each detected hand.
    - For each hand, it loops over the landmarks, and for each landmark, it converts normalized coordinates (ranging from 0 to 1) into pixel coordinates based on the image's width and height.
    - The ID and coordinates of each landmark are printed.

8. **Drawing Hand Landmarks:**
    - `mpDraw.draw_landmarks()`: This draws the landmarks and their connections on the captured image.

9. **FPS Calculation:**
    - The time between frames is used to calculate the FPS (`fps = 1 / (cTime - pTime)`), which is then displayed on the image using `cv2.putText()`.

10. **Display and Exit:**
    - The processed frame (with landmarks and FPS) is shown in a window using `cv2.imshow()`.
    - Press 'q' to exit the loop and stop the webcam feed.

11. **Cleanup:**
    - `cap.release()` releases the webcam.
    - `cv2.destroyAllWindows()` closes all OpenCV windows to clean up resources.
"""
