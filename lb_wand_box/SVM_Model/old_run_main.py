import cv2
import os
import numpy as np
import time
import subprocess
import joblib

from picamera2 import Picamera2


# initialize the camera and configure
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()

model_path = '/home/leftbrain/Documents/src/lb_wand_box/SVM_Model/svm_model_data_gen_rbpi.yml'
label_encoder = '/home/leftbrain/Documents/src/lb_wand_box/SVM_Model/label_encoder.pkl'
test_img_path = '/home/leftbrain/Desktop/Raspberry_Potter/temp/current_img_to_review.jpg'
output_folder = "/home/leftbrain/Desktop/Raspberry_Potter/temp"

os.makedirs(output_folder, exist_ok=True)
svm = cv2.ml.SVM_load(model_path)
le = joblib.load(label_encoder)

confidence_threshold = 0.3

# allow the camera to warmup
time.sleep(1.0)

# Initialize Kalman filter
kalman = cv2.KalmanFilter(4, 2)  # 4 dynamic params, 2 measured params
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # Adjust process noise
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1  # Adjust measurement noise

# State initialization
state = np.zeros((4, 1), dtype=np.float32)  # x, y, dx, dy
measurement = np.zeros((2, 1), dtype=np.float32)  # x, y

# Initialize variables for tracking
prev_points = None
prev_gray = None
point_history = []  # Store points over the last 90 frames
line_history = []   # Store lines (start and end points) for the last 90 frames
history_size = 55
scale_factor = 2  # Factor to extend the line by

# Get the frame size to calculate the minimum movement threshold
frame_width = 640
frame_height = 480

# Minimum movement distance: 2-3% of the diagonal of the frame
min_movement_threshold = 0.005 * np.sqrt(frame_width**2 + frame_height**2)  

params = cv2.SimpleBlobDetector_Params()
params.filterByColor = 1
params.blobColor = 255
params.filterByCircularity = 1
params.minCircularity = 0.5
params.filterByConvexity = 1
params.minConvexity = 0.6
params.filterByArea = 1
params.minArea = 8
params.maxArea = 100

detector = cv2.SimpleBlobDetector_create(params)

# Optical flow parameters
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

last_line_drawn_time = time.time()
clear_frame_threshold = .7

# Function to create the HOG descriptor
def make_hog():
    winSize = (25, 25)  # Use the window size from your training
    blockSize = (10, 10)  # Match these with your training values
    blockStride = (5, 5)
    cellSize = (10, 10)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    return hog

# Preprocess the image (resize and HOG descriptor extraction)
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found at {img_path}")
    
    # Ensure the image is resized to 100x100
    img_resized = cv2.resize(img, (100, 100))
    
    # Compute HOG features
    hog = make_hog()
    hog_desc = hog.compute(img_resized)
    
    # Ensure the HOG features are reshaped and converted to CV_32F
    hog_desc = hog_desc.reshape(1, -1).astype(np.float32)
    
    # # Print shape and data type for debugging
    # print(f"HOG descriptor shape: {hog_desc.shape}")
    # print(f"HOG descriptor data type: {hog_desc.dtype}")
    
    return hog_desc

# Load the pre-trained SVM_Model and predict the label
def predict_image():
    # Preprocess the image to get HOG features
    hog_features = preprocess_image(test_img_path)
    
    # Ensure the HOG features are a 2D array with a single sample (1, -1) shape
    hog_features = hog_features.reshape(1, -1).astype(np.float32)

    # Get the decision function (distance from the hyperplane)
    _, result = svm.predict(hog_features, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
    decision_value = result[0, 0]

    # Check confidence (absolute distance from the decision boundary)
    if abs(decision_value) > confidence_threshold:
        print(f"Confidence: {decision_value}")
        return None  # Model is not confident enough, return None (or "null" equivalent)

    # Run the prediction
    _, label_result = svm.predict(hog_features)

    # The prediction result needs to be in a 1D array format for label decoding
    predicted_label = label_result.flatten()

    # Decode the label using the label encoder
    decoded_label = le.inverse_transform(predicted_label.astype(int))

    return decoded_label[0]  # Return the decoded label


def filter_significant_movement(point_history, threshold=2):
    """ Filters out points that have moved significantly more than the threshold """
    if len(point_history) < 2:
        return False  # Not enough points to compare

    old_point = point_history[-2]
    new_point = point_history[-1]

    distance = np.linalg.norm(np.array(new_point) - np.array(old_point))
    return distance > threshold

def calculate_distance(p1, p2):
    """ Calculate the Euclidean distance between two points """
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Function to save a screenshot
def save_screenshot(frame):
    # Generate a random file name
    filename = 'current_img_to_review.jpg'

    # Create the full path for the file
    file_path = os.path.join(output_folder, filename)

    # Save the image
    cv2.imwrite(file_path, frame)
    # print(f"Screenshot saved to {file_path}")

frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

# Main loop
while True:
    # Capture frame-by-frame for processing
    processing_frame = picam2.capture_array()

    # Create a black background (instead of using the video frame)

    # Convert BGR to HSV and Grayscale
    hsv = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)

    # Define the HSV range for the IR light (you may need to tweak these values)
    lower_hsv = np.array([0, 0, 225])
    upper_hsv = np.array([200, 100, 255])

    # Threshold the HSV image to get only the wand tip
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Apply morphological operation to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Detect blobs in the mask
    keypoints = detector.detect(mask)

    # If keypoints are detected, update Kalman filter
    if isinstance(keypoints, list) and len(keypoints) > 0:
        x, y = int(keypoints[0].pt[0]), int(keypoints[0].pt[1])
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        kalman.correct(measurement)

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2)
            cv2.circle(frame, (x, y), radius + 3, (255, 0, 0), 3)  # Blue thick outline
            cv2.circle(frame, (x, y), radius, (0, 255, 255), -1)   # Filled yellow keypoint
    else:
        # If no keypoint is detected, predict the position using Kalman filter
        prediction = kalman.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])
        cv2.circle(frame, (pred_x, pred_y), 10, (0, 0, 255), 2)  # Red prediction circle

    # Track points using Optical Flow
    line_drawn = False
    if prev_points is not None and prev_gray is not None:
        new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)
        if new_points is not None and status is not None:
            good_new = new_points[status == 1]
            good_old = prev_points[status == 1]

            if len(good_new) > 0 and len(good_old) > 0:
                for (new, old) in zip(good_new, good_old):
                    a, b = new.ravel()
                    c, d = old.ravel()

                    point_history.append((a, b))
                    if len(point_history) > history_size:
                        point_history.pop(0)

                    # Calculate the movement distance
                    movement_distance = calculate_distance((a, b), (c, d))
                    print(f"a: {a}, b: {b} -- c: {c}, d: {d} -- mv: {movement_distance}")
                    # print (str(a) + ", " + str(b) + " - " + str(c) + ", " + str(d));

                    # Only draw lines and store if movement exceeds threshold (2-3% of view)
                    if movement_distance > min_movement_threshold:
                        line_drawn = True
                        last_line_drawn_time = time.time()

                        cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

                        # Draw the line connecting old and new points
                        cv2.line(frame, (int(c), int(d)), (int(a), int(b)), (255, 0, 0), 5)

                        # Store the line segment in line_history for the last 90 frames
                        line_history.append(((int(c), int(d)), (int(a), int(b))))
                        if len(line_history) > history_size:
                            line_history.pop(0)  # Keep only the last 90 frames of lines

                        # Calculate direction vector and extend the line
                        direction = np.array([a - c, b - d])
                        extended_point = np.array([a, b]) + scale_factor * direction
                        extended_x, extended_y = int(extended_point[0]), int(extended_point[1])
                        cv2.line(frame, (int(a), int(b)), (extended_x, extended_y), (0, 0, 255), 3)
    else:
        print("No previous points or grayscale image available")

    # If no new lines were drawn and it's been more than 0.5 seconds, clear the frame
    current_time = time.time()
    # print (current_time - last_line_drawn_time)
    if not line_drawn and (current_time - last_line_drawn_time) > clear_frame_threshold:
        # cv2.imshow('Motion Tracking', cv2.flip(frame, 1))

        # cv2.waitKey(1)
        save_screenshot(frame)

        try:
            predicted_label = predict_image()
            if predicted_label != None:
                print(f"Predicted label index: {predicted_label}")

                if predicted_label == "ascendio":
                    command = ['python3', '/home/leftbrain/Documents/src/lb_wand_box/servo-control/servo.py', "1"]

                if predicted_label == "descendio":
                    command = ['python3', '/home/leftbrain/Documents/src/lb_wand_box/servo-control/servo.py', "0"]

                if predicted_label == 'incendio':
                    command = ['vlc', '/home/leftbrain/Documents/src/lb_wand_box/animations/incendio.mp4', "--play-and-exit", "--fullscreen"]

                if predicted_label == 'leviosa':
                    command = ['vlc', '/home/leftbrain/Documents/src/lb_wand_box/animations/leviosa.mov', "--play-and-exit", "--fullscreen"]

                if predicted_label == 'revelio':
                    command = ['vlc', '/home/leftbrain/Documents/src/lb_wand_box/animations/revelio.mp4', "--play-and-exit", "--fullscreen"]


                # Run the command
                result = subprocess.run(command, capture_output=True, text=True)

        except Exception as e:
            print(f"Error during prediction: {e}")

        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        line_history = []

    # Update previous frame and points for next iteration
    prev_gray = gray.copy()
    if keypoints and len(keypoints) > 0:
        prev_points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

    # Draw all the blue lines stored in line_history
    for idx, line in enumerate(line_history):
        # Calculate the ratio of how old the line is (0 is the oldest, 1 is the newest)
        fade_ratio = idx / (len(line_history) - 1) if len(line_history) > 1 else 1

        # Interpolate between blue and white
        blue = int(255 * (1 - fade_ratio))   # Fades from 255 to 0 (blue to white)
        white = int(255 * fade_ratio)        # Fades from 0 to 255 (blue to white)

        # Define the color based on the fade ratio
        color = (blue, blue, white)  # (R, G, B) -> Blue fades to White

        # Draw the line with the interpolated color
        cv2.line(frame, line[0], line[1], color, 5)

    # Display the resulting frame
    cv2.imshow('Motion Tracking', cv2.flip(frame, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # if cv2.waitKey(1) & 0xFF == ord('s'):

# Release the capture and close windows
cv2.destroyAllWindows()
