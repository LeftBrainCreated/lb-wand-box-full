import cv2
import os
import numpy as np
import time
import subprocess
import joblib
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt



from picamera2 import Picamera2

# Initialize the camera and configure
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()

# Update model paths
model_path = '/home/leftbrain/Documents/src/lb_wand_box/SVM Model/svm_model_data_gen_rbpi.pkl'  # Updated extension to .pkl
label_encoder_path = '/home/leftbrain/Documents/src/lb_wand_box/SVM Model/label_encoder.pkl'
test_img_path = '/home/leftbrain/Desktop/Raspberry_Potter/temp/current_img_to_review.jpg'
output_folder = "/home/leftbrain/Desktop/Raspberry_Potter/temp"

os.makedirs(output_folder, exist_ok=True)

# Load the scikit-learn SVM model and label encoder
svm = joblib.load(model_path)
le = joblib.load(label_encoder_path)

confidence_threshold = 0.75

# Allow the camera to warm up
time.sleep(1.0)

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

# Minimum movement distance: 0.5% of the diagonal of the frame
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
clear_frame_threshold = 0.7

# Function to create the HOG descriptor
def make_hog():
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize \
                            ,nbins,derivAperture,winSigma,histogramNormType \
                            ,L2HysThreshold,gammaCorrection,nlevels, signedGradients)
    return hog

def bbox_and_resize(img, target_size=(100, 100)):
    """
    Resize the image to a lower resolution while retaining color.
    
    Parameters:
    - img: Input image (BGR format as read by OpenCV).
    - target_size: Tuple indicating the target resolution (width, height).
    
    Returns:
    - Resized image with retained color.
    """
    # Convert to grayscale only for thresholding purpose, but keep original color image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to binary image
    ret, thresh = cv2.threshold(gray, 15, 255, 0)

    # Apply bounding box around the symbol
    x, y, w, h = cv2.boundingRect(thresh)

    bp = 4  # Border pixels

    rs = max(0, y - bp)  # Starting row
    re = min(img.shape[0], y + h + bp)  # Ending row
    cs = max(0, x - bp)  # Starting col
    ce = min(img.shape[1], x + w + bp)  # Ending col

    # Crop the image but keep color
    cropped_img = img[rs:re, cs:ce]

    # Resize to a lower resolution while retaining color
    resized_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_LINEAR)

    return resized_img

# Preprocess the image (resize and HOG descriptor extraction)
def preprocess_image(img_path, target_size=(100, 100)):
    """
    Preprocess the image for prediction: resize to the target size while retaining color.
    
    Parameters:
    - img_path: Path to the image file.
    - target_size: Tuple indicating the target resolution (width, height).
    
    Returns:
    - HOG features extracted from the resized color image.
    """
    img = cv2.imread(img_path)  # Read the image in color (BGR format)
    if img is None:
        raise ValueError(f"Image not found at {img_path}")

    # Resize the image to the same resolution used in training
    img_resized = bbox_and_resize(img, target_size=(100, 100))  # Consistent with training

    # Compute HOG features
    hog = make_hog()
    hog_desc = hog.compute(img_resized)

    # Ensure the HOG features are reshaped and converted to the appropriate data type
    hog_desc = hog_desc.reshape(1, -1)

    return hog_desc

# # Load the pre-trained SVM model and predict the label
# def predict_image():
#     # Preprocess the image to get HOG features
#     hog_features = preprocess_image(test_img_path)

#     # Use the scikit-learn SVM model for prediction
#     # Get probability estimates for confidence tracking
#     probabilities = svm.predict_proba(hog_features)
#     max_prob = np.max(probabilities)
#     predicted_label_index = np.argmax(probabilities)

#     # Check if the confidence exceeds the threshold
#     if max_prob < confidence_threshold:
#         print(f"Confidence ({max_prob}) is below the threshold.")
#         return None  # Model is not confident enough

#     # Decode the label using the label encoder
#     predicted_label = le.inverse_transform([predicted_label_index])

#     print(f"Predicted label: {predicted_label[0]}, Confidence: {max_prob}")
#     return predicted_label[0]  # Return the decoded label

def predict_image():
    # Preprocess the image to get HOG features
    hog_features = preprocess_image(test_img_path, target_size=(100, 100))  # Adjust target_size if necessary

    # Use the scikit-learn SVM model for prediction
    # Get probability estimates for confidence tracking
    probabilities = svm.predict_proba(hog_features)[0]  # Get the first (and only) sample's probabilities
    max_prob = np.max(probabilities)
    predicted_label_index = np.argmax(probabilities)

    # Check if the confidence exceeds the threshold
    if max_prob < confidence_threshold:
        print(f"Confidence ({max_prob}) is below the threshold.")
        return None, probabilities  # Return probabilities for visualization

    # Decode the label using the label encoder
    predicted_label = le.inverse_transform([predicted_label_index])

    print(f"Predicted label: {predicted_label[0]}, Confidence: {max_prob}")
    return predicted_label[0], probabilities  # Return probabilities

def plot_probabilities(probabilities, class_labels):
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(class_labels))
    plt.barh(y_pos, probabilities, align='center', alpha=0.7)
    plt.yticks(y_pos, class_labels)
    plt.xlabel('Probability')
    plt.title('Class Probabilities')
    plt.gca().invert_yaxis()  # Highest probability on top
    plt.show()

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
    # Generate a file name
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

    # If keypoints are detected, update tracking points
    if isinstance(keypoints, list) and len(keypoints) > 0:
        x, y = int(keypoints[0].pt[0]), int(keypoints[0].pt[1])

        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2)
            cv2.circle(frame, (x, y), radius + 3, (255, 0, 0), 3)  # Blue thick outline
            cv2.circle(frame, (x, y), radius, (0, 255, 255), -1)   # Filled yellow keypoint
    else:
        # If no keypoint is detected, skip tracking
        pass

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
                    # print(f"a: {a}, b: {b} -- c: {c}, d: {d} -- mv: {movement_distance}")

                    # Only draw lines and store if movement exceeds threshold
                    if movement_distance > min_movement_threshold:
                        line_drawn = True
                        last_line_drawn_time = time.time()

                        cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

                        # Draw the line connecting old and new points
                        cv2.line(frame, (int(c), int(d)), (int(a), int(b)), (255, 0, 0), 5)

                        # Store the line segment in line_history
                        line_history.append(((int(c), int(d)), (int(a), int(b))))
                        if len(line_history) > history_size:
                            line_history.pop(0)  # Keep only the last 'history_size' frames of lines

                        # Calculate direction vector and extend the line
                        direction = np.array([a - c, b - d])
                        extended_point = np.array([a, b]) + scale_factor * direction
                        extended_x, extended_y = int(extended_point[0]), int(extended_point[1])
                        cv2.line(frame, (int(a), int(b)), (extended_x, extended_y), (0, 0, 255), 3)
    else:
        # print("No previous points or grayscale image available")
        pass

    # If no new lines were drawn and it's been more than 0.7 seconds, process the frame
    current_time = time.time()
    if not line_drawn and (current_time - last_line_drawn_time) > clear_frame_threshold:
        save_screenshot(frame)

        try:
            predicted_label, probabilities = predict_image()
            if predicted_label is not None:
                print(f"Predicted label: {predicted_label}")

                # Get the class labels from the label encoder
                class_labels = le.classes_

                # Plot the probabilities
                # plot_probabilities(probabilities, class_labels)

                if predicted_label == "ascendio":
                    command = ['python3', '/home/leftbrain/Documents/src/lb_wand_box/servo-control/servo.py', "1"]
                elif predicted_label == "descendio":
                    command = ['python3', '/home/leftbrain/Documents/src/lb_wand_box/servo-control/servo.py', "0"]
                elif predicted_label == 'incendio':
                    command = ['cvlc', '/home/leftbrain/Documents/src/lb_wand_box/animations/incendio.mp4', "--play-and-exit", "--fullscreen"]
                elif predicted_label == 'leviosa':
                    command = ['cvlc', '/home/leftbrain/Documents/src/lb_wand_box/animations/leviosa.mp4', "--play-and-exit", "--fullscreen"]
                elif predicted_label == 'arresto':
                    command = ['cvlc', '/home/leftbrain/Documents/src/lb_wand_box/animations/arresto.mp4', "--play-and-exit", "--fullscreen"]
                elif predicted_label == 'herbivicus':
                    command = ['cvlc', '/home/leftbrain/Documents/src/lb_wand_box/animations/herbivicus.mp4', "--play-and-exit", "--fullscreen"]
                elif predicted_label == 'revelio':
                    command = ['python3', '/home/leftbrain/Documents/src/lb_wand_box/servo-control/servo.py', "1"]
                else:
                    command = None

                if command:
                    # Run the command
                    result = subprocess.run(command, capture_output=True, text=True)
                    print("Standard Output:", result.stdout)
                    print("Error Output:", result.stderr)                    
                    # You might want to handle the result or any errors here

        except Exception as e:
            print(f"Error during prediction: {e}")

        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        line_history = []
        last_line_drawn_time = time.time()  # Reset the timer after processing

    # Update previous frame and points for next iteration
    prev_gray = gray.copy()
    if keypoints and len(keypoints) > 0:
        prev_points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
    else:
        prev_points = None

    # Draw all the blue lines stored in line_history
    for idx, line in enumerate(line_history):
        # Calculate the ratio of how old the line is (0 is the oldest, 1 is the newest)
        fade_ratio = idx / (len(line_history) - 1) if len(line_history) > 1 else 1

        # Interpolate between blue and white
        blue = int(255 * (1 - fade_ratio))   # Fades from 255 to 0 (blue to white)
        white = int(255 * fade_ratio)        # Fades from 0 to 255 (blue to white)

        # Define the color based on the fade ratio
        color = (blue, blue, white)  # (B, G, R) -> Blue fades to White

        # Draw the line with the interpolated color
        cv2.line(frame, line[0], line[1], color, 5)

    # Display the resulting frame
    cv2.namedWindow("Motion Tracking", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Motion Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Motion Tracking', cv2.flip(frame, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cv2.destroyAllWindows()
