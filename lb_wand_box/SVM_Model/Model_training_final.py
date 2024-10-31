# -*- coding: utf-8 -*-
"""
This file is used to train a support vector machine using scikit-learn.
"""

import gc
import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import joblib
import psutil

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

label_encoder = '/home/leftbrain/Documents/src/lb_wand_box/SVM_Model/label_encoder.pkl'

#%% Functions

def print_memory_usage(stage=""):
    process = psutil.Process()
    mem_info=process.memory_info()
    print(f"[{stage}] Current memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")

def preview_image(img, title='Image Preview'):
    """ Function to display an image using Matplotlib """
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()
    
def visualize_hog(img, hog, cell_size=8):
    # Get the HOG descriptor for the image
    hog_desc = hog.compute(img)
    
    # Get image size and reshape HOG descriptor
    img_height, img_width = img.shape[:2]
    n_cells_x = img_width // cell_size  # Number of cells along x-axis
    n_cells_y = img_height // cell_size  # Number of cells along y-axis

    # Reshape the HOG descriptor to match cells
    hog_desc = hog_desc.reshape(n_cells_y, n_cells_x, -1)

    # Visualization setup
    cell_grid = np.zeros_like(img, dtype=np.float32)
    
    # Create a grid for visualizing HOG gradients
    for y in range(n_cells_y):
        for x in range(n_cells_x):
            # Get the HOG features for the current cell
            cell_hog = hog_desc[y, x]
            
            # Calculate the angle and magnitude of the gradients
            num_bins = cell_hog.shape[0]
            angle_step = 180 // num_bins
            
            for bin_idx in range(num_bins):
                angle = bin_idx * angle_step
                magnitude = cell_hog[bin_idx]

                # Draw the line for the gradient in this bin
                center = (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2)
                direction = (int(center[0] + magnitude * np.cos(np.radians(angle))),
                             int(center[1] - magnitude * np.sin(np.radians(angle))))
                
                cv2.line(cell_grid, center, direction, (255, 255, 255), 1)

    # Normalize the image for better visibility
    cell_grid = cv2.normalize(cell_grid, None, 0, 255, cv2.NORM_MINMAX)

    return cell_grid

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed.
        return img.copy()
    # Calculate skew based on central momemts.
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness.
    M = np.float32([[1, skew, -0.5*img.shape[0]*skew], [0, 1, 0]])
    # Apply affine transform
    img = cv2.warpAffine(img, M, (img.shape[0], img.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

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

def resize_with_aspect_ratio(img, target_size=(300, 300), padding_color=(0, 0, 0)):
    """
    Resize an image to the target size while preserving the aspect ratio.
    Pads the image with the specified color to maintain the aspect ratio.
    
    :param img: The input image in BGR format.
    :param target_size: The size to which the image will be resized.
    :param padding_color: RGB tuple to fill in the padding area.
    :return: The resized image with padding.
    """
    # Calculate the aspect ratio of the original image
    original_height, original_width = img.shape[:2]
    aspect_ratio = original_width / original_height

    # Calculate the target aspect ratio
    target_width, target_height = target_size
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        # Image is wider than target, fit by width
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Image is taller than target, fit by height
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # Resize image to fit within the target dimensions
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Create a new image with padding_color background
    padded_img = np.full((target_height, target_width, 3), padding_color, dtype=np.uint8)

    # Calculate padding offsets
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # Place the resized image on the padded background
    padded_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

    return padded_img

def plot_images(img_data,labels_predicted,label_actual):

    get_ipython().run_line_magic('matplotlib', 'qt')

    ncolrow = math.ceil(math.sqrt(len(img_data)))

    f = plt.figure()
    for i,img in enumerate(img_data):
        ax = f.add_subplot(ncolrow,ncolrow,i+1)
        ax.imshow(img, cmap = 'gray')
        ax.axis('off')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.1)

    plt.show()

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

def data_gen_flip_x(img, label):

    # Just flip other images
    img_flip = np.flip(img.copy(),axis = 1)

    # Flip and rotate for Lshape
    if "Lshape" in label:
        img_flip = np.rot90(img_flip)

    return img_flip


#%% Prepare Images
# Change this to where your images are stored
img_paths = r"/home/leftbrain/Desktop/Raspberry_Potter/Rb_potter_files"

batch_size = 200
current_batch = 0

img_data = []  # Re-initialize within the loop
hog_desc = []
labels = []
hog = make_hog()

all_labels = []
all_data = []

total_images = sum(len(files) for _, _, files in os.walk(img_paths))
num_batches = (total_images // batch_size) + 1

for path, subdirs, files in os.walk(img_paths):
    for name in files:
        img = cv2.imread(os.path.join(path, name))
        img_resize = bbox_and_resize(img)
        # preview_image(img_resize)
        print_memory_usage(f"After processing image {name}")
        img_hog_desc = hog.compute(img_resize)

        img_data.append(img_resize)
        hog_desc.append(img_hog_desc)

        label = os.path.basename(path)
        labels.append(label)

        # if (len(img_data) >= batch_size):
        #     print(f"Processing batch {current_batch + 1}/{num_batches}")

        #     all_labels.extend(labels)
        #     all_data.extend(hog_desc)

        #     # Clear and delete lists
        #     del img_data
        #     del hog_desc
        #     del labels
        #     gc.collect()

        #     # Re-initialize lists to avoid holding references
        #     img_data = []
        #     hog_desc = []
        #     labels = []

        #     current_batch += 1
        #     print_memory_usage(f"After processing batch {current_batch}")

        # Data augmentation by flipping images (optional)
        # img_flip = data_gen_flip_x(img_resize, label)
        # img_hog_desc_flip = hog.compute(img_flip)
        # img_data.append(img_flip)
        # hog_desc.append(img_hog_desc_flip)
        # labels.append(label)

# all_labels.extend(labels)
# all_data.extend(hog_desc)

print_memory_usage(f"After processing all batches")

# Remap Labels to numeric
le = preprocessing.LabelEncoder()
labels_numeric = le.fit_transform(labels)
print('dumping data');
joblib.dump(le, label_encoder)

# Combine data into one array
print ('Combining Data into one array')
hog_desc = [desc.flatten() for desc in hog_desc]  # Flatten each descriptor to a 1D array
data = np.array(hog_desc)

# Create test and training sets
print ('Creating Training Sets')
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels_numeric, test_size=0.1, random_state=42
)

#%% Train Model with scikit-learn
print ('Training scikit-learn')
svm = SVC(kernel='rbf', probability=True, verbose=True, decision_function_shape='ovr')
svm.fit(train_data, train_labels)

# Save trained model
print ('Saving Trained Model')
filepath = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(filepath, "svm_model_data_gen_rbpi.pkl")
joblib.dump(svm, filename)

#%% Test Model
print ('Testing Data')
test_response = svm.predict(test_data)

test_labels_predicted = le.inverse_transform(test_response)
test_labels_actual = le.inverse_transform(test_labels)

acc_array = np.equal(test_response, test_labels)
accuracy = np.mean(acc_array)

print(f"Accuracy: {accuracy * 100:5.2f}%")

print("Misclassified Samples:")
print("Predicted:", test_labels_predicted[~acc_array])
print("Actual:   ", test_labels_actual[~acc_array])

# Confidence tracking using probabilities
probabilities = svm.predict_proba(test_data)
print("Probabilities for the first test sample:", probabilities[0])

# Load the model and test again to verify
svm_loaded = joblib.load(filename)
test_response_loaded = svm_loaded.predict(test_data)

acc_array_loaded = np.equal(test_response_loaded, test_labels)
accuracy_loaded = np.mean(acc_array_loaded)

print(f"Accuracy after loading model: {accuracy_loaded * 100:5.2f}%")

#%% Optional Visualization (commented out)
index = np.random.randint(0, len(img_data))
img = img_data[index]
label = labels[index]
img_flip = data_gen_flip_x(img, label)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_flip, cmap='gray')
plt.title('Flipped')
plt.axis('off')
plt.show()
