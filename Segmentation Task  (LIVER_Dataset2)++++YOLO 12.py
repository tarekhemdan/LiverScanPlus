# Check GPU
!nvidia-smi

# Clear output for better readability
from IPython.display import clear_output
clear_output()

# Install the necessary libraries
!pip install ultralytics roboflow opencv-python pandas matplotlib scikit-learn

# Import necessary libraries
from ultralytics import YOLO
from IPython.display import Image, display
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import pandas as pd
import cv2
from google.colab.patches import cv2_imshow
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
from roboflow import Roboflow
import matplotlib.image as mpimg

##################################################################################################
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="xs542BAQQ4nGbfcWQdcq")
project = rf.workspace("segmentasiliver").project("project-5c-liver-tumor-aikum")
version = project.version(1)
dataset = version.download("yolov12")
                
#################################################################################################                

# Path to dataset
data_path = '/content/Project-5C-Liver-Tumor-1'

# Path to the data.yaml file
data_yaml_path = os.path.join(dataset.location, 'data.yaml')

##################################################################################################
# Train the YOLO model
model = YOLO('yolo12n-seg.yaml')  # Load a pretrained YOLO model

# Train the model
model.train(task='segment', mode='train', data=data_yaml_path, epochs=100)

##################################################################################################
# Validate the model
metrics = model.val()

# Print the evaluation metrics
print("Model Evaluation Metrics:", metrics)

##################################################################################################
# Show Confusion Matrix
from IPython.display import Image, display
display(Image('/content/runs/segment/train/confusion_matrix.png'))

##################################################################################################
# Show Results (image)
display(Image('/content/runs/segment/train/results.png'))

##################################################################################################
# Show results (table)
import pandas as pd

# Load the CSV file into a pandas DataFrame
csv_file_path = '/content/runs/segment/train/results.csv'
df = pd.read_csv(csv_file_path)

# Display the DataFrame
print(df)
##################################################################################################

# Package imports and train path
import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

train_path = '/content/runs/segment/train'
# Show box curves of F1-Confident, Precision-Recall, Precision-Confident and Recall-Confident

# Load your images using OpenCV
img1 = cv2.imread(f'{train_path}/BoxF1_curve.png')
img2 = cv2.imread(f'{train_path}/BoxPR_curve.png')
img3 = cv2.imread(f'{train_path}/BoxP_curve.png')
img4 = cv2.imread(f'{train_path}/BoxR_curve.png')

# Concatenate images horizontally
top_row = cv2.hconcat([img1, img2])
bottom_row = cv2.hconcat([img3, img4])

# Concatenate the two rows vertically
grid = cv2.vconcat([top_row, bottom_row])

# Display the concatenated image
cv2_imshow(grid)

##################################################################################################

# Show mask curves of F1-Confident, Precision-Recall, Precision-Confident and Recall-Confident

# Load your images using OpenCV
img1 = cv2.imread(f'{train_path}/MaskF1_curve.png')
img2 = cv2.imread(f'{train_path}/MaskPR_curve.png')
img3 = cv2.imread(f'{train_path}/MaskP_curve.png')
img4 = cv2.imread(f'{train_path}/MaskR_curve.png')

# Concatenate images horizontally
top_row = cv2.hconcat([img1, img2])
bottom_row = cv2.hconcat([img3, img4])

# Concatenate the two rows vertically
grid = cv2.vconcat([top_row, bottom_row])

# Display the concatenated image
cv2_imshow(grid)

##################################################################################################
# Validation Results: Ground Truth vs Prediction

# Load your images using matplotlib's imread
img1 = mpimg.imread(f'{train_path}/val_batch0_labels.jpg')
img2 = mpimg.imread(f'{train_path}/val_batch0_pred.jpg')
img3 = mpimg.imread(f'{train_path}/val_batch1_labels.jpg')
img4 = mpimg.imread(f'{train_path}/val_batch1_pred.jpg')

# Get image dimensions (assuming all images are of the same size)
img_height, img_width, _ = img1.shape

# Create a figure with subplots, adjusting the figure size to match the image size
fig, axes = plt.subplots(4, 1, figsize=(img_width / 100, 4 * img_height / 100))  # 4 rows, 1 column

# Add a title to the figure with a bigger font size
# fig.suptitle('Validation Results: Ground Truth vs Prediction', fontsize=24)

# Plot the images and add larger titles for each subplot
axes[0].imshow(img1)
axes[0].set_title("Ground Truth - Batch 0", fontsize=15)  # Larger subplot title
axes[1].imshow(img2)
axes[1].set_title("Prediction - Batch 0", fontsize=15)

axes[2].imshow(img3)
axes[2].set_title("Ground Truth - Batch 1", fontsize=15)
axes[3].imshow(img4)
axes[3].set_title("Prediction - Batch 1", fontsize=15)

# Turn off axes for all subplots
for ax in axes:
    ax.axis('off')

# Adjust layout to prevent overlap and show the title
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Display the plot
plt.show()
##################################################################################################
# Predict on test data using trained model

# Load trained model
model = YOLO('/content/runs/segment/train/weights/best.pt')

# Specify the test folder path
test_path = '/content/Project-5C-Liver-Tumor-1/test/images'

# List all images in the test folder
image_files = [os.path.join(test_path, img) for img in os.listdir(test_path) if img.endswith(('.jpg', '.jpeg', '.png'))]

# Predict each images
for img_path in image_files:
    results = model.predict(source=img_path, save=True, conf=0.50)
##################################################################################################

# Plot prediction on test images

# Specify the predict folder path
pred_path = '/content/runs/segment/predict'

# List all images in the predict folder
pred_image_files = [os.path.join(pred_path, img) for img in os.listdir(pred_path) if img.endswith(('.jpg', '.jpeg', '.png'))]

# Number of images per row
images_per_row = 5

# Calculate the number of rows needed
n_rows = len(pred_image_files) // images_per_row + int(len(pred_image_files) % images_per_row != 0)

# Set figure size
fig, axs = plt.subplots(n_rows, images_per_row, figsize=(15, 3 * n_rows))

# Flatten axes if necessary (for easier iteration)
axs = axs.flatten()

# Loop through each image and display it
for i, img_path in enumerate(pred_image_files):
    img = mpimg.imread(img_path)      # Read the image
    axs[i].imshow(img)                # Show the image
    axs[i].axis('off')                # Turn off axis
    axs[i].set_title(f"Image {i+1}")  # Add title

# Hide any extra empty subplots (if the number of images is not a perfect multiple of images_per_row)
for j in range(i+1, len(axs)):
    axs[j].axis('off')  # Hide unused axes

# Display all images
plt.tight_layout()
plt.show()

###############################################################################################
#Write results in File
###############################################################################################
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from google.colab.patches import cv2_imshow
from IPython.display import Image, display
import pandas as pd
from ultralytics import YOLO

# Function to write results to a text file
def write_to_file(file_path, content):
    with open(file_path, 'a') as file:
        file.write(content + '\n')

# File to save results
results_file = '/content/Segmentation_Results.txt'

# Validate the model
metrics = model.val()

# Write evaluation metrics to file
write_to_file(results_file, "Model Evaluation Metrics:")
write_to_file(results_file, str(metrics))

##################################################################################################
# Show Confusion Matrix
confusion_matrix_path = '/content/runs/segment/train/confusion_matrix.png'
display(Image(confusion_matrix_path))
write_to_file(results_file, f"Confusion Matrix saved at: {confusion_matrix_path}")

##################################################################################################
# Show Results (image)
results_image_path = '/content/runs/segment/train/results.png'
display(Image(results_image_path))
write_to_file(results_file, f"Results image saved at: {results_image_path}")

##################################################################################################
# Show results (table)
csv_file_path = '/content/runs/segment/train/results.csv'
df = pd.read_csv(csv_file_path)
write_to_file(results_file, "Results Table:")
write_to_file(results_file, df.to_string())

##################################################################################################
# Package imports and train path
train_path = '/content/runs/segment/train'

# Show box curves of F1-Confident, Precision-Recall, Precision-Confident and Recall-Confident
box_images = [
    f'{train_path}/BoxF1_curve.png',
    f'{train_path}/BoxPR_curve.png',
    f'{train_path}/BoxP_curve.png',
    f'{train_path}/BoxR_curve.png'
]

# Concatenate images horizontally
top_row = cv2.hconcat([cv2.imread(img) for img in box_images[:2]])
bottom_row = cv2.hconcat([cv2.imread(img) for img in box_images[2:]])
grid = cv2.vconcat([top_row, bottom_row])

# Display the concatenated image
cv2_imshow(grid)
write_to_file(results_file, "Box curves displayed.")

##################################################################################################
# Show mask curves of F1-Confident, Precision-Recall, Precision-Confident and Recall-Confident
mask_images = [
    f'{train_path}/MaskF1_curve.png',
    f'{train_path}/MaskPR_curve.png',
    f'{train_path}/MaskP_curve.png',
    f'{train_path}/MaskR_curve.png'
]

# Concatenate images horizontally
top_row = cv2.hconcat([cv2.imread(img) for img in mask_images[:2]])
bottom_row = cv2.hconcat([cv2.imread(img) for img in mask_images[2:]])
grid = cv2.vconcat([top_row, bottom_row])

# Display the concatenated image
cv2_imshow(grid)
write_to_file(results_file, "Mask curves displayed.")

##################################################################################################
# Validation Results: Ground Truth vs Prediction
val_images = [
    f'{train_path}/val_batch0_labels.jpg',
    f'{train_path}/val_batch0_pred.jpg',
    f'{train_path}/val_batch1_labels.jpg',
    f'{train_path}/val_batch1_pred.jpg'
]

# Load images and check for "Normal" class
titles = []
for img_path in val_images:
    if "Normal" in img_path:
        titles.append("Normal")
    else:
        titles.append(os.path.basename(img_path))

# Create a figure with subplots
fig, axes = plt.subplots(4, 1, figsize=(10, 20))

# Plot the images and add titles
for i, (img_path, title) in enumerate(zip(val_images, titles)):
    img = mpimg.imread(img_path)
    axes[i].imshow(img)
    axes[i].set_title(title, fontsize=15)
    axes[i].axis('off')

# Adjust layout and display
plt.tight_layout()
plt.show()
write_to_file(results_file, "Validation Results: Ground Truth vs Prediction displayed.")

##################################################################################################
# Predict on test data using trained model
model = YOLO('/content/runs/segment/train/weights/best.pt')
test_path = '/content/Project-5C-Liver-Tumor-1/test/images'
image_files = [os.path.join(test_path, img) for img in os.listdir(test_path) if img.endswith(('.jpg', '.jpeg', '.png'))]

# Predict each image and save results
for img_path in image_files:
    results = model.predict(source=img_path, save=True, conf=0.50)
    if "Normal" in results:
        write_to_file(results_file, f"Image {os.path.basename(img_path)} classified as Normal.")

##################################################################################################
# Plot prediction on test images
pred_path = '/content/runs/segment/predict'
pred_image_files = [os.path.join(pred_path, img) for img in os.listdir(pred_path) if img.endswith(('.jpg', '.jpeg', '.png'))]

# Number of images per row
images_per_row = 5
n_rows = len(pred_image_files) // images_per_row + int(len(pred_image_files) % images_per_row != 0)

# Set figure size
fig, axs = plt.subplots(n_rows, images_per_row, figsize=(15, 3 * n_rows))
axs = axs.flatten()

# Loop through each image and display it
for i, img_path in enumerate(pred_image_files):
    img = mpimg.imread(img_path)
    axs[i].imshow(img)
    axs[i].axis('off')
    if "Normal" in img_path:
        axs[i].set_title("Normal", fontsize=12)
    else:
        axs[i].set_title(f"Image {i+1}", fontsize=12)

# Hide any extra empty subplots
for j in range(i+1, len(axs)):
    axs[j].axis('off')

# Display all images
plt.tight_layout()
plt.show()
write_to_file(results_file, "Prediction on test images displayed.")

##################################################################################################
# Final message
write_to_file(results_file, "All results have been saved and displayed.")
##################################################################################################
# Zip the runs folder
!zip -r runs.zip runs/

# Download the zip file
from google.colab import files
files.download('runs.zip')