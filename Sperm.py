import math
import numpy as np
from scipy.spatial.distance import cdist,pdist, squareform
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
from skimage.util import invert
from skimage.graph import route_through_array
from scipy.spatial import distance
import copy
from skimage import io, color, measure, morphology, filters
import sys
from scipy.spatial import ConvexHull
from tqdm import tqdm
from skimage.measure import label, regionprops,LineModelND, ransac
from itertools import combinations
from skimage.color import gray2rgb
from skimage.draw import circle_perimeter,polygon
from scipy.ndimage import convolve, distance_transform_edt
from skimage import io, color, filters, morphology
import json
import torch
from skimage import color, filters, morphology


sys.path.append("segment-anything-main")

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


I = "3392"

def set_I(new_value):
    global I
    I = new_value

Sperm_tail=[]







def resize_image(input_path, output_path, width, height):
    # Check if the input file exists
    if os.path.exists(input_path):
        # Open the image
        img = Image.open(input_path)

        # Resize the image using LANCZOS resampling for high quality
        img_resized = img.resize((width, height), Image.Resampling.LANCZOS)

        # Save the resized image
        img_resized.save(output_path)
        #print(f"Resized and saved as {output_path}")
    else:
        print(f"File {input_path} does not exist.")

# Set your parameters here
input_path = f"data/Preprocessing_images/{I}.jpg"  
output_path = "data/original_images/new.jpg" 
width = 1440         
height = 1080       

# Call the resizing function
resize_image(input_path, output_path, width, height)






target_brightness = 220

# Function to adjust the brightness of an image
def adjust_brightness(image_path, target_brightness):
    with Image.open(image_path) as img:
        # Convert the image to HSV and split into channels
        hsv_image = img.convert('HSV')
        h, s, v = hsv_image.split()
        v_array = np.array(v, dtype=np.float64)
        
        # Calculate the current average brightness
        current_brightness = np.mean(v_array)
        
        # Avoid division by zero and extreme changes
        if current_brightness == 0 or current_brightness == target_brightness:
            return img
        
        # Adjust brightness
        v_array *= target_brightness / current_brightness
        v_array[v_array > 255] = 255
        v_array = v_array.astype(np.uint8)
        
        # Merge back the channels and convert to RGB
        adjusted_hsv_image = Image.merge('HSV', (h, s, Image.fromarray(v_array)))
        return adjusted_hsv_image.convert('RGB')


image_path = 'data/original_images/new.jpg' 

adjusted_image = adjust_brightness(image_path, target_brightness)


new_path='data/original_images/new1.jpg'
adjusted_image.save(new_path) 
#print("Brightness adjustment completed for the image.")






# Target RGB values
target_r = 216.5
target_g = 212.5
target_b = 219.5

# Function to adjust RGB channels
def adjust_rgb_channels(image_path, target_r, target_g, target_b):
    with Image.open(image_path) as img:
        # Convert the image to RGB and split into channels
        r, g, b = img.split()
        
        # Calculate the current average values
        r_avg, g_avg, b_avg = map(np.mean, (r, g, b))
        
        # Calculate scaling factors avoiding division by zero
        r_scale = target_r / r_avg if r_avg > 0 else 0
        g_scale = target_g / g_avg if g_avg > 0 else 0
        b_scale = target_b / b_avg if b_avg > 0 else 0
        
        # Scale each channel by its respective factor
        r = (np.array(r) * r_scale).clip(0, 255).astype(np.uint8)
        g = (np.array(g) * g_scale).clip(0, 255).astype(np.uint8)
        b = (np.array(b) * b_scale).clip(0, 255).astype(np.uint8)
        
        # Merge the channels back and return the adjusted image
        adjusted_image = Image.merge('RGB', (Image.fromarray(r), Image.fromarray(g), Image.fromarray(b)))
        return adjusted_image


image_path = 'data/original_images/new1.jpg' 

# Adjust the RGB channels of the single image
adjusted_image = adjust_rgb_channels(image_path, target_r, target_g, target_b)
new_path = 'data/original_images/new2.jpg'
# Save the adjusted image
# You can choose to save it with the same name (overwriting) or with a new name
adjusted_image.save(new_path) 
#print("RGB channel adjustment completed for the image.")







def dehaze_image(img, clip_limit=150, tile_size=(2, 2)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)


def sharpen_image(img, factor=1.235):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel * factor)


def adjust_contrast(img, factor=1.0):
    mean = np.mean(img)
    return np.clip((1 + factor) * (img - mean) + mean, 0, 255).astype(np.uint8)


def process_image_skimage(image_opencv):
    image_rgb = cv2.cvtColor(image_opencv, cv2.COLOR_BGR2RGB)
    gray_sk = color.rgb2gray(image_rgb)
    threshold_value = filters.threshold_otsu(gray_sk)
    binary = gray_sk < threshold_value
    cleaned = morphology.remove_small_objects(binary, min_size=2000)
    mask = np.where(cleaned == 0, 1, 0).astype(bool)
    image_rgb[mask] = [255, 255, 255]
    whitening_mask_sk = np.all(image_rgb > [200, 200, 200], axis=-1)
    image_rgb[whitening_mask_sk] = [255, 255, 255]
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def process_single_image(image_path,news_path):
    image_opencv = cv2.imread(image_path)
    if image_opencv is None:
        print("can't load image")
        return

    img_original_rgb = cv2.cvtColor(image_opencv, cv2.COLOR_BGR2RGB)
    img_new_adjustment = img_original_rgb.copy()
    img_new_adjustment = adjust_contrast(img_new_adjustment,1.0)
    img_new_adjustment = sharpen_image(img_new_adjustment,1.235)
    img_new_adjustment = dehaze_image(img_new_adjustment)
    img_noise_reduced = cv2.fastNlMeansDenoisingColored(img_new_adjustment, None, 20, 20, 7, 21)
    img_final = process_image_skimage(img_noise_reduced)

    
    cv2.imwrite(news_path, img_final)  



image_path = 'data/original_images/new2.jpg'  
news_path = f'data/Preprocessing_images/new2.jpg'

process_single_image(image_path,news_path)

















def resize_image(input_path, output_path, width, height):
    # Check if the input file exists
    if os.path.exists(input_path):
        # Open the image
        img = Image.open(input_path)

        # Resize the image using LANCZOS resampling for high quality
        img_resized = img.resize((width, height), Image.Resampling.LANCZOS)

        # Save the resized image
        img_resized.save(output_path)
        #print(f"Resized and saved as {output_path}")
    else:
        print(f"File {input_path} does not exist.")

# Set your parameters here
input_path = f"data/Preprocessing_images/new2.jpg"  # Path to the input image
output_path = f"data/Preprocessing_images/new2.jpg" # Path to save the resized image
width = 720         # New width for the image
height = 540        # New height for the image

# Call the resizing function
resize_image(input_path, output_path, width, height)
























image = cv2.imread(f'data/Preprocessing_images/{I}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)















sam_checkpoint = "segment-anything-main/sam_vit_h_4b8939.pth"
model_type = "default"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)







mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
masks = np.array([mask for mask in masks if mask['area'] <= 100000])
#masks = np.delete(masks, 0)
np.save('masks2.npy', masks)







# Load the numpy array with allow_pickle enabled
masks = np.load('masks2.npy', allow_pickle=True)

# Load the image from the uploaded file
image_path = f'data/original_images/{I}.jpg'
image = cv2.imread(image_path)

# Convert the image from BGR to RGB and then to HSV
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

# Define the HSV range for purple and green
purple_hsv_min = np.array([100, 30, 20])
purple_hsv_max = np.array([180, 255, 255])
green_hsv_min = np.array([60, 0, 160])
green_hsv_max = np.array([130, 50, 230])

# Extract segmentation masks from each dictionary in the array
segmentation_masks = [mask['segmentation'] for mask in masks if 'segmentation' in mask]

# Placeholder for indices that meet the new criteria with at least 800 purple pixels and 400 green pixels
valid_mask_indices_updated_800_400 = []

# Placeholder for the valid masks
valid_masks = []

# Loop through each segmentation mask with the updated pixel criteria
for index, segmentation_mask in enumerate(segmentation_masks):
    # Apply the segmentation mask to the HSV image
    
    
    #print(segmentation_mask.dtype)
    #print(image_hsv.shape, segmentation_mask.shape)

    segmented_image_hsv = cv2.bitwise_and(image_hsv, image_hsv, mask=segmentation_mask.astype(np.uint8))

    # Create masks for purple and green colors within the segmented area
    mask_purple = cv2.inRange(segmented_image_hsv, purple_hsv_min, purple_hsv_max)
    mask_green = cv2.inRange(segmented_image_hsv, green_hsv_min, green_hsv_max)

    # Find the number of purple and green pixels within the segmented area
    num_purple_pixels = np.count_nonzero(mask_purple)
    num_green_pixels = np.count_nonzero(mask_green)

    # Check if the segmented area contains at least 800 purple pixels and 600 green pixels
    if num_purple_pixels >= 800 and num_green_pixels >= 600:
        valid_mask_indices_updated_800_400.append(index)
        valid_masks.append(segmentation_mask) # Store the valid mask

# Save the valid masks to a new .npy file
np.save('valid_masks.npy', np.array(valid_masks))










image_path = f'data/Preprocessing_images/{I}.jpg'
image = Image.open(image_path)


masks_path = 'valid_masks.npy'
masks = np.load(masks_path)


image_info = (image.size, image.mode)
masks_info = masks.shape








image_np = np.array(image)


for mask in masks:
  
    image_np[mask == True] = [255, 255, 255]


modified_image = Image.fromarray(image_np)
processed_img_path = "modified_image.jpg"
modified_image.save("modified_image.jpg")










image_sk = io.imread("modified_image.jpg")

gray_sk = color.rgb2gray(image_sk)


threshold_value = filters.threshold_otsu(gray_sk)
binary = gray_sk < threshold_value


cleaned = morphology.remove_small_objects(binary, min_size=50)


mask = np.where(cleaned == 0, 1, 0).astype(bool)


image_sk[mask] = [255, 255, 255]


whitening_mask_sk = np.all(image_sk > [200, 200, 200], axis=-1)
image_sk[whitening_mask_sk] = [255, 255, 255]





processed_img_path = "modified_image_dilated.jpg"
cv2.imwrite(processed_img_path, image_sk)




T=True
XX=0
while T!=False:
    

    image = cv2.imread('modified_image_dilated.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    masks = np.array([mask for mask in masks if mask['area'] <= 100000])
    np.save('masks2.npy', masks)



    # Load the numpy array with allow_pickle enabled
    masks = np.load('masks2.npy', allow_pickle=True)

    # Load the image from the uploaded file
    image_path = f'data/original_images/{I}.jpg'
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB and then to HSV
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Define the HSV range for purple and green
    purple_hsv_min = np.array([100, 30, 20])
    purple_hsv_max = np.array([180, 255, 255])
    green_hsv_min = np.array([60, 0, 160])
    green_hsv_max = np.array([130, 50, 230])

    # Extract segmentation masks from each dictionary in the array
    segmentation_masks = [mask['segmentation'] for mask in masks if 'segmentation' in mask]

    # Placeholder for indices that meet the new criteria with at least 800 purple pixels and 400 green pixels
    valid_mask_indices_updated_800_400 = []


   
    # Loop through each segmentation mask with the updated pixel criteria
    for index, segmentation_mask in enumerate(segmentation_masks):
        # Apply the segmentation mask to the HSV image
        segmented_image_hsv = cv2.bitwise_and(image_hsv, image_hsv, mask=segmentation_mask.astype(np.uint8))

        # Create masks for purple and green colors within the segmented area
        mask_purple = cv2.inRange(segmented_image_hsv, purple_hsv_min, purple_hsv_max)
        mask_green = cv2.inRange(segmented_image_hsv, green_hsv_min, green_hsv_max)

        # Find the number of purple and green pixels within the segmented area
        num_purple_pixels = np.count_nonzero(mask_purple)
        num_green_pixels = np.count_nonzero(mask_green)

        # Check if the segmented area contains at least 800 purple pixels and 600 green pixels
        if num_purple_pixels >= 800 and num_green_pixels >= 600:
            valid_mask_indices_updated_800_400.append(index)
            valid_masks.append(segmentation_mask) # Store the valid mask

    # Save the valid masks to a new .npy file
    np.save('valid_masks.npy', np.array(valid_masks))
    # Print out the valid indices
    #print(valid_mask_indices_updated_800_400)
    if XX==len(valid_masks):
        T=False
        break
    XX=len(valid_masks)



    
    image_path = 'modified_image_dilated.jpg'
    image = Image.open(image_path)

    
    masks_path = 'valid_masks.npy'
    masks = np.load(masks_path)

    
    image_np = np.array(image)

        
    for mask in masks:
       
        image_np[mask == True] = [255, 255, 255]

    
    modified_image = Image.fromarray(image_np)   
    processed_img_path = "modified_image.jpg"
       
   
    modified_image.save("modified_image.jpg")
   
   
    #modified_image




    image_sk = io.imread("modified_image.jpg")


    gray_sk = color.rgb2gray(image_sk)


    threshold_value = filters.threshold_otsu(gray_sk)
    binary = gray_sk < threshold_value


    cleaned = morphology.remove_small_objects(binary, min_size=50)


    mask = np.where(cleaned == 0, 1, 0).astype(bool)


    image_sk[mask] = [255, 255, 255]


    whitening_mask_sk = np.all(image_sk > [200, 200, 200], axis=-1)
    image_sk[whitening_mask_sk] = [255, 255, 255]





    processed_img_path = "modified_image_dilated.jpg"
    cv2.imwrite(processed_img_path, image_sk)
    
    
image = cv2.imread('modified_image_dilated.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
masks = np.array([mask for mask in masks if mask['area'] <= 100000])
#masks = np.delete(masks, 0)
np.save('masks2.npy', masks)


masks1 = [np.array(seg['segmentation'], dtype=np.uint8) for seg in masks]

masks1 = np.stack(masks1)


image_path = f'data/original_images/{I}.jpg' 
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_color = np.array([100, 20 , 20])  
upper_color = np.array([180, 255, 255]) 
mask = cv2.inRange(hsv, lower_color, upper_color)

filtered_masks = []
masks222=masks1

for current_mask in masks1:
    # Find the bounding box of the current mask
    rows, cols = np.where(current_mask == 1)
    
    # Check if rows and cols are not empty
    if len(rows) == 0 or len(cols) == 0:
        continue  # skip the current iteration if either rows or cols is empty
    
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    

    
    region_mask = mask[min_row:max_row+1, min_col:max_col+1]
    region_current_mask = current_mask[min_row:max_row+1, min_col:max_col+1]
    
    overlap = np.sum((region_mask == 255) & (region_current_mask == 1))
    total_pixels = np.sum(region_current_mask == 1)
    overlap_percentage = (overlap / total_pixels) * 100
    

    if overlap_percentage > 40:
        filtered_masks.append(current_mask)
        


filtered_masks = np.array(filtered_masks)
np.save('filtered_masks.npy', filtered_masks)



from scipy.spatial.distance import pdist, squareform


# Load the numpy file
file_path = 'filtered_masks.npy'
masks = np.load(file_path)

# Function to find the furthest distance between points in each mask layer
def find_furthest_distance_in_each_layer(masks):
    distances = []
    for layer in masks:
        # Find coordinates of non-background (non-zero) points
        points = np.column_stack(np.where(layer > 0))
        if len(points) > 1:
            # Compute pairwise distances between all points
            pairwise_distances = squareform(pdist(points, 'euclidean'))
            # Find the maximum distance
            max_distance = pairwise_distances.max()
        else:
            # If there is only one point or none, the distance is 0
            max_distance = 0
        distances.append(max_distance)
    return distances

# Calculate the furthest distances for each layer
furthest_distances = find_furthest_distance_in_each_layer(masks)

# Convert the furthest distances list to a NumPy array for comparison
furthest_distances_array = np.array(furthest_distances)

# Filter out layers where the furthest distance is less than 75
masks_filtered = masks[furthest_distances_array <= 75]

# Save the filtered masks back to the original file
np.save(file_path, masks_filtered)






file_path = 'filtered_masks.npy'  
masks = np.load(file_path)





image_path = 'modified_image.jpg'
image = Image.open(image_path)


masks_path = 'filtered_masks.npy'
masks = np.load(masks_path)


image_info = (image.size, image.mode)
masks_info = masks.shape



image_np = np.array(image)


for mask in masks:
    
    image_np[mask == 1] = [255, 255, 255]


modified_image = Image.fromarray(image_np)
processed_img_path = "modified_image.jpg"
modified_image.save("modified_image.jpg")







image_sk = io.imread("modified_image.jpg")


gray_sk = color.rgb2gray(image_sk)


threshold_value = filters.threshold_otsu(gray_sk)
binary = gray_sk < threshold_value


cleaned = morphology.remove_small_objects(binary, min_size=50)


mask = np.where(cleaned == 0, 1, 0).astype(bool)


image_sk[mask] = [255, 255, 255]


whitening_mask_sk = np.all(image_sk > [200, 200, 200], axis=-1)
image_sk[whitening_mask_sk] = [255, 255, 255]





#



processed_img_path = "modified_image_dilated.jpg"
cv2.imwrite(processed_img_path, image_sk)

image22 = cv2.imread(r"modified_image_dilated.jpg")
image22 = cv2.cvtColor(image22, cv2.COLOR_BGR2RGB)


mask_generator22 = SamAutomaticMaskGenerator(sam)
masks22 = mask_generator22.generate(image22)
masks22 = np.array([mask for mask in masks22 if mask['area'] <= 100000])
#masks22 = np.delete(masks22, 0)
np.save("masks22.npy",masks22)




masks1 = [np.array(seg['segmentation'], dtype=np.uint8) for seg in masks22]

masks1 = np.stack(masks1)
image_path = f'data/original_images/{I}.jpg'  
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_color = np.array([100, 33, 20])  
upper_color = np.array([180, 255, 255]) 
mask = cv2.inRange(hsv, lower_color, upper_color)
filtered_masks = list(np.load("filtered_masks.npy"))

masks222 = masks1
new = []
for current_mask in masks1:
    # Find the bounding box of the current mask
    rows, cols = np.where(current_mask == 1)

    # Check if rows and cols are not empty
    if len(rows) == 0 or len(cols) == 0:
        continue  # skip the current iteration if either rows or cols is empty

    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # ... (rest of your code)

    region_mask = mask[min_row:max_row + 1, min_col:max_col + 1]
    region_current_mask = current_mask[min_row:max_row + 1, min_col:max_col + 1]

    overlap = np.sum((region_mask == 255) & (region_current_mask == 1))
    total_pixels = np.sum(region_current_mask == 1)
    overlap_percentage = (overlap / total_pixels) * 100

   
    if overlap_percentage > 50:
        new.append(current_mask)

new = np.array(new)

# Calculate the furthest distances for each layer
furthest_distances = find_furthest_distance_in_each_layer(new)

# Convert the furthest distances list to a NumPy array for comparison
furthest_distances_array = np.array(furthest_distances)

# Filter out layers where the furthest distance is less than 75
masks_filtered = new[furthest_distances_array <= 80]

masks_filtered=list(masks_filtered)
masksfinal=masks_filtered+filtered_masks
masksfinal=np.array(masksfinal)
# Save the filtered masks back to the original file
np.save('filtered_masks.npy', masksfinal)

image_path = "modified_image_dilated.jpg"
image = Image.open(image_path)

image_np = np.array(image)


for mask in masks_filtered:

    image_np[mask == 1] = [255, 255, 255]


modified_image = Image.fromarray(image_np)
processed_img_path = "modified_image_dilated.jpg"
modified_image.save("modified_image_dilated.jpg")



image_sk = io.imread("modified_image_dilated.jpg")


gray_sk = color.rgb2gray(image_sk)


threshold_value = filters.threshold_otsu(gray_sk)
binary = gray_sk < threshold_value


cleaned = morphology.remove_small_objects(binary, min_size=50)


mask = np.where(cleaned == 0, 1, 0).astype(bool)


image_sk[mask] = [255, 255, 255]


whitening_mask_sk = np.all(image_sk > [200, 200, 200], axis=-1)
image_sk[whitening_mask_sk] = [255, 255, 255]





processed_img_path = "modified_image_dilated.jpg"
cv2.imwrite(processed_img_path, image_sk)






image22 = cv2.imread(r"modified_image_dilated.jpg")
image22 = cv2.cvtColor(image22, cv2.COLOR_BGR2RGB)


mask_generator22 = SamAutomaticMaskGenerator(sam)
masks22 = mask_generator22.generate(image22)
masks22 = np.array([mask for mask in masks22 if mask['area'] <= 100000])
#masks22 = np.delete(masks22, 0)
np.save("masks22.npy",masks22)


def calculate_furthest_distance(mask):
    # Find the coordinates of all pixels that belong to the mask
    y_coords, x_coords = np.nonzero(mask)
    points = np.column_stack((x_coords, y_coords))

    # Compute the convex hull of the points
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    # Calculate the pairwise distance between hull points
    distances = cdist(hull_points, hull_points, metric='euclidean')

    # Find the maximum distance
    furthest_distance = np.max(distances)
    return furthest_distance


# Load the mask file with allow_pickle set to True
masks = np.load('masks22.npy', allow_pickle=True)

# Load the image to get its dimensions
image_path = r"modified_image_dilated.jpg"
image = Image.open(image_path)

# Extract the segmentation data from each dictionary
binary_masks = [mask_dict['segmentation'] for mask_dict in masks]

# List to store indices of masks with ratio > 0.2
valid_mask_indices = []

# Iterate over each binary mask and perform calculations
for index, binary_mask in enumerate(binary_masks):
    # Calculate the furthest distance
    furthest_distance = calculate_furthest_distance(binary_mask)

    # Calculate the area of the mask
    area = np.sum(binary_mask)

    # Calculate the ratio
    ratio = area / (furthest_distance ** 2)

    # Check if the ratio is greater than 0.2 and record the index if it is
    if ratio > 0.4:
        valid_mask_indices.append(index)



masks22_path = 'masks22.npy'
masks22 = np.load(masks22_path, allow_pickle=True)


image_path = r"modified_image_dilated.jpg"
image = Image.open(image_path)


image_np = np.array(image)



for idx in valid_mask_indices:
    
    mask = masks22[idx]['segmentation']
   
    image_np[mask] = [255, 255, 255]


modified_image = Image.fromarray(image_np)


save_path = 'new_modified_image.jpg'
modified_image.save(save_path)



image_sk = io.imread('new_modified_image.jpg')


gray_sk = color.rgb2gray(image_sk)


threshold_value = filters.threshold_otsu(gray_sk)
binary = gray_sk < threshold_value


cleaned = morphology.remove_small_objects(binary, min_size=50)


mask = np.where(cleaned == 0, 1, 0).astype(bool)


image_sk[mask] = [255, 255, 255]


whitening_mask_sk = np.all(image_sk > [200, 200, 200], axis=-1)
image_sk[whitening_mask_sk] = [255, 255, 255]





processed_img_path = 'processed_image.jpg'
cv2.imwrite(processed_img_path, image_sk)


masks22 = np.delete(masks22, valid_mask_indices)


np.save("masks22.npy",masks22)












# Load the numpy file
mask_file = 'masks22.npy'  # Replace with your file path
masks = np.load(mask_file, allow_pickle=True)

# Function to remove connected components smaller than a specified size
def remove_small_components(mask, min_size):
    labeled_array = label(mask)
    sizes = np.bincount(labeled_array.ravel())

    # Identify labels of small components
    small_components = np.where(sizes < min_size)[0]

    # Create a mask to remove small components
    remove_mask = np.in1d(labeled_array, small_components).reshape(labeled_array.shape)

    # Set pixels of small components to False
    mask[remove_mask] = False
    return mask

# Analyze the masks to find the number of connected components in each layer
connected_components = [np.max(label(mask['segmentation'])) if 'segmentation' in mask else 0 for mask in masks]


min_pixel_size = 15

for i, num_components in enumerate(connected_components):
    if num_components > 1:
        masks[i]['segmentation'] = remove_small_components(masks[i]['segmentation'], min_pixel_size)

# Save the modified masks back to the file
np.save(mask_file, masks)
masks_path = "masks22.npy"
masks = np.load(masks_path, allow_pickle=True)




def extract_and_skeletonize(data):
    skeletons = []
    for item in data:
        if 'segmentation' in item:
            segmentation_mask = item['segmentation']
            segmentation_mask_contiguous = np.ascontiguousarray(segmentation_mask, dtype=bool)
            skeleton = skeletonize(segmentation_mask_contiguous)
            skeletons.append(skeleton)
    return np.array(skeletons)

def find_branch_points(skeleton):
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    convolved = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    return np.logical_and(skeleton, convolved >= 13)

def find_endpoints(skeleton):
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    convolved = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    return np.logical_and(skeleton, np.isin(convolved, [11, 21]))

def is_fully_connected(skeleton):
    labeled_skeleton, num_components = label(skeleton, return_num=True)
    return num_components == 1

def is_adjacent_to_branch(endpoint, branch_points):
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            neighbor = (endpoint[0] + dx, endpoint[1] + dy)
            if neighbor in branch_points:
                return True
    return False


def process_skeletons(data):
    modified_skeletons = []
    for item in data:
        if 'segmentation' in item:
            # Extract and skeletonize the current layer
            segmentation_mask = item['segmentation']
            segmentation_mask_contiguous = np.ascontiguousarray(segmentation_mask, dtype=bool)
            skeleton = skeletonize(segmentation_mask_contiguous)

            # Find and remove branch points
            branch_points = find_branch_points(skeleton)
            branch_point_coordinates = np.argwhere(branch_points)
            skeleton_without_branches = skeleton.copy()
            for coord in branch_point_coordinates:
                skeleton_without_branches[tuple(coord)] = 0

            if 0>1:
                # If more than 14 branch points, use all as min_branch_points_needed
                pass
            else:
                labeled_skeleton, _ = label(skeleton_without_branches, return_num=True)
                small_components = [region for region in regionprops(labeled_skeleton) if region.area < 25]
                for component in small_components:
                    component_mask = np.zeros_like(skeleton_without_branches, dtype=bool)
                    for coord in component.coords:
                        component_mask[tuple(coord)] = True

                    endpoints = find_endpoints(component_mask)
                    endpoints_coordinates = np.argwhere(endpoints)

                    if len(endpoints_coordinates) != 2:
                        continue

                    branch_points_list = [tuple(coord) for coord in branch_point_coordinates]
                    if not all(
                            is_adjacent_to_branch(endpoint, branch_points_list) for endpoint in endpoints_coordinates):
                        for coord in component.coords:
                            skeleton_without_branches[tuple(coord)] = 0

                # Continue with existing process to find minimum branch points needed for full connectivity
                min_branch_points_needed = None
                min_num_branch_points = len(branch_point_coordinates) + 1
                if len(branch_point_coordinates) > 14:
                    # If more than 14 branch points, use all as min_branch_points_needed
                    min_branch_points_needed = branch_point_coordinates
                else:
                    for num_points_to_add in range(len(branch_point_coordinates) + 1):
                        for branch_subset in combinations(branch_point_coordinates, num_points_to_add):
                            temp_skeleton = skeleton_without_branches.copy()
                            for point in branch_subset:
                                temp_skeleton[tuple(point)] = 1

                            if is_fully_connected(temp_skeleton):
                                if num_points_to_add < min_num_branch_points:
                                    min_num_branch_points = num_points_to_add
                                    min_branch_points_needed = branch_subset
                                break
                        if min_branch_points_needed is not None:
                             break

            # Add the minimum set of branch points back to the skeleton
            if min_branch_points_needed is not None:
                for point in min_branch_points_needed:
                    skeleton_without_branches[tuple(point)] = 1

            # Add the processed skeleton to the list
            modified_skeletons.append(skeleton_without_branches)

    return np.array(modified_skeletons)



# Process all layers of the segmentation data
skeletonized_segmentations = process_skeletons(masks)

output_file = 'skeletonized_segmentations.npy'  # Replace with your desired save path
np.save(output_file, skeletonized_segmentations)





# Function to analyze the connectivity of the skeletons in all masks
def analyze_masks_connectivity(masks_file):
    masks = masks_file
    disconnected_masks_indices = []
    connectcomponents = []

    for index, mask_dict in enumerate(masks):
        mask = mask_dict['segmentation']
        labeled_mask, num_components = label(mask, return_num=True, connectivity=2)
        connectcomponents.append(num_components)

        if num_components > 1:
            # Extract coordinates of pixels in each component and calculate areas
            components_areas = [np.sum(labeled_mask == i) for i in range(1, num_components + 1)]
            coords = [np.column_stack(np.where(labeled_mask == i)) for i in range(1, num_components + 1)]
            
            # Initialize a large initial distance
            min_distance = np.inf
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    distance = np.min(cdist(coords[i], coords[j]))
                    if distance < min_distance:
                        min_distance = distance
                        smallest_area = min(components_areas[i], components_areas[j])

            # Check conditions for adding to disconnected_masks_indices
            if min_distance > 25 or smallest_area < 50:
                disconnected_masks_indices.append(index)

    return disconnected_masks_indices, connectcomponents

# Analyze the connectivity of the skeletons in all masks
disconnected_masks_indices, connectcomponents = analyze_masks_connectivity(masks)




# Function to analyze the connectivity of the skeletons in all masks
def analyze_skeleton_connectivity(masks):
    disconnected_masks_indices = []

    for index, mask_dict in enumerate(masks):
        # Perform skeletonization
        mask = mask_dict['segmentation']
        skeleton = skeletonize(np.ascontiguousarray(mask, dtype=bool))

        # Label connected components in the skeleton
        labeled_skeleton, num_components = label(skeleton, return_num=True)

        # If there are more than one connected components, the skeleton is disconnected
        if num_components > 1:
            disconnected_masks_indices.append(index)

    return disconnected_masks_indices




# Function to detect endpoints in a skeleton
def detect_endpoints(skeleton):
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    neighbors = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
    endpoints = ((neighbors - 10) == 1) & skeleton
    return endpoints


def detect_branch_points(skeleton):
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    neighbors = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
    branch_points = ((neighbors - 10) >= 3) & skeleton
    return branch_points


def endpoints_to_nearest_branch_distance(skeleton, endpoints, branch_points):
    endpoints_coords = np.column_stack(np.where(endpoints))

    # print(f"Mask : {len(endpoints_coords)} coords")
    branch_points_coords = np.column_stack(np.where(branch_points))

    tree = cKDTree(branch_points_coords)
    distances, _ = tree.query(endpoints_coords, k=1)
    return distances


def identify_valid_endpoints_masks(masks, min_distance=25):
    valid_endpoints_masks = []

    for index, mask_dict in enumerate(masks):
        # Perform skeletonization
        mask = mask_dict
        skeleton = mask

        # Detect endpoints and branch points in the skeleton
        endpoints = detect_endpoints(skeleton)
        # print(endpoints)
        branch_points = detect_branch_points(skeleton)

        # If no branch points are detected, consider all endpoints as valid
        if not np.any(branch_points):
            valid_endpoints_count = np.count_nonzero(endpoints)
        else:
            # Calculate the distances from each endpoint to the nearest branch point
            distances_to_nearest_branch = endpoints_to_nearest_branch_distance(skeleton, endpoints, branch_points)

            # Count valid endpoints based on the distance criteria
            valid_endpoints_count = np.sum(distances_to_nearest_branch > min_distance) + 0.5 * np.sum(
                distances_to_nearest_branch <= min_distance)

        # Print the number of endpoints for this mask
        #print(f"Mask {index + 1}: {valid_endpoints_count} endpoints")

        # If there are more than two valid endpoints, add the mask index to the list
        if math.floor(valid_endpoints_count) > 2*connectcomponents[index]:
            valid_endpoints_masks.append(index)

    return valid_endpoints_masks


# Load the masks from the .npy file
# masks = np.load('masks22.npy', allow_pickle=True)

# Identify masks with more than two valid endpoints based on the distance criteria
valid_endpoints_masks_indices = identify_valid_endpoints_masks(skeletonized_segmentations)






def find_point_on_skeleton(skeleton, start, distance):
    current_point = start
    visited = set()
    height, width = skeleton.shape

    for _ in range(abs(distance)):
        visited.add(current_point)
        x, y = current_point
        neighbors = [(x2, y2) for x2 in range(x - 1, x + 2)
                     for y2 in range(y - 1, y + 2)
                     if 0 <= x2 < width and 0 <= y2 < height and (x2, y2) != current_point and skeleton[y2, x2] and (
                     x2, y2) not in visited]

        if not neighbors:
            break  

        # Choose the next point
        current_point = neighbors[0] if distance > 0 else neighbors[-1]

    return current_point


def calculate_angle(point_a, point_b, point_c):
    """
    Calculate the angle formed by three points.
    """
    a = np.array(point_a)
    b = np.array(point_b)
    c = np.array(point_c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


def find_endpoints(skeleton):
    """
    Find endpoints in the skeletonized image.

    Args:
    - skeleton (numpy.ndarray): The skeletonized image.

    Returns:
    - numpy.ndarray: Endpoints in the skeleton.
    """
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    filtered = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    return np.isin(filtered, [11, 21])


def is_near_endpoint(point, endpoints, max_distance=30):
    """
    Check if a point is within a specified distance from the nearest endpoint.

    Args:
    - point (tuple): The point to check.
    - endpoints (numpy.ndarray): Endpoints in the skeleton.
    - max_distance (int): The maximum distance to an endpoint.

    Returns:
    - bool: True if the point is within max_distance from an endpoint, False otherwise.
    """
    distance_map = distance_transform_edt(~endpoints)
    y, x = point
    return distance_map[y, x] <= max_distance


def find_sharp_angle_points(skeleton, distance=10, angle_threshold=130, max_endpoint_distance=30):
    """
    Find sharp angle points on the skeleton that are not within a specified distance from any endpoint.
    """
    sharp_points = []
    endpoints = find_endpoints(skeleton)
    for y in range(skeleton.shape[0]):
        for x in range(skeleton.shape[1]):
            if skeleton[y, x] and not is_near_endpoint((y, x), endpoints, max_endpoint_distance):
                point_a = find_point_on_skeleton(skeleton, (x, y), distance)
                point_b = find_point_on_skeleton(skeleton, (x, y), -distance)
                if point_a != (x, y) and point_b != (x, y):
                    angle = calculate_angle(point_a, (x, y), point_b)
                    if angle < angle_threshold:
                        sharp_points.append((y, x))
    return sharp_points


def mark_points_on_skeleton(skeleton, points):
    """
    Mark given points on the skeleton with red color.

    Args:
    - skeleton (numpy.ndarray): The skeletonized image.
    - points (list of tuples): Points to be marked.

    Returns:
    - numpy.ndarray: Skeleton image with points marked in red.
    """
    marked_skeleton = gray2rgb(skeleton.astype(np.uint8) * 255)  # Convert to RGB for marking
    for point in points:
        y, x = point
        rr, cc = circle_perimeter(y, x, 1)
        marked_skeleton[rr, cc] = [255, 0, 0]  # Red color for marked points
    return marked_skeleton


# Load the skeletonized segmentations
skeletonized_segmentations_path = 'skeletonized_segmentations.npy'
skeletonized_segmentations = np.load(skeletonized_segmentations_path, allow_pickle=True)
masks_with_high_curvature_points = []  # List to hold indices of masks with valid high curvature points
# Process each layer and store sharp points
sharp_points_all_layers = []

for index, skeleton_layer in enumerate(skeletonized_segmentations):
    sharp_points = find_sharp_angle_points(skeleton_layer)
    if sharp_points:  # Check if the list of sharp points is not empty
        masks_with_high_curvature_points.append(index)
    sharp_points_all_layers.append({'layer': index, 'sharp_points': sharp_points})


final_set = set(masks_with_high_curvature_points) | set(valid_endpoints_masks_indices) | set(disconnected_masks_indices)
final = list(final_set)
image = io.imread('processed_image.jpg')  
masks = np.load('masks22.npy', allow_pickle=True)  


new_image = copy.deepcopy(image)




masks_length = len(masks)  
new_list = list(range(masks_length))  

final2 = list(set(new_list) - set(final))



for index in final2:
    mask = masks[index]['segmentation']
    new_image[mask == 1] = 255  

    
    
    
    
    

    
    
    
    
    
    
    
    
    



from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist

def find_closest_points(coords1, coords2):
    """
    Find the closest points between two sets of coordinates.
    """
    distances = cdist(coords1, coords2)
    idx_min = np.argmin(distances)
    min_idx_1, min_idx_2 = np.unravel_index(idx_min, distances.shape)
    return coords1[min_idx_1], coords2[min_idx_2]

def draw_filled_rectangle(mask, point_1, point_2, ratio):
    """
    Draw a filled rectangle on the mask with given points and ratio.
    """
    center_x = (point_1[1] + point_2[1]) / 2
    center_y = (point_1[0] + point_2[0]) / 2
    distance = np.linalg.norm(point_1 - point_2)
    dx = ratio / 2
    dy = distance / 2

    row_min = int(max(0, center_y - dy))
    row_max = int(min(mask.shape[0], center_y + dy))
    col_min = int(max(0, center_x - dx))
    col_max = int(min(mask.shape[1], center_x + dx))

    mask[row_min:row_max, col_min:col_max] = 1

# Load the masks from the .npy file
masks_file = 'masks22.npy'  # Update this path
masks = np.load(masks_file, allow_pickle=True)



for index in final2:
    if connectcomponents[index] == 2:
        mask = masks[index]['segmentation']
        original_pixels = np.sum(mask)
        skeleton = skeletonize(mask)
        skeleton_pixels = np.sum(skeleton)
        ratio = original_pixels / skeleton_pixels if skeleton_pixels > 0 else 0

        labeled_mask, num_components = label(mask, return_num=True, connectivity=1)
        props = regionprops(labeled_mask)

        if num_components == 2:  # Ensure there are exactly two components
            coords_1 = props[0].coords
            coords_2 = props[1].coords
            point_1, point_2 = find_closest_points(coords_1, coords_2)

            draw_filled_rectangle(mask, point_1, point_2, ratio)
            masks[index]['segmentation'] = mask  # Update the mask

# Save the modified masks back to the file
np.save(masks_file, masks)
    
    
    
masks_file = 'masks22.npy'  # Update this path
masks = np.load(masks_file, allow_pickle=True)    
    
        
    
    
    
    
    

    
    
    
    
    
    
    
    
    
for index in final2:
    Sperm_tail.append(masks[index])


new_image_path = 'final111.jpg'  
io.imsave(new_image_path, new_image)


image_sk = io.imread('final111.jpg')


gray_sk = color.rgb2gray(image_sk)


threshold_value = filters.threshold_otsu(gray_sk)
binary = gray_sk < threshold_value


cleaned = morphology.remove_small_objects(binary, min_size=50)


mask = np.where(cleaned == 0, 1, 0).astype(bool)

image_sk[mask] = [255, 255, 255]


whitening_mask_sk = np.all(image_sk > [200, 200, 200], axis=-1)
image_sk[whitening_mask_sk] = [255, 255, 255]





processed_img_path = 'modified_image_dilated0.jpg'
cv2.imwrite(processed_img_path, image_sk)




prev_count = -1
current_count = 0
stagnant_rounds = 0  

X = 0  
while True:
    
    image22 = cv2.imread(f"modified_image_dilated{X}.jpg")
    image22 = cv2.cvtColor(image22, cv2.COLOR_BGR2RGB)

  
    mask_generator22 = SamAutomaticMaskGenerator(sam)
    masks22 = mask_generator22.generate(image22)
    masks22 = np.array([mask for mask in masks22 if mask['area'] <= 100000])
    np.save("masks22.npy", masks22)
    # np.save('masks2.npy', masks)
    masks = np.load('masks22.npy', allow_pickle=True)


    if masks.size == 0:
        #print(f"No masks found for image extracted_image{X}.jpg, skipping to next iteration.")
        X=X-1
        break

    # Load the image to get its dimensions
    image_path = f"modified_image_dilated{X}.jpg"
    image = Image.open(image_path)

    # Extract the segmentation data from each dictionary
    binary_masks = [mask_dict['segmentation'] for mask_dict in masks]

    # List to store indices of masks with ratio > 0.2
    valid_mask_indices = []
    # Iterate over each binary mask and perform calculations
    for index, binary_mask in enumerate(binary_masks):
        # Calculate the furthest distance
        furthest_distance = calculate_furthest_distance(binary_mask)

        # Calculate the area of the mask
        area = np.sum(binary_mask)

        # Calculate the ratio
        ratio = area / (furthest_distance ** 2)

        # Check if the ratio is greater than 0.2 and record the index if it is
        if ratio > 0.2:
            valid_mask_indices.append(index)
    masks22_path = 'masks22.npy'
    masks22 = np.load(masks22_path, allow_pickle=True)

    
    image_path = f"modified_image_dilated{X}.jpg"
    image = Image.open(image_path)

    
    image_np = np.array(image)

   
    for idx in valid_mask_indices:
        
        mask = masks22[idx]['segmentation']
        
        image_np[mask] = [255, 255, 255]

    
    modified_image = Image.fromarray(image_np)

    
    save_path = 'new_modified_image.jpg'
    modified_image.save(save_path)
   




    image_sk = io.imread('new_modified_image.jpg')


    gray_sk = color.rgb2gray(image_sk)


    threshold_value = filters.threshold_otsu(gray_sk)
    binary = gray_sk < threshold_value

 
    cleaned = morphology.remove_small_objects(binary, min_size=20)


    mask = np.where(cleaned == 0, 1, 0).astype(bool)


    image_sk[mask] = [255, 255, 255]


    whitening_mask_sk = np.all(image_sk > [200, 200, 200], axis=-1)
    image_sk[whitening_mask_sk] = [255, 255, 255]

    processed_img_path = 'processed_image.jpg'
    cv2.imwrite(processed_img_path, image_sk)

    masks22 = np.delete(masks22, valid_mask_indices)
    np.save("masks22.npy", masks22)


    # Load the masks file
    masks_path = "masks22.npy"
    masks = np.load(masks_path, allow_pickle=True)
    # Check if the array is empty
    if masks.size == 0:
        #print(f"No masks found for image extracted_image{X}.jpg, skipping to next iteration.")
        X=X-1
        break
    # Check the structure and type of the first item




    # Load the numpy file
    mask_file = 'masks22.npy'  # Replace with your file path
    masks = np.load(mask_file, allow_pickle=True)

    # Analyze the masks to find the number of connected components in each layer
    connected_components = [np.max(label(mask['segmentation'])) if 'segmentation' in mask else 0 for mask in masks]


    min_pixel_size = 10

    for i, num_components in enumerate(connected_components):
        if num_components > 1:
            masks[i]['segmentation'] = remove_small_components(masks[i]['segmentation'], min_pixel_size)

    # Save the modified masks back to the file
    np.save(mask_file, masks)
    masks_path = "masks22.npy"
    masks = np.load(masks_path, allow_pickle=True)


    # Analyze the connectivity of the skeletons in all masks

    # Process all layers of the segmentation data
    skeletonized_segmentations = process_skeletons(masks)

    # Visualizing the modified skeletons in the array

    output_file = 'skeletonized_segmentations.npy'  # Replace with your desired save path
    np.save(output_file, skeletonized_segmentations)

    disconnected_masks_indices = []
    disconnected_masks_indices = analyze_skeleton_connectivity(masks)

    # Load the masks from the .npy file

    # Check if the array is empty

    # Your further operations go here
    valid_endpoints_masks_indices = []
    # Identify masks with more than two valid endpoints based on the distance criteria
    valid_endpoints_masks_indices = identify_valid_endpoints_masks(skeletonized_segmentations)

    # Load the masks from the .npy file
    # masks = np.load('masks22.npy', allow_pickle=True)

    # Visualize each mask excluding endpoints
    skeletonized_segmentations_path = 'skeletonized_segmentations.npy'
    skeletonized_segmentations = np.load(skeletonized_segmentations_path, allow_pickle=True)
    masks_with_high_curvature_points = []  # List to hold indices of masks with valid high curvature points
    # Process each layer and store sharp points
    sharp_points_all_layers = []

    for index, skeleton_layer in enumerate(skeletonized_segmentations):
        sharp_points = find_sharp_angle_points(skeleton_layer)
        if sharp_points:  # Check if the list of sharp points is not empty
            masks_with_high_curvature_points.append(index)
        sharp_points_all_layers.append({'layer': index, 'sharp_points': sharp_points})




    final_set = set(masks_with_high_curvature_points) | set(valid_endpoints_masks_indices) | set(disconnected_masks_indices)
    final = list(final_set)


    image = io.imread('processed_image.jpg') 
    masks = np.load('masks22.npy', allow_pickle=True)  


    new_image = copy.deepcopy(image)


    masks_length = len(masks)  
    new_list = list(range(masks_length)) 

    final2 = list(set(new_list) - set(final))




    for index in final2:
        mask = masks[index]['segmentation']
        new_image[mask == 1] = 255  

    for index in final2:
        Sperm_tail.append(masks[index])

    new_image_path = 'final111.jpg'  
    io.imsave(new_image_path, new_image)



    img_path = 'final111.jpg'


    image_sk = io.imread('final111.jpg')


    gray_sk = color.rgb2gray(image_sk)


    threshold_value = filters.threshold_otsu(gray_sk)
    binary = gray_sk < threshold_value

 
    cleaned = morphology.remove_small_objects(binary, min_size=20)


    mask = np.where(cleaned == 0, 1, 0).astype(bool)


    image_sk[mask] = [255, 255, 255]

 
    whitening_mask_sk = np.all(image_sk > [200, 200, 200], axis=-1)
    image_sk[whitening_mask_sk] = [255, 255, 255]


    processed_img_path = f"modified_image_dilated{X + 1}.jpg"
    cv2.imwrite(processed_img_path, image_sk)





    current_count = len(Sperm_tail)


    if current_count == prev_count:
        stagnant_rounds += 1
    else:
        stagnant_rounds = 0


    prev_count = current_count


    if stagnant_rounds >= 2:
        break  
    #print("X:",X)

    #print("stagnant_rounds:",stagnant_rounds)
    #print("len(Sperm_tail):",len(Sperm_tail))
    X += 1
    if X>=6:
        X=X-1
        break


def extract_segments_from_image(masks_path, image_path):

    masks = np.load(masks_path, allow_pickle=True)


    image = Image.open(image_path)
    image_np = np.array(image)


    extracted_images_paths = []
    for i, mask_dict in enumerate(masks):

        segmentation = mask_dict['segmentation']


        new_image = Image.new("RGB", image.size, (255, 255, 255))
        new_image_np = np.array(new_image)

        new_image_np[segmentation == 1] = image_np[segmentation == 1]

        new_image_pil = Image.fromarray(new_image_np)


        new_image_path = f'extracted_image{i}_0.jpg'
        new_image_pil.save(new_image_path)

        extracted_images_paths.append(new_image_path)

    return extracted_images_paths


masks_path = 'masks22.npy'
# image_path = f'modified_image_dilated{X}.jpg'
image_path = f'data/original_images/{I}.jpg'
extracted_images = extract_segments_from_image(masks_path, image_path)





# Load the image from the saved path after previous processing
image_path = f"modified_image_dilated{X+1}.jpg"  # Replace with your image path
image_array = io.imread(image_path)

# Load the mask file
masks_path = 'masks22.npy'  # Replace with your mask file path
masks_data = np.load(masks_path, allow_pickle=True)

# Apply each mask to the image by setting the corresponding pixels to white
for mask_dict in masks_data:
    mask = mask_dict['segmentation']
    image_array[mask] = [255, 255, 255]  # Set to white

# Convert the image to grayscale
gray_sk = color.rgb2gray(image_array)

# Check if the image has more than one color value
if len(np.unique(gray_sk)) > 1:
    # Apply Otsu's method for thresholding
    threshold_value = filters.threshold_otsu(gray_sk)
    binary = gray_sk < threshold_value

    # Remove small objects from the binary image
    cleaned = morphology.remove_small_objects(binary, min_size=50)

    # Create a mask to convert small particles to white
    mask = np.where(cleaned == 0, 1, 0).astype(bool)

    # Apply the mask to convert small particles to white
    image_array[mask] = [255, 255, 255]

# Convert near-white background to pure white
whitening_mask_sk = np.all(image_array > [200, 200, 200], axis=-1)
image_array[whitening_mask_sk] = [255, 255, 255]

# Convert the numpy array back to an image
final_image_sk = Image.fromarray(image_array.astype(np.uint8))

# Save the final processed image to disk
processed_img_path = 'new_processed_image.jpg'  # Replace with your desired save path
final_image_sk.save(processed_img_path)


def extract_and_place_on_canvas(image_array, min_pixel_size=100):
    # Convert the image to grayscale
    gray_image = color.rgb2gray(image_array)

    # Binarize the image by setting a threshold manually
    binary_image = gray_image < 0.5

    # Label connected regions of the binary image
    labeled_image = measure.label(binary_image, connectivity=2)

    # Find properties of labeled regions
    region_props = measure.regionprops(labeled_image)

    # List to store the output images and their properties for saving later
    output_images = []

    # Loop over each region to check its area and extract it if it's larger than min_pixel_size
    for region in region_props:
        if region.area >= min_pixel_size:
            # Extract the region's bounding box
            minr, minc, maxr, maxc = region.bbox
            # Extract the region from the original image
            region_image = image_array[minr:maxr, minc:maxc]
            # Create a mask for the region
            mask = labeled_image[minr:maxr, minc:maxc] == region.label
            # Initialize a white canvas with the same size as the original image
            canvas = np.ones(image_array.shape, dtype=np.uint8) * 255
            # Place the region on the canvas at the corresponding location
            for i in range(3):  # Assuming image has 3 channels (RGB)
                canvas[minr:maxr, minc:maxc, i] = np.where(mask, region_image[:, :, i], 255)
            output_images.append(canvas)

    return output_images


# Load the image from the file system
image_path = 'new_processed_image.jpg'  # Replace with the path to your image file
image_array = io.imread(image_path)

# Extract large components and place them on a white canvas
extracted_images_on_canvas = extract_and_place_on_canvas(image_array)


# Display and save the extracted images on canvas with the specified naming convention
for idx, img in enumerate(extracted_images_on_canvas, start=len(extracted_images)):  # Start index at 2
    # Save the image with the given naming convention
    
    save_path = f'extracted_image{idx}_0.jpg'  # Replace with your save path
    Image.fromarray(img).save(save_path)




def extract_components_and_update_masks(image_array, masks_path, min_pixel_size=100):
    # Load the existing masks
    existing_masks = np.load(masks_path, allow_pickle=True)

    # Convert the image to grayscale
    gray_image = color.rgb2gray(image_array)

    # Binarize the image by setting a threshold manually
    binary_image = gray_image < 0.5

    # Label connected regions of the binary image
    labeled_image = measure.label(binary_image, connectivity=2)

    # Find properties of labeled regions
    region_props = measure.regionprops(labeled_image)

    # List to store the updated masks
    updated_masks = list(existing_masks)

    # Loop over each region to check its area and extract it if it's larger than min_pixel_size
    for idx, region in enumerate(region_props, start=2):  # Start the index from 2
        if region.area >= min_pixel_size:
            # Create a mask for the region
            mask = labeled_image == region.label

            # Construct mask dictionary similar to the given file format
            mask_dict = {
                'segmentation': mask.astype(bool),
                'area': region.area,
                'bbox': region.bbox,
                'predicted_iou': None,  # Placeholder, as we do not have this data
                'point_coords': None,  # Placeholder, as we do not have this data
                'stability_score': None,  # Placeholder, as we do not have this data
                'crop_box': None  # Placeholder, as we do not have this data
            }

            # Append the new mask dictionary to the list
            updated_masks.append(mask_dict)

    return updated_masks


# Load the image from the file system
image_path = 'new_processed_image.jpg'  # Replace with the path to your image file
image_array = io.imread(image_path)

# Path to the existing masks file
masks_path = 'masks22.npy'  # Replace with the path to your masks file

# Extract components and update masks
updated_masks = extract_components_and_update_masks(image_array, masks_path)

# Save the updated masks to the filesystem
updated_masks_path = 'masks22.npy'  # Replace with your save path
np.save(updated_masks_path, updated_masks)




def extract_and_skeletonize(data):
    """Extract and skeletonize segmentation masks."""
    skeletons = []
    for item in data:
        segmentation_mask = item['segmentation']
        segmentation_mask_contiguous = np.ascontiguousarray(segmentation_mask, dtype=bool)
        skeleton = skeletonize(segmentation_mask_contiguous)
        skeletons.append(skeleton)
    return np.array(skeletons)

def find_branch_points(skeleton):
    """Find branch points in a skeletonized image."""
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    convolved = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    return np.logical_and(skeleton, convolved >= 13)

def find_endpoints(skeleton):
    """Find endpoints in a skeletonized image."""
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    convolved = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    return np.logical_and(skeleton, np.isin(convolved, [11, 21]))

def is_adjacent_to_branch(endpoint, branch_points):
    """Check if an endpoint is adjacent to any of the branch points."""
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            neighbor = (endpoint[0] + dx, endpoint[1] + dy)
            if neighbor in branch_points:
                return True
    return False

def sample_every_other_pixel(coord_list):
    """Sample every other pixel in a list of coordinates."""
    #return coord_list[::2]
    return coord_list


def find_endpoints_of_components(component_coords):
    """Find two endpoints of a component."""
    if len(component_coords) >= 2:
        return [component_coords[0], component_coords[-1]]
    else:
        return [component_coords[0], component_coords[0]]



# Load the data
data_path = 'masks22.npy'
data = np.load(data_path, allow_pickle=True)

skeletonized_segmentations = process_skeletons(data)


masks_with_high_curvature_points = []  # List to hold indices of masks with valid high curvature points
# Process each layer and store sharp points
sharp_points_all_layers = []

for index, skeleton_layer in enumerate(skeletonized_segmentations):
    sharp_points = find_sharp_angle_points(skeleton_layer)
    if sharp_points:  # Check if the list of sharp points is not empty
        masks_with_high_curvature_points.append(index)
    sharp_points_all_layers.append({'layer': index, 'sharp_points': sharp_points})

# Extract and process skeletons
skeletons = extract_and_skeletonize(data)
points_for_sam = []
for index, skeleton in enumerate(skeletons):
    # Find branch points
    branch_points = find_branch_points(skeleton)
    branch_point_coordinates = set(map(tuple, np.argwhere(branch_points)))

    # Initialize high curvature points for the current layer
    high_curvature_points = set()

    # Check and extract high curvature points if they exist for the current layer
    for layer in sharp_points_all_layers:
        if layer['layer'] == index:
            high_curvature_points = set(map(tuple, layer['sharp_points']))
            break

    # Combine branch points and high curvature points
    combined_points = branch_point_coordinates.union(high_curvature_points)

    # Create a copy of the skeleton without these points
    skeleton_without_branches = np.copy(skeleton)
    for coord in combined_points:
        skeleton_without_branches[coord] = 0

    labeled_skeleton, _ = label(skeleton_without_branches, return_num=True)
    small_components = [region for region in regionprops(labeled_skeleton) if region.area < 40]
    branch_points_list = [tuple(coord) for coord in branch_point_coordinates]

    for component in small_components:
        component_coords = component.coords
        component_mask = np.zeros_like(skeleton_without_branches, dtype=bool)

        for coord in component_coords:
            component_mask[tuple(coord)] = True

        endpoints = find_endpoints(component_mask)
        endpoints_coordinates = np.argwhere(endpoints)

        if sum(is_adjacent_to_branch(tuple(endpoint), branch_points_list) for endpoint in
               endpoints_coordinates) != 3:  
            for coord in component_coords:
                skeleton_without_branches[tuple(coord)] = 0

    # Record coordinates of remaining components
    labeled_skeleton, num_components = label(skeleton_without_branches, return_num=True)
    component_dict = {}
    for i in range(1, num_components + 1):
        component_coords = np.argwhere(labeled_skeleton == i)
        component_dict[f'Component_{i}'] = sample_every_other_pixel([tuple(coord[::-1]) for coord in component_coords])

    points_for_sam.append({f'Skeleton_{index + 1}': component_dict})

# 'points_for_sam' now contains the coordinates grouped by each skeleton


# 'points_for_sam' contains the coordinates of sampled pixels from the remaining components
endpoints_for_sam = []
for skeleton_dict in points_for_sam:
    for skeleton_key, components in skeleton_dict.items():
        skeleton_endpoints = {}
        for component_key, component_coords in components.items():
            endpoints = find_endpoints_of_components(component_coords)
            skeleton_endpoints[component_key] = endpoints
        endpoints_for_sam.append({skeleton_key: skeleton_endpoints})

from segment_anything import sam_model_registry, SamPredictor
#print(1)
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white',
               linewidth=1.25)

sam_checkpoint = "segment-anything-main/sam_vit_h_4b8939.pth"
device = "cuda"
model_type = "default"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

for i, skeleton in enumerate(points_for_sam):
    for skeleton_name, components in skeleton.items():
        for component_name, coordinates in components.items():
            # Extracting coordinates of each component
            
            item = coordinates
            image = cv2.imread(f'extracted_image{i}_0.jpg')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
            predictor.set_image(image)
 
            input_point = np.array(item)
            input_label = np.array([1] * len(input_point))
 
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            Sperm_tail.append(masks[0])


#print(2)
def standardize_format(sperm_tail):
    standardized_data = []
    for item in sperm_tail:
        if isinstance(item, dict):

            standardized_item ={'segmentation': item['segmentation']}
        else:

            standardized_item = {'segmentation': item}
        standardized_data.append(standardized_item)
    return standardized_data
Sperm_tail = standardize_format(Sperm_tail)
Sperm_tail=np.array(Sperm_tail)
np.save('Sperm_tail.npy', Sperm_tail)


# Define the function to filter masks based on a threshold
def filter_masks(masks, threshold=50):
    filtered_masks = []

    for mask_dict in masks:
        segmentation = mask_dict['segmentation']
        if np.sum(segmentation) >= threshold:
            filtered_masks.append(mask_dict)

    return filtered_masks


# Load the .npy file
file_path = 'Sperm_tail.npy'
original_masks = np.load(file_path, allow_pickle=True)

# Filter the masks
filtered_masks = filter_masks(original_masks, threshold=450)

np.save('Sperm_tail.npy', np.array(filtered_masks, dtype=object))



# Define the IOU computation function
def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0

# Load the original set of masks
masks_path = 'Sperm_tail.npy'
masks = np.load(masks_path, allow_pickle=True)

# Identify pairs of masks with IOU >= 0.5 and decide which one to keep
masks_to_remove = set()
for i in range(len(masks)):
    for j in range(i+1, len(masks)):
        iou = compute_iou(masks[i]['segmentation'], masks[j]['segmentation'])
        if iou >= 0.5:
            # Compare the sum of True values in each mask
            sum_i = masks[i]['segmentation'].sum()
            sum_j = masks[j]['segmentation'].sum()
            # Add the index of the mask with fewer True values to the removal set
            masks_to_remove.add(j if sum_i >= sum_j else i)

# Remove the identified masks
masks = np.delete(masks, list(masks_to_remove))

# Save the updated masks
updated_masks_path = 'Sperm_tail.npy'
np.save(updated_masks_path, masks)


# Load the masks from the provided file
masks_path = 'filtered_masks.npy'  
masks = np.load(masks_path, allow_pickle=True)

# Identify pairs of masks with IOU >= 0.5 and decide which one to keep
masks_to_remove = set()
for i in range(masks.shape[0]):
    for j in range(i+1, masks.shape[0]):
        iou = compute_iou(masks[i], masks[j])
        if iou >= 0.5:
            # Compare the sum of True values in each mask
            sum_i = masks[i].sum()
            sum_j = masks[j].sum()
            # Add the index of the mask with fewer True values to the removal set
            masks_to_remove.add(j if sum_i >= sum_j else i)

# Remove the identified masks
filtered_masks = np.delete(masks, list(masks_to_remove), axis=0)

# Save the updated masks to a new file
updated_masks_path = 'filtered_masks.npy'  
np.save(updated_masks_path, filtered_masks)





def calculate_max_inscribed_distance(mask):
    """
    Calculate the maximum inscribed distance within a mask.
    """
    # Find all points in the mask
    y_coords, x_coords = np.where(mask)
    points = np.column_stack((x_coords, y_coords))

    # Calculate all pairwise distances and return the max
    if len(points) > 1:
        return max(pdist(points))
    else:
        return 0

def is_touching_border(mask):
    """
    Check if a mask touches the image border.
    """
    # Check if any True values are on the border of the image
    return np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[:, -1])

def process_masks(data):
    """
    Process each mask in the data and return indices of masks to be removed.
    """
    masks_to_remove = []
    for i, item in enumerate(data):
        segmentation_mask = item['segmentation']

        # Calculate the maximum inscribed distance
        max_distance = calculate_max_inscribed_distance(segmentation_mask)

        # Check if the mask touches the border
        touches_border = is_touching_border(segmentation_mask)

        # If max distance is less than 80 pixels and the mask touches the border, mark for removal
        if max_distance < 30 and touches_border:#
            masks_to_remove.append(i)
        elif max_distance <20:
            masks_to_remove.append(i)

    return masks_to_remove

# Load your data
data = np.load('Sperm_tail.npy', allow_pickle=True)

# Process the data
masks_to_remove = process_masks(data)

# Optionally, remove the marked masks
data = np.delete(data, masks_to_remove, axis=0)
np.save('Sperm_tail.npy',data)





# Function to label connected components in a layer
def label_connected_components(layer):
    return label(layer, return_num=True)

# Function to split a mask layer into individual layers for each connected component
def split_layer_into_components(layer):
    labeled, num_components = label_connected_components(layer)
    component_layers = []

    for component in range(1, num_components + 1):
        # Create a layer for each component
        component_layer = (labeled == component)
        component_layers.append(component_layer)

    return component_layers

# Load the mask file
file_path = 'filtered_masks.npy'
masks = np.load(file_path)

# Analyze each layer of the mask for connected components
layers_with_multiple_components = []

for layer_index in range(masks.shape[0]):
    _, num_components = label_connected_components(masks[layer_index])
    if num_components >= 2:
        layers_with_multiple_components.append(layer_index)

# Split layers with multiple components and update the masks array
new_masks = []

for layer_index in range(masks.shape[0]):
    if layer_index in layers_with_multiple_components:
        # Split the layer into individual components
        new_layers = split_layer_into_components(masks[layer_index])
        new_masks.extend(new_layers)
    else:
        # Keep the layer as it is
        new_masks.append(masks[layer_index])

# Convert list back to numpy array and save the updated masks to the file
new_masks = np.array(new_masks)
updated_file_path = 'filtered_masks.npy'
np.save(updated_file_path, new_masks)





# Load the segmentation masks from the provided file
file_path = 'filtered_masks.npy'  # Replace with your file path
masks = np.load(file_path)

# Filter out masks where the number of True pixels is less than 400
filtered_masks = [mask for mask in masks if np.sum(mask) >= 700]

# Save the filtered masks back to the original file
np.save(file_path, filtered_masks)


# Function to calculate the largest distance and draw the longest line
def find_and_draw_longest_line(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None  # If no contours found, return None

    longest_line = None
    max_distance = 0
    # Check each contour
    for contour in contours:
        # Approximate contour to reduce the number of points
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # Check the distance between each pair of points
        for i in range(len(approx)):
            for j in range(i + 1, len(approx)):
                distance = np.linalg.norm(approx[i][0] - approx[j][0])
                if distance > max_distance:
                    max_distance = distance
                    longest_line = (tuple(approx[i][0]), tuple(approx[j][0]))

    # Create an empty canvas with the same dimensions as the mask
    canvas = np.zeros_like(mask)
    if longest_line is not None:
        # Draw the longest line on the canvas
        cv2.line(canvas, longest_line[0], longest_line[1], 255, 1)

    return canvas, longest_line


# Load the numpy array
masks = np.load('filtered_masks.npy')

# Create an array to store the canvases
canvases = []

# Iterate through each mask and find the longest line
for i, mask in enumerate(masks):
    canvas, line = find_and_draw_longest_line(mask)
    if canvas is not None:
        canvases.append(canvas)

# Merge all canvases into one white canvas
final_canvas = np.zeros_like(masks[0])
for canvas in canvases:
    final_canvas = cv2.bitwise_or(final_canvas, canvas)

# Save the final result to a file
output_path = 'final_canvas.jpg'
cv2.imwrite(output_path, final_canvas)


def remove_masks_with_border_objects(masks, threshold=15):
    """
    Remove masks where the number of non-zero pixels touching the border
    of the image is greater than a specified threshold.

    :param masks: A numpy array of masks.
    :param threshold: The maximum allowed number of non-zero pixels on the border.
    :return: A numpy array of masks with border-touching objects removed.
    """
    filtered_masks = []
    for mask in masks:
        # Count non-zero pixels on each edge
        top_edge_count = np.sum(mask[0, :] > 0)
        bottom_edge_count = np.sum(mask[-1, :] > 0)
        left_edge_count = np.sum(mask[:, 0] > 0)
        right_edge_count = np.sum(mask[:, -1] > 0)

        # Total count of non-zero pixels on all edges
        total_edge_count = top_edge_count + bottom_edge_count + left_edge_count + right_edge_count

        # Check if the total count is within the threshold
        if total_edge_count <= threshold:
            filtered_masks.append(mask)

    return np.array(filtered_masks)

# Load the provided file
file_path = 'filtered_masks.npy'  # Replace with your file path
masks = np.load(file_path)

# Applying the function with the new threshold rule
filtered_masks = remove_masks_with_border_objects(masks)

# Saving the filtered masks to a new file
output_file_path = 'filtered_masks.npy'  # Replace with your desired save path
np.save(output_file_path, filtered_masks)


# Function to calculate the largest distance
def find_longest_line(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # If no contours found, return None

    longest_line = None
    max_distance = 0
    # Check each contour
    for contour in contours:
        # Approximate contour to reduce the number of points
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # Check the distance between each pair of points
        for i in range(len(approx)):
            for j in range(i + 1, len(approx)):
                distance = np.linalg.norm(approx[i][0] - approx[j][0])
                if distance > max_distance:
                    max_distance = distance
                    longest_line = (tuple(approx[i][0]), tuple(approx[j][0]))

    return longest_line

# Load the numpy array
masks = np.load('filtered_masks.npy')  # Make sure to have this file in the current directory

# List to store the longest lines with layer information
longest_lines_info = []

# Iterate through each mask and find the longest line
for i, mask in enumerate(masks):
    longest_line = find_longest_line(mask)
    if longest_line:
        # Append a dictionary with layer number and endpoints of the longest line
        longest_lines_info.append({'layer': i, 'endpoints': [longest_line[0], longest_line[1]]})

sperm_tail_data = np.load('Sperm_tail.npy', allow_pickle=True)
# Process all layers of the segmentation data
skeletonized_segmentations = process_skeletons(sperm_tail_data)


output_file = 'skeletonized_segmentations.npy'  # Replace with your desired save path
np.save(output_file, skeletonized_segmentations)


# Function to identify branch points in a skeleton
def find_branch_points(skeleton):
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    convolved = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    return np.logical_and(skeleton, convolved >=13)


# Function to find the nearest branch point for each endpoint
def nearest_branch_point(endpoint, branch_points):
    y, x = np.where(branch_points)
    if len(x) == 0 or len(y) == 0:

        return endpoint, float('inf')

    distances = np.sqrt((x - endpoint[0])**2 + (y - endpoint[1])**2)
    nearest_index = np.argmin(distances)
    return (x[nearest_index], y[nearest_index]), distances[nearest_index]


# Modified function to find endpoints with consideration of branch points
def find_modified_endpoints(skeletons):
    all_endpoints = []
    for index, skeleton in enumerate(skeletons):
        branch_points = find_branch_points(skeleton)
        labeled_skeleton= label(skeleton)
        properties = regionprops(labeled_skeleton)
        endpoints = []
        for prop in properties:
            if prop.extent < 0.5:
                coords = prop.coords
                for coord in coords:
                    if np.sum(skeleton[coord[0]-1:coord[0]+2, coord[1]-1:coord[1]+2]) == 2:
                        nearest_bp, distance = nearest_branch_point((coord[1], coord[0]), branch_points)
                        if distance < 25:
                            endpoints.append(nearest_bp)
                        else:
                            endpoints.append((coord[1], coord[0]))
        endpoints = list(set(endpoints))  # Remove duplicates
        all_endpoints.append({'layer': index, 'endpoints': endpoints})
    return all_endpoints

# Find the endpoints for all layers using region properties
all_endpoints = find_modified_endpoints(skeletonized_segmentations)


# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to find the two farthest points in a list of points
def find_farthest_points(points):
    max_distance = 0
    farthest_points = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            distance = calculate_distance(points[i], points[j])
            if distance > max_distance:
                max_distance = distance
                farthest_points = [points[i], points[j]]
    return farthest_points



# Process each layer to keep only the two farthest points
for layer_data in all_endpoints:
    # Find the two farthest points if there are more than one point
    if len(layer_data['endpoints']) > 1:
        layer_data['endpoints'] = find_farthest_points(layer_data['endpoints'])


def find_point_on_skeleton(skeleton, start, distance):

    current_point = start
    visited = set()
    height, width = skeleton.shape

    for _ in range(distance):
        visited.add(current_point)
        x, y = current_point
        neighbors = [(x2, y2) for x2 in range(x - 1, x + 2)
                     for y2 in range(y - 1, y + 2)
                     if 0 <= x2 < width and 0 <= y2 < height and (x2, y2) != current_point and skeleton[y2, x2] and (
                     x2, y2) not in visited]

        if not neighbors:
            break  

        current_point = neighbors[0]
        visited.add(current_point)

    return current_point


def extend_endpoints_on_skeletons(skeletons, all_endpoints, distance):

    all_endpoints_20 = []

    for item in all_endpoints:
        layer = item['layer']
        endpoints = item['endpoints']
        skeleton = skeletons[layer]

        extended_points = []
        for point in endpoints:
            extended_point = find_point_on_skeleton(skeleton, point, distance)
            extended_points.append(extended_point)

        all_endpoints_20.append({'layer': layer, 'endpoints': extended_points})

    return all_endpoints_20



distance = 5 

all_endpoints_20 = extend_endpoints_on_skeletons(skeletonized_segmentations, all_endpoints, distance)



# Load the image and convert it to an RGB numpy array
image_path = f'data/original_images/{I}.jpg'  
image = Image.open(image_path).convert('RGB')
image_array = np.array(image)
hsv_image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)  

# Function to get the value (brightness) of a point in HSV space
def value_of_point(point, hsv_img_array):
    return hsv_img_array[point[1], point[0], 2]  

def keep_corresponding_minimum_value_point(all_endpoints, all_endpoints_20, hsv_img_array):
    corresponding_minimum_value_points_per_layer = []

    for layer_endpoints, layer_endpoints_20 in zip(all_endpoints, all_endpoints_20):
        layer = layer_endpoints['layer']


        if len(layer_endpoints['endpoints']) != len(layer_endpoints_20['endpoints']):
            continue


        minimum_value_point = None
        minimum_value = float('inf')

        for point, point_20 in zip(layer_endpoints['endpoints'], layer_endpoints_20['endpoints']):
            current_value = value_of_point(point_20, hsv_img_array)
            if current_value < minimum_value:
                minimum_value = current_value
                minimum_value_point = point

        if minimum_value_point is not None:
            corresponding_minimum_value_points_per_layer.append({'layer': layer, 'endpoints': [minimum_value_point]})

    return corresponding_minimum_value_points_per_layer


all_endpoints = keep_corresponding_minimum_value_point(all_endpoints, all_endpoints_20, hsv_image_array)


def find_branch_length(skeleton, start_point, visited, max_length=50):

    length = 0
    current_point = start_point
    while length < max_length:
        length += 1
        x, y = current_point
        neighbors = [(x2, y2) for x2 in range(x - 1, x + 2)
                     for y2 in range(y - 1, y + 2)
                     if 0 <= x2 < skeleton.shape[0] and 0 <= y2 < skeleton.shape[1] and
                     skeleton[x2, y2] and (x2, y2) != current_point and (x2, y2) not in visited]
        if not neighbors:
            break  

        current_point = neighbors[0]  
        visited.add(current_point)

    return length


def find_longest_branch_point(skeleton, current_point, points):

    max_length = 0
    best_point = None
    visited = set(points)
    for neighbor in [(x, y) for x in range(current_point[0] - 1, current_point[0] + 2)
                     for y in range(current_point[1] - 1, current_point[1] + 2)
                     if 0 <= x < skeleton.shape[0] and 0 <= y < skeleton.shape[1] and
                        skeleton[x, y] and (x, y) != current_point and (x, y) not in points]:
        branch_length = find_branch_length(skeleton, neighbor, visited)
        if branch_length > max_length:
            max_length = branch_length
            best_point = neighbor

    return best_point


def find_skeleton_points(skeleton, start_point, num_points=30, max_branch_length=100):

    points = [start_point]
    for _ in range(num_points - 1):
        current_point = points[-1]
        next_point = find_longest_branch_point(skeleton, current_point, points)
        if next_point is None:
            break  

        points.append(next_point)

    return points


def calculate_angle(points):

    points_array = np.array(points)


    model, _ = ransac(points_array, LineModelND, min_samples=2, residual_threshold=1, max_trials=100)


    line_origin = model.params[0]  
    line_direction = model.params[1]  


    last_point = points_array[-1]
    projected_point = line_origin + np.dot((last_point - line_origin), line_direction) * line_direction

    first_point = points_array[0]
    projected_point2 = line_origin + np.dot((first_point - line_origin), line_direction) * line_direction


    direction_to_start = projected_point2 - projected_point


    angle = math.atan2(direction_to_start[0], direction_to_start[1]) * 180 / math.pi


    return angle if angle >= 0 else angle + 360





# Assuming `skeletonized_segments` contains your skeletonized data and `all_endpoints` is your list of endpoints
angles_with_20_points = []

for layer_info in all_endpoints:
    layer = layer_info['layer']
    angles = []
    for endpoint in layer_info['endpoints']:
        # Convert endpoint coordinates (x, y) to (row, column)
        endpoint_rc = (endpoint[1], endpoint[0])
        points = find_skeleton_points(skeletonized_segmentations[layer], endpoint_rc)
        angle = calculate_angle(points)
        angles.append(angle)
    angles_with_20_points.append({'layer': layer, 'angles': angles})




def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def find_matching_tails(heads, tails, threshold):
   
    selected_tails_for_heads = {}
    for head in heads:
        head_layer = head['layer']
        selected_tails = []
        for tail in tails:
            tail_layer = tail['layer']
            for head_endpoint in head['endpoints']:
                for tail_endpoint in tail['endpoints']:
                    if distance(head_endpoint, tail_endpoint) <= threshold:
                        selected_tails.append(tail_layer)
                        break 
                if tail_layer in selected_tails:
                    break  
        selected_tails_for_heads[head_layer] = list(set(selected_tails))  
    return selected_tails_for_heads

selected_tails = find_matching_tails(longest_lines_info, all_endpoints, 27)







# Function to calculate the distance between two points
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to calculate the angle between two vectors
def angle_between_points(p1, p2, p3):
    """
    Calculate the angle between the vector from p1 to p2 and the vector from p2 to p3
    """
    a = (p2[0] - p1[0], p2[1] - p1[1])
    b = (p3[0] - p2[0], p3[1] - p2[1])
    inner_product = a[0]*b[0] + a[1]*b[1]
    len_a = math.sqrt(a[0]**2 + a[1]**2)
    len_b = math.sqrt(b[0]**2 + b[1]**2)
    return math.acos(inner_product / (len_a * len_b))

# Function to find the closest point pair between head and tail
def closest_points(head, tail):
    """
    Find the closest points between a head and a tail.
    Return the points as (head_point, tail_point).
    """
    min_dist = float('inf')
    closest_pair = None
    for h_point in head:
        for t_point in tail:
            dist = distance(h_point, t_point)
            if dist < min_dist:
                min_dist = dist
                closest_pair = (h_point, t_point)
    return closest_pair




# Function to calculate the distance between two points
def distance(p1, p2):

    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)



# Function to calculate the angle between two vectors
def angle_difference(closest_tail_point, h_point, tail_layer, angles_with_20_points):

    angle_to_h_point = math.degrees(math.atan2(h_point[1] - closest_tail_point[1], h_point[0] - closest_tail_point[0]))


    if angle_to_h_point < 0:
        angle_to_h_point += 360


    angle_in_data = next(item for item in angles_with_20_points if item['layer'] == tail_layer)['angles'][0]


    angle_diff = abs(angle_to_h_point - angle_in_data)


    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    return angle_diff


# Function to find the closest point pair between head and tail
def closest_points(head, tail):

    """
    Find the closest points between a head and a tail.
    Return the points as (head_point, tail_point).
    """
    min_dist = float('inf') 
    closest_pair = None 
    for h_point in head:  
        for t_point in tail:  
            dist = distance(h_point, t_point)  
            if dist < min_dist: 
                min_dist = dist  
                closest_pair = (h_point, t_point)  
    return closest_pair  


# Convert the data into a more usable format
heads = {item['layer']: item['endpoints'] for item in longest_lines_info}

tails = {item['layer']: item['endpoints'] for item in all_endpoints}

# Process each head
final_selection3 = {}




for head_layer, tail_layers in selected_tails.items():
    head = heads[head_layer]
    min_angle_diff = float('inf')
    selected_tail = None

    for tail_layer in tail_layers:
        tail = tails[tail_layer]
        closest_head_point, closest_tail_point = closest_points(head, tail)

        for h_point in head:
            if h_point != closest_head_point:
                angle_diff = angle_difference(closest_head_point, h_point, tail_layer, angles_with_20_points)
                if angle_diff < min_angle_diff:
                    min_angle_diff = angle_diff
                    selected_tail = tail_layer

    # Check if the minimum angle difference exceeds 100 degrees
    if min_angle_diff > 40: 
        final_selection3[head_layer] = None
    else:
        final_selection3[head_layer] = selected_tail

# Include heads not in the final selection with None
for head_layer in heads:
    if head_layer not in final_selection3:
        final_selection3[head_layer] = None















while True:
    # Find all heads assigned to each tail
    tail_to_heads = {}
    for head, tail in final_selection3.items():
        if tail is not None:
            tail_to_heads.setdefault(tail, []).append(head)

    # Identify tails with more than one head assigned
    conflict_tails = {tail: heads for tail, heads in tail_to_heads.items() if len(heads) > 1}

    # Break loop if no conflicts
    if not conflict_tails:
        break

    # Resolve conflicts
    for tail, conflicting_heads in conflict_tails.items():
        min_angle = float('inf')
        selected_head = None

        # Find the head with the smallest angle to the tail
        for head in conflicting_heads:
            head_points = heads[head]
            tail_point = tails[tail][0]
            for head_point in head_points:
                angle = angle_between_points(head_point, tail_point, (head_point[0]+1, head_point[1]))  # Example angle calculation
                if angle < min_angle:
                    min_angle = angle
                    selected_head = head

        # Reassign tails for non-selected heads
        for head in conflicting_heads:
            if head != selected_head:
                alternative_tails = [t for t in selected_tails[head] if t != tail and t not in tail_to_heads]
                final_selection3[head] = alternative_tails[0] if alternative_tails else None


head_masks_path = 'filtered_masks.npy'
head_masks = np.load(head_masks_path)


sperm_tail_path = 'Sperm_tail.npy'
sperm_tail = np.load(sperm_tail_path, allow_pickle=True)
segmentations = [segment['segmentation'] for segment in sperm_tail]



final_selection_filtered = {k: v for k, v in final_selection3.items() if v is not None}





combined_masks = {}





original_head_masks = []
original_tail_masks = []


for i, (head_layer, tail_layer) in enumerate(final_selection_filtered.items()):
    head_mask = head_masks[head_layer]
    tail_mask = segmentations[tail_layer]

    original_head_masks.append(head_mask)
    original_tail_masks.append(tail_mask)


    combined_mask = np.maximum(head_mask, tail_mask)


    combined_masks[f'Head_{head_layer}_Tail_{tail_layer}'] = combined_mask










np.save('combined_masks.npy', combined_masks)

np.save('original_head_masks.npy', original_head_masks)
np.save('original_tail_masks.npy', original_tail_masks)












from skimage.draw import polygon
from scipy.spatial import distance
from skimage.measure import label, regionprops


def create_rectangle_from_midpoints(pt1, pt2, width):
    # Calculate direction vector from pt1 to pt2
    direction = np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]])
    length = np.linalg.norm(direction)
    direction = direction / length

    # Calculate perpendicular vector
    perp_direction = np.array([-direction[1], direction[0]])

    # Calculate four corners
    half_width = width / 2
    corner1 = pt1 + perp_direction * half_width
    corner2 = pt1 - perp_direction * half_width
    corner3 = pt2 - perp_direction * half_width
    corner4 = pt2 + perp_direction * half_width

    return np.array([corner1, corner2, corner3, corner4])

def has_multiple_components(mask):
    labeled_mask = label(mask)
    props = regionprops(labeled_mask)
    return len(props) >= 2

def update_masks_with_rectangles(content):
    for key, mask in content.items():
        if has_multiple_components(mask):
            # Label the mask to identify separate regions
            labeled_mask = label(mask)

            # Extract the properties of each region
            regions = regionprops(labeled_mask)

            # Find the closest points between the two largest regions
            min_dist = np.inf
            closest_points = None
            for i in range(len(regions)):
                for j in range(i + 1, len(regions)):
                    region1, region2 = regions[i], regions[j]
                    for coord1 in region1.coords:
                        for coord2 in region2.coords:
                            dist = distance.euclidean(coord1, coord2)
                            if dist < min_dist:
                                min_dist = dist
                                closest_points = (coord1, coord2)

            # Define width of the rectangle
            width = 15

            # Create the rectangle using the midpoints of the closest points
            rectangle_coords = create_rectangle_from_midpoints(closest_points[0], closest_points[1], width)

            # Extract polygon coordinates for drawing the rectangle
            rr, cc = polygon(rectangle_coords[:, 0], rectangle_coords[:, 1], mask.shape)

            # Update the mask with the rectangle
            mask[rr, cc] = 1  # Fill the rectangle area

            # Update the content with the new mask
            content[key] = mask

    return content

# Load the mask file
file_path = 'combined_masks.npy'
masks = np.load(file_path, allow_pickle=True)
content = masks.item()

# Update the masks in the original file
updated_content = update_masks_with_rectangles(content)

# Save the updated masks to a new file
updated_file_path = 'combined_masks.npy'
np.save(updated_file_path, updated_content)










# Load the numpy array
data = np.load('combined_masks.npy', allow_pickle=True)

# Access the contained item
item = data.item()

# Extract values
all_values = []
if isinstance(item, dict):
    all_values.extend(item.values())
elif isinstance(item, (list, tuple)):
    for sub_item in item:
        if isinstance(sub_item, dict):
            all_values.extend(sub_item.values())

# Convert to numpy array
new_array = np.array(all_values, dtype=object)

# Now `new_array` contains all the extracted values


# new_array now contains all the values extracted from the dictionaries








# Load the numpy array
valid_masks_path = 'valid_masks.npy'
valid_masks = np.load(valid_masks_path, allow_pickle=True)

# Access the content if it's a 0-dimensional numpy array
if valid_masks.ndim == 0:
    valid_masks = valid_masks.item()

# Convert boolean values to integers: False to 0, True to 1
valid_masks_converted = np.array(valid_masks).astype(int)

# Now `valid_masks_converted` contains the converted values




Last=list(valid_masks_converted)+list(new_array)


np.save("combined_masks.npy",Last)












# Function to convert mask to polygon points
# Function to convert mask to polygon points without approximation
def mask_to_polygon_points(mask):

    mask = mask.astype(np.uint8)
    

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:

        polygon = contour.reshape(-1, 2).tolist()
        polygons.append(polygon)

    return polygons


# Load the .npy file which contains the masks
npy_file_path = 'combined_masks.npy'








masks = np.load(npy_file_path, allow_pickle=True)


#mask_content = masks.item()





# Define the basic structure of the JSON output
json_output = {
    "version": "0.2.4",  # Assuming a hypothetical version for the output format
    "flags": {},
    "shapes": [],
    "imagePath": f"{I}.jpg",
    "imageData": None,
    "imageHeight": 540,
    "imageWidth": 720,
    "text": ""
}














for index, mask in enumerate(masks):
    polygons = mask_to_polygon_points(mask)
    for polygon in polygons:
        shape_data = {
            "label": "sperm",
            "points": polygon,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        json_output['shapes'].append(shape_data)

# Assuming 'I' is defined earlier in your code or you replace it with appropriate logic to generate file names
json_output_path = f'data/original_images/{I}.json'  # Replace 'I' with appropriate file name logic
with open(json_output_path, 'w') as json_file:
    json.dump(json_output, json_file, indent=4)




json_file_path = f'data/original_images/{I}.json'


with open(json_file_path, 'r') as file:
    data = json.load(file)


data['shapes'] = [shape for shape in data['shapes'] if len(shape['points']) >= 10]


with open(json_file_path, 'w') as file:
    json.dump(data, file, indent=4)
