import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure, morphology

# Load the Real Image

# ----------------------
# Step 1: Load Real Image
# ----------------------
image = cv2.imread('10-11318_6050-PG_17_0310.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
pixel_size = 0.73  # µm/pixel
height, width = gray_image.shape

# ----------------------
# Step 2: Preprocessing with CLAHE and Noise Removal
# ----------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(25, 25))
enhanced_image = clahe.apply(gray_image)
binary_mask = cv2.inRange(enhanced_image, 100, 255)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

# Area filtering
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
filtered_mask = np.zeros_like(cleaned_mask)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= 200:
        filtered_mask[labels == i] = 255

# Final dilation to connect features
final_mask = cv2.dilate(filtered_mask, kernel, iterations=1)
output_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=final_mask)


bgr_output = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
lab_output = cv2.cvtColor(bgr_output, cv2.COLOR_BGR2LAB)



lab = cv2.cvtColor(lab_output, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

# Step 1: Illumination Correction using L channel
blurred = cv2.GaussianBlur(L, (101, 101), 0)
normalized = cv2.subtract(L, blurred)
normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

# Step 2: CLAHE for Contrast Enhancement
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(25, 25))
enhanced = clahe.apply(normalized)

# Step 3: Noise Removal BEFORE Thresholding
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
denoised = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)

# Area filtering
_, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
cleaned_binary = np.zeros_like(binary)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] > 200:
        cleaned_binary[labels == i] = 255

# Step 4: Adaptive Thresholding on Cleaned Image
coating_thresh = cv2.adaptiveThreshold(output_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 8)
substrate_thresh = cv2.adaptiveThreshold(output_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 10)
background_thresh = cv2.adaptiveThreshold(output_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)

coating_thresh_1 = cv2.adaptiveThreshold(output_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 8)
substrate_thresh_1 = cv2.adaptiveThreshold(output_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 15, 8)
background_thresh_1 = cv2.adaptiveThreshold(output_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)

# Step 5: Morphological Cleanup
coating_mask = cv2.morphologyEx(coating_thresh, cv2.MORPH_OPEN, kernel)
substrate_mask = cv2.morphologyEx(substrate_thresh, cv2.MORPH_OPEN, kernel)
background_mask = cv2.morphologyEx(background_thresh, cv2.MORPH_OPEN, kernel)

coating_mask1 = cv2.morphologyEx(coating_thresh_1, cv2.MORPH_OPEN, kernel)
substrate_mask1 = cv2.morphologyEx(substrate_thresh_1, cv2.MORPH_OPEN, kernel)
background_mask1 = cv2.morphologyEx(background_thresh_1, cv2.MORPH_OPEN, kernel)

# Step 6: Derived Masks
covered_coating = coating_mask1.copy()
uncovered_coating = cv2.bitwise_and(substrate_mask, cv2.bitwise_not(coating_mask))
overall_coating = cv2.bitwise_or(covered_coating, uncovered_coating)
#covered_substrate = cv2.bitwise_and(substrate_mask1, covered_coating)

# Step 7: Delamination Detection
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated_substrate = cv2.dilate(substrate_mask, kernel_dilate, iterations=2)
dilated_coating = cv2.dilate(coating_mask, kernel_dilate, iterations=2)
delaminated_mask = cv2.bitwise_and(background_mask, cv2.bitwise_and(dilated_substrate, dilated_coating))

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(delaminated_mask, connectivity=8)
delaminated_clean = np.zeros_like(delaminated_mask)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] > 220:
        delaminated_clean[labels == i] = 255

# Step 8: Region Definitions
regions = {
    'Covered (Coating Side)': covered_coating,
    #'Uncovered (Coating Side)': uncovered_coating,
    'Overall (Coating Side)': overall_coating,
   # 'Covered (Substrate Side)': covered_substrate,
   # 'Delaminated (Substrate Side)': delaminated_clean
}

color_map = {
    'Covered (Coating Side)': (255, 0, 0),
    #'Uncovered (Coating Side)': (0, 255, 0),
    'Overall (Coating Side)': (128, 0, 128),
    #'Covered (Substrate Side)': (0, 0, 255),
    #'Delaminated (Substrate Side)': (255, 255, 0)
}

pixel_size = 0.73  # μm/pixel
lengths = {}

# Step 9: Contour Length Calculation
for label, mask in regions.items():
    contours = measure.find_contours(mask, level=0.5)
    color_output = image.copy()
    total_length = 0

    for contour in contours:
        if len(contour) < 30:
            continue
        contour = np.fliplr(contour).astype(int)
        if label == 'Uncovered (Coating Side)':
            for pt in contour:
                cv2.circle(color_output, tuple(pt), 1, color_map[label], -1)
            total_length += len(contour)
        else:
            for i in range(1, len(contour)):
                pt1 = tuple(contour[i - 1])
                pt2 = tuple(contour[i])
                cv2.line(color_output, pt1, pt2, color_map[label], 1)
                total_length += np.linalg.norm(np.array(pt1) - np.array(pt2))

    lengths[label] = total_length * pixel_size
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(color_output, cv2.COLOR_BGR2RGB))
    plt.title(f"{label}: {lengths[label]:.2f} μm")
    plt.axis('off')
    plt.savefig(f"{label.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    plt.show()

# Step 10: Save to CSV
length_df = pd.DataFrame(list(lengths.items()), columns=['Region', 'Length (μm)'])
length_df.to_csv('Stack_Frontview19_length_results.csv', index=False)

# Print Results
for k, v in lengths.items():
    print(f"{k}: {v:.2f} μm")







# Load the Real Image
image = cv2.imread('10-11318_6050-PG_17_0310.jpg')
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

# Step 1: Illumination Correction using L channel
blurred = cv2.GaussianBlur(L, (101, 101), 0)
normalized = cv2.subtract(L, blurred)
normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

# Step 2: CLAHE for Contrast Enhancement
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(25, 25))
enhanced = clahe.apply(normalized)

# Step 3: Noise Removal BEFORE Thresholding
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
denoised = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)

# Area filtering
_, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
cleaned_binary = np.zeros_like(binary)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] > 200:
        cleaned_binary[labels == i] = 255

# Step 4: Adaptive Thresholding on Cleaned Image
coating_thresh = cv2.adaptiveThreshold(cleaned_binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 8)
substrate_thresh = cv2.adaptiveThreshold(cleaned_binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 10)
background_thresh = cv2.adaptiveThreshold(cleaned_binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)

coating_thresh_1 = cv2.adaptiveThreshold(cleaned_binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 8)
substrate_thresh_1 = cv2.adaptiveThreshold(cleaned_binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 15, 8)
background_thresh_1 = cv2.adaptiveThreshold(cleaned_binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)

# Step 5: Morphological Cleanup
coating_mask = cv2.morphologyEx(coating_thresh, cv2.MORPH_OPEN, kernel)
substrate_mask = cv2.morphologyEx(substrate_thresh, cv2.MORPH_OPEN, kernel)
background_mask = cv2.morphologyEx(background_thresh, cv2.MORPH_OPEN, kernel)

coating_mask1 = cv2.morphologyEx(coating_thresh_1, cv2.MORPH_OPEN, kernel)
substrate_mask1 = cv2.morphologyEx(substrate_thresh_1, cv2.MORPH_OPEN, kernel)
background_mask1 = cv2.morphologyEx(background_thresh_1, cv2.MORPH_OPEN, kernel)

# Step 6: Derived Masks
covered_coating = coating_mask1.copy()
uncovered_coating = cv2.bitwise_and(substrate_mask, cv2.bitwise_not(coating_mask))
#overall_coating = cv2.bitwise_or(covered_coating, uncovered_coating)
covered_substrate = cv2.bitwise_and(substrate_mask1, covered_coating)

# Step 7: Delamination Detection
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated_substrate = cv2.dilate(substrate_mask, kernel_dilate, iterations=2)
dilated_coating = cv2.dilate(coating_mask, kernel_dilate, iterations=2)
delaminated_mask = cv2.bitwise_and(background_mask, cv2.bitwise_and(dilated_substrate, dilated_coating))

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(delaminated_mask, connectivity=8)
delaminated_clean = np.zeros_like(delaminated_mask)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] > 220:
        delaminated_clean[labels == i] = 255

# Step 8: Region Definitions
regions = {
    #'Covered (Coating Side)': covered_coating,
    'Uncovered (Coating Side)': uncovered_coating,
    #'Overall (Coating Side)': overall_coating,
    'Covered (Substrate Side)': covered_substrate,
    'Delaminated (Substrate Side)': delaminated_clean
}

color_map = {
    'Covered (Coating Side)': (255, 0, 0),
    'Uncovered (Coating Side)': (0, 255, 0),
    'Overall (Coating Side)': (128, 0, 128),
    'Covered (Substrate Side)': (0, 0, 255),
    'Delaminated (Substrate Side)': (255, 255, 0)
}

pixel_size = 0.73  # μm/pixel
lengths = {}

# Step 9: Contour Length Calculation
for label, mask in regions.items():
    contours = measure.find_contours(mask, level=0.5)
    color_output = image.copy()
    total_length = 0

    for contour in contours:
        if len(contour) < 30:
            continue
        contour = np.fliplr(contour).astype(int)
        if label == 'Uncovered (Coating Side)':
            for pt in contour:
                cv2.circle(color_output, tuple(pt), 1, color_map[label], -1)
            total_length += len(contour)
        else:
            for i in range(1, len(contour)):
                pt1 = tuple(contour[i - 1])
                pt2 = tuple(contour[i])
                cv2.line(color_output, pt1, pt2, color_map[label], 1)
                total_length += np.linalg.norm(np.array(pt1) - np.array(pt2))

    lengths[label] = total_length * pixel_size
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(color_output, cv2.COLOR_BGR2RGB))
    plt.title(f"{label}: {lengths[label]:.2f} μm")
    plt.axis('off')
    plt.savefig(f"{label.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    plt.show()

# Step 10: Save to CSV
length_df = pd.DataFrame(list(lengths.items()), columns=['Region', 'Length (μm)'])
length_df.to_csv('Stack_Frontview19_length_results1.csv', index=False)

# Print Results
for k, v in lengths.items():
    print(f"{k}: {v:.2f} μm")


