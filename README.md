# Electrode Degradation — Traditional Image Processing Pipeline

This repository contains the baseline, rule-based pipeline used to assess electrode degradation from high-resolution microscope images. It sets a clear reference for performance before moving to data-driven methods. The implementation lives in the notebook Coating_defects_traditional.ipynb and uses OpenCV and scikit-image.

# What the pipeline does

Preprocess images to correct uneven illumination and low local contrast.

Segment coating, substrate, and background into binary masks.

Post-process masks with morphology and Boolean logic to derive defect classes.

Detect delamination by intersecting dilated coating and substrate edges within background regions.

Quantify defect lengths from contours and export results to CSV.

Export visual overlays (PNGs) for each region.

# Method (step-by-step)

** Load image **
Set the input path (e.g., 10-11318_6050-PG_17_0310.jpg) and the pixel size (default 0.73 μm/pixel).

** Illumination correction (L channel) **
Convert to LAB, take L, blur with a large Gaussian kernel (101×101), subtract, then normalize to [0, 255].

**  Contrast enhancement (CLAHE) ** 

Apply CLAHE (clipLimit=3.0, tileGridSize=25×25) to the normalized L image.

** Noise removal (before thresholding) ** 

Use morphological opening with an elliptical kernel (5×5).

** Adaptive Gaussian thresholding ** 

Compute separate masks with tuned parameters:

**  Coating: ** 

binary (blockSize 11, C≈8)

**  Substrate: ** 

inverse binary (e.g., blockSize 11–15, C≈8–10)

**  Background: ** 

inverse binary (blockSize 11, C≈2)

**  Morphological cleanup ** 

Apply opening to each thresholded mask; optionally filter by connected-component area.

Derived masks (Boolean logic)

**  Covered coating: ** 

From cleaned coating mask.

**  Uncovered coating: ** 

substrate_mask AND NOT coating_mask.

**  Overall coating: ** 

union of covered and uncovered.

**  Delamination detection ** 
Dilate coating and substrate edges (3×3 rect kernel, iterations=2) and intersect them inside the background to get delaminated regions. Remove small components (e.g., area ≤220 px).

** Length measurement (μm) ** 
Trace contours (skimage.measure.find_contours), draw polylines, sum segment lengths, and convert to microns using pixel_size.

# Outputs

CSV: e.g., Stack_Frontview19_length_results.csv with region-wise lengths.

PNG overlays per region (saved with region names).

** Inputs and outputs ** 

Input: 

A microscope image (e.g., .jpg/.png) placed in the repo or a /data folder.

Outputs:

*.png overlays for covered/overall coating and other regions.

*_length_results.csv with two columns: Region, Length (μm).

# Requirements

Python 3.9+

OpenCV, NumPy, Matplotlib, Pandas, scikit-image
