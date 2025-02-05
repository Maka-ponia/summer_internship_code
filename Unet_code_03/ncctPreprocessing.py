import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import keyboard  

# Load the NIfTI files
ncct_file = r"Y:/DATASETS/ISLES-2024/batch_1/raw_data/sub-stroke0007/ses-01/sub-stroke0007_ses-01_ncct.nii.gz"
mask_file = r"Y:/DATASETS/ISLES-2024/batch_1/derivatives/sub-stroke0007/ses-02/sub-stroke0007_ses-02_lesion-msk.nii.gz"

mask_img = nib.load(mask_file)
ncct_img = nib.load(ncct_file)

# Get image data as NumPy arrays
ncct_data = ncct_img.get_fdata()
mask_data = mask_img.get_fdata()

# Find slices containing lesions
lesion_slices = [i for i in range(mask_data.shape[2]) if np.any(mask_data[:, :, i] == 1)]
if lesion_slices:
    print(f"Lesion found in slices: {lesion_slices}")
else:
    print("No lesion detected in any slice.")

# Define Window Level (WL) and Window Width (WW) for contrast adjustment
WL = 80  # Stroke Window Level
WW = 260  # Stroke Window Width

# Function to apply windowing (contrast adjustment)
def apply_windowing(img, WL, WW):
    img = (img - (WL - WW / 2)) / WW
    img = np.clip(img, 0, 1)  # Normalize to [0,1]
    return img

def adaptive_windowing(img, percentile_low=5, percentile_high=95):
    """Applies dynamic windowing based on the intensity percentiles of the image."""
    # Calculate the 5th and 95th percentiles of the pixel intensities
    low = np.percentile(img, percentile_low)
    high = np.percentile(img, percentile_high)
    
    # Calculate WL and WW
    WL = (low + high) / 2  # Window level (center of the window)
    WW = high - low        # Window width (range of intensities)
    
    # Apply windowing to the image
    windowed_img = (img - (WL - WW / 2)) / WW
    windowed_img = np.clip(windowed_img, 0, 1)  # Normalize to [0, 1]
    
    return windowed_img

# Function to apply Canny Edge Detection
def apply_canny(image, threshold1=50, threshold2=150):
    image = np.uint8(image * 255)  # Convert to 8-bit
    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)  # Reduce noise
    edges = cv2.Canny(blurred, threshold1, threshold2)  # Apply Canny
    return edges

# Apply thresholding to the mask
mask_binary = (mask_data > 0.5).astype(int)

# Loop through lesion slices in groups of 3
for i in range(0, len(lesion_slices), 3):
    slices_to_show = lesion_slices[i:i+3]  # Process 3 slices at a time
    
    fig, axes = plt.subplots(len(slices_to_show), 3, figsize=(12, 4 * len(slices_to_show)))

    for j, slice_idx in enumerate(slices_to_show):
        # Process NCCT slice
        windowed_img = apply_windowing(ncct_data[:, :, slice_idx], WL, WW)
        # windowed_img = adaptive_windowing(ncct_data[:, :, slice_idx])
        edges = apply_canny(windowed_img)
        # NCCT Scan (Processed)
        axes[j, 0].imshow(windowed_img, cmap="gray")
        axes[j, 0].set_title(f"NCCT - Slice {slice_idx}")
        axes[j, 0].axis('off')

        # Mask alone (Lesion)
        axes[j, 1].imshow(mask_binary[:, :, slice_idx], cmap="jet")
        axes[j, 1].set_title(f"Mask - Slice {slice_idx}")
        axes[j, 1].axis('off')

        # Overlay: NCCT + Mask + Edges
        axes[j, 2].imshow(windowed_img, cmap="gray")
        axes[j, 2].imshow(mask_binary[:, :, slice_idx], cmap="jet", alpha=0.3)
        axes[j, 2].imshow(edges, cmap="inferno", alpha=0.6)  # Overlay edges
        axes[j, 2].set_title(f"Overlay - Slice {slice_idx}")
        axes[j, 2].axis('off')

    plt.tight_layout()
    plt.show()
     # Wait for user input: 'Enter' to continue, 'X' to exit
    print("\nPress 'Enter' to continue or 'X' to exit...")
    while True:
        if keyboard.is_pressed('x'):
            print("\nExiting the loop. Goodbye!")
            exit()  # Terminates the entire script
        elif keyboard.is_pressed('enter'):
            break  # Proceeds to the next set of slices
