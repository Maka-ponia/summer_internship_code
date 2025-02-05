import os
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
import nibabel as nib  # Required to read .nii files
import numpy as np
from tqdm import tqdm
import cv2 
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import math

os.environ["CUDA_VISIBLE_DEVICES"]="6"


def display_sample(image_list, save_path="output", num=0):
    plt.figure(figsize=(10, 10))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(image_list)):
        plt.subplot(1, len(image_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(image_list[i]))
        plt.axis('off')

    # Saves the plot instead of showing it
    
    plt.savefig(f"{save_path}{num}.png")
    plt.close()

# Define the root directory where ISLES-2024 dataset is stored
root_directory = "Y:/DATASETS/ISLES-2024"

def find_nii_files(root_dir):
    ncct_files = []
    mask_files = []
    
    # Traverse all batches (batch_1, batch_2, etc.)
    for batch in os.listdir(root_dir):
        batch_path = os.path.join(root_dir, batch)

        # Ensure it's a valid directory
        if not os.path.isdir(batch_path):
            continue
        
        # Search inside raw_data for NCCT scans
        raw_data_path = os.path.join(batch_path, "raw_data")
        if os.path.exists(raw_data_path):
            for sub in os.listdir(raw_data_path):  # Iterate through sub-stroke#### directories
                sub_path = os.path.join(raw_data_path, sub, "ses-01")
                if os.path.exists(sub_path):
                    ncct_files.extend(glob.glob(os.path.join(sub_path, "*ncct*.nii*")))  # Finds .nii and .nii.gz

        # Search inside derivatives for mask files
        derivatives_path = os.path.join(batch_path, "derivatives")
        if os.path.exists(derivatives_path):
            for sub in os.listdir(derivatives_path):  # Iterate through sub-stroke#### directories
                sub_path = os.path.join(derivatives_path, sub, "ses-02")
                if os.path.exists(sub_path):
                    mask_files.extend(glob.glob(os.path.join(sub_path, "*msk*.nii*")))  # Finds .nii and .nii.gz

    return ncct_files, mask_files

# Get lists of NCCT and mask file paths
ncct_list, mask_list = find_nii_files(root_directory)

# Split NCCT and Mask path lists while maintaining pairing
ncct_train, ncct_test, mask_train, mask_test = train_test_split(
    ncct_list, mask_list, test_size=0.2, random_state=42
)

def normalize_binary_mask(binary_mask):
    """
    Ensure binary mask is in the range [0, 1].
    Assumes the input is a numpy array.
    """
    # Clip values to ensure binary (0 or 1)
    binary_mask = tf.clip_by_value(binary_mask, 0, 1)
    return binary_mask


def normalize_ncct_scan(ncct_scan):
    """
    Normalize NCCT scan to the range [0, 1].
    Assumes the input is a numpy array.
    """
    # Clip values to the typical Hounsfield Unit (HU) range for brain CT scans
    hu_min = -1000  # Air
    hu_max = 1000   # Bone
    ncct_scan = tf.clip_by_value(ncct_scan, hu_min, hu_max)
    
    # Normalize to [0, 1]
    ncct_scan = (ncct_scan - hu_min) / (hu_max - hu_min)
    return ncct_scan

def normalize(ncct_slice, lesion_mask_slice):
    ncct_slice = normalize_ncct_scan(ncct_slice)  # Normalize NCCT scan
    lesion_mask_slice = normalize_binary_mask(lesion_mask_slice)  # Normalize binary mask
    return ncct_slice, lesion_mask_slice

def adaptive_windowing(img, percentile_low=5, percentile_high=95):
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

def create_dataset(ncct_paths, mask_paths):
    # List to store each pair of (NCCT slice, Mask slice)
    ncct_slice_list = []  
    lesion_mask_slice_list = [] 

    # Wrap the zip in tqdm to show progress
    for ncct_path, mask_path in tqdm(zip(ncct_paths, mask_paths), total=len(ncct_paths), desc="Processing Files"):
        # Load the NIfTI files
        ncct_img = nib.load(ncct_path)
        mask_img = nib.load(mask_path)

        # Convert to numpy arrays or other necessary processing
        ncct_data = ncct_img.get_fdata()
        mask_data = mask_img.get_fdata()
        
        # Ensure both have the same number of slices
        if ncct_data.shape[2] != mask_data.shape[2]:
            print(f"Shape mismatch: NCCT has {ncct_data.shape[2]} slices, mask has {mask_data.shape[2]} slices")
            continue
        
        # Find slices containing lesions
        lesion_slices = [i for i in range(mask_data.shape[2]) if np.any(mask_data[:, :, i] == 1)]
        
        # Handle case where no lesion slices were found
        if not lesion_slices:
            continue
        
        # Iterate through lesion slices with a progress bar
        for slice_index in lesion_slices:
            # Access the slice from the mask
            lesion_mask_slice = mask_data[:, :, slice_index]
            
            # Access the slice from the NCCT
            ncct_slice = ncct_data[:, :, slice_index]
            
            # Apply adaptive windowing
            windowed_img = adaptive_windowing(ncct_slice)
            
            # Optional: Apply Canny edge detection if you want to use it as an additional feature
            edges = apply_canny(windowed_img)
            
            # Add a channel dimension (1 channel for grayscale images)
            lesion_mask_slice = np.expand_dims(lesion_mask_slice, axis=-1)
            ncct_slice = np.expand_dims(ncct_slice, axis=-1)
            
            # Resize to (128, 128)
            lesion_mask_slice = tf.image.resize(lesion_mask_slice, (128, 128))
            ncct_slice = tf.image.resize(ncct_slice, (128, 128))
        
            # print("Before normalization:", np.min(ncct_slice), np.max(ncct_slice))
            # print("Before normalization:", np.min(lesion_mask_slice), np.max(lesion_mask_slice))
        
            # Normalize the slices
            ncct_slice, lesion_mask_slice = normalize(ncct_slice, lesion_mask_slice)
        
            # print("After normalization:", np.min(ncct_slice), np.max(ncct_slice))
            # print("After normalization:", np.min(lesion_mask_slice), np.max(lesion_mask_slice))
            
            # Append the slices to the lists
            ncct_slice_list.append(ncct_slice)
            lesion_mask_slice_list.append(lesion_mask_slice)

    # Create a TensorFlow Dataset from the slices
    dataset = tf.data.Dataset.from_tensor_slices((ncct_slice_list, lesion_mask_slice_list))

    return dataset

# Function to display an image and its mask
def show_sample(dataset):
    sample = next(iter(dataset.shuffle(1000).take(1)))  # Take a random sample
    
    ncct_image = sample[0].numpy()  # Extract NCCT image
    mask = sample[1].numpy()  # Extract mask
    
    # Plot the images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(ncct_image.squeeze(), cmap='gray')  # Assuming grayscale NCCT
    ax[0].set_title("NCCT Image")
    ax[0].axis("off")
    
    ax[1].imshow(mask.squeeze(), cmap='jet', alpha=0.7)  # Use 'jet' colormap for mask
    ax[1].set_title("Segmentation Mask")
    ax[1].axis("off")

    plt.show()

# Augments the dataset vertically

def augment_vertical_flip(image, label):
    # Perform vertical flip on the image and label
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        label = tf.image.flip_up_down(label)
    return image, label

def augment_horizontal_flip(image, mask):
    input_image = tf.image.flip_left_right(image)
    input_mask = tf.image.flip_left_right(mask)
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def augment_contrast(image, mask):
    input_image = tf.image.adjust_contrast(image, contrast_factor=2)
    input_mask = mask  # Contrast does not affect the mask
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def augment_random_brightness(image, mask):
    input_image = tf.image.random_brightness(image, max_delta=0.1)  # Adjust brightness by up to 10%
    input_mask = mask  # Brightness does not affect the mask
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def augment_random_saturation(image, mask, lower=0.5, upper=1.5):
    input_image = image
    input_mask = mask
    
    # Apply random saturation to the image
    input_image = tf.image.random_saturation(input_image, lower, upper)
    
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

train_dataset = create_dataset(ncct_train, mask_train)    
test_dataset = create_dataset(ncct_test, mask_test)  

# show_sample(train_dataset)
# Get the cardinality of the dataset, which gives the number of samples
sample_count = train_dataset.cardinality().numpy()
print(f"Total samples: {sample_count}")

train_dataset_vflip = train_dataset.map(augment_vertical_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# train_dataset_contrast = train_dataset.map(augment_contrast, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# train_dataset_rbrightness = train_dataset.map(augment_random_brightness, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset_hflip = train_dataset.map(augment_horizontal_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# train_dataset_rsaturate = train_dataset.map(augment_random_saturation, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_dataset_combined = train_dataset.concatenate(train_dataset_vflip)
# train_dataset_combined = train_dataset_combined.concatenate(train_dataset_contrast)
# train_dataset_combined = train_dataset_combined.concatenate(train_dataset_rbrightness)
train_dataset_combined = train_dataset_combined.concatenate(train_dataset_hflip)
# train_dataset_combined = train_dataset_combined.concatenate(train_dataset_rsaturate)

print("shape checker")
for ncct, mask in train_dataset_combined.take(2):
    print("NCCT shape:", ncct.shape)
    print("Mask shape:", mask.shape)

# Get the cardinality of the dataset, which gives the number of samples
train_dataset_combined_count = train_dataset_combined.cardinality().numpy()
test_dataset_count = test_dataset.cardinality().numpy()

print(f"Total samples after aug: {train_dataset_combined_count}")
show_sample(train_dataset_combined)

BATCH_SIZE = 16
BUFFER_SIZE = 1000

train_dataset_combined = train_dataset_combined.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset_combined = train_dataset_combined.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_dataset = test_dataset.batch(BATCH_SIZE)

# Defines the parts of the unet model 

def double_conv_block(x, n_filters):
    x = layers.Conv2D(n_filters, 3, padding ='same', activation = 'relu', kernel_initializer = 'he_normal')(x)  
    x = layers.Conv2D(n_filters, 3, padding ='same', activation = 'relu', kernel_initializer = 'he_normal')(x)
    return x

def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p

def upsample_blocks(x, conv_features, n_filters):
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding = 'same')(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)
    return x

# Builds the actually model that the image is put through

def build_unet_model(output_channels):
    
    # input
    
    inputs = layers.Input(shape = (128, 128, 1))
    
    # encoder - downsample
    
    f1, p1 = downsample_block(inputs, 64)
    p1 = layers.Dropout(0.1)(p1)  
    f2, p2 = downsample_block(p1, 128)
    p2 = layers.Dropout(0.2)(p2)
    f3, p3 = downsample_block(p2, 256)
    p3 = layers.Dropout(0.3)(p3)
    f4, p4 = downsample_block(p3, 512)
    p4 = layers.Dropout(0.4)(p4)
    
    # Intermediate/Bottle neck block
    
    intermediate_block = double_conv_block(p4, 1024)
    intermediate_block = layers.Dropout(0.5)(intermediate_block)
    
    # decoder - upsample
    
    u6 = upsample_blocks(intermediate_block, f4, 512)
    u6 = layers.Dropout(0.4)(u6)
    u7 = upsample_blocks(u6, f3, 256)
    u7 = layers.Dropout(0.3)(u7)
    u8 = upsample_blocks(u7, f2, 128)
    u8 = layers.Dropout(0.2)(u8)
    u9 = upsample_blocks(u8, f1, 64)
    u9 = layers.Dropout(0.1)(u9)
    
    # output layer
    
    outputs = layers.Conv2D(output_channels, 1, padding = 'same', activation = 'sigmoid')(u9)
    
    # unet model
    
    unet_model = tf.keras.Model(inputs, outputs, name = 'u-Net')
    
    return unet_model

# Build U-Net model for binary segmentation
output_channels = 1  # Change to 1 for binary segmentation
model = build_unet_model(output_channels)

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

# print("model summary")
# for sample in train_dataset_combined.take(5):
#     print(sample)  # Check if the slices are valid

# Trains the model Specify GPU being used by wtf.device context manager

# Train the model
with tf.device('/GPU:6'):
    EPOCHS = 20
    steps_per_epoch = train_dataset_combined.cardinality().numpy() // BATCH_SIZE
    validation_steps = test_dataset.cardinality().numpy() // BATCH_SIZE
    history = model.fit(train_dataset_combined,
                        epochs=EPOCHS,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        validation_data=test_dataset)

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

# Creates a tensor repersenting the models prediction of a given image and then sends it to be printed

def show_predications(dataset=None, num=1):
    number = 0
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display_sample([image[0], mask[0], create_mask(pred_mask)], "images", number)
            number += 1  
            
show_predications(test_dataset, 10)