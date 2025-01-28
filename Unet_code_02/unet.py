import os
import matplotlib
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

os.environ["CUDA_VISIBLE_DEVICES"]="8"

# Check and configure GPUs

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

dataset, info = tfds.load('oxford_iiit_pet', with_info = True)

# Preprocessing Steps

def normalize(input_image, input_mask):
    
    # converts input_image from uint8 to tf.float32 and then divides each pixel value by 255.0, 
    # normalizing the pixel values between 0,1. Convert to zero based indexing
    
    try:
        input_image = tf.cast(input_image, tf.float32) / 255.0
        input_mask = tf.cast(input_mask, tf.float32)
        input_mask = input_mask - 1
        return input_image, input_mask
    except Exception as e:
        print(f"Skipping corrupted data during normalization: {e}")
        return None, None

def load_train_images(sample):
    try:
        # Resize the image and mask to 128x128
        
        input_image = tf.image.resize(sample['image'], (128, 128))
        input_mask = tf.image.resize(sample['segmentation_mask'], (128, 128))
        
        # Normalize images and masks
        
        input_image, input_mask = normalize(input_image, input_mask)
        return input_image, input_mask
    except Exception as e:
        print(f"Skipping corrupted file during training data loading: {e}")
        return None

def load_test_images(sample):
    
    # resize the image to 128X128
    
    try:
        input_image = tf.image.resize(sample['image'], (128, 128))
        input_mask = tf.image.resize(sample['segmentation_mask'], (128, 128))
    
    # normalize the images and masks
    
        input_image, input_mask = normalize(input_image, input_mask)
        return input_image, input_mask
    except Exception as e:
        print(f"Skipping corrupted file during testing data loading: {e}")
        return None  
    
    # Return None for invalid files

# Augments the dataset vertically

def augment_vertical_flip(sample):
    input_image = tf.image.flip_up_down(sample['image'])
    input_mask = tf.image.flip_up_down(sample['segmentation_mask'])
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

# Augments the dataset by using contrast

def augment_contrast(sample):
    input_image = tf.image.adjust_contrast(sample['image'], contrast_factor=2)
    input_mask = sample['segmentation_mask']  # Contrast does not affect the mask
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

# Augments the dataset by resizing 

def augment_random_brightness(sample):
    input_image = tf.image.random_brightness(sample['image'], max_delta=0.1)  # Adjust brightness by up to 10%
    input_mask = sample['segmentation_mask']  # Contrast does not affect the mask
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

# Augments the dataset horizontally 

def augment_horizontal_flip(sample):
    input_image = tf.image.flip_left_right(sample['image'])
    input_mask = tf.image.flip_left_right(sample['segmentation_mask'])
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

# Augment the random rotation 

def augment_random_rotation(sample, max_angle=30):
    # Extract the image and mask from the sample dictionary
    input_image = sample['image']
    input_mask = sample['segmentation_mask']
    
    # Rotate the image by a random angle
    angle = tf.random.uniform([], -max_angle, max_angle, dtype=tf.float32)
    input_image = tfa.image.rotate(input_image, angle)
    
    # Rotate the segmentation mask similarly
    input_mask = tfa.image.rotate(input_mask, angle)
    
    # Normalize the image and mask (optional)
    input_image, input_mask = normalize(input_image, input_mask)
    
    # Return the rotated image and mask
    return input_image, input_mask

# Augment the random saturation 

def augment_random_saturation(sample, lower=0.5, upper=1.5):
    # Extract the image and mask from the sample dictionary
    input_image = sample['image']
    input_mask = sample['segmentation_mask']
    
    # Apply random saturation to the image
    input_image = tf.image.random_saturation(input_image, lower, upper)
    
    # No need to modify the segmentation mask, it's only the image that gets augmented
    # Normalize if necessary
    input_image, input_mask = normalize(input_image, input_mask)
    
    # Return the augmented image and its corresponding mask
    return input_image, input_mask



# Itterates through the dataset and applies the load functions two each data point, 
# which is then placed in another arary

train_dataset_original = dataset['train'].map(load_train_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = dataset['test'].map(load_test_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Augmented datasets

train_dataset_vflip = dataset['train'].map(augment_vertical_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset_contrast = dataset['train'].map(augment_contrast, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset_resize = dataset['train'].map(augment_random_brightness, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset_hflip = dataset['train'].map(augment_horizontal_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# train_dataset_rrotate = dataset['train'].map(augment_random_rotation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset_rsaturate = dataset['train'].map(augment_random_saturation, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Combine all datasets into one

train_dataset_combined = train_dataset_original.concatenate(train_dataset_vflip)
train_dataset_combined = train_dataset_combined.concatenate(train_dataset_contrast)
train_dataset_combined = train_dataset_combined.concatenate(train_dataset_resize)
train_dataset_combined = train_dataset_combined.concatenate(train_dataset_hflip)
# train_dataset_combined = train_dataset_combined.concatenate(train_dataset_rrotate)
train_dataset_combined = train_dataset_combined.concatenate(train_dataset_rsaturate)


def resize_images(image, mask, target_size=(128, 128)):
    image = tf.image.resize(image, target_size)
    mask = tf.image.resize(mask, target_size)
    return image, mask

# Establishes that 64 examples from the dataset will be 
# proccessed at a time during training or testing. Establishes 
# that 1000 datapoints will be stored to be shuffled 

BATCH_SIZE = 16
BUFFER_SIZE = 1000

# stores the dataset in a cache after the first read, shuffles it and then stoes then in a batch by an amount repatatly 
# Grabs data whil data is still being proccesed
train_dataset_combined = train_dataset_combined.map(resize_images)
train_dataset_combined = train_dataset_combined.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset_combined = train_dataset_combined.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_dataset = test_dataset.batch(BATCH_SIZE)

# Prints sample

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

# Calculates the dice_coe 

def dice_coefficient(y_true, y_pred, smooth=1e-6): 
    y_pred = tf.nn.softmax(y_pred) 
    y_true_f = tf.keras.backend.flatten(tf.one_hot(tf.cast(y_true, tf.int32), depth=output_channels))
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

# Calculates the boundary_iou_loss

def boundary_iou_loss(y_true, y_pred):
    # Function to calculate the boundary of the mask (thin boundary).
    def boundary(mask):
        # Get the number of channels in the input (y_true or y_pred)
        channels = mask.shape[-1]  # Extract channel dimension (last dimension)
        
        # Laplacian kernel for boundary detection (3x3)
        
        kernel = tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=tf.float32)
        
        # Adjust kernel depth based on the number of channels in the input
        
        kernel = tf.expand_dims(kernel, axis=-1)  # Make it [3, 3, 1]
        
        # Tile the kernel along the last axis (depth) to match the number of channels in the input
        
        kernel = tf.tile(kernel, [1, 1, channels])  # Repeat kernel along the depth axis
        
        # Add the last dimension to make the kernel [3, 3, channels, 1]
        
        kernel = tf.expand_dims(kernel, axis=-1)  # Now shape [3, 3, channels, 1]
        
        # Cast mask to float32 and add batch dimension: [batch_size, height, width, channels]
        
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, axis=0)  # Add batch dimension: shape [1, height, width, channels]
        
        # Perform convolution to get the boundary
        
        boundary = tf.nn.conv2d(mask, kernel, strides=[1, 1, 1, 1], padding='SAME')
        boundary = tf.abs(boundary)  # Take absolute value for boundary pixels
        return boundary

    # Compute the boundaries for both true and predicted masks
    true_boundary = boundary(y_true)
    pred_boundary = boundary(y_pred)

    # Compute the intersection and union of the boundaries
    intersection = K.sum(true_boundary * pred_boundary)
    union = K.sum(true_boundary) + K.sum(pred_boundary) - intersection

    # Compute Boundary IoU as intersection over union
    boundary_iou = intersection / (union + K.epsilon())  # Adding epsilon to avoid division by zero

    return 1 - boundary_iou  # The loss is 1 minus the IoU (since we want to minimize the loss)

# Calculates the dice_loss

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
    
    y_true_f = tf.reshape(y_true, [-1, tf.shape(y_pred)[-1]])  
    y_pred_f = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])  
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    denominator = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
    dice_coeff = (2.0 * intersection + smooth) / (denominator + smooth)
    
    return 1.0 - tf.reduce_mean(dice_coeff)

# Defines combined loss function (sparse categorical + boundary IoU)

def combined_loss(y_true, y_pred):
    
    # Sparse Categorical Crossentropy loss for pixel-wise accuracy
    
    scce_loss = SparseCategoricalCrossentropy(from_logits=False)(y_true, y_pred)
    
    # Boundary IoU loss for boundary accuracy
    
    bdy_loss = boundary_iou_loss(y_true, y_pred)
    
    # Calculates the dice loss
    
    dice = dice_loss(y_true, y_pred)
    
    # Combine the two losses (adjust the weights if necessary)
    
    total_loss = 0.4 * scce_loss + 0.33 * bdy_loss + 0.33 * dice
    return total_loss

# Define the ReduceLROnPlateau callback

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',   # Monitor validation loss
    factor=0.5,           # Factor by which the learning rate will be reduced
    patience=5,           # Number of epochs with no improvement before reducing
    min_lr=1e-8           # Lower bound for the learning rate
)

# Builds the actually model that the image is put through   

def build_unet_model(output_channels):
    
    # input
    
    inputs = layers.Input(shape = (128, 128, 3))
    
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
    
    outputs = layers.Conv2D(output_channels, 1, padding = 'same', activation = 'softmax')(u9)
    
    # unet model
    
    unet_model = tf.keras.Model(inputs, outputs, name = 'u-Net')
    
    return unet_model

# configures the model for training by specifying its optimizer, loss function, and evaluation metrics

output_channels = 3
model = build_unet_model(output_channels)
model.compile(optimizer='adam',
              loss=combined_loss,
              metrics=['accuracy', dice_coefficient])

# Trains the model Specify GPU being used by wtf.device context manager

with tf.device('/GPU:8'):
    EPOCHS = 20
    steps_per_epoch = info.splits['train'].num_examples // BATCH_SIZE
    validation_steps = info.splits['test'].num_examples // BATCH_SIZE 
    history = model.fit(train_dataset_combined, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_data=test_dataset, callbacks=[reduce_lr])

# Takes a tensor and then apples the labels based on the values in the tensor to create the masks

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