import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

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
        input_mask = input_mask - 1
        return input_image, input_mask
    except Exception as e:
        print(f"Skipping corrupted data during normalization: {e}")
        return None, None

def load_train_images(sample):
    try:
    
    # resize the image to 128X128
        
        input_image = tf.image.resize(sample['image'], (128, 128))
        input_mask = tf.image.resize(sample['segmentation_mask'], (128, 128))

    # data augmentation, randomaly flips an image and mask horizonatlly 

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)
            
    # normalize the images and masks

        input_image, input_mask = normalize(input_image, input_mask)
        return input_image, input_mask
    except Exception as e:
        print(f"Skipping corrupted file during training data loading: {e}")
        return None  
    
    # Return None for invalid files

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

# Itterates through the dataset and applies the load functions two each data point, 
# which is then placed in another arary

train_dataset = dataset['train'].map(load_train_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = dataset['test'].map(load_test_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Establishes that 64 examples from the dataset will be 
# proccessed at a time during training or testing. Establishes 
# that 1000 datapoints will be stored to be shuffled 

BATCH_SIZE = 16
BUFFER_SIZE = 1000

# stores the dataset in a cache after the first read, shuffles it and then stoes then in a batch by an amount repatatly 
# Grabs data whil data is still being proccesed
train_dataset = train_dataset.cache().shuffle(BATCH_SIZE).batch(BATCH_SIZE).repeat()    
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# explanatory Data Analysis

def display_sample(image_list):
    plt.figure(figsize=(10,10))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    
    for i in range(len(image_list)):
        plt.subplot(1, len(image_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(image_list[i]))
        plt.axis('off')
        
    plt.show()
    
#Define

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
    
    inputs = layers.Input(shape = (128, 128, 3))
    
    # encoder - downsample
    
    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)
    
    # Intermediate/Bottle neck block
    
    intermediate_block = double_conv_block(p4, 1024)
    
    # decoder - upsample
    
    u6 = upsample_blocks(intermediate_block, f4, 512)
    u7 = upsample_blocks(u6, f3, 256)
    u8 = upsample_blocks(u7, f2, 128)
    u9 = upsample_blocks(u8, f1, 64)
    
    # output layer
    
    outputs = layers.Conv2D(output_channels, 1, padding = 'same', activation = 'softmax')(u9)
    
    # unet model
    
    unet_model = tf.keras.Model(inputs, outputs, name = 'u-Net')
    
    return unet_model

# for images, masks in train_dataset.take(3):
#     sample_image, sample_mask = images[0], masks[0]
#     #display_sample([sample_image, sample_mask])
#     print(sample_mask)
#     sample_mask(60)


# Define the Dice coefficient
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Compute the Dice coefficient.
    
    Parameters:
    - y_true: Ground truth labels.
    - y_pred: Predicted labels.
    - smooth: A smoothing factor to avoid division by zero.
    
    Returns:
    - Dice coefficient value.
    """
    # Flatten tensors for calculation
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    
    # Compute the intersection and union
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f)
    
    # Return Dice coefficient
    return (2. * intersection + smooth) / (union + smooth)
    
output_channels = 3
model = build_unet_model(output_channels)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', dice_coefficient])

# plot the model doesnt work but seems not important

# tf.keras.utils.plot_model(model, show_shapes = True, expand_nested = False, dpi = 64)

# Train the model

# Create the MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()

# Print the number of devices (GPUs) being used.
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    EPOCHS = 20
    steps_per_epoch = info.splits['train'].num_examples // BATCH_SIZE
    validation_steps = info.splits['test'].num_examples // BATCH_SIZE 
    history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_data=test_dataset)