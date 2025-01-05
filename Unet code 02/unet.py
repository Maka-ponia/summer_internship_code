import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

dataset, info = tfds.load('oxford_iiit_pet', with_info = True)

#Preprocessing Steps

def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = input_mask - 1 # Convert to zero based indexing
    return input_image, input_mask

def load_train_images(sample):
   
    # resize the image
   
    input_image = tf.image.resize(sample['image'], (128, 128))
    input_mask = tf.image.resize(sample['segmentation_mask'], (128, 128))
   
    # data augmentation
   
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    
    # normalize the images
    
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

def load_test_images(sample):
   
    # resize the image
   
    input_image = tf.image.resize(sample['image'], (128, 128))
    input_mask = tf.image.resize(sample['segmentation_mask'], (128, 128))
    
    # normalize the images
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask

train_dataset = dataset['train'].map(load_train_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = dataset['test'].map(load_test_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)

BATCH_SIZE = 64
BUFFER_SIZE = 1000

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
    
def build_unet_model(output_channels):
    
    # input
    
    inputs = layers.Input(shape = (128, 128, 3))
    
    # encoder - downsample
    
    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)
    
    # Intermediate block
    
    intermediate_block = double_conv_block(p4, 1024)
    
    # decoder - upsample
    
    u6 = upsample_blocks(intermediate_block, f4, 512)
    u7 = upsample_blocks(u6, f3, 256)
    u8 = upsample_blocks(u7, f2, 128)
    u9 = upsample_blocks(u8, f1, 64)
    
    # output layer
    
    outputs = layers.Conv2D(output_channels, 1, padding = 'same', activation = 'relu')(u9)
    
    # unet model
    
    unet_model = tf.keras.Model(inputs, outputs, name = 'u-Net')
    
    return unet_model

# for images, masks in train_dataset.take(3):
#     sample_image, sample_mask = images[0], masks[0]
#     #display_sample([sample_image, sample_mask])
#     print(sample_mask)
#     sample_mask(60)
    
output_channels = 3
model = build_unet_model(output_channels)
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# plot the model doesnt work but seems not important

# tf.keras.utils.plot_model(model, show_shapes = True, expand_nested = False, dpi = 64)

# Train the model

EPOCHS = 20

steps_per_epoch = info.splits['train'].num_examples // BATCH_SIZE
validation_steps = info.splits['test'].num_examples // BATCH_SIZE 

history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_data=test_dataset)