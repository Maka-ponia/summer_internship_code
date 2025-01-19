import tensorflow as tf

# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

# Check if TensorFlow can use the GPU
if len(gpus) > 0:
    print(f"Using {len(gpus)} GPU(s)")
else:
    print("No GPUs detected.")
