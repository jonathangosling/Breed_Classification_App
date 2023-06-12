import tensorflow as tf

# Define image size
# Ensure image size is of the correct size for the model
# (same as size used in training model)
IMG_SIZE = 224


def process_image(image_content):
    """
    Turns image into a Tensor
    """
    # Turn image into numerical Tensor with 3 colour channels (RGB)
    image = tf.image.decode_jpeg(image_content, channels=3)
    # Convert colour channel values from 0-255 to 0-1 values
    # (RGB values are 0-255 as standard)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize the image
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image


def create_data_batches(X, y=None, batch_size=1):
    """
    Creates batches of data from image (X) and label (y) pairs.
    Shuffles data if training data but not if validation data.labels.
    Accepts data with no labels (test_data).
    """
    print("Creating data batches...")
    # Create Dataset
    data = tf.data.Dataset.from_tensor_slices(X)
    # Turn Dataset into batches
    # .map applies the image processing function to all images
    # in Dataset .batch creates batches
    data_batch = data.map(process_image).batch(batch_size)
    return data_batch


def get_image_label(image_path, label):
    """
    Creates tuple of (image, label) from input image filepath and label
    """
    image = process_image(image_path)
    return image, label
