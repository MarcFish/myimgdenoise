import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def show(real, fake, n=16):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(real[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(fake[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def scale(image):
    image = image.astype(np.float32)
    return (image - 127.5) / 127.5


def rescale(image):
    return (image * 127.5 + 127.5).astype(np.uint8)


def process_numpy(array, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(array).shuffle(512).batch(batch_size=batch_size, drop_remainder=True).map(_process,tf.data.experimental.AUTOTUNE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def _process(image):
    img = tf.cast(image, tf.float32)
    img = scale(img)
    img = apply_augment(img)
    return img


def apply_augment(image: tf.Tensor) -> tf.Tensor:
    image = tf.image.random_brightness(image, 0.5)
    image = tf.image.random_saturation(image, 0.0, 2.0)
    image = tf.image.random_contrast(image, 0.0, 1.0)
    # image = tf.image.random_flip_left_right(image)
    return image
