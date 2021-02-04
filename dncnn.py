import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from utils import show, scale


class DnCNN(keras.Model):
    def __init__(self):
        super(DnCNN, self).__init__()

    def build(self, input_shape):
        self.model = keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="SAME"),
            keras.layers.LeakyReLU(0.2),
        ])
        for _ in range(16):
            self.model.add(keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="SAME"))
            self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.LeakyReLU(0.2))
        self.model.add(keras.layers.Conv2D(filters=input_shape[-1], kernel_size=3, strides=1, padding="SAME"))

    def call(self, inputs, training=None, mask=None):
        noise = self.model(inputs)
        return inputs - noise


(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = scale(x_train)[..., np.newaxis]
x_test = scale(x_test)[..., np.newaxis]

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape, stddev=2.0)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape, stddev=4.0)

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=-1., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=-1., clip_value_max=1.)

model = DnCNN()
model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5),
              loss=keras.losses.MeanSquaredError())
model.fit(x_train_noisy, x_train_noisy - x_train, epochs=10, shuffle=True, validation_data=(x_test_noisy, x_test_noisy - x_test))
show(x_test_noisy[:16], x_test_noisy[:16]-model(x_test_noisy[:16]))
