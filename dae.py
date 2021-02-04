import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from utils import show, scale


class DenoiseAutoencoder(keras.Model):
    def __init__(self):
        super(DenoiseAutoencoder, self).__init__()

    def build(self, input_shape):
        self.encoder = keras.Sequential([
            keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
        ])
        self.decoder = keras.Sequential([
            keras.layers.Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=2, padding="SAME"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2, padding="SAME"),
        ])

    def call(self, inputs, training=None, mask=None):
        encode = self.encoder(inputs)
        decode = self.decoder(encode)
        return decode


(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = scale(x_train)[..., np.newaxis]
x_test = scale(x_test)[..., np.newaxis]

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape, stddev=2.0)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape, stddev=2.0)

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=-1., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=-1., clip_value_max=1.)

model = DenoiseAutoencoder()
model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5),
              loss=keras.losses.MeanSquaredError())
model.fit(x_train_noisy, x_train, epochs=10, shuffle=True, validation_data=(x_test_noisy, x_test))
show(x_test_noisy, model(x_test_noisy))
