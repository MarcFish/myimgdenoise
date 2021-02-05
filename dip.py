import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from tensorflow.keras.datasets import fashion_mnist, mnist
import numpy as np
from utils import show, scale


class DIP(keras.Model):
    def __init__(self, img_shape):
        super(DIP, self).__init__()
        self.img_shape = img_shape

    def build(self, input_shape):
        self.gen = keras.Sequential([
            keras.layers.InputLayer(input_shape=input_shape),
            keras.layers.Dense(7 * 7 * 256),
            keras.layers.Reshape((7, 7, 256)),
            keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding="SAME",
                                         kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding="SAME",
                                         kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2DTranspose(filters=self.img_shape[-1], kernel_size=5, strides=1, padding="SAME",
                                         kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),
                                         activation="tanh")
        ])
        self.built = True

    def call(self, inputs, training=None, mask=None):
        return self.gen(inputs)


(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = scale(x_train)[..., np.newaxis]
x_test = scale(x_test)[..., np.newaxis]

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape, stddev=2.0)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape, stddev=4.0)

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=-1., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=-1., clip_value_max=1.)

noise_input = np.random.uniform(size=(x_train.shape[0], 100))
model = DIP(x_train.shape[1:])
model.build(noise_input.shape[1:])
model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5),
              loss=keras.losses.MeanSquaredError())
model.fit(noise_input, x_train_noisy, epochs=10, shuffle=True)
show(x_train_noisy[:16], model(noise_input[:16]))
