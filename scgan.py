import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
from tensorflow.keras.datasets import fashion_mnist, mnist
import numpy as np
from utils import show, scale


class ScGAN(keras.Model):
    def __init__(self):
        super(ScGAN, self).__init__()

    def build(self, input_shape):
        self.gen = keras.Sequential([
            keras.layers.InputLayer(input_shape=input_shape),
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="SAME"),
            keras.layers.LeakyReLU(0.2),
        ])
        for _ in range(16):  # 16 in original;
            self.gen.add(keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="SAME"))
            self.gen.add(keras.layers.BatchNormalization())
            self.gen.add(keras.layers.LeakyReLU(0.2))
        self.gen.add(keras.layers.Conv2D(filters=input_shape[-1], kernel_size=3, strides=1, padding="SAME"))

        self.dis = keras.Sequential([
            keras.layers.InputLayer(input_shape=input_shape),
            keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding="SAME"),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2D(filters=128, kernel_size=5, strides=2, padding="SAME"),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="SAME"),
            keras.layers.LeakyReLU(0.2),
            keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding="SAME"),
        ])

        self.w1 = self.add_weight(shape=(1,), initializer=keras.initializers.Zeros(),
                                  constraint=keras.constraints.NonNeg())
        self.w2 = self.add_weight(shape=(1,), initializer=keras.initializers.Zeros(),
                                  constraint=keras.constraints.NonNeg())
        self.w3 = self.add_weight(shape=(1,), initializer=keras.initializers.Zeros(),
                                  constraint=keras.constraints.NonNeg())

        self.built = True

    def call(self, inputs, training=None, mask=None):
        return inputs - self.gen(inputs)

    def _gen_loss(self, real, fake):
        return self.compiled_loss(tf.ones_like(fake), fake)

    def _dis_loss(self, real, fake):
        real_loss = self.compiled_loss(tf.ones_like(real), real)
        fake_loss = self.compiled_loss(tf.zeros_like(fake), fake)
        return real_loss + fake_loss

    def _clean_loss(self, clean):
        return self.compiled_loss(clean, tf.zeros_like(clean))

    def _pn_loss(self, noise, nnoise):
        return self.compiled_loss(nnoise, noise)

    def _rec_loss(self, noise, cnoise):
        return self.compiled_loss(cnoise, noise)

    def compile(self, d_optimizer, g_optimizer, **kwargs):
        super(ScGAN, self).compile(**kwargs)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def train_step(self, data):
        clean_img, noise_img = data
        with tf.GradientTape(persistent=True) as tape:
            noise = self.gen(noise_img)
            rec_img = noise_img - noise
            clean_img_noise = clean_img + noise

            clean_loss = self._clean_loss(clean_img)
            pn_loss = self._pn_loss(noise, self.gen(noise))
            rec_loss = self._rec_loss(noise, self.gen(clean_img_noise))
            real = self.dis(clean_img)
            fake = self.dis(rec_img)
            gen_loss = self._gen_loss(real, fake) + self.w1 * clean_loss + self.w2 * pn_loss + self.w3 * rec_loss
            dis_loss = self._dis_loss(real, fake) + self.w1 * clean_loss + self.w2 * pn_loss + self.w3 * rec_loss

        gen_gradients = tape.gradient(gen_loss, self.gen.trainable_variables)
        dis_gradients = tape.gradient(dis_loss, self.dis.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradients, self.gen.trainable_variables))
        self.d_optimizer.apply_gradients(zip(dis_gradients, self.dis.trainable_variables))

        return {"gen_loss": gen_loss, "dis_loss": dis_loss}


(x_train, _), (x_test, _) = fashion_mnist.load_data()
(y_train, _), (y_test, _) = mnist.load_data()
x_train = scale(x_train)[..., np.newaxis]
x_test = scale(x_test)[..., np.newaxis]
y_train = scale(y_train)[..., np.newaxis]
y_test = scale(y_test)[..., np.newaxis]

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape, stddev=2.0)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape, stddev=4.0)

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=-1., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=-1., clip_value_max=1.)

model = ScGAN()
model.build(x_train.shape[1:])
model.compile(d_optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5),
              g_optimizer=tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=5e-5),
              loss=keras.losses.MeanSquaredError())
model.fit(x_train, x_train_noisy, epochs=10, shuffle=True)  # TODO: split three epoch when fit
show(x_test_noisy[:16], model(x_test_noisy[:16]))
