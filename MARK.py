import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from Parameters import cmaxlen, dmaxlen, cndims, afunction, input_stddev, weights_file


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class MARK(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()

        global cmaxlen, dmaxlen, cndims, afunction, weights_file

        self.cmaxlen = cmaxlen
        self.dmaxlen = dmaxlen
        self.tmaxlen = cmaxlen + dmaxlen
        self.cndims = cndims
        self.input_stddev = input_stddev

        self.context_layer = layers.Dense(cndims, activation='tanh')
        self.noise_stddev = layers.Dense(1, activation='tanh')

        self.core_layers = [ # cndims + dmaxlen <= 1024
            layers.Dense(1024, activation=afunction),
            layers.Dense(1024, activation=afunction),
            layers.Dense(1024, activation=afunction),
            layers.Dense(1024, activation=afunction),
            layers.Dense(1024, activation=afunction),
            layers.Dense(1024, activation=afunction),
            layers.Dense(1024, activation=afunction),
            layers.Dense(256, activation='softmax')
        ]

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self(np.zeros((1, self.tmaxlen), dtype=np.float32))

        self.summary()

        if weights_file != '':
            self.load_weights(weights_file)

        self.interact_text = ""

    def call(self, x):
        x = tf.random.normal(shape=x.shape, mean=x, stddev=self.input_stddev)
        c, x = x[:, :self.cmaxlen], x[:, self.cmaxlen:]
        assert x.shape[1] == self.dmaxlen

        batch_size = x.shape[0]

        c = self.context_layer(c)

        x = tf.concat([c, x], axis=1)
        nse = self.noise_stddev(x)

        assert x.shape[1] == self.dmaxlen + self.cndims

        for idx in range(len(self.core_layers)):
            x = tf.concat([x, tf.random.normal(shape=[batch_size, 1], stddev=nse)], axis=1)
            x = self.core_layers[idx](x)

        return x

    @tf.function(jit_compile=True)
    def fitstep(self, X, Y):
        with tf.GradientTape() as tape:
            out = self(X)
            loss = self.loss(Y, out)

        g = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(g, self.trainable_variables))
        return loss

    def vtop(self, x):
        return x * (x > 0)

    def text2array(self, text):
        text = ''.join([char for char in text if ord(char) < 256])
        text = text[self.vtop(len(text)-self.tmaxlen):]
        text = ' ' * (self.tmaxlen - len(text)) + text
        return [float(ord(char)) for char in text]

    def fit(self, textdir, epochs, lr, epsilon=1e-7):
        self.opt = tf.keras.optimizers.Adam(lr, epsilon=epsilon)

        with open(textdir, 'r') as file:
            text = ''.join(file.readlines())

        text = ''.join([char for char in text if ord(char) < 256])

        X = np.zeros((len(text), self.tmaxlen), dtype=np.float32)
        Y = np.zeros(len(text), dtype=np.uint8)

        print(f'\nEntrenando con {X.shape[0]} ejemplos...\n')

        for idx in range(len(text)):
            X[idx], Y[idx] = self.text2array(text[self.vtop(idx-self.tmaxlen):idx]), ord(text[idx])

        for ep in range(1, epochs + 1, 1):
            print(f'Epoch {ep}/{epochs} | Loss {self.fitstep(X, Y)}')

        self.save_weights('model.h5')

    def interact(self, query, length, live_mode=False):
        self.interact_text += query
        output_text = ""
        char = ''

        if type(length) == str:
            while not char == length:
                self.interact_text = self.interact_text[self.vtop(len(self.interact_text) - self.tmaxlen):]
                X = np.array(self.text2array(self.interact_text), dtype=np.float32)[np.newaxis, ...]
                Y = self(X)[0, :].numpy()
                char = chr(Y.argmax())
                self.interact_text += char
                output_text += char

                if live_mode:
                    print(char, end='')

        elif type(length) == int:
            for i in range(length):
                self.interact_text = self.interact_text[self.vtop(len(self.interact_text) - self.tmaxlen):]
                X = np.array(self.text2array(self.interact_text), dtype=np.float32)[np.newaxis, ...]
                Y = self(X)[0, :].numpy()
                char = chr(Y.argmax())
                self.interact_text += char
                output_text += char

                if live_mode:
                    print(char, end='')
        else:
            raise AttributeError('Para length solo son aceptados valores str o int!')

        return output_text

