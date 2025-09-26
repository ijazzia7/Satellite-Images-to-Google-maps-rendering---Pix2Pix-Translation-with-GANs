import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, BatchNormalization, Flatten, Concatenate, Reshape, Embedding, Multiply
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Dropout
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.models import Model
from keras.preprocessing.image import load_img

# ---------------------------
# Data Loading
# ---------------------------
def load_images(path):
    source_imgs, target_imgs = [], []
    imgs = os.listdir(path)
    for i in tqdm(range(len(imgs))):
        arr = np.array(load_img(f'{path}/{imgs[i]}', target_size=(256, 512)))
        source_imgs.append(arr[:, :256, :])
        target_imgs.append(arr[:, 256:, :])
    return np.array(source_imgs), np.array(target_imgs)

def load_real_samples(train_s, train_t):
    train_s = (train_s - 127.5) / 127.5
    train_t = (train_t - 127.5) / 127.5
    return [train_s, train_t]

# ---------------------------
# Model Components
# ---------------------------
def build_discriminator(image_shape):
    init = RandomNormal(stddev=0.02, seed=90)
    inp_s = Input(shape=image_shape)
    inp_t = Input(shape=image_shape)
    merged = Concatenate()([inp_s, inp_t])

    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d); d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d); d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d); d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d); d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    model = Model([inp_s, inp_t], patch_out)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

def encoding_block(inp, filters, bat=True):
    init = RandomNormal(stddev=0.02)
    d = Conv2D(filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(inp)
    if bat:
        d = BatchNormalization()(d, training=True)
    d = LeakyReLU(alpha=0.2)(d)
    return d

def decoding_block(inp, skip_in, filters, dropout=True):
    init = RandomNormal(stddev=0.02)
    u = Conv2DTranspose(filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(inp)
    u = BatchNormalization()(u, training=True)
    if dropout:
        u = Dropout(0.5)(u)
    u = Concatenate()([u, skip_in])
    u = Activation('relu')(u)
    return u

def build_generator(image_shape):
    init = RandomNormal(stddev=0.02)
    inp_s = Input(shape=image_shape)
    d1 = encoding_block(inp_s, 64, bat=False)
    d2 = encoding_block(d1, 128)
    d3 = encoding_block(d2, 256)
    d4 = encoding_block(d3, 512)
    d5 = encoding_block(d4, 512)
    d6 = encoding_block(d5, 512)
    d7 = encoding_block(d6, 512)

    bottleneck = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    bottleneck = Activation('relu')(bottleneck)

    u1 = decoding_block(bottleneck, d7, 512)
    u2 = decoding_block(u1, d6, 512)
    u3 = decoding_block(u2, d5, 512)
    u4 = decoding_block(u3, d4, 512, dropout=False)
    u5 = decoding_block(u4, d3, 256, dropout=False)
    u6 = decoding_block(u5, d2, 128, dropout=False)
    u7 = decoding_block(u6, d1, 64, dropout=False)

    last = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(u7)
    out_image = Activation('tanh')(last)
    return Model(inp_s, out_image)

def build_gan(g_model, d_model, image_shape):
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    in_src = Input(shape=image_shape)
    gen_out = g_model(in_src)
    dis_out = d_model([in_src, gen_out])
    model = Model(in_src, [dis_out, gen_out])
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return model

# ---------------------------
# Training Helpers
# ---------------------------
def generate_real_samples(dataset, n_samples, patch_shape):
    train_s, train_t = dataset
    ix = np.random.randint(0, train_s.shape[0], n_samples)
    x1, x2 = train_s[ix], train_t[ix]
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return [x1, x2], y

def generate_fake_samples(generator, samples, patch_shape):
    fake = generator.predict(samples, verbose=0)
    y = np.zeros((len(fake), patch_shape, patch_shape, 1))
    return fake, y

def summarize_performance(step, g_model, dataset, n_samples=3):
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + i); plt.axis('off'); plt.imshow(X_realA[i])
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i); plt.axis('off'); plt.imshow(X_fakeB[i])
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples*2 + i); plt.axis('off'); plt.imshow(X_realB[i])
    filename1 = f'plot_{step+1:06d}.png'
    plt.savefig(filename1); plt.close()
    filename2 = f'model_{step+1:06d}.keras'
    g_model.save(filename2)
    print(f'>Saved: {filename1} and {filename2}')

def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    n_patch = d_model.output_shape[1]
    trainA, trainB = dataset
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs
    for i in tqdm(range(n_steps)):
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        d_model.train_on_batch([X_realA, X_realB], y_real)
        d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        gan_model.train_on_batch(X_realA, [y_real, X_realB])
        if (i+1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    train_s, train_t = load_images('/kaggle/input/pix2pix-maps/train')
    val_s, val_t = load_images('/kaggle/input/pix2pix-maps/val')
    dataset = load_real_samples(train_s, train_t)
    image_shape = (256, 256, 3)
    d_model = build_discriminator(image_shape)
    g_model = build_generator(image_shape)
    gan_model = build_gan(g_model, d_model, image_shape)
    train(d_model, g_model, gan_model, dataset, n_epochs=200)
