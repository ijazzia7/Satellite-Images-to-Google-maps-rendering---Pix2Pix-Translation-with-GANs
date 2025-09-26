# Pix2Pix: Image-to-Image Translation using Conditional GANs

This repository provides a TensorFlow/Keras implementation of **Pix2Pix**, a conditional Generative Adversarial Network (cGAN) for image-to-image translation.

Pix2Pix learns a mapping from input images to output images using paired datasets. For example:

- Satellite images to map renderings
- Sketches to realistic photographs
- Daytime to nighttime scenes

The implementation in this repository is demonstrated on the [Pix2Pix Maps Dataset](https://www.kaggle.com/datasets/vikramtiwari/pix2pix-maps), but it can be adapted to any paired image dataset.

---

## Key Features

- **Data Preprocessing**: Splits paired images into source (input) and target (output).
- **Discriminator (PatchGAN)**: A convolutional classifier that determines whether a given image pair (source and target) is real or generated.
- **Generator (U-Net)**: Produces translated images using an encoder–decoder architecture with skip connections for preserving fine details.
- **Adversarial Training**: Combines binary cross-entropy loss with L1 reconstruction loss for stable and realistic outputs.
- **Evaluation and Visualization**: Saves intermediate outputs and model checkpoints during training.
- **Inference**: Apply trained models to new, unseen input images.

---

## Model Architecture

- **Discriminator**: PatchGAN model that classifies image patches as real or fake.
- **Generator**: U-Net encoder–decoder with skip connections to ensure both global coherence and local detail.
- **GAN (Combined Model)**: Trains the generator and discriminator jointly, balancing adversarial and reconstruction objectives.

---

## Training

The training process consists of:

1. Loading and normalizing paired datasets.
2. Training the discriminator on real and generated pairs.
3. Training the generator through the combined GAN model.
4. Saving generated images and model weights periodically.
