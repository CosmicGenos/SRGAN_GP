# SRGAN: Super-Resolution Generative Adversarial Network

This repository contains an implementation of SRGAN (Super-Resolution Generative Adversarial Network) with some modifications to enhance performance and stability. The model is designed to upscale low-resolution images to high-resolution with impressive visual quality.

## Architecture Overview

### Generator

The Generator follows an architecture inspired by the original SRGAN paper, with:

- Initial convolutional layer with PReLU activation
- 16 residual blocks for deep feature extraction, each containing:
  - Convolutional layers with batch normalization
  - PReLU activation functions
  - Skip connections to facilitate gradient flow
- Post-residual convolution followed by a global skip connection
- Upsampling layers using PixelShuffle (2× upscaling in each layer for 4× total)
- Final convolutional layer with tanh activation

The generator effectively learns to transform low-resolution inputs into high-quality, high-resolution outputs while preserving important details and textures.

### Critic (Discriminator)

The Critic follows a WGAN architecture without using sigmoid activations:

- Series of convolutional blocks with increasing channel depth (64→512)
- Instance normalization instead of batch normalization
- LeakyReLU activations with slope 0.2
- Fully connected layers at the end to produce the final score

### Gradient Penalty

This implementation uses a Wasserstein GAN with Gradient Penalty (WGAN-GP) approach. The gradient penalty:

- Enforces the Lipschitz constraint on the critic function
- Improves training stability by preventing gradient explosion/vanishing
- Enables more efficient and effective adversarial learning
- Helps the model converge to better solutions compared to standard GANs

The penalty is applied by:
1. Creating interpolated samples between real and fake images
2. Computing gradients with respect to these samples
3. Penalizing the critic when the gradient norm deviates from 1

## Loss Functions

The model uses a combination of losses:
- MSE (Mean Squared Error) for pixel-wise accuracy
- VGG Loss for perceptual similarity using high-level features
- Adversarial loss from the critic for generating realistic textures

## Datasets

Training is performed on a diverse set of datasets:
- **Flickr2K**: 2650 high-quality images
- **DIV2K**: 800 high-resolution training images with diverse content
- **SRGAN-Faces-4x**: Face-specific dataset with 4x downsampling
- **Urban100**: 100 urban scenes with structures and patterns

## Training Strategy

The model follows a two-phase training strategy:
1. **Pretraining phase**: Optimizes MSE and VGG loss to learn basic super-resolution
2. **SRGAN phase**: Introduces adversarial training with the critic for photo-realistic details

## Results

The model successfully generates high-resolution images with sharp details and sample images are uploaded in the Repository.
