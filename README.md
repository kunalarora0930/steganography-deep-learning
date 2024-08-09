# Steganography with Convolutional Neural Networks
Kaggle Notebook: https://www.kaggle.com/code/cheesecke/steganography-using-deep-learning/


This project implements a deep learning-based approach to image steganography using Convolutional Neural Networks (CNNs). The goal is to hide a secret image within a cover image, enabling the transmission of hidden data without noticeable changes to the cover image. The implementation uses the Tiny ImageNet dataset for training and testing the model.

## Features

- **Preprocessing**: Loads and prepares the Tiny ImageNet dataset.
- **Encoder Network**: Combines the secret and cover images into a single stego image using CNN layers.
- **Decoder Network**: Extracts the secret image from the stego image.
- **Custom Loss Functions**: Uses custom loss functions to optimize the hiding and revealing processes.
- **Training**: Implements a training loop with adjustable learning rates and monitors the loss values.
- **Evaluation**: Analyzes and visualizes the performance of the model using error metrics and histograms.

## Dataset
This project uses tiny-imagenet-200 dataset from kaggle: 
https://www.kaggle.com/datasets/nikhilshingadiya/tinyimagenet200

## Model Architecture

The model consists of two primary components: the Encoder and the Decoder. 

### Encoder Network

The encoder network is responsible for embedding the secret image into the cover image to produce a stego image. 

- **Input Layers**: The network takes two inputs: the cover image and the secret image. Both images are of the same dimensions.
- **Convolutional Layers**: A series of convolutional layers are applied to both inputs to extract features.
- **Concatenation and Processing**: The features from both images are concatenated and further processed through additional convolutional layers to create the stego image.
- **Output Layer**: The output is a single image, the stego image, which visually resembles the cover image while embedding the secret image.

### Decoder Network

The decoder network aims to extract the secret image from the stego image.

- **Input Layer**: The network takes the stego image as input.
- **Convolutional Layers**: Similar to the encoder, it uses convolutional layers to process the stego image and retrieve the embedded secret features.
- **Output Layer**: The final output is an approximation of the original secret image.

## Logic and Code Explanation

1. **Data Loading and Preprocessing**: The dataset is loaded, and images are resized to a consistent shape. Normalization is applied to bring pixel values to the [0, 1] range.

2. **Model Building**: The encoder and decoder are implemented using TensorFlow and Keras. Each network is composed of convolutional layers that progressively extract features.

3. **Custom Loss Functions**: 
   - **Revealing Loss**: This measures the mean squared error (MSE) between the true and predicted secret images.
   - **Full Loss**: This combines the revealing loss with the loss between the true and predicted cover images. This ensures that the stego image closely resembles the cover image while accurately embedding the secret image.

4. **Training Loop**: The model is trained over a number of epochs. The training loop adjusts learning rates and monitors loss values to improve model performance.

5. **Evaluation**: Post-training, the model's performance is evaluated by checking the accuracy of the decoded secret image and comparing the stego image to the cover image.

## Results

The performance of the model can be assessed through visualizations and metrics that evaluate the accuracy of the steganography process. The mean squared error per pixel and histograms of pixel errors provide insights into the effectiveness of the model.

