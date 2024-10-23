# Brain_tumor
# Brain Tumor Detector

This project involves building a detection model using a Convolutional Neural Network (CNN) with TensorFlow and Keras. The dataset consists of brain MRI images sourced from [Kaggle](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection).

## Dataset Overview

The dataset is organized into two folders: `yes` (tumorous) and `no` (non-tumorous), containing a total of **253 Brain MRI images**:
- **Yes (Tumorous)**: 155 images
- **No (Non-tumorous)**: 98 images

## Getting Started

**Note:** Viewing Jupyter notebooks on GitHub can sometimes be problematic. For a better experience, you can use [nbviewer](https://nbviewer.jupyter.org/).

### Data Augmentation

Data augmentation was employed to address the limited dataset size and tackle class imbalance. 

- **Original Dataset**: 155 positive and 98 negative examples (253 images).
- **After Augmentation**: The dataset expanded to 2065 images, comprising:
  - 1085 positive examples
  - 980 negative examples

The augmented images, along with the original ones, can be found in the 'augmented data' folder.

### Data Preprocessing

Preprocessing steps for each image include:

1. Cropping to focus on the brain region.
2. Resizing to (240, 240, 3) to standardize input dimensions.
3. Normalizing pixel values to the range [0, 1].

### Data Splitting

The dataset was split as follows:

- 70% for training
- 15% for validation
- 15% for testing

## Neural Network Architecture

![Neural Network Architecture](convnet_architecture.jpg)

### Architecture Details

The model architecture is designed to handle the input shape of (240, 240, 3) through the following layers:

1. Zero Padding layer (2x2)
2. Convolutional layer (32 filters, 7x7 kernel, stride 1)
3. Batch normalization layer
4. ReLU activation layer
5. Max Pooling layer (4x4)
6. Second Max Pooling layer (4x4)
7. Flatten layer to convert to a one-dimensional vector
8. Dense output layer with one neuron and sigmoid activation (binary classification)

### Rationale for Architecture Choice

While transfer learning with ResNet50 and VGG-16 was initially considered, the models proved too complex for the dataset size and led to overfitting. Given computational constraints (6th gen Intel i7 CPU, 8 GB RAM), a simpler architecture was chosen and successfully trained from scratch.

## Model Training

The model was trained for **24 epochs**. Below are the loss and accuracy plots:

![Loss plot](Loss.PNG)
![Accuracy plot](Accuracy.PNG)

The highest validation accuracy was recorded during the 23rd epoch.

## Results

The best model achieved the following performance metrics on the test set:

- **Accuracy**: 88.7%
- **F1 Score**: 0.88

### Performance Summary

| Metric       | Validation Set | Test Set |
|--------------|----------------|----------|
| Accuracy     | 91%            | 89%      |
| F1 Score     | 0.91           | 0.88     |

These results indicate strong model performance, especially considering the balanced nature of the dataset.
