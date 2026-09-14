# Task 3: Handwritten Digit Recognition

## Objective

Recognize handwritten digits from 28x28 grayscale images using a Convolutional Neural Network (CNN).

## Dataset

MNIST contains 60,000 training images and 10,000 test images of handwritten digits from 0 to 9.

## Workflow

1. Load MNIST
2. Normalize pixel values from 0-255 to 0-1
3. Reshape images for CNN input
4. Build a CNN
5. Train with a validation split
6. Evaluate on the untouched test set
7. Save the trained model
8. Generate loss/accuracy plots
9. Generate a confusion matrix
10. Save example predictions

## CNN architecture

- Conv2D: 32 filters, 3x3
- MaxPooling2D
- Conv2D: 64 filters, 3x3
- MaxPooling2D
- Flatten
- Dense: 128 neurons
- Dropout
- Dense: 10-class softmax output

The model uses sparse categorical cross-entropy because the labels are integer digit IDs.
