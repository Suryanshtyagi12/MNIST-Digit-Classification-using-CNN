✨ MNIST Digit Classification using Convolutional Neural Networks (CNN)
This project demonstrates handwritten digit recognition using the MNIST dataset and a Convolutional Neural Network (CNN). The model classifies grayscale images of digits (0–9) and achieves high accuracy using deep learning techniques.

📊 Dataset Overview
Dataset: MNIST

Images: 70,000 total (60,000 training + 10,000 testing)

Image Size: 28 × 28 pixels, grayscale

Classes: 10 (Digits from 0 to 9)

🧠 Model Architecture (CNN)
The CNN model consists of the following layers:

Layer Type	Description
Input Layer	28×28×1 grayscale image
Conv2D	Extracts image features
MaxPooling2D	Reduces spatial dimensions
Dropout (optional)	Prevents overfitting
Flatten	Converts 2D to 1D vector
Dense Layer(s)	Fully connected neural network
Output Layer	Softmax activation (10 classes)
⚙ Project Workflow
✅ 1. Data Loading & Preprocessing
Load dataset from Keras (tensorflow.keras.datasets.mnist)

Normalize pixel values (0–255 → 0–1)

Reshape data to (28, 28, 1)

One-hot encode labels

✅ 2. Building the CNN Model
Use Sequential() model

Add convolution, pooling, dropout, and dense layers

Compile with:

Optimizer: Adam

Loss: categorical_crossentropy

Metrics: accuracy

✅ 3. Training the Model
Train on training data with validation split

Monitor training and validation accuracy/loss

✅ 4. Model Evaluation
Test on unseen test data

Display final accuracy

Optionally show confusion matrix or sample predictions

✅ 5. Model Saving
Save the trained model as mnist_cnn_model.h5 or .keras

