# Speech Emotion Recognition Using Attention Model

## Overview

This project aims to recognize emotions in speech using an advanced deep learning model that incorporates Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, and an Attention mechanism. The model is trained and evaluated using the TESS Toronto emotional speech set data.

## Features

- **CNN + LSTM Architecture**: Combines the spatial feature extraction capabilities of CNN with the temporal processing power of LSTM.
- **Attention Mechanism**: Enhances the model's ability to focus on significant parts of the speech signal.
- **High Accuracy**: Achieves an average test accuracy rate of 90%, outperforming many existing models.

## Dependencies

The project relies on the following libraries and frameworks:

- TensorFlow
- Keras
- Scikit-learn
- Librosa
- Matplotlib
- Joblib
- Pandas
- NumPy

## Dataset

The primary dataset used in this project is the [TESS Toronto emotional speech set](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess).The dataset includes recordings categorized into seven emotional classes: Happy, Fear, Surprised, Sad, Angry, Disgust, and Neutral. Each recording begins with the phrase "say the word..." followed by a different word, making up 400 unique sentences per emotion. The sentences are uniformly recorded by both a male and a female speaker, resulting in 200 recordings per gender for each emotion. Altogether, the dataset includes 2,800 recordings.

## Data Distribution

The base data must be manually distributed in a random manner to the test folder. Specifically, 40 recordings from each folder in the database should be transferred to the test folder creating a test set of 560 recordings (20% from the data base). The path to this test folder must be specified in the `Use_Our_Model.py` script. The remaining recordings in the folders will be used for training and verification in the `Emotion_Recognition.py` script.

## Installation

To run this project, ensure you have Python installed (version 3.6 or higher) and set up a virtual environment. Then, install the required packages using the following commands:

```bash
pip install tensorflow
pip install keras
pip install scikit-learn
pip install librosa
pip install matplotlib
pip install joblib
pip install pandas
pip install numpy
```

## Feature Extraction

The feature extraction process uses MFCC (Mel Frequency Cepstral Coefficients), which have been identified as the best-performing features for speech emotion recognition. The MFCCs are extracted from the audio files and padded to ensure uniformity in feature array shapes.

## Model Architecture

### Convolutional Neural Network (CNN)

The model starts with four convolutional blocks, each containing:
- **Conv2D Layer**: Extracts spatial features from the input data.
- **BatchNormalization**: Normalizes the activations of the previous layer.
- **Activation Layer (ReLU)**: Adds non-linearity to the model.
- **MaxPooling2D Layer**: Reduces the dimensionality of the feature maps.
- **Dropout Layer**: Prevents overfitting by randomly setting input units to zero.

### Long Short-Term Memory (LSTM) Network

- **TimeDistributed Layer**: Flattens the feature maps to make them suitable for LSTM input.
- **LSTM Layer**: Captures temporal dependencies in the data with return sequences set to true.
- **Attention Layer**: Focuses on important parts of the sequence by assigning weights to each time step.
- **Reshape Layer**: Converts the 2D output of the attention layer to 3D.
- **LSTM Layer**: Processes the weighted sequence data.

### Dense Layer

- **Dense Layer with Softmax Activation**: Outputs the probability distribution over the emotion classes.

### Model Compilation

The model is compiled with a lower learning rate using the Adam optimizer and categorical cross-entropy loss function.

## Usage

### Training the Model

1. **Prepare the Data**: Ensure the dataset is available in the specified directory structure.
2. **Run the Training Script**: Execute the `Emotion_Recognition.py` script to train the model.

```bash
python Emotion_Recognition.py
```

### Evaluating the Model

1. **Prepare the Test Data**: Ensure the test dataset is properly labeled and available.
2. **Run the Evaluation Script**: Execute the `Use_Our_Model.py` script to load the trained model and evaluate its performance.

```bash
python Use_Our_Model.py
```

## Results

The model achieves high accuracy in detecting emotions from speech signals. The confusion matrix and classification report provide detailed insights into the model's performance across different emotions.

### Confusion Matrix

The confusion matrix for the validation set is saved as `confusion_matrix_validation.png` and can be visualized to understand the performance of the model.

### Classification Report

The precision, recall, and F1-score for each emotion are calculated and printed, providing a comprehensive evaluation of the model's accuracy.

## Conclusion

This project demonstrates the effectiveness of combining CNN, LSTM, and Attention mechanisms for speech emotion recognition. The proposed model shows significant improvements over traditional methods and can be used for various applications in healthcare, customer service, and more.

## References

- Singh, J.; Saheer, L.B.; Faust, O. (2023). Speech Emotion Recognition Using Attention Model. *International Journal of Environmental Research and Public Health*, 20, 5140. [Link to article](https://doi.org/10.3390/ijerph20065140)
- [TESS Toronto emotional speech set](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

## License

This project is licensed under the MIT License. See the LICENSE file for details.
