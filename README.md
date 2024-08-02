# Voice_emotion_recognition
Voice emotion recognition by machine learning techniques 

# Speech Emotion Recognition Using Attention Model

## Overview

This project aims to recognize emotions in speech using an advanced deep learning model that incorporates Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, and an Attention mechanism. The model is trained and evaluated using the TESS Toronto emotional speech set data and other datasets combined to improve the robustness and accuracy of emotion detection.

## Features

- **Attention Mechanism**: Enhances the model's ability to focus on significant parts of the speech signal.
- **CNN + LSTM Architecture**: Combines the spatial feature extraction capabilities of CNN with the temporal processing power of LSTM.
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

The primary dataset used in this project is the **TESS Toronto emotional speech set**, supplemented with data from the RAVDESS and SAVEE datasets. The combined dataset includes various emotions: happy, sad, angry, surprise, disgust, calm, fearful, and neutral.

## Installation

To run this project, ensure you have Python installed (version 3.6 or higher) and set up a virtual environment. Then, install the required packages using the following command:

```bash
pip install -r requirements.txt
```

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

### Feature Extraction

The feature extraction process uses MFCC (Mel Frequency Cepstral Coefficients), which have been identified as the best-performing features for speech emotion recognition.

### Attention Layer

A custom attention layer is implemented to allow the model to focus on important parts of the speech signal. This layer enhances the model's ability to capture significant features and improve overall performance.

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

## License

This project is licensed under the MIT License. See the LICENSE file for details.
