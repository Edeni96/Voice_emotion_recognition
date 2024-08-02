import warnings
import os
import tensorflow as tf
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
import librosa.display
import librosa.feature
import joblib
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Suppress warnings and logging
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define custom attention layer class to be loaded with the model
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],), initializer="zeros", trainable=True)
        self.U = self.add_weight(name="att_u", shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.U, axes=1)
        ait = tf.squeeze(ait, -1)
        ait = tf.exp(ait)
        ait /= tf.reduce_sum(ait, axis=1, keepdims=True)
        ait = tf.expand_dims(ait, -1)
        output = inputs * ait
        return tf.reduce_sum(output, axis=1)

# Load and label test dataset
paths = []
labels = []
for dirname, _, filenames in os.walk("C:\\Users\\karni\\OneDrive\\Desktop\\electrical_eng\\4_YEAR_B\\Machine_Learning\\final_course_project\\python files\\Test"):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1].split('.')[0].lower()
        labels.append(label)

print('Dataset is Loaded')

# Create dataframe
df = pd.DataFrame({'speech': paths, 'label': labels})

# Feature extraction function
def extract_features(filename, max_len=130):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # Padding to ensure all feature arrays have the same shape
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

# Extract features
X = np.array([extract_features(x) for x in df['speech']])
X = np.expand_dims(X, -1)
scaler = joblib.load("C:\\Users\\karni\\OneDrive\\Desktop\\electrical_eng\\4_YEAR_B\\Machine_Learning\\final_course_project\\python files\\Projecton_Model\\Scaler.joblib")
X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Encode labels
enc = joblib.load("C:\\Users\\karni\\OneDrive\\Desktop\\electrical_eng\\4_YEAR_B\\Machine_Learning\\final_course_project\\python files\\Projecton_Model\\Encoder.joblib")
y = enc.transform(df[['label']]).toarray()

# Load the model
model = load_model("C:\\Users\\karni\\OneDrive\\Desktop\\electrical_eng\\4_YEAR_B\\Machine_Learning\\final_course_project\\python files\\Projecton_Model\\Model.keras", custom_objects={'AttentionLayer': AttentionLayer})

# Print model summary and save it to a text file
model_summary_text = []
model.summary(print_fn=lambda x: model_summary_text.append(x))

with open("model_summary.txt", "w", encoding="utf-8") as f:
    for line in model_summary_text:
        f.write(line + "\n")

# Predict and evaluate the model
y_pred = model.predict(X)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y, axis=1)
cm = confusion_matrix(y_true, y_pred_classes)

# Calculate precision, recall, and F1-score
report = classification_report(y_true, y_pred_classes, target_names=enc.categories_[0], output_dict=True)
precision = {label: report[label]['precision'] for label in report if label not in ['accuracy', 'macro avg', 'weighted avg']}
recall = {label: report[label]['recall'] for label in report if label not in ['accuracy', 'macro avg', 'weighted avg']}
f1_score = {label: report[label]['f1-score'] for label in report if label not in ['accuracy', 'macro avg', 'weighted avg']}

# Print the table
print(f"{'Emotional Label':<15}{'Precision (Percentage)':<25}{'Recall':<10}{'F1-Score'}")
for label in precision:
    print(f"{label:<15}{precision[label]*100:<25.2f}{recall[label]*100:<10.2f}{f1_score[label]*100:.2f}")

print(f"\n{'Macro Average':<15}{report['macro avg']['precision']*100:<25.2f}{report['macro avg']['recall']*100:<10.2f}{report['macro avg']['f1-score']*100:.2f}")
print(f"{'Weighted Average':<15}{report['weighted avg']['precision']*100:<25.2f}{report['weighted avg']['recall']*100:<10.2f}{report['weighted avg']['f1-score']*100:.2f}")
print(f"\n{'Accuracy':<15}{report['accuracy']*100:.2f}")

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=enc.categories_[0])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix for Test Set')
plt.savefig('confusion_matrix_test.png')
plt.show()
