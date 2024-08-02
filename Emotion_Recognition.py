import warnings
import os
import tensorflow as tf
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
import librosa.display
import librosa.feature
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, LSTM, TimeDistributed, Layer, Activation, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Suppress warnings and logging
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Custom attention layer
class AttentionLayer(Layer):
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

# Load and label dataset
paths = []
labels = []
for dirname, _, filenames in os.walk("C:\\Users\\karni\\OneDrive\\Desktop\\electrical_eng\\4_YEAR_B\\Machine_Learning\\final_course_project\\python files\\TESS Toronto emotional speech set data"):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1].split('.')[0].lower()
        labels.append(label)

print('Dataset is Loaded')

# Create dataframe
df = pd.DataFrame({'speech': paths, 'label': labels})

# Check the number of unique labels
num_classes = len(df['label'].unique())
print(f'Number of classes: {num_classes}')

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
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Encode labels
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']]).toarray()

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.125, random_state=42)

# Define the model
input_shape = (X_train.shape[1], X_train.shape[2], 1)

model = Sequential()

# Adding 4 convolutional blocks
for _ in range(4):
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.01), input_shape=input_shape if _ == 0 else None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.2))

model.add(TimeDistributed(Flatten()))
model.add(LSTM(32, return_sequences=True))
model.add(AttentionLayer())
model.add(Reshape((1, 32)))  # Reshape layer to convert the 2D output from AttentionLayer to 3D
model.add(LSTM(32))
model.add(Dense(num_classes, activation='softmax'))  # Adjusted number of neurons to match number of classes

# Compile the model with a lower learning rate
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])

# Print model summary and save it to a text file
model_summary_text = []
model.summary(print_fn=lambda x: model_summary_text.append(x))

with open("model_summary.txt", "w", encoding="utf-8") as f:
    for line in model_summary_text:
        f.write(line + "\n")

# Define EarlyStopping and ReduceLROnPlateau callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train the model with early stopping and learning rate reduction
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=300, batch_size=64, callbacks=[early_stopping, reduce_lr])

# Save the model, encoder, and scaler
model.save("C:\\Users\\karni\\OneDrive\\Desktop\\electrical_eng\\4_YEAR_B\\Machine_Learning\\final_course_project\\python files\\Projecton_Model\\Model.keras")
joblib.dump(enc, "C:\\Users\\karni\\OneDrive\\Desktop\\electrical_eng\\4_YEAR_B\\Machine_Learning\\final_course_project\\python files\\Projecton_Model\\Encoder.joblib")
joblib.dump(scaler, "C:\\Users\\karni\\OneDrive\\Desktop\\electrical_eng\\4_YEAR_B\\Machine_Learning\\final_course_project\\python files\\Projecton_Model\\Scaler.joblib")

# Plot and save training history
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.savefig('training_history.png')
plt.show()

# Compute confusion matrix and F1 score
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)
cm = confusion_matrix(y_true, y_pred_classes)
f1 = f1_score(y_true, y_pred_classes, average='weighted')

print(f"F1 Score: {f1}")

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=enc.categories_[0])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix for Validation Set')
plt.savefig('confusion_matrix_validation.png')
plt.show()