print("Starting Smart Grid Anomaly Detection Project...")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout

print("Loading dataset...")
data = pd.read_csv("data/smartgrid_dataset.csv")

print("Dataset loaded successfully.")
print("Shape:", data.shape)

X = data.drop("label", axis=1)
y = data["label"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = X.reshape(X.shape[0], X.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Building model...")

model = Sequential([
    Conv1D(32, 2, activation='relu', input_shape=(X.shape[1],1)),
    LSTM(32),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("Training started...")
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

print("Training completed.")

print("Evaluating model...")
y_pred = np.argmax(model.predict(X_test), axis=1)

acc = accuracy_score(y_test, y_pred)

print("\n===== RESULTS =====")
print("Accuracy:", acc)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True)
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy Graph")
plt.legend(["Train", "Validation"])
plt.savefig("results/accuracy.png")

print("Graphs saved inside results folder.")
print("Project completed successfully!")
