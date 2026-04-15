import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def prepare_data(accel_file, gyro_file, label):
    # Load and clean your specific files
    accel = pd.read_csv(accel_file)
    gyro = pd.read_csv(gyro_file)
    accel.columns = ['Time', 'ax', 'ay', 'az'] + list(accel.columns[4:])
    gyro.columns = ['Time', 'gx', 'gy', 'gz'] + list(gyro.columns[4:])
    combined = pd.concat([accel[['ax', 'ay', 'az']], gyro[['gx', 'gy', 'gz']]], axis=1).dropna()
    combined['Label'] = label
    return combined

print("1. Loading your data...")
falls_df = prepare_data('falls_accel.csv', 'falls_gyro.csv', 1)
walking_df = prepare_data('walking_accel.csv', 'walking_gyro.csv', 0)
data = pd.concat([falls_df, walking_df], ignore_index=True)

data['SVM'] = np.sqrt(data['ax']**2 + data['ay']**2 + data['az']**2)
features = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'SVM']
X = data[features]
y = data['Label']

# --- THE GOLDEN RULE OF ML TESTING ---
# Split the data! 70% for studying (train), 30% for the final exam (test)
print("2. Hiding 30% of the data for the final exam...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print("3. Training the AI on the study guide...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("4. Forcing the AI to take the final exam...")
predictions = model.predict(X_test)

# --- GRADE THE EXAM ---
accuracy = accuracy_score(y_test, predictions)
print(f"\n===========================")
print(f" FINAL EXAM SCORE: {accuracy * 100:.2f}%")
print(f"===========================\n")

print("Drawing the Confusion Matrix Report Card...")
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Walking', 'Falling'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Fall Detection Confusion Matrix")
plt.show()
