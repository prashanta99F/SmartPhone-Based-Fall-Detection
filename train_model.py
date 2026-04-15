import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

print("1. Loading and combining your separate sensor files...")

def prepare_data(accel_file, gyro_file, label):
    # Read the separate files
    accel = pd.read_csv(accel_file)
    gyro = pd.read_csv(gyro_file)
    
    # Safely rename the first 4 columns to our standard names (Time, X, Y, Z)
    # This prevents errors if the app uses weird headers like 'Acceleration x (m/s^2)'
    accel.columns = ['Time', 'ax', 'ay', 'az'] + list(accel.columns[4:])
    gyro.columns = ['Time', 'gx', 'gy', 'gz'] + list(gyro.columns[4:])
    
    # Stitch the Accelerometer and Gyroscope data side-by-side
    combined = pd.concat([accel[['ax', 'ay', 'az']], gyro[['gx', 'gy', 'gz']]], axis=1)
    
    # Remove any empty rows if one sensor recorded a split-second longer than the other
    combined = combined.dropna()
    
    # Tag it as a Fall (1) or Walking (0)
    combined['Label'] = label
    return combined

# Load Falls (Label 1)
falls_df = prepare_data('falls_accel.csv', 'falls_gyro.csv', 1)

# Load Walking (Label 0)
walking_df = prepare_data('walking_accel.csv', 'walking_gyro.csv', 0)

print("2. Preparing the final dataset...")
data = pd.concat([falls_df, walking_df], ignore_index=True)

# Calculate the total force (SVM)
data['SVM'] = np.sqrt(data['ax']**2 + data['ay']**2 + data['az']**2)

# Select the sensor columns for the AI
features = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'SVM']
X = data[features]
y = data['Label']

print("3. Training the AI Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print("4. Saving the AI Brain...")
joblib.dump(model, 'fall_detection_model.pkl')
print("\nSUCCESS! You can now run the live detector.")