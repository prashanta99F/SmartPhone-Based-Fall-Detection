import pandas as pd
import numpy as np
import joblib
import time
import requests
from collections import Counter

PHYPHOX_IP = "http://192.168.0.128"
STREAM_URL = f"{PHYPHOX_IP}/get?accX&accY&accZ&gyroX&gyroY&gyroZ"

print("Loading AI Brain...")
model = joblib.load('fall_detection_model.pkl')

window_size = 25 
data_buffer = []
cooldown_timer = 0 

print(f"Connecting to Phyphox at {PHYPHOX_IP}...")
print("\n--- LIVE ML DASHBOARD (Press Ctrl+C to stop) ---\n")

while True:
    try:
        response = requests.get(STREAM_URL, timeout=1)
        data = response.json()
        
        ax = data['buffer']['accX']['buffer'][-1]
        ay = data['buffer']['accY']['buffer'][-1]
        az = data['buffer']['accZ']['buffer'][-1]
        
        gx = data['buffer']['gyroX']['buffer'][-1]
        gy = data['buffer']['gyroY']['buffer'][-1]
        gz = data['buffer']['gyroZ']['buffer'][-1]
        
        row = {'ax': ax, 'ay': ay, 'az': az, 'gx': gx, 'gy': gy, 'gz': gz}
        data_buffer.append(row)
        
        if len(data_buffer) >= window_size:
            window_df = pd.DataFrame(data_buffer)
            # Calculate SVM in m/s^2
            window_df['SVM'] = np.sqrt(window_df['ax']**2 + window_df['ay']**2 + window_df['az']**2)
            
            features = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'SVM']
            X_live = window_df[features]
            
            # 1. Ask the ML model what is happening right now
            predictions = model.predict(X_live)
            raw_prediction = Counter(predictions).most_common(1)[0][0]
            
            # 2. Calculate the max impact for the dashboard
            current_svm = window_df['SVM'].max()
            
            # 3. Print the Live Dashboard showing EXACTLY what number the AI is outputting
            if cooldown_timer == 0:
                print(f"AI Thinks Number: [ {raw_prediction} ] | Max Impact: {current_svm:.2f} m/s²    ", end="\r")
            
            # 4. NEW TRIGGER: If impact breaks 20.0, trigger the alarm instantly!
            if current_svm > 20.0 and cooldown_timer == 0: 
                print("\n\n" + "="*50)
                print(f"  IMPACT DETECTED! The AI's state number was: {raw_prediction}  ")
                print("="*50 + "\n")
                cooldown_timer = 50 
            
            # Remove the oldest frame to keep the sliding window moving
            data_buffer.pop(0)
        
        if cooldown_timer > 0:
            cooldown_timer -= 1
            
        time.sleep(0.02)
        
    except requests.exceptions.RequestException:
        print("Waiting for phone connection... (Make sure Phyphox is playing)", end="\r")
        time.sleep(2)
    except KeyError:
        print("Connected, waiting for sensor data...", end="\r")
        time.sleep(2)