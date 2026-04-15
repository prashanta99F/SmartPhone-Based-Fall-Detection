import urllib.request
import json
import time
import math
import os 
import statistics

# --- CONFIGURATION ---
# Target IP address of the smartphone running Phyphox
PHYPHOX_URL = "http://192.168.0.128" 

# Bypass local proxy settings to ensure direct connection to the phone
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['no_proxy'] = '*'

def get_sensor_data():
    """Fetches live tri-axial accelerometer data from the smartphone over Wi-Fi."""
    try:
        url = f"{PHYPHOX_URL}/get?accX&accY&accZ"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=0.2) as response:
            data = json.loads(response.read().decode())
            
            # Extract the most recent X, Y, Z acceleration values
            try:
                ax = data['buffer']['accX']['buffer'][0]
                ay = data['buffer']['accY']['buffer'][0]
                az = data['buffer']['accZ']['buffer'][0]
            except:
                ax = data['buffer']['accX'][-1]
                ay = data['buffer']['accY'][-1]
                az = data['buffer']['accZ'][-1]
            return ax, ay, az
    except: 
        return None

def analyze_movement():
    """Main continuous monitoring loop and multi-tier statistical classifier."""
    print(f"--- CONNECTED TO {PHYPHOX_URL} ---")
    print("1. Put phone in POCKET (Hoodie or Pants).")
    print("2. Stand still for calibration (3s)...")
    
    #  CALIBRATION PHASE 
    # Establish baseline gravity to account for different phone sensors
    avg_val = 0
    samples = 0
    for i in range(20): 
        val = get_sensor_data()
        if val:
            mag = math.sqrt(val[0]**2 + val[1]**2 + val[2]**2)
            avg_val += mag
            samples += 1
        time.sleep(0.1)
    
    if samples == 0:
        print("[ERROR] No data. Check IP.")
        return

    avg_val /= samples
    scale_factor = 9.81 if avg_val > 5.0 else 1.0
    print(f"--- SYSTEM READY: HIGH SENSITIVITY MODE ---")

    # Initialize rolling data buffers
    history_g = [] 
    history_y = [] 
    
    last_print_time = time.time()
    fall_start_time = None
    
    # Algorithm Thresholds
    FREE_FALL = 0.6    
    IMPACT = 2.0       
    
    # --- CONTINUOUS MONITORING LOOP ---
    while True:
        raw = get_sensor_data()
        if raw:
            # Normalize raw data to standard g-force
            ax, ay, az = raw
            gx = ax / scale_factor
            gy = ay / scale_factor
            gz = az / scale_factor
            
            # Calculate Signal Vector Magnitude (SVM)
            total_g = math.sqrt(gx**2 + gy**2 + gz**2)
            
            # Update 10-tick rolling history buffer
            history_g.append(total_g)
            history_y.append(abs(gy)) 
            if len(history_g) > 10: 
                history_g.pop(0)
                history_y.pop(0)

            now = time.time()

            # TIER 1: Detect pre-impact weightlessness (Freefall)
            if total_g < FREE_FALL:
                if fall_start_time is None: 
                    fall_start_time = now
            
            # TIER 2: Detect sudden impact spike
            elif total_g > IMPACT:
                print(f"\n IMPACT DETECTED ({total_g:.1f}g).")
                
                # Anti-bounce delay to ignore immediate physical reverberations
                print("   -> Settling (1.0s)...")
                time.sleep(1.0)
                
                # TIER 3: Post-Impact Statistical Window
                print("   -> Checking for Recovery (1.0s)...")
                post_impact_g = []
                post_impact_y = []
                
                # Collect exactly 1 second of post-impact kinematic data
                start_collect = time.time()
                while time.time() - start_collect < 1.0:
                    p_raw = get_sensor_data()
                    if p_raw:
                        p_gx = p_raw[0] / scale_factor
                        p_gy = p_raw[1] / scale_factor
                        p_gz = p_raw[2] / scale_factor
                        p_total = math.sqrt(p_gx**2 + p_gy**2 + p_gz**2)
                        
                        post_impact_g.append(p_total)
                        post_impact_y.append(abs(p_gy))
                    time.sleep(0.05)
                
                # Evaluate the collected window data
                if len(post_impact_g) > 2:
                    final_tilt = statistics.mean(post_impact_y)
                    final_energy = statistics.stdev(post_impact_g)
                    
                    print(f"   Tilt: {final_tilt:.2f} | Energy: {final_energy:.2f}")

                    # TIER 4: Classification Logic (ADL vs Genuine Fall)
                  
                    # Rule 1: User is upright (Likely a jump or heavy step)
                    if final_tilt > 0.6:
                        print(f"RECOVERED: Upright (Jump).\n")
                    
                    # Rule 2: User is horizontal, evaluate kinetic energy
                    else:
                        # High kinetic variance means user is moving/recovering
                        if final_energy > 0.45:
                            print(f"RECOVERED: Horizontal but Active (Hoodie Run).\n")
                        # Low variance + Horizontal = Catastrophic Fall (Long Lie)
                        else:
                            print(f" FALL CONFIRMED: Horizontal & Still.\n")
                            
                time.sleep(4)
                
                # Reset buffers for next event
                fall_start_time = None
                history_g = [] 
                continue 

            # Reset logic if gravity normalizes without an impact spike
            elif 0.8 < total_g < 1.5:
                if fall_start_time is not None and (now - fall_start_time) > 0:
                    fall_start_time = None 

# Execute the main function when script is run
if __name__ == "__main__":
    analyze_movement()