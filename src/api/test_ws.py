import websocket
import json
import time
import librosa
import numpy as np
import os
import torch

# --- CONFIG ---
MOCK_MIC_RATE = 44100  # Simulate a 44.1kHz microphone
WS_URL = f"ws://localhost:8000/stream/audio?rate={MOCK_MIC_RATE}"
TEST_FILE = r"C:\dev\archive\Emotions\Angry\03-01-05-01-01-01-01.wav"

def test_streaming():
    if not os.path.exists(TEST_FILE):
        print(f"Test file not found: {TEST_FILE}")
        return

    print(f"Loading test file at {MOCK_MIC_RATE}Hz to simulate high-res mic...")
    # Load audio and resample to the MOCK rate
    speech, _ = librosa.load(TEST_FILE, sr=MOCK_MIC_RATE)
    
    # Connect to WebSocket
    print(f"Connecting to {WS_URL}...")
    try:
        ws = websocket.create_connection(WS_URL)
    except Exception as e:
        print(f"Connection failed: {e}")
        return
    
    try:
        # Send 0.5s chunks of 44.1kHz data
        chunk_size = int(MOCK_MIC_RATE * 0.5)
        
        print("Starting Stream...")
        for i in range(0, len(speech), chunk_size):
            chunk = speech[i:i + chunk_size]
            if len(chunk) == 0: continue

            # Convert to 16-bit PCM
            chunk_int16 = (chunk * 32767).astype(np.int16)
            
            # Send binary data
            ws.send_binary(chunk_int16.tobytes())
            
            # Receive response
            try:
                # Set a longer timeout for resampling latency
                ws.settimeout(0.5)
                result = ws.recv()
                data = json.loads(result)
                
                # Check for the new status field and confidence
                status_marker = "[ALERT]" if data.get('status') == "high_confidence" else "[INFO]"
                print(f"{status_marker} Prediction: {data['emotion']} | Conf: {data['confidence']:.2%} | Status: {data.get('status')}")
                
            except websocket.WebSocketTimeoutException:
                pass
            
            time.sleep(0.5)
            
        print("\nStream Finished.")
        
    except Exception as e:
        print(f"Error during stream: {e}")
    finally:
        ws.close()
        print("WebSocket Closed.")

if __name__ == "__main__":
    test_streaming()