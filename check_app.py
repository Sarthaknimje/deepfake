import requests
import time
import sys

def check_streamlit_app():
    """Check if the Streamlit app is running on port 8501"""
    url = "http://localhost:8501/"
    
    # Try multiple times as the app may take time to start
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✅ Streamlit app is running successfully on {url}")
                return True
        except requests.exceptions.RequestException:
            print(f"Attempt {attempt+1}/{max_retries}: App not responding yet, retrying...")
            time.sleep(2)
    
    print("❌ Failed to connect to the Streamlit app")
    return False

if __name__ == "__main__":
    check_streamlit_app() 