#!/bin/bash

echo "Setting up the Advanced Deepfake Detector environment..."

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv deepfake_env
source deepfake_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing required packages..."
pip install streamlit opencv-python pillow numpy matplotlib scikit-learn streamlit-image-select || pip install --break-system-packages streamlit opencv-python pillow numpy matplotlib scikit-learn streamlit-image-select

# Install keras and tensorflow (lightweight version for compatibility)
echo "Installing deep learning dependencies..."
pip install keras tensorflow-cpu || pip install --break-system-packages keras tensorflow-cpu

# Unzip the model
echo "Unpacking the model files..."
cd code/PretrainedModel/
unzip -o dffnetv2B0.zip

echo "Setup complete! To run the application:"
echo "1. Activate the virtual environment: source deepfake_env/bin/activate"
echo "2. Navigate to: cd code/PretrainedModel/streamlit_deepfake_detector"
echo "3. Run the app: streamlit run final_app.py" 